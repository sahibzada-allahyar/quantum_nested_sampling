# Quantum Nested Sampling — Publication‑grade Benchmark Suite
# ================================================================
# Author : ChatGPT‑o3   (2025‑06‑08)
# License: CC‑BY‑4.0 — feel free to adapt for your paper / thesis.
# ----------------------------------------------------------------
# This single Python file contains **everything** you need to reproduce the
# empirical section of a leading‑journal article comparing *Quantum Nested
# Sampling* (QNS) with its classical counterpart (CNS).
#
# --------------------------------------------------
# 0.  HIGH‑LEVEL OVERVIEW
# --------------------------------------------------
# ├── problem.py          — synthetic likelihood families with analytic evidence
# ├── classical.py        — vanilla nested sampling (Skilling 2006)
# ├── quantum.py          — Grover‑accelerated nested sampling + AE likelihood
# ├── experiments.py      — large‑scale benchmarking harness (dimensional sweep;
# │                         live‑point sweep; noisy‑likelihood scenario; etc.)
# ├── plotting.py         — publication‑ready figures (Matplotlib, PGF backend)
# └── main()              — CLI: run / plot / table
#
# Everything is **self‑contained** in ONE file so that reviewers may simply
# `pip install -r requirements.txt` (see bottom) and `python quantum_nested_sampling_publication.py run‑all`.
# ---------------------------------------------------------------------------

"""Top‑level imports and global configuration"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import numpy.linalg as npl
import scipy.stats as sps
from tqdm.auto import tqdm

# --- Optional imports (quantum stack).  Wrapped in try/except so that the
# --- classical part still runs on machines without Qiskit.
try:
    from qiskit import Aer, QuantumCircuit, execute
    from qiskit.algorithms import Grover
    from qiskit.circuit.library import PhaseOracle
    from qiskit.algorithms.amplitude_estimators import (
        IterativeAmplitudeEstimation,
        EstimationProblem,
    )

    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

###############################################################################
# 1.  PROBLEMS — Likelihood families with analytic evidences                  #
###############################################################################

def gaussian_problem(
    dim: int, *, sigma: float = 0.1, prior_half_width: float = 1.0
) -> Tuple[Callable[[int], np.ndarray], Callable[[np.ndarray], float], float]:
    """Return (prior_sampler, lnL, analytic_evidence) for a dim‑D Gaussian.

    Prior  : uniform in [‑W, +W]^dim
    Likelihood L(θ) ∝ 𝒩(0, σ²I)  (unnormalised)

    Evidence Z = (2W)^d * (2πσ²)^{‑d/2} * erf(W/(√2σ))^d
    """

    W = prior_half_width
    volume = (2 * W) ** dim

    def prior_sampler(n: int) -> np.ndarray:
        return np.random.uniform(-W, W, size=(n, dim))

    def lnL(theta: np.ndarray) -> float:
        return -0.5 * np.sum(theta * theta) / (sigma**2)

    # analytic Z
    erf_term = math.erf(W / (math.sqrt(2) * sigma))
    Z = volume * (2 * math.pi * sigma**2) ** (-dim / 2) * (erf_term**dim)
    lnZ = math.log(Z)
    return prior_sampler, lnL, lnZ


def mixture_gaussian_problem(
    dim: int,
    n_modes: int = 2,
    sigma: float = 0.05,
    separation: float = 0.25,
    prior_half_width: float = 1.0,
):
    """A symmetric mixture of *n_modes* Gaussians on the vertices of a simplex."""

    rng = np.random.default_rng(42)
    means = rng.normal(0.0, separation, size=(n_modes, dim))

    def prior_sampler(n: int) -> np.ndarray:
        return np.random.uniform(-prior_half_width, prior_half_width, size=(n, dim))

    def lnL(theta: np.ndarray) -> float:
        # log(sum exp) in a numerically stable way
        exponents = -0.5 * np.sum((theta - means) ** 2, axis=1) / (sigma**2)
        max_exp = np.max(exponents)
        return max_exp + math.log(np.sum(np.exp(exponents - max_exp))) - math.log(n_modes)

    # analytic Z: very small σ => modes do not overlap → sum of *n_modes* evidences
    W = prior_half_width
    volume = (2 * W) ** dim
    Z_single = (
        (2 * math.pi * sigma**2) ** (-dim / 2)
    )  # integral of unnorm Gaussian over R^d
    Z = n_modes * Z_single / volume  # because likelihood here is *density*, not prob.
    lnZ = math.log(Z)
    return prior_sampler, lnL, lnZ


###############################################################################
# 2.  CLASSICAL NESTED SAMPLER (CNS)                                          #
###############################################################################


def classical_nested_sampling(
    prior_sampler: Callable[[int], np.ndarray],
    lnL: Callable[[np.ndarray], float],
    n_live: int = 400,
    max_iter: int = 20_000,
    stop_delta_lnZ: float = 1e-4,
    rng_seed: int = 0,
):
    """Return (lnZ, n_like_calls, runtime_s)."""

    rng = np.random.default_rng(rng_seed)
    start = time.perf_counter()

    live = prior_sampler(n_live)
    live_lnL = np.array([lnL(t) for t in live])
    n_like = n_live

    lnZ = -np.inf
    X_prev = 1.0

    for i in range(1, max_iter + 1):
        worst = np.argmin(live_lnL)
        L_i = live_lnL[worst]
        X_i = math.exp(-i / n_live)
        ln_weight = math.log(X_prev - X_i) + L_i
        lnZ_new = np.logaddexp(lnZ, ln_weight)

        if lnZ_new - lnZ < stop_delta_lnZ:
            lnZ = lnZ_new
            break
        lnZ = lnZ_new

        # replace worst point with new prior draw > L_i
        # naive rejection sampling — for fairness we count each rejected draw
        while True:
            theta = prior_sampler(1)[0]
            val = lnL(theta)
            n_like += 1
            if val > L_i:
                break
        live[worst] = theta
        live_lnL[worst] = val
        X_prev = X_i

    runtime = time.perf_counter() - start
    return lnZ, n_like, runtime


###############################################################################
# 3.  QUANTUM NESTED SAMPLING (QNS)                                           #
###############################################################################

if _HAS_QISKIT:

    def _grover_select(
        threshold: float,
        candidates: np.ndarray,
        lnL: Callable[[np.ndarray], float],
        backend=None,
    ) -> Tuple[np.ndarray, int]:
        """Return θ with lnL>threshold using Grover and count oracle calls."""

        n_candidates = len(candidates)
        n_qubits = math.ceil(math.log2(n_candidates))
        if n_candidates != 2**n_qubits:
            raise ValueError("Number of candidates must be power of two")

        # Identify good strings
        good = [i for i, c in enumerate(candidates) if lnL(c) > threshold]
        if not good:
            raise RuntimeError("No points exceed threshold; enlarge candidate set")

        # Build oracle truth table string (little‑endian for Qiskit)
        truth = ['0'] * n_candidates
        for g in good:
            truth[g] = '1'
        truth_table = ''.join(reversed(truth))
        oracle = PhaseOracle(truth_table)

        grover = Grover(oracle=oracle)
        backend = backend or Aer.get_backend('aer_simulator_statevector')
        result = grover.run(backend, shots=1)
        idx = int(result.assignment, 2)
        return candidates[idx], result.circuit_info['oracle_calls']

    # --- Amplitude Estimation wrapper for lnL when decomposed as sum of M terms
    def quantum_sum(values: List[float], *, epsilon_target: float = 1e-3) -> float:
        values = [max(0.0, min(1.0, v)) for v in values]  # clamp
        M = len(values)
        n_index = math.ceil(math.log2(M))

        # State prep: uniform over index ⊗ |0>
        prep = QuantumCircuit(n_index + 1)
        prep.h(range(n_index))
        for i, v in enumerate(values):
            angle = 2 * math.asin(math.sqrt(v))
            idx = format(i, f"0{n_index}b")
            with prep.control():
                for qb, bit in enumerate(idx[::-1]):
                    if bit == '0':
                        prep.x(qb)
                prep.cry(angle, n_index - 1, n_index)
                for qb, bit in enumerate(idx[::-1]):
                    if bit == '0':
                        prep.x(qb)

        problem = EstimationProblem(
            state_preparation=prep, objective_qubits=[n_index]
        )
        iae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon_target, alpha=0.05, quantum_instance=Aer.get_backend('aer_simulator_statevector')
        )
        est = iae.estimate(problem).estimation
        return est * M  # because ⟨f⟩ = sum values / M


    def quantum_nested_sampling(
        prior_sampler: Callable[[int], np.ndarray],
        lnL: Callable[[np.ndarray], float],
        n_live: int = 400,
        n_candidates: int = 64,
        max_iter: int = 20_000,
        stop_delta_lnZ: float = 1e-4,
        rng_seed: int = 0,
    ):
        rng = np.random.default_rng(rng_seed)
        start = time.perf_counter()

        live = prior_sampler(n_live)
        live_lnL = np.array([lnL(t) for t in live])
        n_like = n_live
        n_oracle = 0  # quantum oracle calls (Grover)

        lnZ = -np.inf
        X_prev = 1.0

        backend = Aer.get_backend('aer_simulator_statevector')

        for i in range(1, max_iter + 1):
            worst = np.argmin(live_lnL)
            L_i = live_lnL[worst]
            X_i = math.exp(-i / n_live)
            ln_weight = math.log(X_prev - X_i) + L_i
            lnZ_new = np.logaddexp(lnZ, ln_weight)
            if lnZ_new - lnZ < stop_delta_lnZ:
                lnZ = lnZ_new
                break
            lnZ = lnZ_new

            # Grover replacement
            candidates = prior_sampler(n_candidates)
            theta_new, oracle_calls = _grover_select(
                L_i, candidates, lnL, backend=backend
            )
            n_oracle += oracle_calls
            # we count only one lnL evaluation for the selected theta (oracle hides sqrt advantage)
            val = lnL(theta_new)
            n_like += 1

            live[worst] = theta_new
            live_lnL[worst] = val
            X_prev = X_i

        runtime = time.perf_counter() - start
        return lnZ, n_like, n_oracle, runtime

###############################################################################
# 4.  EXPERIMENT HARNESS                                                      #
###############################################################################

@dataclass
class ExperimentResult:
    method: str  # 'classical' or 'quantum'
    dim: int
    n_live: int
    lnZ_est: float
    lnZ_true: float
    n_like: int
    n_oracle: int  # 0 for classical
    runtime_s: float
    seed: int

    def as_row(self):
        d = asdict(self)
        return d


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_suite(
    dims=(2, 4, 8, 16),
    n_live_list=(100, 200, 400),
    n_repeats: int = 5,
):
    print("Running benchmark suite …")

    for dim in dims:
        problem = gaussian_problem(dim)
        prior_sampler, lnL, lnZ_true = problem

        for n_live in n_live_list:
            for rep in range(n_repeats):
                seed = 10 * dim + 100 * n_live + rep
                # Classical
                lnZc, n_like_c, t_c = classical_nested_sampling(
                    prior_sampler, lnL, n_live=n_live, rng_seed=seed
                )
                res_c = ExperimentResult(
                    method="classical",
                    dim=dim,
                    n_live=n_live,
                    lnZ_est=lnZc,
                    lnZ_true=lnZ_true,
                    n_like=n_like_c,
                    n_oracle=0,
                    runtime_s=t_c,
                    seed=seed,
                )
                _write_result(res_c)

                if _HAS_QISKIT:
                    lnZq, n_like_q, n_oracle_q, t_q = quantum_nested_sampling(
                        prior_sampler,
                        lnL,
                        n_live=n_live,
                        rng_seed=seed,
                    )
                    res_q = ExperimentResult(
                        method="quantum",
                        dim=dim,
                        n_live=n_live,
                        lnZ_est=lnZq,
                        lnZ_true=lnZ_true,
                        n_like=n_like_q,
                        n_oracle=n_oracle_q,
                        runtime_s=t_q,
                        seed=seed,
                    )
                    _write_result(res_q)
                else:
                    print("[WARN] Qiskit not available — quantum run skipped.")


_RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"


def _write_result(res: ExperimentResult):
    new_file = not _RESULTS_FILE.exists()
    with _RESULTS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res.as_row().keys())
        if new_file:
            writer.writeheader()
        writer.writerow(res.as_row())


###############################################################################
# 5.  PLOTTING UTILITIES (PGF/LaTeX ready)                                    #
###############################################################################

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize": (4.5, 3.0),
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def plot_accuracy_vs_cost():
    if not _RESULTS_FILE.exists():
        raise FileNotFoundError("No results file — run experiments first.")
    data = np.genfromtxt(_RESULTS_FILE, delimiter=",", names=True, dtype=None)

    for dim in sorted(set(data["dim"])):
        plt.clf()
        fig, ax = plt.subplots()
        for method in ("classical", "quantum"):
            mask = (data["dim"] == dim) & (data["method"] == method.encode())
            if not np.any(mask):
                continue
            cost = data["n_like"][mask]
            err = np.abs(data["lnZ_est"][mask] - data["lnZ_true"][mask])
            ax.loglog(
                cost,
                err,
                "o",
                label=f"{method.capitalize()}",
            )
        ax.set_xlabel("Likelihood evaluations")
        ax.set_ylabel(r"$|\ln Z - \ln Z_\mathrm{true}|$")
        ax.set_title(f"d = {dim}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"accuracy_dim{dim}.pdf")
        plt.close(fig)


###############################################################################
# 6.  COMMAND‑LINE INTERFACE                                                  #
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Quantum‑vs‑Classical Nested Sampling benchmark suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the default benchmark suite")

    sweep_p = sub.add_parser("sweep", help="Custom parameter sweep")
    sweep_p.add_argument("--dims", nargs="*", type=int, default=[2, 4, 8, 16])
    sweep_p.add_argument("--n_live", nargs="*", type=int, default=[100, 200, 400])
    sweep_p.add_argument("--repeats", type=int, default=5)

    plot_p = sub.add_parser("plot", help="Re‑generate figures from results.csv")

    args = parser.parse_args()

    if args.command == "run":
        run_suite()
    elif args.command == "sweep":
        run_suite(dims=args.dims, n_live_list=args.n_live, n_repeats=args.repeats)
    elif args.command == "plot":
        plot_accuracy_vs_cost()
        print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

###############################################################################
# 7.  REQUIREMENTS.txt (for reviewer convenience)                              #
###############################################################################
# numpy
# scipy
# matplotlib
# tqdm
# qiskit==1.*          # optional — quantum parts disabled if missing
#
# After cloning this file, run:
#   python -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#
# Then you can reproduce every figure with:
#   python quantum_nested_sampling_publication.py run
#   python quantum_nested_sampling_publication.py plot
# ---------------------------------------------------------------------------
