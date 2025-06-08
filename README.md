# Quantum Nested Sampling — Publication‑grade Benchmark Suite

This repository contains a complete implementation of Quantum Nested Sampling (QNS) compared with Classical Nested Sampling (CNS) for Bayesian evidence computation.

## Features

- **Classical Nested Sampling**: Standard implementation following Skilling (2006)
- **Quantum Nested Sampling**: Grover-accelerated version with quantum oracle optimization
- **Comprehensive Benchmarking**: Multi-dimensional parameter sweeps and performance analysis
- **Publication-ready Figures**: LaTeX-compatible plots with matplotlib/PGF backend
- **Real-time Logging**: Detailed progress tracking for long-running experiments

## Installation

This project uses Poetry for dependency management. To install:

```bash
# Install with basic dependencies (classical sampling only)
poetry install

# Install with quantum dependencies (includes Qiskit)
poetry install --extras quantum

# Activate the virtual environment
poetry shell
```

## Usage

Run the benchmark suite:

```bash
# Default benchmark (classical and quantum if available)
python quantum_nested_sampling_publication.py run

# Custom parameter sweep
python quantum_nested_sampling_publication.py sweep --dims 2 4 8 --n_live 100 200 --repeats 3

# Generate plots from existing results
python quantum_nested_sampling_publication.py plot
```

## Output

- Results are saved to `results/benchmark_results.csv`
- Figures are saved to `figures/accuracy_dim*.pdf`
- Runtime logs are saved to `quantum_nested_sampling.log`

## License

CC-BY-4.0 — feel free to adapt for your paper/thesis.
