# RouteOpt

Intelligent Route Optimization Using Python

## Project Overview

RouteOpt is an operations research project focused on optimizing delivery routes through mathematical algorithms and data analysis.

## Objectives

- Develop route optimization algorithms.
- Minimize travel distance and operational costs.
- Compare different optimization methods.
- Analyze real world logistics datasets.
- Evaluate operational constraints.

## Technologies

- Python
- NumPy
- Matplotlib
- Google OR-Tools (planned)

## Project Status

In development.

Implemented and tested:

- Nearest Neighbor heuristic
- Brute Force exact search
- 2-opt local search
- Three-algorithm comparison script

Automated tests are in `tests/`. Run them with:

```bash
python3 -m unittest discover -s tests -v
```

## Reproducible benchmark

Run the benchmark directly:

```bash
cd src
python3 benchmark.py
```

This generates raw results and a comparison figure in `results/`. To work interactively with the same workflow, open `notebooks/routeopt_benchmark.ipynb` and run all cells.

The benchmark uses fixed random seeds, 5/8/12/20-customer synthetic instances and three timing repetitions. Brute Force is limited to at most 8 customers to avoid factorial search on impractical instance sizes.
