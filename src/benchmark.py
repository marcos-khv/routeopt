"""Reproducible experimental benchmarking for RouteOpt TSP algorithms."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brute_force import brute_force
from nearest_neighbor import nearest_neighbor, route_distance
from two_opt import two_opt


DEFAULT_CUSTOMER_SIZES = (5, 8, 12, 20)
DEFAULT_SEEDS = (2024, 2025, 2026)
EXACT_MAX_CUSTOMERS = 8


def generate_synthetic_instance(customer_count: int, seed: int) -> np.ndarray:
    """Generate a square [0, 100] instance with a depot at its center."""
    rng = np.random.default_rng(seed)
    customers = rng.uniform(0.0, 100.0, size=(customer_count, 2))
    depot = np.array([[50.0, 50.0]])
    return np.vstack((depot, customers))


def timed_call(function, *args, repetitions: int = 3):
    """Return the median result and median elapsed time over repetitions."""
    timings = []
    result = None

    for _ in range(repetitions):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)

    return result, median(timings)


def run_benchmark(
    customer_sizes=DEFAULT_CUSTOMER_SIZES,
    seeds=DEFAULT_SEEDS,
    repetitions: int = 3,
) -> pd.DataFrame:
    """Run NN, 2-opt, and exact search on reproducible synthetic instances."""
    rows = []

    for customer_count in customer_sizes:
        for seed in seeds:
            locations = generate_synthetic_instance(customer_count, seed)

            (nn_route, nn_time) = timed_call(nearest_neighbor, locations, repetitions=repetitions)[0:2]
            nn_distance = route_distance(locations, nn_route)

            two_opt_result, opt_time = timed_call(
                two_opt,
                locations,
                nn_route,
                repetitions=repetitions,
            )
            improved_route, improved_distance = two_opt_result

            exact_distance = None
            exact_time = None
            nn_gap = None
            opt_gap = None

            if customer_count <= EXACT_MAX_CUSTOMERS:
                (exact_result, exact_time) = timed_call(brute_force, locations, repetitions=repetitions)[0:2]
                _, exact_distance = exact_result
                nn_gap = 100.0 * (nn_distance - exact_distance) / exact_distance
                opt_gap = 100.0 * (improved_distance - exact_distance) / exact_distance

            rows.append({
                "customers": customer_count,
                "seed": seed,
                "nearest_neighbor_distance": nn_distance,
                "two_opt_distance": improved_distance,
                "exact_distance": exact_distance,
                "nearest_neighbor_time_s": nn_time,
                "two_opt_time_s": opt_time,
                "brute_force_time_s": exact_time,
                "nearest_neighbor_gap_pct": nn_gap,
                "two_opt_gap_pct": opt_gap,
                "nearest_neighbor_route": json.dumps(nn_route),
                "two_opt_route": json.dumps(improved_route),
            })

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and median algorithm results for each customer size."""
    return (
        results.groupby("customers", as_index=False)
        .agg(
            nearest_neighbor_mean_distance=("nearest_neighbor_distance", "mean"),
            nearest_neighbor_median_time=("nearest_neighbor_time_s", "median"),
            two_opt_mean_distance=("two_opt_distance", "mean"),
            two_opt_median_time=("two_opt_time_s", "median"),
            exact_mean_distance=("exact_distance", "mean"),
            brute_force_median_time=("brute_force_time_s", "median"),
            mean_nn_gap=("nearest_neighbor_gap_pct", "mean"),
            mean_two_opt_gap=("two_opt_gap_pct", "mean"),
        )
    )


def plot_scalability(results: pd.DataFrame, output_path: Path) -> None:
    """Plot median execution time and mean route distance by customer count."""
    summary = summarize_results(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(summary["customers"], summary["nearest_neighbor_median_time"], marker="o", label="Nearest Neighbor")
    axes[0].plot(summary["customers"], summary["two_opt_median_time"], marker="o", label="2-opt")
    exact = summary[summary["brute_force_median_time"].notna()]
    axes[0].plot(exact["customers"], exact["brute_force_median_time"], marker="o", label="Brute Force")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of customers")
    axes[0].set_ylabel("Median execution time (seconds, log scale)")
    axes[0].set_title("Computational Scalability")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(summary["customers"], summary["nearest_neighbor_mean_distance"], marker="o", label="Nearest Neighbor")
    axes[1].plot(summary["customers"], summary["two_opt_mean_distance"], marker="o", label="2-opt")
    exact_distance = summary[summary["exact_mean_distance"].notna()]
    axes[1].plot(exact_distance["customers"], exact_distance["exact_mean_distance"], marker="o", label="Brute Force")
    axes[1].set_xlabel("Number of customers")
    axes[1].set_ylabel("Mean route distance")
    axes[1].set_title("Solution Quality")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def environment_metadata() -> dict:
    """Capture the environment needed to interpret and reproduce the benchmark."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }


def save_results(results: pd.DataFrame, results_dir: Path) -> tuple[Path, Path]:
    """Persist raw results and benchmark metadata as machine-readable files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "algorithm_benchmark.csv"
    json_path = results_dir / "benchmark_metadata.json"

    results.to_csv(csv_path, index=False)
    metadata = {
        "experiment": "RouteOpt Phase 3 - reproducible algorithm benchmark",
        "customer_sizes": sorted(results["customers"].unique().tolist()),
        "seeds": sorted(results["seed"].unique().tolist()),
        "exact_search_max_customers": EXACT_MAX_CUSTOMERS,
        "environment": environment_metadata(),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    benchmark_results = run_benchmark()
    print(summarize_results(benchmark_results).to_string(index=False))

    csv_path, json_path = save_results(benchmark_results, project_root / "results")
    plot_scalability(benchmark_results, project_root / "results" / "algorithm_benchmark.png")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
