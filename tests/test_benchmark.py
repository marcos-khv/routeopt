import sys
import unittest

import pandas as pd

sys.path.insert(0, "src")

from benchmark import generate_synthetic_instance, run_benchmark, summarize_results


class BenchmarkTests(unittest.TestCase):
    def test_synthetic_instance_is_reproducible(self):
        first = generate_synthetic_instance(5, 2024)
        second = generate_synthetic_instance(5, 2024)

        self.assertEqual(first.shape, (6, 2))
        self.assertTrue((first == second).all())
        self.assertTrue((first[0] == [50.0, 50.0]).all())

    def test_small_benchmark_has_gaps_and_exact_results(self):
        results = run_benchmark(customer_sizes=(5,), seeds=(2024,), repetitions=1)

        self.assertEqual(len(results), 1)
        self.assertFalse(pd.isna(results.iloc[0]["exact_distance"]))
        row = results.iloc[0]
        self.assertGreaterEqual(row["two_opt_gap_pct"], 0.0)
        self.assertLessEqual(row["two_opt_distance"], row["nearest_neighbor_distance"])
        self.assertGreaterEqual(row["ortools_gap_pct"], -1e-9)
        self.assertAlmostEqual(
            row["nearest_neighbor_two_opt_distance"],
            row["two_opt_distance"],
        )

    def test_large_benchmark_does_not_run_exact_search(self):
        results = run_benchmark(customer_sizes=(12,), seeds=(2024,), repetitions=1)

        self.assertTrue(pd.isna(results.iloc[0]["exact_distance"]))
        self.assertTrue(pd.isna(results.iloc[0]["brute_force_time_s"]))
        self.assertTrue(pd.isna(results.iloc[0]["nearest_neighbor_gap_pct"]))
        self.assertTrue(pd.isna(results.iloc[0]["ortools_gap_pct"]))

    def test_summary_has_one_row_per_customer_size(self):
        results = run_benchmark(customer_sizes=(5, 12), seeds=(2024,), repetitions=1)
        summary = summarize_results(results)

        self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
