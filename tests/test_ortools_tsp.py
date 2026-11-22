import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from brute_force import brute_force
from ortools_tsp import ortools_tsp
from nearest_neighbor import route_distance


class OrToolsTspTests(unittest.TestCase):
    def setUp(self):
        self.locations = np.array([
            [50.0, 50.0],
            [10.0, 10.0],
            [90.0, 10.0],
            [90.0, 90.0],
            [10.0, 90.0],
        ])

    def test_route_visits_every_customer_and_returns_to_depot(self):
        route, distance = ortools_tsp(self.locations, time_limit_seconds=0.05)

        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 0)
        self.assertEqual(set(route[:-1]), set(range(len(self.locations))))
        self.assertAlmostEqual(distance, route_distance(self.locations, route))

    def test_small_instance_matches_exact_optimum(self):
        _, exact_distance = brute_force(self.locations)
        _, solver_distance = ortools_tsp(self.locations, time_limit_seconds=0.05)

        self.assertLessEqual(solver_distance, exact_distance + 1e-9)

    def test_rejects_invalid_time_limit(self):
        with self.assertRaises(ValueError):
            ortools_tsp(self.locations, time_limit_seconds=0)


if __name__ == "__main__":
    unittest.main()
