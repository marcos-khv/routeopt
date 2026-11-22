import sys
import unittest

import numpy as np

sys.path.insert(0, "src")

from brute_force import brute_force
from nearest_neighbor import nearest_neighbor, route_distance
from two_opt import two_opt


class TwoOptTests(unittest.TestCase):
    def setUp(self):
        self.locations = np.array([
            [0, 0],
            [8, 1],
            [9, 8],
            [1, 9],
            [4, 4],
        ])

    def test_two_opt_does_not_worsen_route(self):
        initial = nearest_neighbor(self.locations)
        improved, distance = two_opt(self.locations, initial)

        self.assertEqual(improved[0], 0)
        self.assertEqual(improved[-1], 0)
        self.assertEqual(set(improved[:-1]), set(range(len(self.locations))))
        self.assertLessEqual(distance, route_distance(self.locations, initial) + 1e-12)

    def test_two_opt_reaches_brute_force_optimum_for_small_instance(self):
        optimal_route, optimal_distance = brute_force(self.locations)
        initial = nearest_neighbor(self.locations)
        improved_route, improved_distance = two_opt(self.locations, initial)

        self.assertLessEqual(improved_distance, route_distance(self.locations, initial) + 1e-12)
        self.assertAlmostEqual(improved_distance, optimal_distance, places=12)

    def test_two_opt_rejects_open_route(self):
        with self.assertRaises(ValueError):
            two_opt(self.locations, [0, 1, 2, 3, 4])

    def test_single_customer_route_is_valid(self):
        locations = np.array([[0, 0], [3, 4]])
        route = [0, 1, 0]

        improved, distance = two_opt(locations, route)

        self.assertEqual(improved, route)
        self.assertAlmostEqual(distance, 10.0)


if __name__ == "__main__":
    unittest.main()
