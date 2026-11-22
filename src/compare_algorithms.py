
from time import perf_counter

from nearest_neighbor import (
    locations,
    nearest_neighbor,
    route_distance,
)

from brute_force import brute_force
from two_opt import two_opt


# Nearest Neighbor
start = perf_counter()

nn_route = nearest_neighbor(locations)
nn_distance = route_distance(locations, nn_route)

nn_time = perf_counter() - start


# Brute Force
start = perf_counter()

optimal_route, optimal_distance = brute_force(locations)

bf_time = perf_counter() - start


# 2-opt
start = perf_counter()

improved_route, improved_distance = two_opt(locations, nn_route)

opt_time = perf_counter() - start


# Calculate optimality gaps
nn_gap = (
    (nn_distance - optimal_distance)
    / optimal_distance
) * 100

opt_gap = (
    (improved_distance - optimal_distance)
    / optimal_distance
) * 100


print("ROUTEOPT - ALGORITHM COMPARISON")
print("--------------------------------")
print(f"Nearest Neighbor: {nn_distance:.2f} units")
print(f"2-opt:             {improved_distance:.2f} units")
print(f"Brute Force:      {optimal_distance:.2f} units")
print(f"NN optimality gap: {nn_gap:.2f}%")
print(f"2-opt optimality gap: {opt_gap:.2f}%")
print()
print(f"NN execution time: {nn_time:.6f} seconds")
print(f"2-opt execution time: {opt_time:.6f} seconds")
print(f"BF execution time: {bf_time:.6f} seconds")
