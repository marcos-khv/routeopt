
from time import perf_counter

from nearest_neighbor import (
    locations,
    nearest_neighbor,
    route_distance,
)

from brute_force import brute_force


# Nearest Neighbor
start = perf_counter()

nn_route = nearest_neighbor(locations)
nn_distance = route_distance(locations, nn_route)

nn_time = perf_counter() - start


# Brute Force
start = perf_counter()

optimal_route, optimal_distance = brute_force(locations)

bf_time = perf_counter() - start


# Calculate optimality gap
gap = (
    (nn_distance - optimal_distance)
    / optimal_distance
) * 100


print("ROUTEOPT - ALGORITHM COMPARISON")
print("--------------------------------")
print(f"Nearest Neighbor: {nn_distance:.2f} units")
print(f"Brute Force:      {optimal_distance:.2f} units")
print(f"Optimality Gap:   {gap:.2f}%")
print()
print(f"NN execution time: {nn_time:.6f} seconds")
print(f"BF execution time: {bf_time:.6f} seconds")
