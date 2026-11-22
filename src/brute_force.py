
from itertools import permutations
from nearest_neighbor import locations, route_distance


def brute_force(locations, start=0):
    """Find the optimal route by evaluating all permutations."""

    customers = [i for i in range(len(locations)) if i != start]

    best_route = None
    best_distance = float("inf")

    for ordering in permutations(customers):
        route = [start] + list(ordering) + [start]
        distance = route_distance(locations, route)

        if distance < best_distance:
            best_distance = distance
            best_route = route

    return best_route, best_distance


if __name__ == "__main__":
    route, distance = brute_force(locations)

    print("ROUTEOPT - BRUTE FORCE")
    print("----------------------")
    print(f"Optimal route: {route}")
    print(f"Optimal distance: {distance:.2f} units")