"""Two-opt local search for symmetric Traveling Salesperson Problems."""

from nearest_neighbor import calculate_distance, locations, nearest_neighbor, route_distance


def two_opt(locations, route):
    """Improve a closed route by repeatedly reversing improving segments.

    The first and last indices must both identify the depot. Returns a tuple
    containing the improved route and its total distance.
    """
    if len(route) < 4:
        return list(route), route_distance(locations, route)

    if route[0] != route[-1]:
        raise ValueError("The route must start and end at the same depot")

    improved_route = list(route)
    improved = True

    while improved:
        improved = False
        best_distance = route_distance(locations, improved_route)

        for i in range(1, len(improved_route) - 2):
            for j in range(i + 1, len(improved_route) - 1):
                candidate = improved_route.copy()
                candidate[i:j + 1] = reversed(candidate[i:j + 1])

                candidate_distance = route_distance(locations, candidate)

                if candidate_distance < best_distance:
                    improved_route = candidate
                    best_distance = candidate_distance
                    improved = True

    return improved_route, best_distance


if __name__ == "__main__":
    initial_route = nearest_neighbor(locations)
    improved_route, distance = two_opt(locations, initial_route)

    print("ROUTEOPT - 2-OPT")
    print("----------------")
    print(f"Initial route: {initial_route}")
    print(f"Improved route: {improved_route}")
    print(f"Improved distance: {distance:.2f} units")
