
import numpy as np
import matplotlib.pyplot as plt

# Synthetic coordinates
# Index 0 represents the depot
locations = np.array([
    [2, 2],  # Depot
    [5, 8],  # Customer 1
    [8, 3],  # Customer 2
    [1, 7],  # Customer 3
    [9, 9],  # Customer 4
    [6, 5],  # Customer 5
])


def calculate_distance(point_a, point_b):
    """Calculate Euclidean distance between two locations."""
    return np.linalg.norm(point_a - point_b)


def nearest_neighbor(locations, start=0):
    """Construct a route using the nearest neighbor heuristic."""

    unvisited = set(range(len(locations)))
    unvisited.remove(start)

    route = [start]
    current = start

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda i: (
                calculate_distance(locations[current], locations[i]),
                i,
            ),
        )

        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Return to the depot
    route.append(start)

    return route


def route_distance(locations, route):
    """Calculate the total distance travelled."""
    return sum(
        calculate_distance(locations[a], locations[b])
        for a, b in zip(route[:-1], route[1:])
    )


def plot_route(locations, route):
    """Visualize the route and delivery locations."""

    ordered = locations[route]

    plt.figure(figsize=(8, 6))

    plt.plot(
        ordered[:, 0],
        ordered[:, 1],
        marker="o",
        linestyle="-",
    )

    for i, (x, y) in enumerate(locations):
        label = "Depot" if i == 0 else f"Customer {i}"

        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("RouteOpt - Nearest Neighbor")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    route = nearest_neighbor(locations)
    distance = route_distance(locations, route)

    print("ROUTEOPT - FIRST EXPERIMENT")
    print("---------------------------")
    print(f"Route: {route}")
    print(f"Total distance: {distance:.2f} units")

    plot_route(locations, route)
