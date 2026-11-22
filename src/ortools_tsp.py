"""Traveling Salesperson Problem solver using Google OR-Tools."""

from __future__ import annotations

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from nearest_neighbor import calculate_distance, route_distance


def ortools_tsp(
    locations: np.ndarray,
    time_limit_seconds: float = 0.2,
    distance_scale: int = 1000,
) -> tuple[list[int], float]:
    """Solve a single-depot Euclidean TSP with OR-Tools RoutingModel.

    OR-Tools requires integer arc costs, so Euclidean distances are rounded
    after multiplication by ``distance_scale``. The returned distance is
    recalculated using the original floating-point coordinates.
    """
    if len(locations) < 2:
        raise ValueError("At least one depot and one customer are required")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be greater than zero")
    if distance_scale <= 0:
        raise ValueError("distance_scale must be greater than zero")

    node_count = len(locations)
    distance_matrix = [
        [
            int(round(calculate_distance(locations[i], locations[j]) * distance_scale))
            for j in range(node_count)
        ]
        for i in range(node_count)
    ]

    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    whole_seconds = int(time_limit_seconds)
    search_parameters.time_limit.seconds = whole_seconds
    search_parameters.time_limit.nanos = int(
        (time_limit_seconds - whole_seconds) * 1_000_000_000
    )

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools did not find a feasible TSP route")

    route = [0]
    index = routing.Start(0)
    while not routing.IsEnd(index):
        index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

    return route, route_distance(locations, route)
