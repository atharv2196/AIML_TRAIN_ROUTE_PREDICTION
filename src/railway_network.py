"""Small graph-based railway network simulation for educational use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class Route:
    stations: tuple[str, ...]
    cost_km: float


def build_demo_network() -> nx.Graph:
    network = nx.Graph()
    network.add_weighted_edges_from(
        [
            ("Delhi", "Agra", 200),
            ("Agra", "Bhopal", 500),
            ("Bhopal", "Mumbai", 650),
            ("Delhi", "Jaipur", 280),
            ("Jaipur", "Ahmedabad", 650),
            ("Ahmedabad", "Mumbai", 530),
            ("Agra", "Jaipur", 240),
        ],
        weight="distance_km",
    )
    return network


def shortest_route(network: nx.Graph, source: str, destination: str) -> Route:
    stations = nx.shortest_path(network, source, destination, weight="distance_km")
    cost = nx.path_weight(network, stations, weight="distance_km")
    return Route(tuple(stations), float(cost))


def alternative_routes(
    network: nx.Graph, source: str, destination: str, limit: int = 3
) -> list[Route]:
    routes = []
    paths = nx.shortest_simple_paths(network, source, destination, weight="distance_km")
    for path in paths:
        routes.append(Route(tuple(path), float(nx.path_weight(network, path, weight="distance_km"))))
        if len(routes) == limit:
            break
    return routes


def block_track(network: nx.Graph, station_a: str, station_b: str) -> None:
    if not network.has_edge(station_a, station_b):
        raise ValueError(f"Track does not exist: {station_a} - {station_b}")
    network[station_a][station_b]["blocked"] = True
    network.remove_edge(station_a, station_b)


def route_edges(route: Iterable[str]) -> list[tuple[str, str]]:
    stations = list(route)
    return list(zip(stations, stations[1:]))


if __name__ == "__main__":
    graph = build_demo_network()
    print(shortest_route(graph, "Delhi", "Mumbai"))
    print(alternative_routes(graph, "Delhi", "Mumbai"))
