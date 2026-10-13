"""Deterministic multi-train fleet simulation built on the existing railway modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import networkx as nx

from decision_engine import OperationalAction, decide_operation
from integrated_simulation import DelayPredictor, STATION_COORDINATES
from railway_network import Route, alternative_routes, build_demo_network, route_edges, shortest_route
from safety_engine import SafetyResult, SignalState, check_route_safety

DEFAULT_FLEET_SIZE = 24
FLEET_SEED = 42
ACTIVE_ROUTE_STATUSES = frozenset({"PROCEED", "REROUTE"})
CONFLICT_POLICY = "Same-step policy: only PROCEED and REROUTE trains reserve their stored route edges; HOLD and STOP trains do not reserve future route edges."

ROUTE_PAIRS = [
    ("Delhi", "Mumbai"),
    ("Delhi", "Ahmedabad"),
    ("Agra", "Mumbai"),
    ("Agra", "Jaipur"),
    ("Bhopal", "Delhi"),
    ("Bhopal", "Ahmedabad"),
    ("Jaipur", "Mumbai"),
    ("Jaipur", "Bhopal"),
    ("Ahmedabad", "Agra"),
    ("Mumbai", "Delhi"),
    ("Mumbai", "Jaipur"),
    ("Ahmedabad", "Bhopal"),
]


@dataclass(frozen=True)
class RouteEvaluation:
    requested_route: Route | None
    safety_result: SafetyResult
    final_route: Route | None
    final_safety_result: SafetyResult | None
    decision: OperationalAction
    reasons: tuple[str, ...]
    candidates: tuple["CandidateEvaluation", ...] = ()


@dataclass(frozen=True)
class CandidateEvaluation:
    route: Route
    safety_result: SafetyResult
    signal_failures: tuple[str, ...]
    blocked_edges: tuple[tuple[str, str], ...]
    occupied_edges: tuple[tuple[str, str], ...]
    conflict_details: tuple[dict[str, Any], ...]

    @property
    def validated(self) -> bool:
        return self.safety_result.allowed

    @property
    def failure_categories(self) -> tuple[str, ...]:
        categories: list[str] = []
        if self.signal_failures:
            categories.append("RED signal")
        if self.blocked_edges:
            categories.append("blocked track")
        if self.occupied_edges:
            categories.append("occupied track")
        if self.conflict_details:
            categories.append("active train conflict")
        if any("exceeds limit" in reason for reason in self.safety_result.reasons):
            categories.append("speed limit")
        return tuple(categories)


def active_conflicting_routes(other_trains: list[dict[str, Any]] | list[list[str]], selected_train_id: str) -> list[list[str]]:
    """Return routes reserved at the current simulated time step."""
    routes: list[list[str]] = []
    for candidate in other_trains:
        if isinstance(candidate, dict):
            if candidate.get("train_id") == selected_train_id:
                continue
            if candidate.get("operational_status") not in ACTIVE_ROUTE_STATUSES:
                continue
            route = candidate.get("final_route") or candidate.get("current_route")
        else:
            route = candidate
        if route:
            routes.append(list(route))
    return routes


def route_conflict_details(route: Route, other_trains: list[dict[str, Any]] | list[list[str]], selected_train_id: str) -> list[dict[str, Any]]:
    route_edges_set = _edge_set(route.stations)
    details: list[dict[str, Any]] = []
    for candidate in other_trains:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("train_id") == selected_train_id or candidate.get("operational_status") not in ACTIVE_ROUTE_STATUSES:
            continue
        candidate_route = candidate.get("final_route") or candidate.get("current_route") or []
        shared = route_edges_set & _edge_set(candidate_route)
        if shared:
            details.append({"train_id": candidate["train_id"], "operational_status": candidate["operational_status"], "shared_edges": [list(edge) for edge in sorted(shared, key=lambda value: sorted(value))], "route": list(candidate_route), "reserved": True})
    return details


def _route_dict(route: Route | None) -> dict[str, Any] | None:
    if route is None:
        return None
    return {"stations": list(route.stations), "cost_km": route.cost_km}


def _edge_set(route: list[str] | tuple[str, ...]) -> set[frozenset[str]]:
    return {frozenset(edge) for edge in route_edges(route)}


def _copy_graph_without_tracks(graph: nx.Graph, tracks: set[tuple[str, str]]) -> nx.Graph:
    available = graph.copy()
    available.remove_edges_from(track for track in tracks if available.has_edge(*track))
    return available


def evaluate_candidate_routes(
    train: dict[str, Any],
    candidates: list[Route],
    graph: nx.Graph,
    signals: dict[str, SignalState],
    occupied_tracks: set[tuple[str, str]],
    other_trains: list[dict[str, Any]] | list[list[str]],
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    seen_routes: set[tuple[str, ...]] = set()
    normalized_occupied = {frozenset(edge) for edge in occupied_tracks}
    for candidate in candidates:
        if candidate.stations in seen_routes:
            continue
        seen_routes.add(candidate.stations)
        edges = route_edges(candidate.stations)
        signal_failures = tuple(f"RED signal at {station}" for station in candidate.stations[:-1] if signals.get(station, SignalState.GREEN) == SignalState.RED)
        blocked_edges = tuple(edge for edge in edges if not graph.has_edge(*edge))
        occupied_edges = tuple(edge for edge in edges if frozenset(edge) in normalized_occupied)
        conflicts = route_conflict_details(candidate, other_trains, train["train_id"])
        safety = check_route_safety(candidate.stations, signals, occupied_tracks, [item["route"] for item in conflicts], train["speed_kmph"])
        if blocked_edges:
            safety = SafetyResult(False, tuple(safety.reasons) + ("A route track is blocked",))
        evaluations.append(CandidateEvaluation(candidate, safety, signal_failures, blocked_edges, occupied_edges, tuple(conflicts)))
    return evaluations


def _train_features(predictor: DelayPredictor, index: int) -> dict[str, Any]:
    return predictor.row_features(index % len(predictor.frame))


def _initial_station(route: Route, index: int) -> str:
    movement_stations = route.stations[:-1]
    return movement_stations[index % len(movement_stations)]


def evaluate_route(
    train: dict[str, Any],
    graph: nx.Graph,
    signals: dict[str, SignalState],
    occupied_tracks: set[tuple[str, str]],
    other_trains: list[dict[str, Any]] | list[list[str]],
    requested_route: Route | None = None,
) -> RouteEvaluation:
    """Evaluate one requested route and return only a validated final route."""
    if requested_route is None:
        try:
            requested_route = shortest_route(graph, train["current_station"], train["destination"])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            requested_route = None
    conflicts = active_conflicting_routes(other_trains, train["train_id"])
    if requested_route is None:
        safety = SafetyResult(False, ("No connected route is available",))
        decision, reason = decide_operation(train["predicted_delay_minutes"], False, safety)
        return RouteEvaluation(None, safety, None, None, decision, (reason,))
    requested_evaluation = evaluate_candidate_routes(train, [requested_route], graph, signals, occupied_tracks, other_trains)[0]
    safety = requested_evaluation.safety_result
    if safety.allowed:
        decision, reason = decide_operation(train["predicted_delay_minutes"], True, safety)
        return RouteEvaluation(requested_route, safety, requested_route, safety, decision, (reason,), (requested_evaluation,))
    available_graph = _copy_graph_without_tracks(graph, occupied_tracks)
    candidates = alternative_routes(available_graph, train["current_station"], train["destination"], limit=5)
    candidate_evaluations = evaluate_candidate_routes(train, candidates, available_graph, signals, occupied_tracks, other_trains)
    unique_candidates: list[CandidateEvaluation] = []
    seen_routes: set[tuple[str, ...]] = set()
    for candidate_evaluation in [requested_evaluation, *candidate_evaluations]:
        if candidate_evaluation.route.stations not in seen_routes:
            seen_routes.add(candidate_evaluation.route.stations)
            unique_candidates.append(candidate_evaluation)
    for candidate_evaluation in candidate_evaluations:
        candidate = candidate_evaluation.route
        if candidate.stations == requested_route.stations:
            continue
        if candidate_evaluation.validated:
            reason = f"Initial route rejected; safe alternative selected ({', '.join(candidate.stations)})"
            return RouteEvaluation(requested_route, safety, candidate, candidate_evaluation.safety_result, OperationalAction.REROUTE, tuple(safety.reasons) + (reason,), tuple(unique_candidates))
    decision, reason = decide_operation(train["predicted_delay_minutes"], False, safety)
    no_alternative_reason = "No safe alternative route was available"
    return RouteEvaluation(requested_route, safety, None, None, decision, tuple(safety.reasons) + (no_alternative_reason, reason), tuple(unique_candidates))


def apply_route_evaluation(train: dict[str, Any], evaluation: RouteEvaluation) -> dict[str, Any]:
    """Persist an evaluation without presenting an attempted route as approved."""
    requested = list(evaluation.requested_route.stations) if evaluation.requested_route else []
    final = list(evaluation.final_route.stations) if evaluation.final_route else []
    train.update(
        {
            "last_attempted_route": requested,
            "current_route": final,
            "final_route": final,
            "alternative_route": _route_dict(evaluation.final_route) if final and final != requested else None,
            "safety_allowed": evaluation.safety_result.allowed,
            "final_safety_allowed": bool(evaluation.final_safety_result and evaluation.final_safety_result.allowed),
            "safety_reasons": list(evaluation.safety_result.reasons),
            "decision_reasons": list(evaluation.reasons),
            "current_decision": evaluation.decision.value,
            "operational_status": evaluation.decision.value,
            "decision_reason": evaluation.reasons[-1] if evaluation.reasons else "No decision reason recorded",
        }
    )
    return train


def _base_train(predictor: DelayPredictor, index: int, origin: str, destination: str, route: Route) -> dict[str, Any]:
    features = _train_features(predictor, index * 37)
    status = predictor.predict(features)
    current_station = _initial_station(route, index)
    latitude, longitude = STATION_COORDINATES[current_station]
    return {
        "train_id": f"SIM-{index + 1:03d}",
        "origin": origin,
        "destination": destination,
        "current_station": current_station,
        "latitude": latitude,
        "longitude": longitude,
        "speed_kmph": float(48 + ((index * 17) % 68)),
        "direction": f"{current_station} -> {destination}",
        "predicted_delay_status": status,
        "predicted_delay_minutes": 60 if status else 0,
        "model_features": features,
        "scenario": "fleet_simulation",
        "initial_route": list(route.stations),
        "last_attempted_route": list(route.stations),
        "current_route": list(route.stations),
        "final_route": list(route.stations),
        "alternative_route": None,
        "operational_status": "CREATED",
        "current_decision": "PENDING",
        "decision_reason": "Awaiting deterministic safety evaluation",
        "safety_allowed": True,
        "final_safety_allowed": True,
        "safety_reasons": [],
        "signal_state": SignalState.GREEN.value,
        "restricted_signal": None,
        "blocked_track": None,
        "occupied_tracks": [],
    }


def evaluate_fleet_train(
    train: dict[str, Any],
    graph: nx.Graph,
    signals: dict[str, SignalState],
    occupied_tracks: set[tuple[str, str]],
    conflicting_routes: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Recalculate one train using graph availability, safety, and decisions."""
    other_trains: list[dict[str, Any]] | list[list[str]] = conflicting_routes or []
    evaluation = evaluate_route(train, graph, signals, occupied_tracks, other_trains)
    apply_route_evaluation(train, evaluation)
    train["direction"] = f"{train['current_station']} -> {train['destination']}"
    train["signal_state"] = signals.get(train["current_station"], SignalState.GREEN).value
    train["occupied_tracks"] = [list(track) for track in occupied_tracks if evaluation.requested_route and frozenset(track) in _edge_set(evaluation.requested_route.stations)]
    return train


def generate_fleet(
    predictor: DelayPredictor,
    count: int = DEFAULT_FLEET_SIZE,
    seed: int = FLEET_SEED,
) -> list[dict[str, Any]]:
    """Generate a stable, connected fleet with repeatable IDs and varied states."""
    if not 20 <= count <= 50:
        raise ValueError("Fleet size must be between 20 and 50 trains")
    if seed != FLEET_SEED:
        raise ValueError(f"This demonstration supports deterministic seed {FLEET_SEED}")
    graph = build_demo_network()
    fleet: list[dict[str, Any]] = []
    for index in range(count):
        origin, destination = ROUTE_PAIRS[index % len(ROUTE_PAIRS)]
        route = shortest_route(graph, origin, destination)
        train = _base_train(predictor, index, origin, destination, route)
        train_signals = {station: SignalState.GREEN for station in STATION_COORDINATES}
        train_occupied_tracks: set[tuple[str, str]] = set()
        if index % 9 == 3 and len(route.stations) > 2:
            train_signals[route.stations[1]] = SignalState.RED
            train["restricted_signal"] = route.stations[1]
        if index % 6 == 1:
            first_edge = route_edges(route.stations)[0]
            train_occupied_tracks.add(first_edge)
            train["blocked_track"] = list(first_edge)
        evaluate_fleet_train(train, graph, train_signals, train_occupied_tracks)
        fleet.append(train)
    return fleet


def reset_fleet(predictor: DelayPredictor, count: int = DEFAULT_FLEET_SIZE) -> list[dict[str, Any]]:
    return generate_fleet(predictor, count=count, seed=FLEET_SEED)


def fleet_summary(fleet: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "total": len(fleet),
        "delayed": sum(train["predicted_delay_status"] for train in fleet),
        "stopped": sum(train["operational_status"] == "STOP" for train in fleet),
        "rerouted": sum(train["operational_status"] == "REROUTE" for train in fleet),
        "on_hold": sum(train["operational_status"] == "HOLD" for train in fleet),
        "safety_alerts": sum(not train["safety_allowed"] for train in fleet),
        "average_delay_minutes": sum(train["predicted_delay_minutes"] for train in fleet) / max(len(fleet), 1),
    }


def reset_train(train: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(train)
