"""Deterministic safety checks for the railway simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from railway_network import route_edges


class SignalState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reasons: tuple[str, ...]


def check_route_safety(
    route: list[str] | tuple[str, ...],
    signals: dict[str, SignalState],
    occupied_tracks: set[tuple[str, str]],
    conflicting_routes: list[list[str]] | None = None,
    speed_kmph: float = 0.0,
    speed_limit_kmph: float = 120.0,
) -> SafetyResult:
    reasons: list[str] = []
    edges = route_edges(route)
    for station in route[:-1]:
        if signals.get(station, SignalState.GREEN) == SignalState.RED:
            reasons.append(f"RED signal at {station}")
    normalized_occupied = {frozenset(track) for track in occupied_tracks}
    if any(frozenset(edge) in normalized_occupied for edge in edges):
        reasons.append("A route track is occupied")
    for conflict in conflicting_routes or []:
        if any(frozenset(edge) in {frozenset(item) for item in route_edges(conflict)} for edge in edges):
            reasons.append("The proposed route conflicts with another route")
            break
    if speed_kmph > speed_limit_kmph:
        reasons.append(f"Speed {speed_kmph:g} exceeds limit {speed_limit_kmph:g} km/h")
    return SafetyResult(allowed=not reasons, reasons=tuple(reasons))
