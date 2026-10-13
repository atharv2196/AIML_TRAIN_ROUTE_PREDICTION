"""Explainable rule-based operational decisions for the simulation."""

from __future__ import annotations

from enum import Enum

from safety_engine import SafetyResult


class OperationalAction(str, Enum):
    PROCEED = "PROCEED"
    HOLD = "HOLD"
    REROUTE = "REROUTE"
    STOP = "STOP"


def decide_operation(
    predicted_delay_minutes: float,
    route_available: bool,
    safety: SafetyResult,
) -> tuple[OperationalAction, str]:
    if not safety.allowed:
        if route_available:
            return OperationalAction.REROUTE, "Safety check rejected the proposed route"
        return OperationalAction.STOP, "Safety check rejected the route and no alternative is available"
    if not route_available:
        return OperationalAction.HOLD, "No route is currently available"
    if predicted_delay_minutes >= 60:
        return OperationalAction.HOLD, "Predicted delay is at least 60 minutes"
    return OperationalAction.PROCEED, "Route is available and safety checks passed"
