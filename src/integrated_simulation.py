"""Integrated educational railway simulation with repeatable scenarios."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from decision_engine import OperationalAction, decide_operation
from railway_network import Route, alternative_routes, block_track, build_demo_network, shortest_route
from safety_engine import SafetyResult, SignalState, check_route_safety

LEAKAGE_COLUMNS = {"is_delayed", "delay_minutes", "primary_delay_cause", "journey_id", "departure_date"}
STATION_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Agra": (27.1767, 78.0081),
    "Bhopal": (23.2599, 77.4126),
    "Mumbai": (19.0760, 72.8777),
    "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714),
}


@dataclass
class IntegratedTrainState:
    train_id: str
    current_station: str
    destination: str
    latitude: float
    longitude: float
    speed_kmph: float
    predicted_delay_status: int
    current_route: tuple[str, ...]
    operational_status: str
    model_features: dict[str, Any]


@dataclass
class ScenarioResult:
    scenario: str
    initial_state: dict[str, Any]
    predicted_delay_status: int
    initial_route: dict[str, Any]
    initial_safety: dict[str, Any]
    decision: str
    reason: str
    final_route: dict[str, Any]
    final_state: dict[str, Any]


class DelayPredictor:
    def __init__(self, model_path: Path, data_path: Path, rows: int = 20_000) -> None:
        self.pipeline = joblib.load(model_path)
        self.frame = pd.read_csv(data_path, nrows=rows)

    def row_features(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {key: value for key, value in row.drop(labels=list(LEAKAGE_COLUMNS), errors="ignore").to_dict().items()}

    def predict(self, features: dict[str, Any]) -> int:
        values = self.pipeline.predict(pd.DataFrame([features]))
        return int(values[0])

    def find_row(self, desired_status: int | None = None) -> tuple[int, dict[str, Any], int]:
        for index in range(len(self.frame)):
            features = self.row_features(index)
            status = self.predict(features)
            if desired_status is None or status == desired_status:
                return index, features, status
        raise RuntimeError("No suitable local training row was found for the simulation")


def route_dict(route: Route) -> dict[str, Any]:
    return {"stations": list(route.stations), "cost_km": route.cost_km}


def state_for_row(predictor: DelayPredictor, desired_status: int | None = None) -> IntegratedTrainState:
    index, features, status = predictor.find_row(desired_status)
    row = predictor.frame.iloc[index]
    station = "Delhi"
    destination = "Mumbai"
    latitude, longitude = STATION_COORDINATES[station]
    return IntegratedTrainState(
        train_id=str(row["journey_id"]),
        current_station=station,
        destination=destination,
        latitude=latitude,
        longitude=longitude,
        speed_kmph=80.0,
        predicted_delay_status=status,
        current_route=(),
        operational_status="CREATED",
        model_features=features,
    )


def update_after_valid_action(state: IntegratedTrainState, route: Route, action: OperationalAction) -> None:
    state.current_route = route.stations
    if action in {OperationalAction.PROCEED, OperationalAction.REROUTE} and len(route.stations) > 1:
        state.current_station = route.stations[1]
        state.latitude, state.longitude = STATION_COORDINATES[state.current_station]
        state.operational_status = action.value
    else:
        state.operational_status = action.value


def find_safe_alternative(graph, source: str, destination: str, signals, occupied, conflicts) -> tuple[Route | None, SafetyResult | None]:
    for candidate in alternative_routes(graph, source, destination):
        result = check_route_safety(candidate.stations, signals, occupied, conflicts)
        if result.allowed:
            return candidate, result
    return None, None


def run_scenario(name: str, predictor: DelayPredictor, blocked_edge: tuple[str, str] | None = None, red_station: str | None = None) -> ScenarioResult:
    graph = build_demo_network()
    state = state_for_row(predictor, desired_status=0 if name == "normal_operation" else None)
    initial_route = shortest_route(graph, state.current_station, state.destination)
    state.current_route = initial_route.stations
    signals = {red_station: SignalState.RED} if red_station else {}
    occupied = {blocked_edge} if blocked_edge else set()
    if blocked_edge:
        block_track(graph, *blocked_edge)
    initial_state = asdict(state)
    initial_safety = check_route_safety(initial_route.stations, signals, occupied)
    predicted_minutes = 60 if state.predicted_delay_status else 0
    action, reason = decide_operation(predicted_minutes, True, initial_safety)
    final_route = initial_route
    final_safety = initial_safety
    if action == OperationalAction.REROUTE:
        final_route, final_safety = find_safe_alternative(graph, state.current_station, state.destination, signals, occupied, [])
        if final_route is None:
            action = OperationalAction.STOP
            reason = "No safe alternative route was available"
            final_route = initial_route
            final_safety = initial_safety
        else:
            reason = f"Initial route rejected; safe alternative selected ({', '.join(final_route.stations)})"
    update_after_valid_action(state, final_route, action)
    return ScenarioResult(
        scenario=name,
        initial_state=initial_state,
        predicted_delay_status=state.predicted_delay_status,
        initial_route=route_dict(initial_route),
        initial_safety=asdict(initial_safety),
        decision=action.value,
        reason=reason,
        final_route=route_dict(final_route),
        final_state=asdict(state),
    )


def print_result(result: ScenarioResult) -> None:
    print(f"\n=== {result.scenario.upper()} ===")
    print(f"Initial state: {result.initial_state}")
    print(f"Predicted delay status: {result.predicted_delay_status}")
    print(f"Initial route: {result.initial_route}")
    print(f"Safety result: {result.initial_safety}")
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print(f"Final route: {result.final_route}")
    print(f"Final state: {result.final_state}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/ir_train.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/random_forest.joblib"))
    args = parser.parse_args()
    predictor = DelayPredictor(args.model, args.data)
    results = [
        run_scenario("normal_operation", predictor),
        run_scenario("congestion_alternative_route", predictor, blocked_edge=("Delhi", "Agra")),
        run_scenario("red_signal_safety_event", predictor, red_station="Agra"),
    ]
    Path("results").mkdir(exist_ok=True)
    Path("results/integrated_scenarios.json").write_text(json.dumps([asdict(result) for result in results], indent=2, default=str), encoding="utf-8")
    Path("results/integrated_scenarios.txt").write_text("\n".join(f"{result.scenario}: {result.decision} - {result.reason}" for result in results) + "\n", encoding="utf-8")
    for result in results:
        print_result(result)
    print("\nScenario results saved to results/integrated_scenarios.json and results/integrated_scenarios.txt")


if __name__ == "__main__":
    main()
