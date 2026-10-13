"""Automated tests for routing, safety, decisions, and model integration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib

from decision_engine import OperationalAction, decide_operation
from integrated_simulation import DelayPredictor, run_scenario
from railway_network import build_demo_network, block_track, shortest_route
from safety_engine import SafetyResult, SignalState, check_route_safety


class RailwaySystemTests(unittest.TestCase):
    def test_shortest_route(self) -> None:
        route = shortest_route(build_demo_network(), "Delhi", "Mumbai")
        self.assertEqual(route.stations, ("Delhi", "Agra", "Bhopal", "Mumbai"))
        self.assertEqual(route.cost_km, 1350.0)

    def test_alternative_route_after_blocking_track(self) -> None:
        graph = build_demo_network()
        block_track(graph, "Delhi", "Agra")
        route = shortest_route(graph, "Delhi", "Mumbai")
        self.assertEqual(route.stations, ("Delhi", "Jaipur", "Ahmedabad", "Mumbai"))

    def test_red_signal_rejection(self) -> None:
        result = check_route_safety(("Delhi", "Agra", "Mumbai"), {"Agra": SignalState.RED}, set())
        self.assertFalse(result.allowed)
        self.assertIn("RED signal at Agra", result.reasons)

    def test_occupied_track_rejection(self) -> None:
        result = check_route_safety(("Delhi", "Agra", "Mumbai"), {}, {("Delhi", "Agra")})
        self.assertFalse(result.allowed)
        self.assertIn("A route track is occupied", result.reasons)

    def test_decision_outcomes(self) -> None:
        safe = SafetyResult(True, ())
        unsafe = SafetyResult(False, ("RED signal at Agra",))
        self.assertEqual(decide_operation(0, True, safe)[0], OperationalAction.PROCEED)
        self.assertEqual(decide_operation(60, True, safe)[0], OperationalAction.HOLD)
        self.assertEqual(decide_operation(0, True, unsafe)[0], OperationalAction.REROUTE)
        self.assertEqual(decide_operation(0, False, unsafe)[0], OperationalAction.STOP)

    def test_saved_model_loading_and_prediction(self) -> None:
        model = joblib.load(ROOT / "models" / "random_forest.joblib")
        predictor = DelayPredictor(ROOT / "models" / "random_forest.joblib", ROOT / "datasets" / "ir_train.csv")
        _, features, status = predictor.find_row()
        self.assertTrue(hasattr(model, "predict"))
        self.assertIn(status, (0, 1))
        self.assertIn(predictor.predict(features), (0, 1))

    def test_integrated_scenarios(self) -> None:
        predictor = DelayPredictor(ROOT / "models" / "random_forest.joblib", ROOT / "datasets" / "ir_train.csv")
        normal = run_scenario("normal_operation", predictor)
        congestion = run_scenario("congestion_alternative_route", predictor, blocked_edge=("Delhi", "Agra"))
        red_signal = run_scenario("red_signal_safety_event", predictor, red_station="Agra")
        self.assertEqual(normal.decision, "PROCEED")
        self.assertEqual(congestion.decision, "REROUTE")
        self.assertEqual(red_signal.decision, "REROUTE")
        self.assertNotEqual(congestion.initial_state["current_station"], congestion.final_state["current_station"])
        self.assertFalse(red_signal.initial_safety["allowed"])


if __name__ == "__main__":
    unittest.main()
