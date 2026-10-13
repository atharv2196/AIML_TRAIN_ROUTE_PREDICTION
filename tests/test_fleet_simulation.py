"""Tests for the deterministic multi-train fleet simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_simulation import DEFAULT_FLEET_SIZE, apply_route_evaluation, evaluate_candidate_routes, evaluate_route, generate_fleet, reset_fleet
from integrated_simulation import DelayPredictor
from railway_network import Route, build_demo_network, block_track, route_edges
from safety_engine import SignalState


class FleetSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = DelayPredictor(ROOT / "models" / "random_forest.joblib", ROOT / "datasets" / "ir_train.csv")
        cls.fleet = generate_fleet(cls.predictor)

    def test_default_fleet_has_unique_stable_ids(self) -> None:
        self.assertEqual(len(self.fleet), DEFAULT_FLEET_SIZE)
        self.assertEqual(len({train["train_id"] for train in self.fleet}), DEFAULT_FLEET_SIZE)
        reset = reset_fleet(self.predictor)
        self.assertEqual([train["train_id"] for train in self.fleet], [train["train_id"] for train in reset])
        self.assertEqual([train["current_route"] for train in self.fleet], [train["current_route"] for train in reset])

    def test_routes_are_connected_and_have_supported_edges(self) -> None:
        graph = build_demo_network()
        for train in self.fleet:
            self.assertGreaterEqual(len(train["current_route"]), 2)
            for edge in route_edges(train["current_route"]):
                self.assertTrue(graph.has_edge(*edge))

    def test_fleet_contains_varied_predictions_and_decisions(self) -> None:
        decisions = {train["current_decision"] for train in self.fleet}
        predictions = {train["predicted_delay_status"] for train in self.fleet}
        self.assertEqual(predictions, {0, 1})
        self.assertIn("PROCEED", decisions)
        self.assertIn("HOLD", decisions)
        self.assertIn("REROUTE", decisions)

    def test_safety_alerts_have_explicit_reasons(self) -> None:
        alerts = [train for train in self.fleet if not train["safety_allowed"]]
        self.assertGreaterEqual(len(alerts), 1)
        self.assertTrue(all(train["safety_reasons"] for train in alerts))
        self.assertTrue(all(train["current_decision"] in {"REROUTE", "STOP"} for train in alerts))

    def _train(self, current_station: str = "Delhi", destination: str = "Mumbai") -> dict:
        return {
            "train_id": "TEST-001",
            "current_station": current_station,
            "destination": destination,
            "speed_kmph": 80.0,
            "predicted_delay_minutes": 0,
            "operational_status": "PROCEED",
            "final_route": [],
            "current_route": [],
        }

    def test_blocked_track_is_avoided(self) -> None:
        graph = build_demo_network()
        block_track(graph, "Delhi", "Agra")
        evaluation = evaluate_route(self._train(), graph, {}, set(), [])
        self.assertTrue(evaluation.safety_result.allowed)
        self.assertEqual(evaluation.requested_route.stations, ("Delhi", "Jaipur", "Ahmedabad", "Mumbai"))
        self.assertEqual(evaluation.final_route.stations, evaluation.requested_route.stations)

    def test_red_signal_rejects_route_and_unrelated_red_does_not(self) -> None:
        graph = build_demo_network()
        rejected = evaluate_route(self._train(), graph, {"Agra": SignalState.RED}, set(), [])
        unrelated = evaluate_route(self._train(), graph, {"Jaipur": SignalState.RED}, set(), [])
        self.assertFalse(rejected.safety_result.allowed)
        self.assertIn("RED signal at Agra", rejected.safety_result.reasons)
        self.assertTrue(rejected.final_route is not None and rejected.final_route.stations != rejected.requested_route.stations)
        self.assertTrue(unrelated.safety_result.allowed)

    def test_unrelated_signals_do_not_reject_requested_route(self) -> None:
        graph = build_demo_network()
        route = Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0)
        train = self._train()
        evaluation = evaluate_route(train, graph, {"Agra": SignalState.RED, "Bhopal": SignalState.RED}, set(), [], requested_route=route)
        self.assertTrue(evaluation.safety_result.allowed)
        self.assertNotIn("RED signal at Agra", evaluation.safety_result.reasons)
        self.assertNotIn("RED signal at Bhopal", evaluation.safety_result.reasons)

    def test_route_signal_rejects_only_when_station_is_on_route(self) -> None:
        graph = build_demo_network()
        route = Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0)
        evaluation = evaluate_route(self._train(), graph, {"Jaipur": SignalState.RED}, set(), [], requested_route=route)
        self.assertFalse(evaluation.safety_result.allowed)
        self.assertIn("RED signal at Jaipur", evaluation.safety_result.reasons)

    def test_active_conflict_is_separate_from_unrelated_signal_failure(self) -> None:
        graph = build_demo_network()
        route = Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0)
        active_train = {"train_id": "SIM-013", "operational_status": "PROCEED", "final_route": list(route.stations)}
        evaluation = evaluate_route(self._train(), graph, {"Agra": SignalState.RED, "Bhopal": SignalState.RED}, set(), [active_train], requested_route=route)
        self.assertFalse(evaluation.safety_result.allowed)
        self.assertIn("The proposed route conflicts with another route", evaluation.safety_result.reasons)
        self.assertNotIn("RED signal at Agra", evaluation.safety_result.reasons)
        self.assertNotIn("RED signal at Bhopal", evaluation.safety_result.reasons)

    def test_occupied_track_rejection_and_safe_alternative(self) -> None:
        graph = build_demo_network()
        evaluation = evaluate_route(self._train(), graph, {}, {("Delhi", "Agra")}, [])
        self.assertFalse(evaluation.safety_result.allowed)
        self.assertIn("A route track is occupied", evaluation.safety_result.reasons)
        self.assertEqual(evaluation.decision.value, "REROUTE")
        self.assertNotEqual(evaluation.final_route.stations, evaluation.requested_route.stations)

    def test_active_conflict_is_checked_but_hold_route_is_not_reserved(self) -> None:
        graph = build_demo_network()
        active = {"train_id": "TEST-002", "operational_status": "PROCEED", "final_route": ["Delhi", "Agra", "Bhopal", "Mumbai"]}
        held = {"train_id": "TEST-002", "operational_status": "HOLD", "final_route": ["Delhi", "Agra", "Bhopal", "Mumbai"]}
        conflict = evaluate_route(self._train(), graph, {}, set(), [active])
        no_conflict = evaluate_route(self._train(), graph, {}, set(), [held])
        self.assertIn("The proposed route conflicts with another route", conflict.safety_result.reasons)
        self.assertFalse(conflict.safety_result.allowed)
        self.assertTrue(no_conflict.safety_result.allowed)

    def test_no_safe_alternative_returns_stop_and_no_final_route(self) -> None:
        graph = nx.Graph()
        graph.add_weighted_edges_from([("Delhi", "Agra", 200), ("Agra", "Bhopal", 500), ("Bhopal", "Mumbai", 650)], weight="distance_km")
        evaluation = evaluate_route(self._train(), graph, {"Agra": SignalState.RED}, set(), [])
        self.assertEqual(evaluation.decision.value, "STOP")
        self.assertIsNone(evaluation.final_route)
        self.assertIn("No safe alternative route was available", evaluation.reasons)

    def test_apply_evaluation_persists_only_validated_final_route(self) -> None:
        train = self._train()
        evaluation = evaluate_route(train, build_demo_network(), {}, {("Delhi", "Agra")}, [])
        apply_route_evaluation(train, evaluation)
        self.assertEqual(train["last_attempted_route"], list(evaluation.requested_route.stations))
        self.assertEqual(train["final_route"], list(evaluation.final_route.stations))
        self.assertNotEqual(train["last_attempted_route"], train["final_route"])

    def test_reproduced_blocked_route_reports_each_unsafe_candidate(self) -> None:
        graph = build_demo_network()
        block_track(graph, "Delhi", "Agra")
        signals = {"Agra": SignalState.RED, "Bhopal": SignalState.RED}
        route = Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0)
        train = self._train()
        active = [
            {"train_id": "SIM-013", "operational_status": "REROUTE", "final_route": list(route.stations)},
            {"train_id": "SIM-003", "operational_status": "PROCEED", "final_route": ["Agra", "Bhopal", "Mumbai"]},
        ]
        evaluation = evaluate_route(train, graph, signals, set(), active, requested_route=route)
        self.assertIsNone(evaluation.final_route)
        self.assertEqual(evaluation.decision.value, "STOP")
        self.assertGreaterEqual(len(evaluation.candidates), 2)
        self.assertTrue(all(not candidate.validated for candidate in evaluation.candidates))
        all_reasons = [reason for candidate in evaluation.candidates for reason in candidate.safety_result.reasons]
        self.assertIn("RED signal at Agra", all_reasons)
        self.assertIn("RED signal at Bhopal", all_reasons)
        self.assertIn("The proposed route conflicts with another route", all_reasons)
        self.assertTrue(any(detail["train_id"] == "SIM-013" for candidate in evaluation.candidates for detail in candidate.conflict_details))

    def test_candidate_routes_deduplicate_by_ordered_station_sequence(self) -> None:
        graph = build_demo_network()
        routes = [
            Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0),
            Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0),
            Route(("Delhi", "Agra", "Bhopal", "Mumbai"), 1350.0),
        ]
        evaluations = evaluate_candidate_routes(self._train(), routes, graph, {}, set(), [{"train_id": "ACTIVE-001", "operational_status": "PROCEED", "final_route": list(routes[0].stations)}])
        self.assertEqual([evaluation.route.stations for evaluation in evaluations], [routes[0].stations, routes[2].stations])
        self.assertEqual(len({evaluation.route.stations for evaluation in evaluations}), 2)
        self.assertFalse(evaluations[0].safety_result.allowed)
        self.assertTrue(evaluations[1].safety_result.allowed)

    def test_route_evaluation_deduplicates_requested_and_generated_candidate(self) -> None:
        graph = build_demo_network()
        block_track(graph, "Delhi", "Agra")
        requested = Route(("Delhi", "Jaipur", "Ahmedabad", "Mumbai"), 1460.0)
        active_train = {"train_id": "ACTIVE-001", "operational_status": "PROCEED", "final_route": list(requested.stations)}
        evaluation = evaluate_route(self._train(), graph, {}, set(), [active_train], requested_route=requested)
        routes = [candidate.route.stations for candidate in evaluation.candidates]
        self.assertEqual(routes, [
            ("Delhi", "Jaipur", "Ahmedabad", "Mumbai"),
            ("Delhi", "Jaipur", "Agra", "Bhopal", "Mumbai"),
        ])
        self.assertEqual(len(routes), len(set(routes)))


if __name__ == "__main__":
    unittest.main()
