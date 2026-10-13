"""Focused tests for dashboard adapters around existing railway logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app import count_active_conflicts, count_conflict_events, edge_label, make_log_entry, parse_edge, safe_route_options
from railway_network import build_demo_network


class DashboardAdapterTests(unittest.TestCase):
    def test_edge_label_round_trip(self) -> None:
        edge = ("Delhi", "Agra")
        self.assertEqual(parse_edge(edge_label(edge)), edge)

    def test_dashboard_route_options_use_existing_graph(self) -> None:
        routes = safe_route_options(build_demo_network(), "Delhi", "Mumbai")
        self.assertGreaterEqual(len(routes), 2)
        self.assertEqual(routes[0].stations, ("Delhi", "Agra", "Bhopal", "Mumbai"))

    def test_structured_log_and_conflict_counts(self) -> None:
        entry = make_log_entry(
            "SIM-001",
            1,
            ["Delhi", "Agra"],
            {"allowed": False, "reasons": ["The proposed route conflicts with another route"]},
            "REROUTE",
            "Validated alternative selected",
            ["Delhi", "Jaipur"],
            ["The proposed route conflicts with another route"],
        )
        self.assertEqual(entry["requested_route"], "Delhi -> Agra")
        self.assertEqual(entry["final_route"], "Delhi -> Jaipur")
        self.assertEqual(count_conflict_events([entry]), 1)
        self.assertEqual(count_active_conflicts([{"safety_reasons": entry["safety_reasons"]}]), 1)


if __name__ == "__main__":
    unittest.main()
