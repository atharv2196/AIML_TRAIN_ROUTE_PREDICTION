"""Interactive Streamlit dashboard for the educational railway prototype."""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from decision_engine import OperationalAction, decide_operation
from integrated_simulation import (
    LEAKAGE_COLUMNS,
    STATION_COORDINATES,
    DelayPredictor,
    run_scenario,
    update_after_valid_action,
)
from fleet_simulation import CONFLICT_POLICY, DEFAULT_FLEET_SIZE, apply_route_evaluation, evaluate_route, fleet_summary, generate_fleet
from railway_network import Route, alternative_routes, block_track, build_demo_network, route_edges, shortest_route
from safety_engine import SignalState, check_route_safety

DATA_PATH = ROOT / "datasets" / "ir_train.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}
BASE_EDGES = [
    ("Delhi", "Agra"),
    ("Agra", "Bhopal"),
    ("Bhopal", "Mumbai"),
    ("Delhi", "Jaipur"),
    ("Jaipur", "Ahmedabad"),
    ("Ahmedabad", "Mumbai"),
    ("Agra", "Jaipur"),
]

st.set_page_config(page_title="Railway Operations Control", page_icon="🚆", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root {
        --rail-bg: #0B1120;
        --rail-sidebar: #111827;
        --rail-panel: #172235;
        --rail-panel-2: #1E293B;
        --rail-raised: #243247;
        --rail-border: #334155;
        --rail-text: #F8FAFC;
        --rail-secondary: #CBD5E1;
        --rail-muted: #94A3B8;
        --rail-blue: #38BDF8;
        --rail-blue-2: #60A5FA;
        --rail-green: #22C55E;
        --rail-amber: #F59E0B;
        --rail-red: #EF4444;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: var(--rail-bg) !important;
        color: var(--rail-text) !important;
    }
    .stApp, .stApp * { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background: var(--rail-bg) !important; color: var(--rail-text) !important; }
    [data-testid="stHeader"] { background: var(--rail-bg) !important; border-bottom: 1px solid var(--rail-border); }
    [data-testid="stToolbar"] { background: var(--rail-bg) !important; }
    [data-testid="stSidebar"] { background: var(--rail-sidebar) !important; border-right: 1px solid var(--rail-border); }
    [data-testid="stSidebar"] > div:first-child { background: var(--rail-sidebar) !important; }
    [data-testid="stSidebar"] * { color: var(--rail-text); }
    [data-testid="stAppViewContainer"] section.main > div { max-width: 1500px; padding-top: 2rem; }
    h1 { color: var(--rail-text) !important; font-size: 2rem !important; font-weight: 750 !important; letter-spacing: 0 !important; }
    h2, h3 { color: var(--rail-text) !important; font-weight: 700 !important; letter-spacing: 0 !important; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--rail-secondary); }
    [data-testid="stMetric"] { background: var(--rail-panel) !important; border: 1px solid var(--rail-border) !important; border-radius: 10px !important; padding: 16px 18px !important; min-height: 108px; box-shadow: none !important; }
    [data-testid="stMetricLabel"] { color: var(--rail-muted) !important; font-size: 0.78rem !important; font-weight: 650 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: var(--rail-text) !important; font-size: 1.8rem !important; font-weight: 750 !important; }
    [data-testid="stMetricDelta"] { color: var(--rail-blue) !important; }
    [data-baseweb="tab-list"] { background: var(--rail-panel) !important; border: 1px solid var(--rail-border); border-radius: 8px; padding: 4px; gap: 4px; }
    button[data-baseweb="tab"] { background: var(--rail-panel) !important; color: var(--rail-muted) !important; border-radius: 6px !important; font-weight: 650 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background: var(--rail-raised) !important; color: var(--rail-text) !important; }
    [data-baseweb="tab-highlight"] { background: var(--rail-blue) !important; height: 2px !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: var(--rail-panel) !important; border: 1px solid var(--rail-border) !important; border-radius: 10px !important; }
    [data-testid="stExpander"] { background: var(--rail-panel) !important; border: 1px solid var(--rail-border) !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"], [data-testid="stMultiSelect"] div[data-baseweb="select"], [data-testid="stSlider"] {
        background: var(--rail-panel-2) !important; color: var(--rail-text) !important;
    }
    [data-baseweb="select"] > div { background: var(--rail-panel-2) !important; border-color: var(--rail-border) !important; color: var(--rail-text) !important; }
    [data-baseweb="popover"] > div, [data-baseweb="menu"] { background: var(--rail-panel-2) !important; color: var(--rail-text) !important; border: 1px solid var(--rail-border) !important; }
    [data-baseweb="option"] { color: var(--rail-text) !important; background: var(--rail-panel-2) !important; }
    [data-baseweb="option"][aria-selected="true"], [data-baseweb="option"]:hover { background: var(--rail-raised) !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { border: 1px solid var(--rail-border) !important; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button { background: var(--rail-blue-2) !important; color: #07111F !important; border: 1px solid var(--rail-blue) !important; border-radius: 7px !important; font-weight: 750 !important; min-height: 40px; }
    [data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover { background: var(--rail-blue) !important; border-color: var(--rail-blue) !important; }
    [data-testid="stButton"] button:focus, [data-testid="stFormSubmitButton"] button:focus { outline: 2px solid var(--rail-blue) !important; outline-offset: 2px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--rail-border) !important; background: var(--rail-panel) !important; }
    [data-testid="stAlert"] { background: var(--rail-panel-2) !important; border: 1px solid var(--rail-border) !important; color: var(--rail-text) !important; }
    [data-testid="stStatusWidget"] { background: var(--rail-panel) !important; }
    .status-pill { display: inline-block; padding: 5px 11px; border-radius: 999px; font-weight: 800; font-size: 0.72rem; letter-spacing: 0.06em; border: 1px solid currentColor; }
    .safe { color: #07130D; background: var(--rail-green); }
    .warning { color: #241600; background: var(--rail-amber); }
    .danger { color: #FFF5F5; background: var(--rail-red); }
    .sim-note { color: var(--rail-secondary); font-size: 0.88rem; line-height: 1.45; padding: 12px 14px; border-left: 3px solid var(--rail-blue); background: var(--rail-panel) !important; border-top: 1px solid var(--rail-border); border-right: 1px solid var(--rail-border); border-bottom: 1px solid var(--rail-border); }
    hr { border-color: var(--rail-border) !important; }
    [data-testid="stMarkdownContainer"] a { color: var(--rail-blue) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def model_key(name: str) -> str:
    return name.lower().replace(" ", "_")


@st.cache_resource(show_spinner="Loading the saved model...")
def load_predictor(model_name: str) -> DelayPredictor:
    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return DelayPredictor(model_path, DATA_PATH)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fresh_network() -> nx.Graph:
    return build_demo_network()


def edge_label(edge: tuple[str, str]) -> str:
    return f"{edge[0]} - {edge[1]}"


def canonical_edge(edge: tuple[str, str]) -> tuple[str, str]:
    if edge in BASE_EDGES:
        return edge
    reversed_edge = (edge[1], edge[0])
    if reversed_edge in BASE_EDGES:
        return reversed_edge
    return edge


def parse_edge(label: str) -> tuple[str, str]:
    left, right = label.split(" - ", 1)
    return left, right


def initialize_state() -> None:
    if st.session_state.get("fleet_version") != 1:
        for key in ["network", "signals", "occupied_tracks", "decision_log", "route_plan", "last_scenario", "last_scenario_result", "trains", "selected_train"]:
            st.session_state.pop(key, None)
        st.session_state.fleet_version = 1
    if "network" not in st.session_state:
        st.session_state.network = fresh_network()
    if "decision_log" not in st.session_state:
        st.session_state.decision_log = []
    if "route_plan" not in st.session_state:
        st.session_state.route_plan = None
    if "last_scenario" not in st.session_state:
        st.session_state.last_scenario = None
    if "selected_train" not in st.session_state:
        st.session_state.selected_train = None
    if "trains" not in st.session_state:
        predictor = load_predictor("Random Forest")
        st.session_state.trains = generate_fleet(predictor, count=DEFAULT_FLEET_SIZE)
        st.session_state.signals = {station: SignalState.GREEN.value for station in STATION_COORDINATES}
        st.session_state.occupied_tracks = set()
        for train in st.session_state.trains:
            if train.get("restricted_signal"):
                st.session_state.signals[train["restricted_signal"]] = SignalState.RED.value
            if train.get("blocked_track"):
                st.session_state.occupied_tracks.add(canonical_edge(tuple(train["blocked_track"])))
        st.session_state.selected_train = st.session_state.trains[0]["train_id"]


def reset_simulation() -> None:
    for key in ["network", "signals", "occupied_tracks", "decision_log", "route_plan", "last_scenario", "last_scenario_result", "trains", "selected_train"]:
        st.session_state.pop(key, None)
    st.rerun()


def make_log_entry(train_id: str, predicted_delay: int, requested_route: list[str], safety: dict[str, Any], decision: str, reason: str, final_route: list[str], safety_reasons: list[str] | None = None, step: int = 1) -> dict[str, Any]:
    reasons = list(safety_reasons or safety.get("reasons", []))
    return {
        "step": step,
        "action_id": f"{train_id}-{step:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_id": train_id,
        "predicted_delay": "DELAYED" if predicted_delay else "ON TIME",
        "requested_route": " -> ".join(requested_route),
        "safety_allowed": bool(safety["allowed"]),
        "safety_result": "ALLOWED" if safety["allowed"] else "REJECTED",
        "safety_reasons": reasons,
        "decision": decision,
        "reason": reason,
        "final_route": " -> ".join(final_route) if final_route else "NO VALIDATED ROUTE",
    }


def count_conflict_events(logs: list[dict[str, Any]]) -> int:
    return sum(any("conflict" in reason.lower() for reason in log.get("safety_reasons", [])) for log in logs)


def count_active_conflicts(trains: list[dict[str, Any]]) -> int:
    return sum(any("conflict" in reason.lower() for reason in train.get("safety_reasons", [])) for train in trains)


def append_log(train_id: str, predicted_delay: int, requested_route: list[str], safety: dict[str, Any], decision: str, reason: str, final_route: list[str], safety_reasons: list[str] | None = None) -> None:
    reasons = list(safety_reasons or safety.get("reasons", []))
    fingerprint = (train_id, tuple(requested_route), tuple(final_route), decision, tuple(reasons))
    if st.session_state.decision_log and st.session_state.decision_log[-1].get("fingerprint") == fingerprint:
        return
    entry = make_log_entry(train_id, predicted_delay, requested_route, safety, decision, reason, final_route, reasons, len(st.session_state.decision_log) + 1)
    entry["fingerprint"] = fingerprint
    st.session_state.decision_log.append(entry)


def selected_train() -> dict[str, Any]:
    for train in st.session_state.trains:
        if train["train_id"] == st.session_state.selected_train:
            return train
    return st.session_state.trains[0]


def safe_route_options(graph: nx.Graph, source: str, destination: str) -> list[Route]:
    try:
        return alternative_routes(graph, source, destination, limit=3)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def evaluate_selected_route(route: Route | None = None):
    train = selected_train()
    return evaluate_route(
        train,
        st.session_state.network,
        {station: SignalState(value) for station, value in st.session_state.signals.items()},
        st.session_state.occupied_tracks,
        st.session_state.trains,
        requested_route=route,
    )


def plot_network(selected_route: list[str] | None = None, selected_train_id: str | None = None) -> go.Figure:
    graph = st.session_state.network
    selected_edges = {frozenset(edge) for edge in route_edges(selected_route or [])}
    occupied_edges = {frozenset(edge) for edge in st.session_state.occupied_tracks}
    fig = go.Figure()
    for left, right in BASE_EDGES:
        x_values = [STATION_COORDINATES[left][1], STATION_COORDINATES[right][1]]
        y_values = [STATION_COORDINATES[left][0], STATION_COORDINATES[right][0]]
        edge_key = frozenset((left, right))
        if not graph.has_edge(left, right):
            color, width, dash = "#EF4444", 5, "dash"
        elif edge_key in occupied_edges:
            color, width, dash = "#F59E0B", 5, "dot"
        elif edge_key in selected_edges:
            color, width, dash = "#38BDF8", 6, "solid"
        else:
            color, width, dash = "#64748B", 2, "solid"
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", line={"color": color, "width": width, "dash": dash}, hoverinfo="text", text=[edge_label((left, right))] * 2, showlegend=False))
    for name, color, dash in [("Normal track", "#64748B", "solid"), ("Selected route", "#38BDF8", "solid"), ("Occupied track", "#F59E0B", "dot"), ("Blocked track", "#EF4444", "dash")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": color, "width": 4, "dash": dash}, name=name))
    station_names = list(STATION_COORDINATES)
    fig.add_trace(go.Scatter(x=[STATION_COORDINATES[name][1] for name in station_names], y=[STATION_COORDINATES[name][0] for name in station_names], mode="markers+text", text=station_names, textposition="top center", marker={"size": 15, "color": "#f4c95d", "line": {"color": "#fff3c2", "width": 1}}, name="Stations", hovertemplate="%{text}<extra></extra>"))
    for train in st.session_state.trains:
        is_selected = train["train_id"] == selected_train_id
        fig.add_trace(go.Scatter(x=[train["longitude"]], y=[train["latitude"]], mode="markers", marker={"size": 20 if is_selected else 14, "color": "#f28f3b" if train["predicted_delay_status"] else "#55d187", "line": {"color": "white", "width": 2 if is_selected else 1}}, name=f"Train {train['train_id']}", text=[train["train_id"]], hovertemplate="Train %{text}<br>Status: " + train["operational_status"] + "<extra></extra>"))
    fig.update_layout(height=560, template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", margin={"l": 10, "r": 10, "t": 20, "b": 10}, xaxis={"title": "Simulated longitude", "showgrid": False}, yaxis={"title": "Simulated latitude", "showgrid": False}, legend={"orientation": "h", "y": -0.12})
    return fig


def status_badge(label: str, kind: str) -> None:
    st.markdown(f'<span class="status-pill {kind}">{label}</span>', unsafe_allow_html=True)


def overview_tab() -> None:
    trains = st.session_state.trains
    summary = fleet_summary(trains)
    running = sum(train["operational_status"] in {"PROCEED", "REROUTE"} for train in trains)
    columns = st.columns(6)
    for column, label, value in zip(columns, ["Simulated trains", "Running", "Delayed", "On hold", "Safety alerts", "Avg delay (sim.)"], [summary["total"], running, summary["delayed"], summary["on_hold"], summary["safety_alerts"], f"{summary['average_delay_minutes']:.0f} min"]):
        column.metric(label, value)
    st.subheader("Control-room status")
    left, right = st.columns([1, 2])
    with left:
        status_badge("SIMULATION ACTIVE", "safe")
        st.write(f"Selected scenario: **{st.session_state.last_scenario or 'Baseline fleet'}**")
        historical_conflicts = count_conflict_events(st.session_state.decision_log)
        active_conflicts = count_active_conflicts(trains)
        st.write(f"Active route-conflict alerts: **{active_conflicts}**")
        st.write(f"Historical conflict events: **{historical_conflicts}**")
        st.write(f"Last action: **{st.session_state.decision_log[-1]['decision'] if st.session_state.decision_log else 'No manual action yet'}**")
    with right:
        st.markdown("**Workflow**")
        st.write("Simulated train state -> saved delay classifier -> NetworkX route -> deterministic safety verification -> explainable operational decision.")
        st.markdown('<div class="sim-note">All positions, signals, occupancy and actions in this dashboard are simulated. It is not connected to live GNSS, Indian Railways control systems, or certified signaling hardware.</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        status_frame = pd.DataFrame({"Decision": [train["current_decision"] for train in trains]}).value_counts().reset_index(name="Trains")
        st.plotly_chart(px.bar(status_frame, x="Decision", y="Trains", color="Decision", title="Current fleet decisions", template="plotly_dark"), use_container_width=True, key="fleet_decision_chart")
    with chart_right:
        delay_frame = pd.DataFrame({"Prediction": ["DELAYED" if train["predicted_delay_status"] else "ON TIME" for train in trains]}).value_counts().reset_index(name="Trains")
        st.plotly_chart(px.bar(delay_frame, x="Prediction", y="Trains", color="Prediction", title="Model-predicted delay status", template="plotly_dark"), use_container_width=True, key="fleet_delay_chart")


def map_tab() -> None:
    train_ids = [train["train_id"] for train in st.session_state.trains]
    chosen = st.selectbox("Train to highlight", train_ids, index=train_ids.index(st.session_state.selected_train), key="map_train")
    st.session_state.selected_train = chosen
    train = selected_train()
    st.plotly_chart(plot_network(train["final_route"], chosen), use_container_width=True, key="map_network_chart")
    st.caption("Schematic station coordinates are used for visualization; they do not represent a live railway map.")
    st.dataframe(pd.DataFrame([{key: train[key] for key in ["train_id", "current_station", "destination", "operational_status", "final_route"]}]), hide_index=True, use_container_width=True)


def monitoring_tab() -> None:
    search = st.text_input("Search train, origin, destination, or station", key="fleet_search")
    available_statuses = sorted({train["operational_status"] for train in st.session_state.trains})
    selected_statuses = st.multiselect("Operational status", available_statuses, default=available_statuses, key="fleet_status_filter")
    selected_delays = st.multiselect("Prediction filter", ["DELAYED", "ON TIME"], default=["DELAYED", "ON TIME"], key="fleet_delay_filter")
    query = search.strip().lower()
    filtered_trains = [train for train in st.session_state.trains if train["operational_status"] in selected_statuses and ("DELAYED" if train["predicted_delay_status"] else "ON TIME") in selected_delays and (not query or query in " ".join([train["train_id"], train["origin"], train["destination"], train["current_station"]]).lower())]
    rows = []
    for train in filtered_trains:
        rows.append({"Train ID": train["train_id"], "Station": train["current_station"], "Destination": train["destination"], "Speed km/h": train["speed_kmph"], "Delay": "DELAYED" if train["predicted_delay_status"] else "ON TIME", "Status": train["operational_status"], "Route": " -> ".join(train["final_route"]), "Safety": "CLEAR" if train["safety_allowed"] else "ALERT"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    train_ids = [train["train_id"] for train in st.session_state.trains]
    chosen = st.selectbox("Inspect train", train_ids, index=train_ids.index(st.session_state.selected_train), key="monitor_train")
    st.session_state.selected_train = chosen
    train = selected_train()
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.write(f"**{train['train_id']}** | {train['origin']} -> {train['destination']}")
        st.write(f"Current location: **{train['current_station']}** | Direction: **{train['direction']}**")
        st.write(f"Decision: **{train['current_decision']}** | Reason: {train['decision_reason']}")
    with detail_right:
        st.write(f"Planned route: **{' -> '.join(train['current_route'])}**")
        st.write(f"Safety: **{'CLEAR' if train['safety_allowed'] else 'ALERT'}**")
        if train["safety_reasons"]:
            st.error("; ".join(train["safety_reasons"]))
    speed = st.slider("Simulated speed (km/h)", 0, 160, int(train["speed_kmph"]), key=f"speed_{chosen}")
    train["speed_kmph"] = speed
    st.write(f"Current position: **{train['current_station']}** | Destination: **{train['destination']}**")
    if st.button("Advance one valid route segment", key="advance_train"):
        route = tuple(train["final_route"])
        if train["current_station"] not in route or route.index(train["current_station"]) >= len(route) - 1:
            st.info("This simulated train has no further route segment to advance.")
        else:
            remaining = route[route.index(train["current_station"]):]
            state = type("DashboardState", (), train)()
            state.current_route = remaining
            update_after_valid_action(state, Route(remaining, 0.0), OperationalAction.PROCEED)
            train.update({"current_station": state.current_station, "latitude": state.latitude, "longitude": state.longitude, "operational_status": state.operational_status, "final_route": list(state.current_route)})
            st.session_state.decision_log.append({"step": len(st.session_state.decision_log) + 1, "train_id": train["train_id"], "predicted_delay": "DELAYED" if train["predicted_delay_status"] else "ON TIME", "requested_route": " -> ".join(route), "safety_result": "NOT RECHECKED", "decision": "PROCEED", "reason": "Advanced one simulated route segment through the tracking function", "final_route": " -> ".join(train["final_route"])})
            st.success(f"Advanced to {train['current_station']}.")


def route_tab() -> None:
    stations = list(STATION_COORDINATES)
    train_ids = [train["train_id"] for train in st.session_state.trains]
    route_train_id = st.selectbox("Train to route", train_ids, index=train_ids.index(st.session_state.selected_train), key="route_train")
    route_train = next(train for train in st.session_state.trains if train["train_id"] == route_train_id)
    st.session_state.selected_train = route_train_id
    source = st.selectbox("Source station", stations, index=stations.index(route_train["current_station"]) if route_train["current_station"] in stations else 0, key="route_source")
    destination = st.selectbox("Destination station", stations, index=stations.index(route_train["destination"]) if route_train["destination"] in stations else 3, key="route_destination")
    block_labels = [edge_label(edge) for edge in BASE_EDGES]
    selected_blocks = st.multiselect("Tracks to block for this simulation", block_labels, key="route_blocks")
    left, right = st.columns(2)
    with left:
        if st.button("Apply blocks and calculate route", type="primary"):
            if source == destination:
                st.error("Source and destination must be different stations.")
            else:
                st.session_state.network = fresh_network()
                st.session_state.occupied_tracks = set()
                for label in selected_blocks:
                    edge = parse_edge(label)
                    block_track(st.session_state.network, *edge)
                    st.session_state.occupied_tracks.add(edge)
                try:
                    route_train.update({"origin": source, "destination": destination, "current_station": source, "latitude": STATION_COORDINATES[source][0], "longitude": STATION_COORDINATES[source][1], "direction": f"{source} -> {destination}"})
                    evaluation = evaluate_selected_route()
                    chosen = evaluation.requested_route
                    if chosen is None:
                        raise nx.NetworkXNoPath
                    options = safe_route_options(st.session_state.network, source, destination)
                    apply_route_evaluation(route_train, evaluation)
                    validated = evaluation.final_route or chosen
                    st.session_state.route_plan = {"source": source, "destination": destination, "requested": chosen, "selected": validated, "evaluation": evaluation, "alternatives": options}
                    st.success(f"Requested route: {' -> '.join(chosen.stations)} ({chosen.cost_km:g} km)")
                except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
                    st.session_state.route_plan = None
                    st.error(f"No feasible graph route: {error}")
    with right:
        if st.button("Reset route planner"):
            st.session_state.network = fresh_network()
            st.session_state.occupied_tracks = set()
            st.session_state.route_plan = None
            st.rerun()
    plan = st.session_state.route_plan
    if plan:
        evaluation = plan["evaluation"]
        st.write(f"**Requested route:** {' -> '.join(plan['requested'].stations)} | **Cost:** {plan['requested'].cost_km:g} km")
        if evaluation.final_route and evaluation.final_route.stations != evaluation.requested_route.stations:
            st.success(f"**Validated final route:** {' -> '.join(evaluation.final_route.stations)} | **Decision:** {evaluation.decision.value}")
        elif evaluation.final_route is None:
            st.error(f"No validated final route. **Decision:** {evaluation.decision.value}")
        else:
            st.write(f"**Validated final route:** {' -> '.join(evaluation.final_route.stations)} | **Decision:** {evaluation.decision.value}")
        if evaluation.candidates:
            st.write("**Evaluated candidate routes:**")
            candidate_rows = []
            for candidate in evaluation.candidates:
                conflict_text = "\n".join(f"{item['train_id']}: {', '.join('-'.join(edge) for edge in item['shared_edges'])}" for item in candidate.conflict_details)
                reasons = list(candidate.safety_result.reasons)
                if candidate.blocked_edges:
                    reasons.append("Blocked edges: " + ", ".join("-".join(edge) for edge in candidate.blocked_edges))
                if candidate.occupied_edges:
                    reasons.append("Occupied edges: " + ", ".join("-".join(edge) for edge in candidate.occupied_edges))
                candidate_rows.append({"Route": " -> ".join(candidate.route.stations), "Cost km": candidate.route.cost_km, "Status": "VALIDATED SAFE" if candidate.validated else "REJECTED", "Failure categories": ", ".join(candidate.failure_categories) or "None", "Failure reasons": "; ".join(reasons) or "None", "Active conflicts": conflict_text or "None"})
            st.dataframe(pd.DataFrame(candidate_rows), hide_index=True, use_container_width=True)
        st.warning(CONFLICT_POLICY)
        status_badge("REQUESTED ROUTE SAFE" if evaluation.safety_result.allowed else "REQUESTED ROUTE REJECTED", "safe" if evaluation.safety_result.allowed else "danger")
        if evaluation.reasons:
            st.write("Evaluation reasons: " + "; ".join(evaluation.reasons))
    st.plotly_chart(plot_network(plan["selected"].stations if plan else selected_train()["final_route"], st.session_state.selected_train), use_container_width=True, key="route_network_chart")


def safety_tab() -> None:
    plan = st.session_state.route_plan
    route = plan["requested"] if plan else Route(tuple(selected_train()["final_route"]), 0.0)
    st.write(f"Route under review: **{' -> '.join(route.stations)}**")
    signal_station = st.selectbox("Signal station", list(STATION_COORDINATES), key="signal_station")
    signal_value = st.selectbox("Signal aspect", [state.value for state in SignalState], index=[state.value for state in SignalState].index(st.session_state.signals[signal_station]), key="signal_value")
    available_edge_labels = [edge_label(edge) for edge in BASE_EDGES]
    default_occupied_labels = [edge_label(canonical_edge(edge)) for edge in st.session_state.occupied_tracks if edge_label(canonical_edge(edge)) in available_edge_labels]
    occupied_labels = st.multiselect("Occupied tracks", available_edge_labels, default=default_occupied_labels, key="safety_occupied")
    if st.button("Apply safety state and verify route", type="primary"):
        st.session_state.signals[signal_station] = signal_value
        st.session_state.occupied_tracks = {parse_edge(label) for label in occupied_labels}
        evaluation = evaluate_selected_route(route)
        apply_route_evaluation(selected_train(), evaluation)
        if plan:
            plan["evaluation"] = evaluation
            plan["selected"] = evaluation.final_route or route
        if evaluation.safety_result.allowed:
            status_badge("REQUESTED ROUTE ALLOWED", "safe")
        else:
            status_badge("REQUESTED ROUTE REJECTED", "danger")
        if evaluation.safety_result.reasons:
            st.error("; ".join(evaluation.safety_result.reasons))
        st.write(f"Decision engine: **{evaluation.decision.value}**")
        st.write(f"Reason: {evaluation.reasons[-1] if evaluation.reasons else 'No decision reason recorded'}")
        if evaluation.final_route:
            st.success(f"Validated final route: {' -> '.join(evaluation.final_route.stations)}")
        else:
            st.warning("No validated final route was selected.")
        append_log(selected_train()["train_id"], selected_train()["predicted_delay_status"], list(route.stations), {"allowed": evaluation.safety_result.allowed, "reasons": list(evaluation.safety_result.reasons)}, evaluation.decision.value, evaluation.reasons[-1] if evaluation.reasons else "No decision reason recorded", list(evaluation.final_route.stations) if evaluation.final_route else [], list(evaluation.safety_result.reasons))
    st.dataframe(pd.DataFrame([{"Station": station, "Signal": st.session_state.signals[station]} for station in STATION_COORDINATES]), hide_index=True, use_container_width=True)
    st.caption("Safety checks remain deterministic and are never overridden by the ML prediction.")


def prediction_tab() -> None:
    train_ids = [train["train_id"] for train in st.session_state.trains]
    chosen = st.selectbox("Base prediction record", train_ids, key="prediction_train")
    train = next(item for item in st.session_state.trains if item["train_id"] == chosen)
    model_name = st.selectbox("Saved model", list(MODEL_FILES), key="prediction_model")
    features = dict(train["model_features"])
    with st.form("prediction_form"):
        departure_hour = st.slider("Departure hour", 0, 23, int(features.get("departure_hour", 12)))
        distance_km = st.number_input("Distance (km)", min_value=1.0, max_value=5000.0, value=float(features.get("distance_km", 500.0)))
        fog_score = st.number_input("Fog risk score", min_value=0.0, max_value=1.0, value=float(features.get("fog_risk_score", 0.0)))
        utilisation = st.number_input("Seat utilisation (%)", min_value=0.0, max_value=100.0, value=float(features.get("seat_utilisation_pct", 70.0)))
        incoming_delay = st.checkbox("Late incoming rake", value=bool(features.get("late_incoming_rake", 0)))
        submitted = st.form_submit_button("Predict delay", type="primary")
    if submitted:
        features.update({"departure_hour": departure_hour, "distance_km": distance_km, "fog_risk_score": fog_score, "seat_utilisation_pct": utilisation, "late_incoming_rake": int(incoming_delay)})
        try:
            predictor = load_predictor(model_name)
            frame = pd.DataFrame([features])
            prediction = int(predictor.pipeline.predict(frame)[0])
            probability = float(predictor.pipeline.predict_proba(frame)[0, 1])
            if prediction:
                status_badge("PREDICTED DELAY", "warning")
            else:
                status_badge("PREDICTED ON TIME", "safe")
            st.metric("Delay probability", f"{probability:.1%}")
            st.info("This is a model prediction, not a guarantee. It cannot authorize unsafe movement.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


def performance_tab() -> None:
    original = load_csv("model_comparison.csv")
    chronological = load_csv("chronological_model_comparison.csv")
    if original.empty:
        st.warning("Original evaluation results are unavailable.")
        return
    st.subheader("Original random-split experiment")
    st.dataframe(original, hide_index=True, use_container_width=True)
    metric = st.selectbox("Metric", ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"], key="metric_select")
    fig = px.bar(original, x="model", y=metric, color="model", title=f"Original experiment: {metric}", range_y=[0, 1], template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, key="original_metric_chart")
    if not chronological.empty:
        st.subheader("Original versus chronological experiment")
        comparison = pd.concat([original.assign(experiment="Original random split"), chronological.assign(experiment="Improved chronological")], ignore_index=True)
        comparison = comparison[comparison["model"] != "Majority Class Baseline"]
        fig = px.bar(comparison, x="model", y=metric, color="experiment", barmode="group", range_y=[0, 1], title=f"{metric}: random versus chronological", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True, key="experiment_comparison_chart")
        st.dataframe(chronological, hide_index=True, use_container_width=True)
        st.caption("The experiments use different split strategies and feature exclusions; compare them as evaluation views, not as identical benchmarks.")
    image_columns = st.columns(3)
    for column, name in zip(image_columns, ["confusion_random_forest.png", "feature_importance_random_forest.png", "roc_curves.png"]):
        path = GRAPHS_DIR / name
        if path.exists():
            column.image(str(path), caption=name, use_container_width=True)


def scenario_tab() -> None:
    options = {"Normal Operation": ("normal_operation", None, None), "Congestion": ("congestion_alternative_route", ("Delhi", "Agra"), None), "Red Signal Safety Event": ("red_signal_safety_event", None, "Agra")}
    label = st.selectbox("Scenario", list(options), key="scenario_select")
    if st.button("Run selected scenario", type="primary"):
        name, blocked, red = options[label]
        result = run_scenario(name, load_predictor("Random Forest"), blocked_edge=blocked, red_station=red)
        st.session_state.last_scenario = label
        st.session_state.last_scenario_result = result
        append_log(result.final_state["train_id"], result.predicted_delay_status, result.initial_route["stations"], result.initial_safety, result.decision, result.reason, result.final_route["stations"])
    if st.button("Reset simulation", key="scenario_reset"):
        reset_simulation()
    result = st.session_state.get("last_scenario_result")
    if result:
        result_data = asdict(result)
        st.write("**Event log**")
        for title, value in [("Initial train state", result_data["initial_state"]), ("Delay prediction", result_data["predicted_delay_status"]), ("Initial route", result_data["initial_route"]), ("Safety check", result_data["initial_safety"]), ("Decision", result_data["decision"]), ("Reason", result_data["reason"]), ("Final state", result_data["final_state"])]:
            st.write(f"**{title}:** {value}")


def decision_log_tab() -> None:
    if not st.session_state.decision_log:
        st.info("No dashboard actions have been logged yet. Run a scenario or apply a safety check.")
        return
    log = pd.DataFrame(st.session_state.decision_log)
    train_filter = st.multiselect("Filter trains", sorted(log["train_id"].unique()), default=list(log["train_id"].unique()))
    decision_filter = st.multiselect("Filter decisions", sorted(log["decision"].unique()), default=sorted(log["decision"].unique()))
    filtered = log[log["train_id"].isin(train_filter) & log["decision"].isin(decision_filter)]
    display_log = filtered.drop(columns=["fingerprint"], errors="ignore").copy()
    if "safety_reasons" in display_log:
        display_log["safety_reasons"] = display_log["safety_reasons"].apply(lambda reasons: "; ".join(reasons) if isinstance(reasons, list) else str(reasons))
    st.dataframe(display_log, hide_index=True, use_container_width=True)


def main() -> None:
    try:
        initialize_state()
    except Exception as error:
        st.error(f"Dashboard initialization failed: {error}")
        st.info("Check that the virtual environment is active and the saved model and dataset files exist.")
        st.stop()
    with st.sidebar:
        st.title("RAILWAY / OPS")
        st.caption("AI-assisted educational control room")
        status_badge("SIMULATION MODE", "warning")
        st.divider()
        st.write("**Selected train**")
        st.session_state.selected_train = st.selectbox("Train", [train["train_id"] for train in st.session_state.trains], key="sidebar_train")
        if st.button("Reset all simulation state"):
            reset_simulation()
        st.divider()
        st.caption("No live GNSS. No real railway control. Safety rules remain deterministic.")
    st.title("Railway Operations Control Room")
    st.caption("AI-Based Railway Train Tracking, Signal Handling, Route Planning and Automatic Decision-Making System")
    tabs = st.tabs(["Overview", "Railway Map", "Train Monitoring", "Route Planner", "Safety Control", "AI Prediction", "Model Performance", "Scenarios", "Decision Log"])
    with tabs[0]: overview_tab()
    with tabs[1]: map_tab()
    with tabs[2]: monitoring_tab()
    with tabs[3]: route_tab()
    with tabs[4]: safety_tab()
    with tabs[5]: prediction_tab()
    with tabs[6]: performance_tab()
    with tabs[7]: scenario_tab()
    with tabs[8]: decision_log_tab()


if __name__ == "__main__":
    main()
