# AI-Based Railway Operations Prototype

This is an educational simulation, not a certified railway control system. It combines delay classification, graph-based route planning, simulated train localization, deterministic safety rules, and explainable operational decisions.

## Environment

The project uses Python 3.14.5 in `.venv`. Install the pinned core packages with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Inspect data

```powershell
.\.venv\Scripts\python.exe src\inspect_data.py
```

The local `ir_train.csv` contains 1,500,000 rows and the binary target `is_delayed`. The training script excludes target-revealing delay fields, identifiers, and departure date. It uses a reproducible stratified split.

## Train models

The default development run reads 300,000 rows so it is practical on a personal computer. Use `--max-rows 0` for the full CSV.

```powershell
.\.venv\Scripts\python.exe src\train_models.py
```

The command trains Logistic Regression, Random Forest, and XGBoost, then writes model files to `models/`, metrics and reports to `results/`, and graphs to `graphs/`. The current initial experiment uses 300,000 rows, with 240,000 training examples and 60,000 testing examples.

## Railway simulation modules

Set the source directory on the import path when using the modules directly:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\railway_network.py
```

- `railway_network.py`: demo graph, shortest paths, alternatives, and blocked tracks.
- `train_tracking.py`: simulated train records and nearest-station mapping.
- `safety_engine.py`: signal, occupancy, conflict, and speed-limit checks.
- `decision_engine.py`: PROCEED, HOLD, REROUTE, or STOP rules.
- `fleet_simulation.py`: deterministic 24-train fleet generation, model predictions, route states, and safety-gated decisions.

The ML model does not override deterministic safety checks. Live GNSS, national network coverage, certified interlocking, and reinforcement-learning methods are outside this prototype.

## Audit and chronological evaluation

Run the audit without changing the original `model_comparison.csv` or `evaluation_report.txt`:

```powershell
$env:PYTHONPATH = "src"
python src\audit_and_chronological.py --max-rows 20000
```

This verifies the saved-model metrics, adds a majority-class baseline, reviews operational-risk features, and trains a separate chronological experiment. New outputs are written to `results/audit_report.txt`, `results/audit_saved_results.csv`, `results/chronological_model_comparison.csv`, and `results/original_vs_improved_comparison.csv`.

## Integrated demonstration

Run the saved Random Forest model through normal, congestion, and red-signal scenarios:

```powershell
$env:PYTHONPATH = "src"
python src\integrated_simulation.py
```

Scenario output is saved to `results/integrated_scenarios.json` and `results/integrated_scenarios.txt`.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Streamlit dashboard

Install the dashboard dependencies from the project environment and launch the local control-room interface:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The dashboard has nine tabs: System Overview, Interactive Railway Map, Train Monitoring, Route Planner, Signal and Safety Control, AI Delay Prediction, Model Performance, Scenario Demonstration, and Decision Log. It loads saved models and reports rather than retraining when the page opens. Session state retains the stable 24-train fleet, route blocks, signal states, selected trains, scenario results, and logged actions.

The fleet is synthetic and deterministic. Each train has a stable `SIM-###` ID, origin, destination, current station, speed, direction, model-predicted delay, route, safety state, and operational decision. The default fleet contains 24 trains, with connected routes over the small demonstration graph. The normal, congestion, and red-signal scenario demonstrations remain available separately.

Route conflicts use a simplified same-step educational policy: only trains currently in `PROCEED` or `REROUTE` reserve their stored route edges. `HOLD` and `STOP` trains do not reserve future route edges. A route is considered conflicting when it shares an undirected edge with an active train in the same simulated step. This is not a timetable, signaling, or real-world dispatch model.

The map is a schematic Plotly visualization of the existing NetworkX graph. Train positions, signals, occupied tracks, predictions, and decisions are simulated. The dashboard is not connected to live GNSS, Indian Railways systems, signaling hardware, or certified safety infrastructure.
