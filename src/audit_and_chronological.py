"""Audit saved results and run a separate chronological evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "is_delayed"
LEAKAGE_COLUMNS = {TARGET, "delay_minutes", "primary_delay_cause", "journey_id", "departure_date"}
OPERATIONAL_RISK_COLUMNS = {
    "route_historical_ontime_pct": "historical aggregate; valid only if calculated before this journey",
    "late_incoming_rake": "near-real-time operational information; may be unavailable at planning time",
    "zone_congestion_index": "operational congestion signal; timing and calculation need confirmation",
    "maintenance_score": "pre-departure operational information; confirm availability at prediction time",
}


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def metrics(y_true: pd.Series, predicted, probability) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if y_true.nunique() == 2 else float("nan"),
    }


def model_definitions(y_train: pd.Series) -> dict[str, object]:
    positive_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss", scale_pos_weight=positive_weight, random_state=SEED),
    }


def evaluate_saved_models(frame: pd.DataFrame, report: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    features = frame.drop(columns=[column for column in LEAKAGE_COLUMNS if column in frame.columns])
    target = frame[TARGET]
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=SEED)
    rows = []
    max_difference = 0.0
    majority = int(target.mode().iloc[0])
    rows.append({"experiment": "original_saved_verified", "model": "Majority Class Baseline", **metrics(y_test, [majority] * len(y_test), [majority] * len(y_test)), "matches_saved_report": True})
    for _, expected in report.iterrows():
        name = expected["model"]
        pipeline = joblib.load(Path("models") / f"{name.lower().replace(' ', '_')}.joblib")
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        values = metrics(y_test, predicted, probability)
        max_difference = max(max_difference, *(abs(values[key] - expected[key]) for key in values))
        rows.append({"experiment": "original_saved_verified", "model": name, **values, "matches_saved_report": max_difference < 1e-9})
    return pd.DataFrame(rows), max_difference


def chronological_evaluation(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.assign(_parsed_date=pd.to_datetime(frame["departure_date"], errors="coerce")).sort_values("_parsed_date")
    ordered = ordered.dropna(subset=["_parsed_date"])
    split = int(len(ordered) * 0.8)
    train = ordered.iloc[:split]
    test = ordered.iloc[split:]
    excluded = LEAKAGE_COLUMNS | set(OPERATIONAL_RISK_COLUMNS)
    X_train = train.drop(columns=[column for column in excluded | {"_parsed_date"} if column in train.columns or column in test.columns])
    X_test = test.drop(columns=[column for column in excluded if column in test.columns])
    y_train, y_test = train[TARGET], test[TARGET]
    rows = []
    majority = int(y_train.mode().iloc[0])
    rows.append({"experiment": "chronological_improved", "model": "Majority Class Baseline", **metrics(y_test, [majority] * len(y_test), [majority] * len(y_test))})
    for name, estimator in model_definitions(y_train).items():
        pipeline = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"experiment": "chronological_improved", "model": name, **metrics(y_test, predicted, probability)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/ir_train.csv"))
    parser.add_argument("--max-rows", type=int, default=20_000)
    args = parser.parse_args()
    frame = pd.read_csv(args.data, nrows=None if args.max_rows == 0 else args.max_rows)
    frame[TARGET] = frame[TARGET].astype(int)
    saved_report = pd.read_csv("results/model_comparison.csv")
    verified, max_difference = evaluate_saved_models(frame, saved_report)
    improved = chronological_evaluation(frame)
    comparison = pd.concat([verified, improved], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    verified.to_csv("results/audit_saved_results.csv", index=False)
    improved.to_csv("results/chronological_model_comparison.csv", index=False)
    comparison.to_csv("results/original_vs_improved_comparison.csv", index=False)
    dates = pd.to_datetime(frame["departure_date"], errors="coerce")
    risk_lines = [f"- {column}: {description}" for column, description in OPERATIONAL_RISK_COLUMNS.items() if column in frame]
    report_lines = [
        "RAILWAY MODEL AUDIT",
        f"Rows audited: {len(frame):,}",
        f"Date range: {dates.min()} to {dates.max()}",
        f"Saved metrics verification maximum difference: {max_difference:.12f}",
        f"Saved metrics match: {max_difference < 1e-9}",
        "",
        "ORIGINAL EXPERIMENT",
        "Stratified random 80/20 split; leakage columns excluded by the existing pipeline.",
        "",
        "OPERATIONAL FEATURE REVIEW",
        *risk_lines,
        "These fields are not proven leakage, but the improved experiment excludes them unless their availability before prediction is documented.",
        "",
        "IMPROVED EXPERIMENT",
        "Chronological 80/20 split after sorting by departure_date; target-revealing and operational-risk columns excluded.",
        improved.to_string(index=False),
    ]
    Path("results/audit_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
