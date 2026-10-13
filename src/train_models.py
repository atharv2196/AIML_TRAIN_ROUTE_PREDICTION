"""Train and evaluate the three requested classifiers on local data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "is_delayed"
LEAKAGE = {TARGET, "delay_minutes", "primary_delay_cause", "journey_id", "departure_date"}


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/ir_train.csv"))
    parser.add_argument("--max-rows", type=int, default=300_000, help="Rows read from the raw file; 0 means all rows.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("graphs").mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, nrows=None if args.max_rows == 0 else args.max_rows)
    frame[TARGET] = frame[TARGET].astype(int)
    features = frame.drop(columns=[column for column in LEAKAGE if column in frame.columns])
    target = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=args.test_size, stratify=target, random_state=SEED)
    preprocessor = make_preprocessor(X_train)
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss", scale_pos_weight=positive_weight, random_state=SEED),
    }
    results = []
    predictions = {}
    probabilities = {}
    pipelines = {}
    print(f"Dataset: {args.data.name}\nRows used: {len(frame):,}\nFeatures used: {features.shape[1]}\nPrediction target: {TARGET}\nTraining samples: {len(X_train):,}\nTesting samples: {len(X_test):,}\n")
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        training_time = time.perf_counter() - start
        start = time.perf_counter()
        predicted = pipeline.predict(X_test)
        prediction_time = time.perf_counter() - start
        probability = pipeline.predict_proba(X_test)[:, 1]
        predictions[name] = predicted
        probabilities[name] = probability
        results.append({"model": name, "accuracy": accuracy_score(y_test, predicted), "precision": precision_score(y_test, predicted, zero_division=0), "recall": recall_score(y_test, predicted, zero_division=0), "f1": f1_score(y_test, predicted, zero_division=0), "balanced_accuracy": balanced_accuracy_score(y_test, predicted), "roc_auc": roc_auc_score(y_test, probability), "training_time_seconds": training_time, "prediction_time_seconds": prediction_time})
        joblib.dump(pipeline, Path("models") / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"{name}: accuracy={results[-1]['accuracy']:.4f}, precision={results[-1]['precision']:.4f}, recall={results[-1]['recall']:.4f}, f1={results[-1]['f1']:.4f}")
    report = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    report.to_csv("results/model_comparison.csv", index=False)
    best = report.loc[report["f1"].idxmax()]
    lines = ["MODEL COMPARISON", report.to_string(index=False), "", f"BEST MODEL BY F1: {best['model']} ({best['f1']:.4f})", "", "CONFUSION MATRICES"]
    for name, predicted in predictions.items():
        matrix = confusion_matrix(y_test, predicted)
        lines.extend([f"{name}\n{matrix}", classification_report(y_test, predicted, zero_division=0)])
        plt.figure(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(Path("graphs") / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()
    for name in ["Random Forest", "XGBoost"]:
        model = pipelines[name].named_steps["model"]
        feature_names = pipelines[name].named_steps["preprocess"].get_feature_names_out()
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
        plt.figure(figsize=(9, 6))
        importance.sort_values().plot(kind="barh")
        plt.title(f"{name} feature importance (top 20)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(Path("graphs") / f"feature_importance_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()
    metric_plot = report.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    metric_plot.plot(kind="bar", ylim=(0, 1), figsize=(8, 4), title="Model metric comparison")
    plt.tight_layout(); plt.savefig("graphs/model_metrics.png", dpi=150); plt.close()
    report.set_index("model")["training_time_seconds"].plot(kind="bar", figsize=(7, 4), title="Model training time (seconds)")
    plt.tight_layout(); plt.savefig("graphs/training_time.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 5))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probability):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig("graphs/roc_curves.png", dpi=150); plt.close()
    Path("results/evaluation_report.txt").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print("\nTRAINING COMPLETED\nModels saved successfully.\nEvaluation reports saved successfully.\nGraphs generated successfully.")


if __name__ == "__main__":
    main()