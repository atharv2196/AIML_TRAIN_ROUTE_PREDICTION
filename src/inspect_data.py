"""Inspect the local railway datasets and write a concise report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def inspect_csv(path: Path) -> str:
    missing: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    rows = duplicates = 0
    seen: set[tuple[str, ...]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            rows += 1
            values = tuple(row.get(column, "") for column in columns)
            duplicates += values in seen
            seen.add(values)
            for key, value in row.items():
                if not value or not value.strip():
                    missing[key] += 1
            labels[row.get("is_delayed", "")] += 1
    return "\n".join(
        [
            f"Dataset: {path.name}",
            f"Rows: {rows:,}",
            f"Columns ({len(columns)}): {', '.join(columns)}",
            f"Duplicate rows: {duplicates:,}",
            f"Target counts: {dict(labels)}",
            f"Missing values: {dict(missing) if missing else 'none'}",
        ]
    )


def inspect_json(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        features = data.get("features")
        detail = f", features={len(features):,}" if isinstance(features, list) else ""
        return f"{path.name}: object, keys={list(data)[:10]}{detail}"
    return f"{path.name}: {type(data).__name__}, records={len(data):,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("results/data_inspection.txt"))
    args = parser.parse_args()
    lines = [inspect_csv(args.data_dir / "ir_train.csv")]
    lines.extend(inspect_json(args.data_dir / name) for name in ["stations.json", "trains.json", "schedules.json"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print("\n\n".join(lines))


if __name__ == "__main__":
    main()