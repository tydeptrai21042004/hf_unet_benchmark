#!/usr/bin/env python3
"""Aggregate evaluation JSON files into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_COLUMNS = ["model", "split", "dice", "iou", "precision", "recall", "mae", "loss", "num_samples", "checkpoint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark result tables.")
    parser.add_argument("--output-root", type=str, default=".", help="Project root or experiment output root")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing *_metrics.json files")
    parser.add_argument("--save-name", type=str, default="benchmark_results")
    return parser.parse_args()


def find_metric_files(base: Path) -> List[Path]:
    files = sorted(base.rglob("*_metrics.json"))
    if files:
        return files
    return sorted(base.rglob("metrics_*.json"))


def load_rows(files: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        row = {
            "model": payload.get("model", path.stem.replace("_metrics", "")),
            "split": payload.get("split", "test"),
            "checkpoint": payload.get("checkpoint", ""),
            "num_samples": payload.get("num_samples", ""),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in DEFAULT_COLUMNS})


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def save_latex(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Split", "Dice$\\uparrow$", "IoU$\\uparrow$", "Precision$\\uparrow$", "Recall$\\uparrow$", "MAE$\\downarrow$", "Loss"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l c c c c c c}\n")
        f.write("\\toprule\n")
        f.write(" {} \\\n".format(" & ".join(headers)))
        f.write("\\midrule\n")
        for row in rows:
            line = " & ".join(
                [
                    fmt(row.get("model", "")),
                    fmt(row.get("split", "")),
                    fmt(row.get("dice", "")),
                    fmt(row.get("iou", "")),
                    fmt(row.get("precision", "")),
                    fmt(row.get("recall", "")),
                    fmt(row.get("mae", "")),
                    fmt(row.get("loss", "")),
                ]
            )
            f.write(f"{line} \\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    input_dir = Path(args.input_dir) if args.input_dir else output_root / "results" / "tables"
    files = find_metric_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No metric JSON files found under: {input_dir}")
    rows = sorted(load_rows(files), key=lambda x: (str(x.get("split", "")), str(x.get("model", ""))))
    save_csv(rows, output_root / "results" / "tables" / f"{args.save_name}.csv")
    save_latex(rows, output_root / "results" / "tables" / f"{args.save_name}.tex")
    print(f"Exported {len(rows)} rows to {output_root / 'results' / 'tables'}")


if __name__ == "__main__":
    main()
