#!/usr/bin/env python3
"""Aggregate repeated-seed evaluation outputs into mean/std summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

METRIC_COLUMNS = ["dice", "iou", "precision", "recall", "mae", "loss"]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-seed evaluation results into mean/std tables.")
    parser.add_argument("--output-root", type=str, default="outputs", help="Root containing seed_<n>/ subdirectories.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Defaults to auto-detect seed_* folders.")
    parser.add_argument("--save-name", type=str, default="multi_seed_summary")
    return parser.parse_args()



def _parse_seeds(value: str | None) -> List[str]:
    if value is None or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]



def _discover_seed_roots(output_root: Path, seeds: List[str]) -> List[Tuple[str, Path]]:
    if seeds:
        return [(seed, output_root / f"seed_{seed}") for seed in seeds]
    roots: List[Tuple[str, Path]] = []
    for path in sorted(output_root.glob("seed_*")):
        if path.is_dir():
            roots.append((path.name.replace("seed_", "", 1), path))
    return roots



def _find_metric_files(seed_root: Path) -> List[Path]:
    tables_root = seed_root / "results" / "tables"
    if tables_root.is_dir():
        files = sorted(tables_root.rglob("*_metrics.json"))
        if files:
            return files
    return sorted(seed_root.rglob("metrics_*.json"))



def _load_rows(seed_roots: Iterable[Tuple[str, Path]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed, seed_root in seed_roots:
        for path in _find_metric_files(seed_root):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            metrics = payload.get("metrics", {})
            row: Dict[str, object] = {
                "model": payload.get("model", path.stem.replace("_metrics", "")),
                "dataset": payload.get("dataset", ""),
                "split": payload.get("split", "test"),
                "seed": str(payload.get("seed") if payload.get("seed") is not None else seed),
                "checkpoint": payload.get("checkpoint", ""),
                "num_samples": payload.get("num_samples", ""),
            }
            row.update(metrics)
            rows.append(row)
    return rows



def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan



def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("model", "")), str(row.get("dataset", "")), str(row.get("split", "")))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (model, dataset, split), items in sorted(grouped.items()):
        summary: Dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "split": split,
            "num_seeds": len(items),
            "seeds": ",".join(sorted(str(item.get("seed", "")) for item in items)),
        }
        for metric in METRIC_COLUMNS:
            values = [_safe_float(item.get(metric)) for item in items]
            values = [value for value in values if not math.isnan(value)]
            if not values:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_std"] = ""
                summary[f"{metric}_mean_pm_std"] = ""
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(summary)
    return summary_rows



def _save_json(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)



def _save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["model", "dataset", "split", "num_seeds", "seeds"]
    else:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def _save_latex(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model", "Dataset", "Split", "Seeds", "Dice", "IoU", "Precision", "Recall", "MAE", "Loss"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l l c c c c c c c}\n")
        f.write("\\toprule\n")
        f.write(" {} \\\n".format(" & ".join(headers)))
        f.write("\\midrule\n")
        for row in rows:
            parts = [
                str(row.get("model", "")),
                str(row.get("dataset", "")),
                str(row.get("split", "")),
                str(row.get("num_seeds", "")),
                str(row.get("dice_mean_pm_std", "")),
                str(row.get("iou_mean_pm_std", "")),
                str(row.get("precision_mean_pm_std", "")),
                str(row.get("recall_mean_pm_std", "")),
                str(row.get("mae_mean_pm_std", "")),
                str(row.get("loss_mean_pm_std", "")),
            ]
            f.write(" {} \\\n".format(" & ".join(parts)))
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")



def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    seed_roots = _discover_seed_roots(output_root, _parse_seeds(args.seeds))
    if not seed_roots:
        raise FileNotFoundError(f"No seed_* directories found under: {output_root}")
    rows = _load_rows(seed_roots)
    if not rows:
        raise FileNotFoundError(f"No per-seed metric files found under: {output_root}")
    summary_rows = aggregate_rows(rows)
    out_dir = output_root / "results" / "tables"
    _save_json(summary_rows, out_dir / f"{args.save_name}.json")
    _save_csv(summary_rows, out_dir / f"{args.save_name}.csv")
    _save_latex(summary_rows, out_dir / f"{args.save_name}.tex")
    print(f"Aggregated {len(rows)} per-seed rows into {len(summary_rows)} summary rows at: {out_dir}")


if __name__ == "__main__":
    main()
