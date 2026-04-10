#!/usr/bin/env python3
"""Run the full benchmark pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data prep, split generation, training, evaluation, and export.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names for train_all/eval_all")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--zip-path", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_prepare:
        cmd = [py, str(PROJECT_ROOT / "scripts" / "prepare_kvasir_seg.py"), "--data-root", args.data_root, "--image-size", str(args.image_size)]
        if args.source_dir:
            cmd += ["--source-dir", args.source_dir]
        if args.zip_path:
            cmd += ["--zip-path", args.zip_path]
        if args.download_url:
            cmd += ["--download-url", args.download_url]
        run(cmd)

    if not args.skip_splits:
        run([py, str(PROJECT_ROOT / "scripts" / "make_splits.py"), "--data-root", args.data_root, "--seed", str(args.seed)])

    train_cmd = [py, str(PROJECT_ROOT / "scripts" / "train_all.py"), "--config-dir", args.config_dir, "--data-root", args.data_root, "--image-size", str(args.image_size), "--seed", str(args.seed)]
    if args.models:
        train_cmd += ["--models", args.models]
    for flag in [("--batch-size", args.batch_size), ("--epochs", args.epochs), ("--lr", args.lr), ("--device", args.device), ("--output-root", args.output_root), ("--num-workers", args.num_workers)]:
        if flag[1] is not None:
            train_cmd += [flag[0], str(flag[1])]
    run(train_cmd)

    eval_cmd = [py, str(PROJECT_ROOT / "scripts" / "eval_all.py"), "--config-dir", args.config_dir, "--data-root", args.data_root, "--image-size", str(args.image_size)]
    if args.models:
        eval_cmd += ["--models", args.models]
    for flag in [("--batch-size", args.batch_size), ("--device", args.device), ("--output-root", args.output_root), ("--num-workers", args.num_workers)]:
        if flag[1] is not None:
            eval_cmd += [flag[0], str(flag[1])]
    if args.save_predictions:
        eval_cmd.append("--save-predictions")
    if args.save_visualizations:
        eval_cmd.append("--save-visualizations")
    run(eval_cmd)

    export_cmd = [py, str(PROJECT_ROOT / "scripts" / "export_results.py")]
    if args.output_root:
        export_cmd += ["--output-root", args.output_root]
    run(export_cmd)


if __name__ == "__main__":
    main()
