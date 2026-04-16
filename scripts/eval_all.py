#!/usr/bin/env python3
"""Evaluate all or selected models on a chosen split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS = [
    "unet",
    "unet_cbam",
    "unetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "proposal_hf_unet",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all or selected models.")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma-separated model names")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Currently supports kvasir_seg and custom.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = PROJECT_ROOT / "scripts" / "eval_one.py"
    config_dir = PROJECT_ROOT / args.config_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        cmd = [sys.executable, str(script), "--model", model, "--dataset", args.dataset, "--split", args.split]
        cfg = config_dir / f"{model}.yaml"
        if cfg.exists():
            cmd += ["--config", str(cfg)]
        for flag in [
            ("--data-root", args.data_root),
            ("--image-size", args.image_size),
            ("--batch-size", args.batch_size),
            ("--device", args.device),
            ("--output-root", args.output_root),
            ("--num-workers", args.num_workers),
        ]:
            if flag[1] is not None:
                cmd += [flag[0], str(flag[1])]
        if args.save_predictions:
            cmd.append("--save-predictions")
        if args.save_visualizations:
            cmd.append("--save-visualizations")
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
