#!/usr/bin/env python3
"""Train multiple benchmark models sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS = [
    "unet",
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
    parser = argparse.ArgumentParser(description="Train all or selected models sequentially.")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma-separated model names")
    parser.add_argument("--config-dir", type=str, default="configs", help="Directory containing <model>.yaml configs")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = PROJECT_ROOT / "scripts" / "train_one.py"
    config_dir = PROJECT_ROOT / args.config_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        cmd = [sys.executable, str(script), "--model", model]
        cfg = config_dir / f"{model}.yaml"
        if cfg.exists():
            cmd += ["--config", str(cfg)]
        for flag in [
            ("--data-root", args.data_root),
            ("--image-size", args.image_size),
            ("--batch-size", args.batch_size),
            ("--epochs", args.epochs),
            ("--lr", args.lr),
            ("--device", args.device),
            ("--output-root", args.output_root),
            ("--num-workers", args.num_workers),
            ("--seed", args.seed),
        ]:
            if flag[1] is not None:
                cmd += [flag[0], str(flag[1])]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
