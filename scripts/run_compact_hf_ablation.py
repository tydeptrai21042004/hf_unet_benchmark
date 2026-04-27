#!/usr/bin/env python3
"""Run the compact 9-variant HF-U-Net ablation suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = [
    "unet",
    "unet_conv_bottleneck",
    "unet_fft_bottleneck",
    "proposal_hf_unet",
    "hf_unet_wo_hartley",
    "hf_unet_wo_fourier_kernel",
    "hf_unet_wo_residual",
    "hf_unet_encoder_stage4",
    "hf_unet_decoder_stage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact HF-U-Net ablation training/evaluation.")
    parser.add_argument("--dataset", type=str, default="cvc_clinicdb")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs_compact_hf_ablation")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    models = ",".join(ABLATION_MODELS)
    common = [
        "--models", models,
        "--dataset", args.dataset,
        "--config-dir", "configs/ablation",
        "--data-root", args.data_root,
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output-root", args.output_root,
    ]
    for flag, value in [("--batch-size", args.batch_size), ("--epochs", args.epochs), ("--num-workers", args.num_workers)]:
        if value is not None:
            common += [flag, str(value)]

    run([sys.executable, str(PROJECT_ROOT / "scripts" / "train_all.py"), *common])
    if not args.skip_eval:
        run([sys.executable, str(PROJECT_ROOT / "scripts" / "eval_all.py"), *common])
        run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_results.py"), "--output-root", args.output_root])


if __name__ == "__main__":
    main()
