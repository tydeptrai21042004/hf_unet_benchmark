#!/usr/bin/env python3
"""Run the full benchmark pipeline across multiple seeds and aggregate mean/std."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = "unet,unet_cbam,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,proposal_hf_unet"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed benchmarks and aggregate the results.")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=str, default="kvasir_seg")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--zip-path", type=str, default=None)
    parser.add_argument("--download-url", type=str, default=None)
    parser.add_argument("--download-dst", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="42,1337,2024")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    return parser.parse_args()



def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds



def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)



def main() -> None:
    args = parse_args()
    py = sys.executable
    seeds = _parse_seeds(args.seeds)
    benchmark_script = PROJECT_ROOT / "scripts" / "benchmark_all.py"
    aggregate_script = PROJECT_ROOT / "scripts" / "aggregate_seed_results.py"
    base_output_root = Path(args.output_root)

    for index, seed in enumerate(seeds):
        seed_output_root = base_output_root / f"seed_{seed}"
        cmd = [
            py,
            str(benchmark_script),
            "--models", args.models,
            "--dataset", args.dataset,
            "--config-dir", args.config_dir,
            "--data-root", args.data_root,
            "--image-size", str(args.image_size),
            "--device", args.device,
            "--output-root", str(seed_output_root),
            "--seed", str(seed),
        ]
        for flag, value in (
            ("--source-dir", args.source_dir),
            ("--zip-path", args.zip_path),
            ("--download-url", args.download_url),
            ("--download-dst", args.download_dst),
            ("--batch-size", args.batch_size),
            ("--epochs", args.epochs),
            ("--lr", args.lr),
            ("--num-workers", args.num_workers),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        if index > 0:
            cmd += ["--skip-prepare", "--skip-splits"]
        if args.save_predictions:
            cmd.append("--save-predictions")
        if args.save_visualizations:
            cmd.append("--save-visualizations")
        if args.allow_insecure_download:
            cmd.append("--allow-insecure-download")
        _run(cmd)

    _run([
        py,
        str(aggregate_script),
        "--output-root", str(base_output_root),
        "--seeds", ",".join(str(seed) for seed in seeds),
    ])


if __name__ == "__main__":
    main()
