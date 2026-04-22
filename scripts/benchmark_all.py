#!/usr/bin/env python3
"""Run the full benchmark pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset_spec, normalize_dataset_name
from src.utils import resolve_device

DEFAULT_MODELS = "unet,unet_cbam,unetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,proposal_hf_unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data prep, split generation, training, evaluation, and export.")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS, help="Comma-separated model names for train_all/eval_all")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Kvasir-SEG has a default download URL; the other supported polyp datasets can be prepared from source-dir/zip-path or a custom download URL.")
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
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true", help="Pass through to prepare_kvasir_seg.py for Kaggle-like TLS issues.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepared_dataset_exists(data_root: Path, image_size: int) -> bool:
    return (data_root / "processed" / f"images_{image_size}").is_dir() and (data_root / "processed" / f"masks_{image_size}").is_dir()


def split_files_exist(data_root: Path) -> bool:
    splits_root = data_root / "splits"
    return all((splits_root / f"{name}.txt").is_file() for name in ("train", "val", "test"))


def build_prepare_cmd(args: argparse.Namespace, py: str) -> list[str]:
    dataset_name = normalize_dataset_name(args.dataset)
    spec = get_dataset_spec(dataset_name)
    cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"),
        "--dataset",
        dataset_name,
        "--data-root",
        args.data_root,
        "--image-size",
        str(args.image_size),
    ]
    if args.source_dir:
        cmd += ["--source-dir", args.source_dir]
    if args.zip_path:
        cmd += ["--zip-path", args.zip_path]
    if args.download_url:
        cmd += ["--download-url", args.download_url]
    elif spec.default_download_url and not args.source_dir and not args.zip_path:
        cmd += ["--download-url", spec.default_download_url]
    if args.download_dst:
        cmd += ["--download-dst", args.download_dst]
    if getattr(args, "allow_insecure_download", False):
        cmd += ["--allow-insecure-download"]
    return cmd


def main() -> None:
    args = parse_args()
    py = sys.executable
    device = resolve_device(args.device)
    dataset_name = normalize_dataset_name(args.dataset)
    data_root = Path(args.data_root)

    must_prepare = prepared_dataset_exists(data_root, args.image_size) is False
    if must_prepare or not args.skip_prepare:
        run(build_prepare_cmd(args, py))

    must_split = split_files_exist(data_root) is False
    if must_split or not args.skip_splits:
        run([
            py,
            str(PROJECT_ROOT / "scripts" / "make_splits.py"),
            "--dataset",
            dataset_name,
            "--data-root",
            args.data_root,
            "--image-size",
            str(args.image_size),
            "--seed",
            str(args.seed),
        ])

    train_cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "train_all.py"),
        "--models",
        args.models,
        "--dataset",
        dataset_name,
        "--config-dir",
        args.config_dir,
        "--data-root",
        args.data_root,
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
        "--device",
        device,
    ]
    for flag in [("--batch-size", args.batch_size), ("--epochs", args.epochs), ("--lr", args.lr), ("--output-root", args.output_root), ("--num-workers", args.num_workers)]:
        if flag[1] is not None:
            train_cmd += [flag[0], str(flag[1])]
    run(train_cmd)

    eval_cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "eval_all.py"),
        "--models",
        args.models,
        "--dataset",
        dataset_name,
        "--config-dir",
        args.config_dir,
        "--data-root",
        args.data_root,
        "--image-size",
        str(args.image_size),
        "--device",
        device,
    ]
    for flag in [("--batch-size", args.batch_size), ("--output-root", args.output_root), ("--num-workers", args.num_workers), ("--seed", args.seed)]:
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
