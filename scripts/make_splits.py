#!/usr/bin/env python3
"""Create deterministic train/val/test splits for segmentation datasets."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import normalize_dataset_name

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS


def _resolve_image_dir(data_root: Path, image_size: Optional[int] = None) -> Path:
    candidates = []
    if image_size is not None:
        candidates.append(data_root / "processed" / f"images_{image_size}")
    candidates.extend(
        [
            data_root / "processed" / "images",
            data_root / "raw" / "Kvasir-SEG" / "images",
            data_root / "Kvasir-SEG" / "images",
            data_root / "images",
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path

    processed_root = data_root / "processed"
    if processed_root.is_dir():
        dynamic = sorted(p for p in processed_root.iterdir() if p.is_dir() and p.name.startswith("images_"))
        if dynamic:
            return dynamic[0]
    raise FileNotFoundError(f"Could not find image directory under: {data_root}")


def _collect_ids(image_dir: Path) -> List[str]:
    return sorted(p.stem for p in image_dir.iterdir() if p.is_file() and _is_image(p))


def _split_ids(ids: Sequence[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(n - 2, 0)
    remaining = n - n_train
    n_val = min(max(n_val, 1 if remaining >= 2 else 0), max(remaining - 1, 0))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def _write_list(items: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test split text files.")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Currently supports kvasir_seg and custom.")
    parser.add_argument("--data-root", type=str, default="data", help="Benchmark data root.")
    parser.add_argument("--image-size", type=int, default=None, help="Preferred processed image size, e.g. 352.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <data-root>/splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_dataset_name(args.dataset)  # validate early
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1")

    data_root = Path(args.data_root)
    image_dir = _resolve_image_dir(data_root, image_size=args.image_size)
    ids = _collect_ids(image_dir)
    if len(ids) < 3:
        raise RuntimeError(f"Need at least 3 samples to create train/val/test splits, found {len(ids)}")

    train_ids, val_ids, test_ids = _split_ids(ids, args.train_ratio, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "splits"
    _write_list(train_ids, output_dir / "train.txt")
    _write_list(val_ids, output_dir / "val.txt")
    _write_list(test_ids, output_dir / "test.txt")

    print(f"Saved splits to: {output_dir}")
    print(f"dataset={normalize_dataset_name(args.dataset)}")
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} total={len(ids)}")


if __name__ == "__main__":
    main()
