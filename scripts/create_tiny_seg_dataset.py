#!/usr/bin/env python3
"""Create a tiny deterministic binary segmentation dataset for smoke tests.

The dataset follows the repository's generic custom layout:
  <output-root>/processed/images_<image_size>/*.png
  <output-root>/processed/masks_<image_size>/*.png
  <output-root>/splits/{train,val,test}.txt

It is not meant for reporting paper results. Use it only to verify that all
models, dataloaders, training, and evaluation run end-to-end.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic binary segmentation dataset.")
    parser.add_argument("--output-root", type=str, default="data_tiny")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.67)
    parser.add_argument("--val-ratio", type=float, default=0.17)
    return parser.parse_args()


def _draw_sample(image_size: int, index: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    h = w = image_size
    image = Image.new("RGB", (w, h), color=(20, 20, 20))
    mask = Image.new("L", (w, h), color=0)
    draw_img = ImageDraw.Draw(image)
    draw_msk = ImageDraw.Draw(mask)

    cx = rng.randint(w // 4, 3 * w // 4)
    cy = rng.randint(h // 4, 3 * h // 4)
    rx = rng.randint(max(4, w // 10), max(5, w // 4))
    ry = rng.randint(max(4, h // 10), max(5, h // 4))
    bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
    color = (
        int(120 + 80 * math.sin(index + 0.2)),
        int(120 + 80 * math.sin(index * 0.7 + 1.1)),
        int(120 + 80 * math.sin(index * 0.5 + 2.3)),
    )
    draw_img.ellipse(bbox, fill=color)
    draw_msk.ellipse(bbox, fill=255)

    # Add deterministic low-amplitude texture so the task is not completely trivial.
    arr = np.asarray(image).astype(np.int16)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    texture = 10 * np.sin((xx + index) / 5.0) + 8 * np.cos((yy - index) / 7.0)
    arr = np.clip(arr + texture[..., None], 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    return image, mask


def main() -> None:
    args = parse_args()
    if args.num_samples < 6:
        raise ValueError("Use at least 6 samples so train/val/test are non-empty.")
    root = Path(args.output_root)
    image_dir = root / "processed" / f"images_{args.image_size}"
    mask_dir = root / "processed" / f"masks_{args.image_size}"
    split_dir = root / "splits"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ids: list[str] = []
    for i in range(args.num_samples):
        sample_id = f"tiny_{i:04d}"
        ids.append(sample_id)
        image, mask = _draw_sample(args.image_size, i, rng)
        image.save(image_dir / f"{sample_id}.png")
        mask.save(mask_dir / f"{sample_id}.png")

    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * args.train_ratio)))
    n_val = max(1, int(round(n * args.val_ratio)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    splits = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }
    for name, values in splits.items():
        (split_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")

    print(f"Created tiny dataset at {root}")
    print(f"images={image_dir}")
    print(f"masks={mask_dir}")
    print("splits=" + ", ".join(f"{k}:{len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
