#!/usr/bin/env python3
"""Prepare Kvasir-SEG for the HF-U-Net benchmark.

Supports three input modes:
1) Existing extracted dataset folder with images/ and masks/
2) Zip archive containing Kvasir-SEG
3) Optional direct download URL

Outputs a benchmark-friendly layout:
    data/
      raw/Kvasir-SEG/images
      raw/Kvasir-SEG/masks
      processed/images_<size>
      processed/masks_<size>
      processed/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_dataset_root(root: Path) -> Optional[Path]:
    candidates = [
        root,
        root / "Kvasir-SEG",
        root / "kvasir-seg",
        root / "kvasir_seg",
        root / "segmented-images",
    ]
    for cand in candidates:
        img = cand / "images"
        msk = cand / "masks"
        if img.is_dir() and msk.is_dir():
            return cand
    for cand in root.rglob("*"):
        if cand.is_dir() and (cand / "images").is_dir() and (cand / "masks").is_dir():
            return cand
    return None


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS


def _collect_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path]]:
    images = sorted([p for p in image_dir.iterdir() if p.is_file() and _is_image(p)])
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and _is_image(p)}
    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for image_path in images:
        sample_id = image_path.stem
        mask_path = mask_map.get(sample_id)
        if mask_path is None:
            missing.append(sample_id)
            continue
        pairs.append((sample_id, image_path, mask_path))
    if not pairs:
        raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
    if missing:
        print(f"[WARN] Missing masks for {len(missing)} images. They will be skipped.", file=sys.stderr)
    return pairs


def _copy_raw_pairs(pairs: Sequence[Tuple[str, Path, Path]], raw_images: Path, raw_masks: Path) -> None:
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_masks.mkdir(parents=True, exist_ok=True)
    for sample_id, image_path, mask_path in pairs:
        shutil.copy2(image_path, raw_images / f"{sample_id}{image_path.suffix.lower()}")
        shutil.copy2(mask_path, raw_masks / f"{sample_id}{mask_path.suffix.lower()}")


def _resize_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path, size: int) -> Tuple[int, int]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    orig_h, orig_w = image.height, image.width

    image_resized = image.resize((size, size), resample=Image.BILINEAR)
    mask_resized = mask.resize((size, size), resample=Image.NEAREST)
    mask_binary = mask_resized.point(lambda x: 255 if x > 127 else 0)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image_resized.save(out_image)
    mask_binary.save(out_mask)
    return orig_h, orig_w


def _write_metadata(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    dataset_root = _find_dataset_root(extract_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Could not find extracted Kvasir-SEG dataset under {extract_dir}. Expected images/ and masks/ folders."
        )
    return dataset_root


def _maybe_download(url: str, dst: Path) -> Path:
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests is required for download mode. Install it or pass --zip-path/--source-dir.") from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dst.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kvasir-SEG for benchmark training.")
    parser.add_argument("--data-root", type=str, default="data", help="Benchmark data root.")
    parser.add_argument("--source-dir", type=str, default=None, help="Path to extracted Kvasir-SEG folder or its parent.")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to a Kvasir-SEG zip archive.")
    parser.add_argument("--download-url", type=str, default=None, help="Optional URL to download a zip archive.")
    parser.add_argument("--download-dst", type=str, default=None, help="Optional destination path for downloaded zip.")
    parser.add_argument("--image-size", type=int, default=352, help="Output square size for processed images/masks.")
    parser.add_argument("--skip-raw-copy", action="store_true", help="Do not copy files into data/raw/Kvasir-SEG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_root = data_root / "raw" / "Kvasir-SEG"
    processed_root = data_root / "processed"
    processed_images = processed_root / f"images_{args.image_size}"
    processed_masks = processed_root / f"masks_{args.image_size}"

    dataset_root: Optional[Path] = None

    if args.source_dir:
        dataset_root = _find_dataset_root(Path(args.source_dir))
        if dataset_root is None:
            raise FileNotFoundError(f"Could not locate images/ and masks/ under source-dir: {args.source_dir}")
    elif args.zip_path:
        dataset_root = _extract_zip(Path(args.zip_path), data_root / "_tmp_extract")
    elif args.download_url:
        download_dst = Path(args.download_dst) if args.download_dst else data_root / "downloads" / "kvasir_seg.zip"
        zip_path = _maybe_download(args.download_url, download_dst)
        dataset_root = _extract_zip(zip_path, data_root / "_tmp_extract")
    else:
        # Fallback: maybe raw dataset already exists.
        dataset_root = _find_dataset_root(data_root)
        if dataset_root is None:
            raise ValueError(
                "Provide one of --source-dir, --zip-path, or --download-url. "
                "Alternatively, place extracted data under data/raw/Kvasir-SEG/images and masks."
            )

    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    pairs = _collect_pairs(image_dir, mask_dir)

    if not args.skip_raw_copy:
        _copy_raw_pairs(pairs, raw_root / "images", raw_root / "masks")

    metadata_rows: List[dict] = []
    for sample_id, image_path, mask_path in pairs:
        out_image = processed_images / f"{sample_id}.png"
        out_mask = processed_masks / f"{sample_id}.png"
        orig_h, orig_w = _resize_pair(image_path, mask_path, out_image, out_mask, args.image_size)
        metadata_rows.append(
            {
                "id": sample_id,
                "image_path": str(out_image.as_posix()),
                "mask_path": str(out_mask.as_posix()),
                "orig_height": orig_h,
                "orig_width": orig_w,
                "proc_height": args.image_size,
                "proc_width": args.image_size,
            }
        )

    _write_metadata(metadata_rows, processed_root / "metadata.csv")
    print(f"Prepared {len(metadata_rows)} samples at: {processed_root}")
    print(f"Processed images: {processed_images}")
    print(f"Processed masks : {processed_masks}")


if __name__ == "__main__":
    main()
