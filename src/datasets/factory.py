from __future__ import annotations

from pathlib import Path
from typing import Optional

from .kvasir_seg_dataset import KvasirSegDataset, infer_dataset_paths
from .registry import normalize_dataset_name


SUPPORTED_BINARY_SEG_DATASETS = {
    "kvasir_seg",
    "cvc_clinicdb",
    "etis",
    "cvc_colondb",
    "cvc_300",
    "custom",
}


def build_dataset(
    name: str,
    root: str | Path,
    split: Optional[str] = None,
    split_file: Optional[str | Path] = None,
    image_dir: Optional[str | Path] = None,
    mask_dir: Optional[str | Path] = None,
    image_size: Optional[int] = None,
    transform=None,
    return_paths: bool = False,
    strict_pairing: bool = True,
) -> KvasirSegDataset:
    dataset_name = normalize_dataset_name(name)
    if dataset_name not in SUPPORTED_BINARY_SEG_DATASETS:
        raise ValueError(f"Unsupported dataset for the generic binary-seg loader: {dataset_name}")
    return KvasirSegDataset(
        root=root,
        split=split,
        split_file=split_file,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        transform=transform,
        return_paths=return_paths,
        strict_pairing=strict_pairing,
        dataset_name=dataset_name,
    )


__all__ = ["build_dataset", "infer_dataset_paths", "SUPPORTED_BINARY_SEG_DATASETS"]
