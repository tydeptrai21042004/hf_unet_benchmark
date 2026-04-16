from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    default_download_url: Optional[str] = None
    description: str = ""


DATASET_SPECS = {
    "kvasir_seg": DatasetSpec(
        name="kvasir_seg",
        aliases=("kvasir_seg", "kvasir-seg", "kvasir"),
        default_download_url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        description="Kvasir-SEG polyp segmentation dataset.",
    ),
    "custom": DatasetSpec(
        name="custom",
        aliases=("custom", "custom_binary_seg", "custom_segmentation"),
        default_download_url=None,
        description="Custom binary segmentation dataset with matching images/masks and split files.",
    ),
}

_ALIAS_TO_NAME: Dict[str, str] = {}
for _name, _spec in DATASET_SPECS.items():
    for _alias in _spec.aliases:
        _ALIAS_TO_NAME[_alias.lower()] = _name


def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return "kvasir_seg"
    value = str(name).strip().lower().replace(" ", "_")
    if not value:
        return "kvasir_seg"
    try:
        return _ALIAS_TO_NAME[value]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}") from exc


def get_dataset_spec(name: Optional[str] = None) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(name)]


__all__ = ["DatasetSpec", "DATASET_SPECS", "normalize_dataset_name", "get_dataset_spec"]
