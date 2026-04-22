from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    canonical_dir: str
    default_download_url: Optional[str] = None
    description: str = ""


DATASET_SPECS = {
    "kvasir_seg": DatasetSpec(
        name="kvasir_seg",
        aliases=("kvasir_seg", "kvasir-seg", "kvasir"),
        canonical_dir="Kvasir-SEG",
        default_download_url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        description="Kvasir-SEG polyp segmentation dataset.",
    ),
    "cvc_clinicdb": DatasetSpec(
        name="cvc_clinicdb",
        aliases=("cvc_clinicdb", "cvc-clinicdb", "clinicdb", "cvc612", "cvc-612"),
        canonical_dir="CVC-ClinicDB",
        default_download_url=None,
        description="CVC-ClinicDB polyp segmentation dataset.",
    ),
    "etis": DatasetSpec(
        name="etis",
        aliases=("etis", "etis-larib", "etis_larib", "etis-laribpolypdb", "etis_laribpolypdb"),
        canonical_dir="ETIS-LaribPolypDB",
        default_download_url=None,
        description="ETIS-LaribPolypDB polyp segmentation dataset.",
    ),
    "cvc_colondb": DatasetSpec(
        name="cvc_colondb",
        aliases=("cvc_colondb", "cvc-colondb", "colondb", "cvc-colon"),
        canonical_dir="CVC-ColonDB",
        default_download_url=None,
        description="CVC-ColonDB polyp segmentation dataset.",
    ),
    "cvc_300": DatasetSpec(
        name="cvc_300",
        aliases=("cvc_300", "cvc-300", "cvc300"),
        canonical_dir="CVC-300",
        default_download_url=None,
        description="CVC-300 polyp segmentation dataset.",
    ),
    "custom": DatasetSpec(
        name="custom",
        aliases=("custom", "custom_binary_seg", "custom_segmentation"),
        canonical_dir="custom",
        default_download_url=None,
        description="Custom binary segmentation dataset with matching images/masks and split files.",
    ),
}

_ALIAS_TO_NAME: Dict[str, str] = {}
for _name, _spec in DATASET_SPECS.items():
    for _alias in _spec.aliases:
        _ALIAS_TO_NAME[_alias.lower()] = _name

SUPPORTED_DATASETS = tuple(sorted(DATASET_SPECS))


def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return "kvasir_seg"
    value = str(name).strip().lower().replace(" ", "_")
    if not value:
        return "kvasir_seg"
    try:
        return _ALIAS_TO_NAME[value]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}") from exc



def get_dataset_spec(name: Optional[str] = None) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(name)]


__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "SUPPORTED_DATASETS",
    "normalize_dataset_name",
    "get_dataset_spec",
]
