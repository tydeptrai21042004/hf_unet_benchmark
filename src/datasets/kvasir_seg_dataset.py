"""Kvasir-SEG dataset implementation for the HF-U-Net benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_eval_transforms, build_train_transforms

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
Sample = MutableMapping[str, object]


@dataclass(frozen=True)
class KvasirPaths:
    image_dir: Path
    mask_dir: Path


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def _resolve_existing_dir(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _resolve_processed_pair(root: Path, image_size: Optional[int] = None) -> Optional[KvasirPaths]:
    processed_root = root / "processed"
    if not processed_root.is_dir():
        return None

    if image_size is not None:
        image_dir = processed_root / f"images_{image_size}"
        mask_dir = processed_root / f"masks_{image_size}"
        if image_dir.is_dir() and mask_dir.is_dir():
            return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)

    image_dir = processed_root / "images"
    mask_dir = processed_root / "masks"
    if image_dir.is_dir() and mask_dir.is_dir():
        return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)

    suffixes: list[str] = []
    for path in processed_root.iterdir():
        if path.is_dir() and path.name.startswith("images_"):
            suffixes.append(path.name[len("images_"):])
    for suffix in sorted(set(suffixes)):
        image_dir = processed_root / f"images_{suffix}"
        mask_dir = processed_root / f"masks_{suffix}"
        if image_dir.is_dir() and mask_dir.is_dir():
            return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)
    return None


def infer_kvasir_paths(root: str | Path, image_size: Optional[int] = None) -> KvasirPaths:
    """Infer image and mask directories from a benchmark-style root.

    Supported layouts include:

    1) processed benchmark layout
       root/
         processed/
           images_<size>/
           masks_<size>/

    2) raw Kvasir-SEG layout
       root/
         raw/
           Kvasir-SEG/
             images/
             masks/

    3) direct dataset root
       root/
         images/
         masks/
    """
    root = Path(root)

    processed_pair = _resolve_processed_pair(root, image_size=image_size)
    if processed_pair is not None:
        return processed_pair

    image_dir = _resolve_existing_dir(
        [
            root / "raw" / "Kvasir-SEG" / "images",
            root / "Kvasir-SEG" / "images",
            root / "images",
        ]
    )
    mask_dir = _resolve_existing_dir(
        [
            root / "raw" / "Kvasir-SEG" / "masks",
            root / "Kvasir-SEG" / "masks",
            root / "masks",
        ]
    )

    if image_dir is None or mask_dir is None:
        expected = "processed/images_<size> + processed/masks_<size>, raw/Kvasir-SEG/images + raw/Kvasir-SEG/masks, or images + masks"
        raise FileNotFoundError(
            "Could not infer Kvasir-SEG image/mask directories from root: "
            f"{root}. Expected folders like {expected}."
        )

    return KvasirPaths(image_dir=image_dir, mask_dir=mask_dir)


class KvasirSegDataset(Dataset):
    """Binary segmentation dataset for Kvasir-SEG or compatible image/mask layouts."""

    def __init__(
        self,
        root: str | Path,
        split: Optional[str] = None,
        split_file: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        mask_dir: Optional[str | Path] = None,
        image_size: Optional[int] = None,
        transform: Optional[Callable[[Sample], Sample]] = None,
        return_paths: bool = False,
        strict_pairing: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.return_paths = return_paths
        self.strict_pairing = strict_pairing
        self.transform = transform
        self.image_size = image_size

        if image_dir is None or mask_dir is None:
            inferred = infer_kvasir_paths(self.root, image_size=image_size)
            self.image_dir = inferred.image_dir if image_dir is None else Path(image_dir)
            self.mask_dir = inferred.mask_dir if mask_dir is None else Path(mask_dir)
        else:
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        if split_file is None and split is not None:
            candidate = self.root / "splits" / f"{split}.txt"
            if candidate.exists():
                split_file = candidate

        self.samples = self._build_samples(split_file=split_file)
        if not self.samples:
            raise RuntimeError(
                "No valid image-mask pairs found for dataset. "
                f"image_dir={self.image_dir}, mask_dir={self.mask_dir}, split_file={split_file}"
            )

    def _load_split_ids(self, split_file: Optional[str | Path]) -> Optional[List[str]]:
        if split_file is None:
            return None

        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        ids: List[str] = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                ids.append(Path(item).stem)
        return ids

    def _find_image_by_stem(self, directory: Path, stem: str) -> Optional[Path]:
        for ext in VALID_IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _build_samples(self, split_file: Optional[str | Path]) -> List[Tuple[str, Path, Path]]:
        split_ids = self._load_split_ids(split_file)
        samples: List[Tuple[str, Path, Path]] = []

        if split_ids is not None:
            for sample_id in split_ids:
                image_path = self._find_image_by_stem(self.image_dir, sample_id)
                mask_path = self._find_image_by_stem(self.mask_dir, sample_id)

                if image_path is None or mask_path is None:
                    if self.strict_pairing:
                        raise FileNotFoundError(
                            f"Missing image or mask for sample '{sample_id}'. "
                            f"image_dir={self.image_dir}, mask_dir={self.mask_dir}"
                        )
                    continue
                samples.append((sample_id, image_path, mask_path))
            return samples

        image_files = sorted(p for p in self.image_dir.iterdir() if p.is_file() and _is_image_file(p))
        for image_path in image_files:
            sample_id = image_path.stem
            mask_path = self._find_image_by_stem(self.mask_dir, sample_id)
            if mask_path is None:
                if self.strict_pairing:
                    raise FileNotFoundError(
                        f"No corresponding mask found for image '{image_path.name}' in {self.mask_dir}"
                    )
                continue
            samples.append((sample_id, image_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id, image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        sample: Dict[str, object] = {
            "id": sample_id,
            "image": image,
            "mask": mask,
            "orig_size": (image.height, image.width),
        }

        if self.return_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_ids(self) -> List[str]:
        return [sample_id for sample_id, _, _ in self.samples]


def build_kvasir_datasets(
    root: str | Path,
    image_size: int | Sequence[int] = 352,
    normalize: bool = True,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    return_paths: bool = False,
) -> Dict[str, KvasirSegDataset]:
    """Convenience factory for train/val/test datasets."""
    train_transform = build_train_transforms(image_size=image_size, normalize=normalize)
    eval_transform = build_eval_transforms(image_size=image_size, normalize=normalize)
    resolved_size = image_size if isinstance(image_size, int) else None

    datasets = {
        "train": KvasirSegDataset(
            root=root,
            split=train_split,
            image_size=resolved_size,
            transform=train_transform,
            return_paths=return_paths,
        ),
        "val": KvasirSegDataset(
            root=root,
            split=val_split,
            image_size=resolved_size,
            transform=eval_transform,
            return_paths=return_paths,
        ),
        "test": KvasirSegDataset(
            root=root,
            split=test_split,
            image_size=resolved_size,
            transform=eval_transform,
            return_paths=return_paths,
        ),
    }
    return datasets
