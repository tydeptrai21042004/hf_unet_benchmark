from __future__ import annotations

import math
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_seed_results, prepare_kvasir_seg
from src.datasets import build_dataset, infer_dataset_paths, normalize_dataset_name



def _write_pair(image_path: Path, mask_path: Path, value: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(value, value, value)).save(image_path)
    Image.new("L", (8, 8), color=255 if value % 2 else 0).save(mask_path)



def test_prepare_find_dataset_root_supports_pranet_style_bundle(tmp_path: Path):
    dataset_root = tmp_path / "TestDataset" / "ETIS-LaribPolypDB"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "masks").mkdir(parents=True)
    found = prepare_kvasir_seg._find_dataset_root(tmp_path, "etis")
    assert found == dataset_root



def test_build_dataset_supports_clinicdb_with_split_file(tmp_path: Path):
    root = tmp_path / "data"
    image_dir = root / "raw" / "CVC-ClinicDB" / "images"
    mask_dir = root / "raw" / "CVC-ClinicDB" / "masks"
    for index in range(3):
        stem = f"sample_{index}"
        _write_pair(image_dir / f"{stem}.png", mask_dir / f"{stem}.png", index)
    splits = root / "splits"
    splits.mkdir(parents=True)
    (splits / "train.txt").write_text("sample_0\nsample_1\n", encoding="utf-8")

    dataset = build_dataset(name="clinicdb", root=root, split="train", image_size=None)
    assert len(dataset) == 2
    first = dataset[0]
    assert first["image"].size == (8, 8)



def test_infer_dataset_paths_supports_processed_layout_for_cvc300(tmp_path: Path):
    (tmp_path / "processed" / "images_352").mkdir(parents=True)
    (tmp_path / "processed" / "masks_352").mkdir(parents=True)
    paths = infer_dataset_paths(tmp_path, dataset_name="cvc_300", image_size=352)
    assert paths.image_dir.name == "images_352"
    assert paths.mask_dir.name == "masks_352"



def test_normalize_dataset_name_handles_added_polyp_datasets():
    assert normalize_dataset_name("CVC-ClinicDB") == "cvc_clinicdb"
    assert normalize_dataset_name("CVC-ColonDB") == "cvc_colondb"
    assert normalize_dataset_name("ETIS-Larib") == "etis"
    assert normalize_dataset_name("CVC300") == "cvc_300"



def test_aggregate_seed_results_computes_mean_and_std():
    rows = [
        {"model": "unet", "dataset": "kvasir_seg", "split": "test", "seed": "1", "dice": 0.8, "iou": 0.7, "precision": 0.9, "recall": 0.85, "mae": 0.1, "loss": 0.2},
        {"model": "unet", "dataset": "kvasir_seg", "split": "test", "seed": "2", "dice": 1.0, "iou": 0.9, "precision": 0.95, "recall": 0.9, "mae": 0.2, "loss": 0.4},
    ]
    summary = aggregate_seed_results.aggregate_rows(rows)
    assert len(summary) == 1
    row = summary[0]
    assert math.isclose(float(row["dice_mean"]), 0.9)
    assert math.isclose(float(row["dice_std"]), math.sqrt(0.02), rel_tol=1e-6)
    assert row["dice_mean_pm_std"] == "0.9000 ± 0.1414"
    assert row["num_seeds"] == 2
