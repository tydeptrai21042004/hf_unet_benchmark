from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_all, make_splits, prepare_kvasir_seg
from src.datasets import get_dataset_spec, infer_kvasir_paths, normalize_dataset_name
from src.utils import resolve_device, should_pin_memory


def test_resolve_device_auto_prefers_cuda_when_available():
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"
        assert should_pin_memory("auto") is True


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_missing():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"
        assert resolve_device("cuda") == "cpu"
        assert should_pin_memory("auto") is False


def test_normalize_dataset_name_accepts_aliases():
    assert normalize_dataset_name("kvasir") == "kvasir_seg"
    assert normalize_dataset_name("kvasir-seg") == "kvasir_seg"
    assert normalize_dataset_name("custom") == "custom"


def test_kvasir_dataset_spec_has_default_download_url():
    spec = get_dataset_spec("kvasir_seg")
    assert spec.default_download_url is not None
    assert spec.default_download_url.endswith("kvasir-seg.zip")


def test_infer_kvasir_paths_prefers_requested_processed_size(tmp_path: Path):
    (tmp_path / "processed" / "images_256").mkdir(parents=True)
    (tmp_path / "processed" / "masks_256").mkdir(parents=True)
    (tmp_path / "processed" / "images_352").mkdir(parents=True)
    (tmp_path / "processed" / "masks_352").mkdir(parents=True)

    paths = infer_kvasir_paths(tmp_path, image_size=256)
    assert paths.image_dir.name == "images_256"
    assert paths.mask_dir.name == "masks_256"


def test_infer_kvasir_paths_falls_back_to_any_processed_pair(tmp_path: Path):
    (tmp_path / "processed" / "images_512").mkdir(parents=True)
    (tmp_path / "processed" / "masks_512").mkdir(parents=True)

    paths = infer_kvasir_paths(tmp_path, image_size=352)
    assert paths.image_dir.name == "images_512"
    assert paths.mask_dir.name == "masks_512"


def test_make_splits_resolve_image_dir_prefers_requested_size(tmp_path: Path):
    (tmp_path / "processed" / "images_320").mkdir(parents=True)
    (tmp_path / "processed" / "images_352").mkdir(parents=True)
    resolved = make_splits._resolve_image_dir(tmp_path, image_size=352)
    assert resolved.name == "images_352"


def test_prepare_script_detects_prepared_dataset(tmp_path: Path):
    (tmp_path / "processed" / "images_352").mkdir(parents=True)
    (tmp_path / "processed" / "masks_352").mkdir(parents=True)
    assert prepare_kvasir_seg.prepared_dataset_exists(tmp_path, 352) is True
    assert prepare_kvasir_seg.prepared_dataset_exists(tmp_path, 256) is False


def test_benchmark_build_prepare_cmd_injects_default_download_url():
    args = Namespace(
        dataset="kvasir_seg",
        data_root="data",
        image_size=352,
        source_dir=None,
        zip_path=None,
        download_url=None,
        download_dst=None,
    )
    cmd = benchmark_all.build_prepare_cmd(args, py="python")
    spec = get_dataset_spec("kvasir_seg")
    assert "--download-url" in cmd
    assert spec.default_download_url in cmd


def test_benchmark_prepared_dataset_and_split_helpers(tmp_path: Path):
    assert benchmark_all.prepared_dataset_exists(tmp_path, 352) is False
    (tmp_path / "processed" / "images_352").mkdir(parents=True)
    (tmp_path / "processed" / "masks_352").mkdir(parents=True)
    assert benchmark_all.prepared_dataset_exists(tmp_path, 352) is True

    assert benchmark_all.split_files_exist(tmp_path) is False
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    for name in ("train", "val", "test"):
        (splits / f"{name}.txt").write_text("a\n", encoding="utf-8")
    assert benchmark_all.split_files_exist(tmp_path) is True
