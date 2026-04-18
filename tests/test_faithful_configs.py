from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAITHFUL_DIR = PROJECT_ROOT / "configs" / "faithful"
MODELS = [
    "unet",
    "unet_cbam",
    "unetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "proposal_hf_unet_lite",
    "proposal_hf_unet",
]


def load_cfg(name: str):
    with (FAITHFUL_DIR / f"{name}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_faithful_configs_share_dataset_and_optimizer_recipe():
    reference = load_cfg(MODELS[0])
    keys = [
        ("data", "image_size"),
        ("data", "batch_size"),
        ("data", "augmentation"),
        ("train", "epochs"),
        ("train", "optimizer"),
        ("train", "scheduler"),
        ("train", "t_max"),
        ("train", "eta_min"),
        ("train", "mixed_precision"),
        ("train", "grad_clip"),
        ("train", "seed"),
        ("train", "threshold"),
    ]
    for model_name in MODELS[1:]:
        cfg = load_cfg(model_name)
        for section, key in keys:
            assert cfg[section][key] == reference[section][key], (model_name, section, key)


def test_faithful_configs_enable_model_specific_supervision_when_needed():
    assert load_cfg("unetpp")["model"]["deep_supervision"] is True
    assert load_cfg("pranet")["train"]["loss"] == "structure"
    assert load_cfg("polyp_pvt")["train"]["aux_output_weights"] == [1.0]
    assert load_cfg("cfanet")["train"]["boundary_weight"] > 0.0
    assert load_cfg("hsnet")["model"]["faithful_output"] is True
    assert len(load_cfg("hsnet")["train"]["aux_output_weights"]) == 4
