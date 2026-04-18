from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAPER_FAIR_DIR = PROJECT_ROOT / "configs" / "paper_fair"
MODELS = ["unet", "unet_cbam", "unetpp", "pranet", "acsnet", "hardnet_mseg", "polyp_pvt", "caranet", "cfanet", "hsnet", "proposal_hf_unet"]


def load_cfg(name: str):
    with (PAPER_FAIR_DIR / f"{name}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_paper_fair_configs_share_same_training_recipe():
    keys = [
        ("data", "image_size"),
        ("data", "batch_size"),
        ("data", "augmentation"),
        ("train", "epochs"),
        ("train", "lr"),
        ("train", "weight_decay"),
        ("train", "optimizer"),
        ("train", "scheduler"),
        ("train", "t_max"),
        ("train", "eta_min"),
        ("train", "mixed_precision"),
        ("train", "grad_clip"),
        ("train", "seed"),
        ("train", "loss"),
        ("train", "threshold"),
        ("train", "aux_loss_weight"),
        ("train", "aux_warmup_epochs"),
        ("train", "aux_ramp_epochs"),
        ("eval", "loss"),
        ("eval", "threshold"),
    ]
    reference = load_cfg(MODELS[0])
    for model_name in MODELS[1:]:
        cfg = load_cfg(model_name)
        for section, key in keys:
            assert cfg[section][key] == reference[section][key], (model_name, section, key, cfg[section][key], reference[section][key])


def test_paper_fair_proposal_disables_proposal_only_training_helpers():
    cfg = load_cfg("proposal_hf_unet")
    assert cfg["train"]["aux_loss_weight"] == 0.0
    assert cfg["train"]["aux_warmup_epochs"] == 0
    assert cfg["train"]["aux_ramp_epochs"] == 0
    assert cfg["model"]["hf_alpha_start"] == cfg["model"]["hf_alpha"]
    assert cfg["model"]["hf_alpha_warmup_epochs"] == 0
    assert cfg["model"]["use_hf_regularizer"] is False


def test_paper_fair_output_root_is_separate_from_default_runs():
    for model_name in MODELS:
        cfg = load_cfg(model_name)
        assert cfg["experiment"]["output_root"] == "outputs_paper_fair"


def test_paper_fair_scheduler_horizon_matches_epoch_budget():
    for model_name in MODELS:
        cfg = load_cfg(model_name)
        assert cfg["train"]["epochs"] == 30
        assert cfg["train"]["t_max"] == 30
