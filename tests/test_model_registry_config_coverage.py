from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401  # side-effect registration
from src.models.registry import get_model_class

BENCHMARK_MODELS = [
    "unet",
    "unet_cbam",
    "unetpp",
    "resunetpp",
    "pranet",
    "acsnet",
    "hardnet_mseg",
    "polyp_pvt",
    "caranet",
    "cfanet",
    "hsnet",
    "csca_unet",
    "proposal_hf_unet",
]


@pytest.mark.parametrize("config_dir", ["configs", "configs/fair", "configs/paper_fair", "configs/faithful"])
def test_every_benchmark_model_has_config_and_is_registered(config_dir: str):
    cfg_dir = PROJECT_ROOT / config_dir
    for model_name in BENCHMARK_MODELS:
        cfg_path = cfg_dir / f"{model_name}.yaml"
        assert cfg_path.exists(), (config_dir, model_name)

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == model_name
        model_cls = get_model_class(model_name)
        assert model_cls.__name__, model_name


def _literal_default_models_from_script(script_name: str):
    tree = ast.parse((PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_MODELS":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str):
                        return [item.strip() for item in value.split(",") if item.strip()]
                    return list(value)
    raise AssertionError(f"DEFAULT_MODELS not found in {script_name}")


@pytest.mark.parametrize("script_name", ["train_all.py", "eval_all.py", "benchmark_all.py"])
def test_default_benchmark_scripts_include_every_reported_model(script_name: str):
    defaults = _literal_default_models_from_script(script_name)
    missing = [model for model in BENCHMARK_MODELS if model not in defaults]
    assert not missing, (script_name, missing, defaults)


def test_official_faithful_csca_config_is_kept_separate_from_fast_fair_csca_config():
    official = yaml.safe_load((PROJECT_ROOT / "configs/official_faithful/csca_unet.yaml").read_text(encoding="utf-8"))
    paper_fair = yaml.safe_load((PROJECT_ROOT / "configs/paper_fair/csca_unet.yaml").read_text(encoding="utf-8"))

    assert official["model"]["attention_mode"] == "paper"
    assert official["model"]["deep_supervision"] is True
    assert official["model"]["faithful_output"] is True
    assert official["train"]["loss"] == "structure"

    assert paper_fair["model"]["attention_mode"] == "efficient"
    assert paper_fair["model"]["deep_supervision"] is False
    assert paper_fair["train"]["loss"] == "bce_dice"
