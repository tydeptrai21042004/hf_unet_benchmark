from __future__ import annotations

from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model


CONFIG_DIR = PROJECT_ROOT / "configs"
FAITHFUL_DIR = CONFIG_DIR / "faithful"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_hf_unet_lite_is_parameter_matched_to_unet():
    with (CONFIG_DIR / "unet.yaml").open("r", encoding="utf-8") as f:
        unet_cfg = yaml.safe_load(f)
    with (FAITHFUL_DIR / "proposal_hf_unet_lite.yaml").open("r", encoding="utf-8") as f:
        lite_cfg = yaml.safe_load(f)

    unet = build_model("unet", unet_cfg["model"])
    lite = build_model("proposal_hf_unet_lite", lite_cfg["model"])

    delta = abs(count_params(unet) - count_params(lite))
    assert delta <= 50000, delta
