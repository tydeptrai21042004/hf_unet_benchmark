from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss, parse_model_output
from src.losses import BCEDiceLoss, StructureLoss
from src.models import build_model

FAITHFUL_DIR = PROJECT_ROOT / "configs" / "faithful"

EXPECTED_AUX = {
    "unetpp": 3,
    "pranet": 3,
    "acsnet": 3,
    "polyp_pvt": 1,
    "caranet": 3,
    "cfanet": 4,
}


def load_cfg(name: str):
    with (FAITHFUL_DIR / f"{name}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_faithful_configs_build_and_expose_expected_output_contracts():
    x = torch.randn(1, 3, 64, 64)
    for model_name, expected_aux in EXPECTED_AUX.items():
        cfg = load_cfg(model_name)
        model = build_model(model_name, cfg["model"])
        model.eval()
        with torch.no_grad():
            output = model(x)
        parsed = parse_model_output(output)
        assert parsed.main.shape == (1, 1, 64, 64), model_name
        assert len(parsed.aux) == expected_aux, (model_name, len(parsed.aux), expected_aux)
        if model_name == "cfanet":
            assert parsed.boundary is not None
            assert parsed.boundary.shape == (1, 1, 64, 64)


def test_compute_supervised_loss_supports_auxiliary_and_boundary_terms():
    masks = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)
    output = {
        "main": torch.randn(2, 1, 32, 32),
        "aux": [torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)],
        "boundary": torch.randn(2, 1, 32, 32),
    }
    total_loss, log_items, parsed = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=BCEDiceLoss(),
        aux_loss_fn=BCEDiceLoss(),
        aux_weights=[0.5, 1.0],
        boundary_loss_fn=BCEDiceLoss(),
        boundary_weight=0.25,
    )
    assert torch.isfinite(total_loss)
    assert parsed.main.shape == masks.shape
    assert "aux_loss" in log_items
    assert "boundary_loss" in log_items


def test_structure_loss_can_train_faithful_multistage_models():
    cfg = load_cfg("pranet")
    model = build_model("pranet", cfg["model"])
    x = torch.randn(1, 3, 64, 64)
    masks = torch.randint(0, 2, (1, 1, 64, 64), dtype=torch.float32)
    output = model(x)
    total_loss, _, parsed = compute_supervised_loss(
        output,
        masks,
        main_loss_fn=StructureLoss(),
        aux_loss_fn=StructureLoss(),
        aux_weights=[1.0, 1.0, 1.0],
    )
    assert torch.isfinite(total_loss)
    assert len(parsed.aux) == 3
