from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_all, benchmark_fair, benchmark_faithful, eval_all, train_all
from src.engine.output_utils import parse_model_output
from src.models import build_model


def test_build_hsnet_from_registry():
    model = build_model(
        "hsnet",
        config={
            "in_channels": 3,
            "num_classes": 1,
            "channels": [16, 32, 64, 128, 256],
            "transformer_channels": [16, 32, 64, 128],
            "decoder_channels": 32,
        },
    )
    assert model is not None


def test_hsnet_forward_shape_matches_binary_segmentation_contract():
    model = build_model(
        "hsnet",
        config={
            "in_channels": 3,
            "num_classes": 1,
            "channels": [16, 32, 64, 128, 256],
            "transformer_channels": [16, 32, 64, 128],
            "decoder_channels": 32,
            "faithful_output": True,
        },
    )
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    parsed = parse_model_output(y)
    assert parsed.main.shape == (2, 1, 128, 128)
    assert len(parsed.aux) == 4
    assert parsed.extras is not None
    assert "msp_weights" in parsed.extras


def test_cli_model_lists_include_hsnet():
    assert "hsnet" in train_all.DEFAULT_MODELS
    assert "hsnet" in eval_all.DEFAULT_MODELS
    assert "hsnet" in benchmark_fair.MODELS
    assert "hsnet" in benchmark_faithful.MODELS
    assert "hsnet" in benchmark_all.DEFAULT_MODELS.split(",")
