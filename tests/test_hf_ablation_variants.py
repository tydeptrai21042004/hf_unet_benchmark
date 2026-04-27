from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
torch.set_num_threads(1)
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401  # side-effect registration
from src.models import build_model
from src.models.proposal.hf_ablation import ConvBottleneck, FFTGFNetLikeBottleneck, IdentityTransform2d
from src.models.proposal.hf_bottleneck import HFBottleneck
from src.models.registry import get_model_class

ABLATION_MODELS = [
    "unet",
    "unet_conv_bottleneck",
    "unet_fft_bottleneck",
    "proposal_hf_unet",
    "hf_unet_wo_hartley",
    "hf_unet_wo_fourier_kernel",
    "hf_unet_wo_residual",
    "hf_unet_encoder_stage4",
    "hf_unet_decoder_stage",
]


def _tiny_cfg(model_name: str) -> dict:
    if model_name == "unet":
        return {
            "in_channels": 3,
            "num_classes": 1,
            "channels": (1, 2, 4, 8, 16),
            "norm": "gn",
            "act": "relu",
        }
    return {
        "in_channels": 3,
        "num_classes": 1,
        "channels": (1, 2, 4, 8, 16),
        "norm": "gn",
        "act": "relu",
        "hf_block_norm": "gn",
        "hf_block_act": "gelu",
        "hf_expansion": 1.0,
        "hf_alpha": 0.25,
        "hf_alpha_warmup_epochs": 0,
        "use_gate": True,
        "use_hf_regularizer": True,
    }


@pytest.mark.parametrize("model_name", ABLATION_MODELS)
def test_compact_ablation_model_is_registered_and_outputs_correct_shape(model_name: str):
    get_model_class(model_name)
    model = build_model(model_name, config=_tiny_cfg(model_name)).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_compact_ablation_blocks_match_their_declared_purpose():
    conv = build_model("unet_conv_bottleneck", config=_tiny_cfg("unet_conv_bottleneck"))
    fft = build_model("unet_fft_bottleneck", config=_tiny_cfg("unet_fft_bottleneck"))
    no_hartley = build_model("hf_unet_wo_hartley", config=_tiny_cfg("hf_unet_wo_hartley"))
    no_kernel = build_model("hf_unet_wo_fourier_kernel", config=_tiny_cfg("hf_unet_wo_fourier_kernel"))
    no_residual = build_model("hf_unet_wo_residual", config=_tiny_cfg("hf_unet_wo_residual"))
    stage4 = build_model("hf_unet_encoder_stage4", config=_tiny_cfg("hf_unet_encoder_stage4"))
    decoder = build_model("hf_unet_decoder_stage", config=_tiny_cfg("hf_unet_decoder_stage"))

    assert isinstance(conv.block, ConvBottleneck)
    assert isinstance(fft.block, FFTGFNetLikeBottleneck)
    assert isinstance(no_hartley.block, HFBottleneck)
    assert isinstance(no_hartley.block.hartley, IdentityTransform2d)
    assert isinstance(no_kernel.block, HFBottleneck)
    assert isinstance(no_kernel.block.mixer, torch.nn.Identity)
    assert isinstance(no_residual.block, HFBottleneck)
    assert no_residual.ablation == "wo_residual"
    assert stage4.placement == "encoder_stage4"
    assert decoder.placement == "decoder_stage"


def test_ablation_configs_exist_and_build():
    cfg_dir = PROJECT_ROOT / "configs" / "ablation"
    for model_name in ABLATION_MODELS:
        cfg_path = cfg_dir / f"{model_name}.yaml"
        assert cfg_path.exists(), model_name
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == model_name
        model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
        # Use tiny channels for speed while preserving the model's ablation key.
        model_cfg.update(_tiny_cfg(model_name))
        build_model(model_name, config=model_cfg)


def test_hf_ablation_backward_smoke():
    model = build_model("hf_unet_wo_hartley", config=_tiny_cfg("hf_unet_wo_hartley"))
    model.train()
    x = torch.randn(2, 3, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
