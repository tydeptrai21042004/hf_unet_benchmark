from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.trainer import Trainer
from src.losses import BCEDiceLoss
from src.models import build_model
from src.models.common.blocks import CBAM
from src.models.proposal.hf_bottleneck import (
    FrequencyMixer,
    HartleyTransform2d,
    HFBottleneck,
)

torch.set_num_threads(1)

CONFIG_DIR = PROJECT_ROOT / "configs"

# Prefer paper_fair. Fall back to fair if your repository still uses that name.
FAIR_CONFIG_DIR = CONFIG_DIR / "paper_fair"
if not FAIR_CONFIG_DIR.exists():
    FAIR_CONFIG_DIR = CONFIG_DIR / "fair"


def load_cfg(name: str, fair: bool = False):
    base = FAIR_CONFIG_DIR if fair else CONFIG_DIR
    with (base / f"{name}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_cfg(name: str, fair: bool = False):
    cfg = dict(load_cfg(name, fair=fair)["model"])
    cfg.pop("name", None)
    return cfg


def test_hartley_is_almost_involutive_on_real_inputs():
    ht = HartleyTransform2d()
    x = torch.randn(2, 8, 17, 19)
    y = ht(ht(x))
    assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)


def test_frequency_mixer_has_learnable_frequency_response():
    mixer = FrequencyMixer(
        channels=8,
        expansion=1.0,
        dropout=0.0,
        act="identity",
        init_hw=(7, 5),
    )

    assert hasattr(mixer, "response")
    assert mixer.response.ndim == 4
    assert mixer.response.shape[0] == 1
    assert mixer.response.shape[-2:] == (7, 5)
    assert mixer.response.requires_grad


@torch.no_grad()
def test_frequency_response_changes_output_by_frequency_location():
    mixer = FrequencyMixer(
        channels=1,
        expansion=1.0,
        dropout=0.0,
        act="identity",
        rank=1,
        init_hw=(8, 8),
    )

    mixer.in_proj.weight.fill_(1.0)
    mixer.out_proj.weight.fill_(1.0)

    x = torch.randn(1, 1, 8, 8)

    mixer.response.zero_()
    mixer.response[:, :, 0, 0] = 2.0
    y_low = mixer(x)

    mixer.response.zero_()
    mixer.response[:, :, 3, 4] = 2.0
    y_mid = mixer(x)

    assert not torch.allclose(y_low, y_mid)


@torch.no_grad()
def test_hf_bottleneck_frequency_response_is_nonlocal():
    block = HFBottleneck(
        channels=1,
        expansion=1.0,
        alpha=1.0,
        dropout=0.0,
        use_se=False,
        use_gate=False,
        norm="gn",
        act="relu",
        mixer_act="identity",
        identity_init=False,
        projection="identity",
        mixer_rank=1,
        mixer_init_hw=(16, 16),
    )

    block.mixer.in_proj.weight.fill_(1.0)
    block.mixer.out_proj.weight.fill_(1.0)
    block.mixer.response.zero_()

    # Activate a single non-DC frequency. This should create a global spatial
    # pattern after the inverse Hartley transform.
    block.mixer.response[:, :, 1, 2] = 4.0

    x = torch.zeros(1, 1, 16, 16)
    x[:, :, 8, 8] = 1.0

    y = block(x)

    hf_component = y - x

    local_mask = torch.zeros_like(hf_component, dtype=torch.bool)
    local_mask[:, :, 7:10, 7:10] = True
    nonlocal_energy = hf_component.masked_fill(local_mask, 0.0).abs().sum()

    assert nonlocal_energy.item() > 1e-4


def test_hf_bottleneck_preserves_shape_and_gradients():
    m = HFBottleneck(
        channels=32,
        expansion=1.5,
        alpha=0.5,
        dropout=0.0,
        norm="gn",
        act="relu",
        projection="linear",
    )

    x = torch.randn(2, 32, 32, 32, requires_grad=True)
    y = m(x)

    assert y.shape == x.shape

    y.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_auxiliary_regularization_is_scalar_and_finite():
    model = build_model("proposal_hf_unet", config=model_cfg("proposal_hf_unet"))

    x = torch.randn(1, 3, 128, 128)
    logits = model(x)
    reg = model.auxiliary_regularization()

    assert logits.shape == (1, 1, 128, 128)
    assert reg.ndim == 0
    assert torch.isfinite(reg)


@torch.no_grad()
def test_hf_bottleneck_uses_theory_faithful_projection_config():
    cfg = load_cfg("proposal_hf_unet")
    cfg["model"]["hf_projection"] = "linear"

    model = build_model("proposal_hf_unet", config=cfg["model"])

    pre = model.hf_bottleneck.pre
    post = model.hf_bottleneck.post

    assert isinstance(pre, nn.Conv2d)
    assert isinstance(post, nn.Conv2d)
    assert pre.kernel_size == (1, 1)
    assert post.kernel_size == (1, 1)


def test_build_model_ignores_name_inside_config_dict():
    cfg = load_cfg("unet")["model"]
    _ = build_model("unet", config=cfg)


@torch.no_grad()
def test_hf_and_unet_use_same_backbone_channels_in_fair_setup():
    hf_cfg = load_cfg("proposal_hf_unet", fair=True)["model"]
    unet_cfg = load_cfg("unet", fair=True)["model"]

    assert tuple(hf_cfg["channels"]) == tuple(unet_cfg["channels"])


@torch.no_grad()
def test_proposal_decoder_has_no_cbam_by_default_in_fair_setup():
    model = build_model("proposal_hf_unet", config=model_cfg("proposal_hf_unet", fair=True))

    assert not any(isinstance(m, CBAM) for m in model.decoder.modules())


@torch.no_grad()
def test_unet_cbam_control_contains_cbam_modules():
    model = build_model("unet_cbam", config=model_cfg("unet_cbam", fair=True))

    assert any(isinstance(m, CBAM) for m in model.decoder.modules())


@torch.no_grad()
def test_hf_alpha_warmup_changes_runtime_alpha():
    model = build_model("proposal_hf_unet", config=model_cfg("proposal_hf_unet", fair=True))

    model.set_epoch(0)
    alpha0 = model.hf_bottleneck.alpha

    model.set_epoch(5)
    alpha5 = model.hf_bottleneck.alpha

    model.set_epoch(10)
    alpha10 = model.hf_bottleneck.alpha

    assert alpha0 <= alpha5 <= alpha10
    assert abs(alpha10 - model.hf_alpha_target) < 1e-8


def test_aux_weight_schedule_warms_up_then_ramps():
    model = build_model("proposal_hf_unet", config=model_cfg("proposal_hf_unet", fair=True))

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn=BCEDiceLoss(),
        device="cpu",
        mixed_precision=False,
        aux_loss_weight=0.1,
        aux_warmup_epochs=10,
        aux_ramp_epochs=20,
    )

    assert trainer._current_aux_weight(1) == 0.0
    assert 0.0 < trainer._current_aux_weight(20) < 0.1
    assert abs(trainer._current_aux_weight(40) - 0.1) < 1e-8


@torch.no_grad()
def test_identity_friendly_init_keeps_first_forward_close_to_input():
    model = HFBottleneck(
        channels=16,
        alpha=0.5,
        norm="gn",
        act="relu",
        identity_init=True,
        projection="linear",
    )

    x = torch.randn(1, 16, 32, 32)
    y = model(x)

    rel = (y - x).abs().mean().item() / (x.abs().mean().item() + 1e-6)

    assert rel < 0.1


@torch.no_grad()
def test_all_models_produce_binary_logit_map_in_fair_setup():
    model_names = [
        "unet",
        "unet_cbam",
        "unetpp",
        "pranet",
        "acsnet",
        "hardnet_mseg",
        "polyp_pvt",
        "caranet",
        "proposal_hf_unet",
    ]

    for model_name in model_names:
        model = build_model(model_name, config=model_cfg(model_name, fair=True))

        x = torch.randn(1, 3, 128, 128)
        y = model(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape[0] == 1
        assert y.shape[1] == 1
        assert y.shape[-2:] == (128, 128)
        assert torch.isfinite(y).all()
