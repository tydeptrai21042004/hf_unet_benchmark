from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.models.common.official_backbones import (
    OfficialHarDNetEncoder,
    OfficialPVTv2Backbone,
    OfficialRes2NetEncoder,
)
from src.models.common.paper_baselines import (
    AdaptiveSelectionModule,
    AxialReverseAttention,
    BoundaryAggregationModule,
    BoundaryPredictionNetwork,
    CFPModule,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    CrossFeatureFusion,
    CrossSemanticAttention,
    DenseAggregation,
    GlobalContextModule,
    HybridSemanticComplementaryModule,
    LocalContextAttention,
    MultiScalePredictionModule,
    ReverseAttentionBranch,
    RFBModified,
    SimilarityAggregationModule,
)

CONFIG_DIR = PROJECT_ROOT / "configs"
FAITHFUL_DIR = CONFIG_DIR / "faithful"
OFFICIAL_FAITHFUL_DIR = CONFIG_DIR / "official_faithful"


def _load_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_from_default(name: str):
    cfg = _load_cfg(CONFIG_DIR / f"{name}.yaml")
    return build_model(name, cfg["model"])


def _count_modules(model, module_type: type) -> int:
    return sum(isinstance(m, module_type) for m in model.modules())


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("pranet", {RFBModified: 3, DenseAggregation: 1, ReverseAttentionBranch: 3, OfficialRes2NetEncoder: 1}),
        ("acsnet", {LocalContextAttention: 4, GlobalContextModule: 1, AdaptiveSelectionModule: 4, OfficialRes2NetEncoder: 1}),
        ("caranet", {CFPModule: 1, RFBModified: 3, DenseAggregation: 1, AxialReverseAttention: 3, OfficialRes2NetEncoder: 1}),
        ("hardnet_mseg", {RFBModified: 3, DenseAggregation: 1, OfficialHarDNetEncoder: 1}),
        ("polyp_pvt", {CascadedFusionModule: 1, CamouflageIdentificationModule: 1, SimilarityAggregationModule: 1, OfficialPVTv2Backbone: 1}),
        ("cfanet", {BoundaryPredictionNetwork: 1, CrossFeatureFusion: 7, BoundaryAggregationModule: 4}),
        ("hsnet", {CrossSemanticAttention: 4, HybridSemanticComplementaryModule: 4, MultiScalePredictionModule: 1, OfficialRes2NetEncoder: 1, OfficialPVTv2Backbone: 1}),
    ],
)
def test_default_models_match_paper_module_counts(model_name: str, expected: dict[type, int]):
    model = _build_from_default(model_name)
    for module_type, count in expected.items():
        assert _count_modules(model, module_type) == count, (model_name, module_type.__name__)


def test_hsnet_msp_emits_five_stage_logits_and_normalized_weights():
    cfg = _load_cfg(FAITHFUL_DIR / "hsnet.yaml")
    model = build_model("hsnet", cfg["model"])
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert "msp_weights" in output
    assert "stage_logits" in output
    assert len(output["stage_logits"]) == 5
    weights = output["msp_weights"]
    assert weights.ndim == 1
    assert weights.numel() == 5
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype), atol=1e-5)
    assert torch.all(weights >= 0)


def test_cfanet_faithful_output_keeps_boundary_branch_and_four_aux_predictions():
    cfg = _load_cfg(FAITHFUL_DIR / "cfanet.yaml")
    model = build_model("cfanet", cfg["model"])
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["main"].shape == (1, 1, 64, 64)
    assert output["boundary"].shape == (1, 1, 64, 64)
    assert len(output["aux"]) == 4
    for aux in output["aux"]:
        assert aux.shape == (1, 1, 64, 64)


@pytest.mark.parametrize(
    "name,backbone_key,url_key",
    [
        ("pranet", "backbone_impl", "backbone_checkpoint_url"),
        ("acsnet", "backbone_impl", "backbone_checkpoint_url"),
        ("caranet", "backbone_impl", "backbone_checkpoint_url"),
        ("hardnet_mseg", "backbone_impl", "backbone_checkpoint_url"),
        ("polyp_pvt", "backbone_impl", "backbone_checkpoint_url"),
        ("hsnet", "backbone_impl", None),
    ],
)
def test_official_faithful_configs_request_official_backbones(name: str, backbone_key: str, url_key: str | None):
    cfg = _load_cfg(OFFICIAL_FAITHFUL_DIR / f"{name}.yaml")
    model_cfg = cfg["model"]
    assert model_cfg[backbone_key] == "official"
    assert model_cfg.get("backbone_pretrained", False) is True
    if url_key is not None:
        assert model_cfg.get(url_key)
    if name == "hsnet":
        assert model_cfg.get("cnn_backbone_checkpoint_url")
        assert model_cfg.get("transformer_backbone_checkpoint_url")


@pytest.mark.parametrize(
    "name,main_loss",
    [
        ("pranet", "structure"),
        ("caranet", "structure"),
        ("hardnet_mseg", "structure"),
        ("polyp_pvt", "structure"),
        ("acsnet", "bce_dice"),
        ("hsnet", "bce_dice"),
    ],
)
def test_official_faithful_configs_preserve_model_specific_loss_choices(name: str, main_loss: str):
    cfg = _load_cfg(OFFICIAL_FAITHFUL_DIR / f"{name}.yaml")
    assert cfg["train"]["loss"] == main_loss
