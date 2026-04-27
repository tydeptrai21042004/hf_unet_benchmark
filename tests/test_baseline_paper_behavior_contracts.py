from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss
from src.losses import BCEDiceLoss
from src.models.baselines.csca_unet import CSCAUNet, CSCASpatialAttention
from src.models.common.paper_baselines import (
    GlobalContextModule,
    LocalContextAttention,
    MultiScalePredictionModule,
    ReverseAttentionBranch,
)


def test_reverse_attention_branch_is_conditioned_by_coarse_prediction_map():
    """PraNet/CaraNet-style RA must not ignore its previous prediction guide."""
    torch.manual_seed(0)
    branch = ReverseAttentionBranch(in_channels=4, mid_channels=4, depth=2, kernel_size=3).eval()
    feat = torch.randn(1, 4, 8, 8)
    low_confidence_logits = torch.full((1, 1, 8, 8), -6.0)
    high_confidence_logits = torch.full((1, 1, 8, 8), 6.0)

    with torch.no_grad():
        out_low = branch(feat, low_confidence_logits)
        out_high = branch(feat, high_confidence_logits)

    assert out_low.shape == (1, 1, 8, 8)
    assert out_high.shape == out_low.shape
    assert torch.isfinite(out_low).all()
    assert torch.isfinite(out_high).all()
    assert not torch.allclose(out_low, out_high), "RA output must change when the guide map changes."


def test_acsnet_local_context_attention_emphasizes_uncertain_regions():
    """ACSNet LCA should focus more on uncertain prediction regions than confident ones."""
    lca = LocalContextAttention(channels=3).eval()
    lca.refine = nn.Identity()
    feat = torch.ones(1, 3, 4, 4)
    uncertain_logits = torch.zeros(1, 1, 4, 4)
    confident_logits = torch.full((1, 1, 4, 4), 10.0)

    with torch.no_grad():
        uncertain = lca(feat, uncertain_logits)
        confident = lca(feat, confident_logits)

    assert torch.all(uncertain > confident)
    assert torch.allclose(uncertain, torch.full_like(uncertain, 2.0), atol=1e-6)
    assert torch.all(confident < 1.001)


def test_acsnet_global_context_module_returns_one_feature_per_decoder_level():
    gcm = GlobalContextModule(in_channels=16, decoder_channels=(8, 4, 2, 1)).eval()
    x = torch.randn(1, 16, 4, 4)
    refs = [
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 2, 16, 16),
        torch.randn(1, 1, 32, 32),
    ]

    with torch.no_grad():
        outs = gcm(x, refs)

    assert len(outs) == len(refs)
    for out, ref in zip(outs, refs):
        assert out.shape[0] == ref.shape[0]
        assert out.shape[1] == ref.shape[1]
        assert out.shape[-2:] == ref.shape[-2:]
        assert torch.isfinite(out).all()


def test_csca_unet_faithful_output_has_main_plus_five_supervision_maps():
    model = CSCAUNet(
        in_channels=3,
        num_classes=1,
        channels=(2, 4, 8, 16, 32, 64),
        deep_supervision=True,
        faithful_output=True,
        attention_mode="efficient",
    ).eval()
    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out = model(x)

    assert set(out) == {"main", "aux"}
    assert out["main"].shape == (1, 1, 64, 64)
    assert len(out["aux"]) == 5
    assert all(aux.shape == out["main"].shape for aux in out["aux"])
    assert all(torch.isfinite(aux).all() for aux in out["aux"])


def test_csca_spatial_attention_paper_mode_runs_on_square_feature_maps():
    """The paper-style matmul branch is expensive but should still be valid on square crops."""
    attn = CSCASpatialAttention(channels=4, decay=2, attention_mode="paper").eval()
    low = torch.randn(1, 4, 4, 4)
    high = torch.randn(1, 4, 4, 4)

    with torch.no_grad():
        out = attn(low, high)

    assert out.shape == low.shape
    assert torch.isfinite(out).all()


def test_hsnet_multiscale_prediction_weights_are_normalized_and_trainable():
    msp = MultiScalePredictionModule(num_scales=3, refine=True)
    logits = [
        torch.randn(2, 1, 8, 8, requires_grad=True),
        torch.randn(2, 1, 16, 16, requires_grad=True),
        torch.randn(2, 1, 32, 32, requires_grad=True),
    ]

    fused, weights = msp(logits, target=(32, 32))
    loss = fused.mean()
    loss.backward()

    assert fused.shape == (2, 1, 32, 32)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(msp.weight_logits.grad).all()
    assert msp.weight_logits.grad.abs().sum() > 0


def test_unified_loss_supports_auxiliary_and_boundary_outputs_together():
    criterion = BCEDiceLoss()
    main = torch.zeros(2, 1, 16, 16, requires_grad=True)
    aux = [torch.zeros(2, 1, 16, 16, requires_grad=True) for _ in range(2)]
    boundary = torch.zeros(2, 1, 16, 16, requires_grad=True)
    masks = torch.zeros(2, 1, 16, 16)
    masks[:, :, 4:12, 5:11] = 1.0

    total, logs, parsed = compute_supervised_loss(
        {"main": main, "aux": aux, "boundary": boundary},
        masks,
        main_loss_fn=criterion,
        aux_loss_fn=criterion,
        aux_weights=[0.25, 0.5],
        boundary_loss_fn=criterion,
        boundary_weight=0.75,
    )
    total.backward()

    assert parsed.boundary is boundary
    assert len(parsed.aux) == 2
    assert logs["aux_weight"] == 0.375
    assert logs["boundary_weight"] == 0.75
    assert torch.isfinite(total)
    assert main.grad is not None and main.grad.abs().sum() > 0
    assert boundary.grad is not None and boundary.grad.abs().sum() > 0
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in aux)
