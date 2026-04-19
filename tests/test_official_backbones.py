from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common.official_backbones import OfficialHarDNetEncoder, OfficialPVTv2Backbone, OfficialRes2NetEncoder


def test_official_res2net_encoder_outputs_requested_pyramid_channels():
    model = OfficialRes2NetEncoder(channels=(16, 32, 64, 128, 256), pretrained=False)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        feats = model(x)
    assert [t.shape[1] for t in feats] == [16, 32, 64, 128, 256]
    assert [t.shape[-2:] for t in feats] == [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]


def test_official_pvtv2_backbone_outputs_requested_channels():
    model = OfficialPVTv2Backbone(embed_dims=(16, 32, 64, 128), variant="pvt_v2_b0", pretrained=False, image_size=32)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        feats = model(x)
    assert [t.shape[1] for t in feats] == [16, 32, 64, 128]
    assert [t.shape[-2:] for t in feats] == [(8, 8), (4, 4), (2, 2), (1, 1)]


def test_official_hardnet_encoder_outputs_requested_channels():
    model = OfficialHarDNetEncoder(channels=(16, 32, 64, 128, 256), pretrained=False)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        feats = model(x)
    assert [t.shape[1] for t in feats] == [16, 32, 64, 128, 256]
    assert [t.shape[-2:] for t in feats] == [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
