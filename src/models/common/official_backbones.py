from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import warnings

import torch
import torch.nn as nn

from .utils import ensure_tuple_channels
from ..vendor.hardnet import HarDBlock  # re-exported for tests/contracts
from ..vendor.res2net_v1b import Res2Net, Bottle2neck, res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from ..vendor.pvt_v2_compat import pvt_v2_b0, pvt_v2_b0_fast, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_fast
from ..vendor.hardnet import HarDNet


def _load_state_dict(model: nn.Module, checkpoint: str | None = None, url: str | None = None, strict: bool = False) -> bool:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            warnings.warn(f"Checkpoint not found: {checkpoint}")
            return False
        state = torch.load(path, map_location="cpu")
    elif url:
        try:
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        except Exception as exc:  # pragma: no cover - network-dependent
            warnings.warn(f"Failed to download checkpoint from {url}: {exc}")
            return False
    else:
        return False
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=strict)
    return True


class _Projection(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OfficialRes2NetEncoder(nn.Module):
    """Official Res2Net backbone adapter with projected multi-scale outputs.

    Returns five features matching the benchmark contract: [H/2, H/4, H/8, H/16, H/32].
    """

    VARIANTS = {
        "res2net50_v1b_26w_4s": (lambda pretrained=False: res2net50_v1b_26w_4s(pretrained=pretrained), (64, 256, 512, 1024, 2048)),
        "res2net101_v1b_26w_4s": (lambda pretrained=False: res2net101_v1b_26w_4s(pretrained=pretrained), (64, 256, 512, 1024, 2048)),
        "res2net50_v1b_26w_4s_fast": (lambda pretrained=False: Res2Net(Bottle2neck, [1, 1, 1, 1], baseWidth=26, scale=4), (64, 256, 512, 1024, 2048)),
    }

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        variant: str = "res2net50_v1b_26w_4s",
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("OfficialRes2NetEncoder expects five output channel values.")
        if in_channels != 3:
            raise ValueError("OfficialRes2NetEncoder currently supports RGB input only.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported Res2Net variant: {variant}")
        ctor, raw_channels = self.VARIANTS[variant]
        self.backbone = ctor(pretrained=False)
        if checkpoint or checkpoint_url:
            _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
        elif pretrained:
            warnings.warn(
                "OfficialRes2NetEncoder pretrained=True was requested without a checkpoint path/url. "
                "Please provide model.backbone_checkpoint or model.backbone_checkpoint_url for deterministic faithful runs."
            )
        self.channels = channels
        self.raw_channels = raw_channels
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(raw_channels, channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        feats = [x0, x1, x2, x3, x4]
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


class OfficialPVTv2Backbone(nn.Module):
    """Official PVTv2 backbone adapter with projected stage outputs."""

    VARIANTS = {
        "pvt_v2_b0": (pvt_v2_b0, (32, 64, 160, 256)),
        "pvt_v2_b0_fast": (pvt_v2_b0_fast, (32, 64, 160, 256)),
        "pvt_v2_b1": (pvt_v2_b1, (64, 128, 320, 512)),
        "pvt_v2_b2": (pvt_v2_b2, (64, 128, 320, 512)),
        "pvt_v2_b2_fast": (pvt_v2_b2_fast, (64, 128, 320, 512)),
    }

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: Sequence[int] = (32, 64, 128, 256),
        variant: str = "pvt_v2_b2",
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
        image_size: int = 352,
    ) -> None:
        super().__init__()
        embed_dims = ensure_tuple_channels(embed_dims)
        if len(embed_dims) != 4:
            raise ValueError("OfficialPVTv2Backbone expects four output stage channel values.")
        if in_channels != 3:
            raise ValueError("OfficialPVTv2Backbone currently supports RGB input only.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported PVTv2 variant: {variant}")
        ctor, raw_channels = self.VARIANTS[variant]
        self.backbone = ctor(img_size=image_size, in_chans=in_channels)
        if checkpoint or checkpoint_url:
            _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
        elif pretrained:
            warnings.warn(
                "OfficialPVTv2Backbone pretrained=True was requested without a checkpoint path/url. "
                "Please provide model.pvt_checkpoint or model.pvt_checkpoint_url for deterministic faithful runs."
            )
        self.channels = tuple(int(c) for c in embed_dims)
        self.raw_channels = raw_channels
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(raw_channels, self.channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone.forward_features(x)
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


class OfficialHarDNetEncoder(nn.Module):
    """Official HarDNet-68 backbone adapter with projected outputs."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        arch: int = 68,
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("OfficialHarDNetEncoder expects five output channel values.")
        if in_channels != 3:
            raise ValueError("OfficialHarDNetEncoder currently supports RGB input only.")
        self.backbone = HarDNet(arch=arch, pretrained=False)
        if checkpoint or checkpoint_url:
            _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
        elif pretrained:
            warnings.warn(
                "OfficialHarDNetEncoder pretrained=True was requested without a checkpoint path/url. "
                "Please provide model.backbone_checkpoint or model.backbone_checkpoint_url for deterministic faithful runs."
            )
        self.raw_channels = (32, 128, 256, 640, 1024)
        self.channels = tuple(int(c) for c in channels)
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(self.raw_channels, self.channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base = self.backbone.base
        s0 = base[0](x)
        x = base[1](s0)
        x = base[2](x)
        x = base[3](x)
        s1 = base[4](x)
        x = base[5](s1)
        x = base[6](x)
        s2 = base[7](x)
        x = base[8](s2)
        x = base[9](x)
        x = base[10](x)
        x = base[11](x)
        s3 = base[12](x)
        x = base[13](s3)
        x = base[14](x)
        s4 = base[15](x)
        feats = [s0, s1, s2, s3, s4]
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


__all__ = [
    "OfficialRes2NetEncoder",
    "OfficialPVTv2Backbone",
    "OfficialHarDNetEncoder",
    "HarDBlock",
]
