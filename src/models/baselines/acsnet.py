from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import ASPP, CBAM, UpBlock
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model


class AdaptiveContextSelection(nn.Module):
    def __init__(self, channels: int, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.aspp = ASPP(channels, channels, rates=(1, 3, 6, 9), norm=norm, act=act)
        self.attn = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.aspp(x))


@register_model("acsnet")
class ACSNetLite(nn.Module):
    """Simplified ACSNet-style benchmark model with adaptive context selection."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="res", norm=norm, act=act)
        self.context = AdaptiveContextSelection(channels[-1], norm=norm, act=act)
        self.up4 = UpBlock(channels[4], channels[3], channels[3], norm=norm, act=act, use_cbam=True)
        self.up3 = UpBlock(channels[3], channels[2], channels[2], norm=norm, act=act, use_cbam=True)
        self.up2 = UpBlock(channels[2], channels[1], channels[1], norm=norm, act=act, use_cbam=True)
        self.up1 = UpBlock(channels[1], channels[0], channels[0], norm=norm, act=act, use_cbam=True)
        self.head = nn.Conv2d(channels[0], num_classes, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x0, x1, x2, x3, x4 = feats
        x4 = self.context(x4)
        d3 = self.up4(x4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)
        return self.head(d0)
