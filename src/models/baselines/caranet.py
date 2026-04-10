from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import ASPP, CBAM, ReverseAttentionRefine
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights, resize_to
from ..registry import register_model


class ContextAwareFusion(nn.Module):
    def __init__(self, channels: tuple[int, int, int], out_channels: int = 64, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        c2, c3, c4 = channels
        self.l2 = nn.Conv2d(c2, out_channels, 1)
        self.l3 = nn.Conv2d(c3, out_channels, 1)
        self.l4 = nn.Conv2d(c4, out_channels, 1)
        self.aspp = ASPP(out_channels * 3, out_channels, norm=norm, act=act)
        self.cbam = CBAM(out_channels)
        self.head = nn.Conv2d(out_channels, 1, 1)

    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        p2 = self.l2(x2)
        p3 = resize_to(self.l3(x3), p2)
        p4 = resize_to(self.l4(x4), p2)
        fused = self.aspp(torch.cat([p2, p3, p4], dim=1))
        fused = self.cbam(fused)
        return self.head(fused)


@register_model("caranet")
class CaraNetLite(nn.Module):
    """Simplified CaraNet-style model with context-aware fusion and reverse attention refinement."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256, 512), norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CaraNetLite currently supports binary segmentation only.")
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="res", norm=norm, act=act)
        self.caf = ContextAwareFusion((channels[2], channels[3], channels[4]), out_channels=64, norm=norm, act=act)
        self.ra4 = ReverseAttentionRefine(channels[4], out_channels=1, norm=norm, act=act)
        self.ra3 = ReverseAttentionRefine(channels[3], out_channels=1, norm=norm, act=act)
        self.ra2 = ReverseAttentionRefine(channels[2], out_channels=1, norm=norm, act=act)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        coarse = self.caf(x3, x4, x5)
        logit4 = self.ra4(x5, coarse)
        logit3 = self.ra3(x4, logit4)
        logit2 = self.ra2(x3, logit3)
        return resize_to(logit2, x)
