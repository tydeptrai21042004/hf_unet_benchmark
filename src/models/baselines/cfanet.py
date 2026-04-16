from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import ConvNormAct, SqueezeExcitation
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights, resize_to
from ..registry import register_model


class BoundaryPath(nn.Module):
    def __init__(self, c1: int, c2: int, out_channels: int = 32, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.p1 = ConvNormAct(c1, out_channels, 3, norm=norm, act=act)
        self.p2 = ConvNormAct(c2, out_channels, 3, norm=norm, act=act)
        self.gate = nn.Sequential(
            ConvNormAct(out_channels * 2, out_channels, 3, norm=norm, act=act),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvNormAct(out_channels * 2, out_channels, 3, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, 3, norm=norm, act=act),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        p1 = self.p1(x1)
        p2 = resize_to(self.p2(x2), p1)
        gated = p2 * self.gate(torch.cat([p1, p2], dim=1))
        return self.refine(torch.cat([p1, gated], dim=1))


class CrossLevelAggregation(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], out_channels: int = 64, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        c2, c3, c4, c5 = channels
        self.p2 = ConvNormAct(c2, out_channels, 1, padding=0, norm=norm, act=act)
        self.p3 = ConvNormAct(c3, out_channels, 1, padding=0, norm=norm, act=act)
        self.p4 = ConvNormAct(c4, out_channels, 1, padding=0, norm=norm, act=act)
        self.p5 = ConvNormAct(c5, out_channels, 1, padding=0, norm=norm, act=act)
        self.a4 = nn.Sequential(
            ConvNormAct(out_channels * 2, out_channels, 3, norm=norm, act=act),
            SqueezeExcitation(out_channels),
        )
        self.a3 = nn.Sequential(
            ConvNormAct(out_channels * 3, out_channels, 3, norm=norm, act=act),
            SqueezeExcitation(out_channels),
        )
        self.a2 = nn.Sequential(
            ConvNormAct(out_channels * 3, out_channels, 3, norm=norm, act=act),
            SqueezeExcitation(out_channels),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(out_channels * 4, out_channels, 3, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, 3, norm=norm, act=act),
        )

    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor) -> torch.Tensor:
        p2 = self.p2(x2)
        p3 = self.p3(x3)
        p4 = self.p4(x4)
        p5 = self.p5(x5)

        a4 = self.a4(torch.cat([p4, resize_to(p5, p4)], dim=1))
        a3 = self.a3(torch.cat([p3, resize_to(a4, p3), resize_to(p5, p3)], dim=1))
        a2 = self.a2(torch.cat([p2, resize_to(a3, p2), resize_to(a4, p2)], dim=1))
        return self.fuse(torch.cat([a2, resize_to(a3, a2), resize_to(a4, a2), resize_to(p5, a2)], dim=1))


@register_model("cfanet")
class CFANetLite(nn.Module):
    """Benchmark-friendly CFANet-style model.

    This is a simplified implementation that preserves the core benchmark-facing
    ideas used for comparison in this repository: cross-level feature aggregation
    and a lightweight boundary-aware fusion path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        aggregation_channels: int = 64,
        boundary_channels: int = 32,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("CFANetLite expects exactly five encoder stages.")
        c1, c2, c3, c4, c5 = channels
        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="res", norm=norm, act=act)
        self.boundary = BoundaryPath(c1, c2, out_channels=boundary_channels, norm=norm, act=act)
        self.aggregation = CrossLevelAggregation((c2, c3, c4, c5), out_channels=aggregation_channels, norm=norm, act=act)
        self.decoder = nn.Sequential(
            ConvNormAct(aggregation_channels + boundary_channels, aggregation_channels, 3, norm=norm, act=act),
            ConvNormAct(aggregation_channels, max(aggregation_channels // 2, num_classes), 3, norm=norm, act=act),
        )
        self.head = nn.Conv2d(max(aggregation_channels // 2, num_classes), num_classes, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        boundary = self.boundary(x1, x2)
        context = self.aggregation(x2, x3, x4, x5)
        fused = self.decoder(torch.cat([resize_to(context, boundary), boundary], dim=1))
        return self.head(fused)
