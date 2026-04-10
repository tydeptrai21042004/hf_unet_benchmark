from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import DepthwiseSeparableConv, UpBlock, ConvNormAct
from ..common.utils import init_weights
from ..registry import register_model


class HarDBlockLite(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, layers: int = 4, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        current = in_channels
        outs = []
        for _ in range(layers):
            conv = DepthwiseSeparableConv(current, growth_rate, norm=norm, act=act)
            self.layers.append(conv)
            outs.append(growth_rate)
            current += growth_rate
        self.out_channels = in_channels + sum(outs)
        self.project = ConvNormAct(self.out_channels, self.out_channels, 1, padding=0, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            inp = torch.cat(feats, dim=1)
            feats.append(layer(inp))
        return self.project(torch.cat(feats, dim=1))


@register_model("hardnet_mseg")
class HarDNetMSEGLite(nn.Module):
    """Efficient segmentation model inspired by HarDNet-MSEG, simplified for unified benchmarking."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, base_channels, 3, stride=1, norm=norm, act=act),
            ConvNormAct(base_channels, base_channels, 3, stride=1, norm=norm, act=act),
        )
        self.pool = nn.MaxPool2d(2)
        self.b1 = HarDBlockLite(base_channels, base_channels // 2, norm=norm, act=act)
        self.b2 = HarDBlockLite(self.b1.out_channels, base_channels, norm=norm, act=act)
        self.b3 = HarDBlockLite(self.b2.out_channels, base_channels * 2 // 1, norm=norm, act=act)
        self.b4 = HarDBlockLite(self.b3.out_channels, base_channels * 2, norm=norm, act=act)

        c1, c2, c3, c4 = self.b1.out_channels, self.b2.out_channels, self.b3.out_channels, self.b4.out_channels
        self.up3 = UpBlock(c4, c3, c3, norm=norm, act=act)
        self.up2 = UpBlock(c3, c2, c2, norm=norm, act=act)
        self.up1 = UpBlock(c2, c1, c1, norm=norm, act=act)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(c1, base_channels, 3, norm=norm, act=act),
        )
        self.head = nn.Conv2d(base_channels, num_classes, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.b1(self.pool(s0))
        s2 = self.b2(self.pool(s1))
        s3 = self.b3(self.pool(s2))
        s4 = self.b4(self.pool(s3))
        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.final_up(d1)
        return self.head(d0)
