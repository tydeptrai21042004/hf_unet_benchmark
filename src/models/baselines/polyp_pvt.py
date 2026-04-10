from __future__ import annotations

import torch
import torch.nn as nn

from ..common.blocks import ConvNormAct, SelfAttention2d, UpBlock
from ..common.utils import init_weights
from ..registry import register_model


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        self.proj = ConvNormAct(in_channels, out_channels, 3, stride=stride, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PVTStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, heads: int, norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        self.patch = PatchEmbed(in_channels, out_channels, stride=2, norm=norm, act=act)
        self.blocks = nn.Sequential(*[SelfAttention2d(out_channels, num_heads=heads) for _ in range(depth)])
        self.ffn = nn.Sequential(
            ConvNormAct(out_channels, out_channels, 3, norm=norm, act=act),
            ConvNormAct(out_channels, out_channels, 3, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = self.blocks(x)
        return self.ffn(x)


@register_model("polyp_pvt")
class PolypPVTLite(nn.Module):
    """Conv-attention hybrid inspired by Polyp-PVT, simplified for a single-codebase benchmark."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256), norm: str = "bn", act: str = "gelu") -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c0, 3, norm=norm, act=act),
            ConvNormAct(c0, c0, 3, norm=norm, act=act),
        )
        self.s1 = PVTStage(c0, c1, depth=1, heads=2, norm=norm, act=act)
        self.s2 = PVTStage(c1, c2, depth=2, heads=4, norm=norm, act=act)
        self.s3 = PVTStage(c2, c3, depth=2, heads=8, norm=norm, act=act)
        self.up2 = UpBlock(c3, c2, c2, norm=norm, act="relu", use_cbam=True)
        self.up1 = UpBlock(c2, c1, c1, norm=norm, act="relu", use_cbam=True)
        self.up0 = UpBlock(c1, c0, c0, norm=norm, act="relu", use_cbam=True)
        self.head = nn.Conv2d(c0, num_classes, 1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.s1(x0)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        d2 = self.up2(x3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)
        return self.head(d0)
