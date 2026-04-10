from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import DoubleConv, DownBlock, ResidualBlock, DepthwiseSeparableConv
from .utils import ensure_tuple_channels


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (32, 64, 128, 256, 512), block: str = "double", norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        self.channels = channels

        if block == "res":
            stem = ResidualBlock(in_channels, channels[0], norm=norm, act=act)
        elif block == "sep":
            stem = nn.Sequential(
                DepthwiseSeparableConv(in_channels, channels[0], norm=norm, act=act),
                DepthwiseSeparableConv(channels[0], channels[0], norm=norm, act=act),
            )
        else:
            stem = DoubleConv(in_channels, channels[0], norm=norm, act=act)

        self.stem = stem
        self.downs = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1], block=block, norm=norm, act=act)
            for i in range(len(channels) - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = [self.stem(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        return feats


__all__ = ["PyramidEncoder"]
