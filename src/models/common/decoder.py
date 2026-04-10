from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import UpBlock


class UNetDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], norm: str = "bn", act: str = "relu", use_cbam: bool = False) -> None:
        super().__init__()
        channels = tuple(channels)
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.blocks.append(
                UpBlock(channels[i], channels[i - 1], channels[i - 1], norm=norm, act=act, use_cbam=use_cbam)
            )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        skips = list(reversed(features[:-1]))
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x


__all__ = ["UNetDecoder"]
