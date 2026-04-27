from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..common.blocks import ASPP, ConvNormAct, ResidualBlock, SqueezeExcitation
from ..common.utils import ensure_tuple_channels, init_weights, resize_to
from ..registry import register_model


class ResUNetPPAttentionGate(nn.Module):
    """Attention gate used in the ResUNet++ decoder.

    The gate receives a low-level skip tensor and a high-level decoder tensor,
    projects both to an intermediate space, and produces a spatial attention mask
    for the skip tensor before concatenation with the upsampled decoder feature.
    """

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        inter_channels = max(int(inter_channels), 1)
        self.skip_proj = ConvNormAct(skip_channels, inter_channels, kernel_size=1, padding=0, norm=norm, act=act)
        self.gate_proj = ConvNormAct(gate_channels, inter_channels, kernel_size=1, padding=0, norm=norm, act=act)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True) if act.lower() == "relu" else nn.GELU()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = resize_to(gate, skip)
        attn = self.psi(self.act(self.skip_proj(skip) + self.gate_proj(gate)))
        return skip * attn


class ResUNetPPDecoderBlock(nn.Module):
    """Upsampling block with attention-gated skip fusion and residual refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.attention = ResUNetPPAttentionGate(
            skip_channels=skip_channels,
            gate_channels=in_channels,
            inter_channels=max(skip_channels // 2, 1),
            norm=norm,
            act=act,
        )
        self.fuse = ResidualBlock(in_channels + skip_channels, out_channels, norm=norm, act=act)
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = resize_to(x, skip)
        skip = self.attention(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.se(self.fuse(x))


@register_model("resunetpp")
class ResUNetPlusPlus(nn.Module):
    """ResUNet++ baseline for medical image / polyp segmentation.

    This implementation keeps the paper-level building blocks used by
    ResUNet++: residual convolution blocks, squeeze-and-excitation gates,
    an ASPP bridge, and attention-gated decoder skip fusion. The output is a
    single raw-logit segmentation map with the same spatial size as the input,
    matching the benchmark's common training/evaluation contract.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
        aspp_rates: Sequence[int] = (1, 6, 12, 18),
        **_: object,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("ResUNetPlusPlus expects five channel values, e.g. [32, 64, 128, 256, 512].")
        c0, c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ResidualBlock(in_channels, c0, norm=norm, act=act),
            SqueezeExcitation(c0),
        )
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(c0, c1, norm=norm, act=act),
            SqueezeExcitation(c1),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(c1, c2, norm=norm, act=act),
            SqueezeExcitation(c2),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(c2, c3, norm=norm, act=act),
            SqueezeExcitation(c3),
        )

        self.bridge = ASPP(c3, c4, rates=tuple(int(r) for r in aspp_rates), norm=norm, act=act)
        self.bridge_se = SqueezeExcitation(c4)

        self.dec3 = ResUNetPPDecoderBlock(c4, c3, c3, norm=norm, act=act)
        self.dec2 = ResUNetPPDecoderBlock(c3, c2, c2, norm=norm, act=act)
        self.dec1 = ResUNetPPDecoderBlock(c2, c1, c1, norm=norm, act=act)
        self.dec0 = ResUNetPPDecoderBlock(c1, c0, c0, norm=norm, act=act)

        self.out_aspp = ASPP(c0, c0, rates=(1, 3, 6), norm=norm, act=act)
        self.seg_head = nn.Conv2d(c0, num_classes, kernel_size=1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bridge_se(self.bridge(s3))
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        out = self.seg_head(self.out_aspp(d0))
        return resize_to(out, x)


__all__ = ["ResUNetPlusPlus", "ResUNetPPAttentionGate", "ResUNetPPDecoderBlock"]
