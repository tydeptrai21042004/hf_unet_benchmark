from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import resize_to


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "gn":
        groups = 8 if channels >= 8 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm: {norm}")


def _make_act(act: str) -> nn.Module:
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {act}")


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm: str = "bn",
        act: str = "relu",
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            _make_norm(norm, out_channels),
            _make_act(act),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            ConvNormAct(in_channels, mid_channels, 3, norm=norm, act=act),
            ConvNormAct(mid_channels, out_channels, 3, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, in_channels, kernel_size, groups=in_channels, norm=norm, act=act),
            ConvNormAct(in_channels, out_channels, 1, padding=0, norm=norm, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn", act: str = "relu", bottleneck: bool = False) -> None:
        super().__init__()
        if bottleneck:
            hidden = max(out_channels // 4, 16)
            self.block = nn.Sequential(
                ConvNormAct(in_channels, hidden, 1, padding=0, norm=norm, act=act),
                ConvNormAct(hidden, hidden, 3, norm=norm, act=act),
                nn.Conv2d(hidden, out_channels, 1, bias=False),
                _make_norm(norm, out_channels),
            )
        else:
            self.block = nn.Sequential(
                ConvNormAct(in_channels, out_channels, 3, norm=norm, act=act),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                _make_norm(norm, out_channels),
            )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _make_norm(norm, out_channels),
        )
        self.act = _make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction=reduction)
        self.spatial = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.spatial(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18), norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(ConvNormAct(in_channels, out_channels, 1, padding=0, norm=norm, act=act))
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                        _make_norm(norm, out_channels),
                        _make_act(act),
                    )
                )
        self.project = ConvNormAct(out_channels * len(rates), out_channels, 1, padding=0, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: str = "double", norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        if block == "res":
            conv = ResidualBlock(in_channels, out_channels, norm=norm, act=act)
        elif block == "sep":
            conv = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, norm=norm, act=act),
                DepthwiseSeparableConv(out_channels, out_channels, norm=norm, act=act),
            )
        else:
            conv = DoubleConv(in_channels, out_channels, norm=norm, act=act)
        self.block = nn.Sequential(nn.MaxPool2d(2), conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, mode: str = "bilinear", norm: str = "bn", act: str = "relu", use_cbam: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, norm=norm, act=act)
        self.attn = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = resize_to(x, skip, mode=self.mode)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)


class FusionBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        self.pre = ConvNormAct(channels, channels, 3, norm=norm, act=act)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        return y * self.gate(y) + x


class ReverseAttentionRefine(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, mid_channels: Optional[int] = None, norm: str = "bn", act: str = "relu") -> None:
        super().__init__()
        mid_channels = mid_channels or max(in_channels // 2, 16)
        self.conv = nn.Sequential(
            ConvNormAct(in_channels, mid_channels, 3, norm=norm, act=act),
            ConvNormAct(mid_channels, mid_channels, 3, norm=norm, act=act),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse = resize_to(coarse_logits, feat)
        reverse_mask = 1.0 - torch.sigmoid(coarse)
        refined = self.conv(feat * reverse_mask)
        return refined + coarse


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.proj(out) + x


__all__ = [
    "ConvNormAct",
    "DoubleConv",
    "DepthwiseSeparableConv",
    "ResidualBlock",
    "SqueezeExcitation",
    "SpatialGate",
    "CBAM",
    "ASPP",
    "DownBlock",
    "UpBlock",
    "FusionBlock",
    "ReverseAttentionRefine",
    "SelfAttention2d",
]
