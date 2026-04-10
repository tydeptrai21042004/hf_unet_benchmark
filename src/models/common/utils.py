from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


def resize_to(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)


def upsample(x: torch.Tensor, scale_factor: int = 2, mode: str = "bilinear") -> torch.Tensor:
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters() if not trainable_only else (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def ensure_tuple_channels(channels: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(c) for c in channels)
    if len(out) < 2:
        raise ValueError("At least two channel values are required.")
    return out


def safe_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = torch.sigmoid(x)
    return y.clamp(min=eps, max=1.0 - eps)


__all__ = [
    "init_weights",
    "resize_to",
    "upsample",
    "count_parameters",
    "ensure_tuple_channels",
    "safe_sigmoid",
]
