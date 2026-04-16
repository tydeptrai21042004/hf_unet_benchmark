from __future__ import annotations

from typing import Optional

import torch


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve runtime device string.

    Rules
    -----
    - None, "", and "auto" select CUDA when available, otherwise CPU.
    - Explicit CUDA requests fall back to CPU when CUDA is unavailable.
    - Other device strings are returned unchanged.
    """

    if requested is None:
        requested = "auto"

    value = str(requested).strip()
    lowered = value.lower()
    if lowered in {"", "auto", "cuda_if_available"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if lowered.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return value


def should_pin_memory(device: Optional[str] = None) -> bool:
    return resolve_device(device).startswith("cuda")


__all__ = ["resolve_device", "should_pin_memory"]
