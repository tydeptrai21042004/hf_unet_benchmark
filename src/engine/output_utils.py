from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class ParsedModelOutput:
    main: torch.Tensor
    aux: list[torch.Tensor]
    boundary: torch.Tensor | None = None
    extras: dict[str, Any] | None = None


def parse_model_output(output: Any) -> ParsedModelOutput:
    if torch.is_tensor(output):
        return ParsedModelOutput(main=output, aux=[])
    if isinstance(output, dict):
        main = output.get("main")
        if main is None or not torch.is_tensor(main):
            raise TypeError("Model output dict must contain a tensor under key 'main'.")
        aux = [t for t in output.get("aux", []) if torch.is_tensor(t)]
        boundary = output.get("boundary")
        if boundary is not None and not torch.is_tensor(boundary):
            raise TypeError("Model output key 'boundary' must be a tensor or None.")
        extras = {k: v for k, v in output.items() if k not in {"main", "aux", "boundary"}}
        return ParsedModelOutput(main=main, aux=aux, boundary=boundary, extras=extras)
    if isinstance(output, (list, tuple)):
        tensors = [t for t in output if torch.is_tensor(t)]
        if not tensors:
            raise TypeError("List/tuple model outputs must contain at least one tensor.")
        return ParsedModelOutput(main=tensors[-1], aux=tensors[:-1])
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def expand_aux_weights(num_aux: int, aux_weights: float | Sequence[float] | None) -> list[float]:
    if num_aux <= 0:
        return []
    if aux_weights is None:
        return [1.0] * num_aux
    if isinstance(aux_weights, (int, float)):
        return [float(aux_weights)] * num_aux
    weights = [float(w) for w in aux_weights]
    if not weights:
        return [1.0] * num_aux
    if len(weights) < num_aux:
        weights = weights + [weights[-1]] * (num_aux - len(weights))
    return weights[:num_aux]


def masks_to_boundaries(masks: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError("Expected masks with shape [B, C, H, W].")
    pad = kernel_size // 2
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    return boundary


def compute_supervised_loss(
    output: Any,
    masks: torch.Tensor,
    *,
    main_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    aux_weights: float | Sequence[float] | None = None,
    boundary_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    boundary_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float], ParsedModelOutput]:
    parsed = parse_model_output(output)
    main_loss = main_loss_fn(parsed.main, masks)
    total_loss = main_loss
    log_items: dict[str, float] = {
        "main_loss": float(main_loss.detach().item()),
        "seg_loss": float(main_loss.detach().item()),
    }

    aux_values: list[float] = []
    if parsed.aux:
        aux_loss_fn = aux_loss_fn or main_loss_fn
        weights = expand_aux_weights(len(parsed.aux), aux_weights)
        for aux_tensor, weight in zip(parsed.aux, weights):
            aux_loss = aux_loss_fn(aux_tensor, masks)
            total_loss = total_loss + float(weight) * aux_loss
            aux_values.append(float(aux_loss.detach().item()))
        if aux_values:
            log_items["aux_loss"] = float(sum(aux_values) / len(aux_values))
            log_items["aux_weight"] = float(sum(weights) / len(weights))

    if parsed.boundary is not None and boundary_loss_fn is not None and boundary_weight > 0.0:
        boundary_target = masks_to_boundaries(masks)
        boundary_loss = boundary_loss_fn(parsed.boundary, boundary_target)
        total_loss = total_loss + float(boundary_weight) * boundary_loss
        log_items["boundary_loss"] = float(boundary_loss.detach().item())
        log_items["boundary_weight"] = float(boundary_weight)

    log_items["loss"] = float(total_loss.detach().item())
    return total_loss, log_items, parsed


__all__ = [
    "ParsedModelOutput",
    "parse_model_output",
    "compute_supervised_loss",
    "expand_aux_weights",
    "masks_to_boundaries",
]
