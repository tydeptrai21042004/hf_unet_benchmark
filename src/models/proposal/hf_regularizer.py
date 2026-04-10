from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .hf_bottleneck import HFRegularizationTerms, HFBottleneck


class HFRegularizer(nn.Module):
    def __init__(
        self,
        lambda_smoothness: float = 1e-4,
        lambda_identity: float = 1e-4,
        lambda_energy: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_smoothness = lambda_smoothness
        self.lambda_identity = lambda_identity
        self.lambda_energy = lambda_energy

    def forward(self, terms: Optional[HFRegularizationTerms]) -> torch.Tensor:
        if terms is None:
            return torch.tensor(0.0)
        return (
            self.lambda_smoothness * terms.spectral_smoothness
            + self.lambda_identity * terms.near_identity
            + self.lambda_energy * terms.energy_penalty
        )

    def from_module(self, module: HFBottleneck) -> torch.Tensor:
        terms = module.regularization_terms()
        if terms is None:
            device = next(module.parameters()).device
            return torch.zeros((), device=device)
        return self(terms)


__all__ = ["HFRegularizer"]
