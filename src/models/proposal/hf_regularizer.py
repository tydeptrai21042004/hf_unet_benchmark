from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .hf_bottleneck import HFRegularizationTerms, HFBottleneck


class HFRegularizer(nn.Module):
    """Auxiliary regularizer for the HF bottleneck.

    The new terms are closer to the paper's stability interpretation:
      - response_smoothness controls variation of the learned M_K(p,q);
      - response_magnitude keeps the frequency response small;
      - stability_penalty discourages excessive residual amplification.
    """

    def __init__(
        self,
        lambda_smoothness: float = 1.0e-4,
        lambda_identity: float = 1.0e-4,
        lambda_energy: float = 1.0e-4,
        lambda_response_smoothness: float = 1.0e-4,
        lambda_response_magnitude: float = 1.0e-4,
        lambda_stability: float = 1.0e-4,
    ) -> None:
        super().__init__()

        self.lambda_smoothness = float(lambda_smoothness)
        self.lambda_identity = float(lambda_identity)
        self.lambda_energy = float(lambda_energy)
        self.lambda_response_smoothness = float(lambda_response_smoothness)
        self.lambda_response_magnitude = float(lambda_response_magnitude)
        self.lambda_stability = float(lambda_stability)

    def forward(self, terms: Optional[HFRegularizationTerms]) -> torch.Tensor:
        if terms is None:
            return torch.tensor(0.0)

        loss = (
            self.lambda_smoothness * terms.spectral_smoothness
            + self.lambda_identity * terms.near_identity
            + self.lambda_energy * terms.energy_penalty
        )

        if terms.response_smoothness is not None:
            loss = loss + self.lambda_response_smoothness * terms.response_smoothness

        if terms.response_magnitude is not None:
            loss = loss + self.lambda_response_magnitude * terms.response_magnitude

        if terms.stability_penalty is not None:
            loss = loss + self.lambda_stability * terms.stability_penalty

        return loss

    def from_module(self, module: HFBottleneck) -> torch.Tensor:
        terms = module.regularization_terms()

        if terms is None:
            device = next(module.parameters()).device
            return torch.zeros((), device=device)

        return self(terms)


__all__ = ["HFRegularizer"]
