from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..common.blocks import ConvNormAct, SqueezeExcitation


@dataclass
class HFRegularizationTerms:
    spectral_smoothness: torch.Tensor
    near_identity: torch.Tensor
    energy_penalty: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.spectral_smoothness + self.near_identity + self.energy_penalty


class HartleyTransform2d(nn.Module):
    """2D discrete Hartley transform using the FFT identity.

    With orthonormal FFT normalization, applying the Hartley transform twice
    reconstructs the original tensor for real-valued inputs up to numerical error.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.fft.fft2(x, norm="ortho")
        return z.real - z.imag


class FrequencyMixer(nn.Module):
    def __init__(self, channels: int, expansion: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(int(channels * expansion), channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HFBottleneck(nn.Module):
    """Hartley-Fourier bottleneck for real-valued feature maps.

    Pipeline:
    1. local pre-projection
    2. Hartley transform per channel
    3. learnable frequency/channel mixing
    4. inverse Hartley transform
    5. gated residual fusion
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.hartley = HartleyTransform2d()
        self.pre = ConvNormAct(channels, channels, 3, norm="bn", act="gelu")
        self.mixer = FrequencyMixer(channels, expansion=expansion, dropout=dropout)
        self.post = ConvNormAct(channels, channels, 3, norm="bn", act="gelu")
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self._last_terms: Optional[HFRegularizationTerms] = None

    def regularization_terms(self) -> Optional[HFRegularizationTerms]:
        return self._last_terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pre(x)
        freq = self.hartley(x)
        mixed = self.mixer(freq)
        restored = self.hartley(mixed)
        restored = self.post(restored)
        restored = self.se(restored)
        gate = self.gate(restored)
        out = identity + self.alpha * gate * restored

        smoothness = (mixed[:, :, 1:, :] - mixed[:, :, :-1, :]).abs().mean() + (mixed[:, :, :, 1:] - mixed[:, :, :, :-1]).abs().mean()
        near_identity = (gate.mean() - 0.5).abs()
        energy_penalty = (restored.pow(2).mean() - identity.pow(2).mean()).abs()
        self._last_terms = HFRegularizationTerms(
            spectral_smoothness=smoothness,
            near_identity=near_identity,
            energy_penalty=energy_penalty,
        )
        return out


__all__ = ["HFRegularizationTerms", "HartleyTransform2d", "HFBottleneck"]
