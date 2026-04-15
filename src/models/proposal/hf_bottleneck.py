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
    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        dropout: float = 0.0,
        act: str = "identity",
    ) -> None:
        super().__init__()
        hidden = max(int(channels * expansion), channels)
        self.in_proj = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act_name = act.lower()
        if self.act_name == "identity":
            self.act = nn.Identity()
        elif self.act_name == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.act_name == "gelu":
            self.act = nn.GELU()
        elif self.act_name == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported mixer activation: {act}")
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.out_proj(x)
        return x


class HFBottleneck(nn.Module):
    """Hartley-Fourier bottleneck for real-valued feature maps.

    The bottleneck keeps the residual form explicit and starts close to identity,
    which makes the block easier to optimize without changing the underlying
    Hartley-Fourier interpretation.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_se: bool = False,
        use_gate: bool = True,
        norm: str = "bn",
        act: str = "relu",
        mixer_act: str = "identity",
        gate_init_bias: float = -2.0,
        identity_init: bool = True,
        gate_identity_target: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_alpha = float(alpha)
        self.alpha = float(alpha)
        self.use_gate = bool(use_gate)
        self.gate_identity_target = float(gate_identity_target)

        self.hartley = HartleyTransform2d()
        self.pre = ConvNormAct(channels, channels, 3, norm=norm, act=act)
        self.mixer = FrequencyMixer(channels, expansion=expansion, dropout=dropout, act=mixer_act)
        self.post = ConvNormAct(channels, channels, 3, norm=norm, act=act)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()
        self._last_terms: Optional[HFRegularizationTerms] = None

        if identity_init:
            self._apply_identity_friendly_init(gate_init_bias=gate_init_bias)

    def _apply_identity_friendly_init(self, gate_init_bias: float) -> None:
        nn.init.kaiming_normal_(self.mixer.in_proj.weight, nonlinearity="linear")
        nn.init.normal_(self.mixer.out_proj.weight, mean=0.0, std=1.0e-3)
        if self.use_gate:
            gate_conv = self.gate[1]
            nn.init.zeros_(gate_conv.weight)
            if gate_conv.bias is not None:
                nn.init.constant_(gate_conv.bias, gate_init_bias)

    def regularization_terms(self) -> Optional[HFRegularizationTerms]:
        return self._last_terms

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pre(x)
        freq = self.hartley(x)
        mixed = self.mixer(freq)
        restored = self.hartley(mixed)
        restored = self.post(restored)
        restored = self.se(restored)
        if self.use_gate:
            gate = self.gate(restored)
        else:
            gate = torch.ones_like(restored)
        out = identity + self.alpha * gate * restored

        smoothness = (mixed[:, :, 1:, :] - mixed[:, :, :-1, :]).abs().mean() + (mixed[:, :, :, 1:] - mixed[:, :, :, :-1]).abs().mean()
        if self.use_gate:
            near_identity = (gate.mean() - self.gate_identity_target).abs()
        else:
            near_identity = restored.new_zeros(())
        energy_penalty = (restored.pow(2).mean() - identity.pow(2).mean()).abs()
        self._last_terms = HFRegularizationTerms(
            spectral_smoothness=smoothness,
            near_identity=near_identity,
            energy_penalty=energy_penalty,
        )
        return out


__all__ = ["HFRegularizationTerms", "HartleyTransform2d", "HFBottleneck", "FrequencyMixer"]
