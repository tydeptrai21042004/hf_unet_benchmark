from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import ConvNormAct, SqueezeExcitation


@dataclass
class HFRegularizationTerms:
    spectral_smoothness: torch.Tensor
    near_identity: torch.Tensor
    energy_penalty: torch.Tensor

    # New terms closer to the paper's spectral-stability discussion.
    response_smoothness: Optional[torch.Tensor] = None
    response_magnitude: Optional[torch.Tensor] = None
    stability_penalty: Optional[torch.Tensor] = None

    @property
    def total(self) -> torch.Tensor:
        total = self.spectral_smoothness + self.near_identity + self.energy_penalty
        if self.response_smoothness is not None:
            total = total + self.response_smoothness
        if self.response_magnitude is not None:
            total = total + self.response_magnitude
        if self.stability_penalty is not None:
            total = total + self.stability_penalty
        return total


class HartleyTransform2d(nn.Module):
    """2D discrete Hartley transform using the FFT identity.

    For real input x,

        H(x) = Re(FFT2(x)) - Im(FFT2(x)).

    With orthonormal FFT normalization, H(H(x)) reconstructs x up to
    numerical precision.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        fft_input = x

        # FFT on CUDA does not always support fp16/bfloat16 robustly.
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            fft_input = x.float()

        z = torch.fft.fft2(fft_input, norm="ortho")
        y = z.real - z.imag

        if y.dtype != orig_dtype and orig_dtype in (torch.float16, torch.bfloat16):
            y = y.to(orig_dtype)

        return y


class FrequencyMixer(nn.Module):
    """Frequency-dependent Hartley-domain mixer.

    This replaces the old frequency-independent 1x1 mixer.

    The mixer implements a low-rank approximation of the paper's frequency-wise
    channel mixing matrix

        M_K(p, q) = W_out diag(1 + r(p, q)) W_in,

    where r(p, q) is a learnable frequency response.

    Input and output are real Hartley-domain tensors of shape [B, C, H, W].
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        dropout: float = 0.0,
        act: str = "identity",
        rank: Optional[int] = None,
        init_hw: Sequence[int] = (22, 22),
        init_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()

        if len(init_hw) != 2:
            raise ValueError("init_hw must contain exactly two integers: (H, W).")

        self.channels = int(channels)
        self.rank = int(rank) if rank is not None else max(int(channels * expansion), channels)
        self.init_hw = (int(init_hw[0]), int(init_hw[1]))

        self.in_proj = nn.Conv2d(self.channels, self.rank, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.rank, self.channels, kernel_size=1, bias=False)

        # Learnable frequency response r(p,q).
        # Shape: [1, rank, H0, W0]. It is interpolated if the runtime bottleneck
        # resolution differs from H0 x W0.
        self.response = nn.Parameter(torch.empty(1, self.rank, self.init_hw[0], self.init_hw[1]))

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

        self.reset_parameters(init_scale=init_scale)

    def reset_parameters(self, init_scale: float = 1.0e-3) -> None:
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity="linear")
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_scale)
        nn.init.normal_(self.response, mean=0.0, std=init_scale)

    def _response_for_size(self, h: int, w: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        r = self.response

        if r.shape[-2:] != (h, w):
            r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)

        return r.to(device=device, dtype=dtype)

    def response_regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.response

        if r.shape[-2] > 1:
            smooth_h = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().mean()
        else:
            smooth_h = r.new_zeros(())

        if r.shape[-1] > 1:
            smooth_w = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().mean()
        else:
            smooth_w = r.new_zeros(())

        response_smoothness = smooth_h + smooth_w
        response_magnitude = r.pow(2).mean()

        return response_smoothness, response_magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)

        # Frequency-dependent response. The "1 + r" form keeps the response
        # close to a stable identity-like modulation at initialization.
        r = self._response_for_size(z.shape[-2], z.shape[-1], dtype=z.dtype, device=z.device)
        z = z * (1.0 + r)

        # Keep identity for the paper-faithful linear core.
        # Non-identity activation can be used only for empirical variants.
        z = self.act(z)

        z = self.drop(z)
        z = self.out_proj(z)
        return z


class HFBottleneck(nn.Module):
    """Hartley-Fourier bottleneck for real-valued feature maps.

    Paper-faithful core:

        B_{K,alpha}(X) = X + alpha T_K X,

    where

        H(T_K X)(p,q) = M_K(p,q) H(X)(p,q).

    Practical options:
      - projection='identity': closest to the theory.
      - projection='linear': 1x1 linear pre/post projection.
      - projection='conv': legacy 3x3 ConvNormAct projection.
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
        projection: str = "linear",
        mixer_rank: Optional[int] = None,
        mixer_init_hw: Sequence[int] = (22, 22),
    ) -> None:
        super().__init__()

        self.channels = int(channels)
        self.base_alpha = float(alpha)
        self.alpha = float(alpha)
        self.use_gate = bool(use_gate)
        self.gate_identity_target = float(gate_identity_target)
        self.projection = projection.lower()

        self.hartley = HartleyTransform2d()

        self.pre = self._make_projection(
            channels=self.channels,
            mode=self.projection,
            norm=norm,
            act=act,
        )

        self.mixer = FrequencyMixer(
            channels=self.channels,
            expansion=expansion,
            dropout=dropout,
            act=mixer_act,
            rank=mixer_rank,
            init_hw=mixer_init_hw,
        )

        self.post = self._make_projection(
            channels=self.channels,
            mode=self.projection,
            norm=norm,
            act=act,
        )

        self.se = SqueezeExcitation(self.channels) if use_se else nn.Identity()

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()

        self._last_terms: Optional[HFRegularizationTerms] = None

        if identity_init:
            self._apply_identity_friendly_init(gate_init_bias=gate_init_bias)

    @staticmethod
    def _make_projection(channels: int, mode: str, norm: str, act: str) -> nn.Module:
        mode = mode.lower()

        if mode == "identity":
            return nn.Identity()

        if mode == "linear":
            return nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        if mode == "conv":
            # Legacy mode. Useful only for empirical variants.
            return ConvNormAct(channels, channels, kernel_size=3, norm=norm, act=act)

        raise ValueError(
            f"Unsupported HF projection mode: {mode}. "
            "Use one of: 'identity', 'linear', 'conv'."
        )

    @staticmethod
    def _init_projection_as_identity(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1) and module.in_channels == module.out_channels:
                nn.init.dirac_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_identity_friendly_init(self, gate_init_bias: float) -> None:
        # Make 1x1 projections start as identity if used.
        self._init_projection_as_identity(self.pre)
        self._init_projection_as_identity(self.post)

        # Keep the HF residual initially small.
        if isinstance(self.mixer, FrequencyMixer):
            self.mixer.reset_parameters(init_scale=1.0e-3)

        if self.use_gate:
            gate_conv = self.gate[1]
            nn.init.zeros_(gate_conv.weight)
            if gate_conv.bias is not None:
                nn.init.constant_(gate_conv.bias, gate_init_bias)

    def regularization_terms(self) -> Optional[HFRegularizationTerms]:
        return self._last_terms

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _mixer_response_terms(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.mixer, "response_regularization_terms"):
            response_smoothness, response_magnitude = self.mixer.response_regularization_terms()
            return response_smoothness, response_magnitude

        zero = torch.zeros((), device=device)
        return zero, zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x_projected = self.pre(x)

        # Hartley on signal.
        freq = self.hartley(x_projected)

        # Frequency-dependent channel mixing: M_K(p,q) H(X)(p,q).
        mixed = self.mixer(freq)

        # Inverse Hartley. Since H is involutive under orthonormal normalization,
        # applying H again reconstructs the spatial-domain signal.
        restored = self.hartley(mixed)

        restored = self.post(restored)
        restored = self.se(restored)

        if self.use_gate:
            gate = self.gate(restored)
        else:
            gate = torch.ones_like(restored)

        hf_residual = gate * restored
        out = identity + self.alpha * hf_residual

        # Feature-domain spectral smoothness.
        if mixed.ndim == 4 and mixed.shape[-2] > 1:
            smooth_h = (mixed[:, :, 1:, :] - mixed[:, :, :-1, :]).abs().mean()
        else:
            smooth_h = mixed.new_zeros(())

        if mixed.ndim == 4 and mixed.shape[-1] > 1:
            smooth_w = (mixed[:, :, :, 1:] - mixed[:, :, :, :-1]).abs().mean()
        else:
            smooth_w = mixed.new_zeros(())

        spectral_smoothness = smooth_h + smooth_w

        if self.use_gate:
            near_identity = (gate.mean() - self.gate_identity_target).abs()
        else:
            near_identity = restored.new_zeros(())

        eps = 1.0e-6
        identity_energy = identity.pow(2).mean().detach() + eps
        restored_energy = restored.pow(2).mean()

        # Energy-control penalty closer to a normalized stability measure.
        energy_penalty = (restored_energy / identity_energy - 1.0).abs()

        # Penalize overly large residual amplification.
        residual_energy = (self.alpha * hf_residual).pow(2).mean()
        stability_penalty = torch.relu(residual_energy / identity_energy - 1.0)

        response_smoothness, response_magnitude = self._mixer_response_terms(device=identity.device)

        self._last_terms = HFRegularizationTerms(
            spectral_smoothness=spectral_smoothness,
            near_identity=near_identity,
            energy_penalty=energy_penalty,
            response_smoothness=response_smoothness,
            response_magnitude=response_magnitude,
            stability_penalty=stability_penalty,
        )

        return out


__all__ = [
    "HFRegularizationTerms",
    "HartleyTransform2d",
    "FrequencyMixer",
    "HFBottleneck",
]
