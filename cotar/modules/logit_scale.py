from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["LogitScale"]


class LogitScale(nn.Module):
    """A learnable temperature, held as a log scale and clamped so a runaway scale
    cannot saturate the softmax."""

    def __init__(self, init_scale: float = 1.0 / 0.07, max_scale: float = 100.0) -> None:
        super().__init__()
        if max_scale <= 0:
            raise ValueError(f"max_scale must be positive, got {max_scale}.")
        if not 0 < init_scale <= max_scale:
            raise ValueError(f"init_scale must be in (0, {max_scale}], got {init_scale}.")

        self._max_log_scale = math.log(max_scale)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self._max_log_scale).exp()

    @property
    def temperature(self) -> torch.Tensor:
        return self().reciprocal()
