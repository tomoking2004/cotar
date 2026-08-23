from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn

__all__ = ["INIT_SCALE", "LogitScale"]

# The scale the contrastive term starts at, written as the reciprocal of the quantity
# it is set by: a starting temperature of 0.07. Defined here because this is what holds
# it; every other default reads it from here rather than restating the number.
INIT_SCALE: Final = 1 / 0.07


class LogitScale(nn.Module):
    """A learnable temperature, held as a log scale and clamped so a runaway scale
    cannot saturate the softmax.
    """

    def __init__(self, init_scale: float = INIT_SCALE, max_scale: float = 100.0) -> None:
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
