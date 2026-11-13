"""Pairwise comparisons within a batch, shared by the alignment loss and the metrics."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
import torch.nn.functional as F

__all__ = ["pairwise_cosine", "pairwise_equal"]


def pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    """`(N, N)` cosine similarities between the rows of `features`, in float32.

    Autocast is switched off around the Gram matrix deliberately: it routes matmul to
    bf16, which would undo the `.float()` below and return cosines good to only two or
    three decimal digits. Both callers need far finer — the contrastive loss scales these
    by its logit scale (1/0.07, about 14, at the start) before a logsumexp, and
    `separation_d` is a difference of cosine means that the arms are compared on.
    """
    with torch.autocast(features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=-1)
        return z @ z.t()


def pairwise_equal(labels: Sequence[Hashable], device: torch.device) -> torch.Tensor:
    """`(N, N)` bool: whether rows `i` and `j` carry the same label."""
    ids: dict[Hashable, int] = {}
    codes = torch.tensor([ids.setdefault(label, len(ids)) for label in labels], device=device)
    return codes[:, None] == codes[None, :]
