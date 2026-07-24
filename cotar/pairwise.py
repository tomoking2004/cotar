from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
import torch.nn.functional as F

__all__ = ["pairwise_cosine", "pairwise_equal"]


def pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    # Autocast is switched off around the Gram matrix deliberately: it routes
    # matmul to bf16, which would undo the `.float()` below and return cosines
    # good to only ~3 decimal digits. Both callers need far finer — the
    # contrastive loss scales these by ~14 before a logsumexp, and `separation_d`
    # is a difference of cosine means that the arms are compared on.
    with torch.autocast(features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=-1)
        return z @ z.t()


def pairwise_equal(labels: Sequence[Hashable], device: torch.device) -> torch.Tensor:
    ids: dict[Hashable, int] = {}
    codes = torch.tensor([ids.setdefault(label, len(ids)) for label in labels], device=device)
    return codes[:, None] == codes[None, :]
