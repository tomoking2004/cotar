from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from .pairwise import pairwise_cosine, pairwise_equal

__all__ = ["supervised_contrastive_loss"]


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: Sequence[Hashable],
    logit_scale: float | torch.Tensor,
) -> torch.Tensor:
    if logit_scale <= 0:
        raise ValueError(f"logit_scale must be positive, got {logit_scale}.")
    if features.size(0) != len(labels):
        raise ValueError(f"features has {features.size(0)} rows but {len(labels)} labels.")

    logits = pairwise_cosine(features) * logit_scale
    logits.fill_diagonal_(float("-inf"))

    positives = pairwise_equal(labels, features.device).fill_diagonal_(False)
    pos_count = positives.sum(dim=1)
    has_positive = pos_count > 0
    if not has_positive.any():
        # Nothing to pull together. Return a zero that is still attached to the
        # graph, so the parameters this loss reaches keep receiving a gradient.
        return features.float().sum() * 0.0

    log_denom = torch.logsumexp(logits, dim=1)
    # Rows without a positive are dropped by `has_positive` below, but they must not
    # divide by zero first: the NaN that produces would flow back through the
    # division into every other row's gradient.
    pos_mean = torch.where(positives, logits, 0.0).sum(dim=1) / pos_count.clamp(min=1)
    return (log_denom - pos_mean)[has_positive].mean()
