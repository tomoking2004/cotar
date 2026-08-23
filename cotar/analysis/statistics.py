"""How the study turns three seeds into a number, defined once.

The research document fixes three decisions: an arm is summarised by the mean over the
seeds with an unbiased standard deviation beside it, arms are compared as differences
*within* a seed, and an interval is Student's t on those three differences. They live
here rather than in each script because two scripts that formed an interval differently
would put two meanings behind one bracket in the same document.
"""

from __future__ import annotations

import math
import statistics
from typing import TypedDict

__all__ = [
    "CONFIDENCE",
    "PairedDifference",
    "Summary",
    "paired_difference",
    "summarize",
    "t_critical",
]

CONFIDENCE = 0.95


class Summary(TypedDict):
    """One arm's value across the seeds.

    ``range`` is carried beside ``sd`` because the document quotes both: the spread of
    the unchanged arm is the yardstick a between-arm difference is held against, and it
    is read once as a standard deviation and once as the width actually observed.
    """

    mean: float
    sd: float
    range: float
    per_seed: dict[str, float]


class PairedDifference(TypedDict):
    """One arm minus another, paired seed by seed.

    ``excludes_zero`` is stored rather than left to each reader: whether an interval
    clears zero is the claim a document makes out of it, and recomputing that from a
    rounded bracket is how a claim drifts from its number.
    """

    mean: float
    sd: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    per_seed: dict[str, float]


def t_critical(df: int, confidence: float = CONFIDENCE) -> float:
    """The two-sided critical value of Student's t.

    Only two degrees of freedom are implemented, which is what three seeds give, and
    that case is exact in closed form: for a t with two degrees of freedom the
    distribution function is ``F(t) = ½ + t / (2√(2 + t²))``, which inverts to
    ``t = a√(2 / (1 − a²))`` with ``a = 2q − 1``. Computed rather than read off a table
    so that the interval in the document has a derivation instead of a citation.

    A different seed count raises, rather than quietly reusing the wrong quantile: the
    general case needs the inverse incomplete beta, and that is a dependency this
    project does not otherwise carry.
    """
    if df != 2:
        raise ValueError(
            f"Only df = 2 (three seeds) is implemented, got {df}. A different seed "
            f"count needs the general Student-t quantile."
        )
    a = 2 * (1 - (1 - confidence) / 2) - 1
    return a * math.sqrt(2 / (1 - a**2))


def summarize(by_seed: dict[int, float]) -> Summary:
    """An arm's mean and spread over the seeds, keeping what they were computed from."""
    values = [by_seed[seed] for seed in sorted(by_seed)]
    if len(values) < 2:
        raise ValueError(f"Need at least two seeds to report a spread, got {len(values)}.")
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "range": max(values) - min(values),
        "per_seed": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
    }


def paired_difference(
    left: dict[int, float], right: dict[int, float], confidence: float = CONFIDENCE
) -> PairedDifference:
    """``left − right``, differenced within each seed before anything is averaged.

    The seeds are the unit of pairing, so the two groups must have been run at the same
    ones; differencing across mismatched seeds would compare initialisations rather than
    arms, and silently.
    """
    if set(left) != set(right):
        raise ValueError(f"Seeds differ: {sorted(left)} against {sorted(right)}.")
    seeds = sorted(left)
    differences = [left[seed] - right[seed] for seed in seeds]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    margin = t_critical(len(differences) - 1, confidence) * sd / math.sqrt(len(differences))
    low, high = mean - margin, mean + margin
    return {
        "mean": mean,
        "sd": sd,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": low > 0 or high < 0,
        "per_seed": {str(seed): left[seed] - right[seed] for seed in seeds},
    }
