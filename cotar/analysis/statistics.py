"""How the study turns its seeds into a number, defined once.

The research document fixes three decisions: an arm is summarised by the mean over the
seeds with an unbiased standard deviation beside it, arms are compared as differences
*within* a seed, and an interval is Student's t on those differences — three of them for
the main results, five for the robustness measurement. They live
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

    The point ``t`` with ``P(T ≤ t) = 1 − (1 − confidence) / 2`` for ``df`` degrees of
    freedom — 4.303 for the three seeds of the reported experiment (``df = 2``), 2.776 for
    five. Computed rather than read off a table so that the interval in the document has a
    derivation instead of a citation, and without a statistics dependency the project
    does not otherwise carry: for ``t ≥ 0`` the distribution function is
    ``F(t) = 1 − ½ I_x(df/2, ½)`` with ``x = df / (df + t²)``, where ``I`` is the
    regularised incomplete beta function, and ``F`` is inverted by bisection to machine
    precision. Two degrees of freedom have a closed form, ``t = a√(2 / (1 − a²))`` with
    ``a = 2q − 1``, which is used as it stands: the reported analyses were computed with
    it, and they reproduce to the last digit only if it is.
    """
    if df < 1:
        raise ValueError(f"Need at least one degree of freedom (two seeds), got {df}.")
    target = 1 - (1 - confidence) / 2
    if df == 2:
        a = 2 * target - 1
        return a * math.sqrt(2 / (1 - a**2))
    low, high = 0.0, 1.0
    while _t_cdf(high, df) < target:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        if _t_cdf(mid, df) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _t_cdf(t: float, df: int) -> float:
    """Student's t distribution function at ``t ≥ 0``."""
    return 1 - 0.5 * _incomplete_beta(df / (df + t * t), df / 2, 0.5)


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """The regularised incomplete beta function ``I_x(a, b)``, by its continued fraction.

    The fraction converges fast for ``x < (a + 1) / (a + b + 2)``; above that point the
    symmetry ``I_x(a, b) = 1 − I_{1−x}(b, a)`` brings the argument back below it.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _incomplete_beta(1 - x, b, a)
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    # Lentz's method for the continued fraction 1 / (1 + d1 / (1 + d2 / (1 + …))).
    tiny = 1e-300
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    fraction = d
    for m in range(1, 1000):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1 + numerator * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + numerator / c
            c = c if abs(c) > tiny else tiny
            fraction *= c * d
        if abs(c * d - 1) < 1e-15:
            break
    return math.exp(log_front) * fraction / a


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
