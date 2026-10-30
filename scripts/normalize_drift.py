"""Read the drift measurement against other yardsticks, question by question. No GPU.

`probe_drift.py` reports drift as a percentage of the image-swap control's mean and as
Cohen's d, and context.md §A.6 decides on the first at layer 16. Both rest on the control
being a fair scale in every arm, and the arms do not share one: the alignment loss
reshaped the space the vector sits in, so the proposal's control is about five times the
baseline's and its spread relative to its own mean is nearly twice as wide. Whether the
decision survives that is a question the per-run summaries cannot answer, because they
keep only means and standard deviations. This re-reads the distances `probe_drift.py`
saved per question, and asks it three ways:

- **What the control looks like.** Per run and layer: its mean, its coefficient of
  variation, its skew, the ratio of its 90th to its 10th percentile, and the share of its
  total that the top tenth of questions carry.
- **What Cohen's d is made of.** d is (mu_D - mu_C) / sigma_p, so in units of the control's
  mean it is -(1 - R) / (sigma_p / mu_C), with R the ratio over 100. The pooled variance of
  `cohens_d` weights each side by its count less one, and a family of k variants has k
  drift pairs per control distance, so the pooled variance is split into the control's
  share. Beside it are two forms that drop the count weighting — the two variances averaged
  with equal weight, and the control's standard deviation alone (Glass's delta) — and two
  overlaps that assume no shape at all: the probability that a drift falls below a control
  distance over every pairing of the two samples (`auc`), and the share of drift pairs that
  fall below their own question's control distance (`below_own`).
- **Whether the ratio's direction depends on the scale.** The median of the drift over the
  median of the control; the geometric mean over questions and variants of each drift over
  its own question's control (`per_question`); the mean ratio with the control's top one
  per cent of questions left out (`trimmed`); and the drift over the mean distance between
  different questions of the same signature, which each run's final evaluation measured on
  the unperturbed report set (`same_signature` — one minus the same-signature similarity,
  a scale built from no image at all).

None of these replaces the decision quantity, which was fixed before the run. They are
read beside it, and every one is compared across arms exactly as it is: seed-paired, with
the interval of `cotar.analysis.statistics`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cotar.analysis import (
    ANY,
    ARMS,
    analysis_path,
    eval_report,
    paired_difference,
    site,
    summarize,
)
from cotar.training import cohens_d
from cotar.utils import save_json

DRIFT_PATH = analysis_path("probe_drift.py").with_suffix(".pt")
OUT_PATH   = analysis_path(__file__)

CONTROL = "other_image"

# Left out of the trimmed ratio: the questions whose control distance is in the top 1%.
TRIM = 0.99


def control_shape(c: torch.Tensor) -> dict[str, float]:
    """One run's image-swap control at one layer, as a distribution."""
    c = c.double()
    mean, sd = c.mean().item(), c.std().item()
    p10, p90 = torch.quantile(c, torch.tensor([0.1, 0.9], dtype=torch.double)).tolist()
    top = c.sort(descending=True).values[: c.numel() // 10]
    return {
        "mean": mean,
        "cv": sd / mean,
        "skew": (((c - mean) / sd) ** 3).mean().item(),
        "p90_over_p10": p90 / p10,
        "top_tenth_share": top.sum().item() / c.sum().item(),
    }


def below_probability(d: torch.Tensor, c: torch.Tensor) -> float:
    """P(D < C) over every pairing of a drift with a control distance.

    Counted against the sorted control rather than read off joint ranks: a tie between a
    drift and a control distance is then never "below", whereas ranks break ties in
    whatever order the sort happens to leave them — which differs between machines.
    """
    above = c.numel() - torch.searchsorted(c.sort().values, d, right=True)
    return above.sum().item() / (c.numel() * d.numel())


def family_measures(
    stack: torch.Tensor, c: torch.Tensor, same_signature: float | None
) -> dict[str, float]:
    """Every reading of one family at one layer in one run.

    `stack` is `(k, N)`: the family's k variants, each a drift per question; `c` is the
    control per question. Percentages are on the same scale as `probe_drift.py`'s ratio.
    `same_signature` is the scale of that name, or `None` at a layer it was not measured.
    """
    stack, c = stack.double(), c.double()
    d = stack.flatten()
    mu_d, mu_c = d.mean().item(), c.mean().item()
    var_d, var_c = d.var().item(), c.var().item()
    weighted_d, weighted_c = (d.numel() - 1) * var_d, (c.numel() - 1) * var_c
    pooled = math.sqrt((weighted_d + weighted_c) / (d.numel() + c.numel() - 2))
    kept = c[c <= torch.quantile(c, TRIM)]
    measures = {
        "ratio": 100 * mu_d / mu_c,
        "cohens_d": cohens_d(d, c),
        "pooled_sd_over_control_mean": pooled / mu_c,
        "drift_sd_over_control_mean": math.sqrt(var_d) / mu_c,
        "control_share_of_pooled": weighted_c / (weighted_d + weighted_c),
        "d_equal_weight": (mu_d - mu_c) / math.sqrt((var_d + var_c) / 2),
        "glass_delta": (mu_d - mu_c) / math.sqrt(var_c),
        "auc": 100 * below_probability(d, c),
        "below_own": 100 * (stack < c).double().mean().item(),
        "median": 100 * d.median().item() / c.median().item(),
        "per_question": 100 * math.exp((stack / c).log().mean().item()),
        "trimmed": 100 * mu_d / kept.mean().item(),
    }
    if same_signature is not None:
        measures["same_signature"] = 100 * mu_d / same_signature
    return measures


def compared(
    by_run: Callable[[str, int], dict[str, float]], seeds: list[int]
) -> dict[str, Any]:
    """Every measure of `by_run` summarised per arm and differenced within each seed."""
    readings = {(arm, seed): by_run(arm, seed) for arm in ARMS for seed in seeds}
    reference, *aligned = ARMS
    pairs = [(arm, reference) for arm in aligned] + [tuple(aligned)]
    out: dict[str, Any] = {}
    for measure in readings[ARMS[0], seeds[0]]:
        by_arm = {arm: {seed: readings[arm, seed][measure] for seed in seeds} for arm in ARMS}
        out[measure] = {
            "arms": {arm: summarize(values) for arm, values in by_arm.items()},
            "differences": {
                f"{left} - {right}": paired_difference(by_arm[left], by_arm[right])
                for left, right in pairs
            },
        }
    return out


def main() -> None:
    samples = torch.load(DRIFT_PATH, weights_only=False)
    layers: list[int] = samples["layers"]
    family_of: dict[str, str] = samples["families"]
    families = [*dict.fromkeys(family_of.values()), ANY]
    members = {
        family: [v for v in samples["variants"] if family == ANY or family_of[v] == family]
        for family in families
    }
    # Whichever seeds `probe_drift.py` measured: its runs are filed as `arm/seed`.
    seeds = sorted({int(key.split("/")[1]) for key in samples if "/" in key})
    # The same-signature similarity is measured at the site the loss touched, so this
    # scale exists for that layer only.
    site_layer, _ = site(seeds)
    same_signature = {
        (arm, seed): 1 - eval_report(arm, seed)["representation_stability"]["intra_sim"]
        for arm in ARMS for seed in seeds
    }

    def distances(arm: str, seed: int) -> dict[str, torch.Tensor]:
        return samples[f"{arm}/{seed}"]["distances"]

    def shape(i: int) -> Callable[[str, int], dict[str, float]]:
        return lambda arm, seed: control_shape(distances(arm, seed)[CONTROL][:, i])

    def measured(i: int, family: str) -> Callable[[str, int], dict[str, float]]:
        def by_run(arm: str, seed: int) -> dict[str, float]:
            at = distances(arm, seed)
            stack = torch.stack([at[name][:, i] for name in members[family]])
            scale = same_signature[arm, seed] if layers[i] == site_layer else None
            return family_measures(stack, at[CONTROL][:, i], scale)
        return by_run

    result: dict[str, Any] = {"layers": layers, "controls": {}, "families": {}}
    for i, layer in enumerate(layers):
        result["controls"][str(layer)] = compared(shape(i), seeds)
        result["families"][str(layer)] = {
            family: compared(measured(i, family), seeds) for family in families
        }

    print_headline(result, site_layer)
    save_json(result, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")


def print_headline(result: dict[str, Any], layer: int) -> None:
    """The union of every family at one layer: each measure per arm, and the paired
    differences with their intervals."""
    print(f"every family pooled ({ANY}) at layer {layer}: mean over seeds, and the paired "
          f"difference with its 95% interval (* excludes 0)")
    for measure, entry in result["families"][str(layer)][ANY].items():
        row = f"{measure:>28}" + "".join(
            f" {entry['arms'][arm]['mean']:9.3f}" for arm in ARMS
        )
        for key, d in entry["differences"].items():
            row += (f" │ {key}: {d['mean']:+8.3f} [{d['ci_low']:+8.3f}, {d['ci_high']:+8.3f}]"
                    + ("*" if d["excludes_zero"] else " "))
        print(row)


if __name__ == "__main__":
    main()
