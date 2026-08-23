"""The trade-off the study states at one point, read as a curve. No GPU.

The reported experiment fixes the alignment weight at 0.1 and concludes that the
signature becomes far more readable while the answers cost almost nothing. Both halves of
that sentence are single measurements, and a single measurement cannot say whether the
readability is bought cheaply or the cost simply had not started yet.

This puts the two quantities side by side at every weight that has been run: what the
representation carries, measured the way §5.1 measures it, against what the official
evaluator scores. Weight 0 and weight 0.1 come from the reported nine runs; the rest come
from `sweep.py`, under its own timestamp.

Runs at one seed carry no interval, and none is printed for them — the shape of the curve
is what one seed is for, and an interval drawn through single points would suggest a
precision that was not paid for.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F

from cotar.analysis import (
    SEEDS,
    TIMESTAMP,
    analysis_path,
    keep_frequent,
    paired_difference,
    probe_accuracy,
    reported_accuracy,
    representations,
    run_config,
    run_id,
    scorable,
    split_mask,
    splitter,
    summarize,
)
from cotar.config import cfg
from cotar.utils import save_json

# Filled in from what `sweep.py` prints. Empty until a sweep has been run, in which case
# only the two points the reported experiment already measured are shown.
SWEEP_TIMESTAMP = ""
SWEEP_SEEDS = (42,)

# The variants to read, as (variant name, arm). The empty variant is the reported
# experiment: its baseline is the alignment weight 0 point and its proposal the 0.1 one.
# The weight itself is never written here — it is read from each run's own config, so a
# point cannot be filed under a weight it was not trained with.
REPORTED = (("", "baseline"), ("", "proposal"))
SWEPT = (("lambda0.03", "proposal"), ("lambda0.3", "proposal"), ("lambda1.0", "proposal"))

OUT_PATH = analysis_path(__file__)


def point(
    variant: str, arm: str, seeds: tuple[int, ...], timestamp: str
) -> dict[str, Any]:
    """One configuration: what it was trained with, and what it came to.

    The probe is fitted per seed on that seed's representations, exactly as §5.1 fits it,
    and the split is drawn from the same fixed source — so a number here and a number
    there differ by the run and by nothing else.
    """
    weights = {run_config(arm, seed, variant, timestamp)["align_weight"] for seed in seeds}
    if len(weights) != 1:
        raise SystemExit(
            f"{variant or 'reported'} {arm}: seeds disagree about the alignment weight "
            f"({sorted(weights)}), so they are not one point on the curve."
        )

    probe, accuracy = {}, {}
    for seed in seeds:
        features, signatures, _ = representations(arm, seed, variant, timestamp)
        keep, _, labels = keep_frequent(signatures)
        rows, sub_labels, sub_train, n_classes = scorable(
            labels, split_mask(len(keep), splitter())
        )
        x = F.normalize(features[keep][rows].float(), dim=-1)
        probe[seed] = probe_accuracy(x, sub_labels, sub_train, n_classes)
        accuracy[seed] = reported_accuracy(arm, seed, variant, timestamp)

    # The constrained layer is deliberately absent: a run records it in its checkpoint
    # extras, and a snapshot excludes the checkpoints. For a swept run the variant name
    # is what says which layer it was, and for the reported runs the document does.
    return {
        "arm": arm,
        "variant": variant,
        "align_weight": weights.pop(),
        "seeds": list(seeds),
        "probe_accuracy": summarize(probe),
        "accuracy": summarize(accuracy),
        "_probe_per_seed": probe,
        "_accuracy_per_seed": accuracy,
    }


if __name__ == "__main__":
    if not cfg.gqa.testdev_questions.exists():
        raise SystemExit(f"testdev questions not found at {cfg.gqa.testdev_questions}.")

    wanted = [(variant, arm, SEEDS, TIMESTAMP) for variant, arm in REPORTED]
    if SWEEP_TIMESTAMP:
        wanted += [(variant, arm, SWEEP_SEEDS, SWEEP_TIMESTAMP) for variant, arm in SWEPT]
    else:
        print("SWEEP_TIMESTAMP is empty — showing only the two points the reported "
              "experiment measured.\nRun sweep.py, then put the timestamp it prints "
              "here.\n")

    missing = [
        run_id(arm, seed, variant, timestamp)
        for variant, arm, seeds, timestamp in wanted
        for seed in seeds
        if not (cfg.snapshots_root / run_id(arm, seed, variant, timestamp)).exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} snapshot(s) not found under {cfg.snapshots_root}:\n"
            + "\n".join(f"  {name}" for name in missing)
        )

    points = [point(variant, arm, seeds, ts) for variant, arm, seeds, ts in wanted]
    points.sort(key=lambda p: p["align_weight"])

    # Every point against the alignment-free one, which is the only reference the curve
    # has: "costs almost nothing" is a claim about the distance from weight 0.
    zero = next((p for p in points if p["align_weight"] == 0), None)

    print(f"{'weight':>8}  {'variant':>12}  {'probe':>8}  {'accuracy':>9}  "
          f"{'vs weight 0':>12}  seeds")
    for p in points:
        against = ""
        if zero is not None and p is not zero:
            shared = sorted(set(p["_accuracy_per_seed"]) & set(zero["_accuracy_per_seed"]))
            if len(shared) >= 2:
                d = paired_difference(
                    {s: p["_accuracy_per_seed"][s] for s in shared},
                    {s: zero["_accuracy_per_seed"][s] for s in shared},
                )
                p["accuracy_vs_zero"] = d
                against = f"{d['mean']:+.2f} {'*' if d['excludes_zero'] else ''}"
            elif shared:
                delta = p["_accuracy_per_seed"][shared[0]] - zero["_accuracy_per_seed"][shared[0]]
                p["accuracy_vs_zero"] = {"mean": delta, "seeds": shared}
                against = f"{delta:+.2f}"
        print(f"{p['align_weight']:>8}  {p['variant'] or p['arm']:>12}  "
              f"{p['probe_accuracy']['mean']:>7.1%}  {p['accuracy']['mean']:>9.2f}  "
              f"{against:>12}  {','.join(map(str, p['seeds']))}")

    if zero is not None and len(points) > 2:
        best = max(points, key=lambda p: p["probe_accuracy"]["mean"])
        print(f"\nprobe accuracy peaks at weight {best['align_weight']} "
              f"({best['probe_accuracy']['mean']:.1%}), "
              f"{best['probe_accuracy']['mean'] - zero['probe_accuracy']['mean']:+.1%} "
              f"over no alignment")
    print("\n* interval excludes 0. Points with one seed carry no interval.")

    save_json({
        "timestamp": TIMESTAMP,
        "sweep_timestamp": SWEEP_TIMESTAMP,
        "points": [{k: v for k, v in p.items() if not k.startswith("_")} for p in points],
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
