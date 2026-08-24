"""Every run-level number the document states, out of the nine runs' own files. No GPU.

The probes ask new questions of the saved representations. This asks none: it reads what
the runs already recorded and does the arithmetic the document does — the mean over
seeds, the spread, and the paired interval between arms. Done here rather than by hand
because a number computed once in a text editor is a number nobody can check afterwards,
and the geometry and the losses are quoted in the document as freely as the probe scores
are.

Two references are computed beside the measurements, so that a reader does not have to
supply them:

- **the collapse value.** The alignment loss of a batch whose representations all point
  the same way is ``log(B - 1)``, whatever the temperature. An arm sitting at that value
  extracted nothing from its labels, which is the claim the shuffled arm rests on.
- **the epoch's batch count.** The sampler draws by signature, so what an epoch came to
  is a fact about the run rather than a division of the dataset size.
"""

from __future__ import annotations

import math
from datetime import timedelta

from cotar.analysis import (
    ARMS,
    CONFIDENCE,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    constrained_layer,
    eval_report,
    paired_difference,
    reported_accuracy,
    run_config,
    run_id,
    summarize,
    training_batches,
    training_duration,
)
from cotar.utils import save_json

# What the arms are compared on, and where each is read from a run's report. `loss` is
# deliberately absent: it is the objective, and the aligned arms carry an extra term in
# it, so it is not the same quantity in all three.
GEOMETRY = ("separation_d", "intra_sim", "inter_sim", "separation")
LOSSES = ("lm_loss", "align_loss")

OUT_PATH = analysis_path(__file__)


def batch_size(arm: str, seed: int) -> int:
    """The batch the alignment loss saw, which sets the collapse value it is read against.

    Taken from the run's own config rather than defaulted: the collapse value is
    `log(B - 1)`, so a batch size assumed instead of read would move the line the
    shuffled arm is held to, and move it silently.
    """
    config = run_config(arm, seed)
    if "batch_size" not in config:
        raise SystemExit(
            f"{run_id(arm, seed)}: config.json records no batch_size, so the collapse "
            f"value log(B - 1) cannot be derived. It is recorded only when it differs "
            f"from the trainer default, which this run's did not."
        )
    return int(config["batch_size"])


def constrained_site() -> tuple[int, int]:
    """The layer the nine runs constrained, and the width of its vector.

    Read back from the runs rather than restated from the trainer's default, for the
    same reason the batch size is: the document names the layer in every claim it makes,
    and a default is what the code would do today, not what these runs did. Runs that
    disagree are refused — nine runs constraining different layers are not one
    experiment, and averaging them would hide that.
    """
    sites = {(arm, seed): constrained_layer(arm, seed) for arm in ARMS for seed in SEEDS}
    distinct = set(sites.values())
    if len(distinct) != 1:
        raise SystemExit(
            "the runs did not constrain the same site, so they are not one experiment: "
            + ", ".join(f"{run_id(arm, seed)} at layer {layer} ({width}-d)"
                        for (arm, seed), (layer, width) in sorted(sites.items()))
        )
    return distinct.pop()


def collapse_value(size: int) -> float:
    """What the alignment loss reads when every representation points the same way.

    With all cosines equal to 1 the log-sum-exp and the positive term differ by exactly
    ``log(B - 1)``: the temperature cancels, so this is a property of the batch size
    alone and an arm can be held to it without knowing what its temperature did.
    """
    return math.log(size - 1)


if __name__ == "__main__":
    runs = {(arm, seed): eval_report(arm, seed) for arm in ARMS for seed in SEEDS}
    for (arm, seed), report in runs.items():
        recorded = report["config"]
        if recorded["arm"] != arm or recorded["seed"] != seed:
            raise SystemExit(
                f"{arm} seed{seed}: the report says it is {recorded['arm']} "
                f"seed{recorded['seed']} — the runs are not filed under what they are."
            )

    measurements: dict[str, dict[str, dict[int, float]]] = {}
    for arm in ARMS:
        by_metric: dict[str, dict[int, float]] = {}
        for metric in GEOMETRY:
            by_metric[metric] = {
                seed: runs[arm, seed]["representation_stability"][metric] for seed in SEEDS
            }
        for metric in (*LOSSES, "accuracy"):
            values = {
                seed: (
                    reported_accuracy(arm, seed)
                    if metric == "accuracy"
                    else runs[arm, seed]["test_metrics"][metric]
                )
                for seed in SEEDS
                if metric != "align_loss" or "align_loss" in runs[arm, seed]["test_metrics"]
            }
            if values:
                by_metric[metric] = values
        measurements[arm] = by_metric

    summaries = {
        arm: {metric: summarize(values) for metric, values in by_metric.items()}
        for arm, by_metric in measurements.items()
    }

    # Every arm against every other, on the two quantities the arms are entitled to be
    # compared on. The geometry is left out on purpose: it is what the loss optimises
    # directly, so a difference in it confirms the plumbing rather than the hypothesis.
    #
    # Subtracted in the direction the document reads them — the other arms against
    # baseline, then the two aligned arms against each other. A difference filed under
    # the opposite sign would have to be negated by whoever quotes it, and that is a
    # step at which a sign gets lost.
    reference, *aligned = ARMS
    comparisons = {
        f"{left}_minus_{right}": {
            metric: paired_difference(measurements[left][metric], measurements[right][metric])
            for metric in ("accuracy", "lm_loss")
        }
        for left, right in [*((arm, reference) for arm in aligned), tuple(aligned)]
    }

    sizes = {batch_size(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(sizes) != 1:
        raise SystemExit(f"The runs used different batch sizes ({sorted(sizes)}); "
                         f"the collapse value is not one number across them.")
    size = sizes.pop()
    collapse = collapse_value(size)

    batches = {(arm, seed): training_batches(arm, seed) for arm in ARMS for seed in SEEDS}
    durations = {(arm, seed): training_duration(arm, seed) for arm in ARMS for seed in SEEDS}
    total = sum(durations.values(), timedelta())
    mean_duration = total / len(durations)

    # ── report ───────────────────────────────────────────────────────────────
    print(f"{len(runs)} runs under {TIMESTAMP}: {', '.join(ARMS)} × seeds "
          f"{', '.join(map(str, SEEDS))}\n")

    width = max(len(m) for by in measurements.values() for m in by)
    for arm in ARMS:
        print(arm)
        for metric, summary in summaries[arm].items():
            print(f"  {metric:>{width}}  {summary['mean']:9.4f} ± {summary['sd']:.4f}"
                  f"  (range {summary['range']:.4f})"
                  f"   {' / '.join(f'{v:.4f}' for v in summary['per_seed'].values())}")
        print()

    print(f"alignment loss when every representation points the same way: "
          f"log(B − 1) = log({size - 1}) = {collapse:.4f}")
    for arm in ARMS:
        if "align_loss" in summaries[arm]:
            reached = summaries[arm]["align_loss"]["mean"]
            print(f"  {arm:>9}  reached {reached:.4f}"
                  f"   ({collapse - reached:+.4f} against collapse)")

    print(f"\npaired differences, {CONFIDENCE:.0%} interval on three seeds")
    for pair, metrics in comparisons.items():
        for metric, d in metrics.items():
            mark = "excludes 0" if d["excludes_zero"] else "includes 0"
            print(f"  {pair:>19}  {metric:>9}  {d['mean']:+7.3f}"
                  f"  [{d['ci_low']:+7.3f}, {d['ci_high']:+7.3f}]  {mark}")

    layer, width = constrained_site()
    print(f"\nall {len(runs)} runs constrained layer {layer}, {width} dimensions wide")

    steps = sorted(set(batches.values()))
    print(f"\none epoch = {steps[0]:,} batches" if len(steps) == 1
          else f"\nepoch length differs across runs: {steps}")
    print(f"training took {mean_duration} on average, {total} over all {len(runs)} runs")

    analysis_json = {
        "timestamp": TIMESTAMP,
        "seeds": list(SEEDS),
        "batch_size": size,
        "constrained_layer": layer,
        "representation_width": width,
        "collapse_align_loss": collapse,
        "arms": summaries,
        "paired_differences": comparisons,
        "training": {
            "batches_per_epoch": steps[0] if len(steps) == 1 else steps,
            "mean_duration": str(mean_duration),
            "total_duration": str(total),
            "per_run_duration": {
                f"{arm}_seed{seed}": str(durations[arm, seed])
                for arm in ARMS
                for seed in SEEDS
            },
        },
    }
    save_json(analysis_json, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
