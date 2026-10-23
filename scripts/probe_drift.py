"""Ask whether alignment made the representation steadier under image perturbation.

context.md §1.1 starts from Wani et al.'s observation that a VLM's hidden state can move far
while its answer stays put, and §8.1 is the question that observation leaves for this
study: when the image is perturbed in ways that preserve the answer, does the aligned
representation move less than the unaligned one, the same, or more? Nothing so far has
measured it — every analysis before this one read the representations of the unperturbed
report set.

This measures it the way the paper does (see `cotar.analysis.drift` and
`cotar.analysis.perturbations`): one pass over the report set per perturbed variant and
per control, cosine distance to the base pass, read as a percentage of the control's mean
and as Cohen's d, at the site the loss touched and at the final layer. Three arms at three
seeds, so the comparison is the document's usual paired one.

Set `LIMIT` to a few dozen to prove the wiring before committing the GPU to the full set:
twenty-three variants, a control and the base pass make twenty-five passes per run, and a
pass over the full report set takes about a minute and a half on the machine that trained
the runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from train4all.utils import empty_cuda_cache

from cotar.analysis import (
    ANY,
    ARMS,
    CHECKPOINT_NAME,
    CONFIDENCE,
    CONTROLS,
    FAMILIES,
    SEEDS,
    TIMESTAMP,
    Variant,
    analysis_path,
    measure_run,
    paired_difference,
    require_checkpoints,
    site,
    summarize,
    summarize_run,
    trained_model,
    wani_variants,
)
from cotar.config import cfg
from cotar.models import build_smolvlm
from cotar.utils import save_json

# `None` is the whole report set; a few dozen proves the wiring in minutes.
LIMIT: int | None = None

# The evaluation batch the runs themselves used.
BATCH = 32

# Which numbers the arms are compared on across seeds, per family and layer. The ratio
# to the image-swap control is the document's quantity; the rest are kept beside it.
COMPARED = (
    ("other_image", "ratio_mean"),
    ("other_image", "cohens_d"),
    ("other_item", "ratio_mean"),
    ("drift_mean",),
    ("unchanged", "ratio_mean"),
)
# The output-side rates, per family, compared the same way.
RATES = ("flip_rate", "vulnerability", "accuracy")

OUT_PATH     = analysis_path(__file__)
SAMPLES_PATH = OUT_PATH.with_suffix(".pt")

Results = dict[str, dict[str, dict[str, Any]]]
"""arm → seed → what `summarize_run` returned."""


def _value(entry: dict[str, Any], path: Sequence[str]) -> float:
    for key in path:
        entry = entry[key]
    return float(entry)  # type: ignore[arg-type]


def print_run(summary: dict[str, Any]) -> None:
    """One run's families, one row each: drift, its percentage of the image-swap control
    and Cohen's d at every layer, then the output-side rates."""
    layers = [str(layer) for layer in summary["layers"]]
    print(f"{'family':>14} {'pairs':>7}" + "".join(
        f" │ L{layer}: {'drift':>7} {'%ctrl':>7} {'d':>6}" for layer in layers
    ) + f" │ {'IFR':>6} {'IV':>6} {'acc':>6}")
    for family, entry in summary["families"].items():
        row = f"{family:>14} {entry['pairs']:>7,}"
        for layer in layers:
            at = entry["by_layer"][layer]
            row += (f" │      {at['drift_mean']:7.4f} {at['other_image']['ratio_mean']:6.1f}%"
                    f" {at['other_image']['cohens_d']:6.2f}")
        row += (f" │ {100 * entry['flip_rate']:5.1f}% {100 * entry['vulnerability']:5.1f}%"
                f" {100 * entry['accuracy']:5.1f}%")
        print(row)
    for control in CONTROLS:
        print(f"{control:>14} {'':>7}" + "".join(
            f" │      {summary['controls'][control][layer]['mean']:7.4f} {'(ctrl)':>7} {'':>6}"
            for layer in layers
        ))
    print(f"{'base accuracy':>14} {100 * summary['base_accuracy']:6.1f}% (exact match)\n")


def compare(results: Results, layers: Sequence[int]) -> dict[str, Any]:
    """The arms summarised over seeds and differenced within each seed, for every family,
    layer and compared quantity — context.md §4.3's arithmetic, through `statistics`."""
    reference, *aligned = ARMS
    pairs = [(arm, reference) for arm in aligned] + [tuple(aligned)]
    seeds = sorted(int(seed) for seed in results[reference])
    families = list(results[reference][str(seeds[0])]["families"])

    def compared(by_arm: dict[str, dict[int, float]]) -> dict[str, Any]:
        return {
            "arms": {arm: summarize(values) for arm, values in by_arm.items()},
            "differences": {
                f"{left} - {right}": paired_difference(by_arm[left], by_arm[right])
                for left, right in pairs
            },
        }

    def entry(arm: str, seed: int, family: str) -> dict[str, Any]:
        return results[arm][str(seed)]["families"][family]

    comparison: dict[str, Any] = {}
    for family in families:
        comparison[family] = {
            str(layer): {
                "/".join(path): compared({
                    arm: {
                        seed: _value(entry(arm, seed, family)["by_layer"][str(layer)], path)
                        for seed in seeds
                    }
                    for arm in ARMS
                })
                for path in COMPARED
            }
            for layer in layers
        }
        for rate in RATES:
            comparison[family][rate] = compared({
                arm: {seed: float(entry(arm, seed, family)[rate]) for seed in seeds}
                for arm in ARMS
            })
    return comparison


def print_comparison(comparison: dict[str, Any], layer: int) -> None:
    """The headline: drift as a percentage of the image-swap control at one layer, per arm,
    with the paired differences against baseline."""
    reference, *aligned = ARMS
    print(f"drift as % of the image-swap control at layer {layer}, mean over seeds, and the "
          f"paired difference with its {round(100 * CONFIDENCE)}% interval")
    print(f"{'family':>14}" + "".join(f" {arm:>10}" for arm in ARMS)
          + "".join(f" │ {f'{arm} − {reference}':>28}" for arm in aligned))
    for family, by_layer in comparison.items():
        at = by_layer[str(layer)]["other_image/ratio_mean"]
        row = f"{family:>14}" + "".join(f" {at['arms'][arm]['mean']:9.1f}%" for arm in ARMS)
        for arm in aligned:
            d = at["differences"][f"{arm} - {reference}"]
            row += (f" │ {d['mean']:+7.1f}pt [{d['ci_low']:+7.1f}, {d['ci_high']:+7.1f}]"
                    + ("*" if d["excludes_zero"] else " "))
        print(row)
    print("* the interval excludes zero\n")


def main(variants: Sequence[Variant]) -> None:
    require_checkpoints()
    site_layer, _ = site()
    _, processor = build_smolvlm("500M")
    print(f"{len(variants)} perturbed variants in {len(FAMILIES)} families, plus the base "
          f"pass and the image-swap control: {len(variants) + 2} passes per run")
    print(f"report set: {cfg.gqa.testdev_questions}"
          + (f", first {LIMIT} questions" if LIMIT is not None else "") + "\n")

    results: Results = {arm: {} for arm in ARMS}
    samples: dict[str, Any] = {}
    layers: tuple[int, ...] | None = None
    for arm in ARMS:
        for seed in SEEDS:
            print(f"[{arm} seed{seed}]")
            # The site the loss touched, and the final layer: the paper's hook point.
            vlm, _ = trained_model(arm, seed, (site_layer, -1), cfg.device)
            if layers is None:
                layers = vlm.layers
            elif vlm.layers != layers:
                raise SystemExit(f"{arm} seed{seed} reads layers {vlm.layers}, others {layers}.")
            run = measure_run(
                vlm, processor, cfg.gqa.testdev_questions, cfg.gqa.images, variants,
                batch_size=BATCH, num_workers=cfg.num_workers, limit=LIMIT, progress=print,
            )
            del vlm
            empty_cuda_cache()

            results[arm][str(seed)] = summarize_run(run, variants)
            samples[f"{arm}/{seed}"] = {
                "base_correct": run.base_correct,
                "distances": run.distances,
                "flipped": run.flipped,
                "correct": run.correct,
            }
            if "question_ids" not in samples:
                samples["question_ids"] = run.question_ids
                samples["image_ids"] = run.image_ids
            elif samples["question_ids"] != run.question_ids:
                raise SystemExit(f"{arm} seed{seed} measured other questions than the first run.")
            print()
            print_run(results[arm][str(seed)])

    if layers is None:
        raise SystemExit("no run was measured.")
    comparison = compare(results, layers)
    print_comparison(comparison, site_layer)

    save_json({
        "timestamp": TIMESTAMP,
        "checkpoint": CHECKPOINT_NAME,
        "layers": list(layers),
        "site_layer": site_layer,
        "limit": LIMIT,
        "batch_size": BATCH,
        "families": [*FAMILIES, ANY],
        "variants": [{"family": v.family, "name": v.name} for v in variants],
        "controls": list(CONTROLS),
        "runs": results,
        "comparison": comparison,
    }, OUT_PATH)
    torch.save(
        {
            "timestamp": TIMESTAMP,
            "layers": list(layers),
            "variants": [v.name for v in variants],
            "families": {v.name: v.family for v in variants},
            **samples,
        },
        SAMPLES_PATH,
    )
    print(f"written: {OUT_PATH}\nwritten: {SAMPLES_PATH}")


if __name__ == "__main__":
    main(wani_variants())
