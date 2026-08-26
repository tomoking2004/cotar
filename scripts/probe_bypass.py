"""Ask whether the alignment was routed around, or merely turned out to be neutral.

context.md §5.3 reports that accuracy did not move. Two different worlds produce that
number. In one the aligned structure is part of how the answer gets made and simply costs
nothing; in the other the model satisfied the loss in directions the rest of the network
never reads, and the probe in §5.1 is reading a side channel. Removing the projection head
(§2.3) blocks the cheapest escape — the representation itself has to move, and it did —
but it does not block this one.

Breaking the site and watching the output is not the way to tell them apart: a knocked-out
layer gets compensated for downstream, so an unchanged output says nothing about whether
the site was used (McGrath et al.). What this asks instead is where the readable structure
*sits*. The answer vocabulary is closed, so the rows of the output layer that correspond to
it span the directions that decide which answer wins. Project the representation onto that
span and onto its complement, probe each, and compare against a random subspace of the same
width: if the signature is concentrated where the output layer reads, it was not routed
around.

The rows are centred before the span is taken. An uncentred span is dominated by whatever
all the answer rows share, and a component common to every answer shifts every logit alike
— it cannot decide between answers, which is the only thing "the output layer reads it"
can usefully mean here.

Two properties make the result legible without a second opinion. The full-space fit is run
alongside the projections, so the numbers can be checked against the ones §5.1 already
reports — a mismatch means the checkpoint and the saved representations are not from the
same run. And the width `m` is swept, because a conclusion that survives only at one width
is a property of that width.

This is the first of the two stages in context.md §7.1. It answers "is the signature in the
directions the output layer reads", not "did this arm come to depend on the site less than
that one" — the latter needs the propagation from the site to the final layer, and is only
worth its cost if this stage comes back undecided.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cotar.analysis import (
    ARMS,
    CHECKPOINT_NAME,
    MIN_COUNT,
    MODEL,
    SEEDS,
    TIMESTAMP,
    WHERE,
    Arm,
    analysis_path,
    answer_token_rows,
    gain,
    keep_frequent,
    majority_floor,
    output_basis,
    probe_accuracy,
    random_basis,
    reported_gain,
    representations,
    require_checkpoints,
    scorable,
    scores,
    split_mask,
    splitter,
    summarize_places,
    weights,
)
from cotar.config import cfg
from cotar.utils import load_json, save_json

# Widths to sweep. Bounded above by the answer vocabulary's own rank, which is one less
# than the number of distinct first tokens, so 128 stays comfortably inside it.
DIMS = (8, 32, 128)

BASIS_SEED = 0

OUT_PATH = analysis_path(__file__)


def answer_span(
    arm: Arm, seed: int, answer_rows: torch.Tensor, m: int
) -> tuple[torch.Tensor, str]:
    """One run's answer directions, and the layer they were read from.

    The checkpoint is opened and let go inside this call. It carries the whole trained
    model, and the fits that follow take minutes — nothing should hold two gigabytes of
    parameters alive through them when a single matrix is what was wanted.

    Taken once at the widest `m` and sliced for the narrower ones: the decomposition does
    not depend on how many of its directions are asked for, so the leading `m` columns are
    the same however they are obtained.
    """
    trained = weights(arm, seed, model=MODEL)
    return output_basis(trained.output_weight, answer_rows, m), trained.output_layer


if __name__ == "__main__":
    require_checkpoints()

    # The row order and the kept rows are the ones §5.1 probes, so its numbers are the
    # yardstick the full-space fit below is checked against.
    _, signatures, _ = representations(ARMS[0], SEEDS[0])
    keep, _, labels = keep_frequent(signatures)
    rows, sub_labels, sub_train, n_classes = scorable(labels, split_mask(len(keep), splitter()))

    testdev = load_json(cfg.gqa.testdev_questions)
    answer_rows = answer_token_rows({entry["answer"] for entry in testdev.values()}, MODEL)

    print(f"{int(rows.sum()):,} questions, {n_classes} signatures, "
          f"floor {majority_floor(sub_labels, sub_train):.1%}")
    print(f"{len(answer_rows):,} output-layer rows from the answer vocabulary\n")

    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            span, source = answer_span(arm, seed, answer_rows, max(DIMS))
            features, _, _ = representations(arm, seed)
            # Normalised once, exactly as §5.1 does, and not again after projecting: a
            # re-normalised residual would be rescaled to unit length however little of
            # the vector it holds, which is the very thing being measured.
            x = F.normalize(features[keep][rows].float(), dim=-1)

            entry: dict[str, Any] = {
                "output_layer": source,
                "full": probe_accuracy(x, sub_labels, sub_train, n_classes),
                "by_dim": {},
            }
            print(f"[{arm} seed{seed}] output layer: {source}")
            print(f"{f'full space ({x.size(1)})':>26}  {entry['full']:6.1%}"
                  f"   ← compare with §5.1")

            for m in DIMS:
                place = scores(
                    x,
                    span[:, :m],
                    random_basis(x.size(1), m, BASIS_SEED),
                    sub_labels, sub_train, n_classes,
                )
                entry["by_dim"][str(m)] = place
                print(f"{f'm = {m}':>26}  "
                      f"in {place['in_output_span']:6.1%} / {place['in_random_span']:6.1%}"
                      f"   outside {place['outside_output_span']:6.1%}"
                      f" / {place['outside_random_span']:6.1%}"
                      f"   (output / random)")
            results[arm][str(seed)] = entry
            print()

    # ── the comparison the verdict rests on ──────────────────────────────────
    # Not the level anywhere, and not the gain anywhere either: both survive whatever the
    # subspace is. What decides is the *difference* between the output layer's span and a
    # random span of the same width. If the alignment's gain is larger inside U than inside
    # a random subspace, the structure it built is where the answer is read; if it is larger
    # outside U than outside a random one, the structure sits where the output does not read.
    # If neither difference appears, this stage has not decided and §7.1's second stage runs.
    summary = summarize_places(results, DIMS, ARMS, SEEDS)
    print(f"{'':>10}" + "".join(f"{f'm={m}':>44}" for m in DIMS))
    print(f"{'':>10}" + "".join(
        f"{'in U':>11}{'in rand':>11}{'out U':>11}{'out rand':>11}" for _ in DIMS))
    for arm in ARMS:
        print(f"{arm:>10}" + "".join(
            f"{summary[arm][f'{where}_{m}']:>11.1%}" for m in DIMS for where in WHERE
        ))

    print("\ngain over baseline (proposal − baseline), against the random span of each width")
    for m in DIMS:
        cells = []
        for side in ("in", "outside"):
            u, r = (gain(summary, f"{side}_{kind}_span_{m}") for kind in ("output", "random"))
            cells.append(f"{side:>7}  U {u:+5.1f}pt  random {r:+5.1f}pt  Δ {u - r:+5.2f}pt")
        print(f"  m = {m:>3}:  " + "   ".join(cells))
    reported = reported_gain()
    against = (
        f"← §5.1 reports {reported:+.1f}pt" if reported is not None else "← compare with §5.1"
    )
    print(f"  full space:  {gain(summary, 'full'):+5.1f}pt   {against}")

    save_json({
        "timestamp": TIMESTAMP,
        "checkpoint": CHECKPOINT_NAME,
        "dims": list(DIMS),
        "answer_rows": len(answer_rows),
        "min_count": MIN_COUNT,
        "questions": int(rows.sum()),
        "classes": n_classes,
        "floor": majority_floor(sub_labels, sub_train),
        "runs": results,
        "means": summary,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
