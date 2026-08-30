"""Ask whether the representation carries the procedure *compositionally*. No training run.

The signature probe has a standing objection: the alignment loss maximises how separable
the signatures are, and the probe measures how separable they are, so a high score could
be the loss admiring itself. This probe removes that. The loss is told only "same
signature or not" — never which operators a signature is built from, and never that two
signatures share any. So:

- the label is the **set of operators**, not the signature, and
- the split is **signature-disjoint**: every signature in the test half is one the probe
  never saw, assembled from operators it did.

The target drops what a signature keeps — order, repetition, and the operators too rare to
score — so a test signature can share its target vector with a training one
(`select > verify rel` and `select > filter depth > verify rel` do). `target_unseen` counts
the test half that does not, and the scores are over the whole test half.

Recovering the operators of an unseen signature is therefore something the loss never
asked for. If the aligned representation does it better, the gain is structure, not the
loss reading back its own objective.

The same wording control as the signature probe applies, since GQA's questions are
generated from the programs.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F

from cotar.analysis import (
    ARMS,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    fit_linear,
    representations,
    split_mask,
    splitter,
    surface_matrices,
)
from cotar.config import cfg
from cotar.utils import load_json, save_json

MIN_POSITIVE = 50      # an operator needs enough questions on both sides to be scored
OUT_PATH     = analysis_path(__file__)


def auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """Area under the ROC curve, by ranks. 0.5 is chance."""
    n_pos = int(positive.sum())
    n_neg = positive.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1.0, scores.numel() + 1)
    return ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def operator_aucs(logits: torch.Tensor, gold: torch.Tensor) -> list[float]:
    """One AUC per operator, over the rows given, dropping those with a single class.

    Both the stopping score and the reported score are means over these, so they are
    taken from here rather than each recomputing which operators can be scored.
    """
    areas = (auc(logits[:, k], gold[:, k]) for k in range(gold.size(1)))
    return [area for area in areas if not math.isnan(area)]


def macro_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean AUC over the operators that could be scored."""
    areas = operator_aucs(logits, targets.bool())
    return sum(areas) / len(areas) if areas else float("nan")


def evaluate(
    features: torch.Tensor, targets: torch.Tensor, train: torch.Tensor
) -> dict[str, float]:
    """Fit one linear map for all operators at once; score it on the unseen signatures."""
    # Stopped on the macro AUC it reports, read on the validation slice. Exact-set match
    # rides along on the same map: it is a thresholded reading of these scores, and giving
    # it a stopping point of its own would report two maps as if they were one probe.
    w, b, _ = fit_linear(
        features, targets, train, F.binary_cross_entropy_with_logits, targets.size(1),
        macro_auc,
    )
    with torch.no_grad():
        logits = features[~train] @ w + b
    gold = targets[~train].bool()
    areas = operator_aucs(logits, gold)
    return {
        "macro_auc": sum(areas) / len(areas),
        "exact_set_match": ((logits > 0) == gold).all(dim=1).float().mean().item(),
    }


if __name__ == "__main__":
    _, signatures, question_ids = representations(ARMS[0], SEEDS[0])
    testdev = load_json(cfg.gqa.testdev_questions)
    operator_sets = [s.split(" > ") for s in signatures]

    counts = Counter(o for ops in operator_sets for o in ops)
    operators = sorted(o for o, c in counts.items() if c >= MIN_POSITIVE)
    column = {o: k for k, o in enumerate(operators)}
    targets = torch.zeros(len(signatures), len(operators))
    for row, ops in enumerate(operator_sets):
        for o in ops:
            if o in column:
                targets[row, column[o]] = 1.0

    # Signature-disjoint: whole signatures fall on one side, so every signature in the
    # test half is new even though its operators are not.
    unique = sorted(set(signatures))
    side = dict(zip(unique, split_mask(len(unique), splitter()).tolist(), strict=True))
    train = torch.tensor([side[s] for s in signatures])

    # How much of the test half is new *as a target*. Two signatures with the same
    # operators in a different order, or differing only in an operator below
    # MIN_POSITIVE, have identical target rows — and the probe saw one of them.
    vector = {s: tuple(targets[row].tolist()) for row, s in enumerate(signatures)}
    seen = {vector[s] for s in unique if side[s]}
    unseen_signatures = [s for s in unique if not side[s] and vector[s] not in seen]
    unseen_rows = sum(vector[s] not in seen for s in signatures if not side[s])
    target_unseen = {
        "signatures": len(unseen_signatures),
        "questions": unseen_rows,
        "share_of_test_questions": unseen_rows / int((~train).sum()),
    }

    # How many operators can be scored is settled by the split alone — an operator needs
    # both a positive and a negative among the unseen signatures — so it is counted once
    # here rather than returned identically by every fit.
    unseen = targets[~train]
    operators_scored = int(((unseen > 0).any(dim=0) & (unseen == 0).any(dim=0)).sum())

    # Both surface forms, on the same terms as the signature probe (see `cotar.analysis.probing`).
    surface = surface_matrices([testdev[qid]["question"].lower().split() for qid in question_ids])

    print(f"{len(signatures):,} testdev questions, {len(unique)} signatures"
          f" ({int(train.sum()):,} questions from {sum(side.values())} signatures for training,"
          f" the rest from {len(unique) - sum(side.values())} unseen ones)")
    print(f"{len(operators)} operators with {MIN_POSITIVE}+ questions,"
          f" {operators_scored} of them scorable on the unseen half")
    print(f"target vector unseen for {target_unseen['questions']:,} of {int((~train).sum()):,}"
          f" test questions ({target_unseen['share_of_test_questions']:.1%}),"
          f" {target_unseen['signatures']} of {len(unique) - sum(side.values())} signatures\n")

    results: dict[str, dict[str, float]] = {}
    for form, matrix in surface.items():
        results[form] = evaluate(matrix, targets, train)
        print(f"{form:>22}  AUC {results[form]['macro_auc']:6.3f}"
              f"   exact set {results[form]['exact_set_match']:6.1%}")

    for arm in ARMS:
        runs = []
        for seed in SEEDS:
            features, _, _ = representations(arm, seed)
            runs.append(evaluate(F.normalize(features.float(), dim=-1), targets, train))
        results[arm] = {
            k: sum(r[k] for r in runs) / len(runs) for k in ("macro_auc", "exact_set_match")
        } | {
            "per_seed_auc": dict(
                zip(map(str, SEEDS), (r["macro_auc"] for r in runs), strict=True)
            )
        }
        print(f"{arm:>22}  AUC {results[arm]['macro_auc']:6.3f}"
              f"   exact set {results[arm]['exact_set_match']:6.1%}")

    save_json({
        "timestamp": TIMESTAMP,
        "min_positive": MIN_POSITIVE,
        "operators": len(operators),
        "operators_scored": operators_scored,
        "signatures": len(unique),
        "train_signatures": sum(side.values()),
        "split": "signature-disjoint",
        "target_unseen": target_unseen,
        "results": results,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
