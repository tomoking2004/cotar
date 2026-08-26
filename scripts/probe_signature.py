"""Ask what the constrained site carries, and whether it is more than the question's wording.

`separation_d` says how far apart the signature clusters sit in cosine geometry, which is
the quantity the alignment loss optimises directly. It does not say whether the signature
is *readable* there. A linear probe does.

Readable, though, is not yet meaningful. GQA's questions are generated from the very
programs the signature comes from, so the wording and the procedure are correlated by
construction — a probe could score well by reading nothing but surface form. Two controls
separate them:

- a probe on the question's wording, which is pure surface and nothing else, and
- a **prefix-disjoint** split, where no first-few-words prefix is shared between the two
  halves, so a probe that keys on the wording has to generalise past the wordings it saw.

A representation that beats the surface probe, and holds up when the wordings are new, is
carrying the procedure in a form the surface does not hand it for free.

The arms are probed through the same splits with the same settings, so the only difference
between their numbers is the representation itself.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from cotar.analysis import (
    ARMS,
    FORMAT_FEATURES,
    MIN_COUNT,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    classify,
    format_matrix,
    keep_frequent,
    majority_floor,
    predictions,
    probe_accuracy,
    representations,
    residualize,
    scorable,
    split_mask,
    splitter,
    surface_matrices,
    surface_vocabularies,
)
from cotar.config import cfg
from cotar.utils import load_json, save_json

PREFIX_LEN = 4
OUT_PATH   = analysis_path(__file__)


if __name__ == "__main__":
    _, signatures, question_ids = representations(ARMS[0], SEEDS[0])
    testdev = load_json(cfg.gqa.testdev_questions)

    keep, classes, labels = keep_frequent(signatures)
    words = [testdev[question_ids[i]]["question"].lower().split() for i in keep]
    prefixes = [" ".join(w[:PREFIX_LEN]) for w in words]
    formats = format_matrix(words, [testdev[question_ids[i]]["answer"] for i in keep])

    # Neither surface form is the ceiling on its own; the ceiling the representation has
    # to beat is whichever scores higher, so both are measured (see `cotar.analysis.probing`).
    surface = surface_matrices(words)
    distinct = surface_vocabularies(words)

    # Both splits are drawn from one source, in this order.
    generator = splitter()
    # Rows fall on either side independently: the same wording can appear in both.
    by_row = split_mask(len(keep), generator)
    # Whole prefixes fall on one side, so every wording in the test half is new.
    unique = sorted(set(prefixes))
    side = dict(zip(unique, split_mask(len(unique), generator).tolist(), strict=True))
    splits = {
        "random": by_row,
        "prefix-disjoint": torch.tensor([side[p] for p in prefixes]),
    }

    print(f"{len(keep):,} of {len(signatures):,} testdev questions kept"
          f" ({len(classes)} signatures with {MIN_COUNT}+ rows,"
          f" {len(unique):,} distinct {PREFIX_LEN}-word prefixes)")
    print("surface vocabularies: " + ", ".join(
        f"{form} {matrix.size(1):,} of {distinct[form]:,}"
        f"{' (capped)' if distinct[form] > matrix.size(1) else ''}"
        for form, matrix in surface.items()) + "\n")

    results: dict[str, dict[str, object]] = {}
    for name, train in splits.items():
        rows, sub_labels, sub_train, n_classes = scorable(labels, train)

        surface_accuracy = {
            form: probe_accuracy(matrix[rows], sub_labels, sub_train, n_classes)
            for form, matrix in surface.items()
        }
        # What the format features reach on their own — the counterpart of the source
        # accuracy Sahoo et al. ask to be reported beside any cross-dataset probe.
        format_only = probe_accuracy(formats[rows], sub_labels, sub_train, n_classes)

        entry: dict[str, object] = {
            "questions": int(rows.sum()),
            "classes": n_classes,
            "floor": majority_floor(sub_labels, sub_train),
            "surface": surface_accuracy,
            "surface_ceiling": max(surface_accuracy.values()),
            "format_only": format_only,
        }
        print(f"[{name} split] {entry['questions']:,} questions, {entry['classes']} signatures")
        print(f"{'floor (majority)':>22}  {entry['floor']:6.1%}")
        for form, accuracy in surface_accuracy.items():
            print(f"{form:>22}  {accuracy:6.1%}")
        print(f"{'format only':>22}  {format_only:6.1%}"
              f"   ({', '.join(FORMAT_FEATURES)})")

        for arm in ARMS:
            accuracies, residual = [], []
            for seed in SEEDS:
                features, _, _ = representations(arm, seed)
                # Unit-norm rows: the alignment loss and the stability metrics both act on
                # direction alone, so the probe is asked the same question they are.
                x = F.normalize(features[keep][rows].float(), dim=-1)
                accuracies.append(probe_accuracy(x, sub_labels, sub_train, n_classes))
                # The same probe on what is left once format is regressed out. Residualised
                # after normalising, so both probes read the same vectors up to that step.
                residual.append(
                    probe_accuracy(
                        residualize(x, formats[rows]), sub_labels, sub_train, n_classes
                    )
                )
            entry[arm] = {
                "mean": sum(accuracies) / len(accuracies),
                "per_seed": dict(zip(map(str, SEEDS), accuracies, strict=True)),
                "residualized_mean": sum(residual) / len(residual),
                "residualized_per_seed": dict(zip(map(str, SEEDS), residual, strict=True)),
            }
            spread = max(accuracies) - min(accuracies)
            print(f"{arm:>22}  {entry[arm]['mean']:6.1%}  (seed spread {spread:.1%})"
                  f"   │ residualised {entry[arm]['residualized_mean']:6.1%}")
        print()
        results[name] = entry

    # ── is the procedure readable on the questions the model gets wrong? ──────
    # If it reads just as well there, the errors are not procedure confusion, and a
    # better procedure representation has nothing to fix. This is the ceiling on what
    # alignment could ever do for accuracy.
    #
    # The gap in fact runs the other way — the probe reads the errors slightly *better* —
    # and that is composition rather than a finding. The share of the most common
    # signature is measured beside it to show why: the class the probe finds easiest is
    # also the one the model gets wrong most often. Measured over every testdev question,
    # not the probed subset, because it is a claim about where the errors fall.
    gold = {qid: entry["answer"] for qid, entry in testdev.items()}
    kept = torch.tensor(keep)
    top_signature = Counter(signatures).most_common(1)[0][0]
    is_top = torch.tensor([s == top_signature for s in signatures])

    by_correctness: dict[str, dict[str, float]] = {}
    top_share: dict[str, dict[str, float]] = {}
    print("probe accuracy split by whether the model answered correctly (random split),"
          f"\nand how much of each side is `{top_signature}`")
    for arm in ARMS:
        on_correct, on_wrong, gaps, top_right, top_wrong = [], [], [], [], []
        for seed in SEEDS:
            answered = predictions(arm, seed)
            correct = torch.tensor([answered[qid] == gold[qid] for qid in question_ids])
            top_right.append(is_top[correct].float().mean().item())
            top_wrong.append(is_top[~correct].float().mean().item())

            right = correct[kept]
            features, _, _ = representations(arm, seed)
            x = F.normalize(features[keep].float(), dim=-1)
            hit = classify(x, labels, by_row, len(classes)) == labels
            on_correct.append(hit[~by_row & right].float().mean().item())
            on_wrong.append(hit[~by_row & ~right].float().mean().item())
            gaps.append(on_correct[-1] - on_wrong[-1])
        by_correctness[arm] = {
            "on_correct": sum(on_correct) / len(on_correct),
            "on_wrong": sum(on_wrong) / len(on_wrong),
            "gap": sum(gaps) / len(gaps),
        }
        top_share[arm] = {
            "on_correct": sum(top_right) / len(top_right),
            "on_wrong": sum(top_wrong) / len(top_wrong),
        }
        c, w, g = (by_correctness[arm][k] for k in ("on_correct", "on_wrong", "gap"))
        print(f"{arm:>22}  correct {c:6.1%}   wrong {w:6.1%}   gap {100 * g:+5.1f}pt"
              f"   │ top signature: correct {top_share[arm]['on_correct']:5.1%}"
              f"   wrong {top_share[arm]['on_wrong']:5.1%}")

    save_json({
        "timestamp": TIMESTAMP,
        "min_count": MIN_COUNT,
        "prefix_len": PREFIX_LEN,
        # Both, because the cap bites for one form and not the other: `columns` is what
        # the probe was given, `distinct` is how much there was to give.
        "vocabulary": {
            form: {"columns": matrix.size(1), "distinct": distinct[form]}
            for form, matrix in surface.items()
        },
        "splits": results,
        "by_correctness": by_correctness,
        "top_signature": {"signature": top_signature, "share": top_share},
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
