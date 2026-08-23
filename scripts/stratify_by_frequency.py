"""Re-read a finished experiment through the task-signature distribution. No GPU.

Overall testdev accuracy is dominated by whichever signatures are frequent, so an effect
confined to rare procedures cannot show up in it. This splits the same predictions by how
often each question's signature was seen in training and scores the arms inside each
stratum, which is the measurement that tells "no effect" apart from "measured where an
effect could not appear".

Everything here comes from files already on disk: the questions files and each run's
predictions. The overall accuracy is recomputed on the way through and checked against
what the official evaluator reported, so a silent mismatch in scoring cannot pass as a
result.

That recomputation is a *check*, not the reported number. Stratifying needs a per-question
verdict, which the official evaluator does not hand back, so accuracy is scored again here
and the `all` row is what those verdicts come to. The reported accuracy is the official
evaluator's own, summarised by `summarize_runs.py`; the two agree to within the hundredth
of a point the check enforces, and quoting them interchangeably is how a document ends up
with two spreads for one quantity.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path

from cotar.analysis import (
    ARMS,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    paired_difference,
    predictions,
    reported_accuracy,
    run_id,
    summarize,
)
from cotar.config import cfg
from cotar.data import task_signature
from cotar.utils import load_json, save_json

# Open-ended on the right because one signature alone holds a quarter of the data:
# without a top bucket the head would swamp whatever the tail is doing.
BUCKETS  = ((0, 0), (1, 10), (11, 100), (101, 1_000), (1_001, None))
OUT_PATH = analysis_path(__file__)


def label_of(lo: int, hi: int | None) -> str:
    """What a bucket is called — `0`, `1-10`, `1001+`. The name a stratum carries into
    the report, so it is spelled here alone.
    """
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def bucket_of(count: int) -> str:
    for lo, hi in BUCKETS:
        if count >= lo and (hi is None or count <= hi):
            return label_of(lo, hi)
    raise ValueError(f"no bucket for {count}")


def row(label: str, questions: int, binary: float, by_arm: dict[str, dict[int, float]]) -> str:
    """One printed line: each arm's mean ± sd over the seeds, then proposal − baseline
    with the interval that says whether the stratum shows anything at all.

    The total is that same line over every question rather than one stratum, so it is
    built here too instead of being formatted a second time beneath the table.
    """
    cells = "  ".join(
        f"{(s := summarize(by_arm[arm]))['mean']:>7.2f} ±{s['sd']:5.2f}" for arm in ARMS
    )
    gap = paired_difference(by_arm["proposal"], by_arm["baseline"])
    mark = "*" if gap["excludes_zero"] else " "
    return (f"{label:>10} {questions:>10,} {binary:6.1%}  {cells}   {gap['mean']:+6.2f}"
            f"  [{gap['ci_low']:+6.2f}, {gap['ci_high']:+6.2f}]{mark}")


def read_questions(path: Path) -> dict[str, dict[str, str]]:
    """`question_id -> {signature, answer}` for every question with a program."""
    return {
        qid: {"signature": task_signature(entry["semantic"]), "answer": entry["answer"]}
        for qid, entry in load_json(path).items()
        if entry.get("semantic")
    }


if __name__ == "__main__":
    train   = read_questions(cfg.gqa.train_questions)
    testdev = read_questions(cfg.gqa.testdev_questions)
    train_counts = Counter(q["signature"] for q in train.values())

    strata = {qid: bucket_of(train_counts[q["signature"]]) for qid, q in testdev.items()}
    labels = [label_of(lo, hi) for lo, hi in BUCKETS]
    sizes = Counter(strata.values())

    # Accuracy differs sharply between strata, and it is worth knowing why before reading
    # anything into it: yes/no questions score far higher than open ones, so a stratum's
    # level says more about the mix of question types it holds than about how rare its
    # procedures are. The arm-to-arm comparison is unaffected — the arms answer the same
    # questions — but the levels are not comparable across rows.
    binary_share = {
        label: statistics.mean(
            testdev[qid]["answer"] in {"yes", "no"}
            for qid, s in strata.items() if s == label
        )
        for label in labels if sizes[label]
    }

    per_run: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    overall: dict[str, dict[int, float]] = defaultdict(dict)
    for arm in ARMS:
        for seed in SEEDS:
            answered = predictions(arm, seed)
            correct: dict[str, list[int]] = defaultdict(list)
            for qid, q in testdev.items():
                correct[strata[qid]].append(int(answered[qid] == q["answer"]))
            per_run[arm][seed] = {
                label: 100 * statistics.mean(correct[label]) for label in labels if correct[label]
            }
            scored = [hit for hits in correct.values() for hit in hits]
            overall[arm][seed] = 100 * statistics.mean(scored)

            official = reported_accuracy(arm, seed)
            if abs(overall[arm][seed] - official) > 0.01:
                raise SystemExit(
                    f"{run_id(arm, seed)}: recomputed {overall[arm][seed]:.2f} but the "
                    f"official evaluator reported {official:.2f} — the scoring here is not "
                    f"faithful."
                )

    # Each stratum's arms, seed by seed — what the table is printed from and what the
    # differences below are taken over, so the two cannot come from different readings.
    strata_arms = {
        label: {arm: {seed: per_run[arm][seed][label] for seed in SEEDS} for arm in ARMS}
        for label in labels
        if sizes[label]
    }

    print("accuracy by how often the signature was seen in training (mean over 3 seeds, ± sd)")
    print(f"{'train freq':>10} {'questions':>10} {'yes/no':>7}  "
          + "  ".join(f"{arm:>14}" for arm in ARMS)
          + "   proposal−baseline  95% interval  (* excludes 0)")
    for label, in_stratum in strata_arms.items():
        print(row(label, sizes[label], binary_share[label], in_stratum))
    all_binary = statistics.mean(q["answer"] in {"yes", "no"} for q in testdev.values())
    print(row("all", len(testdev), all_binary, overall))

    save_json({
        "timestamp": TIMESTAMP,
        "strata_sizes": {label: sizes[label] for label in labels if sizes[label]},
        "strata_binary_share": binary_share,
        "accuracy_by_stratum": {
            arm: {str(seed): per_run[arm][seed] for seed in SEEDS} for arm in ARMS
        },
        "accuracy_overall": {
            arm: {str(seed): overall[arm][seed] for seed in SEEDS} for arm in ARMS
        },
        # The claim each row carries is a difference and whether its interval clears
        # zero, so both are filed rather than left for whoever quotes the row to redo.
        "proposal_minus_baseline": {
            label: paired_difference(arms["proposal"], arms["baseline"])
            for label, arms in ({**strata_arms, "all": overall}).items()
        },
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
