"""Measure every dataset-level number context.md states. No GPU, no trained model.

These are the claims a reader can check before any model is involved: how the task
signatures are distributed, how many questions cannot be paired, whether the three
splits really are disjoint, and whether GQA's own notion of an entailed question
agrees with the signature. Measured once by hand or by a throwaway script, such
numbers become unaccountable the moment the script is gone — the document asserts
them and nothing can be re-run to check. They live here so that a single run
reproduces all of them together, and so that a number that drifts is caught rather
than believed.

Everything reads the questions files on disk. Nothing depends on a training run.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection
from itertools import combinations
from pathlib import Path
from typing import Any

from cotar.analysis import analysis_path
from cotar.config import cfg
from cotar.data import GQADataset, task_signature
from cotar.utils import load_json, save_json

TOP_K    = 10
SPLITS   = {
    "train":   cfg.gqa.train_questions,
    "val":     cfg.gqa.val_questions,
    "testdev": cfg.gqa.testdev_questions,
}
# The split every reported number is measured on. Named once because two measurements
# below are pinned to it, and they must stay pinned to the same one.
REPORTED = "testdev"
# Signature statistics are reported for the split that is trained on and the split that
# is reported on; `val` only ever selects, and no number in the document comes from it.
PROFILED = ("train", REPORTED)
OUT_PATH = analysis_path(__file__)


def split_profile(questions_path: Path) -> dict[str, Any]:
    """How the procedures are distributed, how many questions have no partner, and how
    wide the answer vocabulary is.

    Read through `GQADataset` rather than the raw file so that the counts are the ones
    training actually saw: questions without a program, or whose image is absent, never
    reach a batch and must not reach these numbers either.
    """
    dataset = GQADataset(questions_path, cfg.gqa.images, require_program=True)
    counts = Counter(dataset.signatures)
    sizes = sorted(counts.values(), reverse=True)
    total = sum(sizes)
    return {
        "questions": total,
        # Closed enough to enumerate — which is what lets §A.4 read the answer directions
        # straight off the output embedding instead of running the model.
        "answer_vocabulary": len({record["answer"] for record in dataset.samples}),
        "signatures": len(sizes),
        # A signature holding one sample is a question with no same-signature sibling, so
        # it can form no positive pair and the grouped sampler drops it.
        "unpairable_questions": sizes.count(1),
        "top1_share": sizes[0] / total,
        f"top{TOP_K}_share": sum(sizes[:TOP_K]) / total,
        "most_common": [
            {"signature": signature, "count": n} for signature, n in counts.most_common(TOP_K)
        ],
    }


def split_ids(questions_path: Path) -> dict[str, set[str]]:
    """The question ids and image ids of one split, keeping neither file in memory."""
    entries: dict[str, dict[str, Any]] = load_json(questions_path)
    return {
        "question": set(entries),
        "image": {entry["imageId"] for entry in entries.values()},
    }


def disjointness(ids: dict[str, dict[str, set[str]]]) -> list[dict[str, Any]]:
    """Every split pair, by both kinds of id — an image shared across splits leaks as
    surely as a question does, and only the question ids are disjoint by construction.
    """
    return [
        {"left": left, "right": right, "kind": kind,
         "overlap": len(ids[left][kind] & ids[right][kind])}
        for left, right in combinations(ids, 2)
        for kind in ("question", "image")
    ]


def entailment_agreement(questions_path: Path) -> dict[str, Any]:
    """How often GQA's entailed pairs share a task signature, and what the differences are.

    GQA's consistency metric requires consistent answers across pairs it marks as
    entailed. Where such a pair has two different signatures, the official metric wants
    the pair aligned and the alignment loss pushes it apart — so this rate measures how
    far the signature is from GQA's own notion of "the same question, asked twice".

    The rate alone invites the reading that the signature is simply wrong. What kind of
    difference it is decides that, so it is measured here rather than argued: GQA's
    entailment relates questions about the same scene, and a signature is a procedure.
    Two questions can share a scene and be answered by different procedures — and if that
    is what the mismatches are, the rate is the two notions doing different jobs rather
    than one of them failing. `answer_kind` is the sharpest form of the question, because
    a yes/no question and an open one cannot be the same procedure whatever else they
    share, and `most_common_mismatches` shows what the rest of the difference is made of.

    Reported both ways on purpose. GQA lists most of these relations from both ends, so
    walking every question's `entailed` list visits such a relation twice; counting the
    visits rather than the pairs double-weights whichever relations happen to be listed
    symmetrically, and moves the rate by several points. `unordered` is the honest pair
    count. `ordered` is kept beside it so that the two can never be confused for each
    other again by whoever reads the number next.
    """
    entries: dict[str, dict[str, Any]] = load_json(questions_path)
    signature = {
        qid: task_signature(entry["semantic"])
        for qid, entry in entries.items()
        if entry.get("semantic")
    }
    visits = [
        (qid, other)
        for qid, entry in entries.items()
        if qid in signature
        for other in entry.get("entailed", ())
        if other != qid and other in signature
    ]
    pairs = {frozenset(visit) for visit in visits}

    def tally[T](items: Collection[T], differs: Callable[[T], bool]) -> dict[str, Any]:
        total = len(items)
        mismatched = sum(differs(item) for item in items)
        return {
            "pairs": total,
            "mismatched": mismatched,
            "mismatch_rate": mismatched / total if total else 0.0,
        }

    def differs(pair: frozenset[str]) -> bool:
        return len({signature[qid] for qid in pair}) > 1

    # Yes/no against open, by the answer itself rather than by the operator that produced
    # it: the claim is about what kind of question a reader sees, not about how the
    # program is written.
    def mixes_answer_kind(pair: frozenset[str]) -> bool:
        return len({entries[qid]["answer"] in ("yes", "no") for qid in pair}) > 1

    split = {True: [p for p in pairs if differs(p)], False: [p for p in pairs if not differs(p)]}
    mismatched_pairs = split[True]

    return {
        "unordered": tally(pairs, differs),
        "ordered": tally(visits, lambda visit: signature[visit[0]] != signature[visit[1]]),
        "listed_from_both_sides": sum(
            qid in entries.get(other, {}).get("entailed", ()) for qid, other in visits
        ),
        "answer_kind": {
            name: {
                "pairs": len(group),
                "mixes_yes_no_with_open": sum(mixes_answer_kind(p) for p in group),
                "rate": (
                    sum(mixes_answer_kind(p) for p in group) / len(group) if group else 0.0
                ),
            }
            for name, group in (("mismatched", mismatched_pairs), ("matched", split[False]))
        },
        "most_common_mismatches": [
            {"signatures": list(sigs), "pairs": n}
            for sigs, n in Counter(
                tuple(sorted(signature[qid] for qid in pair)) for pair in mismatched_pairs
            ).most_common(TOP_K)
        ],
    }


def operator_names_in_questions(questions_path: Path) -> dict[str, Any]:
    """How often an operator puts its own name into the question.

    The wording and the signature are correlated because GQA generates the wording from
    the program, and that correlation has a mechanism worth measuring rather than
    asserting: some operators name themselves in the question, so a probe on the wording
    reads those off for free. Most do not, and that is the other half of the picture —
    the correlation is not a simple name-copy, which is why a probe on the wording lands
    far below one on the representation.

    An operator counts as named when its own name occurs as a whole word sequence in the
    lowercased question. Reported per operator rather than as one average, because the
    leak is all-or-nothing per operator and an average would describe no operator.

    Measured on the reported split alone: it is the split every probe number comes from,
    and the cost is quadratic in operators times questions.
    """
    entries: dict[str, dict[str, Any]] = load_json(questions_path)
    questions = [
        ({step["operation"] for step in entry["semantic"]}, entry["question"].lower())
        for entry in entries.values()
        if entry.get("semantic")
    ]
    asked = Counter(operator for operators, _ in questions for operator in operators)
    named = {
        operator: sum(
            1
            for operators, question in questions
            if operator in operators and pattern.search(question)
        )
        for operator in asked
        for pattern in [re.compile(rf"\b{re.escape(operator)}\b")]
    }
    return {
        "operators": len(asked),
        "named_operators": sum(1 for operator in asked if named[operator]),
        # Only the operators that ever leak: the other 100-odd are the finding's other
        # half, and their count above says it without a row of zeroes each.
        "by_operator": [
            {
                "operator": operator,
                "questions": asked[operator],
                "named": named[operator],
                "rate": named[operator] / asked[operator],
            }
            for operator in sorted(
                asked, key=lambda o: (-named[o] / asked[o], -asked[o])
            )
            if named[operator]
        ],
    }


if __name__ == "__main__":
    ids = {name: split_ids(path) for name, path in SPLITS.items()}
    overlaps = disjointness(ids)

    print("split sizes")
    for name in SPLITS:
        print(f"  {name:>8}  {len(ids[name]['question']):>9,} questions"
              f"  ·  {len(ids[name]['image']):>7,} images")

    print("\nsplit disjointness (an overlap of 0 is what the document claims)")
    for row in overlaps:
        print(f"  {row['left']:>8} vs {row['right']:<8} {row['kind']:>8} id"
              f"  {row['overlap']:>9,}")
    leaks = [row for row in overlaps if row["overlap"]]
    print(f"  → {len(overlaps)} pairs checked, {len(leaks)} with any overlap")

    entailment = entailment_agreement(SPLITS["val"])
    print("\nentailed pairs within val (both sides present, both with a program)")
    for convention in ("unordered", "ordered"):
        tally = entailment[convention]
        print(f"  {convention:>9}  {tally['pairs']:>7,} pairs"
              f"  ·  {tally['mismatched']:>7,} with different signatures"
              f"  ({tally['mismatch_rate']:.1%})")
    print(f"  → {entailment['listed_from_both_sides']:,} of"
          f" {entailment['ordered']['pairs']:,} visits are the return leg of a relation"
          f" already seen")

    print("\n  what kind of difference a mismatch is")
    for name, kind in entailment["answer_kind"].items():
        print(f"    {name:>10}  {kind['mixes_yes_no_with_open']:>7,} of {kind['pairs']:>7,}"
              f"  pair a yes/no question with an open one  ({kind['rate']:.1%})")
    print("    most common mismatched signature pairs")
    for entry in entailment["most_common_mismatches"][:5]:
        left, right = entry["signatures"]
        print(f"      {entry['pairs']:>6,}  {left}   ×   {right}")

    profiles = {name: split_profile(SPLITS[name]) for name in PROFILED}
    for name, profile in profiles.items():
        print(f"\n{name} signatures")
        print(f"  questions    {profile['questions']:>9,}")
        print(f"  signatures   {profile['signatures']:>9,}")
        print(f"  answers      {profile['answer_vocabulary']:>9,}  (distinct)")
        print(f"  unpairable   {profile['unpairable_questions']:>9,}"
              f"  (single-sample signatures, dropped by grouping)")
        print(f"  top 1        {profile['top1_share']:>9.1%}")
        print(f"  top {TOP_K:<2}      {profile[f'top{TOP_K}_share']:>9.1%}")
        for entry in profile["most_common"][:5]:
            print(f"    {entry['count']:>7,}  {entry['signature']}")

    naming = operator_names_in_questions(SPLITS[REPORTED])
    print(f"\noperators naming themselves in the question ({REPORTED})")
    print(f"  {naming['named_operators']} of {naming['operators']} operators ever do."
          f"  The rest never put their name in the wording.")
    for entry in naming["by_operator"]:
        print(f"    {entry['operator']:>16}  {entry['named']:>6,} of {entry['questions']:>6,}"
              f"  {entry['rate']:>7.1%}")

    save_json({
        "splits": {name: {"questions": len(ids[name]["question"]),
                          "images": len(ids[name]["image"])} for name in SPLITS},
        "disjointness": overlaps,
        "entailment_vs_signature": entailment,
        "signature_profiles": profiles,
        "operator_names_in_questions": {"split": REPORTED} | naming,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
