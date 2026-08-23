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

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cotar.analysis import (
    ARMS,
    MIN_COUNT,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    keep_frequent,
    majority_floor,
    probe_accuracy,
    representations,
    run_id,
    scorable,
    split_mask,
    splitter,
)
from cotar.config import cfg
from cotar.utils import load_json, save_json

# Widths to sweep. Bounded above by the answer vocabulary's own rank, which is one less
# than the number of distinct first tokens, so 128 stays comfortably inside it.
DIMS = (8, 32, 128)

# `trainer.test(..., use_best=True)` scored the run with this checkpoint, so this is the
# one whose output layer belongs beside the reported numbers.
CHECKPOINT_NAME = "best.pth"

BASIS_SEED = 0

# The tokenizer has to be the one these weights were trained with, and the run records
# which that was: `run_training` files it into the checkpoint's extras. Stated here so
# the tokenizer can be built before the first checkpoint is opened, and checked against
# the record on every run so the two can never quietly disagree.
MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

OUT_PATH = analysis_path(__file__)


# ── the output layer ─────────────────────────────────────────────────────────


def _state_dict(blob: Any, path: Path) -> dict[str, torch.Tensor]:
    """The parameter mapping inside a checkpoint, whichever key the trainer filed it under."""
    if isinstance(blob, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            inner = blob.get(key)
            if isinstance(inner, dict) and any(isinstance(v, torch.Tensor) for v in inner.values()):
                return inner
        if any(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob
        raise SystemExit(
            f"{path}: no parameter mapping found. Top-level keys: {sorted(blob)[:20]}"
        )
    raise SystemExit(f"{path}: expected a dict, found {type(blob).__name__}.")


def _recorded_model(blob: dict[str, Any]) -> str | None:
    """The base checkpoint this run was trained from, if the trainer filed it."""
    for value in blob.values():
        if isinstance(value, dict) and isinstance(value.get("checkpoint"), str):
            return value["checkpoint"]
    return None


def checkpoint_path(run: str) -> Path:
    return cfg.runs_root / run / "checkpoints" / CHECKPOINT_NAME


def require_checkpoints(runs: list[str]) -> None:
    """Fail before the first fit rather than after the last one.

    Every run is needed for the comparison to mean anything, and the fits ahead take long
    enough that discovering a missing checkpoint at the end would cost the whole pass.
    """
    if missing := [run for run in runs if not checkpoint_path(run).exists()]:
        raise SystemExit(
            f"{len(missing)} of {len(runs)} checkpoints are missing under {cfg.runs_root}:\n"
            + "\n".join(f"  {checkpoint_path(run)}" for run in missing)
            + "\n\nThese are excluded from the snapshots by design, so they exist only on the "
              "machine that ran the training. Nothing here can be measured without them."
        )


def output_weight(run: str) -> tuple[torch.Tensor, str]:
    """The output layer's weight matrix `(V, H)`, read straight out of the checkpoint.

    The model is never built: only this one matrix is wanted, and instantiating the VLM to
    reach it would pull the whole checkpoint onto a device for nothing.

    Small language models often tie the output layer to the input embedding, in which case
    the checkpoint carries no `lm_head` at all and the embedding *is* the output layer.
    Both are accepted, and which one was used is recorded — the two are the same matrix
    when tied, and reading the wrong one when they are not would be silent.
    """
    path = checkpoint_path(run)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        recorded = _recorded_model(blob)
        if recorded is not None and recorded != MODEL:
            raise SystemExit(f"{path}: trained on {recorded}, but this script assumes {MODEL}.")
    state = _state_dict(blob, path)

    for suffix, name in (("lm_head.weight", "lm_head"), ("embed_tokens.weight", "tied embedding")):
        hits = [k for k in state if k.endswith(suffix)]
        if len(hits) > 1:  # the text tower's, not the vision tower's
            hits = [k for k in hits if "text" in k] or hits
        if len(hits) == 1:
            return state[hits[0]].detach().float(), f"{name} ({hits[0]})"
    raise SystemExit(
        f"{path}: found neither an lm_head nor an embedding to use as the output layer.\n"
        f"  keys ending in '.weight': {[k for k in state if k.endswith('.weight')][:20]}"
    )


def answer_token_rows(answers: set[str]) -> torch.Tensor:
    """The output-layer rows the answer vocabulary can be decided on.

    The first token of an answer is where the answer is committed to: by the time a later
    token is emitted the earlier ones are in the context and the choice is already made.
    Answers are encoded both bare and with a leading space, because whether the template
    puts a space before the answer decides which token id comes first and getting it wrong
    would silently select a different set of rows.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids: set[int] = set()
    for answer in answers:
        for text in (answer, f" {answer}"):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if encoded:
                ids.add(int(encoded[0]))
    return torch.tensor(sorted(ids))


def output_basis(weight: torch.Tensor, token_rows: torch.Tensor, m: int) -> torch.Tensor:
    """An orthonormal `(H, m)` basis for the answer rows' leading directions, centred."""
    answers = weight[token_rows]
    centred = answers - answers.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return vh[:m].T.contiguous()


def random_basis(hidden: int, m: int, seed: int) -> torch.Tensor:
    """An orthonormal `(H, m)` basis with no relation to the output layer."""
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(hidden, m, generator=generator)
    q, _ = torch.linalg.qr(gaussian)
    return q[:, :m].contiguous()


# ── the measurement ──────────────────────────────────────────────────────────


def inside(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Coordinates within the subspace — `m` numbers per row."""
    return x @ basis


def outside(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """What the subspace does not hold — still `H` numbers per row, with that part removed."""
    return x - (x @ basis) @ basis.T


WHERE = ("in_output_span", "in_random_span", "outside_output_span")


def gain(summary: dict[str, Any], key: str) -> float:
    """Proposal minus baseline at one place, in percentage points."""
    return 100 * (summary["proposal"][key] - summary["baseline"][key])


def reported_gain() -> float | None:
    """The full-space gain §5.1 reports, read from the file that measured it.

    Restating the number here would give it a second home to drift from, so it is taken
    from `probe_signature.py`'s own output. That analysis may not have been run, in which
    case the pointer to the section stands on its own.
    """
    path = analysis_path("probe_signature.py")
    if not path.exists():
        return None
    split = load_json(path)["splits"]["random"]
    return 100 * (split["proposal"]["mean"] - split["baseline"]["mean"])


if __name__ == "__main__":
    require_checkpoints([run_id(arm, seed) for arm in ARMS for seed in SEEDS])

    # The row order and the kept rows are the ones §5.1 probes, so its numbers are the
    # yardstick the full-space fit below is checked against.
    _, signatures, _ = representations(ARMS[0], SEEDS[0])
    keep, _, labels = keep_frequent(signatures)
    rows, sub_labels, sub_train, n_classes = scorable(labels, split_mask(len(keep), splitter()))

    testdev = load_json(cfg.gqa.testdev_questions)
    answer_rows = answer_token_rows({entry["answer"] for entry in testdev.values()})

    print(f"{int(rows.sum()):,} questions, {n_classes} signatures, "
          f"floor {majority_floor(sub_labels, sub_train):.1%}")
    print(f"{len(answer_rows):,} output-layer rows from the answer vocabulary\n")

    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            weight, source = output_weight(run_id(arm, seed))
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
                u = output_basis(weight, answer_rows, m)
                r = random_basis(x.size(1), m, BASIS_SEED)
                scores = dict(zip(WHERE, (
                    probe_accuracy(inside(x, u), sub_labels, sub_train, n_classes),
                    probe_accuracy(inside(x, r), sub_labels, sub_train, n_classes),
                    probe_accuracy(outside(x, u), sub_labels, sub_train, n_classes),
                ), strict=True))
                entry["by_dim"][str(m)] = scores
                print(f"{f'm = {m}':>26}  "
                      f"in {scores['in_output_span']:6.1%}   "
                      f"random {scores['in_random_span']:6.1%}   "
                      f"outside {scores['outside_output_span']:6.1%}")
            results[arm][str(seed)] = entry
            print()

    # ── the comparison the verdict rests on ──────────────────────────────────
    # Not the level inside the span, which any wide-enough subspace reaches, but how much
    # of the alignment's gain survives there. If the gain over baseline is present inside
    # the output layer's span, the structure alignment built is where the answer is read.
    summary: dict[str, Any] = {}
    print(f"{'':>10}" + "".join(f"{f'm={m}':>34}" for m in DIMS))
    print(f"{'':>10}" + "".join(f"{'in-span':>12}{'random':>11}{'outside':>11}" for _ in DIMS))
    for arm in ARMS:
        means = {
            f"{where}_{m}": sum(
                results[arm][str(seed)]["by_dim"][str(m)][where] for seed in SEEDS
            ) / len(SEEDS)
            for m in DIMS
            for where in WHERE
        }
        means["full"] = sum(results[arm][str(s)]["full"] for s in SEEDS) / len(SEEDS)
        summary[arm] = means
        print(f"{arm:>10}" + "".join(
            f"{means[f'{where}_{m}']:>11.1%}" for m in DIMS for where in WHERE
        ))

    print("\ngain over baseline (proposal − baseline), by where it is measured")
    for m in DIMS:
        gains = (
            f"{where.replace('_', ' ')} {gain(summary, f'{where}_{m}'):+5.1f}pt"
            for where in WHERE
        )
        print(f"  m = {m:>3}:  " + "   ".join(gains))
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
