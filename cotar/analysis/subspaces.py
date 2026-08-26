"""Reading a representation inside a subspace and outside it, and the controls that make
either reading mean something.

Both stages of context.md §7.1 ask the same question of the same rows and differ only in
where the subspace comes from: the first takes the output layer's answer directions as
they stand, the second pulls them back to the constrained site through the network in
between. Everything after that — the four places a probe is run, and the statistic the
verdict rests on — is shared, and lives here so the two stages cannot answer the same
question two ways.

**Both sides need the random twin.** A subspace wide enough holds the signature whatever
it is built from, so scoring well inside `U` is not evidence. A complement that keeps most
of the dimensions loses almost nothing whatever was removed, so scoring well outside `U`
is not evidence either — and that is the side the bypass hypothesis lives on, since bypass
means the structure sits where the output does *not* read. What carries information is the
difference between `U` and a random subspace of the same width, at each of the two places.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any

import torch
from transformers import AutoTokenizer

from ..utils import load_json
from .experiment import analysis_path
from .probing import probe_accuracy

__all__ = [
    "WHERE",
    "answer_token_rows",
    "delta",
    "gain",
    "inside",
    "output_basis",
    "outside",
    "random_basis",
    "reported_gain",
    "scores",
    "seed_deltas",
    "summarize_places",
]

# The four readings, in the order they are reported. Each subspace is read from both
# sides, and each side is paired with its random twin — see the module docstring.
WHERE = (
    "in_output_span",
    "in_random_span",
    "outside_output_span",
    "outside_random_span",
)


def answer_token_rows(answers: set[str], model: str) -> torch.Tensor:
    """The output-layer rows the answer vocabulary can be decided on.

    The first token of an answer is where the answer is committed to: by the time a later
    token is emitted the earlier ones are in the context and the choice is already made.
    Answers are encoded both bare and with a leading space, because whether the template
    puts a space before the answer decides which token id comes first, and getting it
    wrong would silently select a different set of rows.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    ids: set[int] = set()
    for answer in answers:
        for text in (answer, f" {answer}"):
            if encoded := tokenizer(text, add_special_tokens=False).input_ids:
                ids.add(int(encoded[0]))
    return torch.tensor(sorted(ids))


def output_basis(weight: torch.Tensor, token_rows: torch.Tensor, m: int) -> torch.Tensor:
    """An orthonormal `(H, m)` basis for the answer rows' leading directions, centred.

    The rows are centred before the span is taken. A component common to every answer row
    shifts every logit alike, so it cannot decide between answers — and deciding between
    them is the only thing "the output layer reads this direction" can usefully mean.
    """
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


def inside(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Coordinates within the subspace — `m` numbers per row."""
    return x @ basis


def outside(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """What the subspace does not hold — still `H` numbers per row, with that part removed.

    Not renormalised afterwards: scaling the residual back to unit length would restore
    the magnitude the projection removed, which is part of what is being measured.
    """
    return x - (x @ basis) @ basis.T


def scores(
    x: torch.Tensor,
    basis: torch.Tensor,
    control: torch.Tensor,
    labels: torch.Tensor,
    train: torch.Tensor,
    n_classes: int,
) -> dict[str, float]:
    """One run's four readings at one width, keyed by `WHERE`."""
    return dict(
        zip(
            WHERE,
            (
                probe_accuracy(inside(x, basis), labels, train, n_classes),
                probe_accuracy(inside(x, control), labels, train, n_classes),
                probe_accuracy(outside(x, basis), labels, train, n_classes),
                probe_accuracy(outside(x, control), labels, train, n_classes),
            ),
            strict=True,
        )
    )


def gain(summary: dict[str, dict[str, float]], key: str) -> float:
    """Proposal minus baseline at one place, in percentage points."""
    return 100 * (summary["proposal"][key] - summary["baseline"][key])


def delta(summary: dict[str, dict[str, float]], m: int, side: str) -> float:
    """The statistic the verdict rests on, at one width and one side, in percentage points.

    What alignment gained in `U` at this side, less what it gained in a random subspace of
    the same width. Neither the level nor the gain decides anything on its own — both
    survive whatever the subspace is built from (see the module docstring) — so this
    difference is what the two stages of context.md §7.1 are read on.
    """
    return gain(summary, f"{side}_output_span_{m}") - gain(summary, f"{side}_random_span_{m}")


def seed_deltas(
    runs: dict[str, dict[str, Any]], m: int, side: str, seeds: Sequence[int]
) -> list[float]:
    """The same statistic before the seeds are averaged.

    Their mean is `delta`, which is all the mean can say. What it cannot say is whether the
    three runs agree about the sign — and a Δ whose seeds disagree is a Δ the experiment
    has not established the direction of.
    """
    def place(arm: str, seed: int, kind: str) -> float:
        return runs[arm][str(seed)]["by_dim"][str(m)][f"{side}_{kind}_span"]

    return [
        100 * (
            (place("proposal", seed, "output") - place("baseline", seed, "output"))
            - (place("proposal", seed, "random") - place("baseline", seed, "random"))
        )
        for seed in seeds
    ]


def summarize_places(
    runs: dict[str, dict[str, Any]],
    dims: Sequence[int],
    arms: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Each arm's readings, averaged over the seeds.

    Keyed `<place>_<width>` for the four places at each width, and `full` for the fit on
    the whole vector beside them. Shared because averaging over seeds is a decision the
    study makes once, not a shape each stage may arrive at its own way.
    """
    return {
        arm: {
            f"{where}_{m}": mean(
                runs[arm][str(seed)]["by_dim"][str(m)][where] for seed in seeds
            )
            for m in dims
            for where in WHERE
        }
        | {"full": mean(runs[arm][str(seed)]["full"] for seed in seeds)}
        for arm in arms
    }


def reported_gain() -> float | None:
    """The full-space gain §5.1 reports, read from the file that measured it.

    Both stages fit the whole vector alongside their projections, and that fit has to
    reproduce this number — a mismatch means the checkpoint and the saved representations
    are not from the same run. Restating the number would give it a second home to drift
    from, so it comes from `probe_signature.py`'s own output. That analysis may not have
    been run, in which case the pointer to the section stands on its own.
    """
    path = analysis_path("probe_signature.py")
    if not path.exists():
        return None
    split = load_json(path)["splits"]["random"]
    return 100 * (split["proposal"]["mean"] - split["baseline"]["mean"])
