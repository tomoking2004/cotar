"""The linear probe the analyses read representations with, and the control it is held to.

A probe asks what a representation carries in a form a linear map can take out. Three
decisions make its numbers comparable across the arms, and each is here rather than in
the scripts so that no two probes can make them differently.

**Both surface forms are measured.** GQA's questions are generated from the very programs
the labels come from, so a probe on the wording alone bounds how much of the procedure the
question gives away for free. Adjacent word pairs carry word order, but they crowd a
shared vocabulary cap and can cost more than the order buys — which form is the harder
control is a question for the measurement, not an assumption.

**The fit is stopped by a held-out slice, not by a step count.** A shared step budget is
not a shared setting: the aligned arm's representations are nearly linearly separable
before fitting begins and settle within a few hundred steps, while the baseline's and the
shuffled arm's are still climbing. Cut everyone off together and the arms that fit slowly
are reported below what they carry — a difference in fitting speed reads as a difference
in content.

**The stopping score is the metric the probe reports, not the training loss.** Held-out
cross-entropy turns upward as soon as a fit grows confident, which happens far earlier for
a wide sparse vocabulary than for a 960-dimensional representation while both are still
gaining accuracy. Stopping on loss would cut the surface controls short and flatter the
representations they exist to hold to account.

**Format is residualised, not just competed against.** A probe on the wording says what
the words alone reach; it does not say what the representation still holds once the task's
*format* is removed from it. Sahoo et al. (2026) show a case where the two differ
completely: probes separating three reasoning benchmarks scored 100% and fell to chance
once source identity, option count and response length were regressed out. Two of those
three cannot arise here — every question comes from one corpus, and none is multiple
choice — but length can, so it is regressed out rather than assumed harmless.

Removing a format feature also removes whatever the signature legitimately shares with it:
a procedure ending in an existence check really does produce a yes/no answer. The
residualised score is therefore a floor on what survives format, not an unbiased estimate
of it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from itertools import pairwise

import torch
import torch.nn.functional as F

__all__ = [
    "FORMAT_FEATURES",
    "MIN_COUNT",
    "SPLIT_SEED",
    "TRAIN_FRAC",
    "classify",
    "fit_linear",
    "format_matrix",
    "keep_frequent",
    "majority_floor",
    "probe_accuracy",
    "residualize",
    "scorable",
    "split_mask",
    "splitter",
    "surface_matrices",
    "surface_vocabularies",
]

TRAIN_FRAC   = 0.5
SPLIT_SEED   = 0
MIN_COUNT    = 10     # a signature needs enough rows to land on both sides of a split
VOCAB_SIZE   = 8_000
MAX_STEPS    = 8_000
PATIENCE     = 1_000  # steps without a better validation score before stopping
EVAL_EVERY   = 50
VAL_FRAC     = 0.2    # held out of the fitting half to choose the stopping point
LR           = 0.05
WEIGHT_DECAY = 1e-4
RIDGE_ALPHA  = 1.0    # penalty for the format→representation fit that is then subtracted

# What "format" is taken to be here. Source identity and option count — two of the three
# features Sahoo et al. regress out — cannot vary in this study: every question comes from
# one corpus and none is multiple choice. What remains is length, on both sides of the
# exchange, plus whether the answer is a yes/no at all.
FORMAT_FEATURES = ("question_words", "answer_characters", "answer_is_yes_no")


# ── fitting ──────────────────────────────────────────────────────────────────


def fit_linear(
    features: torch.Tensor,
    targets: torch.Tensor,
    train: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_out: int,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A linear map fitted on the fitting half, returned at its best validation step.

    `score_fn` is the reported metric, higher being better, read on the held-out slice.
    """
    rows = train.nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    shuffled = rows[torch.randperm(len(rows), generator=generator)]
    cut = max(1, int(len(shuffled) * VAL_FRAC))
    val, fit_rows = shuffled[:cut], shuffled[cut:]

    w = torch.zeros(features.size(1), n_out, requires_grad=True)
    b = torch.zeros(n_out, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_step = -float("inf"), 0
    best_w, best_b = w.detach().clone(), b.detach().clone()
    for step in range(1, MAX_STEPS + 1):
        opt.zero_grad()
        loss_fn(features[fit_rows] @ w + b, targets[fit_rows]).backward()
        opt.step()
        if step % EVAL_EVERY:
            continue
        with torch.no_grad():
            score = score_fn(features[val] @ w + b, targets[val])
        if score > best:
            best, best_step = score, step
            best_w, best_b = w.detach().clone(), b.detach().clone()
        elif step - best_step >= PATIENCE:
            break
    return best_w, best_b, best_step


def classify(
    features: torch.Tensor, labels: torch.Tensor, train: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """The class this probe predicts for every row, fitted on the training half.

    Every row rather than the test half alone, because one caller scores the split and
    another asks which individual questions the probe got right.
    """
    w, b, _ = fit_linear(
        features, labels, train, F.cross_entropy, n_classes,
        lambda logits, y: (logits.argmax(dim=1) == y).float().mean().item(),
    )
    with torch.no_grad():
        return (features @ w + b).argmax(dim=1)


def probe_accuracy(
    features: torch.Tensor, labels: torch.Tensor, train: torch.Tensor, n_classes: int
) -> float:
    """Top-1 accuracy of that probe on the test half."""
    predicted = classify(features, labels, train, n_classes)
    return (predicted[~train] == labels[~train]).float().mean().item()


def majority_floor(labels: torch.Tensor, train: torch.Tensor) -> float:
    """What the largest class alone scores on the test half — the floor a probe beats."""
    counts = Counter(labels[~train].tolist())
    return max(counts.values()) / sum(counts.values())


# ── which rows and classes a probe is asked about ────────────────────────────


def keep_frequent(
    signatures: list[str], min_count: int = MIN_COUNT
) -> tuple[list[int], list[str], torch.Tensor]:
    """Which rows a probe is asked about, the classes among them, and each row's label.

    A signature seen only a handful of times can land entirely on one side of a split,
    where it is either unlearnable or unscorable, so only common enough ones are kept.
    """
    counts = Counter(signatures)
    keep = [i for i, s in enumerate(signatures) if counts[s] >= min_count]
    classes = sorted({signatures[i] for i in keep})
    index = {s: c for c, s in enumerate(classes)}
    return keep, classes, torch.tensor([index[signatures[i]] for i in keep])


def splitter(seed: int = SPLIT_SEED) -> torch.Generator:
    """The source every split is drawn from, so that a rerun repeats the same halves."""
    return torch.Generator().manual_seed(seed)


def split_mask(n: int, generator: torch.Generator, frac: float = TRAIN_FRAC) -> torch.Tensor:
    """A fitting/test mask over `n` items, drawn independently per item.

    The generator is passed in rather than made here because a script that draws two
    splits draws them in sequence from one source — which halves it gets depends on that
    order, and hiding the source would hide the dependence.
    """
    return torch.rand(n, generator=generator) < frac


def scorable(
    labels: torch.Tensor, train: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The split restricted to the classes present on both of its sides.

    A class the probe never sees cannot be learnt, and one absent from the test half
    cannot be scored; either way it only blurs the number. Returns which rows survived,
    and the labels, mask and class count of what is left.
    """
    usable = set(labels[train].tolist()) & set(labels[~train].tolist())
    rows = torch.tensor([label.item() in usable for label in labels])
    remap = {c: i for i, c in enumerate(sorted(usable))}
    return (
        rows,
        torch.tensor([remap[label.item()] for label in labels[rows]]),
        train[rows],
        len(usable),
    )


# ── the format control ───────────────────────────────────────────────────────


def format_matrix(words: list[list[str]], answers: list[str]) -> torch.Tensor:
    """The task-format features of each question, one row per question.

    Columns are `FORMAT_FEATURES`, standardised so that the ridge penalty falls on each
    equally — a word count and a 0/1 flag are otherwise penalised on wildly different
    scales, and the one measured in larger units would survive the fit untouched.
    """
    rows = torch.tensor(
        [
            [float(len(w)), float(len(a)), float(a in ("yes", "no"))]
            for w, a in zip(words, answers, strict=True)
        ]
    )
    spread = rows.std(dim=0, keepdim=True)
    return (rows - rows.mean(dim=0, keepdim=True)) / spread.clamp(min=1e-8)


def residualize(features: torch.Tensor, formats: torch.Tensor) -> torch.Tensor:
    """What the representation holds beyond what its question's format predicts.

    A ridge fit from the format features onto the representation, subtracted from it. The
    fit uses no signature label, so nothing about the answer the probe is later asked for
    enters here, and it is fitted on every row rather than the probe's training half for
    the same reason — there is no label to leak.

    Not renormalised afterwards. Scaling the residual back to unit length would restore
    the very magnitude the subtraction removed, and a row that lost most of its length
    would come back looking as long as one that lost none.
    """
    ones = torch.ones(len(formats), 1, dtype=formats.dtype)
    design = torch.cat([formats, ones], dim=1)
    penalty = torch.eye(design.size(1), dtype=design.dtype) * RIDGE_ALPHA
    penalty[-1, -1] = 0.0  # the intercept carries no format information to penalise
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ features)
    return features - design @ weights


# ── the surface control ──────────────────────────────────────────────────────


_SURFACES: dict[str, Callable[[list[str]], list[str]]] = {
    "words": lambda ws: ws,
    "words_and_pairs": lambda ws: ws + [f"{a}_{b}" for a, b in pairwise(ws)],
}


def surface_matrices(words: list[list[str]]) -> dict[str, torch.Tensor]:
    """A 0/1 bag-of-grams matrix per surface form, one row per question in the order given."""
    return {name: _encode(words, grams) for name, grams in _SURFACES.items()}


def surface_vocabularies(words: list[list[str]]) -> dict[str, int]:
    """How many distinct grams each surface form has *before* `VOCAB_SIZE` truncates it.

    Reported beside the matrices because the cap bites for only one of the two forms, and
    a column count alone cannot say which: a form under the cap keeps every gram, while
    one over it silently drops its rarest.
    """
    return {
        name: len({gram for ws in words for gram in grams(ws)})
        for name, grams in _SURFACES.items()
    }


def _encode(words: list[list[str]], grams: Callable[[list[str]], list[str]]) -> torch.Tensor:
    counts = Counter(gram for ws in words for gram in grams(ws))
    column = {gram: c for c, (gram, _) in enumerate(counts.most_common(VOCAB_SIZE))}
    matrix = torch.zeros(len(words), len(column))
    for row, ws in enumerate(words):
        for gram in grams(ws):
            if gram in column:
                matrix[row, column[gram]] = 1.0
    return matrix
