"""Every number this study reports.

Answers are decoded here, scored here, and representations compared here; the
official GQA evaluator is run here. Nothing outside this module measures the model —
the trainer adds only its own objective's terms (`lm_loss`, `align_loss` and the
learned `temperature`), which it forms and so alone can report. `Evaluator` is the
single door in: `measure` returns all of a batch's measurements, `report` all of an
accumulated epoch's.
"""

from __future__ import annotations

import json
import math
import re
import string
import subprocess
import sys
import tempfile
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from .data import Batch
from .pairwise import pairwise_cosine, pairwise_equal
from .types import VLM, VLMProcessor
from .utils import save_json

__all__ = [
    "MAX_STABILITY_SAMPLES",
    "normalize_answer",
    "exact_match_accuracy",
    "decode_answers",
    "cohens_d",
    "layer_names",
    "flatten_layers",
    "representation_stability",
    "evaluate_gqa",
    "Evaluator",
]

# The similarity matrix behind the stability metrics is quadratic in the sample count,
# so the epoch-level estimate is capped here rather than at each caller.
MAX_STABILITY_SAMPLES: Final = 4_000

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_MAX_ANSWER_TOKENS = 16
_STABILITY_KEYS = ("intra_sim", "inter_sim", "separation", "separation_d")


# ── Answers ───────────────────────────────────────────────────────────────────


def normalize_answer(answer: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(answer.lower().translate(_PUNCTUATION).split())


def exact_match_accuracy(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but {len(references)} references.")
    if not references:
        return {"accuracy": float("nan")}
    hits = sum(
        normalize_answer(p) == normalize_answer(r)
        for p, r in zip(predictions, references, strict=True)
    )
    return {"accuracy": hits / len(references)}


def decode_answers(vlm: VLM, processor: VLMProcessor, batch: Batch) -> list[str]:
    """Greedily generate an answer for each sample from its question prefix.

    The batch is right-padded — question then answer — because that is what the loss
    needs, but batched generation needs every sequence to end at the same column, so the
    questions are repacked left-padded to a common width.
    """
    input_ids   = batch["input_ids"]    # (B, T) right-padded question + answer
    prompt_lens = batch["prompt_lens"]  # (B,) question token count per sample

    width = max(prompt_lens)
    q_ids  = input_ids.new_full((len(prompt_lens), width), processor.pad_token_id)
    q_mask = input_ids.new_zeros((len(prompt_lens), width))
    for i, n in enumerate(prompt_lens):
        q_ids[i, width - n:]  = input_ids[i, :n]
        q_mask[i, width - n:] = 1

    output_ids = vlm.generate(
        input_ids=q_ids,
        attention_mask=q_mask,
        pixel_values=batch["pixel_values"],
        pixel_attention_mask=batch.get("pixel_attention_mask"),
        max_new_tokens=_MAX_ANSWER_TOKENS,
    )
    return processor.decode(output_ids[:, width:])


# ── Representations ───────────────────────────────────────────────────────────


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Standardised mean difference — scale-free, and so the one arms are compared on."""
    na, nb = a.numel(), b.numel()
    if na < 2 or nb < 2:
        return float("nan")
    pooled = (((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2)).sqrt()
    if pooled == 0:
        return float("nan")
    return ((a.mean() - b.mean()) / pooled).item()


def layer_names(representations: torch.Tensor, layers: Sequence[int] | None = None) -> list[str]:
    """What to call each slice of a `(N, L, H)` stack's layer axis — `L16`, `L20`, …

    `layers` is the model's own list, in the order it stacked them; without it the
    layers are named by position, which reads the same but cannot be trusted to be the
    block number.
    """
    depth = representations.size(1)
    if layers is None:
        return [f"L{i}" for i in range(depth)]
    if len(layers) != depth:
        raise ValueError(
            f"Representations carry {depth} layers but the model names {len(layers)}: {layers}."
        )
    return [f"L{layer}" for layer in layers]


def flatten_layers(by_layer: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-layer metric dicts as the one flat dict a training step reports.

    Each metric appears per layer as ``separation_d/L16``, and bare as the mean over the
    layers that defined it — one number to watch when several layers are constrained.
    A single layer gets the bare names alone: exactly what runs reported back when only
    one layer *could* be constrained, so those runs stay comparable.

    Undefined values are dropped, not averaged: a batch holding no same-signature pair
    (or no different-signature pair) leaves a layer's metrics NaN, and one such layer
    must not erase the layers that did measure something.
    """
    finite = {
        name: {metric: value for metric, value in scores.items() if math.isfinite(value)}
        for name, scores in by_layer.items()
    }
    if len(finite) == 1:
        return next(iter(finite.values()))

    means: dict[str, list[float]] = {}
    for scores in finite.values():
        for metric, value in scores.items():
            means.setdefault(metric, []).append(value)
    return {metric: sum(values) / len(values) for metric, values in means.items()} | {
        f"{metric}/{name}": value
        for name, scores in finite.items()
        for metric, value in scores.items()
    }


def representation_stability(
    representations: torch.Tensor, signatures: Sequence[Hashable]
) -> dict[str, float]:
    """How much more alike same-signature representations are than different ones.

    `separation` is the gap between the two mean cosine similarities and `separation_d`
    its Cohen's d, which is scale-free and so the one the arms are compared on. All four
    are NaN unless the sample holds a pair of each kind.
    """
    if representations.size(0) != len(signatures):
        raise ValueError(
            f"{representations.size(0)} representations but {len(signatures)} signatures."
        )

    sim = pairwise_cosine(representations)
    same = pairwise_equal(signatures, sim.device)
    off_diagonal = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)

    intra = sim[same & off_diagonal]
    inter = sim[~same & off_diagonal]
    if intra.numel() == 0 or inter.numel() == 0:
        return dict.fromkeys(_STABILITY_KEYS, float("nan"))

    intra_mean, inter_mean = intra.mean(), inter.mean()
    return {
        "intra_sim": intra_mean.item(),
        "inter_sim": inter_mean.item(),
        "separation": (intra_mean - inter_mean).item(),
        "separation_d": cohens_d(intra, inter),
    }


# ── Official GQA ──────────────────────────────────────────────────────────────

_EVAL_SCRIPT = Path(__file__).parent / "data" / "_gqa_eval.py"
_SCORE_LINE = re.compile(r"^(\w+): ([\d.]+)")
_SUBSET_LINE = re.compile(r"^Evaluating (\d+) of \d+ questions \((\d+) skipped")
_GROUP_TITLE = re.compile(r"^Accuracy / (.+):$")
_GROUP_ROW = re.compile(r"^ {2}(.+): ([\d.]+)% \((\d+) questions\)$")


def evaluate_gqa(
    predictions: dict[str, str],
    *,
    questions_path: str | Path,
    choices_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the official GQA evaluator and return everything it scores.

    The evaluator is a script; it runs as a subprocess with its output captured, so
    this prints nothing and only returns a dict keyed by lowercase metric name.
    Percentage metrics are in [0, 100]; ``distribution`` is a chi-square value (lower
    is better). The dict also carries ``scored``/``skipped``: how many questions were
    evaluated versus dropped for lacking a prediction.

    The evaluator additionally breaks accuracy down by question structure, semantics,
    reasoning-step count and word count; those land under ``accuracy_per`` as
    ``{group: {bucket: accuracy}}``. Accuracy against the number of reasoning steps
    is the compositional axis — the breakdown a procedural-alignment claim lives or
    dies on.

    ``choices_path`` adds ``validity``/``plausibility``, the only two scores that need
    the answer-option lists; GQA ships those for train/val alone, so on testdev it is
    simply omitted and the remaining scores are unaffected.
    """
    # The evaluator reads its predictions from a file, so hand it one and take it
    # away again — a directory rather than a NamedTemporaryFile because Windows
    # will not let a second process open a temp file we still hold.
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = Path(tmpdir) / "predictions.json"
        with predictions_path.open("w", encoding="utf-8") as f:
            json.dump(_official_format(predictions), f)
        proc = subprocess.run(
            [
                # -P keeps the script's own directory off sys.path. It sits inside
                # this package, and a module of ours that shares a name with a stdlib
                # one would otherwise shadow it as the subprocess starts up.
                sys.executable, "-P", str(_EVAL_SCRIPT),
                "--questions", str(questions_path),
                "--predictions", str(predictions_path),
                *(["--choices", str(choices_path)] if choices_path else []),
            ],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"GQA evaluation script failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    scores = _parse_scores(proc.stdout)
    if not scores:
        raise RuntimeError(f"No scores parsed from GQA evaluation output:\n{proc.stdout.strip()}")
    return scores | _parse_subset(proc.stdout) | {"accuracy_per": _parse_groups(proc.stdout)}


def _official_format(predictions: dict[str, str]) -> list[dict[str, str]]:
    return [{"questionId": qid, "prediction": answer} for qid, answer in predictions.items()]


def _parse_scores(output: str) -> dict[str, float]:
    matches = (_SCORE_LINE.match(line) for line in output.splitlines())
    return {m.group(1).lower(): float(m.group(2)) for m in matches if m}


def _parse_subset(output: str) -> dict[str, float]:
    """The ``Evaluating N of M questions (K skipped ...)`` line the script prints, so
    the subset a run was actually scored on is visible in the result.

    The script is this package's own copy at a fixed path and prints that line
    unconditionally, so it cannot merely be missing: its absence means the copy has
    been edited out of contract. Said out loud here rather than left to surface as a
    pair of keys quietly gone from every ``eval.json``.
    """
    for line in output.splitlines():
        if m := _SUBSET_LINE.match(line):
            return {"scored": float(m.group(1)), "skipped": float(m.group(2))}
    raise RuntimeError(
        f"No 'Evaluating N of M questions' line in GQA evaluation output:\n{output.strip()}"
    )


def _parse_groups(output: str) -> dict[str, dict[str, float]]:
    """The per-type accuracy tables the script prints after the scores: an
    ``Accuracy / <group>:`` heading followed by indented ``<bucket>: <pct>%
    (<n> questions)`` rows. Buckets carrying no questions never appear, so an empty
    group simply comes back empty.
    """
    groups: dict[str, dict[str, float]] = {}
    current: dict[str, float] | None = None
    for line in output.splitlines():
        if title := _GROUP_TITLE.match(line):
            current = groups.setdefault(title.group(1).replace(" ", "_"), {})
        elif (row := _GROUP_ROW.match(line)) and current is not None:
            current[row.group(1)] = float(row.group(2))
        elif not line.strip():
            current = None
    return groups


# ── The one door in ───────────────────────────────────────────────────────────


class Evaluator:
    """Every metric a run reports.

    `measure` returns all of a batch's metrics. On an evaluation pass it also keeps
    the predictions and representations that `report` turns into the epoch-level
    numbers; `reset` clears that accumulation between epochs.

    Without `gqa_questions_path` there is nothing to score the predictions against
    and `report` omits the official scores — but `save_predictions` still writes
    them, so the run can be scored offline later.
    """

    def __init__(
        self,
        vlm: VLM,
        processor: VLMProcessor,
        *,
        gqa_questions_path: Path | str | None = None,
        gqa_choices_path: Path | str | None = None,
        max_stability_samples: int = MAX_STABILITY_SAMPLES,
        seed: int = 0,
    ) -> None:
        self._vlm             = vlm
        self._processor       = processor
        self._questions_path  = Path(gqa_questions_path) if gqa_questions_path else None
        self._choices_path    = Path(gqa_choices_path)   if gqa_choices_path   else None
        self._max_samples     = max_stability_samples
        self._seed            = seed
        self._layers          = vlm.layers

        self._predictions: dict[str, str] = {}
        self._representations: list[torch.Tensor] = []
        self._signatures: list[str] = []
        self._question_ids: list[str] = []

    # ── measurement ───────────────────────────────────────────────────────────

    def measure(
        self, batch: Batch, representation: torch.Tensor, *, training: bool = False
    ) -> dict[str, float]:
        """All of this batch's metrics, from a `(B, L, H)` representation.

        A training step gets the representation metrics alone. Accuracy is
        generation-based, and generating dwarfs the step it would ride along with:
        ~16 decoder passes against one forward and one backward, each re-reading the
        whole prefix while gradient checkpointing holds the model in train mode with
        the KV cache off. It is recovered in eval anyway, where the number actually
        decides. An evaluation pass decodes, scores the answers against gold, and
        keeps what `report` needs.
        """
        # A batch holding no same-signature pair (or no different-signature pair) leaves
        # a layer's metrics undefined; `flatten_layers` drops those rather than let them
        # poison the epoch mean.
        metrics = flatten_layers(
            self._by_layer(
                representation,
                lambda hidden: representation_stability(hidden, batch["task_signatures"]),
            )
        )
        if training:
            return metrics

        answers = decode_answers(self._vlm, self._processor, batch)
        self._predictions.update(
            (qid, normalize_answer(answer))
            for qid, answer in zip(batch["question_ids"], answers, strict=True)
        )
        self._representations.append(representation.float().cpu())
        self._signatures += batch["task_signatures"]
        self._question_ids += batch["question_ids"]
        return metrics | exact_match_accuracy(answers, batch["answers"])

    # ── reporting ─────────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        """Everything scored over a whole accumulated evaluation epoch: the official
        GQA scores, and stability over far more samples than a single batch holds.
        """
        report: dict[str, Any] = {}
        if self._questions_path is not None:
            report["official_gqa"] = evaluate_gqa(
                self._predictions,
                questions_path=self._questions_path,
                choices_path=self._choices_path,
            )

        representations, signatures = self._subsample()
        # One layer reports flat, as it always has; several nest under their layer name
        # rather than averaging, because here there is room to read them apart.
        by_layer = self._by_layer(
            representations,
            lambda hidden: representation_stability(hidden, signatures),
        )
        report["representation_stability"] = (
            next(iter(by_layer.values())) if len(by_layer) == 1 else by_layer
        )
        return report

    def save_predictions(self, path: Path | str) -> None:
        """Write the predictions in the official GQA format, so any run can be
        re-scored and compared offline whether or not it could be scored now.
        """
        save_json(_official_format(self._predictions), path)

    def save_representations(self, path: Path | str) -> None:
        """Persist the accumulated representations, with the signature and question id of
        each.

        This study is *about* these vectors, and a run that keeps only what it reduced
        them to keeps one number. Every open question about the geometry — centre them and
        measure again; is the separation procedure or phrasing; what does a held-out
        signature look like — is then answerable from this file on a laptop, and none of
        them answerable without it short of taking the GPU back.

        They are float32 on purpose: the raw separation between same- and
        different-signature cosines is around 0.02, which is the scale of float16's
        rounding error, and half precision would put the noise floor above the effect.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "representations": torch.cat(self._representations),  # (N, L, H) float32
                # Which block each slice of the layer axis came from. Without it the
                # axis is anonymous, and a file that constrained layers 8 and 16 is
                # indistinguishable from one that constrained 16 and 24.
                "layers": list(self._layers) if self._layers is not None else None,
                "signatures": self._signatures,
                "question_ids": self._question_ids,
            },
            path,
        )

    def reset(self) -> None:
        self._predictions.clear()
        self._representations.clear()
        self._signatures.clear()
        self._question_ids.clear()

    # ── internals ─────────────────────────────────────────────────────────────

    def _by_layer(
        self,
        representations: torch.Tensor,
        score: Callable[[torch.Tensor], dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """`score` applied to each layer of a `(N, L, H)` stack, keyed by layer name.

        Every representation metric is defined over a set of vectors, and the layers are
        separate sets — so each is scored on its own, whatever the caller then does with
        the results.
        """
        return {
            name: score(hidden)
            for name, hidden in zip(
                layer_names(representations, self._layers),
                representations.unbind(dim=1),
                strict=True,
            )
        }

    def _subsample(self) -> tuple[torch.Tensor, list[str]]:
        """A seeded subsample of the accumulated representations, capped at
        ``max_stability_samples``.
        """
        representations = torch.cat(self._representations)
        signatures = self._signatures
        if len(signatures) > self._max_samples:
            generator = torch.Generator().manual_seed(self._seed)
            chosen = torch.randperm(len(signatures), generator=generator)[: self._max_samples]
            representations = representations[chosen]
            signatures = [signatures[i] for i in chosen.tolist()]
        return representations, signatures
