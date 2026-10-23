"""How far a representation moves when the image is perturbed, read against how far an
unrelated image would move it — the measurement context.md §8.1 asks for.

Wani et al. (arXiv:2602.06652) measure the cosine distance 1 − cos between the hidden state
of a base input and that of a perturbed one, and read it against a control: the distances
between the hidden states of unrelated inputs. Drift is reported as a percentage of the
control's mean, and the two distributions are compared by Cohen's d, so a drift well below
the control (d far below zero) is a local deformation and a drift that overlaps it is a move
to where unrelated inputs sit. That form is kept here, and so are their two output-side
rates: the share of perturbed pairs whose predicted answer changed (instance flip rate), and
the share of base samples with at least one such change across a family's variants (image
vulnerability — here per question, since the base sample of this study is a question about
an image).

**Two hook points, from one forward pass.** The paper reads the final layer at the last
context token. This study's question is about the site the alignment loss touched, so that
layer is read at the same position, and the final layer is read beside it as the point of
contact with the paper. Both come out of a single pass, because the model pools every
layer it is built to read.

**One pass per condition, and the base pass is the anchor.** Every variant and the control
is one pass over the report set, in file order, with the image transformed before the
processor sees it. Distances are taken row by row against the base pass, so the rows are
checked to be the same questions in the same order before any are compared.

**Generation is greedy and the answers are matched exactly.** Flips compare the perturbed
prediction with the base prediction after the same normalisation the training-time
evaluator applies; the accuracy beside them is exact match against the gold answer, not the
official GQA evaluator's score, which the base pass of each run already has on record.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import ImageTransform, build_gqa_dataloader
from ..models import SmolVLM
from ..training import cohens_d, decode_answers, normalize_answer
from ..types import VLMProcessor
from .perturbations import FAMILIES, Variant, image_swap, other_item_partners

__all__ = [
    "ANY",
    "CONTROLS",
    "Pass",
    "RunDrift",
    "cosine_distance",
    "family_names",
    "measure_run",
    "run_pass",
    "summarize_run",
]

# The union of every family, reported as a family of its own, as the paper's "Any" row.
ANY = "any"

# The two readings of "unrelated": the same question of another image, and another
# question of another image (see `cotar.analysis.perturbations`).
CONTROLS = ("other_image", "other_item")


@dataclass(frozen=True)
class Pass:
    """What one pass over the report set leaves: a vector per question per layer, and
    the answer the model gave — `None` when the pass did not generate."""

    question_ids: list[str]
    image_ids: list[str]
    answers: list[str]
    """The gold answers, normalised."""
    representations: torch.Tensor
    """`(N, L, H)` float32, on the CPU."""
    predictions: list[str] | None
    """Greedy answers, normalised."""


@dataclass(frozen=True)
class RunDrift:
    """Everything one run's passes measured, question by question."""

    layers: tuple[int, ...]
    question_ids: list[str]
    image_ids: list[str]
    base_correct: torch.Tensor
    """`(N,)` bool — whether the base pass answered correctly."""
    distances: dict[str, torch.Tensor]
    """Variant or control name → `(N, L)` cosine distance to the base representation."""
    flipped: dict[str, torch.Tensor]
    """Variant name → `(N,)` bool — whether the predicted answer changed."""
    correct: dict[str, torch.Tensor]
    """Variant name → `(N,)` bool — whether the perturbed pass answered correctly."""


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`1 − cos` between matching rows of two `(N, L, H)` stacks, as `(N, L)`."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {tuple(a.shape)} against {tuple(b.shape)}.")
    return 1 - F.cosine_similarity(a.float(), b.float(), dim=-1)


def run_pass(
    vlm: SmolVLM,
    processor: VLMProcessor,
    questions_path: str | Path,
    images_dir: str | Path,
    *,
    transform: ImageTransform | None = None,
    generate: bool = True,
    batch_size: int,
    num_workers: int,
    limit: int | None,
) -> Pass:
    """One pass over the report set with the images transformed, in file order.

    The loader is right-padded with labels because that is what gives it the prompt
    lengths the model pools at; the labels themselves are not handed to the model, so no
    loss is formed. Generation repacks the questions left-padded on its own.
    """
    loader: DataLoader[Any] = build_gqa_dataloader(
        questions_path,
        images_dir=images_dir,
        processor=processor,
        batch_size=batch_size,
        with_labels=True,
        limit=limit,
        transform=transform,
        num_workers=num_workers,
    )
    representations: list[torch.Tensor] = []
    question_ids: list[str] = []
    answers: list[str] = []
    predictions: list[str] = []
    for batch in loader:
        on_device = {
            key: value.to(vlm.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            out = vlm(
                input_ids=on_device["input_ids"],
                attention_mask=on_device["attention_mask"],
                pixel_values=on_device["pixel_values"],
                pixel_attention_mask=on_device.get("pixel_attention_mask"),
                prompt_lens=batch["prompt_lens"],
            )
            if out["representation"] is None:
                raise RuntimeError("the model produced no representation: build it with layers.")
            representations.append(out["representation"].float().cpu())
            if generate:
                answered = decode_answers(vlm, processor, on_device)
                predictions += [normalize_answer(a) for a in answered]
        question_ids += batch["question_ids"]
        answers += [normalize_answer(a) for a in batch["answers"]]

    # The dataset lists what the loader iterated, in the same order; the image ids are
    # read from it because a batch carries only what the trainer needs.
    records = loader.dataset.samples  # type: ignore[attr-defined]
    if [r["question_id"] for r in records] != question_ids:
        raise RuntimeError("the loader did not yield the dataset's questions in order.")
    return Pass(
        question_ids=question_ids,
        image_ids=[r["image_id"] for r in records],
        answers=answers,
        representations=torch.cat(representations),
        predictions=predictions if generate else None,
    )


def measure_run(
    vlm: SmolVLM,
    processor: VLMProcessor,
    questions_path: str | Path,
    images_dir: str | Path,
    variants: Sequence[Variant],
    *,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    progress: Callable[[str], None] | None = None,
) -> RunDrift:
    """Every pass one run needs, and the distances and flips between them.

    The base pass comes first and anchors the rest; the image-swap control is a pass of
    its own without generation; the other-item control costs no pass, being a distance
    between two base vectors.
    """
    if vlm.layers is None:
        raise ValueError("the model reads no layer: build it with the layers to measure.")

    def passed(name: str, transform: ImageTransform | None, generate: bool) -> Pass:
        started = time.perf_counter()
        result = run_pass(
            vlm, processor, questions_path, images_dir,
            transform=transform, generate=generate,
            batch_size=batch_size, num_workers=num_workers, limit=limit,
        )
        if progress is not None:
            progress(f"{name:>18}  {len(result.question_ids):>6,} questions  "
                     f"{time.perf_counter() - started:6.1f}s")
        return result

    base = passed("base", None, generate=True)
    if base.predictions is None:
        raise AssertionError("the base pass generates.")
    gold = base.answers
    base_predictions = base.predictions
    base_correct = torch.tensor([p == g for p, g in zip(base_predictions, gold, strict=True)])

    distances: dict[str, torch.Tensor] = {}
    flipped: dict[str, torch.Tensor] = {}
    correct: dict[str, torch.Tensor] = {}
    for variant in variants:
        perturbed = passed(variant.name, variant.transform, generate=True)
        if perturbed.question_ids != base.question_ids:
            raise RuntimeError(f"{variant.name}: the pass saw different questions from the base.")
        if perturbed.predictions is None:
            raise AssertionError("a variant pass generates.")
        distances[variant.name] = cosine_distance(perturbed.representations, base.representations)
        flipped[variant.name] = torch.tensor(
            [p != b for p, b in zip(perturbed.predictions, base_predictions, strict=True)]
        )
        correct[variant.name] = torch.tensor(
            [p == g for p, g in zip(perturbed.predictions, gold, strict=True)]
        )

    swapped = passed(
        "other_image", image_swap(images_dir, base.image_ids), generate=False
    )
    if swapped.question_ids != base.question_ids:
        raise RuntimeError("other_image: the control pass saw different questions from the base.")
    distances["other_image"] = cosine_distance(swapped.representations, base.representations)

    partners = other_item_partners(base.question_ids, base.image_ids)
    distances["other_item"] = cosine_distance(
        base.representations[partners], base.representations
    )

    return RunDrift(
        layers=vlm.layers,
        question_ids=base.question_ids,
        image_ids=base.image_ids,
        base_correct=base_correct,
        distances=distances,
        flipped=flipped,
        correct=correct,
    )


def family_names(variants: Sequence[Variant], family: str) -> list[str]:
    """The variants a family is made of — every variant for `ANY`."""
    return [v.name for v in variants if family == ANY or v.family == family]


def _against(drift: torch.Tensor, control: torch.Tensor) -> dict[str, float]:
    """Drift read against one control: as a percentage of the control's mean, with the
    standard deviation of that percentage over the pairs, and as Cohen's d between the
    two distributions (negative when drift is the smaller)."""
    scale = 100 / control.mean().item()
    return {
        "ratio_mean": drift.mean().item() * scale,
        "ratio_sd": drift.std().item() * scale,
        "cohens_d": cohens_d(drift, control),
    }


def summarize_run(run: RunDrift, variants: Sequence[Variant]) -> dict[str, Any]:
    """One run's numbers, per family and per layer, with the controls they were read against.

    A family's pairs are its variants' pairs pooled: eight shifts give eight pairs per
    question, one rescale gives one. The flip rate is over those pairs; the vulnerability
    is over questions, counting a question once however many of its variants flipped.
    """
    layers = [str(layer) for layer in run.layers]
    controls = {
        control: {
            layer: {
                "mean": run.distances[control][:, i].mean().item(),
                "sd": run.distances[control][:, i].std().item(),
            }
            for i, layer in enumerate(layers)
        }
        for control in CONTROLS
    }

    families: dict[str, dict[str, Any]] = {}
    for family in (*FAMILIES, ANY):
        names = family_names(variants, family)
        if not names:
            continue
        flips = torch.stack([run.flipped[n] for n in names])      # (k, N)
        hits = torch.stack([run.correct[n] for n in names])       # (k, N)
        by_layer: dict[str, Any] = {}
        for i, layer in enumerate(layers):
            drift = torch.cat([run.distances[n][:, i] for n in names])
            unchanged = drift[~flips.flatten()]
            other_image = run.distances["other_image"][:, i]
            by_layer[layer] = {
                "drift_mean": drift.mean().item(),
                "drift_sd": drift.std().item(),
                "other_image": _against(drift, other_image),
                "other_item": _against(drift, run.distances["other_item"][:, i]),
                "unchanged": {
                    "pairs": int(unchanged.numel()),
                    "ratio_mean": (
                        unchanged.mean().item() * 100 / other_image.mean().item()
                        if unchanged.numel() else float("nan")
                    ),
                },
            }
        families[family] = {
            "variants": names,
            "pairs": int(flips.numel()),
            "flip_rate": flips.float().mean().item(),
            "vulnerability": flips.any(dim=0).float().mean().item(),
            "accuracy": hits.float().mean().item(),
            "by_layer": by_layer,
        }

    return {
        "layers": list(run.layers),
        "questions": len(run.question_ids),
        "base_accuracy": run.base_correct.float().mean().item(),
        "controls": controls,
        "families": families,
    }
