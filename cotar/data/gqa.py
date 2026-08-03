from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from train4all.utils import UnifiedLogger

from ..types import Encoding, VLMProcessor
from ..utils import load_json

__all__ = [
    "task_signature",
    "GQASample",
    "GQADataset",
    "MPerSignatureSampler",
    "Batch",
    "build_gqa_dataloader",
]


def task_signature(program: list[dict[str, Any]]) -> str:
    """The operator sequence of a functional program, e.g. `select > relate > query`.

    Arguments are dropped on purpose: matching on them makes a positive pair almost
    the same question, which is trivially alike and says nothing about shared
    procedure.
    """
    return " > ".join(step["operation"] for step in program) or "none"


@dataclass(slots=True)
class GQASample:
    image: Image.Image
    question: str
    answer: str
    program: list[dict[str, Any]]
    question_id: str

    @property
    def signature(self) -> str:
        return task_signature(self.program)


class GQADataset(Dataset[GQASample]):
    def __init__(
        self,
        questions_path: str | Path,
        images_dir: str | Path,
        *,
        require_program: bool = False,
        limit: int | None = None,
        group_by_signature: bool = False,
        min_samples_per_signature: int = 2,
        logger: UnifiedLogger | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        raw: dict[str, dict[str, Any]] = load_json(questions_path)
        # Enumerate the image directory once instead of probing each question's
        # image: train_balanced alone asks 943k questions about 72k images, so a
        # stat per question is ~943k syscalls to learn what one listing of the
        # ~149k files already says.
        available = {path.stem for path in self.images_dir.iterdir()}

        ordered: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        skipped: Counter[str] = Counter()
        for question_id, entry in raw.items():
            program = entry.get("semantic", [])
            if require_program and not program:
                skipped["no program"] += 1
                continue
            if entry["imageId"] not in available:
                skipped["missing images"] += 1
                continue
            record = {
                "question_id": question_id,
                "image_id": entry["imageId"],
                "question": entry["question"],
                "answer": entry["answer"],
                "program": program,
            }
            if group_by_signature:
                groups[task_signature(program)].append(record)
            else:
                # Ungrouped, ``limit`` is simply the leading slice of the file — the
                # faithful order, and the one evaluation reports against.
                ordered.append(record)
                if limit is not None and len(ordered) >= limit:
                    break

        if group_by_signature:
            # Grouped, ``limit`` is applied to the whole file rather than to a leading
            # slice. A slice long enough to hold `limit // min` distinct signatures would
            # have to run far past `limit` rows — one GQA signature alone is a quarter of
            # them — so capping the complete picture is both simpler and more
            # representative of the signature distribution.
            self.samples = self._cap_signature_groups(groups, limit, min_samples_per_signature)
            singletons = sum(
                len(members)
                for members in groups.values()
                if len(members) < min_samples_per_signature
            )
            if logger and singletons:
                logger.log(
                    f"Dropped {singletons} samples without {min_samples_per_signature}+ "
                    f"task-signature siblings.",
                    level="info",
                )
        else:
            self.samples = ordered

        if logger:
            for reason, count in skipped.items():
                logger.log(f"Skipped {count} samples ({reason}).", level="warn")

    @staticmethod
    def _cap_signature_groups(
        groups: dict[str, list[dict[str, Any]]],
        limit: int | None,
        min_samples_per_signature: int,
    ) -> list[dict[str, Any]]:
        """Select usable samples from signature groups, honouring ``limit`` exactly.

        Groups smaller than ``min_samples_per_signature`` cannot form positive pairs and
        are dropped. The budget is then *spread* over the surviving signatures rather than
        poured into the first ones: as many signatures as it can afford are opened at
        ``min_samples_per_signature`` each, and the remainder tops them up evenly. Filling
        greedily instead would hand a small ``limit`` almost entirely to the largest
        signature — on GQA one signature is a quarter of the data — leaving
        ``MPerSignatureSampler`` fewer distinct signatures than a batch needs, and a
        contrastive batch is built out of *classes*. ``limit`` is met exactly whenever the
        data allows, so the step count stays predictable; only a genuine shortage of usable
        samples falls short.
        """
        valid = [
            members
            for members in groups.values()
            if len(members) >= min_samples_per_signature
        ]
        if limit is None:
            return [record for members in valid for record in members]

        opened = valid[: limit // min_samples_per_signature]
        counts = [min_samples_per_signature] * len(opened)

        # Even top-up: raise the open signatures towards a common level, each capped by
        # its own size, until the budget is spent or every one of them is exhausted.
        budget = limit - sum(counts)
        while budget > 0:
            room = [i for i, members in enumerate(opened) if counts[i] < len(members)]
            if not room:
                break
            share = max(1, budget // len(room))
            for i in room:
                take = min(share, len(opened[i]) - counts[i], budget)
                counts[i] += take
                budget -= take
                if budget == 0:
                    break

        return [record for members, count in zip(opened, counts, strict=True)
                for record in members[:count]]

    @property
    def signatures(self) -> list[str]:
        """One task signature per sample, in dataset order — what the sampler groups by."""
        return [task_signature(record["program"]) for record in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GQASample:
        record = self.samples[idx]
        return GQASample(
            image=Image.open(self.images_dir / f"{record['image_id']}.jpg").convert("RGB"),
            question=record["question"],
            answer=record["answer"],
            program=record["program"],
            question_id=record["question_id"],
        )


class MPerSignatureSampler(Sampler[list[int]]):
    """Batches of M samples drawn from each of a few randomly chosen signatures.

    A contrastive batch is built out of classes: too few distinct signatures and it
    has too few negatives, too few samples per signature and it has no positives.
    """

    def __init__(
        self,
        signatures: Sequence[str],
        signatures_per_batch: int,
        samples_per_signature: int = 2,
        *,
        min_samples_per_signature: int = 2,
        num_batches: int | None = None,
        seed: int | None = 0,
    ) -> None:
        if signatures_per_batch < 1:
            raise ValueError(f"signatures_per_batch must be >= 1, got {signatures_per_batch}.")
        if samples_per_signature < 2:
            raise ValueError(f"samples_per_signature must be >= 2, got {samples_per_signature}.")
        if min_samples_per_signature < 2:
            raise ValueError(
                f"min_samples_per_signature must be >= 2, got {min_samples_per_signature}."
            )

        indices_by_signature: dict[str, list[int]] = defaultdict(list)
        for index, signature in enumerate(signatures):
            indices_by_signature[signature].append(index)
        self.pools = [
            indices
            for indices in indices_by_signature.values()
            if len(indices) >= min_samples_per_signature
        ]
        if len(self.pools) < signatures_per_batch:
            raise ValueError(
                f"need >= signatures_per_batch={signatures_per_batch} signatures with "
                f"{min_samples_per_signature}+ samples, got {len(self.pools)}."
            )

        self.signatures_per_batch = signatures_per_batch
        self.samples_per_signature = samples_per_signature
        self.seed = seed if seed is not None else random.Random().randrange(2**63)
        num_samples = sum(len(pool) for pool in self.pools)
        self.num_batches = num_batches or max(
            1, num_samples // (signatures_per_batch * samples_per_signature)
        )
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        k = self.samples_per_signature
        for _ in range(self.num_batches):
            batch: list[int] = []
            for pool in rng.sample(self.pools, self.signatures_per_batch):
                take = min(k, len(pool))
                batch += rng.sample(pool, take) + rng.choices(pool, k=k - take)
            yield batch


class _SampleMeta(TypedDict):
    answers: list[str]
    question_ids: list[str]
    task_signatures: list[str]


class Batch(Encoding, _SampleMeta):
    pass


class _GQACollator:
    def __init__(self, processor: VLMProcessor, *, with_labels: bool) -> None:
        self.processor = processor
        self.with_labels = with_labels

    def __call__(self, batch: list[GQASample]) -> Batch:
        questions = [sample.question for sample in batch]
        images = [sample.image for sample in batch]
        answers = [sample.answer for sample in batch]
        encoding = self.processor.encode(
            questions,
            images,
            answers if self.with_labels else None,
            padding_side="right" if self.with_labels else "left",
        )
        return {
            **encoding,
            "answers": answers,
            "question_ids": [sample.question_id for sample in batch],
            "task_signatures": [sample.signature for sample in batch],
        }


def build_gqa_dataloader(
    questions_path: str | Path,
    images_dir: str | Path,
    processor: VLMProcessor,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    drop_last: bool = False,
    group_by_signature: bool = False,
    samples_per_signature: int = 2,
    min_samples_per_signature: int = 2,
    with_labels: bool = False,
    require_program: bool = False,
    limit: int | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int | None = 0,
    logger: UnifiedLogger | None = None,
) -> DataLoader[GQASample]:
    if group_by_signature and (shuffle or drop_last):
        raise ValueError(
            "group_by_signature forms batches via a custom batch_sampler, which "
            "owns ordering and batch sizing; leave shuffle and drop_last False."
        )
    dataset = GQADataset(
        questions_path, images_dir,
        require_program=require_program,
        limit=limit,
        group_by_signature=group_by_signature,
        min_samples_per_signature=min_samples_per_signature,
        logger=logger,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": _GQACollator(processor, with_labels=with_labels),
        "persistent_workers": num_workers > 0,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
    }

    if group_by_signature:
        sampler = MPerSignatureSampler(
            dataset.signatures,
            signatures_per_batch=max(1, batch_size // samples_per_signature),
            samples_per_signature=samples_per_signature,
            min_samples_per_signature=min_samples_per_signature,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **loader_kwargs
    )
