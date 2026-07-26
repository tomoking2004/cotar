from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["cfg"]

WORK_ROOT = Path.home() / "TMU" / "Master" / "study" / "cotar"


@dataclass(frozen=True)
class GQAPaths:
    root: Path

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def train_questions(self) -> Path:
        return self.root / "questions" / "train_balanced_questions.json"

    @property
    def val_questions(self) -> Path:
        return self.root / "questions" / "val_balanced_questions.json"

    @property
    def val_choices(self) -> Path:
        return self.root / "eval" / "val_choices.json"

    @property
    def testdev_questions(self) -> Path:
        return self.root / "questions" / "testdev_balanced_questions.json"


@dataclass(frozen=True)
class Config:
    seed: int | None = 42
    work_root: Path = WORK_ROOT

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_workers(self) -> int:
        # Workers inherit the parent's parsed questions copy-on-write, so this is
        # bounded by host RAM rather than by cores — 12 exhausted a 30GB machine.
        return 6 if torch.cuda.is_available() else 0

    @property
    def datasets_root(self) -> Path:
        return self.work_root / "datasets"

    @property
    def runs_root(self) -> Path:
        return self.work_root / "runs"

    @property
    def gqa(self) -> GQAPaths:
        return GQAPaths(self.datasets_root / "gqa")


cfg = Config()
