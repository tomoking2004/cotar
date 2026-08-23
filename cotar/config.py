from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["cfg"]

# Two roots, because the project's files divide cleanly in two.
#
# This checkout holds what travels: the code, the documents, and the snapshots and
# analyses they cite. Finding those relative to the package means a clone reads its own
# results wherever it sits, and no script has to rediscover where the repository is.
REPO_ROOT = Path(__file__).resolve().parent.parent

# The working area holds what must not travel: the dataset, and the run directories with
# their weights. Kept outside the checkout because it is synced, and tens of gigabytes do
# not belong in a synced folder.
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
    repo_root: Path = REPO_ROOT
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
    def snapshots_root(self) -> Path:
        # In the checkout, not beside `runs_root`: a snapshot is a run without its
        # weights, small enough to commit, and committing it is the point — the results
        # then travel with the code that produced them.
        return self.repo_root / "snapshots"

    @property
    def analyses_root(self) -> Path:
        return self.repo_root / "analyses"

    @property
    def gqa(self) -> GQAPaths:
        return GQAPaths(self.datasets_root / "gqa")


cfg = Config()
