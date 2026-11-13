"""Where the study's files sit, named once as `cfg`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["cfg"]

# Two roots, because the project's files divide cleanly in two.
#
# This checkout holds what travels: the code, the documents, and the snapshots and
# analyses they cite. Finding those relative to the package means a clone reads its own
# results wherever it sits, and no script has to rediscover where the repository is.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# The working area holds what must not travel: the dataset, and the run directories with
# their weights — tens of gigabytes. It is a fixed place under the home directory, not
# found from the package, so a clone elsewhere reads the same data. Where the checkout
# itself sits at that place, its `.gitignore` keeps datasets/ and runs/ out of version
# control.
_WORK_ROOT = Path.home() / "TMU" / "Master" / "study" / "cotar"

# runs/ alone can also live on removable media: a USB copy of the lab machine's runs/,
# carried over for the analyses that need the weights this checkout does not hold.
# `run_roots` (below) lists it after the local runs/, and readers probe it per run.
# $COTAR_RUNS_ROOT names a copy explicitly for when the drive letter is not E:; the
# datasets stay put either way.
_USB_RUNS_ROOT = Path("E:/runs")


@dataclass(frozen=True)
class GQAPaths:
    """The files of one GQA download, under its `root`."""

    root: Path

    @property
    def images(self) -> Path:
        """The image directory: one `<imageId>.jpg` per image."""
        return self.root / "images"

    @property
    def train_questions(self) -> Path:
        """The balanced training questions."""
        return self.root / "questions" / "train_balanced_questions.json"

    @property
    def val_questions(self) -> Path:
        """The balanced validation questions."""
        return self.root / "questions" / "val_balanced_questions.json"

    @property
    def testdev_questions(self) -> Path:
        """The balanced test-dev questions: the report set."""
        return self.root / "questions" / "testdev_balanced_questions.json"


@dataclass(frozen=True)
class Config:
    """The study's directories, and the defaults that depend on the machine."""

    repo_root: Path = _REPO_ROOT
    work_root: Path = _WORK_ROOT

    @property
    def device(self) -> str:
        """`cuda` where a GPU is visible, `cpu` otherwise."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_workers(self) -> int:
        """How many worker processes load data: six with a GPU, none (the main process) without."""
        # Workers inherit the parent's parsed questions copy-on-write, so this is
        # bounded by host RAM rather than by cores — 12 exhausted a 30GB machine.
        return 6 if torch.cuda.is_available() else 0

    @property
    def datasets_root(self) -> Path:
        """Where the datasets sit, in the working area."""
        return self.work_root / "datasets"

    @property
    def runs_root(self) -> Path:
        """Where a new run is written: always the local working area.

        Reading an already-trained run is a different question — it may sit on a carried
        copy instead — and goes through `run_roots`, never through this.
        """
        return self.work_root / "runs"

    @property
    def run_roots(self) -> tuple[Path, ...]:
        """Every root an already-trained run may be read from, most trusted first.

        An explicit $COTAR_RUNS_ROOT first, then the local runs/, then a USB copy mounted
        at E:/runs. Only roots that exist are listed, and readers probe them per run — a
        decision made once for the whole list would let a local runs/ holding only debug
        runs shadow the USB copy that holds the reported nine.
        """
        override = os.environ.get("COTAR_RUNS_ROOT")
        candidates = [Path(override)] if override else []
        candidates += [self.work_root / "runs", _USB_RUNS_ROOT]
        return tuple(root for root in candidates if root.exists())

    @property
    def snapshots_root(self) -> Path:
        """Where the snapshots sit: in the checkout, not beside `runs_root`.

        A snapshot is a run without its weights, small enough to commit, and committing it
        is the point — the results then travel with the code that produced them.
        """
        return self.repo_root / "snapshots"

    @property
    def analyses_root(self) -> Path:
        """Where the analyses file what they measure."""
        return self.repo_root / "analyses"

    @property
    def gqa(self) -> GQAPaths:
        """The GQA files, under the datasets root."""
        return GQAPaths(self.datasets_root / "gqa")


cfg = Config()
