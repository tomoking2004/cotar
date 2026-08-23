"""The experiment the study reports, and how to read back what it left behind.

Every analysis reads the same nine runs — three arms at three seeds, filed under one
timestamp. Naming them here means the scripts cannot drift apart about which experiment
is being reported, and pointing them all at a later one is a single edit.

The readers below are the only door onto a run's files. Going through them keeps the
layout of a snapshot in one place, and lets the check that the runs line up row for row
happen on every read instead of being remembered by each caller.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from ..config import cfg
from ..training import ARMS, Arm
from ..utils import load_json

__all__ = [
    "ARMS",
    "SEEDS",
    "TIMESTAMP",
    "analysis_path",
    "eval_report",
    "predictions",
    "reported_accuracy",
    "representations",
    "run_config",
    "run_id",
    "training_batches",
    "training_duration",
]

TIMESTAMP = "20260727-002344"
SEEDS = (42, 43, 44)


def run_id(arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP) -> str:
    """The directory name one run left behind.

    `variant` names a configuration other than the reported one — a sweep's alignment
    weight or layer — and is empty for the nine runs the study reports. `timestamp` says
    which experiment: a sweep is filed under its own, and its runs are still comparable
    to the reported ones because a seed fixes the initialisation and the batch order,
    not the day the run happened.

    Composed here rather than shared with `training.run`, which spells the same name for
    the directory it is about to write. The two look like one convention duplicated, but
    they state different facts: that one is what this checkout writes next, and this one
    is what already sits on disk — which has to stay readable after the writing side
    changes, because the runs written under the old spelling do not rename themselves. A
    shared builder would also have to live in the machinery, and the machinery names no
    experiment, while `arm`, `seed` and `variant` are the experiment's own vocabulary.
    """
    return "_".join(part for part in (timestamp, variant, arm, f"seed{seed}") if part)


def _snapshot(arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP) -> Path:
    return cfg.snapshots_root / run_id(arm, seed, variant, timestamp)


_order: tuple[list[str], list[str]] | None = None


def representations(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> tuple[torch.Tensor, list[str], list[str]]:
    """A run's saved testdev representations, with the signature and id of each row.

    Every run evaluated testdev in the same order, so a row means the same question in
    all nine. That is checked here rather than assumed: the arms are compared row by row,
    and an order that differed would compare different questions without saying so. The
    check spans variants too, which is what lets a sweep's runs be read beside the
    reported ones.
    """
    global _order
    saved = torch.load(
        _snapshot(arm, seed, variant, timestamp) / "metrics" / "representations.pt",
        weights_only=False,
    )
    # `(N, L, H)` over the constrained layers; the study constrains one, and that is the
    # layer every analysis reads.
    features = saved["representations"][:, 0, :]
    signatures, question_ids = saved["signatures"], saved["question_ids"]
    if _order is None:
        _order = (signatures, question_ids)
    elif (signatures, question_ids) != _order:
        raise SystemExit(
            f"{run_id(arm, seed, variant, timestamp)}: testdev is in a different order "
            f"from the first run read, so its rows do not line up with the others'."
        )
    return features, signatures, question_ids


def predictions(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> dict[str, str]:
    """What a run answered, by question id."""
    filed = load_json(_snapshot(arm, seed, variant, timestamp) / "metrics" / "predictions.json")
    return {entry["questionId"]: entry["prediction"] for entry in filed}


def eval_report(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> dict[str, Any]:
    """Everything a run's final evaluation recorded: its identity, the batch-averaged
    test metrics, the official GQA scores, and the epoch-level representation geometry.
    """
    return load_json(_snapshot(arm, seed, variant, timestamp) / "metrics" / "eval.json")


def run_config(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> dict[str, Any]:
    """The arguments a run's trainer was constructed with — its `config.json`.

    Distinct from `eval_report`, which records what the run *was* and what it *measured*.
    Settings the trainer was merely handed live only here: the batch size, in particular,
    which decides the value a collapsed alignment loss would take.
    """
    return load_json(_snapshot(arm, seed, variant, timestamp) / "config.json")


def reported_accuracy(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> float:
    """The accuracy the official GQA evaluator gave the run, as a percentage."""
    return eval_report(arm, seed, variant, timestamp)["official_gqa"]["accuracy"]


def training_batches(arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP) -> int:
    """How many batches the training phase actually stepped through.

    Counted from the recorded steps rather than derived from the dataset size, because
    what an epoch comes to is the sampler's to decide — it draws by signature, and the
    count is the one thing that says how many draws that came to.
    """
    steps = load_json(_snapshot(arm, seed, variant, timestamp) / "metrics" / "step_metrics.json")
    return len(next(iter(steps.values()))["train"])


_DURATION = re.compile(r"Training completed\. Duration: (\d+):(\d\d):(\d\d)")


def training_duration(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> timedelta:
    """How long the run's training phase took, as its own log recorded it.

    Read from the log rather than timed again: this is a fact about the machine that
    ran it, and that machine is not the one reading this back.
    """
    log = (_snapshot(arm, seed, variant, timestamp) / "log.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    if not (m := _DURATION.search(log)):
        raise SystemExit(
            f"{run_id(arm, seed, variant, timestamp)}: no training duration in log.txt."
        )
    hours, minutes, seconds = (int(g) for g in m.groups())
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def analysis_path(script: str) -> Path:
    """Where an analysis files its measurements: `analyses/`, under its own name.

    Derived from the script rather than spelled out in it, so one analysis can only ever
    produce the one file that carries its name.
    """
    return cfg.analyses_root / f"{Path(script).stem.replace('_', '-')}.json"
