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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from ..config import cfg
from ..training import ARMS, Arm
from ..utils import load_json

__all__ = [
    "ARMS",
    "CHECKPOINT_NAME",
    "MODEL",
    "SEEDS",
    "TIMESTAMP",
    "Arm",
    "Weights",
    "analysis_path",
    "checkpoint_path",
    "constrained_layer",
    "eval_report",
    "predictions",
    "reported_accuracy",
    "representations",
    "require_checkpoints",
    "run_config",
    "run_id",
    "training_batches",
    "training_duration",
    "weights",
]

TIMESTAMP       = "20260829-134859"
SEEDS           = (42, 43, 44)
# The base checkpoint the reported runs were trained from. Asserted here rather than read
# back, so the analyses need no checkpoint on disk to know it — `weights` checks it against
# whatever a checkpoint *does* record, and stays silent where nothing was recorded.
MODEL           = "HuggingFaceTB/SmolVLM-500M-Instruct"
# `trainer.test(..., use_best=True)` scored the runs with this checkpoint, so this is the
# one whose weights belong beside the reported numbers.
CHECKPOINT_NAME = "best.pth"


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


def constrained_layer(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> tuple[int, int]:
    """Which layer a run constrained, and how wide that layer's vector is.

    `representations` drops the layer axis, because the study constrains one layer and
    every analysis reads it. Its *identity* is what this returns instead: the document
    names the layer in every claim it makes, and a reader with the checkout has nowhere
    else to confirm it — the trainer wrote it into the checkpoint, and the checkpoints
    do not travel. Runs that constrained more than one are refused rather than reported
    by their first, since nothing downstream would know which layer it had been handed.
    """
    saved = torch.load(
        _snapshot(arm, seed, variant, timestamp) / "metrics" / "representations.pt",
        weights_only=False,
    )
    layers = list(saved["layers"])
    if len(layers) != 1:
        raise SystemExit(
            f"{run_id(arm, seed, variant, timestamp)}: constrained {len(layers)} layers "
            f"({layers}), but the study's analyses all read a single one."
        )
    return int(layers[0]), int(saved["representations"].size(-1))


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


# The output layer under either of the two names a small language model files it under.
# Which one a checkpoint uses is recorded rather than assumed: tied to the input embedding
# the two are the same matrix, untied they are not, and reading the wrong one is silent.
_OUTPUT_LAYER = (("lm_head.weight", "lm_head"), ("embed_tokens.weight", "tied embedding"))


@dataclass(frozen=True)
class Weights:
    """What a run's checkpoint holds that an analysis asks of it.

    Both products of one read: a 6GB blob is not opened twice to answer two questions
    about the same run.
    """

    parameters: dict[str, torch.Tensor]
    """The VLM's own state dict, keyed as `SmolVLM` names its parameters."""
    output_weight: torch.Tensor
    """The output layer's matrix, `(V, H)`, in float32."""
    output_layer: str
    """Which key it was read from, and what kind of layer that key is."""


def checkpoint_path(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP
) -> Path:
    """Where a run's weights sit — the first root that actually holds them.

    The one reader here that does not go to a snapshot. Snapshots exclude the checkpoints
    because those are nearly all of a run's bytes, so the weights exist only on the
    machine that trained the run — or on a copy carried over from it, which is why every
    root in `cfg.run_roots` is probed rather than the local one alone. A run found
    nowhere resolves under `runs_root`, so the caller's error names the place a fresh
    run would have put it.
    """
    relative = Path(run_id(arm, seed, variant, timestamp)) / "checkpoints" / CHECKPOINT_NAME
    for root in cfg.run_roots:
        if (root / relative).exists():
            return root / relative
    return cfg.runs_root / relative


def require_checkpoints(variant: str = "", timestamp: str = TIMESTAMP) -> None:
    """Fail before the first measurement rather than after the last one.

    Every run is needed for the comparison to mean anything, and what follows a missing
    checkpoint takes long enough that finding out at the end would cost the whole pass.
    """
    missing = [
        checkpoint_path(arm, seed, variant, timestamp)
        for arm in ARMS
        for seed in SEEDS
        if not checkpoint_path(arm, seed, variant, timestamp).exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ARMS) * len(SEEDS)} checkpoints are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nThese are excluded from the snapshots by design, so they exist only on "
              "the machine that ran the training. Nothing here can be measured without them."
        )


def weights(
    arm: Arm, seed: int, variant: str = "", timestamp: str = TIMESTAMP, *, model: str | None = None
) -> Weights:
    """A run's trained parameters, and the output layer picked out of them.

    train4all files one state dict per registered model, keyed under `models` by the name
    it was registered with — and the study's trainer registers two, the VLM and the learned
    temperature. Which of them is the language model is settled by looking for the output
    layer rather than by naming it: a name spelled here as well as in the trainer is a name
    that can be renamed in one place only. A checkpoint that *is* a bare state dict is
    accepted too — this reads weights that already exist, and a checkpoint older than the
    code reading it is the ordinary case.

    `model` is the base checkpoint the caller assumes these weights were trained from. When
    the run recorded one — the reported runs did — the two are checked against each other;
    for a run trained before the trainer recorded it, nothing is checked and nothing is
    guessed.
    """
    path = checkpoint_path(arm, seed, variant, timestamp)
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Checkpoints are excluded from the snapshots by "
            f"design, so they are only on the machine that ran the training."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise SystemExit(f"{path}: expected a dict, found {type(blob).__name__}.")

    recorded = _recorded_model(blob)
    if model is not None and recorded is not None and recorded != model:
        raise SystemExit(f"{path}: trained on {recorded}, but the caller assumes {model}.")

    states = _state_dicts(blob, path)
    for suffix, kind in _OUTPUT_LAYER:
        hits = [(name, key) for name, state in states.items() for key in state
                if key.endswith(suffix)]
        if len(hits) > 1:  # the text tower's, not the vision tower's
            hits = [hit for hit in hits if "text" in hit[1]] or hits
        if len(hits) == 1:
            name, key = hits[0]
            return Weights(
                parameters=states[name],
                output_weight=states[name][key].detach().float(),
                output_layer=f"{kind} ({name}.{key})" if name else f"{kind} ({key})",
            )
    raise SystemExit(
        f"{path}: found neither an lm_head nor an embedding to use as the output layer.\n"
        f"  models: {sorted(states)}"
    )


def _state_dicts(blob: dict[str, Any], path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Every parameter mapping in a checkpoint, by the name it is filed under.

    A bare `state_dict()` has no name of its own and comes back under the empty one.
    """
    def holds_tensors(value: Any) -> bool:
        return isinstance(value, dict) and any(
            isinstance(v, torch.Tensor) for v in value.values()
        )

    models = blob.get("models")
    if isinstance(models, dict):
        found = {name: state for name, state in models.items() if holds_tensors(state)}
        if found:
            return found
    for key in ("model", "model_state_dict", "state_dict"):
        if holds_tensors(blob.get(key)):
            return {"": blob[key]}
    if holds_tensors(blob):
        return {"": blob}
    raise SystemExit(f"{path}: no parameters found. Top-level keys: {sorted(blob)[:20]}")


def _recorded_model(blob: dict[str, Any]) -> str | None:
    """The base checkpoint a run was trained from, if the trainer filed it under `extras`.

    The nine reported runs were trained before it did, so for them this is `None`.
    """
    extras = blob.get("extras")
    recorded = extras.get("checkpoint") if isinstance(extras, dict) else None
    return recorded if isinstance(recorded, str) else None


def analysis_path(script: str) -> Path:
    """Where an analysis files its measurements: `analyses/`, under its own name.

    Derived from the script rather than spelled out in it, so one analysis can only ever
    produce the one file that carries its name.
    """
    return cfg.analyses_root / f"{Path(script).stem.replace('_', '-')}.json"
