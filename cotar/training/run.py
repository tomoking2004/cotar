"""One training run, and the settings that decide it.

Two scripts train: one runs the three arms at three seeds to produce the experiment the
study reports, and one sweeps a setting the study fixed. They must build the run the same
way or their numbers cannot be put on the same axis, so the building is here and neither
script has its own copy.

`Settings` carries every choice a run is decided by beyond which arm and seed it is. Its
defaults are the reported experiment's values, so a sweep states only what it changes and
the rest is visibly the same as what it is being compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from train4all import Phase
from train4all.utils import empty_cuda_cache

from ..config import cfg
from ..data import build_gqa_dataloader
from ..models import Size, build_smolvlm
from ..utils import make_run_id
from .logit_scale import INIT_SCALE
from .trainer import Arm, Trainer

__all__ = ["Settings", "run_training"]


@dataclass(frozen=True)
class Settings:
    """What one run is, apart from its arm and seed. Defaults are the reported experiment.

    `debug` is not a setting of the experiment but a way to prove the wiring before
    committing hours to it: it shrinks every split so a run finishes in minutes, and
    marks the run directory so a smoke test can never be mistaken for a measurement.
    """

    model_size: Size = "500M"
    layers: tuple[int, ...] = (16,)
    align_weight: float = 0.1
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    samples_per_signature: int = 2
    train_limit: int | None = None
    val_limit: int | None = 2048
    testdev_limit: int | None = None
    debug: bool = False

    def as_debug(self) -> Settings:
        """The same settings cut down to a smoke test."""
        return replace(
            self, debug=True, batch_size=8, train_limit=64, val_limit=64, testdev_limit=64
        )


def run_training(
    arm: Arm, seed: int, settings: Settings, *, timestamp: str, variant: str = ""
) -> None:
    """Train one arm at one seed, and leave a run directory and a snapshot behind.

    Everything is built fresh here — model, loaders, trainer. That is what makes the
    comparison legitimate: a reused sampler would carry its epoch counter into the next
    run and feed it different batches, and reused weights would not be a fresh start.
    Built this way, runs sharing a seed see exactly the same data in exactly the same
    order whatever else differs, so a difference between them is a difference of setting
    rather than of draw — and that holds across experiments, not only within one.
    """
    name = f"{variant}_{arm}_seed{seed}" if variant else f"{arm}_seed{seed}"
    run_id = make_run_id(name, debug=settings.debug, ts=timestamp)

    vlm, processor = build_smolvlm(settings.model_size, layers=settings.layers)
    trainer = Trainer(
        vlm, processor,
        arm=arm,
        align_weight=settings.align_weight,
        init_scale=INIT_SCALE,
        num_epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        max_grad_norm=settings.max_grad_norm,
        accumulation_steps=settings.accumulation_steps,
        monitor="accuracy",
        monitor_mode="max",
        device=cfg.device,
        seed=seed,
        run_dir=cfg.runs_root / run_id,
        # Everything the run produced except the weights, mirrored after every epoch and
        # again after the final evaluation. The checkpoints are nearly all of a run
        # directory's bytes, and the part every later question is asked of — the metrics,
        # the predictions, the representations, the log — is a rounding error beside them.
        # Dropping them is what leaves a copy small enough to keep for every run at once.
        run_snapshot_dir=cfg.snapshots_root / run_id,
        run_snapshot_exclude=["checkpoints"],
        record_step_metrics=True,
        pbar_metric_names=["accuracy", "loss"],
        debug_mode=settings.debug,
        use_dashboard=True,
        gqa_questions_path=cfg.gqa.testdev_questions,
    )

    loader_kwargs: dict[str, Any] = {
        "images_dir": cfg.gqa.images,
        "processor": processor,
        "batch_size": settings.batch_size,
        "with_labels": True,
        "num_workers": cfg.num_workers,
        "seed": seed,
        "logger": trainer.logger,
    }
    train_loader = build_gqa_dataloader(
        cfg.gqa.train_questions,
        group_by_signature=True,
        samples_per_signature=settings.samples_per_signature,
        min_samples_per_signature=settings.samples_per_signature,
        require_program=True,
        limit=settings.train_limit,
        **loader_kwargs,
    )
    train_eval_loader = build_gqa_dataloader(
        cfg.gqa.train_questions, limit=settings.val_limit, **loader_kwargs
    )
    val_loader = build_gqa_dataloader(
        cfg.gqa.val_questions, limit=settings.val_limit, **loader_kwargs
    )
    testdev_loader = build_gqa_dataloader(
        cfg.gqa.testdev_questions, limit=settings.testdev_limit, **loader_kwargs
    )

    # What reproducing this run needs and no trainer argument carries — what this VLM is,
    # and the loaders' settings, neither of which the trainer is constructed with.
    trainer.update_checkpoint_extras({
        "checkpoint": vlm.checkpoint,
        "layers": vlm.layers,
        "attn_implementation": vlm.attn_implementation,
        "samples_per_signature": settings.samples_per_signature,
        "limits": {
            "train": settings.train_limit,
            "val": settings.val_limit,
            "testdev": settings.testdev_limit,
        },
    })

    trainer.train(
        Phase("train", train_loader, training=True),
        Phase("train_eval", train_eval_loader),
        Phase("val", val_loader),
    )

    # Free the parsed question sets before test() forks its workers: each child inherits
    # them copy-on-write, and refcounting alone dirties enough pages to copy them per
    # worker — which is what exhausted the host here. The trainer holds no reference to a
    # phase once train() returns, so these are the last ones.
    del train_loader, train_eval_loader, val_loader
    empty_cuda_cache()

    trainer.test(Phase("testdev", testdev_loader), use_best=True)
