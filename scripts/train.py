"""Run the whole experiment: every arm at every seed, into one timestamped set.

Each run gets a fresh model, fresh loaders and a fresh trainer. That is what makes the
comparison legitimate — a reused sampler would carry its epoch counter into the next
arm and feed it different batches, and reused weights would not be a fresh start. Built
this way, the arms of one seed see exactly the same data in exactly the same order, and
differ only in `arm`; the seeds differ in that order and in initialisation, which is
what makes a difference between arms distinguishable from run-to-run noise.
"""

from __future__ import annotations

import os

# Must be set before the CUDA allocator is first used (i.e. before .to("cuda")).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from itertools import product
from typing import Any, get_args

from train4all import Phase
from train4all.utils import empty_cuda_cache

from cotar.config import cfg
from cotar.data import build_gqa_dataloader
from cotar.models import build_smolvlm
from cotar.modules import INIT_SCALE
from cotar.trainer import Arm, Trainer
from cotar.utils import make_run_id, timestamp


# Derived, not a setting: the arms and their order are `Arm`'s to define, and this is
# the study's independent variable — it may not drift from the type the trainer checks.
ARMS: tuple[Arm, ...] = get_args(Arm)

# experiment
DEBUG                 = False
SEEDS                 = (42, 43, 44)

# model
MODEL_SIZE            = "500M"
LAYERS                = (16,)

# trainer
# The contrastive term's starting temperature is absent from this block on purpose:
# `cotar.modules` holds that number as `INIT_SCALE` (imported above, passed below), and
# restating it here would give one number a second home to drift from.
ALIGN_WEIGHT          = 0.1
EPOCHS                = 1
BATCH_SIZE            = 32 if not DEBUG else 8
LR                    = 1e-5
MAX_GRAD_NORM         = 1.0
ACCUMULATION_STEPS    = 1

# loader
SAMPLES_PER_SIGNATURE = 2
TRAIN_LIMIT           = None if not DEBUG else 64
VAL_LIMIT             = 2048 if not DEBUG else 64
TESTDEV_LIMIT         = None if not DEBUG else 64


def run_arm(arm: Arm, seed: int, ts: str) -> None:
    vlm, processor = build_smolvlm(MODEL_SIZE, layers=LAYERS)
    trainer = Trainer(
        vlm, processor,
        arm=arm,
        align_weight=ALIGN_WEIGHT,
        init_scale=INIT_SCALE,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        max_grad_norm=MAX_GRAD_NORM,
        accumulation_steps=ACCUMULATION_STEPS,
        monitor="accuracy",
        monitor_mode="max",
        device=cfg.device,
        seed=seed,
        run_dir=cfg.runs_root / make_run_id(f"{arm}_seed{seed}", debug=DEBUG, ts=ts),
        record_step_metrics=True,
        pbar_metric_names=["accuracy", "loss"],
        debug_mode=DEBUG,
        use_dashboard=True,
        gqa_questions_path=cfg.gqa.testdev_questions,
    )

    loader_kwargs: dict[str, Any] = dict(
        images_dir=cfg.gqa.images,
        processor=processor,
        batch_size=BATCH_SIZE,
        with_labels=True,
        num_workers=cfg.num_workers,
        seed=seed,
        logger=trainer.logger,
    )
    train_loader = build_gqa_dataloader(
        cfg.gqa.train_questions,
        group_by_signature=True,
        samples_per_signature=SAMPLES_PER_SIGNATURE,
        min_samples_per_signature=SAMPLES_PER_SIGNATURE,
        require_program=True,
        limit=TRAIN_LIMIT,
        **loader_kwargs,
    )
    train_eval_loader = build_gqa_dataloader(
        cfg.gqa.train_questions,
        limit=VAL_LIMIT,
        **loader_kwargs,
    )
    val_loader = build_gqa_dataloader(
        cfg.gqa.val_questions,
        limit=VAL_LIMIT,
        **loader_kwargs,
    )
    testdev_loader = build_gqa_dataloader(
        cfg.gqa.testdev_questions,
        limit=TESTDEV_LIMIT,
        **loader_kwargs,
    )

    # What reproducing this run needs and no trainer argument carries — what this VLM is,
    # and the loaders' settings, neither of which the trainer is constructed with.
    trainer.update_checkpoint_extras({
        "checkpoint": vlm.checkpoint,
        "layers": vlm.layers,
        "attn_implementation": vlm.attn_implementation,
        "samples_per_signature": SAMPLES_PER_SIGNATURE,
        "limits": {"train": TRAIN_LIMIT, "val": VAL_LIMIT, "testdev": TESTDEV_LIMIT},
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

    # `testdev` is the split this study reports, and the phase name is what the metric
    # tables, the plots and `eval.json` file the final numbers under — so the name says
    # which split they came from rather than merely that they came last.
    trainer.test(Phase("testdev", testdev_loader), use_best=True)


if __name__ == "__main__":
    ts = timestamp()
    total = len(SEEDS) * len(ARMS)

    # Seed-major, so an interrupted experiment leaves whole arm comparisons behind
    # rather than the same arm at every seed and nothing to compare it against.
    for i, (seed, arm) in enumerate(product(SEEDS, ARMS), start=1):
        print(f"\n{'=' * 79}\n▶️  Run {i}/{total}: {arm} (seed {seed})\n{'=' * 79}")
        run_arm(arm, seed, ts)

        # Here rather than at the end of `run_arm`: the model, its optimizer state and
        # the trainer holding both are still live locals until that call returns, so a
        # collection inside it cannot reach them however late it is placed.
        empty_cuda_cache()

    print(f"\n{'=' * 79}\n✅ All {total} runs complete.\n{'=' * 79}\n")
