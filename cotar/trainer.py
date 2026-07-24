from __future__ import annotations

import random
import zlib
from pathlib import Path
from typing import Any, Literal, get_args

import torch
from schedulefree import RAdamScheduleFree
from train4all import BaseTrainer, Phase

from .data import Batch
from .losses import supervised_contrastive_loss
from .metrics import MAX_STABILITY_SAMPLES, Evaluator
from .modules import LogitScale
from .types import IGNORE_INDEX, VLM, VLMProcessor
from .utils import save_json

__all__ = ["Arm", "Trainer"]

Arm = Literal["baseline", "proposal", "ablation"]


class Trainer(BaseTrainer):
    """Trains a VLM on GQA, optionally aligning same-task internal representations.

    ``arm`` selects one of the study's three conditions, and is the *only* knob that
    separates them — data, grouping and schedule are held identical, so any difference
    in results traces to the alignment term alone:

    ``"baseline"``
        Alignment off. The language-modelling loss alone.
    ``"proposal"``
        Supervised-contrastive alignment over operator-sequence task signatures, so
        same-program samples attract. Our method.
    ``"ablation"``
        The same loss at the same ``align_weight``, over the same class and
        positive-pair counts, but with the pairing decorrelated from the program —
        separating our signal from a generic contrastive-regularisation effect.

    ``align_weight`` is the magnitude of that auxiliary term; ``"baseline"`` pins it to
    zero whatever is passed, so the three arms differ by ``arm`` and nothing else.

    Every metric comes from :class:`~cotar.metrics.Evaluator`; nothing is measured here.

    The trainer holds no schedule: every pass a run makes — training, the held-in
    overfitting check, validation, and the final evaluation — is a
    :class:`~train4all.Phase` handed to ``train()`` or ``test()`` by the caller (see
    `scripts/train.py`). The final report is written for whichever pass ``test()`` ran,
    under whatever name the caller gave it.
    """

    def __init__(
        self,
        vlm: VLM,
        processor: VLMProcessor,
        *,
        arm: Arm = "baseline",
        align_weight: float = 0.1,
        init_scale: float = 1 / 0.07,
        # evaluation
        gqa_questions_path: Path | str | None = None,
        gqa_choices_path: Path | str | None = None,
        max_stability_samples: int = MAX_STABILITY_SAMPLES,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # `Arm` is a type-only Literal, so a misspelt arm would otherwise fall through
        # to the proposal settings and be *recorded* under the misspelling — the arm is
        # this study's independent variable, and it may not be assigned by accident.
        if arm not in get_args(Arm):
            raise ValueError(f"arm must be one of {get_args(Arm)}, got {arm!r}.")

        self.vlm            = vlm
        self.processor      = processor
        self.arm            = arm
        self.align_weight   = 0.0 if arm == "baseline" else align_weight
        self.align_pairing  = "shuffled" if arm == "ablation" else "signature"
        self.logit_scale    = LogitScale(init_scale=init_scale)

        self._is_training = True
        self._evaluator   = Evaluator(
            vlm,
            processor,
            gqa_questions_path=gqa_questions_path,
            gqa_choices_path=gqa_choices_path,
            max_stability_samples=max_stability_samples,
            seed=self.seed or 0,
        )

        # `seed` is not repeated here: `BaseTrainer` already records it in `config.json`
        # whenever it differs from the constructor default, which every run does.
        self.update_config({
            "arm": self.arm,
            "align_weight": self.align_weight,
            "align_pairing": self.align_pairing,
            "init_scale": init_scale,
        })

    # ── train4all interface ───────────────────────────────────────────────────

    def setup(self) -> None:
        self.set_models({"vlm": self.vlm, "logit_scale": self.logit_scale})
        self.set_optimizer(
            RAdamScheduleFree(params=self.get_trainable_params(), lr=self.learning_rate)
        )

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        output = self.vlm(**batch)
        loss, representation = output["loss"], output["representation"]
        if loss is None:
            raise ValueError("Model returned no loss; ensure `labels` are in the batch.")
        if representation is None:
            raise ValueError(
                "Every step measures the representation, so the model must expose one: "
                "build it with a layer, e.g. build_smolvlm('500M', layers=16)."
            )
        # Cache this forward's representation so the metric pass can reuse it rather
        # than pay for a second forward.
        self.set_cache("representation", representation.detach())

        # Whatever this returns is reported as `loss` — in *every* phase, evaluation
        # included, since BaseTrainer computes it under no-grad to score val and test too.
        # Fold the auxiliary term in and that number stops meaning the same thing in each
        # arm: baseline reports a language-modelling loss and the other two report it plus
        # a contrastive term, so an arm carrying a larger auxiliary loss looks like a worse
        # language model when it may be an identical one. Both terms are cached separately
        # for `compute_metrics`, which reports `lm_loss` — the only one of them that is the
        # same quantity in all three arms, and therefore the only one they may be compared
        # on. The returned objective is unchanged: it is what the optimizer must see.
        self.set_cache("lm_loss", loss.detach())
        if not self.align_weight:
            return loss
        align_loss = self._alignment_loss(representation, batch)
        self.set_cache("align_loss", align_loss.detach())
        return loss + self.align_weight * align_loss

    def compute_metrics(self, batch: Batch) -> dict[str, float]:
        # `BaseTrainer` puts its own `loss` first and insertion order is what the tables
        # and exports are read in, so the terms it decomposes into follow it directly:
        # `lm_loss` exists to be told apart from `loss`, and cannot be from a distance.
        metrics = self._evaluator.measure(
            batch, self.get_cache("representation"), training=self._is_training
        )
        return self._losses() | self._temperature() | metrics

    def get_batch_weight(self, batch: Batch) -> int:
        # The model's loss is a mean over the supervised tokens, so metric averaging (and
        # gradient accumulation) must weight each batch by that same denominator — not by
        # its sample count, which the prompt and image tokens are absent from.
        return int((batch["labels"] != IGNORE_INDEX).sum())

    # ── lifecycle hooks ───────────────────────────────────────────────────────

    def on_set_training_mode(self, training: bool) -> None:
        self._is_training = training
        # RAdamScheduleFree keeps separate train/eval parameter views; evaluation
        # must read the averaged weights, so the two switch in lockstep.
        if isinstance(self._optimizer, RAdamScheduleFree):
            (self._optimizer.train if training else self._optimizer.eval)()

    def on_phase_start(self, epoch: int | None, phase: Phase) -> None:
        # Every phase scores itself from a clean slate — the evaluator accumulates
        # predictions and representations for `report`, and one phase's must never be
        # weighed into the next's.
        self._evaluator.reset()

    def on_phase_end(
        self, epoch: int | None, phase: Phase, metrics: dict[str, float]
    ) -> None:
        # What marks the final evaluation is that it belongs to no epoch: `test()` runs
        # its phase outside the loop and hands this hook `None`, where every pass
        # `train()` makes carries the epoch it ran in. Keying on the *name* instead
        # would tie the report to one spelling of it — and the name is the caller's to
        # choose, so a pass renamed after the split it measures would silently stop
        # reporting.
        if epoch is not None:
            return

        # Persist predictions so any arm can be scored offline with the official
        # evaluator and compared across runs.
        self._evaluator.save_predictions(self.get_metrics_path("predictions"))
        # And the representations themselves — the object of the whole study, and what
        # every later question about the geometry has to be asked of.
        self._evaluator.save_representations(
            self.get_metrics_path("representations").with_suffix(".pt")
        )
        report = self._evaluator.report()
        for name, scores in report.items():
            self.print_dict_tree(scores, header=f"📋 {name}")

        # Carry the run's identity and headline metrics into the report so the comparison
        # reads one self-describing file per run. The *seed* belongs to that identity as
        # squarely as the arm does — the run directory no longer names it, so this is the
        # only place it survives. So does the phase: which split these numbers came from
        # is the caller's choice, and a report that did not name it would be a column of
        # scores with no split attached. What the trainer ran is what the report says it ran.
        path = self.get_metrics_path("eval")
        save_json(
            {
                "config": {
                    "arm": self.arm,
                    "seed": self.seed,
                    "align_weight": self.align_weight,
                    "align_pairing": self.align_pairing,
                    "phase": phase.name,
                },
                "test_metrics": metrics,
                **report,
            },
            path,
        )
        self.print(f"📄 Test report saved: {path.name}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _alignment_loss(self, representation: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Supervised-contrastive loss over same-task representations, `(B, L, H)` in.

        Each constrained layer is scored on its own and the scores averaged, so a layer
        is asked to separate signatures *within itself* — the claim the hypothesis makes
        of every layer it is applied to. Concatenating the layers into one long vector
        would instead ask a single question of the stack, and answer it mostly in
        whichever layer has the largest norm; the residual stream grows through depth, so
        that layer is knowable in advance and it is not the one anybody chose.

        The temperature is shared across layers: it is the strength of the constraint,
        and holding it common is what makes the per-layer terms commensurable enough to
        average. Averaging (not summing) also keeps `align_weight` meaning the same thing
        whether one layer is constrained or five.
        """
        labels = self._pair_labels(batch["task_signatures"], batch["question_ids"])
        logit_scale = self.logit_scale()
        per_layer = torch.stack([
            supervised_contrastive_loss(self._project(layer), labels, logit_scale)
            for layer in representation.unbind(dim=1)
        ])
        return per_layer.mean()

    def _project(self, representation: torch.Tensor) -> torch.Tensor:
        """Subspace seam for the alignment target — identity for now.

        The settled design constrains the *raw* hidden state directly (no learnable
        head), so a rise in separation is attributable to `h` itself and the hypothesis
        stays falsifiable. Whether to confine the contrastive gradient to a fixed
        subspace — a random orthogonal ``U``, or the model's verbalisable "workspace"
        directions — is deliberately left open until the representation is measured;
        that projection would slot in here."""
        return representation

    def _pair_labels(self, signatures: list[str], question_ids: list[str]) -> list[str]:
        """Labels defining the contrastive positives — the sole knob the arms vary.

        The ablation permutes the batch's signatures rather than inventing labels, so
        the class histogram (and with it the positive-pair count) is preserved exactly
        while the pairing is decorrelated from the program. Seeding the permutation on
        the question ids keeps it deterministic per batch, and therefore reproducible."""
        if self.align_pairing == "signature":
            return signatures
        seed = zlib.crc32("".join(question_ids).encode()) ^ (self.seed or 0)
        shuffled = list(signatures)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _losses(self) -> dict[str, float]:
        """The objective's terms, reported apart.

        `loss` is the objective, and in the aligned arms the objective is not the language
        model's loss — see `compute_loss`. `lm_loss` is, in all three, so it is what the
        arms are compared on; `align_loss` is the auxiliary term before `align_weight`
        scales it, and only exists where there is one.
        """
        losses = {"lm_loss": float(self.get_cache("lm_loss"))}
        if self.align_weight:
            losses["align_loss"] = float(self.get_cache("align_loss"))
        return losses

    def _temperature(self) -> dict[str, float]:
        # Only meaningful while aligning: the temperature is learned through the
        # contrastive term, and stays at its init otherwise.
        return {"temperature": float(self.logit_scale.temperature)} if self.align_weight else {}
