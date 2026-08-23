"""Vary one setting the reported experiment fixed, so a single point becomes a curve.

The study measured one configuration: alignment weight 0.1, on layer 16. Everything it
concludes is therefore about that point, and "why 0.1?" has no answer in it. This runs
the neighbouring points so that the trade-off between what the representation carries and
what the answers cost can be read as a shape rather than asserted from one measurement.

Two points are already measured and are deliberately absent below: weight 0 *is* the
baseline arm, and weight 0.1 *is* the reported proposal. A run repeats neither.

The runs land under their own timestamp, and are still comparable to the reported nine:
a seed fixes the initialisation and the batch order, so a run here at seed 42 sees
exactly what the reported runs at seed 42 saw. `summarize_sweep.py` reads both sets
together on that basis — give it the timestamp this prints.

Start with `DEBUG = True`. It finishes in minutes and proves the wiring before a day of
GPU time is committed to it.
"""

from __future__ import annotations

import os

# Must be set before the CUDA allocator is first used (i.e. before .to("cuda")).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from itertools import product

from train4all.utils import empty_cuda_cache

from cotar.training import Arm, Settings, run_training
from cotar.utils import timestamp

DEBUG = False

# One seed first. The curve's shape is visible at one seed, and only the point worth
# reporting needs the other two — three seeds at every point costs three times the GPU
# hours to sharpen intervals nobody reads until the shape says where to look.
SEEDS = (42,)

# The arm every variant runs. The sweep asks what the alignment weight does, and only the
# signature-labelled arm has an alignment to weight.
ARM: Arm = "proposal"

# What each variant changes, against defaults that are the reported experiment's values.
# The name becomes part of the run directory, so it is what `summarize_sweep.py` asks for.
LAMBDA_VARIANTS: tuple[tuple[str, Settings], ...] = (
    ("lambda0.03", Settings(align_weight=0.03)),
    ("lambda0.3", Settings(align_weight=0.3)),
    ("lambda1.0", Settings(align_weight=1.0)),
)

# The other question the study leaves open: "why layer 16?"
LAYER_VARIANTS: tuple[tuple[str, Settings], ...] = (
    ("layer8", Settings(layers=(8,))),
    ("layer24", Settings(layers=(24,))),
)

# Which sweep this run performs. One at a time, never both at once — a curve mixing two
# changed settings has two explanations and settles neither.
VARIANTS = LAMBDA_VARIANTS


if __name__ == "__main__":
    ts = timestamp()
    total = len(SEEDS) * len(VARIANTS)

    print(f"sweep timestamp: {ts}"
          f"\n  → put this in summarize_sweep.py as SWEEP_TIMESTAMP\n")

    for i, (seed, (variant, settings)) in enumerate(product(SEEDS, VARIANTS), start=1):
        if DEBUG:
            settings = settings.as_debug()
        print(f"\n{'=' * 79}\n▶️  Run {i}/{total}: {variant} {ARM} (seed {seed})"
              f"  ·  weight {settings.align_weight}, layers {settings.layers}"
              f"\n{'=' * 79}")
        run_training(ARM, seed, settings, timestamp=ts, variant=variant)

        # Here rather than inside `run_training`: the model, its optimizer state and the
        # trainer holding both are still live locals until that call returns.
        empty_cuda_cache()

    print(f"\n{'=' * 79}\n✅ All {total} runs complete under {ts}.\n{'=' * 79}\n")
