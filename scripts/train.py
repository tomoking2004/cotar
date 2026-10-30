"""Run the experiment the study reports: every arm at every seed, into one timestamped set.

The arms are not a setting of this script — `cotar.training` defines them, and this runs
every one of them. How a run is built is not a setting either: `cotar.training.Settings`
holds that, so that this and the sweep beside it build runs the same way and their
numbers land on one axis.

`SEEDS` is the batch this runs next. The reported experiment ran 42, 43 and 44
(`20260829-134859`); context.md §8.1 adds 45 and 46 to it for the robustness question
alone, so this now trains those two and nothing it has trained before. Setting it back to
the first three reruns the reported batch.

Set `DEBUG` to prove the wiring in minutes before committing a day to it.
"""

from __future__ import annotations

import os

# Must be set before the CUDA allocator is first used (i.e. before .to("cuda")).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from itertools import product

from train4all.utils import empty_cuda_cache

from cotar.training import ARMS, Settings, run_training
from cotar.utils import timestamp

DEBUG    = False
SEEDS    = (45, 46)
SETTINGS = Settings()


if __name__ == "__main__":
    settings = SETTINGS.as_debug() if DEBUG else SETTINGS
    ts = timestamp()
    total = len(SEEDS) * len(ARMS)

    print(f"timestamp: {ts}"
          f"\n  → the analyses find this batch by it: name it in cotar/analysis/experiment.py "
          f"(ADDED_TIMESTAMP for seeds 45 and 46)\n")

    # Seed-major, so an interrupted experiment leaves whole arm comparisons behind
    # rather than the same arm at every seed and nothing to compare it against.
    for i, (seed, arm) in enumerate(product(SEEDS, ARMS), start=1):
        print(f"\n{'=' * 79}\n▶️  Run {i}/{total}: {arm} (seed {seed})\n{'=' * 79}")
        run_training(arm, seed, settings, timestamp=ts)

        # Here rather than inside `run_training`: the model, its optimizer state and the
        # trainer holding both are still live locals until that call returns, so a
        # collection inside it cannot reach them however late it is placed.
        empty_cuda_cache()

    print(f"\n{'=' * 79}\n✅ All {total} runs complete.\n{'=' * 79}\n")
