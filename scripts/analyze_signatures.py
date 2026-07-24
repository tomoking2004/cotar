"""Count the task-signature distribution of a questions file. No training needed."""

from __future__ import annotations

import statistics
from collections import Counter
from itertools import pairwise

from cotar.config import cfg
from cotar.data import GQADataset

QUESTIONS_PATH = cfg.gqa.train_questions
IMAGES_DIR     = cfg.gqa.images
TOP_K          = 20
BUCKETS        = (1, 2, 3, 5, 9, 17, 33, 65)


if __name__ == "__main__":
    dataset = GQADataset(QUESTIONS_PATH, IMAGES_DIR, require_program=True)
    counts = Counter(dataset.signatures)
    if not counts:
        raise SystemExit(f"No samples with a program found (check {QUESTIONS_PATH}).")

    sizes = sorted(counts.values(), reverse=True)
    n_samples, n_sigs = sum(sizes), len(sizes)
    singletons = sizes.count(1)
    retained = n_samples - singletons

    print(f"file         {QUESTIONS_PATH.name}")
    print(f"samples      {n_samples:,}")
    print(f"signatures   {n_sigs:,}  (max {sizes[0]:,} · median {statistics.median(sizes):.0f}"
          f" · mean {n_samples / n_sigs:.1f} samples each)")
    print(f"singletons   {singletons:,}  ({singletons / n_samples:.1%} of samples,"
          f" dropped by grouping)")
    print(f"pairable     {n_sigs - singletons:,} signatures → {retained:,} samples"
          f" ({retained / n_samples:.1%} retained)")

    print(f"\ntop {TOP_K} signatures by sample count")
    for sig, c in counts.most_common(TOP_K):
        print(f"  {c:>7,}  {sig}")

    print("\ngroup-size distribution (signatures per size bucket)")
    for lo, hi in pairwise((*BUCKETS, None)):
        n = sum(c >= lo and (hi is None or c < hi) for c in sizes)
        if not n:
            continue
        label = str(lo) if hi == lo + 1 else f"{lo}+" if hi is None else f"{lo}-{hi - 1}"
        print(f"  {label:>6}  {n:,} signatures")
