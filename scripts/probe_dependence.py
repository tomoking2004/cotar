"""Ask how hard the readout leans on the site, and whether alignment changed that.

context.md §5.5 answered where the aligned structure sits and could not answer whether the
model uses it. The two readings it left open — neutral and bypassed — differ in *dependence*:
bypass means the answer came to rely on the constrained site less, neutrality means it did
not. §4.5's second stage already computes the quantity that separates them and throws it
away, because orthonormalising a pulled-back direction keeps its bearing and discards its
length.

The length is what this keeps. For one prompt and one answer direction `u`, the
vector-Jacobian product gives how far the readout moves along `u` per unit of movement at
the site. That number carries units from both ends, and alignment is free to have changed
either — a run whose site vectors are simply shorter would look less depended-upon without
anything about the computation having changed. So it is reported as an **elasticity**,

    e = |J^T u| * |h| / |r|

— the relative movement of the readout along `u` produced by a relative movement of the
site. A one percent nudge at the site moves the readout `e` percent along `u`.

**The lengths are averaged, not the vectors.** §4.5 averages the gradient vectors over
prompts because it wants one subspace; the mean of vectors that disagree in direction is
short, which is a statement about agreement rather than about magnitude. Dependence is a
magnitude, so the norms are taken first and averaged after.

**The prompt count is checked rather than assumed.** §4.5 averaged over 64 prompts without
showing that 64 was enough. Both quantities are reported at prefixes of the same prompt
set, so a reader can see whether they had settled — the elasticity by its running mean, and
§4.5's own subspace by the cosine between the prefix's mean direction and the full one.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from train4all.utils import empty_cuda_cache

from cotar.analysis import (
    ARMS,
    CHECKPOINT_NAME,
    MODEL,
    SEEDS,
    TIMESTAMP,
    analysis_path,
    answer_token_rows,
    jacobian_batches,
    load_for_jacobian,
    output_basis,
    require_checkpoints,
    site,
    vjp_at_site,
)
from cotar.config import cfg
from cotar.models import build_smolvlm
from cotar.utils import load_json, save_json

# The widths §4.5 swept, so dependence is read on the same axis as where the structure sits.
DIMS = (8, 32, 128)

# The prompts and batching §4.5 used, so the two measurements describe the same linearisation.
PROMPTS = 64
BATCH = 8

# Prefixes the running values are reported at, to show whether PROMPTS was enough.
CHECKPOINTS = (8, 16, 32, 48, 64)

OUT_PATH = analysis_path(__file__)


def elasticities(
    vlm: Any, batches: list[dict[str, Any]], directions: torch.Tensor
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Per-prompt elasticities `(P, m)`, and the mean pulled direction after each prompt.

    The second return is §4.5's own quantity accumulated prompt by prompt, kept so its
    convergence can be read from the same backward passes rather than assumed.
    """
    per_prompt: list[torch.Tensor] = []
    running: dict[int, torch.Tensor] = {}
    total = torch.zeros_like(directions)
    seen = 0
    for batch in batches:
        grads, site_norm, readout_norm = vjp_at_site(vlm, batch, directions)
        # grads is (B, m, H); scale each prompt's row by its own two lengths.
        scale = (site_norm / readout_norm).unsqueeze(1)
        per_prompt.append(grads.norm(dim=-1) * scale)
        for row in range(grads.size(0)):
            total += grads[row].T
            seen += 1
            if seen in CHECKPOINTS:
                running[seen] = (total / seen).clone()
    return torch.cat(per_prompt), running


def settled(running: dict[int, torch.Tensor]) -> list[float]:
    """How close the mean direction was to its final value, at each checkpoint prefix."""
    final = running[max(CHECKPOINTS)]
    return [
        F.cosine_similarity(running[n], final, dim=0).abs().mean().item()
        for n in CHECKPOINTS
    ]


if __name__ == "__main__":
    require_checkpoints()
    layer, hidden_width = site()

    testdev = load_json(cfg.gqa.testdev_questions)
    answer_rows = answer_token_rows({entry["answer"] for entry in testdev.values()}, MODEL)
    _, processor = build_smolvlm("500M")
    batches = jacobian_batches(processor, PROMPTS, BATCH)
    counted = sum(len(b["prompt_lens"]) for b in batches)

    print(f"site: layer {layer}, {hidden_width} dimensions wide")
    print(f"{len(answer_rows):,} output-layer rows from the answer vocabulary")
    print(f"elasticity averaged over {counted} prompts in {len(batches)} batches\n")

    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            vlm, weight, source = load_for_jacobian(arm, seed, layer, cfg.device)
            span = output_basis(weight, answer_rows, max(DIMS))
            per_prompt, running = elasticities(vlm, batches, span)
            del vlm
            empty_cuda_cache()

            entry: dict[str, Any] = {
                "output_layer": source,
                "by_dim": {
                    str(m): {
                        "elasticity": per_prompt[:, :m].mean().item(),
                        "running": [per_prompt[:n, :m].mean().item() for n in CHECKPOINTS],
                    }
                    for m in DIMS
                },
                "direction_settled": settled(running),
            }
            results[arm][str(seed)] = entry
            print(f"[{arm} seed{seed}] output layer: {source}")
            for m in DIMS:
                by = entry["by_dim"][str(m)]
                print(f"{f'm = {m}':>26}  elasticity {by['elasticity']:7.4f}"
                      f"   prefixes " + " ".join(f"{v:6.4f}" for v in by["running"]))
            print(f"{'direction settled':>26}  "
                  + " ".join(f"{v:5.3f}" for v in entry["direction_settled"])
                  + f"   ← prefixes {CHECKPOINTS}\n")

    # ── the comparison the verdict rests on ──────────────────────────────────
    # Shorter for proposal than for baseline means the answer came to lean on the site
    # less, which is bypass. The same means the structure spread without the site losing
    # its hold, which is neutrality. Longer would mean alignment deepened the dependence,
    # and would have to be squared with §5.3's unmoved accuracy.
    summary = {
        arm: {
            str(m): mean(results[arm][str(s)]["by_dim"][str(m)]["elasticity"] for s in SEEDS)
            for m in DIMS
        }
        for arm in ARMS
    }
    print(f"{'':>10}" + "".join(f"{f'm={m}':>14}" for m in DIMS))
    for arm in ARMS:
        print(f"{arm:>10}" + "".join(f"{summary[arm][str(m)]:>14.4f}" for m in DIMS))

    print("\nproposal relative to baseline")
    for m in DIMS:
        b, p = summary["baseline"][str(m)], summary["proposal"][str(m)]
        seeds = [
            100 * (results["proposal"][str(s)]["by_dim"][str(m)]["elasticity"]
                   / results["baseline"][str(s)]["by_dim"][str(m)]["elasticity"] - 1)
            for s in SEEDS
        ]
        print(f"  m = {m:>3}:  {100 * (p / b - 1):+6.1f}%   seeds "
              + " ".join(f"{v:+6.1f}%" for v in seeds))

    save_json({
        "timestamp": TIMESTAMP,
        "checkpoint": CHECKPOINT_NAME,
        "site_layer": layer,
        "dims": list(DIMS),
        "answer_rows": len(answer_rows),
        "jacobian_prompts": counted,
        "prefixes": list(CHECKPOINTS),
        "runs": results,
        "means": summary,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
