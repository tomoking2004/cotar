"""Ask the bypass question again, with the output layer's directions carried to the site.

This is the second of the two stages in context.md §4.5, and it runs because the first did
not decide. There, the answer directions were read off the output layer and applied to the
sixteenth layer's vector as they stood — the two spaces have the same width, but they are
not the same space, and the first stage assumed they were. What it found is what a bad
approximation looks like: the output layer's span held no more of the signature than a
random span of the same width did, in *every* arm, alignment or none. A subspace that is
no better than random before anything was optimised is not the subspace the output reads.

So carry the directions instead of assuming them. Sixteen layers sit between the site and
the readout; linearise them and each answer direction has an image at the site, which is
the direction that actually moves the answer. That is J-lens (Gurnee et al.). Only the
vector-Jacobian product is wanted, so one backward pass per direction suffices and the
`H x H` Jacobian is never formed.

**The average is over prompts, at the constrained position — not over positions.** Gurnee
et al. average over both, because they are after a general layer-to-layer map. Here the
site is one position: the last prompt token, the one the alignment loss touched and the
one the first answer token is chosen from. The image and question positions carry hidden
states the alignment never constrained, and averaging them in would blur the map with
directions belonging to vectors that are not under study. §4.5 records this departure from
Gurnee et al., who average on both axes.

Everything downstream is the first stage's, unchanged: the same rows §5.1 probes, the same
four places, the same random twin at each width, and the same verdict — not the level
anywhere and not the gain anywhere, but the difference between the pulled-back span and a
random span of the same width.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train4all.utils import empty_cuda_cache

from cotar.analysis import (
    ARMS,
    CHECKPOINT_NAME,
    MODEL,
    SEEDS,
    TIMESTAMP,
    WHERE,
    Arm,
    analysis_path,
    answer_token_rows,
    constrained_layer,
    delta,
    gain,
    keep_frequent,
    majority_floor,
    output_basis,
    probe_accuracy,
    random_basis,
    reported_gain,
    representations,
    require_checkpoints,
    scorable,
    scores,
    seed_deltas,
    split_mask,
    splitter,
    summarize_places,
    weights,
)
from cotar.config import cfg
from cotar.data import build_gqa_dataloader
from cotar.models import SmolVLM, build_smolvlm
from cotar.types import VLMProcessor
from cotar.utils import load_json, save_json

# The same widths the first stage swept, so the two are read on one axis.
DIMS       = (8, 32, 128)
BASIS_SEED = 0

# How much of testdev the Jacobian is averaged over, and in how many pieces. The same
# prompts are used for every run, so a difference between two pullbacks is a difference of
# weights. One backward per direction per batch, and a backward costs ~11ms here, so the
# whole nine runs spend about two minutes inside autograd.
PROMPTS    = 64
BATCH      = 8

OUT_PATH = analysis_path(__file__)


def site() -> tuple[int, int]:
    """The layer the nine runs constrained, and the width of its vector.

    Refused if they disagree: nine runs constraining different layers are not one
    experiment, and there would be no single site to pull anything back to.
    """
    sites = {constrained_layer(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(sites) != 1:
        raise SystemExit(f"the runs constrained different sites ({sorted(sites)}).")
    return sites.pop()


def load(arm: Arm, seed: int, layer: int, device: str) -> tuple[SmolVLM, torch.Tensor, str]:
    """One run's trained model, ready to differentiate, and its output layer.

    Evaluation mode, because what is being linearised is the network as it answers rather
    than as it trains — and because the model turns gradient checkpointing on for CUDA,
    which the layers apply only while training.
    """
    trained = weights(arm, seed, model=MODEL)
    vlm, _ = build_smolvlm("500M", layers=(layer,))
    vlm.load_state_dict(trained.parameters)
    return vlm.to(device).eval(), trained.output_weight, trained.output_layer


def pull_back(
    vlm: SmolVLM, batches: list[dict[str, Any]], directions: torch.Tensor
) -> torch.Tensor:
    """Every direction carried from the output layer back to the site — `(H, m)` in, `(H, m)` out.

    Each column is one backward pass per batch, accumulated over prompts and divided at the
    end. The gradient comes back for every position; the row taken is the site's own, which
    is the direct dependence of the readout on the vector the alignment constrained.
    """
    pulled = torch.zeros_like(directions)
    counted = 0
    for batch in batches:
        prompt_lens = batch["prompt_lens"]
        with torch.enable_grad():
            hidden, readout = vlm.readout_from_site(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                pixel_attention_mask=batch.get("pixel_attention_mask"),
                prompt_lens=prompt_lens,
            )
            rows = torch.arange(len(prompt_lens), device=hidden.device)
            last = torch.tensor(prompt_lens, device=hidden.device) - 1
            for j in range(directions.size(1)):
                direction = directions[:, j].to(readout.device, readout.dtype)
                grad, = torch.autograd.grad(
                    readout,
                    hidden,
                    grad_outputs=direction.repeat(len(prompt_lens), 1),
                    retain_graph=j + 1 < directions.size(1),
                )
                pulled[:, j] += grad[rows, last].float().sum(dim=0).cpu()
        counted += len(prompt_lens)
    return pulled / counted


def orthonormal(pulled: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis for the pulled-back directions, `(H, m)`.

    Taken once at the widest `m` and sliced for the narrower ones: a reduced QR builds its
    columns in order, so the leading `m` of it are exactly what QR of the leading `m`
    columns would give. The order is the answer rows' own — their leading principal
    directions — which is the order the first stage swept.
    """
    q, _ = torch.linalg.qr(pulled)
    return q.contiguous()


def rotation(pulled: torch.Tensor, original: torch.Tensor) -> float:
    """How far the layers between the site and the readout turn the answer directions,
    as a mean absolute cosine between each direction and its image.

    The first stage assumed this was 1 — that a direction the output layer reads is the
    same direction at the site. Measuring it says what that assumption was worth, and it
    is the reason this stage exists rather than an ornament on it. Near 1 and the first
    stage's subspace was the right one after all; near 0 and it was measuring a subspace
    with no particular relation to the answer.
    """
    return F.cosine_similarity(pulled, original, dim=0).abs().mean().item()


def prompts(processor: VLMProcessor) -> list[dict[str, Any]]:
    """The testdev batches the Jacobian is averaged over, on the device, kept in memory.

    Held rather than re-read per run: nine models are differentiated against exactly the
    same inputs, which is what makes the nine pullbacks comparable.
    """
    loader: DataLoader[Any] = build_gqa_dataloader(
        cfg.gqa.testdev_questions,
        images_dir=cfg.gqa.images,
        processor=processor,
        batch_size=BATCH,
        with_labels=True,
        limit=PROMPTS,
        num_workers=0,
    )
    return [
        {
            key: value.to(cfg.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for batch in loader
    ]


if __name__ == "__main__":
    require_checkpoints()
    layer, hidden_width = site()

    # The rows and the split are §5.1's, so the full-space fit below is checked against a
    # number that already exists rather than against itself.
    _, signatures, _ = representations(ARMS[0], SEEDS[0])
    keep, _, labels = keep_frequent(signatures)
    rows, sub_labels, sub_train, n_classes = scorable(labels, split_mask(len(keep), splitter()))

    testdev = load_json(cfg.gqa.testdev_questions)
    answer_rows = answer_token_rows({entry["answer"] for entry in testdev.values()}, MODEL)

    _, processor = build_smolvlm("500M")
    batches = prompts(processor)

    print(f"{int(rows.sum()):,} questions, {n_classes} signatures, "
          f"floor {majority_floor(sub_labels, sub_train):.1%}")
    print(f"{len(answer_rows):,} output-layer rows from the answer vocabulary")
    print(f"site: layer {layer}, {hidden_width} dimensions wide")
    print(f"Jacobian averaged over {sum(len(b['prompt_lens']) for b in batches)} prompts "
          f"in {len(batches)} batches, at the last prompt token\n")

    results: dict[str, Any] = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in SEEDS:
            vlm, weight, source = load(arm, seed, layer, cfg.device)
            span = output_basis(weight, answer_rows, max(DIMS))
            pulled = pull_back(vlm, batches, span)
            basis, turned = orthonormal(pulled), rotation(pulled, span)
            del vlm
            empty_cuda_cache()

            features, _, _ = representations(arm, seed)
            x = F.normalize(features[keep][rows].float(), dim=-1)

            entry: dict[str, Any] = {
                "output_layer": source,
                "direction_rotation": turned,
                "full": probe_accuracy(x, sub_labels, sub_train, n_classes),
                "by_dim": {},
            }
            print(f"[{arm} seed{seed}] output layer: {source}")
            print(f"{'directions turned':>26}  mean |cos| {turned:.3f}"
                  f"   ← 1.000 would mean the first stage needed no pullback")
            print(f"{f'full space ({x.size(1)})':>26}  {entry['full']:6.1%}"
                  f"   ← compare with §5.1")
            for m in DIMS:
                place = scores(
                    x,
                    basis[:, :m],
                    random_basis(x.size(1), m, BASIS_SEED),
                    sub_labels, sub_train, n_classes,
                )
                entry["by_dim"][str(m)] = place
                print(f"{f'm = {m}':>26}  "
                      f"in {place['in_output_span']:6.1%} / {place['in_random_span']:6.1%}"
                      f"   outside {place['outside_output_span']:6.1%}"
                      f" / {place['outside_random_span']:6.1%}"
                      f"   (pulled back / random)")
            results[arm][str(seed)] = entry
            print()

    # ── the comparison the verdict rests on ──────────────────────────────────
    # As in the first stage: what decides is the difference between the pulled-back span
    # and a random span of the same width, at each of the two places. A difference inside
    # says the structure alignment built is where the answer is read; a difference outside
    # says it sits where the output does not read. If the first stage's parity persists
    # here — with the approximation it was blamed on now removed — then the parity is the
    # finding, and what it says is that alignment reorganised the whole geometry rather
    # than any particular subspace of it.
    summary = summarize_places(results, DIMS, ARMS, SEEDS)
    for arm in ARMS:
        summary[arm]["direction_rotation"] = sum(
            results[arm][str(seed)]["direction_rotation"] for seed in SEEDS
        ) / len(SEEDS)
    print(f"{'':>10}" + "".join(f"{f'm={m}':>44}" for m in DIMS))
    print(f"{'':>10}" + "".join(
        f"{'in J(U)':>11}{'in rand':>11}{'out J(U)':>11}{'out rand':>11}" for _ in DIMS))
    for arm in ARMS:
        print(f"{arm:>10}" + "".join(
            f"{summary[arm][f'{where}_{m}']:>11.1%}" for m in DIMS for where in WHERE
        ))

    print("\ngain over baseline (proposal − baseline), against the random span of each width")
    for m in DIMS:
        for side in ("in", "outside"):
            u, r = (gain(summary, f"{side}_{kind}_span_{m}") for kind in ("output", "random"))
            seeds = seed_deltas(results, m, side, SEEDS)
            print(f"  m = {m:>3} {side:>7}  J(U) {u:+5.1f}pt  random {r:+5.1f}pt"
                  f"   Δ {delta(summary, m, side):+5.2f}pt"
                  f"   seeds " + " ".join(f"{value:+5.2f}" for value in seeds))
    reported = reported_gain()
    against = (
        f"← §5.1 reports {reported:+.1f}pt" if reported is not None else "← compare with §5.1"
    )
    print(f"  full space:  {gain(summary, 'full'):+5.1f}pt   {against}")

    save_json({
        "timestamp": TIMESTAMP,
        "checkpoint": CHECKPOINT_NAME,
        "stage": "pullback",
        "site_layer": layer,
        "dims": list(DIMS),
        "answer_rows": len(answer_rows),
        "jacobian_prompts": sum(len(b["prompt_lens"]) for b in batches),
        "questions": int(rows.sum()),
        "classes": n_classes,
        "floor": majority_floor(sub_labels, sub_train),
        "runs": results,
        "means": summary,
    }, OUT_PATH)
    print(f"\nwritten: {OUT_PATH}")
