"""Differentiating the readout with respect to the constrained site.

Two measurements need the same backward pass and differ only in what they keep of it.
context.md §4.5 asks *where* the answer directions land at the site, and keeps the mean
direction. §7.1 asks *how far* the readout moves when the site does, and keeps the length.
Both live here so the two cannot disagree about which layer, which position, or which
prompts they are talking about.

**One backward per direction, and the `H x H` Jacobian is never formed.** A vector-Jacobian
product is all either measurement wants, and the graph is retained only while directions
remain.

**The row taken is the site's own.** The gradient comes back for every position, but the
alignment loss touched one — the last prompt token — and the readout for a prompt depends
on that prompt's rows alone, so each row of the returned gradient is that prompt's own
`J^T u` rather than a sum over the batch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import cfg
from ..data import build_gqa_dataloader
from ..models import SmolVLM, build_smolvlm
from ..types import VLMProcessor
from .experiment import ARMS, MODEL, SEEDS, Arm, constrained_layer, weights

__all__ = [
    "jacobian_batches",
    "load_for_jacobian",
    "site",
    "vjp_at_site",
]


def site() -> tuple[int, int]:
    """The layer the nine runs constrained, and the width of its vector.

    Refused if they disagree: nine runs constraining different layers are not one
    experiment, and there would be no single site to differentiate towards.
    """
    sites = {constrained_layer(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(sites) != 1:
        raise SystemExit(f"the runs constrained different sites ({sorted(sites)}).")
    return sites.pop()


def load_for_jacobian(
    arm: Arm, seed: int, layer: int, device: str
) -> tuple[SmolVLM, torch.Tensor, str]:
    """One run's trained model, ready to differentiate, and its output layer.

    Evaluation mode, because what is being linearised is the network as it answers rather
    than as it trains — and because the model turns gradient checkpointing on for CUDA,
    which the layers apply only while training.
    """
    trained = weights(arm, seed, model=MODEL)
    vlm, _ = build_smolvlm("500M", layers=(layer,))
    vlm.load_state_dict(trained.parameters)
    return vlm.to(device).eval(), trained.output_weight, trained.output_layer


def jacobian_batches(
    processor: VLMProcessor, prompts: int, batch_size: int
) -> list[dict[str, Any]]:
    """The report-set batches to differentiate against, on the device, kept in memory.

    Held rather than re-read per run: nine models are differentiated against exactly the
    same inputs, which is what makes their readings comparable.
    """
    loader: DataLoader[Any] = build_gqa_dataloader(
        cfg.gqa.testdev_questions,
        images_dir=cfg.gqa.images,
        processor=processor,
        batch_size=batch_size,
        with_labels=True,
        limit=prompts,
        num_workers=0,
    )
    return [
        {
            key: value.to(cfg.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for batch in loader
    ]


def vjp_at_site(
    vlm: SmolVLM, batch: dict[str, Any], directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batch's gradients at the site, and the two lengths they have to be read against.

    Returns `(B, m, H)` gradients — prompt by prompt, direction by direction — beside the
    norm of each prompt's site vector and of its readout. A gradient on its own is not
    comparable across the arms: it says how far the readout moves per unit of movement at
    the site, in whatever units those two happen to have, and alignment is free to have
    changed either. The caller divides.
    """
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
        grads = []
        for j in range(directions.size(1)):
            direction = directions[:, j].to(readout.device, readout.dtype)
            grad, = torch.autograd.grad(
                readout,
                hidden,
                grad_outputs=direction.repeat(len(prompt_lens), 1),
                retain_graph=j + 1 < directions.size(1),
            )
            grads.append(grad[rows, last].float().cpu())
        # Detached before leaving the graph's scope. A norm taken on `readout` while it
        # still tracks gradients keeps the whole forward alive through the returned
        # tensor, and a caller that holds one per batch holds every batch's graph.
        site_norm = hidden[rows, last].detach().float().norm(dim=-1).cpu()
        readout_norm = readout.detach().float().norm(dim=-1).cpu()
    return torch.stack(grads, dim=1), site_norm, readout_norm
