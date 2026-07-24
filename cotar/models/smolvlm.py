from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

import huggingface_hub
import PIL.Image
import torch
import torch.nn as nn
from PIL.Image import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
from transformers import logging as transformers_logging

from ..types import IGNORE_INDEX, Encoding, PaddingSide, VLMOutput

__all__ = ["Size", "SmolVLMProcessor", "SmolVLM", "build_smolvlm"]

Size = Literal["256M", "500M", "2.2B"]

_CHECKPOINTS: dict[Size, str] = {
    "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "2.2B": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}


def _silence_transformers() -> None:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    huggingface_hub.utils.disable_progress_bars()


def _disable_cuda_graph_capture_checks() -> None:
    # transformers' cache code probes `is_current_stream_capturing()` during
    # generation; without a CUDA context the probe itself raises. Stubbing it to
    # False keeps CPU-only debugging runnable and is a no-op wherever CUDA exists.
    def not_capturing() -> bool:
        return False

    torch.cuda.is_current_stream_capturing = not_capturing
    torch.cuda.graphs.is_current_stream_capturing = not_capturing


_silence_transformers()
if not torch.cuda.is_available():
    _disable_cuda_graph_capture_checks()


def _attn_implementation() -> str:
    """The fastest attention kernel this machine can actually run.

    FlashAttention-2 is a separate wheel that only builds on Linux, so probe the
    import rather than the OS: a lab machine that never got `flash-attn` installed
    then falls back to PyTorch's SDPA instead of failing at load time.
    """
    if not torch.cuda.is_available():
        return "eager"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


class SmolVLMProcessor:
    def __init__(self, size: Size = "500M") -> None:
        # Without splitting, each image is squared to 512x512 and encoded as a
        # single fixed-length token sequence — one image, one fixed prompt prefix
        # length. `_image_expansion` below relies on that constancy: it measures
        # the expansion once here rather than per sample in `_prompt_lens`.
        self._processor = AutoProcessor.from_pretrained(
            _CHECKPOINTS[size],
            do_image_splitting=False,
        )
        self._image_expansion = self._measure_image_expansion()

    @property
    def pad_token_id(self) -> int:
        return self._processor.tokenizer.pad_token_id or 0

    def encode(
        self,
        texts: list[str],
        images: list[Image],
        targets: list[str] | None = None,
        *,
        padding_side: PaddingSide = "left",
    ) -> Encoding:
        if targets is not None and padding_side != "right":
            raise ValueError(
                "Right padding is required with targets: labels and prompt lengths "
                "assume the prompt is a leading prefix of each row."
            )
        answers: Sequence[str | None] = targets if targets is not None else [None] * len(texts)
        prompts = [
            self._build_prompt(text, answer)
            for text, answer in zip(texts, answers, strict=True)
        ]

        self._processor.tokenizer.padding_side = padding_side
        images_per_sample = [[image] for image in images]
        raw = self._processor(
            text=prompts, images=images_per_sample, return_tensors="pt", padding=True
        )

        encoding: Encoding = {
            "input_ids": raw["input_ids"],
            "attention_mask": raw["attention_mask"],
            "pixel_values": raw["pixel_values"],
        }
        if "pixel_attention_mask" in raw:
            encoding["pixel_attention_mask"] = raw["pixel_attention_mask"]
        if targets is not None:
            prompt_lens = self._prompt_lens(texts, prompts)
            encoding["prompt_lens"] = prompt_lens
            encoding["labels"] = self._mask_labels(
                raw["input_ids"], raw["attention_mask"], prompt_lens
            )
        return encoding

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        return self._processor.batch_decode(token_ids, skip_special_tokens=True)

    def _build_prompt(self, text: str, target: str | None) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]
        if target is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": target}]}
            )
        return self._processor.apply_chat_template(messages, add_generation_prompt=target is None)

    def _measure_image_expansion(self) -> int:
        """How many tokens one image adds beyond its single `<image>` placeholder.

        Measured once against the real processor, rather than read off its internal
        wrapping-token constants, so this keeps working across transformers versions.
        Valid as a constant only because `do_image_splitting=False` makes every
        image expand to the same length regardless of its content — `_prompt_lens`
        depends on that to reuse this single measurement for every sample.
        """
        dummy_prompt = self._build_prompt("x", None)
        placeholder_len = len(
            self._processor.tokenizer(dummy_prompt, add_special_tokens=True).input_ids
        )
        dummy_image = PIL.Image.new("RGB", (32, 32))
        expanded = self._processor(text=[dummy_prompt], images=[[dummy_image]], return_tensors="pt")
        return int(expanded["attention_mask"].sum()) - placeholder_len

    def _prompt_lens(self, texts: list[str], full_prompts: list[str]) -> list[int]:
        tokenizer = self._processor.tokenizer
        prompt_only = [self._build_prompt(text, None) for text in texts]
        text_prompt_ids = tokenizer(prompt_only, add_special_tokens=True).input_ids
        text_full_ids = tokenizer(full_prompts, add_special_tokens=True).input_ids

        lengths: list[int] = []
        for prompt_ids, full_ids in zip(text_prompt_ids, text_full_ids, strict=True):
            if prompt_ids != full_ids[: len(prompt_ids)]:
                raise ValueError(
                    "Prompt tokens changed when the answer was appended — the "
                    "tokenizer is context-sensitive at the prompt/answer boundary.\n"
                    f"  prompt-only : {prompt_ids}\n"
                    f"  full prefix : {full_ids[: len(prompt_ids)]}"
                )
            lengths.append(len(prompt_ids) + self._image_expansion)
        return lengths

    def _mask_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: list[int]
    ) -> torch.Tensor:
        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX
        return labels


class SmolVLM(nn.Module):
    """SmolVLM, reporting a pooled representation per sample alongside its loss.

    `layers` names the transformer blocks whose hidden states are read — one, or several
    in the order given — and the position is the last prompt token. Which layers and
    which position to align is the study's central lever, so the model owns it:
    everything downstream — the alignment loss and the stability metrics alike — is
    defined over the ``(B, L, H)`` stack this returns, and none of them gets to choose
    it. The layer axis is always there, single layer or not, so nothing downstream has
    to branch on how many were asked for. Leave `layers` unset and no representation is
    produced (nor are hidden states even computed).
    """

    _model: PreTrainedModel

    def __init__(self, size: Size = "500M", layers: int | Sequence[int] | None = None) -> None:
        super().__init__()
        self.layers = _as_layers(layers)

        # Weights stay fp32 — they are what the optimizer updates, and an update of
        # relative size ~1e-3 (lr 1e-5 on weights of order 1e-2) is *below* bf16's
        # rounding step, so bf16 master weights would silently discard every step
        # and the model would not learn at all. Speed — and on CUDA, correctness —
        # comes from AMP instead: `_autocast` below casts matmuls to bf16 while the
        # parameters accumulate in fp32. This is the model's own invariant, not
        # something every caller must remember to wrap: FlashAttention-2 (see
        # `_attn_implementation`) flatly refuses fp32 tensors, so a caller that
        # forgot would not just run slow, it would crash outright.
        #
        # The checkpoint declares `model_type: idefics3`, so the concrete class must
        # be resolved from the config, not hard-coded: SmolVLM-256M/500M load as
        # Idefics3ForConditionalGeneration and only SmolVLM2-2.2B as SmolVLM*.
        self._model = AutoModelForImageTextToText.from_pretrained(
            _CHECKPOINTS[size],
            torch_dtype=torch.float32,
            attn_implementation=_attn_implementation(),
        )

        self._check_layers()

        if torch.cuda.is_available():
            self._model.gradient_checkpointing_enable()
            self._model.config.use_cache = False

    def _check_layers(self) -> None:
        """Reject a layer the model does not have, here rather than at the first forward.

        `hidden_states` is indexed with a plain integer, so an out-of-range layer would
        otherwise surface as an `IndexError` from inside `forward` — after the weights
        are on the GPU and the loaders are built.
        """
        if self.layers is None:
            return
        depth = self._model.config.get_text_config().num_hidden_layers
        if out_of_range := [layer for layer in self.layers if not -depth - 1 <= layer <= depth]:
            raise ValueError(
                f"Layers {out_of_range} are outside this model: it has {depth} blocks, so "
                f"hidden states run 0 (the embedding) to {depth} (the last block's output)."
            )

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _autocast(self) -> torch.autocast:
        """bf16 autocast on CUDA, a transparent no-op on CPU.

        Applied here rather than left to the caller, so `forward` and `generate`
        are dtype-safe on their own — train4all's `BaseTrainer` already wraps every
        call it makes in its own autocast, and nesting two bf16-on-CUDA contexts is
        a no-op, so this changes nothing for it. What it fixes is every *other*
        caller (`scripts/generate.py`, a REPL, a future script) that calls this
        model directly and has no autocast of its own.
        """
        return torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        prompt_lens: list[int] | None = None,
        **_: Any,
    ) -> VLMOutput:
        with self._autocast():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                labels=labels,
                output_hidden_states=self.layers is not None,
            )
        representation = None
        if self.layers is not None and prompt_lens is not None:
            representation = torch.stack(
                [self._pool(out.hidden_states[layer], prompt_lens) for layer in self.layers],
                dim=1,
            )
        return VLMOutput(loss=out.loss, representation=representation)

    @staticmethod
    def _pool(hidden_states: torch.Tensor, prompt_lens: list[int]) -> torch.Tensor:
        """The hidden state at each row's last prompt token — (B, T, H) → (B, H)."""
        rows = torch.arange(len(prompt_lens), device=hidden_states.device)
        last = torch.tensor(prompt_lens, device=hidden_states.device) - 1
        return hidden_states[rows, last]

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        with self._autocast():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                # Greedy: argmax at every step, neither sampled nor searched. The
                # score is exact match against a single gold answer, so sampling
                # would put a coin flip between the weights and the metric, and beam
                # search would answer with a search over sequences rather than with
                # the model itself. Both are pinned rather than inherited because a
                # checkpoint ships its own generation_config.json, which overrides
                # the library defaults and can silently turn either one on.
                do_sample=False,
                num_beams=1,
                use_cache=True,
                **kwargs,
            )
        return cast(torch.Tensor, output_ids)


def _as_layers(layers: int | Sequence[int] | None) -> tuple[int, ...] | None:
    """One layer, or several, as the tuple the rest of the class reads.

    A bare `int` is still the ordinary case and stays spellable as one. The order given
    is kept: it is the order of the representation's layer axis, and so the order every
    per-layer metric and the saved tensor are labelled in.
    """
    if layers is None:
        return None
    if isinstance(layers, int):
        return (layers,)
    chosen = tuple(layers)
    if not chosen:
        raise ValueError("`layers` is empty; pass None to run without a representation.")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"`layers` repeats a layer: {chosen}.")
    return chosen


def build_smolvlm(
    size: Size = "500M",
    layers: int | Sequence[int] | None = None,
) -> tuple[SmolVLM, SmolVLMProcessor]:
    return SmolVLM(size, layers=layers), SmolVLMProcessor(size)
