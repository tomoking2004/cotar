"""The contract between a model implementation and everything that uses one."""

from __future__ import annotations

from typing import (
    Any,
    Final,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import torch
from PIL.Image import Image

__all__ = [
    "IGNORE_INDEX",
    "VLM",
    "Encoding",
    "PaddingSide",
    "VLMOutput",
    "VLMProcessor",
]

# The label value cross-entropy skips: prompt tokens, image tokens, and padding.
IGNORE_INDEX: Final = -100

PaddingSide = Literal["left", "right"]


class Encoding(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    pixel_attention_mask: NotRequired[torch.Tensor]
    labels: NotRequired[torch.Tensor]
    prompt_lens: NotRequired[list[int]]


class VLMOutput(TypedDict):
    """`representation` is the study's object of interest: `(B, L, H)` — one pooled
    vector per sample per constrained layer, taken at the layers and position the model
    was configured to read, in the order of `VLM.layers`. The layer axis is present
    even when a single layer was asked for, so nothing downstream branches on the
    count. It is `None` when the model was built without a layer, or when the batch
    carries no prompt lengths to pool at.
    """

    loss: torch.Tensor | None
    representation: torch.Tensor | None


@runtime_checkable
class VLMProcessor(Protocol):
    def encode(
        self,
        texts: list[str],
        images: list[Image],
        targets: list[str] | None = None,
        *,
        padding_side: PaddingSide = "left",
    ) -> Encoding:
        ...

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        ...

    @property
    def pad_token_id(self) -> int:
        ...


@runtime_checkable
class VLM(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @property
    def layers(self) -> tuple[int, ...] | None:
        """The layers `representation` is read at, in its layer-axis order — `None`
        when the model produces none. Part of the contract because the metrics name
        themselves after these, and only the model knows them.
        """
        ...

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        prompt_lens: list[int] | None = None,
        **kwargs: Any,
    ) -> VLMOutput:
        ...

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...
