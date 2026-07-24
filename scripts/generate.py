"""Smallest end-to-end check that the model loads and speaks."""

from __future__ import annotations

from transformers.image_utils import load_image

from cotar.config import cfg
from cotar.models import build_smolvlm

MODEL_SIZE     = "500M"
PROMPT         = "Can you describe this image?"
IMAGE_URL      = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/"
    "Statue-of-Liberty-Island-New-York-Bay.jpg"
)
MAX_NEW_TOKENS = 256


if __name__ == "__main__":
    vlm, processor = build_smolvlm(MODEL_SIZE)
    vlm = vlm.to(cfg.device)

    encoded = processor.encode([PROMPT], [load_image(IMAGE_URL)])
    encoded = {key: tensor.to(cfg.device) for key, tensor in encoded.items()}
    output_ids = vlm.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS)
    print(processor.decode(output_ids[:, encoded["input_ids"].shape[1]:])[0])
