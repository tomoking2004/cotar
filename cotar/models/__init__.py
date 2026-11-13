"""The model the study trains, behind the contract of `cotar.types`."""

from .smolvlm import Size, SmolVLM, SmolVLMProcessor, build_smolvlm

__all__ = ["Size", "SmolVLM", "SmolVLMProcessor", "build_smolvlm"]
