from __future__ import annotations

import os

from devmind.config import load_environment
from devmind.llm.base import LLMProvider
from devmind.llm.deterministic import DeterministicProvider
from devmind.llm.openai_provider import OpenAIProvider


def create_llm_provider(
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    load_environment()
    provider_name = (provider or os.getenv("DEVMIND_LLM_PROVIDER", "deterministic")).lower()
    if provider_name == "deterministic":
        return DeterministicProvider()
    if provider_name == "openai":
        model_name = model or os.getenv("DEVMIND_LLM_MODEL", "gpt-5.5")
        return OpenAIProvider(model=model_name)
    raise ValueError(f"Unsupported LLM provider: {provider_name}")
