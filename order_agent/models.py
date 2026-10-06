"""Pick the chat model from a "provider:name" string, so the graph doesn't care which one runs."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

DEFAULT = "ollama:gpt-oss:20b"


def load(spec: str = DEFAULT) -> BaseChatModel:
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # OLLAMA_HOST lets a container reach Ollama on the host or in another container.
        return ChatOllama(model=name, temperature=0,
                          base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, max_tokens=2048)  # reads ANTHROPIC_API_KEY
    raise ValueError(f"Unknown provider {provider!r}; use ollama:<model> or anthropic:<model>.")
