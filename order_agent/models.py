"""Pick the chat model from a "provider:name" string, so the graph doesn't care which one runs."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

DEFAULT = "ollama:gpt-oss:20b"
# Per call: the most tokens one reply may produce (reasoning included) and the longest wait for
# the server, so one degenerate generation can't stall a run. A normal reply uses well under 1,000.
MAX_TOKENS = 4096
TIMEOUT_S = 120.0
MIN_TOKENS = 512  # below this, ordinary replies would be cut off
ENV = "ORDER_AGENT_MODEL"


def configured() -> str:
    """The model the API and the CLI use: ORDER_AGENT_MODEL if set, else the default."""
    return os.environ.get(ENV) or DEFAULT


def load(spec: str = DEFAULT, max_tokens: int = MAX_TOKENS, timeout: float = TIMEOUT_S) -> BaseChatModel:
    if max_tokens < MIN_TOKENS or timeout <= 0:
        raise ValueError(f"max_tokens must be at least {MIN_TOKENS} and timeout above 0.")
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # OLLAMA_HOST lets a container reach Ollama on the host or in another container.
        return ChatOllama(model=name, temperature=0, num_predict=max_tokens, client_kwargs={"timeout": timeout},
                          base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # model/max_tokens/timeout are pydantic aliases the type stubs don't list. Reads ANTHROPIC_API_KEY.
        return ChatAnthropic(model=name, max_tokens=max_tokens, timeout=timeout)  # type: ignore[call-arg]
    raise ValueError(f"Unknown provider {provider!r}; use ollama:<model> or anthropic:<model>.")


def hit_cap(message) -> bool:
    """True when a reply stopped because it reached the token cap, not because it was done."""
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("done_reason") == "length" or meta.get("stop_reason") == "max_tokens"
