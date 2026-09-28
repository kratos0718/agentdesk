"""Thin client for any OpenAI-compatible chat API, with token and cost tracking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import settings


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * settings.llm_input_cost
                + self.output_tokens * settings.llm_output_cost) / 1_000_000

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class LLMClient:
    def __init__(self) -> None:
        from openai import OpenAI  # imported lazily so offline mode needs no SDK

        self._client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=30)
        self.model = settings.llm_model

    def chat(self, messages: list[dict], json_mode: bool = False) -> tuple[str, Usage]:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0, **kwargs)
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else Usage()
        text = resp.choices[0].message.content or ""
        # Reasoning models (Qwen3, DeepSeek-R1) prepend their thinking; keep only the reply.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip(), usage

    def chat_json(self, messages: list[dict]) -> tuple[dict, Usage]:
        text, usage = self.chat(messages, json_mode=True)
        return json.loads(text), usage


_client: LLMClient | None = None


def get_llm() -> LLMClient | None:
    """Return a client when an API key is configured, otherwise None (offline mode)."""
    global _client
    if not settings.online:
        return None
    if _client is None:
        _client = LLMClient()
    return _client
