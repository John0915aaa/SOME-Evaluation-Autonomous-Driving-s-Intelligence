"""Shared OpenAI API helpers for subjective evaluation."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from openai import OpenAI

GPT4O_MODEL = "gpt-4o"


def _read_api_keys() -> list[str]:
    keys_text = os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY")
    if not keys_text:
        raise RuntimeError("Please set OPENAI_API_KEY or OPENAI_API_KEYS before running LLM evaluation.")
    keys = [key.strip() for key in keys_text.replace(";", ",").replace("\n", ",").split(",") if key.strip()]
    if not keys:
        raise RuntimeError("No valid API key found in OPENAI_API_KEY or OPENAI_API_KEYS.")
    return keys


class OpenAIClientPool:
    def __init__(self) -> None:
        self.api_keys = _read_api_keys()
        self.current_key_idx = 0

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_keys[self.current_key_idx])

    def rotate_key(self) -> None:
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)


def call_chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    client_pool: OpenAIClientPool,
    model: str = GPT4O_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            request = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": list(messages),
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            completion = client_pool.get_client().chat.completions.create(**request)
            content = completion.choices[0].message.content
            if content:
                return content.strip()
            raise RuntimeError("Empty response from LLM.")
        except Exception as exc:
            last_error = exc
            client_pool.rotate_key()
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"LLM request failed after {max_retries} retries: {last_error}") from last_error
