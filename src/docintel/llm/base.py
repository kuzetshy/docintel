"""Interfaces of the LLM layer and the shared call pipeline.

`ChatModel` and `Embedder` are the only types the rest of the app depends on.
`BaseChatModel` implements the parts every provider needs (timeout, retries,
latency, cost) once, so an adapter only translates requests and errors.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from docintel.llm.errors import LLMError, LLMTimeoutError
from docintel.llm.pricing import estimate_cost
from docintel.llm.types import Generation, Message, RawGeneration


@runtime_checkable
class ChatModel(Protocol):
    """Anything that can answer a conversation."""

    @property
    def model(self) -> str: ...

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> Generation: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Documents and queries are embedded separately: many embedding models
    expect a different prefix or instruction for each.
    """

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.retryable


class BaseChatModel(ABC):
    """Template for chat adapters: subclasses implement `_generate` and `stream`."""

    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        retry_max_wait_s: float = 10.0,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._retry_max_wait_s = retry_max_wait_s

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> Generation:
        """Call the provider with a per-attempt timeout and retries.

        Only errors marked `retryable` (429, 5xx, timeouts) are retried, with
        exponential backoff plus jitter. Latency covers all attempts.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        started = time.perf_counter()
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1, max=self._retry_max_wait_s),
            reraise=True,
        )
        raw: RawGeneration | None = None
        async for attempt in retrying:
            with attempt:
                raw = await self._generate_with_timeout(
                    messages, system, max_tokens, temperature
                )
        assert raw is not None  # tenacity either returns a result or raises

        return Generation(
            text=raw.text,
            model=self._model,
            usage=raw.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=estimate_cost(self._model, raw.usage),
        )

    async def _generate_with_timeout(
        self,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float | None,
    ) -> RawGeneration:
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._generate(messages, system, max_tokens, temperature)
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"{self._model} did not answer within {self._timeout_s}s"
            ) from exc

    @abstractmethod
    async def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float | None,
    ) -> RawGeneration:
        """One provider call. Must raise `LLMError` subclasses on failure."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive. Not retried: chunks may be shown."""
