"""Deterministic fakes for tests and offline development (no network, no cost)."""

import hashlib
import math
from collections.abc import AsyncIterator, Sequence

from docintel.llm.base import BaseChatModel
from docintel.llm.types import Message, RawGeneration, Usage


class FakeChatModel(BaseChatModel):
    """Returns scripted replies, or echoes the last user message.

    `failures` are raised one by one on the first calls, which lets tests
    check retry behaviour without a real provider.
    """

    def __init__(
        self,
        replies: Sequence[str] | None = None,
        *,
        failures: Sequence[Exception] = (),
        model: str = "fake",
        timeout_s: float = 5.0,
        max_attempts: int = 3,
        retry_max_wait_s: float = 0.0,
    ) -> None:
        super().__init__(
            model,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_max_wait_s=retry_max_wait_s,
        )
        self._replies = list(replies or [])
        self._failures = list(failures)
        self.calls: list[list[Message]] = []

    def _next_reply(self, messages: Sequence[Message]) -> str:
        if self._replies:
            return self._replies.pop(0)
        return f"echo: {messages[-1].content}"

    async def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float | None,
    ) -> RawGeneration:
        self.calls.append(list(messages))
        if self._failures:
            raise self._failures.pop(0)
        text = self._next_reply(messages)
        prompt_words = sum(len(m.content.split()) for m in messages)
        return RawGeneration(
            text=text,
            usage=Usage(input_tokens=prompt_words, output_tokens=len(text.split())),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        for word in self._next_reply(messages).split(" "):
            yield word + " "


class FakeEmbedder:
    """Hash-based unit vectors: same text gives the same vector, no semantics."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i % len(digest)] - 127.5 for i in range(self._dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
