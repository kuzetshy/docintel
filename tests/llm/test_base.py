"""The shared pipeline in BaseChatModel: retries, timeouts, latency, cost."""

import asyncio
from collections.abc import Sequence

import pytest

from docintel.llm import (
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
)
from docintel.llm.fake import FakeChatModel
from docintel.llm.types import RawGeneration

pytestmark = pytest.mark.anyio

HELLO = [Message(role="user", content="hello there")]


async def test_returns_text_usage_latency_and_cost() -> None:
    model = FakeChatModel(["general kenobi"])

    result = await model.generate(HELLO)

    assert result.text == "general kenobi"
    assert result.model == "fake"
    assert result.usage.input_tokens == 2
    assert result.usage.output_tokens == 2
    assert result.latency_ms >= 0
    assert result.cost_usd == 0.0


async def test_retries_retryable_errors_then_succeeds() -> None:
    model = FakeChatModel(
        ["ok"],
        failures=[LLMRateLimitError("429"), LLMUnavailableError("503")],
        max_attempts=3,
    )

    result = await model.generate(HELLO)

    assert result.text == "ok"
    assert len(model.calls) == 3


async def test_gives_up_after_max_attempts() -> None:
    model = FakeChatModel(
        failures=[LLMRateLimitError("429")] * 5,
        max_attempts=2,
    )

    with pytest.raises(LLMRateLimitError):
        await model.generate(HELLO)
    assert len(model.calls) == 2


async def test_does_not_retry_request_errors() -> None:
    model = FakeChatModel(failures=[LLMRequestError("bad api key")])

    with pytest.raises(LLMRequestError):
        await model.generate(HELLO)
    assert len(model.calls) == 1


async def test_timeout_is_converted_and_retried() -> None:
    class SlowModel(FakeChatModel):
        async def _generate(
            self,
            messages: Sequence[Message],
            system: str | None,
            max_tokens: int,
            temperature: float | None,
        ) -> RawGeneration:
            self.calls.append(list(messages))
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

    model = SlowModel(timeout_s=0.01, max_attempts=2)

    with pytest.raises(LLMTimeoutError):
        await model.generate(HELLO)
    assert len(model.calls) == 2


async def test_rejects_empty_messages() -> None:
    with pytest.raises(ValueError):
        await FakeChatModel().generate([])


async def test_stream_yields_chunks() -> None:
    chunks = [c async for c in FakeChatModel(["a b c"]).stream(HELLO)]

    assert "".join(chunks).split() == ["a", "b", "c"]
