"""Gemini adapter, tested against a stub client: no network, no key."""

from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from google import genai
from google.genai import errors, types

from docintel.llm import (
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
    Message,
)
from docintel.llm.gemini import GeminiChatModel, map_error, to_contents


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": f"status {code}"}})


class StubModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> types.GenerateContentResponse:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return cast(types.GenerateContentResponse, outcome)


def make_model(outcomes: list[Any]) -> tuple[GeminiChatModel, StubModels]:
    models = StubModels(outcomes)
    client = cast(genai.Client, SimpleNamespace(aio=SimpleNamespace(models=models)))
    model = GeminiChatModel(
        api_key="unused", model="gemini-3.5-flash-lite", client=client
    )
    model._retry_max_wait_s = 0  # no real sleeping between retries in tests
    return model, models


def response(
    text: str, prompt: int, output: int, thoughts: int = 0
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)])
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=thoughts,
        ),
    )


def test_assistant_role_becomes_model() -> None:
    contents = to_contents(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
    )

    assert [c.role for c in contents] == ["user", "model"]
    assert contents[1].parts is not None
    assert contents[1].parts[0].text == "hello"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (api_error(429), LLMRateLimitError),
        (api_error(500), LLMUnavailableError),
        (api_error(503), LLMUnavailableError),
        (api_error(408), LLMUnavailableError),
        (api_error(400), LLMRequestError),
        (api_error(403), LLMRequestError),
        (httpx2.ConnectError("refused"), LLMUnavailableError),
    ],
)
def test_error_mapping(exc: Exception, expected: type[LLMError]) -> None:
    assert type(map_error(exc)) is expected


@pytest.mark.anyio
async def test_generate_maps_usage_and_cost() -> None:
    model, stub = make_model([response("Paris", prompt=10, output=2, thoughts=3)])

    result = await model.generate(
        [Message(role="user", content="Capital of France?")],
        system="Be brief.",
        max_tokens=50,
    )

    assert result.text == "Paris"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5  # thinking tokens count as output
    assert result.cost_usd == pytest.approx((10 * 0.30 + 5 * 2.50) / 1_000_000)
    config = cast(types.GenerateContentConfig, stub.requests[0]["config"])
    assert config.system_instruction == "Be brief."
    assert config.max_output_tokens == 50


@pytest.mark.anyio
async def test_generate_retries_on_429() -> None:
    model, stub = make_model([api_error(429), response("ok", 1, 1)])

    result = await model.generate([Message(role="user", content="hi")])

    assert result.text == "ok"
    assert len(stub.requests) == 2


@pytest.mark.anyio
async def test_generate_does_not_retry_bad_request() -> None:
    model, stub = make_model([api_error(400), response("never", 1, 1)])

    with pytest.raises(LLMRequestError):
        await model.generate([Message(role="user", content="hi")])
    assert len(stub.requests) == 1
