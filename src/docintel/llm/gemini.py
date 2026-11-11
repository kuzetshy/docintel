"""Google Gemini adapter (google-genai SDK).

This is the only module allowed to import `google.genai`.
"""

from collections.abc import AsyncIterator, Sequence

import httpx2
from google import genai
from google.genai import errors, types

from docintel.llm.base import BaseChatModel
from docintel.llm.errors import (
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMUnavailableError,
)
from docintel.llm.types import Message, RawGeneration, Usage

# Gemini calls the assistant role "model".
_ROLE_MAP = {"user": "user", "assistant": "model"}


def to_contents(messages: Sequence[Message]) -> list[types.Content]:
    """Convert our messages to Gemini `Content` objects."""
    return [
        types.Content(role=_ROLE_MAP[m.role], parts=[types.Part(text=m.content)])
        for m in messages
    ]


def map_error(exc: Exception) -> LLMError:
    """Translate SDK and network errors into provider-neutral ones."""
    if isinstance(exc, errors.APIError):
        detail = f"Gemini API error {exc.code}: {exc.message}"
        if exc.code == 429:
            return LLMRateLimitError(detail)
        if exc.code == 408 or exc.code >= 500:
            return LLMUnavailableError(detail)
        return LLMRequestError(detail)
    if isinstance(exc, httpx2.TransportError):
        return LLMUnavailableError(f"Network error talking to Gemini: {exc}")
    return LLMError(f"Unexpected Gemini error: {exc!r}")


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    if metadata is None:
        return Usage()
    # Thinking tokens are billed as output, so count them as output.
    output = (metadata.candidates_token_count or 0) + (
        metadata.thoughts_token_count or 0
    )
    return Usage(input_tokens=metadata.prompt_token_count or 0, output_tokens=output)


class GeminiChatModel(BaseChatModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        client: genai.Client | None = None,
    ) -> None:
        super().__init__(model, timeout_s=timeout_s, max_attempts=max_attempts)
        # Retries live in BaseChatModel; turn off the SDK's own so they don't stack.
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    def _config(
        self, system: str | None, max_tokens: int, temperature: float | None
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    async def _generate(
        self,
        messages: Sequence[Message],
        system: str | None,
        max_tokens: int,
        temperature: float | None,
    ) -> RawGeneration:
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=to_contents(messages),
                config=self._config(system, max_tokens, temperature),
            )
        except (errors.APIError, httpx2.TransportError) as exc:
            raise map_error(exc) from exc
        return RawGeneration(
            text=response.text or "", usage=_usage(response.usage_metadata)
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=to_contents(messages),
                config=self._config(system, max_tokens, temperature),
            )
            async for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except (errors.APIError, httpx2.TransportError) as exc:
            raise map_error(exc) from exc
