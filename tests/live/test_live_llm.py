"""Live tests against real Gemini and a real fastembed model.

Skipped by default (network, model download, rate limits). Run with:

    uv run pytest -m live
"""

import math

import pytest

from docintel.config import Settings
from docintel.llm import Message, create_chat_model, create_embedder

pytestmark = pytest.mark.live


@pytest.mark.anyio
async def test_gemini_answers() -> None:
    settings = Settings(llm_provider="gemini")
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    model = create_chat_model(settings)

    result = await model.generate(
        [Message(role="user", content="What is 2 + 2? Reply with the number only.")],
        max_tokens=20,
    )

    assert "4" in result.text
    assert result.usage.input_tokens > 0
    assert result.cost_usd is not None


def test_fastembed_puts_similar_texts_closer() -> None:
    embedder = create_embedder(Settings(embedding_provider="fastembed"))

    query = embedder.embed_query("How long can personal data be stored?")
    related, unrelated = embedder.embed_documents(
        [
            "Personal data shall be kept no longer than necessary.",
            "The recipe calls for two cups of flour.",
        ]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.hypot(*a) * math.hypot(*b))

    assert len(query) == embedder.dimension == 384
    assert cosine(query, related) > cosine(query, unrelated)
