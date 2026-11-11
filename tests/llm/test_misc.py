"""Pricing, fake embedder, settings and factory."""

import math

import pytest
from pydantic import SecretStr

from docintel.config import Settings
from docintel.llm import ChatModel, Embedder, Usage, create_chat_model, create_embedder
from docintel.llm.fake import FakeChatModel, FakeEmbedder
from docintel.llm.gemini import GeminiChatModel
from docintel.llm.pricing import estimate_cost


def test_estimate_cost_uses_per_million_prices() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert estimate_cost("gemini-3.5-flash-lite", usage) == pytest.approx(2.80)


def test_estimate_cost_unknown_model_is_none() -> None:
    assert estimate_cost("no-such-model", Usage(input_tokens=5)) is None


def test_fake_embedder_is_deterministic_unit_vectors() -> None:
    embedder = FakeEmbedder(dimension=16)

    a1, a2, b = embedder.embed_documents(["alpha", "alpha", "beta"])

    assert a1 == a2
    assert a1 != b
    assert len(a1) == 16
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0)
    assert embedder.embed_query("alpha") == a1


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_model == "gemini-2.5-flash-lite"
    assert settings.gemini_api_key is not None
    assert "secret-value" not in repr(settings)


def test_factory_builds_fakes() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="fake",
        embedding_provider="fake",
    )

    chat = create_chat_model(settings)
    embedder = create_embedder(settings)

    assert isinstance(chat, FakeChatModel)
    assert isinstance(chat, ChatModel)
    assert isinstance(embedder, FakeEmbedder)
    assert isinstance(embedder, Embedder)


def test_factory_builds_gemini() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="gemini",
        gemini_api_key=SecretStr("test-key"),
    )

    assert isinstance(create_chat_model(settings), GeminiChatModel)


def test_factory_requires_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None, llm_provider="gemini")  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_chat_model(settings)
