"""Build the configured chat model and embedder from settings.

Provider modules are imported lazily, so running with `fake` providers
does not import (or require) any provider SDK.
"""

from docintel.config import Settings
from docintel.llm.base import ChatModel, Embedder


def create_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "fake":
        from docintel.llm.fake import FakeChatModel

        return FakeChatModel()

    if settings.gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY is not set (see .env.example)")
    from docintel.llm.gemini import GeminiChatModel

    return GeminiChatModel(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        max_attempts=settings.llm_max_attempts,
    )


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        from docintel.llm.fake import FakeEmbedder

        return FakeEmbedder()

    from docintel.llm.embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder(
        settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
