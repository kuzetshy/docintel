"""End-to-end RAG on the sample documents: real fastembed, real Gemini,
in-memory Qdrant. Run with `uv run pytest -m live`."""

from pathlib import Path

import pytest

from docintel.config import Settings
from docintel.ingestion.pipeline import ingest_directory
from docintel.llm import create_chat_model, create_embedder
from docintel.rag.service import RagService
from docintel.retrieval.store import VectorStore, create_qdrant_client

pytestmark = [pytest.mark.live, pytest.mark.anyio]

SAMPLES = Path(__file__).parents[2] / "data" / "samples"


@pytest.fixture
async def service() -> RagService:
    settings = Settings()
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    embedder = create_embedder(settings)
    store = VectorStore(create_qdrant_client(":memory:"), "live", embedder.dimension)
    await ingest_directory(
        SAMPLES,
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return RagService(
        chat=create_chat_model(settings),
        embedder=embedder,
        store=store,
        score_threshold=settings.retrieval_score_threshold,
    )


async def test_answers_storage_question_from_gdpr_article_5(
    service: RagService,
) -> None:
    answer = await service.ask("How long can personal data be stored?")

    assert answer.found, answer.answer
    assert any(
        c.source == "gdpr.md" and c.section and c.section.startswith("Article 5")
        for c in answer.citations
    ), answer.citations


async def test_off_topic_question_is_not_found(service: RagService) -> None:
    answer = await service.ask("What is the capital of France?")

    assert not answer.found, answer.answer
