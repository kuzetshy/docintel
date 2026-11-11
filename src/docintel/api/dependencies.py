"""FastAPI dependencies. Tests replace them with `app.dependency_overrides`."""

import asyncio

from docintel.config import get_settings
from docintel.llm import create_chat_model, create_embedder
from docintel.rag.service import RagService
from docintel.retrieval.store import VectorStore, create_qdrant_client

_service: RagService | None = None
_lock = asyncio.Lock()


async def get_rag_service() -> RagService:
    """Build the RAG service once per process, on the first request."""
    global _service
    async with _lock:
        if _service is None:
            settings = get_settings()
            embedder = create_embedder(settings)
            store = VectorStore(
                create_qdrant_client(settings.qdrant_url),
                settings.qdrant_collection,
                embedder.dimension,
            )
            await store.ensure_collection()
            _service = RagService(
                chat=create_chat_model(settings),
                embedder=embedder,
                store=store,
                top_k=settings.retrieval_top_k,
                score_threshold=settings.retrieval_score_threshold,
                max_tokens=settings.answer_max_tokens,
            )
    return _service
