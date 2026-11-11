from collections.abc import AsyncIterator

import pytest

from docintel.llm.fake import FakeEmbedder
from docintel.retrieval.store import VectorStore, create_qdrant_client


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests (marked `anyio`) on asyncio only."""
    return "asyncio"


@pytest.fixture
async def store() -> AsyncIterator[VectorStore]:
    """A fresh in-memory Qdrant collection per test."""
    client = create_qdrant_client(":memory:")
    vector_store = VectorStore(client, "test_chunks", FakeEmbedder().dimension)
    await vector_store.ensure_collection()
    yield vector_store
    await client.close()
