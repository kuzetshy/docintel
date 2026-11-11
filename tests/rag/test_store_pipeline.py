from pathlib import Path

import pytest

from docintel.ingestion.chunker import chunk_document
from docintel.ingestion.models import Document
from docintel.ingestion.pipeline import ingest_directory
from docintel.llm.fake import FakeEmbedder
from docintel.retrieval.store import VectorStore

pytestmark = pytest.mark.anyio

SAMPLES = Path(__file__).parents[2] / "data" / "samples"
EMBEDDER = FakeEmbedder()


def doc(text: str, source: str = "a.md") -> Document:
    return Document(id=f"id-{hash(text)}", title="A", source=source, text=text)


async def put(store: VectorStore, document: Document) -> None:
    chunks = chunk_document(document, size=100, overlap=0)
    await store.replace_source(
        document.source, chunks, EMBEDDER.embed_documents([c.text for c in chunks])
    )


async def test_ensure_collection_is_idempotent(store: VectorStore) -> None:
    await store.ensure_collection()
    await store.ensure_collection()

    assert await store.count() == 0


async def test_search_finds_exact_text_with_full_score(store: VectorStore) -> None:
    await put(store, doc("alpha\nbeta"))

    hits = await store.search(
        EMBEDDER.embed_query("alpha\nbeta"), top_k=3, score_threshold=0.99
    )

    assert len(hits) == 1
    assert hits[0].chunk.text == "alpha\nbeta"
    assert hits[0].score == pytest.approx(1.0)


async def test_reingesting_same_source_replaces_old_chunks(store: VectorStore) -> None:
    await put(store, doc("\n".join(f"old line {i} " + "o" * 30 for i in range(10))))
    old_count = await store.count()

    await put(store, doc("new short text"))

    assert old_count > 1
    assert await store.count() == 1


async def test_other_sources_are_untouched(store: VectorStore) -> None:
    await put(store, doc("first", source="a.md"))
    await put(store, doc("second", source="b.md"))
    await put(store, doc("first again", source="a.md"))

    assert await store.count() == 2


async def test_replace_source_rejects_mismatched_vectors(store: VectorStore) -> None:
    chunks = chunk_document(doc("text"), size=100, overlap=0)

    with pytest.raises(ValueError):
        await store.replace_source("a.md", chunks, [])


async def test_ingest_samples_is_idempotent(store: VectorStore) -> None:
    async def run() -> int:
        report = await ingest_directory(
            SAMPLES, embedder=EMBEDDER, store=store, chunk_size=800, chunk_overlap=120
        )
        assert report.documents == 2
        return report.chunks

    first = await run()
    second = await run()

    assert first == second == await store.count()
