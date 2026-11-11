"""Ingestion pipeline: load -> chunk -> embed -> store."""

import asyncio
from pathlib import Path

from pydantic import BaseModel

from docintel.ingestion.chunker import chunk_document
from docintel.ingestion.loader import load_directory
from docintel.llm import Embedder
from docintel.retrieval.store import VectorStore


class IngestReport(BaseModel):
    documents: int
    chunks: int


async def ingest_directory(
    directory: Path,
    *,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestReport:
    await store.ensure_collection()
    documents = load_directory(directory)
    total_chunks = 0
    for document in documents:
        chunks = chunk_document(document, size=chunk_size, overlap=chunk_overlap)
        # Embedding is CPU-bound and synchronous: keep the event loop free.
        vectors = await asyncio.to_thread(
            embedder.embed_documents, [c.text for c in chunks]
        )
        await store.replace_source(document.source, chunks, vectors)
        total_chunks += len(chunks)
    return IngestReport(documents=len(documents), chunks=total_chunks)
