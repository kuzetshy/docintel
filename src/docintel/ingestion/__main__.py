"""CLI: `python -m docintel.ingestion <directory>`."""

import argparse
import asyncio
from pathlib import Path

from docintel.config import get_settings
from docintel.ingestion.pipeline import ingest_directory
from docintel.llm import create_embedder
from docintel.retrieval.store import VectorStore, create_qdrant_client


async def _run(directory: Path) -> None:
    settings = get_settings()
    embedder = create_embedder(settings)
    store = VectorStore(
        create_qdrant_client(settings.qdrant_url),
        settings.qdrant_collection,
        embedder.dimension,
    )
    report = await ingest_directory(
        directory,
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(
        f"Ingested {report.documents} documents, {report.chunks} chunks; "
        f"collection now holds {await store.count()} chunks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into Qdrant.")
    parser.add_argument("directory", type=Path)
    asyncio.run(_run(parser.parse_args().directory))


if __name__ == "__main__":
    main()
