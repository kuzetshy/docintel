"""Qdrant-backed chunk store.

This is the only module allowed to import `qdrant_client`.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, models

from docintel.ingestion.models import Chunk


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


def create_qdrant_client(url: str) -> AsyncQdrantClient:
    """`:memory:` gives an in-process store (tests); anything else is a URL."""
    if url == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(url=url)


class VectorStore:
    def __init__(
        self, client: AsyncQdrantClient, collection: str, dimension: int
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    async def ensure_collection(self) -> None:
        """Create the collection if missing. Safe to call many times."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )
        # Index the field we filter on when replacing a document.
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    async def replace_source(
        self, source: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Make the store hold exactly these chunks for `source`.

        Old chunks of the same source (e.g. a previous version of the file)
        are deleted first, so re-ingesting never leaves stale or duplicate data.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchValue(value=source)
                        )
                    ]
                )
            ),
        )
        if not chunks:
            return
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=chunk.id, vector=list(vector), payload=chunk.model_dump()
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    async def search(
        self, vector: Sequence[float], *, top_k: int, score_threshold: float
    ) -> list[ScoredChunk]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    async def count(self) -> int:
        result = await self._client.count(collection_name=self._collection, exact=True)
        return result.count
