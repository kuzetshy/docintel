"""Question answering over the document store (thin RAG slice).

retrieve -> (nothing relevant? "not found" without an LLM call)
         -> prompt with numbered sources -> LLM -> parse citations
"""

import asyncio
import re
import time

from pydantic import BaseModel, ConfigDict, Field

from docintel.llm import ChatModel, Embedder, Message, Usage
from docintel.rag.prompt import NOT_FOUND_MARKER, SYSTEM_PROMPT, build_user_message
from docintel.retrieval.store import ScoredChunk, VectorStore

NOT_FOUND_ANSWER = "I could not find the answer in the documents."

# Matches [1], [2][3] and [1, 2].
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SNIPPET_CHARS = 300


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int = Field(description="The [n] used in the answer text.")
    title: str
    section: str | None
    source: str
    chunk_index: int
    score: float
    snippet: str


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    found: bool
    citations: list[Citation]
    model: str | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    latency_ms: float


def cited_numbers(text: str, max_number: int) -> list[int]:
    """Source numbers cited in `text`, in order of first use, valid ones only."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(text):
        for raw in group.split(","):
            number = int(raw)
            if 1 <= number <= max_number and number not in seen:
                seen.append(number)
    return seen


def _citation(number: int, hit: ScoredChunk) -> Citation:
    chunk = hit.chunk
    snippet = chunk.text
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return Citation(
        number=number,
        title=chunk.title,
        section=chunk.section,
        source=chunk.source,
        chunk_index=chunk.index,
        score=round(hit.score, 4),
        snippet=snippet,
    )


class RagService:
    def __init__(
        self,
        *,
        chat: ChatModel,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.6,
        max_tokens: int = 2048,
    ) -> None:
        self._chat = chat
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._max_tokens = max_tokens

    async def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        started = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        vector = await asyncio.to_thread(self._embedder.embed_query, question)
        hits = await self._store.search(
            vector, top_k=top_k or self._top_k, score_threshold=self._score_threshold
        )
        if not hits:
            return Answer(
                answer=NOT_FOUND_ANSWER,
                found=False,
                citations=[],
                latency_ms=elapsed_ms(),
            )

        generation = await self._chat.generate(
            [Message(role="user", content=build_user_message(question, hits))],
            system=SYSTEM_PROMPT,
            max_tokens=self._max_tokens,
        )
        text = generation.text.strip()
        found = bool(text) and NOT_FOUND_MARKER not in text
        citations = (
            [_citation(n, hits[n - 1]) for n in cited_numbers(text, len(hits))]
            if found
            else []
        )
        return Answer(
            answer=text if found else NOT_FOUND_ANSWER,
            found=found,
            citations=citations,
            model=generation.model,
            usage=generation.usage,
            cost_usd=generation.cost_usd,
            latency_ms=elapsed_ms(),
        )
