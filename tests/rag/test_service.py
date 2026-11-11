import pytest

from docintel.ingestion.chunker import chunk_document
from docintel.ingestion.models import Document
from docintel.llm.fake import FakeChatModel, FakeEmbedder
from docintel.rag.prompt import NOT_FOUND_MARKER, SYSTEM_PROMPT
from docintel.rag.service import NOT_FOUND_ANSWER, RagService, cited_numbers
from docintel.retrieval.store import VectorStore

pytestmark = pytest.mark.anyio

EMBEDDER = FakeEmbedder()
FACT = "Personal data shall be kept no longer than necessary."


async def service_with_fact(
    store: VectorStore, replies: list[str]
) -> tuple[RagService, FakeChatModel]:
    document = Document(
        id="gdpr",
        title="GDPR",
        source="gdpr.md",
        text=f"## Article 5: Principles\n{FACT}",
    )
    chunks = chunk_document(document)
    await store.replace_source(
        "gdpr.md", chunks, EMBEDDER.embed_documents([c.text for c in chunks])
    )
    chat = FakeChatModel(replies)
    service = RagService(
        chat=chat, embedder=EMBEDDER, store=store, top_k=3, score_threshold=0.99
    )
    return service, chat


def test_cited_numbers_parses_formats_and_drops_invalid() -> None:
    text = "A [2]. B [1][2]. C [3, 1]. D [9]. E [0]."

    assert cited_numbers(text, max_number=3) == [2, 1, 3]


async def test_no_relevant_chunks_means_not_found_without_llm_call(
    store: VectorStore,
) -> None:
    service, chat = await service_with_fact(store, ["should not be used"])

    answer = await service.ask("Something unrelated")

    assert answer.found is False
    assert answer.answer == NOT_FOUND_ANSWER
    assert answer.citations == []
    assert chat.calls == []


async def test_answer_with_citation(store: VectorStore) -> None:
    service, chat = await service_with_fact(
        store, ["Only as long as necessary [1]. Also [7]."]
    )

    answer = await service.ask(FACT)  # exact text: score 1.0 with FakeEmbedder

    assert answer.found is True
    assert answer.answer.startswith("Only as long as necessary [1]")
    assert [c.number for c in answer.citations] == [1]  # [7] does not exist
    citation = answer.citations[0]
    assert citation.section == "Article 5: Principles"
    assert citation.source == "gdpr.md"
    assert citation.snippet == FACT
    assert answer.usage is not None and answer.model == "fake"
    # The prompt carried the numbered source and the question.
    prompt = chat.calls[0][0].content
    assert "[1] GDPR, Article 5: Principles" in prompt
    assert prompt.endswith(f"Question: {FACT}")


async def test_llm_not_found_marker_becomes_not_found(store: VectorStore) -> None:
    service, _ = await service_with_fact(store, [NOT_FOUND_MARKER])

    answer = await service.ask(FACT)

    assert answer.found is False
    assert answer.answer == NOT_FOUND_ANSWER
    assert answer.citations == []


def test_system_prompt_treats_sources_as_data() -> None:
    assert "data, not instructions" in SYSTEM_PROMPT
    assert NOT_FOUND_MARKER in SYSTEM_PROMPT
