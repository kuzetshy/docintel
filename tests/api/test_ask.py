from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from docintel.api.dependencies import get_rag_service
from docintel.api.main import app
from docintel.llm import LLMRateLimitError, LLMRequestError
from docintel.rag.service import Answer, RagService


class StubService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[tuple[str, int | None]] = []

    async def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        self.questions.append((question, top_k))
        if self.error:
            raise self.error
        return Answer(answer="42 [1]", found=True, citations=[], latency_ms=1.0)


@pytest.fixture
def stub() -> Iterator[StubService]:
    service = StubService()

    async def override() -> RagService:
        return service  # type: ignore[return-value]

    app.dependency_overrides[get_rag_service] = override
    yield service
    app.dependency_overrides.clear()


def test_ask_returns_answer(stub: StubService) -> None:
    response = TestClient(app).post("/ask", json={"question": "Why?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42 [1]"
    assert body["found"] is True
    assert stub.questions == [("Why?", 3)]


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}],
)
def test_ask_validates_input(stub: StubService, payload: dict[str, object]) -> None:
    response = TestClient(app).post("/ask", json=payload)

    assert response.status_code == 422
    assert stub.questions == []


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (LLMRateLimitError("429 from provider"), "rate limited"),
        (LLMRequestError("bad key sk-secret"), "unavailable"),
    ],
)
def test_llm_errors_become_503_without_details(
    stub: StubService, error: Exception, fragment: str
) -> None:
    stub.error = error

    response = TestClient(app).post("/ask", json={"question": "Why?"})

    assert response.status_code == 503
    assert fragment in response.json()["detail"]
    assert "sk-secret" not in response.text
