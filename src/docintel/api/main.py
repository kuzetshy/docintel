"""FastAPI application entry point."""

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from docintel.api.dependencies import get_rag_service
from docintel.llm import LLMError, LLMRateLimitError
from docintel.rag.service import Answer, RagService

app = FastAPI(title="DocIntel", version="0.1.0")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Provider failures become 503 for the client, without leaking details."""
    detail = (
        "The language model is rate limited, try again shortly."
        if isinstance(exc, LLMRateLimitError)
        else "The language model is unavailable."
    )
    return JSONResponse(status_code=503, content={"detail": detail})


@app.get("/health")
def health() -> HealthResponse:
    """Report that the API process is up."""
    return HealthResponse()


@app.post("/ask")
async def ask(
    request: AskRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> Answer:
    """Answer a question from the ingested documents, with citations."""
    return await service.ask(request.question, top_k=request.top_k)
