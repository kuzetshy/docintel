"""FastAPI application entry point."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DocIntel", version="0.1.0")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@app.get("/health")
def health() -> HealthResponse:
    """Report that the API process is up."""
    return HealthResponse()
