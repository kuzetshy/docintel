"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one typed place.

    Values come from real environment variables first, then from `.env`.
    Secrets use `SecretStr`, so they never show up in logs or reprs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chat model
    llm_provider: Literal["gemini", "fake"] = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1)
    gemini_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: Literal["fastembed", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str | None = None

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docintel_chunks"

    # Chunking (characters)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    # Retrieval and answering
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    # Below this cosine similarity a chunk is treated as unrelated. If no chunk
    # passes, /ask answers "not found" without calling the LLM. 0.6 was picked
    # on the samples with bge-small: on-topic top scores 0.66-0.86, off-topic
    # 0.43-0.53 (14 questions, 2026-11-11). Recalibrate on the eval set.
    retrieval_score_threshold: float = Field(default=0.6, ge=-1, le=1)
    answer_max_tokens: int = Field(default=2048, ge=64)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
