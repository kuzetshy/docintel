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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
