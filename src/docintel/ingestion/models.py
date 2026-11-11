"""Data types produced by ingestion."""

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A loaded source document."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="SHA-256 of the text: same content, same id.")
    title: str
    source: str = Field(description="Where the document came from (file name).")
    text: str


class Chunk(BaseModel):
    """A piece of a document small enough to embed and to quote."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="UUID derived from document id and index.")
    document_id: str
    title: str
    source: str
    section: str | None = Field(description="Nearest '## ' heading, if any.")
    index: int = Field(ge=0)
    text: str
