"""Load text documents from disk (Markdown and plain text for now)."""

import hashlib
from pathlib import Path

from docintel.ingestion.models import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    title = next(
        (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
        path.stem,
    )
    return Document(
        id=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        title=title,
        source=path.name,
        text=text,
    )


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in a directory, sorted by name."""
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix in SUPPORTED_SUFFIXES
    )
    return [load_document(p) for p in paths]
