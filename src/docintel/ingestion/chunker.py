"""Line-based chunking that respects Markdown sections.

Rules, in order:
1. A chunk never crosses a `## ` heading, so every chunk belongs to exactly
   one section (for legal texts: one article).
2. Lines are packed into a chunk until adding the next one would exceed
   `size` characters.
3. The next chunk starts with the last lines of the previous one, up to
   `overlap` characters, so a sentence cut at a boundary keeps its context.
4. A single line longer than `size` is split on word boundaries.

Structure-aware parsing of PDF/DOCX comes in week 2; this is enough for
Markdown and plain text.
"""

import uuid

from docintel.ingestion.models import Chunk, Document

# Fixed namespace: the same document and index always give the same chunk id.
_CHUNK_NAMESPACE = uuid.UUID("6f1c2b1e-6b8f-4b55-9a51-3d3f6c1f0a10")


def _split_long_line(line: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in line.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _pack(lines: list[str], size: int, overlap: int) -> list[str]:
    """Pack lines into chunks of at most `size` characters with overlap."""
    units: list[str] = []
    for line in lines:
        units.extend(_split_long_line(line, size) if len(line) > size else [line])

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n".join([*current, unit])) > size:
            chunks.append("\n".join(current))
            # Carry trailing lines over as overlap, if they leave room.
            carried: list[str] = []
            for previous in reversed(current):
                if len("\n".join([previous, *carried, unit])) > size:
                    break
                if len("\n".join([previous, *carried])) > overlap:
                    break
                carried.insert(0, previous)
            current = carried
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Split text into (heading, body lines). Lines before any `## ` get None."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("# "):
            continue  # document title, already stored on the Document
        else:
            sections[-1][1].append(line)
    return [(heading, body) for heading, body in sections if body]


def chunk_document(
    document: Document, size: int = 800, overlap: int = 120
) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    for heading, lines in _sections(document.text):
        for text in _pack(lines, size, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document.id}:{index}")),
                    document_id=document.id,
                    title=document.title,
                    source=document.source,
                    section=heading,
                    index=index,
                    text=text,
                )
            )
    return chunks
