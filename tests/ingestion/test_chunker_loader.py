from pathlib import Path

import pytest

from docintel.ingestion.chunker import chunk_document
from docintel.ingestion.loader import load_directory, load_document
from docintel.ingestion.models import Document

SAMPLES = Path(__file__).parents[2] / "data" / "samples"


def make_doc(text: str) -> Document:
    return Document(id="doc1", title="T", source="t.md", text=text)


def test_chunks_never_cross_sections() -> None:
    doc = make_doc(
        "# T\n\n## Article 1: A\n\nalpha line\n\n## Article 2: B\n\nbeta line"
    )

    chunks = chunk_document(doc, size=500, overlap=50)

    assert [(c.section, c.text) for c in chunks] == [
        ("Article 1: A", "alpha line"),
        ("Article 2: B", "beta line"),
    ]


def test_chunks_respect_size_and_overlap() -> None:
    lines = [f"line {i} " + "x" * 40 for i in range(20)]
    doc = make_doc("\n".join(lines))

    chunks = chunk_document(doc, size=200, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # The last line of a chunk opens the next one (overlap).
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert second.text.startswith(first.text.splitlines()[-1])


def test_long_line_is_split_on_words() -> None:
    doc = make_doc(" ".join(["word"] * 100))

    chunks = chunk_document(doc, size=50, overlap=0)

    assert all(len(c.text) <= 50 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == ["word"] * 100


def test_chunk_ids_are_deterministic_and_unique() -> None:
    doc = make_doc("\n".join(f"line {i} " + "y" * 50 for i in range(30)))

    first = chunk_document(doc, size=200, overlap=0)
    second = chunk_document(doc, size=200, overlap=0)

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert [c.index for c in first] == list(range(len(first)))


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_document(make_doc("text"), size=100, overlap=100)


def test_load_document_takes_title_and_hashes_content(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# My Title\n\nBody", encoding="utf-8")

    doc = load_document(path)
    again = load_document(path)

    assert doc.title == "My Title"
    assert doc.source == "note.md"
    assert doc.id == again.id and len(doc.id) == 64


def test_load_document_falls_back_to_file_name(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("no heading here", encoding="utf-8")

    assert load_document(path).title == "plain"


def test_load_directory_skips_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    assert [d.source for d in load_directory(tmp_path)] == ["a.txt", "b.md"]


def test_sample_documents_chunk_by_article() -> None:
    docs = load_directory(SAMPLES)

    chunks = [c for d in docs for c in chunk_document(d, size=800, overlap=120)]

    assert {d.source for d in docs} == {"gdpr.md", "eu-ai-act.md"}
    # Each document opens with a source/licence preface (no section), then
    # every chunk belongs to an article.
    preface = [c for c in chunks if c.section is None]
    sections = [c.section for c in chunks if c.section is not None]
    assert len(preface) == len(docs)
    assert all(c.index == 0 and "eur-lex" in c.text for c in preface)
    assert sections and all(s.startswith("Article ") for s in sections)
    assert all(len(c.text) <= 800 for c in chunks)
