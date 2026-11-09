"""Smoke tests: every package from the project layout can be imported."""

import importlib

import pytest

MODULES = [
    "docintel",
    "docintel.agent",
    "docintel.api",
    "docintel.extraction",
    "docintel.ingestion",
    "docintel.llm",
    "docintel.observability",
    "docintel.rag",
    "docintel.retrieval",
    "docintel.safety",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    assert importlib.import_module(module).__name__ == module
