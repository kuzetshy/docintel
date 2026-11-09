# DocIntel

Portfolio project 
A document intelligence platform: RAG with citations, structured extraction,
a LangGraph agent, automated evaluation and tracing.

Full plan: docs/plan.md. Read it before starting any task.

## Stack
Python 3.12, uv, FastAPI, Pydantic, LangChain / LangGraph / LlamaIndex,
Qdrant (or pgvector), PostgreSQL, Langfuse, Docker Compose, GitHub Actions,
pytest, ruff, mypy.

## Rules
- Type hints everywhere. Code must pass `ruff` and `mypy`.
- All LLM calls go through `src/docintel/llm`. No direct provider SDK calls elsewhere.
- Secrets live only in `.env` (git-ignored). Never hardcode or commit keys.
- Every feature ships with tests. Build a thin vertical slice first, then widen.
- Quality metrics are computed in `eval/` and recorded in README.
- Small commits, conventional commit messages (feat:, fix:, chore:, docs:, test:).
- Before large changes, show a short plan and wait for confirmation.
- Use only open or synthetic data. No real personal data.
- Prefer cheap models during development to limit API cost.

## Commands
- Install: `uv sync`
- Run API: `uv run uvicorn docintel.api.main:app --reload`
- Tests: `uv run pytest`
- Lint/types: `uv run ruff check . && uv run mypy src`
- Stack: `docker compose up -d`

## Current stage (update after every changes!)
Week 1, in progress.
Done: repo skeleton (uv, src/docintel packages, eval/, tests/, infra/),
ruff, mypy (strict) and pytest with a smoke test.
Next: pre-commit secret scanning, .env.example, FastAPI /health, CI,
Docker Compose (API, Qdrant, Postgres, Langfuse), LLM abstraction,
basic /ask endpoint.