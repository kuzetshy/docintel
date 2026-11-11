# DocIntel

Portfolio project 
A document intelligence platform: RAG with citations, structured extraction,
a LangGraph agent, automated evaluation and tracing.

Full plan: docs/plan.md (local only, git-ignored). Read it before starting
any task. The public summary of the plan lives in README.

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
- At the end of every session, append a study section to `docs/study.md`
  (local only, git-ignored; in Russian): what we built and why, key concepts to read up on, links to
  official docs, and likely interview questions with short answers.

## Commands
- Install: `uv sync`
- Git hooks (once after clone): `uv run pre-commit install`
- Run API: `uv run uvicorn docintel.api.main:app --reload`
- Tests: `uv run pytest` (offline; live provider tests: `uv run pytest -m live`)
- Lint/types: `uv run ruff check . && uv run mypy src`
- Stack (api, postgres, qdrant): `docker compose up -d`
- Stack + Langfuse: `docker compose --profile observability up -d`
- Ingest samples: `docker compose exec api python -m docintel.ingestion /app/data/samples`
  (locally: `uv run python -m docintel.ingestion data/samples`)

## Current stage (update after every changes!)
Week 1 done: `docker compose up` + ingest gives working Q&A with citations.
Done: repo skeleton, ruff/mypy strict/pytest, pre-commit gitleaks, CI
(quality, docker smoke test, secrets), Dockerfile + Compose (api, postgres,
qdrant; Langfuse v4 stack under the `observability` profile), LLM layer
(`src/docintel/llm`: ChatModel/Embedder protocols, Gemini adapter on
gemini-3.5-flash-lite free tier, local fastembed bge-small, fakes, retries,
timeouts, cost), settings in `src/docintel/config.py`, thin RAG slice:
Markdown loader + section-aware chunker (`ingestion/`), Qdrant store with
idempotent re-ingest (`retrieval/store.py`), RagService with score threshold
0.6 + NOT_FOUND marker and [n] citations (`rag/`), POST /ask. Sample data:
GDPR Art. 5, 6, 17 and AI Act Art. 5 in `data/samples/`.
Ruff TID251 bans provider SDK and qdrant_client imports outside their module.
Next (week 2): ingestion of PDF/DOCX, tables, OCR, metadata; 20 documents
load without errors. Consider a README quickstart.
