# DocIntel: Project Plan

## Goal
Build a cloud-agnostic GenAI platform that demonstrates the skills most
junior GenAI / AI Engineer vacancies ask for: RAG, structured outputs,
agents, evaluation, tracing, guardrails, backend, CI/CD and cloud basics.

Target direction: applied LLM engineering (backend + LLMOps), not model
training or research.

## Capabilities
1. **Ask**: answers over uploaded documents with citations; refuses when
   the answer is not in the documents.
2. **Extract**: documents to validated JSON (invoice, contract, resume).
3. **Agent**: multi-step tasks with tools and human confirmation.
4. **Observe**: evaluation, tracing, cost and latency tracking.

## Architecture
```
Frontend (Streamlit)
      |
FastAPI -- auth (JWT, roles) -- rate limit
  |-- /ingest   parse (PDF, DOCX, OCR) -> chunk -> embed -> vector store
  |-- /ask      hybrid search -> rerank -> answer with citations
  |-- /extract  Pydantic schemas -> structured output -> validate + retry
  |-- /agent    LangGraph: search, read-only SQL, calculator, human-in-the-loop

Cross-cutting: Langfuse tracing, eval pipeline, guardrails, logging, CI
```

## Repository layout
```
src/docintel/
  api/  llm/  ingestion/  retrieval/  rag/
  extraction/  agent/  safety/  observability/
eval/  tests/  infra/  docs/  .github/workflows/
```

## Modules

### LLM layer
One interface (`generate`, `embed`, `stream`). Adapters: OpenAI, Anthropic,
Azure OpenAI, Ollama. Retries, timeouts, token and cost accounting.

### Ingestion
PDF/DOCX parsing, tables, OCR for scans, structure-aware chunking, metadata
(document, page, section). Idempotent uploads via content hash.

### Retrieval and RAG
Vector + BM25 hybrid search, reranking, metadata and permission filters.
Answers always cite sources. Low confidence leads to an explicit
"not found in documents". Chunk size, top-k and embedding model are configurable.

### Extraction
Pydantic schemas for 3 document types. Loop: extract, validate, on error
re-ask the model with the validation error. Per-field accuracy metric.

### Agent
LangGraph graph with tools: document search, read-only text-to-SQL
(query validated before execution), calculator. Human approval before
side-effect actions. Step and token budget limits.

### Safety
PII masking in inputs and logs. Prompt injection defense (document text is
data, never instructions). Document-level permissions enforced in retrieval.
Rate limiting.

### Observability and evaluation
- Every request traced in Langfuse.
- Eval dataset of 50+ questions with reference answers and sources.
- Metrics: faithfulness, answer relevance, retrieval hit rate, correct
  refusal rate, extraction field accuracy.
- Experiment table in README (chunk size, hybrid vs vector, reranker on/off).
- Fast eval subset runs in CI and fails the build on a significant drop.

## Timeline (8 weeks, about 10-15 h/week)

| Week | Work | Done when |
|---|---|---|
| 1 | Repo, Docker Compose, CI, LLM abstraction, basic /ask | `docker compose up` gives a working Q&A |
| 2 | Ingestion: PDF, DOCX, tables, OCR, metadata | 20 documents load without errors |
| 3 | Hybrid search, reranking, citations, refusal | answers include sources |
| 4 | Eval dataset, metrics, experiments, Langfuse | results table in README |
| 5 | Extraction: schemas, retry, accuracy | accuracy report on 30 documents |
| 6 | LangGraph agent, text-to-SQL, human-in-the-loop | 3 demo scenarios work |
| 7 | Safety, auth, roles, cache, cost dashboard, CI regression | CI blocks quality drops |
| 8 | Cloud deploy, IaC, docs, demo video | public repo and live demo |

Cut order if time runs short: MCP, IaC, text-to-SQL, OCR.
Never cut: evaluation, tracing, tests, CI.

## Cloud strategy
Develop locally. Deploy in week 8 to the first available of: Azure
(free account or student program), Google Cloud Run, AWS, or a VPS with Docker.
Minimum: managed container, secrets in a secret store, service account with
least privilege, one app deployable from `infra/`.
README includes a "Cloud mapping" section (containers, secrets, IAM,
vector search, LLM service across Azure, AWS, GCP).

## Demo data
Open or synthetic only: EU AI Act, GDPR, public annual reports, generated
invoices and contracts (with noise and scans), a demo sales database.

## Packaging
- README: problem, architecture diagram, quickstart, metrics table,
  limitations and next steps.
- 3-5 short ADRs (why this vector store, why this chunking).
- 2-minute demo video, Langfuse trace screenshots.

## Risks
1. Scope creep: always finish a vertical slice first.
2. Evaluating by eye: keep the dataset and numbers.
3. API cost: cheap models for development, spending limit at the provider.
4. Leaked secrets: pre-commit secret scanning from day one.