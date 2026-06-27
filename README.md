---
title: Cargo RAG Service
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Cargo RAG Service

A retrieval-augmented generation (RAG) microservice that answers questions about cargo,
shipping, and customs — for example:

- *"What does CUS-HOLD mean?"*
- *"Which documents are required for a dangerous goods shipment?"*
- *"Who pays the customs duty?"*

Built as a **learning + portfolio project** to understand every layer of a production RAG
system: embeddings, vector search, grounded prompting, and provider-agnostic LLM calls.
A separate .NET shipment-tracking app calls this service over HTTP.

---

## Live demo

Deployed on a Hugging Face Docker Space:

**https://dethx-ragtest.hf.space**

- `GET /health` is **public** — a cheap liveness check (no LLM call). Try
  [`/health`](https://dethx-ragtest.hf.space/health).
- `POST /ask` requires an `Authorization: Bearer <RAG_API_KEY>` header.

## Calling the API

```bash
curl -X POST https://dethx-ragtest.hf.space/ask \
     -H "Authorization: Bearer $RAG_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"question": "who pays the customs duty?"}'
```

Every answer comes back **with the source chunks it was grounded on** — an
answer without sources is a bug, since grounding is the whole point of RAG:

```json
{
  "answer": "By default, the importer (the receiver) pays import duty and VAT...",
  "sources": [
    { "source": "shipping-faq.txt", "score": 0.6626, "text": "Q: Who pays the customs duty and VAT? A: ..." }
  ]
}
```

---

## Architecture

The key design decision is the **offline / online split**:

```
OFFLINE (run once, or when docs change)
  data/docs/  →  src/ingest.py  →  data/index/  (FAISS vector index)

ONLINE (per request)
  query  →  embed  →  retrieve top-k chunks  →  grounded prompt  →  LLM  →  answer + sources
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — local, free, no API key.
  The *same* model embeds documents at ingest time and queries at runtime.
- **Vector store**: FAISS (in-memory, saved to disk). Built offline, loaded read-only at startup.
- **Generation LLM**: provider-agnostic — Anthropic Claude, Grok/Groq/OpenRouter
  (OpenAI-compatible), Google Gemini, or local Ollama, selected by the `LLM_PROVIDER`
  environment variable.

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and fill in your keys
copy env.example .env        # Windows
# cp env.example .env        # macOS / Linux
```

---

## Running

### Step 1 — Ingest (offline, run once or after doc changes)

```powershell
python -m src.ingest
```

Reads every `.md`, `.txt`, and `.pdf` from `data/docs/`, splits into chunks, embeds them
with the local model, and saves the FAISS index to `data/index/`.

### Step 2 — Test retrieval (Milestone 1 checkpoint)

Interactive REPL (loads model + index once; query many times):

```powershell
python -m src.search
ask> what does CUS-HOLD mean?
ask> k=6 which documents are required?
ask> exit
```

One-shot mode:

```powershell
python -m src.search "who pays the customs duty?"
```

### Step 3 — Run the API

```powershell
uvicorn src.main:app --reload
```

Then in another terminal:

`/ask` is protected (fail-closed): set `RAG_API_KEY` in `.env` and send it as a
Bearer token, or every request returns `401`.

```powershell
# Ask a question
curl -X POST http://127.0.0.1:8000/ask `
     -H "Authorization: Bearer $env:RAG_API_KEY" `
     -H "Content-Type: application/json" `
     -d '{"question": "who pays the customs duty?"}'
```

Sample response:

```json
{
  "answer": "By default, the importer (the receiver) pays import duty and VAT. The exception is when the shipment is sent under the Incoterm DDP (Delivered Duty Paid), in which case the seller pays everything including import duty.",
  "sources": [
    {
      "source": "shipping-faq.txt",
      "score": 0.6626,
      "text": "Q: Who pays the customs duty and VAT? A: By default the importer..."
    },
    {
      "source": "incoterms-2020-summary.md",
      "score": 0.7493,
      "text": "DDP — Delivered Duty Paid. The seller delivers to the destination and pays everything..."
    }
  ]
}
```

**Switching providers is one env change** — set `LLM_PROVIDER` in `.env` to
`anthropic`, `openai_compatible`, `gemini`, or `ollama`. No code change required.

```powershell
# Health check — cheap liveness probe (no LLM call); reports the effective provider/model
curl http://127.0.0.1:8000/health
```

---

## Deployment

Deployed as a **Hugging Face Docker Space** (the container listens on port `7860`).

- The prebuilt **FAISS index** and the **embedding model** are baked into the Docker image
  at build time. Cold starts are fast and the container is self-contained — no model
  download on a non-root, ephemeral filesystem.
- The cloud LLM runs via `LLM_PROVIDER=openai_compatible` pointed at **Groq**; embeddings
  stay **local** (the same sentence-transformers model used at ingest).
- Runtime configuration — LLM provider, Groq base URL and model, and the API key — is set
  through the Space's **Variables & Secrets**, never committed to the repo.

---

## Testing

```powershell
pytest
```

The test suite builds a fresh FAISS index from the real corpus and asserts that known
questions retrieve the expected source files — proving retrieval quality before any LLM
is connected.

---

## Project layout

```
data/docs/      source cargo/customs documents (.md, .txt, .pdf)
data/index/     prebuilt FAISS index (output of ingest; committed so it ships in the image)
src/config.py   env settings + shared get_embeddings()
src/ingest.py   offline: load -> chunk -> embed -> save index
src/rag.py      retrieve -> (M2) grounded prompt -> generate
src/search.py   CLI test harness: REPL or one-shot retrieval
src/main.py     (M2) FastAPI app, POST /ask
tests/          pytest — retrieval correctness assertions
journal/        dated notes after each milestone
```

---

## Environment variables

See `env.example` for the full list. Key ones:

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` \| `openai_compatible` \| `gemini` \| `ollama` | `anthropic` |
| `RAG_API_KEY` | Shared secret for `POST /ask` (Bearer token); unset → every request `401` | _(empty)_ |
| `EMBEDDING_MODEL` | HuggingFace model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOP_K` | Chunks returned per query | `4` |
| `PORT` | Port the API listens on (via `python -m src.main`) | `7860` |

For Groq (the deployed setup), set `LLM_PROVIDER=openai_compatible` plus
`OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_MODEL`, and `OPENAI_COMPAT_API_KEY`.

---

## Status

| Milestone | Status |
|---|---|
| M1: Ingestion + retrieval (no LLM) | ✅ Done |
| M2: LLM generation + FastAPI `/ask` | ✅ Done |
| M3: Deployment (Hugging Face Docker Space + auth + rate limit) | ✅ [Live](https://dethx-ragtest.hf.space) |
