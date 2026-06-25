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

### Step 3 — Run the API *(Milestone 2)*

```powershell
uvicorn src.main:app --reload
# POST /ask  {"question": "..."}
```

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
data/index/     prebuilt FAISS index (output of ingest; gitignored)
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
| `LLM_PROVIDER` | `anthropic` \| `openai_compatible` \| `gemini` \| `ollama` | `gemini` |
| `EMBEDDING_MODEL` | HuggingFace model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOP_K` | Chunks returned per query | `4` |

---

## Status

| Milestone | Status |
|---|---|
| M1: Ingestion + retrieval (no LLM) | ✅ Done |
| M2: LLM generation + FastAPI `/ask` | 🔲 Next |
| M3: Free deployment | 🔲 Planned |
