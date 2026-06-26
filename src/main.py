"""
main.py — FastAPI application.

Endpoints:
    POST /ask   {"question": "...", "k": 4}
                -> {"answer": "...", "sources": [...]}

    GET  /health -> {"status": "ok"|"degraded", "provider": "...",
                     "model": "...", "llm": "reachable"|"<error reason>"}

The index and LLM are loaded once at startup via the FastAPI lifespan hook,
not on every request — the FAISS index can be large and the embedding model
takes a few seconds to warm up.

Error handling:
    - LLM unavailable (Ollama down, bad key, wrong model) -> 502 with JSON error.
    - Missing FAISS index -> 503 with JSON error (run `python -m src.ingest` first).
    - All other unexpected errors -> 500.
"""

from contextlib import asynccontextmanager

import src.config as config
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.llm import check_llm, get_chat_model
from src.rag import generate, get_index


# ---------------------------------------------------------------------------
# Lifespan — warm index + model caches at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and LLM client once before serving requests."""
    get_index()        # fails fast with a clear message if the index is missing
    get_chat_model()   # validates provider config at startup
    yield
    # (nothing to clean up on shutdown for this simple service)


app = FastAPI(
    title="Cargo RAG Service",
    description="Answers cargo/customs questions grounded in retrieved documents.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    k: int | None = None   # override TOP_K for this request; None -> use config


class Source(BaseModel):
    source: str    # filename (e.g. "customs-status-glossary.md")
    score: float   # L2 distance; lower = more similar
    text: str      # the chunk text


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question using retrieved chunks as grounding context.

    Always returns the answer alongside the source chunks it was based on.
    An answer with no sources is a bug — grounding is the whole point of RAG.
    """
    try:
        result = generate(req.question, k=req.k)
    except SystemExit:
        # load_index() calls sys.exit(1) when the index is missing.
        return JSONResponse(
            status_code=503,
            content={"error": "FAISS index not found. Run `python -m src.ingest` first."},
        )
    except Exception as exc:
        # LLM errors: Ollama not running, bad/missing API key, wrong model name, etc.
        provider = config.LLM_PROVIDER
        return JSONResponse(
            status_code=502,
            content={"error": f"LLM provider '{provider}' unavailable: {exc}"},
        )

    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )


@app.get("/health")
def health():
    """Real liveness check — probes the LLM provider, not just config.

    Returns status "ok" only when the configured provider actually responds.
    Status "degraded" means the service is running but the LLM is unreachable.
    """
    provider = config.LLM_PROVIDER
    model = {
        "ollama":            config.OLLAMA_MODEL,
        "anthropic":         config.ANTHROPIC_MODEL,
        "openai_compatible": config.OPENAI_COMPAT_MODEL,
        "gemini":            config.GEMINI_MODEL,
    }.get(provider, "unknown")

    ok, detail = check_llm()

    return {
        "status":   "ok" if ok else "degraded",
        "provider": provider,
        "model":    model,
        "llm":      detail,
    }
