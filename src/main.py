"""
main.py — FastAPI application.

Endpoints:
    POST /ask   {"question": "...", "k": 4}
                Authorization: Bearer <RAG_API_KEY>
                -> {"answer": "...", "sources": [...]}

    GET  /health -> {"status": "ok", "provider": "...", "model": "..."}

The index and LLM are loaded once at startup via the FastAPI lifespan hook,
not on every request — the FAISS index can be large and the embedding model
takes a few seconds to warm up.

Security:
    - POST /ask requires a shared-secret API key (Authorization: Bearer header).
      Fail-closed: if RAG_API_KEY is unset, every /ask request returns 401.
    - /ask is rate-limited to 10 requests/minute per real client IP.
      X-Forwarded-For is trusted (the service runs behind HF's proxy).
      The API key, not the rate limit, is the primary security boundary.
    - GET /health is open — it's a cheap liveness probe (no LLM call).

Error handling:
    - Unauthenticated or rate-limited requests -> 401 / 429.
    - LLM unavailable (bad key, wrong model) -> 502 with JSON error.
    - Missing FAISS index -> 503 with JSON error (run `python -m src.ingest` first).
    - All other unexpected errors -> 500.
"""

import secrets
from contextlib import asynccontextmanager

import src.config as config
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.llm import get_chat_model
from src.rag import generate, get_index


# ---------------------------------------------------------------------------
# Rate limiter — in-memory, per real client IP
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Extract the real client IP, honoring X-Forwarded-For from HF's proxy.

    Takes the first (leftmost) address in XFF — that's the original client.
    Falls back to the direct remote address when the header is absent.
    """
    xff = request.headers.get("X-Forwarded-For")
    return xff.split(",")[0].strip() if xff else get_remote_address(request)


limiter = Limiter(key_func=_client_ip)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Try again in a moment."},
    )


# ---------------------------------------------------------------------------
# Auth dependency — shared-secret API key (Bearer token)
# ---------------------------------------------------------------------------

def require_api_key(authorization: str | None = Header(default=None)):
    """Check the Authorization: Bearer <key> header against RAG_API_KEY.

    Raises 401 if the header is missing, malformed, or the key doesn't match.
    Uses secrets.compare_digest to avoid timing-based key inference.
    Fail-closed: if RAG_API_KEY is unset, every request is rejected.

    Note: auth resolves before the rate limiter runs, so unauthenticated
    requests don't consume the client's rate-limit budget.
    """
    expected = config.RAG_API_KEY
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    if not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
def ask(request: Request, req: AskRequest):
    """Answer a question using retrieved chunks as grounding context.

    Requires a valid API key (Authorization: Bearer header).
    Rate-limited to 10 requests/minute per client IP.
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
        # LLM errors: bad/missing API key, wrong model name, provider down, etc.
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
    """Cheap liveness check — returns 200 immediately without calling the LLM.

    Safe for frequent keep-warm pings and deploy probes (no tokens billed,
    no provider rate-limit consumption). Reports the configured provider and
    model so operators can verify the right settings are in place.
    """
    provider = config.LLM_PROVIDER
    model = {
        "ollama":            config.OLLAMA_MODEL,
        "anthropic":         config.ANTHROPIC_MODEL,
        "openai_compatible": config.OPENAI_COMPAT_MODEL,
        "gemini":            config.GEMINI_MODEL,
    }.get(provider, "unknown")

    return {"status": "ok", "provider": provider, "model": model}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=config.PORT)
