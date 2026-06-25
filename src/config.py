"""
config.py — load every setting from environment variables (via .env).

All other modules import from here; nothing else reads os.environ directly.
The embedding model is exposed through get_embeddings() so ingest.py and rag.py
are guaranteed to use the exact same model instance configuration.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Load .env (silently ignored if the file doesn't exist)
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent          # repo root
DOCS_DIR = ROOT_DIR / "data" / "docs"
INDEX_DIR = ROOT_DIR / "data" / "index"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# ---------------------------------------------------------------------------
# LLM provider (read now; not used until Milestone 2)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- OpenAI-compatible (Grok / Groq / OpenRouter) ---
OPENAI_COMPAT_API_KEY  = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "")
OPENAI_COMPAT_MODEL    = os.getenv("OPENAI_COMPAT_MODEL", "")

# --- Google Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1")


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model.

    Called by both ingest.py (offline) and rag.py (online). Using the same
    function guarantees the same model is used on both sides — a hard
    requirement: if they differ, retrieved chunks will be garbage.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
