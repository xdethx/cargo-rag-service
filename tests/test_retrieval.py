"""
test_retrieval.py — verify that ingestion + retrieval work end-to-end.

The session-scoped fixture builds a real FAISS index from data/docs/ into a
temporary directory so tests are self-contained (no dependency on a pre-built
data/index/). The embedding model is downloaded once per pytest session.

These tests prove the Milestone-1 checkpoint: known questions must surface
the correct source file in the top-k results before any LLM is connected.
"""

import pytest

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCS_DIR, get_embeddings
from src.rag import retrieve


# ---------------------------------------------------------------------------
# Session fixture — build a fresh index from the real corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def index(tmp_path_factory):
    """Build and return a FAISS index from data/docs/ (runs once per session)."""
    tmp_dir = tmp_path_factory.mktemp("index")

    # --- Load ---
    docs = []
    supported = {".md", ".txt", ".pdf"}
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() not in supported:
            continue
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)

    # --- Split ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # --- Embed + index ---
    embeddings = get_embeddings()
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(str(tmp_dir))

    # Reload from disk (exercises the full round-trip)
    return FAISS.load_local(
        str(tmp_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def top_sources(index, question: str, k: int = 4) -> list[str]:
    """Return the source filenames from the top-k results."""
    results = retrieve(index, question, k=k)
    return [r["source"] for r in results]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cus_hold_retrieves_glossary(index):
    """'CUS-HOLD' is defined in the status glossary — it should be the top hit."""
    sources = top_sources(index, "what does CUS-HOLD mean?")
    assert "customs-status-glossary.md" in sources, (
        f"Expected customs-status-glossary.md in results, got: {sources}"
    )


def test_customs_duty_payment_retrieves_relevant_docs(index):
    """Duty payment is covered by the glossary (DUTY-DUE) and/or document requirements."""
    sources = top_sources(index, "who pays the customs duty?", k=4)
    relevant = {"customs-status-glossary.md", "document-requirements.md"}
    assert relevant & set(sources), (
        f"Expected at least one of {relevant} in results, got: {sources}"
    )


def test_document_requirements_retrieves_correct_file(index):
    """Questions about required shipping documents should surface document-requirements.md."""
    sources = top_sources(index, "which documents are required for a shipment?", k=4)
    assert "document-requirements.md" in sources, (
        f"Expected document-requirements.md in results, got: {sources}"
    )


def test_retrieve_returns_score_and_text(index):
    """Each result must carry source, score, and text fields."""
    results = retrieve(index, "customs clearance", k=2)
    assert len(results) == 2
    for r in results:
        assert "source" in r
        assert "score" in r
        assert "text" in r
        assert isinstance(r["score"], float)
        assert r["text"]  # non-empty


def test_k_override(index):
    """Passing k=2 must return exactly 2 results."""
    results = retrieve(index, "incoterms", k=2)
    assert len(results) == 2


def test_k_override_larger(index):
    """Passing k=6 must return up to 6 results (corpus is large enough)."""
    results = retrieve(index, "shipment", k=6)
    assert len(results) == 6
