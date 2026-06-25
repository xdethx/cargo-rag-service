"""
rag.py — retrieval (and, in Milestone 2, generation).

Public API for Milestone 1:
    index = load_index()
    results = retrieve(index, question, k=4)

Each result is a plain dict:
    {"source": "customs-status-glossary.md", "score": 0.42, "text": "..."}

Score is the FAISS L2 distance (lower = more similar). It is returned as-is
so callers can decide how to display or threshold it.
"""

import sys

from langchain_community.vectorstores import FAISS

from src.config import INDEX_DIR, TOP_K, get_embeddings


def load_index() -> FAISS:
    """Load the prebuilt FAISS index from disk.

    Exits with a helpful message if the index hasn't been built yet, rather
    than showing a raw traceback.
    """
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        print(
            f"Index not found at {INDEX_DIR}.\n"
            "Run `python -m src.ingest` first to build it."
        )
        sys.exit(1)

    embeddings = get_embeddings()
    # allow_dangerous_deserialization is required by recent FAISS versions when
    # loading from pickle; safe here because we wrote the file ourselves.
    index = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return index


def retrieve(index: FAISS, question: str, k: int | None = None) -> list[dict]:
    """Embed `question` and return the top-k most similar chunks.

    Returns a list of dicts, each with:
        source  — filename the chunk came from (e.g. "customs-status-glossary.md")
        score   — L2 distance (float); lower means more similar
        text    — the chunk text
    """
    k = k or TOP_K
    # similarity_search_with_score returns [(Document, float), ...] sorted by score.
    raw_results = index.similarity_search_with_score(question, k=k)

    results = []
    for doc, score in raw_results:
        results.append({
            "source": doc.metadata.get("source", "unknown"),
            "score":  round(float(score), 4),
            "text":   doc.page_content,
        })
    return results
