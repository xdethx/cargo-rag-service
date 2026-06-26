"""
rag.py — retrieval and generation.

Milestone 1 public API (unchanged):
    index = load_index()
    results = retrieve(index, question, k=4)

Milestone 2 additions:
    result = generate(question, k=4)
    # -> {"answer": "...", "sources": [{"source": "...", "score": 0.42, "text": "..."}]}

The index and LLM are cached as module-level singletons (get_index / get_chat_model) so
they load once at startup — never per request. generate() accepts optional index/llm
arguments so tests can inject a fake LLM without touching the module state.
"""

import sys

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import INDEX_DIR, TOP_K, get_embeddings

# Module-level cache — populated on first call to get_index() / get_llm().
_index: FAISS | None = None
_llm = None


def get_index() -> FAISS:
    """Return the cached FAISS index, loading it on the first call.

    Call this at API startup to warm the cache; subsequent calls return instantly.
    """
    global _index
    if _index is None:
        _index = load_index()
    return _index


def get_llm():
    """Return the cached LLM client, building it on the first call."""
    global _llm
    if _llm is None:
        from src.llm import get_chat_model  # imported here to avoid circular imports
        _llm = get_chat_model()
    return _llm


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


# ---------------------------------------------------------------------------
# Generation (Milestone 2)
# ---------------------------------------------------------------------------

def build_prompt(question: str, chunks: list[dict]) -> list:
    """Build the messages list for the LLM call.

    Returns [SystemMessage, HumanMessage]. Built by hand so every field is
    visible — no opaque chain or prompt template magic.

    The system message constrains the model to the provided context only.
    The human message lists each chunk labelled with its source file, then
    asks the question.
    """
    system = SystemMessage(content=(
        "You are a cargo and customs assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not found in the context, say clearly: "
        "'This information is not in the provided documents.' "
        "Do not guess or add information from outside the context."
    ))

    # Format each chunk as [source: filename]\n<text>
    context_lines = []
    for chunk in chunks:
        context_lines.append(f"[source: {chunk['source']}]\n{chunk['text']}")
    context_block = "\n\n".join(context_lines)

    human = HumanMessage(content=(
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    ))

    return [system, human]


def generate(
    question: str,
    k: int | None = None,
    index: FAISS | None = None,
    llm=None,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    Steps (explicit, interview-explainable):
      1. Retrieve top-k chunks from the FAISS index.
      2. Build a prompt: system instruction + context block + question.
      3. Call the LLM.
      4. Return the answer text AND the source chunks it was grounded on.

    Args:
        question: the user's question.
        k:        number of chunks to retrieve (defaults to TOP_K from config).
        index:    optional FAISS index for testing (defaults to module cache).
        llm:      optional chat model for testing (defaults to module cache).

    Returns:
        {"answer": str, "sources": list[dict]}
        Each source dict: {"source": filename, "score": float, "text": str}
    """
    # Step 1 — retrieve
    idx = index if index is not None else get_index()
    chunks = retrieve(idx, question, k)

    # Step 2 — build grounded prompt
    messages = build_prompt(question, chunks)

    # Step 3 — call the LLM
    model = llm if llm is not None else get_llm()
    response = model.invoke(messages)

    # Step 4 — return answer + sources
    return {
        "answer":  response.content,
        "sources": chunks,
    }
