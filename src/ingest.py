"""
ingest.py — OFFLINE ingestion step.

Run this whenever documents in data/docs/ change:
    python -m src.ingest

Steps (kept explicit so every line is interview-explainable):
  1. Load each file from data/docs/ using the right loader for its extension.
  2. Normalize metadata["source"] to the basename (e.g. "customs-status-glossary.md").
  3. Split documents into overlapping chunks.
  4. Embed chunks with the local sentence-transformers model.
  5. Save the FAISS index to data/index/.

This script NEVER runs on the request path — it is an offline build step.
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCS_DIR, INDEX_DIR, get_embeddings

# ---------------------------------------------------------------------------
# Chunking settings
# chunk_size=1000 chars keeps each chunk within the context a short paragraph;
# overlap=150 prevents a sentence from being cut cleanly at the boundary.
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


def load_documents():
    """Load every supported file from DOCS_DIR.

    .md and .txt files use TextLoader with UTF-8 encoding (the corpus has
    em-dashes and other non-ASCII chars; Windows default cp1252 would crash).
    .pdf files use PyPDFLoader (one Document per page).
    """
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

        # Normalize source to just the filename so retrieval output is readable.
        for doc in loaded:
            doc.metadata["source"] = path.name

        docs.extend(loaded)
        print(f"  loaded {path.name}  ({len(loaded)} page/section(s))")

    return docs


def split_documents(docs):
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    """Embed chunks and build the FAISS index."""
    embeddings = get_embeddings()
    # FAISS.from_documents embeds every chunk and stores (vector, metadata) pairs.
    index = FAISS.from_documents(chunks, embeddings)
    return index


def main():
    print(f"Loading documents from {DOCS_DIR} ...")
    docs = load_documents()
    if not docs:
        print("No supported documents found. Add .md, .txt, or .pdf files to data/docs/.")
        sys.exit(1)

    print(f"\nSplitting {len(docs)} document(s) into chunks ...")
    chunks = split_documents(docs)
    print(f"  {len(chunks)} chunks total (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("\nEmbedding chunks and building FAISS index ...")
    index = build_index(chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(INDEX_DIR))

    print(f"\nDone. {len(docs)} file(s) -> {len(chunks)} chunks -> saved index to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
