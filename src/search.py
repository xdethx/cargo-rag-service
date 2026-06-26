"""
search.py — CLI test harness for Milestone 1 retrieval (no LLM).

Two modes, same retrieval logic:

  One-shot (pass question as argument):
      python -m src.search "who pays the customs duty?"

  REPL (no argument — keeps model+index in memory across questions):
      python -m src.search
      ask> who pays the customs duty?
      ask> k=6 which documents are required?
      ask> exit

Per-query k override: prefix the question with  k=<n>  (e.g. "k=6 my question").
Otherwise TOP_K from config is used.

Score shown is FAISS L2 distance — lower means more similar.
"""

import sys

# Force UTF-8 output so non-ASCII corpus text (em-dashes, arrows, etc.) doesn't
# raise UnicodeEncodeError on Windows consoles that default to cp1252/cp1254.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag import load_index, retrieve

SNIPPET_LENGTH = 200  # characters to show from each chunk


def _parse_question(raw: str) -> tuple[str, int | None]:
    """Split an optional 'k=N ' prefix from the question text.

    Returns (question, k_override).  k_override is None if not supplied.
    """
    raw = raw.strip()
    if raw.lower().startswith("k="):
        parts = raw.split(None, 1)       # ["k=6", "rest of question"]
        try:
            k_val = int(parts[0][2:])
            question = parts[1] if len(parts) > 1 else ""
            return question, k_val
        except (ValueError, IndexError):
            pass
    return raw, None


def _show_results(results: list[dict]) -> None:
    """Print retrieved chunks in a readable format."""
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        snippet = r["text"].replace("\n", " ")[:SNIPPET_LENGTH]
        if len(r["text"]) > SNIPPET_LENGTH:
            snippet += "..."
        print(f"\n  [{i}] {r['source']}  (L2 dist {r['score']:.4f}, lower=better)")
        print(f"      {snippet}")


def run_once(index, question_raw: str) -> None:
    """Run a single retrieval and print results."""
    question, k_override = _parse_question(question_raw)
    if not question:
        print("Empty question — nothing to retrieve.")
        return
    results = retrieve(index, question, k=k_override)
    _show_results(results)


def run_repl(index) -> None:
    """Interactive REPL — loads once, queries many times."""
    print("Loaded index. Type a question to retrieve chunks.")
    print("Prefix with 'k=N ' to override top-k (e.g. 'k=6 which documents?').")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            raw = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if raw.lower() in {"exit", "quit", ""}:
            if raw.lower() in {"exit", "quit"}:
                print("Bye.")
            break

        run_once(index, raw)
        print()


def main() -> None:
    # Load the model and index once — shared by both modes.
    print("Loading embedding model and index ...")
    index = load_index()
    print("Ready.\n")

    if len(sys.argv) > 1:
        # One-shot mode: question passed as CLI argument.
        question_raw = " ".join(sys.argv[1:])
        run_once(index, question_raw)
    else:
        # REPL mode: interactive loop.
        run_repl(index)


if __name__ == "__main__":
    main()
