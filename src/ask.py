"""
ask.py — CLI for full RAG: retrieve + generate + print answer and sources.

  One-shot:  python -m src.ask "who pays the customs duty?"
  REPL:      python -m src.ask   (then type questions; 'exit' to quit)

k=N prefix overrides top-k for one question (e.g. "k=6 which documents?").
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag import generate, get_index, get_llm


def _parse_question(raw: str) -> tuple[str, int | None]:
    raw = raw.strip()
    if raw.lower().startswith("k="):
        parts = raw.split(None, 1)
        try:
            return (parts[1] if len(parts) > 1 else ""), int(parts[0][2:])
        except (ValueError, IndexError):
            pass
    return raw, None


def _show(result: dict) -> None:
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['source']}  (L2 dist {s['score']:.4f})")


def run_once(question_raw: str) -> None:
    question, k = _parse_question(question_raw)
    if not question:
        print("Empty question.")
        return
    _show(generate(question, k=k))


def run_repl() -> None:
    print("Type a question to get an answer + sources. 'exit' to quit.\n")
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
        run_once(raw)
        print()


def main() -> None:
    print("Loading embedding model, index, and LLM ...")
    get_index()
    get_llm()
    print("Ready.\n")

    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        run_repl()


if __name__ == "__main__":
    main()
