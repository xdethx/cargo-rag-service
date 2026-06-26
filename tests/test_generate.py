"""
test_generate.py — test the generation step without a live LLM.

A capturing fake LLM is injected via generate(..., llm=fake) so these tests
run entirely offline. They verify:
  a) generate() returns the correct shape: {answer, sources}
  b) sources are non-empty and contain the expected filename for a known question
  c) the fake LLM received a prompt that contains the retrieved chunk text
     (proves the grounding wiring is correct — the context block reaches the model)

The session-scoped `index` fixture is intentionally duplicated from
test_retrieval.py rather than shared via conftest.py, to keep tests readable
and independent.
"""

import pytest
from langchain_core.messages import AIMessage

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCS_DIR, get_embeddings
from src.rag import build_prompt, generate


# ---------------------------------------------------------------------------
# Fake LLM — captures the messages it received; returns a fixed answer
# ---------------------------------------------------------------------------

class CapturingFakeLLM:
    """Stand-in for any LangChain chat model.

    .invoke() stores the messages list so tests can inspect the prompt,
    then returns a fixed AIMessage — no network call, no cost, always fast.
    """
    def __init__(self, reply: str = "Fake grounded answer."):
        self.reply = reply
        self.last_messages: list | None = None

    def invoke(self, messages: list):
        self.last_messages = messages
        return AIMessage(content=self.reply)


# ---------------------------------------------------------------------------
# Session fixture — same real corpus, built once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def index(tmp_path_factory):
    """Build a FAISS index from the real data/docs/ corpus (once per session)."""
    tmp_dir = tmp_path_factory.mktemp("gen_index")

    docs = []
    supported = {".md", ".txt", ".pdf"}
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() not in supported:
            continue
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    embeddings = get_embeddings()
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(str(tmp_dir))
    return FAISS.load_local(str(tmp_dir), embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_returns_answer_and_sources(index):
    """generate() must return a non-empty answer string and a non-empty sources list."""
    fake = CapturingFakeLLM()
    result = generate("what does CUS-HOLD mean?", index=index, llm=fake)

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)
    assert result["answer"]          # non-empty
    assert len(result["sources"]) > 0


def test_generate_sources_contain_expected_file(index):
    """Sources for a CUS-HOLD question must include the status glossary."""
    fake = CapturingFakeLLM()
    result = generate("what does CUS-HOLD mean?", index=index, llm=fake)

    source_files = [s["source"] for s in result["sources"]]
    assert "customs-status-glossary.md" in source_files, (
        f"Expected customs-status-glossary.md in sources, got: {source_files}"
    )


def test_generate_passes_context_to_llm(index):
    """The LLM must receive a prompt containing text from the retrieved chunks.

    This is the grounding wiring test: if context doesn't reach the model,
    the answer is hallucinated rather than grounded. We verify the human
    message contains the chunk text that was retrieved.
    """
    fake = CapturingFakeLLM()
    result = generate("what does CUS-HOLD mean?", index=index, llm=fake)

    assert fake.last_messages is not None, "LLM was never called"
    # The human message (index 1) should contain the retrieved chunk text.
    human_content = fake.last_messages[1].content
    assert "CUS-HOLD" in human_content, (
        "Retrieved context about CUS-HOLD was not passed to the LLM"
    )


def test_generate_answer_is_fake_llm_reply(index):
    """generate() must return exactly what the LLM replied (no post-processing)."""
    reply = "Test answer: the importer pays."
    fake = CapturingFakeLLM(reply=reply)
    result = generate("who pays the customs duty?", index=index, llm=fake)

    assert result["answer"] == reply


def test_generate_k_override_controls_source_count(index):
    """Passing k=2 must return exactly 2 sources."""
    fake = CapturingFakeLLM()
    result = generate("shipment documents", k=2, index=index, llm=fake)

    assert len(result["sources"]) == 2


def test_build_prompt_structure():
    """build_prompt() must return [SystemMessage, HumanMessage] with expected content."""
    from langchain_core.messages import HumanMessage, SystemMessage

    chunks = [
        {"source": "glossary.md", "score": 0.3, "text": "CUS-HOLD means customs hold."},
        {"source": "faq.txt",     "score": 0.5, "text": "Duty is paid by the importer."},
    ]
    messages = build_prompt("what does CUS-HOLD mean?", chunks)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    human_text = messages[1].content
    assert "glossary.md" in human_text
    assert "CUS-HOLD means customs hold." in human_text
    assert "what does CUS-HOLD mean?" in human_text
