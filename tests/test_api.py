"""
test_api.py — test auth and rate limiting on POST /ask.

All tests are hermetic: the FAISS index, LLM client, and generate() are
monkeypatched so no real LLM call or disk access happens.

Three scenarios:
  1. Missing/invalid API key -> 401.
  2. Valid API key -> 200 with answer and sources.
  3. Valid API key, 11 requests from the same IP -> first 10 = 200, 11th = 429.

Each test uses a distinct X-Forwarded-For IP so the module-level in-memory
limiter counters don't bleed across tests.
"""

import pytest
from fastapi.testclient import TestClient

import src.config as config
from src.main import app

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CANNED_RESULT = {
    "answer": "CUS-HOLD means the shipment is under customs hold.",
    "sources": [
        {
            "source": "customs-status-glossary.md",
            "score": 0.42,
            "text": "CUS-HOLD: shipment is under customs hold.",
        }
    ],
}

VALID_KEY = "test-secret-key-for-unit-tests"


# ---------------------------------------------------------------------------
# Fixture — patch lifespan deps + generate(), set known RAG_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """Return a TestClient with the app's expensive deps patched out.

    - get_index() and get_chat_model() return None so the lifespan doesn't
      try to load the FAISS index or connect to a provider.
    - generate() returns a canned result so /ask never calls the real RAG flow.
    - RAG_API_KEY is set to VALID_KEY so auth tests have a known secret.
    """
    monkeypatch.setattr("src.main.get_index", lambda: None)
    monkeypatch.setattr("src.main.get_chat_model", lambda: None)
    monkeypatch.setattr("src.main.generate", lambda question, k=None: CANNED_RESULT)
    monkeypatch.setattr(config, "RAG_API_KEY", VALID_KEY)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

def test_ask_without_key_returns_401(client):
    """A request with no Authorization header must be rejected with 401."""
    resp = client.post(
        "/ask",
        json={"question": "what does CUS-HOLD mean?"},
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    assert resp.status_code == 401


def test_ask_with_wrong_key_returns_401(client):
    """A request with an incorrect API key must be rejected with 401."""
    resp = client.post(
        "/ask",
        json={"question": "what does CUS-HOLD mean?"},
        headers={
            "Authorization": "Bearer wrong-key",
            "X-Forwarded-For": "10.0.0.2",
        },
    )
    assert resp.status_code == 401


def test_ask_with_valid_key_returns_200(client):
    """A request with the correct API key must return 200 with answer and sources."""
    resp = client.post(
        "/ask",
        json={"question": "what does CUS-HOLD mean?"},
        headers={
            "Authorization": f"Bearer {VALID_KEY}",
            "X-Forwarded-For": "10.0.0.3",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert len(body["sources"]) > 0


# ---------------------------------------------------------------------------
# Rate limit test
# ---------------------------------------------------------------------------

def test_ask_rate_limit_returns_429_on_11th_request(client):
    """After 10 successful requests from the same IP, the 11th must return 429.

    All 11 requests carry the valid API key so they pass auth and reach the
    limiter. Uses a distinct XFF IP (10.0.0.99) not used by any other test.
    """
    headers = {
        "Authorization": f"Bearer {VALID_KEY}",
        "X-Forwarded-For": "10.0.0.99",
    }
    payload = {"question": "what does CUS-HOLD mean?"}

    for i in range(10):
        resp = client.post("/ask", json=payload, headers=headers)
        assert resp.status_code == 200, f"Request {i+1} should be 200, got {resp.status_code}"

    # 11th request must be rate-limited
    resp = client.post("/ask", json=payload, headers=headers)
    assert resp.status_code == 429
