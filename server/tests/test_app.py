# Tests the FastAPI app (app.py): / and /ask routes.
# Covers: root health check, /ask happy path with mocked LLM,
# /ask returning a friendly 500 on unexpected errors, and 422 on missing "question".
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def _fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_root_endpoint_reports_running():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "RAGChef backend is running"}


def test_ask_endpoint_returns_mocked_answer(monkeypatch):
    monkeypatch.setattr(
        app_module.rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_response("Mocked answer."),
    )

    resp = client.post("/ask", json={"question": "How do I make dumplings?"})

    assert resp.status_code == 200
    assert resp.json() == {"answer": "Mocked answer."}


def test_ask_endpoint_returns_500_with_friendly_detail_on_unexpected_error(monkeypatch):
    def raise_error(question):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module.rag, "ask", raise_error)

    resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 500
    assert "went wrong" in resp.json()["detail"].lower()


def test_ask_endpoint_rejects_missing_question_field():
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_stream_endpoint_streams_list_answer_without_llm_call(monkeypatch):
    def fail(**kwargs):
        raise AssertionError("LLM should not be called for a rule-matched list route")

    monkeypatch.setattr(app_module.rag.client.chat.completions, "create", fail)

    resp = client.post("/ask/stream", json={"question": "Recommend a few dessert recipes"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.startswith("Here are some recipes you might like:")


def test_ask_stream_endpoint_returns_streamed_answer(monkeypatch):
    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            return _fake_response("How do I make Kung Pao Chicken?")
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Mocked "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="streamed answer."))]
            ),
        ]

    monkeypatch.setattr(app_module.rag.client.chat.completions, "create", fake_create)

    resp = client.post("/ask/stream", json={"question": "How do I make Kung Pao Chicken?"})

    assert resp.status_code == 200
    assert resp.text == "Mocked streamed answer."


def test_ask_stream_endpoint_rejects_missing_question_field():
    resp = client.post("/ask/stream", json={})
    assert resp.status_code == 422
