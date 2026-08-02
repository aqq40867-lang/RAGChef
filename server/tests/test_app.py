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
