# Tests the FastAPI app (app.py): / and /ask routes.
# Covers: root health check, /ask happy path with mocked LLM,
# /ask returning a friendly 500 on unexpected errors, and 422 on missing "question".
#
# 中文: LLM 调用通过 rag.llm(LangChain 的 ChatOpenAI)进行,所以这里跟
# test_ask_mocked.py 一样,在 ChatOpenAI 类上 monkeypatch invoke/stream,
# 而不是旧版直接 mock rag.client.chat.completions.create。
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI

import app as app_module

client = TestClient(app_module.app)


def _fake_invoke(text):
    def invoke(self, *args, **kwargs):
        return AIMessage(content=text)

    return invoke


def _fake_stream(*texts):
    def stream(self, *args, **kwargs):
        for t in texts:
            yield AIMessageChunk(content=t)

    return stream


def test_root_endpoint_reports_running():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "RAGChef backend is running"}


def test_ask_endpoint_returns_mocked_answer(monkeypatch):
    monkeypatch.setattr(ChatOpenAI, "invoke", _fake_invoke("Mocked answer."))

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
    def fail(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a rule-matched list route")
        yield  # pragma: no cover - makes this a generator function, never reached

    monkeypatch.setattr(ChatOpenAI, "stream", fail)

    resp = client.post("/ask/stream", json={"question": "Recommend a few dessert recipes"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.startswith("Here are some recipes you might like:")


def test_ask_stream_endpoint_returns_streamed_answer(monkeypatch):
    # query_rewrite() makes one non-streaming call (ChatOpenAI.invoke), then
    # generation streams via ChatOpenAI.stream -- mirror both.
    monkeypatch.setattr(
        ChatOpenAI, "invoke", _fake_invoke("How do I make Kung Pao Chicken?")
    )
    monkeypatch.setattr(ChatOpenAI, "stream", _fake_stream("Mocked ", "streamed answer."))

    resp = client.post("/ask/stream", json={"question": "How do I make Kung Pao Chicken?"})

    assert resp.status_code == 200
    assert resp.text == "Mocked streamed answer."


def test_ask_stream_endpoint_rejects_missing_question_field():
    resp = client.post("/ask/stream", json={})
    assert resp.status_code == 422
