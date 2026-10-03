# Tests the FastAPI app (app.py): / and /ask routes.
# Covers: root health check, /ask happy path with mocked LLM,
# /ask returning a friendly 500 on unexpected errors, and 422 on missing "question".
#
# 中文: LLM 调用通过 rag.llm(openai 官方 SDK 的 OpenAI 客户端)进行,所以这里
# 跟 test_ask_mocked.py 一样,直接在 app_module.rag.llm.chat.completions 这个
# 实例上 monkeypatch .create,而不是 patch 某个类。
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _fake_create(text):
    def create(*args, **kwargs):
        return _FakeResponse(text)

    return create


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChunkChoice(content)]


def _fake_stream_create(*texts):
    def create(*args, **kwargs):
        return (_FakeChunk(t) for t in texts)

    return create


def test_root_endpoint_reports_running():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "RAGChef backend is running"}


def test_ask_endpoint_returns_mocked_answer(monkeypatch):
    monkeypatch.setattr(
        app_module.rag.llm.chat.completions, "create", _fake_create("Mocked answer.")
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
    def fail(*args, **kwargs):
        raise AssertionError("LLM should not be called for a rule-matched list route")

    monkeypatch.setattr(app_module.rag.llm.chat.completions, "create", fail)

    resp = client.post("/ask/stream", json={"question": "Recommend a few dessert recipes"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.startswith("Here are some recipes you might like:")


def test_ask_stream_endpoint_returns_streamed_answer(monkeypatch):
    # query_rewrite() and the streamed generation both go through the same
    # chat.completions.create(), distinguished only by stream=True.
    def fake_create(*args, stream=False, **kwargs):
        if stream:
            return (_FakeChunk(t) for t in ["Mocked ", "streamed answer."])
        return _FakeResponse("How do I make Kung Pao Chicken?")

    monkeypatch.setattr(app_module.rag.llm.chat.completions, "create", fake_create)

    resp = client.post("/ask/stream", json={"question": "How do I make Kung Pao Chicken?"})

    assert resp.status_code == 200
    assert resp.text == "Mocked streamed answer."


def test_ask_stream_endpoint_rejects_missing_question_field():
    resp = client.post("/ask/stream", json={})
    assert resp.status_code == 422
