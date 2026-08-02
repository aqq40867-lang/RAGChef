import os
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from rag import SimpleRAG, LLM_UNAVAILABLE_MESSAGE

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.md")


def _make_rag():
    return SimpleRAG(RECIPES_PATH)


def _fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_ask_returns_llm_answer_without_hitting_network(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_response("Mocked answer about Kung Pao Chicken."),
    )

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert answer == "Mocked answer about Kung Pao Chicken."


def test_ask_passes_configured_model(monkeypatch):
    rag = _make_rag()
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return _fake_response("ok")

    monkeypatch.setattr(rag.client.chat.completions, "create", fake_create)
    rag.ask("How do I make dumplings?")

    assert seen["model"] == rag.model


def test_ask_empty_question_never_calls_llm(monkeypatch):
    rag = _make_rag()

    def fail(**kwargs):
        raise AssertionError("LLM should not be called for an empty question")

    monkeypatch.setattr(rag.client.chat.completions, "create", fail)

    answer = rag.ask("   ")

    assert "enter a question" in answer.lower()


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        ),
        APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com")),
    ],
)
def test_ask_returns_friendly_message_on_llm_failure(monkeypatch, error):
    rag = _make_rag()

    def raise_error(**kwargs):
        raise error

    monkeypatch.setattr(rag.client.chat.completions, "create", raise_error)

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert answer == LLM_UNAVAILABLE_MESSAGE
