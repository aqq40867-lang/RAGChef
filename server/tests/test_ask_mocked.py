import os
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from rag import (
    ROUTE_DETAIL,
    ROUTE_GENERAL,
    ROUTE_LIST,
    NO_RESULTS_MESSAGE,
    SimpleRAG,
    LLM_UNAVAILABLE_MESSAGE,
)

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes")


def _make_rag():
    return SimpleRAG(RECIPES_PATH)


def _fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _fake_stream_chunks(*texts):
    """Builds a fake streaming response: an iterable of chunk objects whose
    .choices[0].delta.content pieces are the given texts, in order --
    mirrors the shape the real OpenAI client yields for stream=True chat
    completions (as opposed to _fake_response()'s non-streaming .message
    shape).
    """
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
        for t in texts
    ]


def _make_streaming_aware_create(stream_texts, non_stream_text="detail"):
    """Builds a fake `create(**kwargs)` that returns a streaming or
    non-streaming fake response depending on the `stream` kwarg, so a single
    monkeypatch can support a full ask_stream() call (which may make both
    non-streaming calls, e.g. query_rewrite, and one streaming call for
    generation).
    """

    def fake_create(**kwargs):
        if kwargs.get("stream"):
            return iter(_fake_stream_chunks(*stream_texts))
        return _fake_response(non_stream_text)

    return fake_create


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


def test_query_router_returns_general_on_unparseable_response(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: "not a real route")

    assert rag.query_router("anything") == ROUTE_GENERAL


def test_query_router_returns_general_when_llm_call_fails(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: None)

    assert rag.query_router("anything") == ROUTE_GENERAL


def test_query_router_returns_classified_route(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: " List ")

    # Case/whitespace in the raw LLM reply shouldn't matter.
    assert rag.query_router("recommend a few dishes") == ROUTE_LIST


def test_query_rewrite_falls_back_to_original_question_on_failure(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0.3: None)

    assert rag.query_rewrite("give me something to cook") == "give me something to cook"


def test_generate_list_answer_lists_dish_names_without_duplicates():
    rag = _make_rag()
    kung_pao = next(d for d in rag.documents if d.dish_name == "kung pao chicken")
    dumplings = next(d for d in rag.documents if d.dish_name == "dumplings jiaozi")

    answer = rag._generate_list_answer([kung_pao, dumplings, kung_pao])

    assert answer.count("Kung Pao Chicken") == 1
    assert "Dumplings Jiaozi" in answer


def test_generate_list_answer_empty_docs_returns_no_results_message():
    rag = _make_rag()
    assert rag._generate_list_answer([]) == NO_RESULTS_MESSAGE


def test_ask_rule_matched_list_route_never_calls_llm_at_all(monkeypatch):
    rag = _make_rag()

    def fail(**kwargs):
        raise AssertionError(
            "LLM should not be called: 'recommend a few' is caught by "
            "_infer_route(), and list generation is pure Python."
        )

    monkeypatch.setattr(rag.client.chat.completions, "create", fail)

    answer = rag.ask("Recommend a few dessert recipes")

    assert answer.startswith("Here are some recipes you might like:")


def test_ask_rule_matched_detail_route_still_calls_rewrite_and_generate(monkeypatch):
    # "How do I make ..." is caught by _infer_route() as ROUTE_DETAIL, so the
    # router call is skipped, but query_rewrite and generation still each
    # need their own LLM call -- 2 calls total, not 0 and not the old 3.
    rag = _make_rag()
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        return _fake_response("How do I make Kung Pao Chicken?")

    monkeypatch.setattr(rag.client.chat.completions, "create", fake_create)

    rag.ask("How do I make Kung Pao Chicken?")

    assert call_count["n"] == 2


def test_infer_route_matches_list_trigger():
    rag = _make_rag()
    assert rag._infer_route("Recommend a few vegetarian dishes") == "list"


def test_infer_route_matches_detail_trigger():
    rag = _make_rag()
    assert rag._infer_route("How do I make Kung Pao Chicken?") == "detail"


def test_infer_route_returns_none_for_ambiguous_question():
    rag = _make_rag()
    assert rag._infer_route("What's the difference between a casserole and a hotpot?") is None


def test_classify_and_rewrite_parses_valid_json(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag,
        "_complete",
        lambda prompt, temperature=0: '{"route": "general", "rewritten": "what is al dente"}',
    )

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what is al dente"


def test_classify_and_rewrite_strips_markdown_code_fence(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag,
        "_complete",
        lambda prompt, temperature=0: '```json\n{"route": "list", "rewritten": "easy dishes"}\n```',
    )

    route, rewritten = rag._classify_and_rewrite("something easy")

    assert route == ROUTE_LIST
    assert rewritten == "easy dishes"


def test_classify_and_rewrite_falls_back_on_invalid_json(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0: "not json at all")

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what does al dente mean?"


def test_classify_and_rewrite_falls_back_when_llm_call_fails(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_complete", lambda prompt, temperature=0: None)

    route, rewritten = rag._classify_and_rewrite("what does al dente mean?")

    assert route == ROUTE_GENERAL
    assert rewritten == "what does al dente mean?"


def test_ask_ambiguous_question_uses_combined_classify_and_rewrite_call(monkeypatch):
    # No list/detail trigger words, so _infer_route() returns None and ask()
    # must fall back to the single combined LLM call instead of two separate
    # router/rewrite calls.
    rag = _make_rag()
    call_count = {"n": 0}

    def fake_complete(prompt, temperature=0.3):
        call_count["n"] += 1
        return '{"route": "general", "rewritten": "what does al dente mean"}'

    monkeypatch.setattr(rag, "_complete", fake_complete)

    rag.ask("What does al dente mean?")

    # 1 call for the combined classify+rewrite, 1 for generation.
    assert call_count["n"] == 2


def test_ask_detail_route_produces_structured_sections(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "query_router", lambda question: ROUTE_DETAIL)
    monkeypatch.setattr(rag, "query_rewrite", lambda question: question)
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_response(
            "## Overview\nA classic dish.\n## Ingredients\n- chicken\n## Steps\n1. Cook it.\n## Tips\nServe hot."
        ),
    )

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert "## Ingredients" in answer
    assert "## Steps" in answer


# ---------------------------------------------------------------------------
# Streaming (_raw_stream_complete / _stream_with_no_results_guard / ask_stream)
# ---------------------------------------------------------------------------


def test_raw_stream_complete_yields_deltas_in_order(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_stream_chunks("Hello", " world"),
    )

    assert list(rag._raw_stream_complete("irrelevant prompt")) == ["Hello", " world"]


def test_raw_stream_complete_yields_nothing_on_auth_failure(monkeypatch):
    rag = _make_rag()

    def raise_error(**kwargs):
        raise AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        )

    monkeypatch.setattr(rag.client.chat.completions, "create", raise_error)

    assert list(rag._raw_stream_complete("irrelevant prompt")) == []


def test_stream_guard_flushes_buffer_once_diverged(monkeypatch):
    rag = _make_rag()
    # "No relevant thing" stops being a possible NO_RESULTS_MESSAGE prefix
    # partway through (the real message continues "...information..."), so
    # everything buffered up to that point should flush as one chunk, then
    # the rest should stream straight through unbuffered.
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_stream_chunks("No", " relevant", " thing", " here"),
    )

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    assert chunks == ["No relevant thing", " here"]
    assert state == {}


def test_stream_guard_detects_exact_no_results_message(monkeypatch):
    rag = _make_rag()
    half = len(NO_RESULTS_MESSAGE) // 2
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_stream_chunks(
            NO_RESULTS_MESSAGE[:half], NO_RESULTS_MESSAGE[half:]
        ),
    )

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    # Nothing should reach the caller -- ask_stream() relies on this to try
    # TheMealDB before anything is shown to the user.
    assert chunks == []
    assert state == {"is_no_results": True}


def test_stream_guard_marks_failed_when_call_fails(monkeypatch):
    rag = _make_rag()

    def raise_error(**kwargs):
        raise AuthenticationError(
            message="bad key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com")),
            body=None,
        )

    monkeypatch.setattr(rag.client.chat.completions, "create", raise_error)

    state = {}
    chunks = list(rag._stream_with_no_results_guard("prompt", 0.3, state))

    assert chunks == []
    assert state == {"failed": True}


def test_ask_stream_list_route_yields_single_chunk_and_never_calls_llm(monkeypatch):
    rag = _make_rag()

    def fail(**kwargs):
        raise AssertionError("LLM should not be called for a rule-matched list route")

    monkeypatch.setattr(rag.client.chat.completions, "create", fail)

    chunks = list(rag.ask_stream("Recommend a few dessert recipes"))

    assert len(chunks) == 1
    assert chunks[0].startswith("Here are some recipes you might like:")


def test_ask_stream_detail_route_joins_to_full_answer(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        _make_streaming_aware_create(
            ["## Overview\n", "A classic dish."],
            non_stream_text="How do I make Kung Pao Chicken?",
        ),
    )

    chunks = list(rag.ask_stream("How do I make Kung Pao Chicken?"))

    # Streamed in more than one piece, but joins to exactly what the
    # non-streaming generation would have produced from the same reply.
    assert len(chunks) >= 1
    assert "".join(chunks) == "## Overview\nA classic dish."


def test_ask_stream_empty_question_yields_prompt_message_without_llm_call(monkeypatch):
    rag = _make_rag()

    def fail(**kwargs):
        raise AssertionError("LLM should not be called for an empty question")

    monkeypatch.setattr(rag.client.chat.completions, "create", fail)

    assert list(rag.ask_stream("   ")) == ["Please enter a question."]


def test_ask_stream_withholds_no_results_message_until_fallback_resolves(monkeypatch):
    from rag import Recipe

    rag = _make_rag()
    fallback_recipe = Recipe(
        text="# Moussaka\n\n## Ingredients\n- eggplant\n\n## Steps\n1. Bake it.",
        source="themealdb:999",
        dish_name="moussaka",
        category="Other",
        difficulty="Unknown",
    )
    monkeypatch.setattr(
        rag, "_themealdb_fallback", lambda question, category=None: fallback_recipe
    )

    call_log = {"stream_calls": 0}

    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            return _fake_response("How do I make Moussaka?")
        call_log["stream_calls"] += 1
        if call_log["stream_calls"] == 1:
            return _fake_stream_chunks(NO_RESULTS_MESSAGE)
        return _fake_stream_chunks("## Overview\n", "Real Moussaka answer.")

    monkeypatch.setattr(rag.client.chat.completions, "create", fake_create)

    chunks = list(rag.ask_stream("How do I make Moussaka?"))
    answer = "".join(chunks)

    # The withheld NO_RESULTS_MESSAGE from the first (unsuccessful) attempt
    # must never reach the caller -- only the TheMealDB-backed answer should.
    assert NO_RESULTS_MESSAGE not in answer
    assert answer == "## Overview\nReal Moussaka answer."
    assert call_log["stream_calls"] == 2


def test_ask_stream_yields_no_results_message_when_fallback_also_finds_nothing(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_themealdb_fallback", lambda question, category=None: None)

    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            return _fake_response("How do I make Moussaka?")
        return _fake_stream_chunks(NO_RESULTS_MESSAGE)

    monkeypatch.setattr(rag.client.chat.completions, "create", fake_create)

    chunks = list(rag.ask_stream("How do I make Moussaka?"))

    assert "".join(chunks) == NO_RESULTS_MESSAGE
