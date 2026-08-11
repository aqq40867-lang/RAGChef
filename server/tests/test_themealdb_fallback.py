"""Tests for the TheMealDB fallback used when the local library has no match.

All tests here stub out SimpleRAG._themealdb_get (and, where relevant,
_rank_documents) instead of hitting the real TheMealDB API or DeepSeek, so
the suite stays fast and network-free like the rest of tests/.
"""

import os
from types import SimpleNamespace

import rag as rag_module
from rag import ROUTE_GENERAL, ROUTE_LIST, SimpleRAG

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes")

# A trimmed real-shaped TheMealDB "meal" object (see themealdb.com/api.php),
# used across several tests below.
_KUNG_PAO_MEAL = {
    "idMeal": "52945",
    "strMeal": "Kung Pao Chicken",
    "strCategory": "Chicken",
    "strArea": "Chinese",
    "strInstructions": "Marinate the chicken.\nStir-fry until cooked through.\nAdd the sauce and toss.",
    "strIngredient1": "Chicken",
    "strMeasure1": "500g",
    "strIngredient2": "Peanuts",
    "strMeasure2": "100g",
    "strIngredient3": "",
    "strMeasure3": "",
}


def _make_rag():
    return SimpleRAG(RECIPES_PATH)


def _fake_llm_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_themealdb_dish_query_strips_question_phrasing():
    assert (
        SimpleRAG._themealdb_dish_query("How do I make Kung Pao Chicken?")
        == "Kung Pao Chicken"
    )
    assert SimpleRAG._themealdb_dish_query("Recipe for beef wellington") == "beef wellington"
    # Already-bare dish names should pass through unchanged.
    assert SimpleRAG._themealdb_dish_query("Ma Po Tofu") == "Ma Po Tofu"


def test_themealdb_meal_to_recipe_builds_expected_markdown():
    recipe = SimpleRAG._themealdb_meal_to_recipe(_KUNG_PAO_MEAL)

    assert recipe.dish_name == "kung pao chicken"
    assert recipe.category == "Meat"  # "Chicken" maps to "Meat"
    assert recipe.difficulty == "Unknown"
    assert recipe.source == "themealdb:52945"
    assert "# Kung Pao Chicken" in recipe.text
    assert "## Ingredients" in recipe.text
    assert "- 500g Chicken" in recipe.text
    assert "- 100g Peanuts" in recipe.text
    assert "## Steps" in recipe.text
    assert "1. Marinate the chicken." in recipe.text
    assert "2. Stir-fry until cooked through." in recipe.text


def test_themealdb_fallback_returns_none_when_disabled(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag_module, "THEMEALDB_ENABLED", False)

    def fail(*args, **kwargs):
        raise AssertionError("TheMealDB should not be called when disabled")

    monkeypatch.setattr(rag, "_themealdb_get", fail)

    assert rag._themealdb_fallback("How do I make a soufflé?") is None


def test_themealdb_fallback_returns_none_when_search_and_filter_both_miss(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_themealdb_get", lambda path, params: {"meals": None})

    assert rag._themealdb_fallback("How do I make a soufflé?", category="Dessert") is None


def test_themealdb_fallback_falls_back_to_category_filter_when_name_search_misses(monkeypatch):
    rag = _make_rag()
    calls = []

    def fake_get(path, params):
        calls.append((path, params))
        if path == "/search.php":
            return {"meals": None}
        if path == "/filter.php":
            return {"meals": [{"idMeal": "52945"}]}
        if path == "/lookup.php":
            assert params == {"i": "52945"}
            return {"meals": [_KUNG_PAO_MEAL]}
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(rag, "_themealdb_get", fake_get)

    recipe = rag._themealdb_fallback("something vegetarian", category="Vegetable")

    assert recipe is not None
    assert recipe.dish_name == "kung pao chicken"
    assert [c[0] for c in calls] == ["/search.php", "/filter.php", "/lookup.php"]


def test_ask_falls_back_to_themealdb_when_local_library_has_no_match(monkeypatch):
    rag = _make_rag()
    monkeypatch.setattr(rag, "_rank_documents", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rag, "_themealdb_get", lambda path, params: {"meals": [_KUNG_PAO_MEAL]}
    )
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_llm_response("Answer grounded in Kung Pao Chicken."),
    )

    answer = rag.ask("How do I make a dish nowhere in the local library?")

    assert answer == "Answer grounded in Kung Pao Chicken."


def test_ask_falls_back_to_themealdb_when_llm_finds_nothing_relevant(monkeypatch):
    # This is what actually happens in practice: _rank_documents always
    # returns its best-effort top_k local matches (see its docstring), even
    # when none of them are relevant, so "nothing relevant" only shows up
    # once the LLM itself says so. Deliberately does NOT stub
    # _rank_documents, so this exercises the real retrieval path (weak/
    # irrelevant local matches) rather than an artificially empty one.
    rag = _make_rag()
    monkeypatch.setattr(rag, "query_router", lambda question: ROUTE_GENERAL)
    monkeypatch.setattr(rag, "query_rewrite", lambda question: question)
    monkeypatch.setattr(
        rag, "_themealdb_get", lambda path, params: {"meals": [_KUNG_PAO_MEAL]}
    )

    responses = iter([rag_module.NO_RESULTS_MESSAGE, "Answer using TheMealDB content."])
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_llm_response(next(responses)),
    )

    answer = rag.ask("How do I make Moussaka?")

    assert answer == "Answer using TheMealDB content."


def test_ask_does_not_call_themealdb_when_local_docs_found(monkeypatch):
    rag = _make_rag()

    def fail(*args, **kwargs):
        raise AssertionError("TheMealDB fallback should not run when local docs were found")

    monkeypatch.setattr(rag, "_themealdb_fallback", fail)
    monkeypatch.setattr(
        rag.client.chat.completions,
        "create",
        lambda **kwargs: _fake_llm_response("Local answer."),
    )

    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert answer == "Local answer."


def test_ask_list_route_never_uses_themealdb_fallback(monkeypatch):
    rag = _make_rag()

    def fail(*args, **kwargs):
        raise AssertionError("List route should not use the TheMealDB fallback")

    monkeypatch.setattr(rag, "query_router", lambda question: ROUTE_LIST)
    monkeypatch.setattr(rag, "_rank_documents", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag, "_themealdb_fallback", fail)

    answer = rag.ask("Recommend a few dishes from a category that doesn't exist")

    assert answer == rag_module.NO_RESULTS_MESSAGE
