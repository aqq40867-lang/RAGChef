import os

from rag import SimpleRAG

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.md")


def _make_rag():
    return SimpleRAG(RECIPES_PATH)


def test_knowledge_base_has_fifty_recipes():
    rag = _make_rag()
    assert len(rag.documents) == 50


def test_retrieve_returns_relevant_document():
    rag = _make_rag()
    results = rag.retrieve("How do I make Kung Pao Chicken?", top_k=1)
    assert len(results) == 1
    assert "Kung Pao Chicken" in results[0]


def test_retrieve_top_k_is_capped_at_document_count():
    rag = _make_rag()
    results = rag.retrieve("soup", top_k=1000)
    assert len(results) == len(rag.documents)


def test_retrieve_soup_query_returns_a_soup_recipe():
    rag = _make_rag()
    results = rag.retrieve("How do I make egg drop soup?", top_k=1)
    assert "Soup" in results[0]


def test_retrieve_finds_british_recipe():
    rag = _make_rag()
    results = rag.retrieve("How do I make fish and chips?", top_k=1)
    assert "Fish and Chips" in results[0]
