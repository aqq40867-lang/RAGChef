# Test the retrieval logic in rag.py itself
# whether the knowledge base loads correctly
# whether keyword retrieval is accurate
# top_k edge cases, validation of parent-child chunking
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


def test_documents_are_split_into_child_chunks():
    rag = _make_rag()
    # Each recipe has a title, an Ingredients section, and a Steps section,
    # so there should be strictly more child chunks than parent documents.
    assert len(rag.child_chunks) > len(rag.documents)
    assert len(rag.child_chunks) == len(rag.child_to_parent)


def test_child_chunk_matches_map_back_to_full_parent_document():
    rag = _make_rag()
    # A narrow, ingredient-specific question should match a child chunk
    # (the Ingredients section) but retrieve() must still return the whole
    # parent recipe, not just that fragment.
    results = rag.retrieve("What ingredients do I need for Kung Pao Chicken?", top_k=1)
    assert "## Steps" in results[0]
    assert "## Ingredients" in results[0]


def test_retrieve_does_not_return_duplicate_parents_for_same_recipe():
    rag = _make_rag()
    # Even if multiple child chunks of the same recipe rank highly, the
    # parent recipe should only appear once in the results.
    results = rag.retrieve("Kung Pao Chicken ingredients and steps", top_k=5)
    assert len(results) == len(set(results))
