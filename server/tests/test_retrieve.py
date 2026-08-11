# Test the retrieval logic in rag.py itself
# whether the knowledge base loads correctly
# whether keyword retrieval is accurate
# top_k edge cases, validation of parent-child chunking
import os

from rag import SimpleRAG

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes")


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
    # A query with no category/difficulty trigger words (see
    # test_retrieve_infers_category_filter_from_question below for those),
    # so nothing narrows the search and top_k=1000 should return every
    # recipe in the library.
    results = rag.retrieve("What can I cook tonight?", top_k=1000)
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


def test_recipes_are_tagged_with_category_and_difficulty_metadata():
    rag = _make_rag()
    kung_pao = next(doc for doc in rag.documents if doc.dish_name == "kung pao chicken")
    assert kung_pao.category == "Meat"
    # 4 numbered steps in the source file -> "Medium" per the step-count heuristic.
    assert kung_pao.difficulty == "Medium"

    dessert_names = {doc.dish_name for doc in rag.documents if doc.category == "Dessert"}
    assert "mango pudding" in dessert_names
    assert "victoria sponge cake" in dessert_names


def test_retrieve_ranks_parent_by_number_of_matching_sections():
    rag = _make_rag()
    # A question that echoes both the ingredients and the steps of one
    # recipe should rank that recipe above one that only matches a single
    # section, because it accumulates more shortlist hits.
    results = rag.retrieve(
        "chicken breast peanuts dried chili peppers marinate stir-fry sauce mixture",
        top_k=1,
    )
    assert "Kung Pao Chicken" in results[0]


def test_retrieve_infers_category_filter_from_question():
    rag = _make_rag()
    # "vegetarian" should trigger an automatic category=Vegetable filter, so
    # every returned recipe should actually be tagged Vegetable, even though
    # the query itself never names a specific dish.
    results = rag.retrieve("Suggest a vegetarian dish", top_k=10)
    assert results  # sanity: the filter shouldn't wipe out every candidate
    categories = {
        doc.category
        for doc in rag.documents
        if doc.text in results
    }
    assert categories == {"Vegetable"}


def test_retrieve_infers_difficulty_filter_from_question():
    rag = _make_rag()
    results = rag.retrieve("What's an easy recipe for a beginner?", top_k=10)
    assert results
    difficulties = {doc.difficulty for doc in rag.documents if doc.text in results}
    assert difficulties == {"Easy"}


def test_retrieve_explicit_filter_overrides_inference():
    rag = _make_rag()
    # No category/difficulty trigger words in the question text, but an
    # explicit filter is still honored.
    results = rag.retrieve("What's for dinner?", top_k=10, category="Dessert")
    assert results
    categories = {doc.category for doc in rag.documents if doc.text in results}
    assert categories == {"Dessert"}


def test_retrieve_filter_matching_nothing_falls_back_to_unfiltered_search():
    rag = _make_rag()
    # No recipe is difficulty="Hard" in this knowledge base (max is
    # "Medium"), so an explicit Hard filter should be dropped rather than
    # returning an empty list.
    results = rag.retrieve("How do I make Kung Pao Chicken?", top_k=1, difficulty="Hard")
    assert len(results) == 1


def test_documents_split_on_third_level_headings_too():
    # _split_by_headings should break out "### " sub-sections from their
    # parent "## " section instead of leaving them merged together.
    chunks = SimpleRAG._split_by_headings(
        "# Dish\n\n## Notes\nGeneral notes.\n\n### Simple Version\nDo less.\n\n### Advanced Version\nDo more.\n"
    )
    assert len(chunks) == 4
    assert chunks[0].startswith("# Dish")
    assert chunks[1].startswith("## Notes")
    assert chunks[2].startswith("### Simple Version")
    assert chunks[3].startswith("### Advanced Version")
