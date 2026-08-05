# Real end-to-end test against the live DeepSeek API (no mocking).
# Covers: SimpleRAG.ask() returns a non-empty string answer for a real query.
# Skipped by default (needs real DEEPSEEK_API_KEY + RUN_INTEGRATION_TESTS=1),
# since it costs DeepSeek credit and requires network access.
import os

import pytest

from rag import SimpleRAG

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "recipes.md")

_has_real_key = bool(os.environ.get("DEEPSEEK_API_KEY")) and os.environ.get(
    "DEEPSEEK_API_KEY"
) != "test-key-for-unit-tests"
_opted_in = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_has_real_key and _opted_in),
    reason="Set a real DEEPSEEK_API_KEY and RUN_INTEGRATION_TESTS=1 to run this test.",
)


def test_ask_real_deepseek_api_returns_an_answer():
    rag = SimpleRAG(RECIPES_PATH)
    answer = rag.ask("How do I make Kung Pao Chicken?")

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
