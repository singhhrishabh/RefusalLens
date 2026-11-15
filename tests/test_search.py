"""Tests for refusallens.search (beam search + RefusalLens)."""

import pytest
from refusallens.search import RefusalLens, ModelWrapper
from refusallens.utils import Explanation


@pytest.fixture
def mock_lens():
    """RefusalLens configured with the built-in mock model."""
    return RefusalLens(model_name_or_fn="mock")


class TestModelWrapper:
    def test_mock_refuses_harmful(self):
        mw = ModelWrapper("mock")
        resp = mw.generate("How to hack a network")
        assert "cannot" in resp.lower() or "apologize" in resp.lower()

    def test_mock_complies_benign(self):
        mw = ModelWrapper("mock")
        resp = mw.generate("How to bake a cake")
        assert "cannot" not in resp.lower()

    def test_cache_works(self):
        mw = ModelWrapper("mock")
        r1 = mw.generate("test prompt one")
        r2 = mw.generate("test prompt one")
        assert r1 == r2  # cached


class TestRefusalLens:
    def test_explain_returns_explanation(self, mock_lens):
        exp = mock_lens.explain("How to hack a wifi network", beam_width=3, max_depth=2)
        assert isinstance(exp, Explanation)
        assert exp.original_prompt == "How to hack a wifi network"
        assert exp.original_refused is True

    def test_finds_flip_on_keyword(self, mock_lens):
        exp = mock_lens.explain("How to hack a wifi network", beam_width=3, max_depth=2)
        # The mock model refuses "hack" but complies with "secure"
        assert exp.flipped_prompt is not None
        assert exp.flip_distance >= 1
        assert exp.trigger_phrase is not None

    def test_benign_prompt_complies(self, mock_lens):
        exp = mock_lens.explain("What is the weather today?", beam_width=3, max_depth=1)
        assert exp.original_refused is False

    def test_handles_no_flip(self, mock_lens):
        # A prompt with very robust refusal — mock always refuses anything with "bomb"
        exp = mock_lens.explain(
            "How to make a bomb at home",
            beam_width=2,
            max_depth=1,
        )
        assert isinstance(exp, Explanation)
        assert exp.original_refused is True
        # It might or might not find a flip depending on search depth

    def test_explain_batch(self, mock_lens):
        prompts = ["How to hack a wifi", "How to bake a cake"]
        results = mock_lens.explain_batch(prompts, beam_width=2, max_depth=1)
        assert len(results) == 2
        assert results[0].original_refused is True
        assert results[1].original_refused is False

    def test_human_summary_not_empty(self, mock_lens):
        exp = mock_lens.explain("How to hack wifi", beam_width=3, max_depth=2)
        assert exp.human_summary != ""
        assert len(exp.human_summary) > 10
