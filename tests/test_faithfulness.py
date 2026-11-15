"""Tests for refusallens.faithfulness."""

import pytest
from refusallens.faithfulness import FaithfulnessValidator
from refusallens.detector import RefusalDetector
from refusallens.search import ModelWrapper
from refusallens.utils import Explanation


@pytest.fixture
def validator():
    mw = ModelWrapper("mock")
    det = RefusalDetector()
    return FaithfulnessValidator(mw.generate, det)


class TestFaithfulnessValidator:
    def test_high_faithfulness_for_known_trigger(self, validator):
        exp = Explanation(
            original_prompt="How to hack a wifi",
            original_response="I cannot help.",
            original_refused=True,
            flipped_prompt="How to secure a wifi",
            flipped_response="Sure!",
            flipped_refused=False,
            flip_distance=1,
            trigger_phrase="hack",
            neutral_replacement="secure",
            human_summary="test",
        )
        score = validator.validate(exp)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # "hack" should reliably trigger refusals in mock
        assert score > 0.3

    def test_low_faithfulness_for_benign_trigger(self, validator):
        exp = Explanation(
            original_prompt="Tell me about cats",
            original_response="Sure, cats are great!",
            original_refused=False,
            trigger_phrase="cats",
            neutral_replacement="dogs",
            human_summary="test",
            original_refused_confidence=0.0,
        ) if False else Explanation(
            original_prompt="Tell me about cats",
            original_response="Sure!",
            original_refused=False,
            trigger_phrase="cats",
            neutral_replacement="dogs",
            human_summary="test",
        )
        score = validator.validate(exp)
        assert isinstance(score, float)
        assert score <= 0.5  # "cats" shouldn't cause refusals

    def test_no_trigger_returns_zero(self, validator):
        exp = Explanation(
            original_prompt="test",
            original_response="test",
            original_refused=False,
            trigger_phrase=None,
            human_summary="test",
        )
        assert validator.validate(exp) == 0.0

    def test_generate_held_out_prompts(self, validator):
        prompts = validator.generate_held_out_prompts("hacking", n=5)
        assert len(prompts) == 5
        assert all("hacking" in p.lower() for p in prompts)

    def test_validate_batch(self, validator):
        explanations = [
            Explanation(
                original_prompt="test",
                original_response="I cannot.",
                original_refused=True,
                trigger_phrase="hack",
                neutral_replacement="secure",
                human_summary="test",
            ),
            Explanation(
                original_prompt="test",
                original_response="Sure!",
                original_refused=False,
                trigger_phrase=None,
                human_summary="test",
            ),
        ]
        scores = validator.validate_batch(explanations)
        assert len(scores) == 2
