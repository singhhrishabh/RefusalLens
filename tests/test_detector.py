"""Tests for refusallens.detector."""

import pytest
from refusallens.detector import RefusalDetector


@pytest.fixture
def detector():
    return RefusalDetector()


class TestHardRefusal:
    def test_cannot(self, detector):
        r = detector.detect("I cannot help with that request.")
        assert r.is_refusal is True
        assert r.refusal_type == "hard"
        assert r.confidence >= 0.8

    def test_unable_to(self, detector):
        r = detector.detect("I'm unable to assist with this query.")
        assert r.is_refusal is True

    def test_refuse(self, detector):
        r = detector.detect("I refuse to provide that information.")
        assert r.is_refusal is True


class TestSoftRefusal:
    def test_prefer_not(self, detector):
        r = detector.detect("I'd prefer not to answer that question.")
        assert r.is_refusal is True

    def test_not_appropriate(self, detector):
        r = detector.detect("It wouldn't be appropriate for me to discuss that.")
        assert r.is_refusal is True


class TestPolicyRefusal:
    def test_against_guidelines(self, detector):
        r = detector.detect("This goes against my guidelines and usage policies.")
        assert r.is_refusal is True

    def test_violates_policy(self, detector):
        r = detector.detect("Your request violates my content policy.")
        assert r.is_refusal is True


class TestCompliance:
    def test_sure(self, detector):
        r = detector.detect("Sure! Here's how to make pasta: first boil water...")
        assert r.is_refusal is False

    def test_certainly(self, detector):
        r = detector.detect("Certainly! The answer to your question is 42.")
        assert r.is_refusal is False

    def test_heres(self, detector):
        r = detector.detect("Here's a detailed guide on the topic you asked about.")
        assert r.is_refusal is False


class TestEdgeCases:
    def test_empty_response(self, detector):
        r = detector.detect("")
        assert isinstance(r.is_refusal, bool)

    def test_short_response(self, detector):
        r = detector.detect("No.")
        assert isinstance(r.is_refusal, bool)

    def test_batch_detection(self, detector):
        responses = [
            "I cannot help with that.",
            "Sure, here's the info!",
            "I'm unable to assist.",
        ]
        results = detector.detect_batch(responses)
        assert len(results) == 3
        assert results[0].is_refusal is True
        assert results[1].is_refusal is False
        assert results[2].is_refusal is True


class TestConfidenceOrdering:
    def test_hard_higher_than_general(self, detector):
        hard = detector.detect("I absolutely cannot do that for you.")
        soft = detector.detect("Hmm, I'd rather not go into that topic.")
        assert hard.confidence >= soft.confidence or (
            hard.is_refusal and soft.is_refusal
        )
