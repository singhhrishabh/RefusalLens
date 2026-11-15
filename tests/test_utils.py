"""Tests for refusallens.utils."""

import pytest
from refusallens.utils import (
    Edit,
    RefusalResult,
    Explanation,
    ResponseCache,
    word_level_distance,
    get_changed_tokens,
    extract_trigger_phrase,
    normalize_text,
    COMMON_SYNONYMS,
)


class TestWordLevelDistance:
    def test_identical(self):
        assert word_level_distance("hello world", "hello world") == 0

    def test_one_substitution(self):
        assert word_level_distance("how to hack a wifi", "how to secure a wifi") == 1

    def test_one_deletion(self):
        assert word_level_distance("how to hack a wifi", "how to a wifi") == 1

    def test_one_insertion(self):
        assert word_level_distance("how to hack wifi", "how to hack a wifi") == 1

    def test_empty_strings(self):
        assert word_level_distance("", "") == 0

    def test_all_different(self):
        assert word_level_distance("a b c", "x y z") == 3


class TestGetChangedTokens:
    def test_single_change(self):
        result = get_changed_tokens("how to hack a wifi", "how to secure a wifi")
        assert any(orig == "hack" and new == "secure" for orig, new in result)

    def test_no_change(self):
        result = get_changed_tokens("hello world", "hello world")
        assert result == []

    def test_deletion(self):
        result = get_changed_tokens("how to hack a wifi", "how to a wifi")
        assert len(result) > 0


class TestExtractTriggerPhrase:
    def test_single_word_trigger(self):
        trigger, replacement = extract_trigger_phrase(
            "how to hack a wifi", "how to secure a wifi"
        )
        assert trigger is not None
        assert replacement is not None

    def test_no_difference(self):
        trigger, replacement = extract_trigger_phrase("hello world", "hello world")
        assert trigger is None or trigger == ""


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache()
        cache.put("hello", "world")
        assert cache.get("hello") == "world"

    def test_miss(self):
        cache = ResponseCache()
        assert cache.get("missing") is None

    def test_query_caches(self):
        cache = ResponseCache()
        call_count = 0

        def model_fn(prompt):
            nonlocal call_count
            call_count += 1
            return f"response to {prompt}"

        r1 = cache.query(model_fn, "test prompt")
        r2 = cache.query(model_fn, "test prompt")
        assert r1 == r2
        assert call_count == 1  # second call used cache


class TestEditDataclass:
    def test_create_edit(self):
        e = Edit(
            original_text="how to hack",
            edited_text="how to secure",
            edit_type="token_substitution",
            description="Replaced 'hack' with 'secure'",
            distance=1,
            changed_tokens=[("hack", "secure")],
        )
        assert e.distance == 1
        assert e.edit_type == "token_substitution"

    def test_default_changed_tokens(self):
        e = Edit(
            original_text="a",
            edited_text="b",
            edit_type="test",
            description="test",
            distance=1,
        )
        assert e.changed_tokens == []


class TestNormalizeText:
    def test_lowercase_strip(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""


class TestCommonSynonyms:
    def test_has_entries(self):
        assert len(COMMON_SYNONYMS) > 20

    def test_known_keys(self):
        assert "hack" in COMMON_SYNONYMS
        assert "exploit" in COMMON_SYNONYMS
