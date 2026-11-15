"""Token-level editor for RefusalLens.

Performs word/token substitutions and deletions to find minimal counterfactual edits.
Supports dictionary-based neutral substitutions, WordNet synonyms, and word deletion.
"""

import re
from typing import List, Tuple, Set
from refusallens.editors.base import BaseEditor
from refusallens.utils import (
    Edit,
    COMMON_SYNONYMS,
    word_level_distance,
    get_changed_tokens,
)

# Try importing nltk WordNet, fallback gracefully if not installed/downloaded
_WORDNET_AVAILABLE = False
try:
    from nltk.corpus import wordnet as wn
    # Quick probe
    wn.synsets("test")
    _WORDNET_AVAILABLE = True
except Exception:
    _WORDNET_AVAILABLE = False


class TokenEditor(BaseEditor):
    """Generates candidate prompt edits by substituting or deleting individual tokens/words."""

    def __init__(self, use_wordnet: bool = True):
        self.synonyms = dict(COMMON_SYNONYMS)
        self.use_wordnet = use_wordnet and _WORDNET_AVAILABLE

    def _get_synonyms(self, word: str) -> List[str]:
        """Retrieve potential replacement words for a given word."""
        clean = word.lower().strip(".,!?:;\"'()[]{}")
        replacements: List[str] = []

        # 1. From predefined sensitive/neutral map
        if clean in self.synonyms:
            replacements.append(self.synonyms[clean])

        # 2. WordNet synonyms if enabled
        if self.use_wordnet:
            try:
                synsets = wn.synsets(clean)
                for syn in synsets[:3]:
                    for lemma in syn.lemmas()[:3]:
                        name = lemma.name().replace("_", " ").lower()
                        if name != clean and name not in replacements:
                            replacements.append(name)
            except Exception:
                pass

        return replacements

    def generate_candidates(self, prompt: str, budget: int = 3) -> List[Edit]:
        """Generate candidate edits under the given word-level distance budget."""
        words = prompt.split()
        if not words:
            return []

        candidates: List[Edit] = []
        seen_texts: Set[str] = {prompt}

        # 1. Token substitutions (single word)
        for i, word in enumerate(words):
            clean_word = word.strip(".,!?:;\"'()[]{}")
            if not clean_word:
                continue

            # Preserving punctuation
            prefix = word[: len(word) - len(word.lstrip(".,!?:;\"'()[]{}"))]
            suffix = word[len(word.rstrip(".,!?:;\"'()[]{}")) :]

            replacements = self._get_synonyms(clean_word)
            for rep in replacements:
                # Preserve capitalization
                if clean_word.isupper():
                    rep_formatted = rep.upper()
                elif clean_word.istitle():
                    rep_formatted = rep.capitalize()
                else:
                    rep_formatted = rep

                new_word = f"{prefix}{rep_formatted}{suffix}"
                new_words = list(words)
                new_words[i] = new_word
                new_text = " ".join(new_words)

                if new_text not in seen_texts:
                    seen_texts.add(new_text)
                    dist = word_level_distance(prompt, new_text)
                    if dist <= budget:
                        candidates.append(
                            Edit(
                                original_text=prompt,
                                edited_text=new_text,
                                edit_type="token_substitution",
                                description=f"Replaced '{clean_word}' with '{rep}'",
                                distance=dist,
                                changed_tokens=[(clean_word, rep)],
                            )
                        )

        # 2. Deletions (if prompt has >= 3 words)
        if len(words) >= 3:
            for i, word in enumerate(words):
                clean_word = word.strip(".,!?:;\"'()[]{}")
                # Only delete meaningful or sensitive tokens
                if clean_word.lower() in self.synonyms or len(clean_word) > 3:
                    new_words = words[:i] + words[i + 1 :]
                    new_text = " ".join(new_words)
                    if new_text not in seen_texts:
                        seen_texts.add(new_text)
                        dist = word_level_distance(prompt, new_text)
                        if dist <= budget:
                            candidates.append(
                                Edit(
                                    original_text=prompt,
                                    edited_text=new_text,
                                    edit_type="deletion",
                                    description=f"Removed '{clean_word}'",
                                    distance=dist,
                                    changed_tokens=[(clean_word, "")],
                                )
                            )

        # 3. Double token substitutions (if budget >= 2)
        if budget >= 2 and len(candidates) < 50:
            sensitive_indices = [
                i
                for i, w in enumerate(words)
                if w.lower().strip(".,!?:;\"'()[]{}") in self.synonyms
            ]
            if len(sensitive_indices) >= 2:
                for idx_a in sensitive_indices:
                    for idx_b in sensitive_indices:
                        if idx_a < idx_b:
                            wa = words[idx_a].strip(".,!?:;\"'()[]{}")
                            wb = words[idx_b].strip(".,!?:;\"'()[]{}")
                            repa = self.synonyms.get(wa.lower())
                            repb = self.synonyms.get(wb.lower())
                            if repa and repb:
                                new_words = list(words)
                                new_words[idx_a] = repa
                                new_words[idx_b] = repb
                                new_text = " ".join(new_words)
                                if new_text not in seen_texts:
                                    seen_texts.add(new_text)
                                    dist = word_level_distance(prompt, new_text)
                                    if dist <= budget:
                                        candidates.append(
                                            Edit(
                                                original_text=prompt,
                                                edited_text=new_text,
                                                edit_type="token_substitution",
                                                description=f"Replaced '{wa}' with '{repa}' and '{wb}' with '{repb}'",
                                                distance=dist,
                                                changed_tokens=[(wa, repa), (wb, repb)],
                                            )
                                        )

        # Sort by edit distance
        candidates.sort(key=lambda e: e.distance)
        return candidates[:30]
