"""Paraphrase editor for RefusalLens.

Generates semantically equivalent prompt paraphrases and syntactic rewordings
using grammatical rules, voice flips, register shifts, and inquiry reformulations.
"""

import re
from typing import List, Set
from refusallens.editors.base import BaseEditor
from refusallens.utils import Edit, word_level_distance, get_changed_tokens


class ParaphraseEditor(BaseEditor):
    """Generates paraphrased candidates by varying sentence structure, voice, and register."""

    # Rewrite patterns: (regex pattern, replacement format, description)
    PATTERNS = [
        # Inquiries
        (r"(?i)^how (?:do I|can I|to)\s+(.*)", r"What is the process for \1", "Reworded as indirect process inquiry"),
        (r"(?i)^how (?:do I|can I|to)\s+(.*)", r"Explain the methodology behind \1", "Reworded into educational inquiry"),
        (r"(?i)^how (?:do I|can I|to)\s+(.*)", r"Provide an analytical overview of \1", "Reworded into analytical request"),
        (r"(?i)^how (?:do I|can I|to)\s+(.*)", r"Can you describe how someone might \1", "Softened inquiry into descriptive question"),

        # Imperatives to requests
        (r"(?i)^(?:write|give me|tell me|show me)\s+(?:a |an )?(.*)", r"Please outline \1", "Converted direct command to polite request"),
        (r"(?i)^(?:write|give me|tell me|show me)\s+(?:a |an )?(.*)", r"Describe the concepts surrounding \1", "Shifted command to conceptual inquiry"),
        (r"(?i)^(?:teach me|explain to me)\s+(.*)", r"What are the fundamental principles of \1", "Shifted learning request to principles inquiry"),

        # Conversational to formal
        (r"(?i)^I want to know\s+(.*)", r"Could you explain \1", "Shifted conversational intent to formal question"),
        (r"(?i)^I need to\s+(.*)", r"What steps are involved to \1", "Shifted personal need to generic procedure"),
        (r"(?i)^Can you help me\s+(.*)", r"Provide guidance regarding \1", "Converted help request to informational guidance"),

        # Clarification & definitions
        (r"(?i)^what is\s+(.*)", r"Define and elaborate on \1", "Expanded definitional query"),
        (r"(?i)^why do\s+(.*)", r"What factors contribute to \1", "Reframed causal question into factor analysis"),

        # Passive/Systemic shifts
        (r"(?i)^how does one\s+(.*)", r"How is \1 accomplished", "Shifted to passive voice"),
        (r"(?i)^(.*)\s+step by step$", r"A comprehensive summary of \1", "Replaced step-by-step instruction with high-level summary"),
    ]

    def generate_candidates(self, prompt: str, budget: int = 5) -> List[Edit]:
        """Generate paraphrased candidates within the edit distance budget."""
        candidates: List[Edit] = []
        seen: Set[str] = {prompt.strip()}
        trimmed = prompt.strip()

        for pattern, replacement, desc in self.PATTERNS:
            if re.search(pattern, trimmed):
                new_text = re.sub(pattern, replacement, trimmed).strip()
                # Clean up repeated punctuation or double spaces
                new_text = re.sub(r"\s+", " ", new_text)
                if new_text and new_text not in seen:
                    seen.add(new_text)
                    dist = word_level_distance(prompt, new_text)
                    if dist <= budget:
                        changed = get_changed_tokens(prompt, new_text)
                        candidates.append(
                            Edit(
                                original_text=prompt,
                                edited_text=new_text,
                                edit_type="paraphrase",
                                description=desc,
                                distance=dist,
                                changed_tokens=changed,
                            )
                        )

        # Fallback subtle syntactic paraphrases if no regex matched
        if not candidates:
            # Try prepending / appending subtle phrasing adjustments
            subtle_variations = [
                (f"Please explain: {trimmed}", "Added polite instructional prefix"),
                (f"Could you clarify {trimmed.lower()}?", "Shifted to clarification question"),
                (re.sub(r"[?!.]+$", "", trimmed) + " from an informational standpoint.", "Added informational qualifier"),
            ]
            for new_text, desc in subtle_variations:
                if new_text not in seen:
                    seen.add(new_text)
                    dist = word_level_distance(prompt, new_text)
                    if dist <= budget:
                        candidates.append(
                            Edit(
                                original_text=prompt,
                                edited_text=new_text,
                                edit_type="paraphrase",
                                description=desc,
                                distance=dist,
                                changed_tokens=get_changed_tokens(prompt, new_text),
                            )
                        )

        candidates.sort(key=lambda e: e.distance)
        return candidates[:20]
