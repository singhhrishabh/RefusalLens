"""
Shared utilities for RefusalLens.

Provides data classes, caching, text-comparison helpers, prompt builders,
and a synonym dictionary used across every module in the package.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
import hashlib
import re
import difflib
import string

__all__ = [
    # Data classes
    "Edit",
    "RefusalResult",
    "Explanation",
    # Caching
    "ResponseCache",
    # Text helpers
    "word_level_distance",
    "get_changed_tokens",
    "extract_trigger_phrase",
    "compute_semantic_similarity",
    "normalize_text",
    "build_chat_prompt",
    # Constants
    "COMMON_SYNONYMS",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Edit:
    """A single edit operation applied to a prompt.

    Attributes:
        original_text: The text before the edit.
        edited_text: The text after the edit.
        edit_type: One of ``"token_substitution"``, ``"paraphrase"``,
            ``"framing"``, or ``"deletion"``.
        description: Human-readable summary, e.g.
            ``"Replaced 'hack' with 'secure'"``.
        distance: Word-level Levenshtein distance between *original_text*
            and *edited_text*.
        changed_tokens: List of ``(original_word, replacement_word)`` tuples
            that were altered by this edit.
    """

    original_text: str
    edited_text: str
    edit_type: str
    description: str
    distance: int
    changed_tokens: list = field(default_factory=list)


@dataclass
class RefusalResult:
    """Result produced by the refusal detector.

    Attributes:
        is_refusal: ``True`` when the response is classified as a refusal.
        confidence: Detector confidence in ``[0.0, 1.0]``.
        matched_pattern: The regex / keyword pattern that fired, if any.
        refusal_type: One of ``"hard"``, ``"soft"``, ``"partial"``,
            or ``"none"``.
    """

    is_refusal: bool
    confidence: float
    matched_pattern: Optional[str] = None
    refusal_type: str = "none"


@dataclass
class Explanation:
    """Complete refusal explanation returned by :pyclass:`RefusalLens`.

    Attributes:
        original_prompt: The user prompt that was analysed.
        original_response: The model's verbatim response.
        original_refused: Whether the original response was a refusal.
        flipped_prompt: The minimally-edited prompt that flipped the
            refusal decision (``None`` if no flip was found).
        flipped_response: The model's response to the flipped prompt.
        flipped_refused: Whether the flipped response is a refusal.
        edit_chain: Ordered list of :class:`Edit` objects that transform
            *original_prompt* into *flipped_prompt*.
        flip_distance: Total word-level edit distance; ``-1`` means no
            flip was found.
        trigger_phrase: The contiguous span in the original prompt that
            triggered the refusal.
        neutral_replacement: The neutral phrase that replaced the trigger.
        human_summary: A one-sentence, human-readable explanation.
        faithfulness_score: Optional faithfulness validation score in
            ``[0.0, 1.0]``.
    """

    original_prompt: str
    original_response: str
    original_refused: bool
    flipped_prompt: Optional[str] = None
    flipped_response: Optional[str] = None
    flipped_refused: Optional[bool] = None
    edit_chain: list = field(default_factory=list)
    flip_distance: int = -1
    trigger_phrase: Optional[str] = None
    neutral_replacement: Optional[str] = None
    human_summary: str = ""
    faithfulness_score: Optional[float] = None
    alternative_candidates: list = field(default_factory=list)
    semantic_similarity: Optional[float] = None
    harm_category: Optional[str] = None
    mode: str = "refusal_to_compliance"

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to a serializable dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "original_response": self.original_response,
            "original_refused": self.original_refused,
            "flipped_prompt": self.flipped_prompt,
            "flipped_response": self.flipped_response,
            "flipped_refused": self.flipped_refused,
            "flip_distance": self.flip_distance,
            "trigger_phrase": self.trigger_phrase,
            "neutral_replacement": self.neutral_replacement,
            "human_summary": self.human_summary,
            "faithfulness_score": self.faithfulness_score,
            "semantic_similarity": self.semantic_similarity,
            "harm_category": self.harm_category,
            "mode": self.mode,
            "edit_chain": [
                {
                    "edit_type": getattr(e, "edit_type", "unknown"),
                    "description": getattr(e, "description", ""),
                    "distance": getattr(e, "distance", 0),
                    "original_text": getattr(e, "original_text", ""),
                    "edited_text": getattr(e, "edited_text", ""),
                }
                for e in self.edit_chain
            ],
            "alternative_candidates": self.alternative_candidates,
        }


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Cache model responses to avoid redundant API calls.

    Uses a SHA-256 hash of the prompt as the cache key so that even very
    long prompts produce compact, collision-resistant keys.

    Example::

        cache = ResponseCache()
        response = cache.query(my_model_fn, "How do I sort a list?")
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory cache."""
        self._cache: Dict[str, str] = {}

    # -- low-level accessors ------------------------------------------------

    def get(self, prompt: str) -> Optional[str]:
        """Return the cached response for *prompt*, or ``None``."""
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self._cache.get(key)

    def put(self, prompt: str, response: str) -> None:
        """Store *response* under the key derived from *prompt*."""
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self._cache[key] = response

    # -- convenience --------------------------------------------------------

    def query(self, model_fn: Callable[[str], str], prompt: str) -> str:
        """Return a (possibly cached) response for *prompt*.

        If the prompt has been seen before the cached response is returned
        immediately; otherwise *model_fn(prompt)* is called, the result is
        cached, and then returned.

        Args:
            model_fn: A callable that accepts a prompt string and returns
                the model's response string.
            prompt: The prompt to send to the model.

        Returns:
            The model response string.
        """
        cached = self.get(prompt)
        if cached is not None:
            return cached
        response = model_fn(prompt)
        self.put(prompt, response)
        return response

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Text comparison helpers
# ---------------------------------------------------------------------------


def word_level_distance(a: str, b: str) -> int:
    """Compute the word-level Levenshtein distance between two strings.

    Words are obtained by splitting on whitespace.  Each word is an atomic
    token — substitution / insertion / deletion each cost 1.

    This uses the standard dynamic-programming algorithm with an
    *O(n × m)* matrix where *n* and *m* are the word counts of *a* and *b*.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The minimum number of single-word edits (insertions, deletions, or
        substitutions) required to transform *a* into *b*.

    Examples:
        >>> word_level_distance("the quick brown fox", "the slow brown fox")
        1
        >>> word_level_distance("hello world", "hello beautiful world")
        1
    """
    words_a: List[str] = a.split()
    words_b: List[str] = b.split()

    n = len(words_a)
    m = len(words_b)

    # dp[i][j] = edit distance between words_a[:i] and words_b[:j]
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if words_a[i - 1] == words_b[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )

    return dp[n][m]


def get_changed_tokens(
    original: str, edited: str
) -> List[Tuple[str, str]]:
    """Return a list of ``(original_word, new_word)`` tuples for words that
    were changed between *original* and *edited*.

    Uses :class:`difflib.SequenceMatcher` to align the two word sequences
    and then emits a pair for every *replace* operation.  Insertions appear
    as ``("", new_word)`` and deletions as ``(old_word, "")``.

    Args:
        original: The original text.
        edited: The edited text.

    Returns:
        A list of ``(original_word, new_word)`` tuples describing every
        word-level change.

    Examples:
        >>> get_changed_tokens("how to hack wifi", "how to secure wifi")
        [('hack', 'secure')]
    """
    words_a = original.split()
    words_b = edited.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    changes: List[Tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            # Pair up replaced words one-to-one; excess on either side is
            # treated as insertion / deletion.
            orig_span = words_a[i1:i2]
            edit_span = words_b[j1:j2]
            max_len = max(len(orig_span), len(edit_span))
            for k in range(max_len):
                orig_word = orig_span[k] if k < len(orig_span) else ""
                edit_word = edit_span[k] if k < len(edit_span) else ""
                changes.append((orig_word, edit_word))
        elif tag == "delete":
            for word in words_a[i1:i2]:
                changes.append((word, ""))
        elif tag == "insert":
            for word in words_b[j1:j2]:
                changes.append(("", word))

    return changes


def extract_trigger_phrase(
    original: str, edited: str
) -> Tuple[str, str]:
    """Find the contiguous changed region between *original* and *edited*.

    Walks the diff opcodes produced by :func:`difflib.SequenceMatcher` and
    merges all contiguous non-equal spans into a single trigger /
    replacement pair.  When changes are scattered, the span covers
    everything from the first change to the last (inclusive of any
    intervening equal words).

    Args:
        original: The original prompt text.
        edited: The edited prompt text.

    Returns:
        A ``(trigger_phrase, replacement_phrase)`` tuple.  Both are
        whitespace-joined word spans.  If the two texts are identical
        both strings are empty.

    Examples:
        >>> extract_trigger_phrase(
        ...     "how to hack into wifi",
        ...     "how to secure wifi access"
        ... )
        ('hack into wifi', 'secure wifi access')
    """
    words_a = original.split()
    words_b = edited.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    opcodes = matcher.get_opcodes()

    # Find the index range in each word list that spans all non-equal ops.
    first_a: Optional[int] = None
    last_a: Optional[int] = None
    first_b: Optional[int] = None
    last_b: Optional[int] = None

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        # Update the boundaries to the widest span that covers every change.
        if first_a is None or i1 < first_a:
            first_a = i1
        if last_a is None or i2 > last_a:
            last_a = i2
        if first_b is None or j1 < first_b:
            first_b = j1
        if last_b is None or j2 > last_b:
            last_b = j2

    if first_a is None:
        # No differences found.
        return ("", "")

    trigger_phrase = " ".join(words_a[first_a:last_a])
    replacement_phrase = " ".join(words_b[first_b:last_b])
    return (trigger_phrase, replacement_phrase)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute lexical and sequence semantic retention between two texts in [0.0, 1.0].

    Blends word-level Jaccard overlap with Levenshtein-based sequence similarity
    to gauge how closely the counterfactual preserves the original prompt's structure.
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    words1 = [w.strip(string.punctuation).lower() for w in text1.split() if w.strip(string.punctuation)]
    words2 = [w.strip(string.punctuation).lower() for w in text2.split() if w.strip(string.punctuation)]
    set1, set2 = set(words1), set(words2)
    jaccard = len(set1 & set2) / max(len(set1 | set2), 1)

    seq = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return round(0.5 * jaccard + 0.5 * seq, 3)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalise *text* for comparison purposes.

    * Convert to lowercase.
    * Strip all ASCII punctuation.
    * Collapse runs of whitespace to a single space.
    * Strip leading / trailing whitespace.

    Args:
        text: Arbitrary input text.

    Returns:
        The normalised string.

    Examples:
        >>> normalize_text("Hello, World!  How's it going?")
        'hello world hows it going'
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_chat_prompt(
    prompt: str, system_msg: Optional[str] = None
) -> List[Dict[str, str]]:
    """Build a chat-style messages list from a user prompt.

    The returned list follows the OpenAI / HuggingFace chat format::

        [
            {"role": "system", "content": "..."},   # only if system_msg
            {"role": "user",   "content": "..."},
        ]

    Args:
        prompt: The user's prompt text.
        system_msg: An optional system message prepended to the
            conversation.

    Returns:
        A list of message dicts with ``"role"`` and ``"content"`` keys.

    Examples:
        >>> build_chat_prompt("hi")
        [{'role': 'user', 'content': 'hi'}]
        >>> build_chat_prompt("hi", system_msg="Be helpful.")
        [{'role': 'system', 'content': 'Be helpful.'}, {'role': 'user', 'content': 'hi'}]
    """
    messages: List[Dict[str, str]] = []
    if system_msg is not None:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    return messages


# ---------------------------------------------------------------------------
# Common synonyms mapping
# ---------------------------------------------------------------------------

COMMON_SYNONYMS: Dict[str, str] = {
    # Security / hacking
    "hack": "secure",
    "hacking": "securing",
    "hacker": "researcher",
    "exploit": "utilise",
    "exploiting": "utilising",
    "crack": "solve",
    "cracking": "solving",
    "bypass": "navigate",
    "breach": "access",
    "intrusion": "access",
    "phishing": "outreach",
    "malware": "software",
    "ransomware": "software",
    "trojan": "program",
    "rootkit": "toolkit",
    "vulnerability": "weakness",
    "backdoor": "access point",
    # Violence / weapons
    "kill": "stop",
    "killing": "stopping",
    "murder": "end",
    "attack": "test",
    "attacking": "testing",
    "bomb": "device",
    "bombing": "deployment",
    "weapon": "tool",
    "weapons": "tools",
    "gun": "device",
    "explosive": "material",
    "shoot": "aim",
    "shooting": "aiming",
    "destroy": "remove",
    "destroying": "removing",
    "assassinate": "replace",
    "strangle": "restrain",
    "stab": "poke",
    # Drugs / substances
    "drug": "medicine",
    "drugs": "medicines",
    "narcotic": "medication",
    "cocaine": "substance",
    "heroin": "compound",
    "meth": "chemical",
    "overdose": "overuse",
    # Theft / fraud
    "steal": "take",
    "stealing": "taking",
    "theft": "transfer",
    "rob": "collect",
    "robbery": "collection",
    "fraud": "error",
    "counterfeit": "replica",
    "forge": "craft",
    "forging": "crafting",
    "scam": "offer",
    "launder": "process",
    "embezzle": "redirect",
    # Harmful / dangerous
    "poison": "substance",
    "toxic": "strong",
    "dangerous": "challenging",
    "illegal": "restricted",
    "illicit": "unofficial",
    "smuggle": "transport",
    "smuggling": "transporting",
    "torture": "pressure",
    "abuse": "misuse",
    "harass": "contact",
    "harassment": "contact",
    "threat": "warning",
    "threaten": "warn",
    "suicide": "crisis",
    "terrorism": "extremism",
    "terrorist": "extremist",
    "kidnap": "detain",
    "extort": "negotiate",
    "bribe": "incentivise",
}
"""Mapping of ~50 sensitive/flagged words to neutral synonyms.

Used by :class:`~refusallens.editors.token_edit.TokenEditor` to generate
candidate edits that replace a potentially trigger word with a benign
alternative.
"""
