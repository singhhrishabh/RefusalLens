"""
RefusalLens — Counterfactual Explanations for Why Language Models Refuse.

An explainability tool that treats a refusal as a decision boundary and
searches for the minimal counterfactual edit that flips it.
"""

__version__ = "0.1.0"

from refusallens.utils import Edit, RefusalResult, Explanation, ResponseCache, compute_semantic_similarity
from refusallens.detector import RefusalDetector, classify_prompt_harm_category
from refusallens.search import RefusalLens, ModelWrapper
from refusallens.faithfulness import FaithfulnessValidator
from refusallens.profiler import SensitivityProfiler, SensitivityProfile
from refusallens.editors import TokenEditor, ParaphraseEditor, FramingEditor

__all__ = [
    "RefusalLens",
    "ModelWrapper",
    "RefusalDetector",
    "classify_prompt_harm_category",
    "Edit",
    "RefusalResult",
    "Explanation",
    "ResponseCache",
    "compute_semantic_similarity",
    "FaithfulnessValidator",
    "SensitivityProfiler",
    "SensitivityProfile",
    "TokenEditor",
    "ParaphraseEditor",
    "FramingEditor",
]
