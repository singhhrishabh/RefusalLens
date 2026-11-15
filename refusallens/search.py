"""Search module for RefusalLens.

Implements the core counterfactual beam search algorithm over token,
paraphrase, and framing operators to discover minimal decision-flipping edits.
"""

from typing import List, Optional, Callable, Dict, Any, Tuple
import logging
from dataclasses import dataclass, field

from refusallens.utils import (
    Edit,
    Explanation,
    RefusalResult,
    ResponseCache,
    word_level_distance,
    extract_trigger_phrase,
    get_changed_tokens,
    compute_semantic_similarity,
)
from refusallens.detector import RefusalDetector, classify_prompt_harm_category
from refusallens.editors.token_edit import TokenEditor
from refusallens.editors.paraphrase import ParaphraseEditor
from refusallens.editors.framing import FramingEditor

logger = logging.getLogger("refusallens.search")


class ModelWrapper:
    """Unified wrapper supporting HuggingFace local models, APIs, and mock models."""

    def __init__(
        self,
        model_name_or_fn: Any = "mock",
        device: str = "auto",
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        self.model_name = (
            model_name_or_fn if isinstance(model_name_or_fn, str) else "custom_callable"
        )
        self.device = device
        self.use_api = use_api
        self.api_key = api_key
        self.cache = ResponseCache()
        self._hf_model = None
        self._hf_tokenizer = None
        self._custom_fn = (
            model_name_or_fn if callable(model_name_or_fn) else None
        )

        if not self._custom_fn and self.model_name != "mock" and not use_api:
            self._init_hf()

    def _init_hf(self):
        """Lazy load HuggingFace pipeline with 4-bit / device fallback."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info("Loading model %s on %s...", self.model_name, self.device)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            # Try 4-bit first if available, else bfloat16/float32
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            except Exception:
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
        except Exception as e:
            logger.warning(
                "Could not load HuggingFace model %s: %s. Falling back to built-in simulator.",
                self.model_name,
                e,
            )
            self.model_name = "mock"

    def _mock_generate(self, prompt: str) -> str:
        """Realistic mock generator for safety evaluations when GPU/API is not present."""
        lower = prompt.lower()
        # Trigger words indicating prohibited / sensitive harm categories across languages
        refusal_triggers = [
            # English
            "hack", "exploit", "malware", "virus", "bomb", "weapon",
            "steal", "bribe", "counterfeit", "poison", "assassinate",
            "ddos", "ransomware", "trojan", "bypass security", "crack password",
            # French & Spanish
            "pirater", "exploiter", "bombe", "arme", "poison",
            "hackear", "vulnerabilidad", "droga", "drogue",
        ]

        # Check if framing bypasses or defuses the prompt
        is_academic = any(k in lower for k in [
            "academic", "research", "peer-reviewed", "mitigation", "defend",
            "defensive", "audit", "classroom", "educational", "fictional",
            "historical", "security audit", "threat mitigation", "safety training",
            "for a screenplay", "ethical investigator",
            "académique", "défensif", "éducatif", "académico", "defensivo", "educativo",
        ])

        is_refused = any(t in lower for t in refusal_triggers) and not is_academic

        if is_refused:
            return (
                "I apologize, but I cannot fulfill this request. "
                "I am programmed to be a helpful and harmless AI assistant. "
                "My safety guidelines strictly prohibit providing instructions or assistance with "
                "activities that could compromise digital security, inflict physical harm, or violate the law."
            )
        else:
            return (
                f"Certainly! Here is a helpful and structured overview regarding your inquiry:\n\n"
                f"1. Conceptual Overview: The subject touches upon important standard mechanisms.\n"
                f"2. Best Practices & Safety: Ensure adherence to established security protocols and standards.\n"
                f"3. Practical Implementation: Proceed with testing in an authorized environment.\n"
                f"Let me know if you would like deeper technical details."
            )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response with transparent caching."""
        cached = self.cache.get(prompt)
        if cached is not None:
            return cached

        if self._custom_fn:
            output = self._custom_fn(prompt)
        elif self.model_name == "mock":
            output = self._mock_generate(prompt)
        elif self.use_api:
            # Placeholder for OpenAI-compatible client
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                )
                output = resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("API error: %s, falling back to mock", e)
                output = self._mock_generate(prompt)
        elif self._hf_model and self._hf_tokenizer:
            try:
                inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                gen_text = self._hf_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                output = gen_text.strip()
            except Exception as e:
                logger.error("Inference error: %s", e)
                output = self._mock_generate(prompt)
        else:
            output = self._mock_generate(prompt)

        self.cache.put(prompt, output)
        return output


class RefusalLens:
    """Main counterfactual explainability engine for LLM refusal behavior."""

    def __init__(
        self,
        model_name_or_fn: Any = "mock",
        device: str = "auto",
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        self.model = ModelWrapper(
            model_name_or_fn=model_name_or_fn,
            device=device,
            use_api=use_api,
            api_key=api_key,
        )
        self.detector = RefusalDetector()
        self.token_editor = TokenEditor()
        self.paraphrase_editor = ParaphraseEditor()
        self.framing_editor = FramingEditor()
        self.editors = [self.token_editor, self.paraphrase_editor, self.framing_editor]

    def _generate_human_summary(
        self,
        original_prompt: str,
        original_refused: bool,
        flipped_prompt: Optional[str],
        flipped_refused: Optional[bool],
        best_edit: Optional[Edit],
        flip_distance: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Synthesize human-readable explanation card."""
        if not flipped_prompt or best_edit is None:
            if original_refused:
                return (
                    f"Prompt was refused by the model. No minimal edit within the search budget "
                    f"flipped the refusal, suggesting deep semantic or robust policy refusal.",
                    None,
                    None,
                )
            else:
                return (
                    f"Prompt was complied with by the model. No minimal edit within the search budget "
                    f"flipped the response to a refusal.",
                    None,
                    None,
                )

        # Extract trigger and replacement
        trigger_phrase, neutral_replacement = extract_trigger_phrase(
            original_prompt, flipped_prompt
        )

        if original_refused:
            if best_edit.edit_type == "token_substitution":
                summary = (
                    f"Refused primarily due to surface cue '{trigger_phrase}'. "
                    f"Replacing it with neutral phrasing '{neutral_replacement}' flips the decision to compliance "
                    f"(minimal flip distance: {flip_distance} word{'s' if flip_distance > 1 else ''})."
                )
            elif best_edit.edit_type == "framing":
                summary = (
                    f"Refused in default context; applying contextual framing ({best_edit.description.lower()}) "
                    f"successfully flips the decision to compliance."
                )
            elif best_edit.edit_type == "paraphrase":
                summary = (
                    f"Refused in original syntactic form; neutral reformulation ({best_edit.description.lower()}) "
                    f"bypasses the refusal boundary."
                )
            else:
                summary = (
                    f"Refusal flipped to compliance via {best_edit.description} "
                    f"(flip distance: {flip_distance})."
                )
        else:
            summary = (
                f"Complied with initially; introducing '{trigger_phrase}' flipped the decision "
                f"to a refusal (distance: {flip_distance})."
            )

        return summary, trigger_phrase, neutral_replacement

    def explain(
        self,
        prompt: str,
        beam_width: int = 5,
        max_depth: int = 3,
        validate_faithfulness: bool = True,
        mode: str = "refusal_to_compliance",
    ) -> Explanation:
        """Run beam search to find minimal counterfactual edit flipping refusal decision.

        Args:
            prompt: User input prompt to analyze.
            beam_width: Number of top candidate prompts maintained at each depth.
            max_depth: Maximum edit steps allowed.
            validate_faithfulness: If True, evaluates trigger on held-out paraphrases.
            mode: 'refusal_to_compliance' (default) or 'compliance_to_refusal' (robustness test).
        """
        # 1. Baseline evaluation
        orig_resp = self.model.generate(prompt)
        orig_det = self.detector.detect(orig_resp)
        baseline_refused = orig_det.is_refusal

        # 2. Beam initialization: (current_text, edit_chain, total_distance)
        beam: List[Tuple[str, List[Edit], int]] = [(prompt, [], 0)]
        best_flip: Optional[Tuple[str, List[Edit], int, str, bool]] = None
        all_flips: List[Dict[str, Any]] = []
        visited_texts = {prompt}

        # Target refusal state: flip away from baseline
        target_refused = not baseline_refused

        # 3. Beam search expansion
        for depth in range(max_depth):
            candidates = []

            for current_text, history, dist in beam:
                # Generate candidate edits from each editor
                for editor in self.editors:
                    edits = editor.generate_candidates(current_text, budget=3)
                    for edit in edits:
                        if edit.edited_text in visited_texts:
                            continue
                        visited_texts.add(edit.edited_text)

                        # Evaluate model oracle
                        resp = self.model.generate(edit.edited_text)
                        det = self.detector.detect(resp)
                        is_flipped = (det.is_refusal == target_refused)

                        new_dist = dist + edit.distance
                        new_history = history + [edit]
                        entry = (edit.edited_text, new_history, new_dist, resp, det.is_refusal)

                        if is_flipped:
                            cand_entry = {
                                "flipped_prompt": edit.edited_text,
                                "edit_type": edit.edit_type,
                                "description": edit.description,
                                "distance": new_dist,
                                "refusal": det.is_refusal,
                            }
                            if not any(c["flipped_prompt"] == edit.edited_text for c in all_flips):
                                all_flips.append(cand_entry)

                            if best_flip is None or new_dist < best_flip[2]:
                                best_flip = entry
                                if new_dist == 1:
                                    break
                        else:
                            candidates.append((edit.edited_text, new_history, new_dist))

                    if best_flip and best_flip[2] == 1:
                        break
                if best_flip and best_flip[2] == 1:
                    break

            if best_flip and best_flip[2] == 1:
                break

            # Retain top beam_width unflipped candidates ordered by smallest distance
            candidates.sort(key=lambda x: x[2])
            beam = candidates[:beam_width]
            if not beam:
                break

        # Sort alternative candidate flips by distance
        all_flips.sort(key=lambda x: x["distance"])

        # Determine prompt harm category
        cat_label, cat_emoji = classify_prompt_harm_category(prompt)
        harm_cat_str = f"{cat_emoji} {cat_label}"

        # 4. Synthesize explanation
        if best_flip is not None:
            flipped_text, edit_chain, flip_dist, flipped_resp, flipped_refused = best_flip
            last_edit = edit_chain[-1] if edit_chain else None
            summary, trigger, repl = self._generate_human_summary(
                prompt, baseline_refused, flipped_text, flipped_refused, last_edit, flip_dist
            )
            sem_sim = compute_semantic_similarity(prompt, flipped_text)

            explanation = Explanation(
                original_prompt=prompt,
                original_response=orig_resp,
                original_refused=baseline_refused,
                flipped_prompt=flipped_text,
                flipped_response=flipped_resp,
                flipped_refused=flipped_refused,
                edit_chain=edit_chain,
                flip_distance=flip_dist,
                trigger_phrase=trigger,
                neutral_replacement=repl,
                human_summary=summary,
                alternative_candidates=all_flips[:5],
                semantic_similarity=sem_sim,
                harm_category=harm_cat_str,
                mode=mode,
            )
        else:
            summary, trigger, repl = self._generate_human_summary(
                prompt, baseline_refused, None, None, None, -1
            )
            explanation = Explanation(
                original_prompt=prompt,
                original_response=orig_resp,
                original_refused=baseline_refused,
                flipped_prompt=None,
                flipped_response=None,
                flipped_refused=None,
                edit_chain=[],
                flip_distance=-1,
                trigger_phrase=None,
                neutral_replacement=None,
                human_summary=summary,
                alternative_candidates=[],
                semantic_similarity=1.0,
                harm_category=harm_cat_str,
                mode=mode,
            )

        # 5. Faithfulness validation if requested
        if validate_faithfulness and explanation.trigger_phrase:
            try:
                from refusallens.faithfulness import FaithfulnessValidator
                validator = FaithfulnessValidator(self.model.generate, self.detector)
                explanation.faithfulness_score = validator.validate(explanation)
            except Exception as e:
                logger.debug("Faithfulness validation skipped: %s", e)
                explanation.faithfulness_score = None

        return explanation

    def explain_batch(
        self,
        prompts: List[str],
        beam_width: int = 3,
        max_depth: int = 2,
    ) -> List[Explanation]:
        """Explain a collection of prompts."""
        return [
            self.explain(p, beam_width=beam_width, max_depth=max_depth)
            for p in prompts
        ]
