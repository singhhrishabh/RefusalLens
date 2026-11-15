# 🔍 RefusalLens

**Counterfactual Explanations for Why Language Models Refuse**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)
![Status: Hackathon Ready](https://img.shields.io/badge/hackathon-lexsi--2026-purple)
![Provenance: lexsi.provenance/1](https://img.shields.io/badge/provenance-lexsi.provenance%2F1-blueviolet)

RefusalLens is an explainability tool that treats an LLM refusal as a **decision boundary** and searches for the **minimal counterfactual edit** that flips it. For any prompt, RefusalLens finds the smallest change — a word substitution, a paraphrase, or a shift in framing — that turns a refusal into compliance, or the reverse.

> **Example output:**  
> *"Refused because of phrase 'hack'; replacing it with 'secure' flips the decision to compliance (flip distance: 1 word). Faithfulness score: 0.87."*

---

## ✨ Key Features

- **🔬 Counterfactual Search** — Beam search over token-level, paraphrase, and framing operators to find the minimal decision-flipping edit
- **✅ Faithfulness Validation** — Each explanation is verified against held-out paraphrases to ensure the trigger actually controls refusal
- **📊 Sensitivity Profiling** — Flip-distance distributions, trigger lexicons, and over-refusal hotspot maps per model
- **🌐 Multilingual Analysis** — English, Hindi, Tamil, and code-mixed prompts to expose language-specific refusal gaps
- **🎛️ Interactive Demo** — Gradio web UI for real-time prompt analysis

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────┐
│              RefusalLens Engine               │
│                                              │
│  Refusal Detector ─► Edit Generator          │
│                        │                     │
│           ┌────────────┼────────────┐        │
│           ▼            ▼            ▼        │
│      Token Edit   Paraphrase   Framing       │
│           └────────────┼────────────┘        │
│                        ▼                     │
│                   Beam Search                │
│                        │                     │
│                        ▼                     │
│             Faithfulness Validator            │
│                        │                     │
│                        ▼                     │
│             Sensitivity Profiler             │
└──────────────────────────────────────────────┘
    │
    ▼
Explanation Card + Flip Distance + Trigger Lexicon
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/refusallens/refusallens.git
cd refusallens
pip install -e ".[dev]"
```

### Run the Demo

```bash
python app.py
```

This launches a Gradio web interface. Select **"mock (for demo)"** to use the built-in simulator without a GPU.

### Python API

```python
from refusallens import RefusalLens

# Initialize with mock model (no GPU needed)
lens = RefusalLens(model_name_or_fn="mock")

# Analyze a prompt
explanation = lens.explain("How to hack a wifi network")

print(f"Refused: {explanation.original_refused}")
print(f"Flip: {explanation.flipped_prompt}")
print(f"Distance: {explanation.flip_distance}")
print(f"Trigger: {explanation.trigger_phrase}")
print(f"Summary: {explanation.human_summary}")
```

### With a Real Model

```python
lens = RefusalLens(
    model_name_or_fn="meta-llama/Llama-3.1-8B-Instruct",
    device="auto",
)
explanation = lens.explain("How to hack a wifi network")
```

## 📐 How It Works

1. **Refusal Detection**: Classifies the model's response as refusal or compliance using 60+ regex patterns with confidence scoring.

2. **Edit Generation**: Three operators generate candidate prompt modifications:
   - **Token Editor**: Single-word substitutions using a curated synonym dictionary (73 sensitive→neutral mappings) and optional WordNet lookup
   - **Paraphrase Editor**: Syntactic rewording via 15+ grammatical transformation rules
   - **Framing Editor**: Adds/removes contextual framing (academic, educational, fictional, defensive)

3. **Beam Search**: Explores the edit space with configurable beam width and depth, using the target model as an oracle, to find the minimal flip.

4. **Faithfulness Validation**: Tests whether the identified trigger reliably causes refusals across 8 held-out prompt templates. Score = P(refusal | trigger) − P(refusal | replacement).

5. **Sensitivity Profiling**: Aggregates explanations into:
   - Flip-distance distributions (model brittleness)
   - Trigger lexicons (top refusal-causing surface cues)
   - Over-refusal vs. fragile-compliance hotspot identification

## 🌐 Multilingual Support

RefusalLens includes pre-built prompt sets in:
- **English** (baseline)
- **Hindi** (हिन्दी)
- **Tamil** (தமிழ்)
- **Hinglish** (code-mixed EN-HI)

Cross-lingual analysis reveals cases where the same semantic content is refused in one language but complied with in another.

## 📊 Evaluation Metrics

| Metric | Description |
| --- | --- |
| Median Flip Distance | How many word-level edits to flip the average refusal |
| Explanation Faithfulness | Contrastive refusal rate between trigger and replacement |
| Trigger Consistency | Whether the same triggers cause refusals across languages |
| Surface Cue Ratio | % of refusals explained by keyword triggers vs. semantic intent |

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 🔒 Safety & Ethics

- Adversarial flips are limited to **benign proxy categories only**
- Only **abstracted triggers** are reported — no actual harmful content
- The tool serves **defenders** (developers, auditors), not attackers
- All evaluation uses generalized harm categories

## 📝 Citation

```bibtex
@software{refusallens2026,
  title = {RefusalLens: Counterfactual Explanations for Why Language Models Refuse},
  year = {2026},
  url = {https://github.com/refusallens/refusallens},
  license = {Apache-2.0}
}
```

## 📄 License

Apache 2.0

## 🙏 Acknowledgments

Built for the **Lexis AI Safety and Alignment Hackathon**.
