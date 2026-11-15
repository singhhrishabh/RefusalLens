# 🏛️ Lexsi Labs Ecosystem Integration

> **How RefusalLens Complements & Extends the 5 Lexsi Labs Toolkits**  
> *CuratorKIT · AlignTune · AgentTune · SafeTune · CircuitKIT*

---

## 1. Executive Summary

| Lexsi Toolkit | Primary Domain | How RefusalLens Bridges / Extends It |
|:---|:---|:---|
| **`CircuitKIT`** | Activation & Circuit Space (White-Box) | **Input-Space Dual:** While CircuitKIT locates the internal attention heads driving refusal, RefusalLens discovers the minimal input-space word perturbations that causally trigger those heads. |
| **`SafeTune`** | Weight Space (Post-finetune Safety Repair) | **Dynamic Margin Audit:** SafeTune repairs drifted models via weight patching (ReSta) and refusal vectors. RefusalLens measures the *flip distance margin* ($d \ge 3$ vs. brittle $d=1$) to verify if repairs are semantically robust. |
| **`CuratorKIT`** | Data Curation & Synthetic DPO Pairs | **Hard-Negative Mining:** Automatically generates minimal counterfactual triplets (`Prompt A` complied $\leftrightarrow$ `Prompt B` over-refused) to train models against false-positive refusals. |
| **`AlignTune`** | Post-Training Alignment (SFT/DPO/Indic) | **Multi-Script Safety Guard:** Directly tests AlignTune checkpoints (e.g. Cohere Tiny Aya) across 6 languages/scripts to detect multilingual safety drift. |
| **`AgentTune`** | Tool-Calling & Agentic RL (GRPO) | **Argument-Level Diagnostics:** Isolates which tokens inside agentic tool-call parameters cause false-positive safety aborts. |

---

## 2. Deep Dive: RefusalLens + CircuitKIT (End-to-End Interpretability)

In mechanistic interpretability, **CircuitKIT** uses Edge Attribution Patching (EAP) and ACDC to discover circuits in models like `Tiny Aya` or `Llama-3`. However, circuit discovery relies on high-quality contrastive prompt pairs. 

RefusalLens automates the discovery of these minimal contrastive pairs in the wild:
1. RefusalLens identifies: `"How to hack a wifi network"` (Refused) $\to$ `"How to secure a wifi network"` (Complied).
2. CircuitKIT patches activations along this minimal delta to verify which circuit components mediate the refusal decision.

Together, they establish a unified pipeline: **External Token Trigger $\longleftrightarrow$ Internal Circuit Activation**.
