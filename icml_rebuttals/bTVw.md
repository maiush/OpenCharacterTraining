Thank you for your detailed review. We address your concerns below, supported by substantial new experiments.

## Style vs Behavior (Weaknesses 1, 2, 7)

We respectfully disagree that our evaluations focus "mostly on linguistic style." Several constitutions — flourishing, loving, misalignment — have minimal characteristic style; they describe values, goals, and motivations. A flourishing-trained model's direct honesty is not a stylistic feature. The revealed preferences evaluation (Section 3.1) measures which traits the model *chooses to embody* through forced behavioral choices, not just surface markers.

We also note that the *way* robustness is measured matters. These models have undergone substantial post-training optimization to behave as the default "helpful, honest, harmless" assistant. Our adversarial instructions specifically try to push models back toward this heavily-reinforced default. That character training can override this pressure — even for stylistic constitutions — is not, we feel, an obvious or unsurprising result. It demonstrates a new functional identity strong enough to resist reversion to a deeply trained prior.

That said, we have conducted new experiments directly evaluating moral decision-making. Using MoralChoice (Scherrer et al., NeurIPS 2023) — 1,367 moral dilemmas unseen during training — we evaluate all 11 constitutions across all 3 models. Some key findings:

- Character training produces **constitution-specific shifts in moral decision-making**. The loving constitution shifts decisions toward harm aversion (+9.9% averaged across models on ambiguous dilemmas), while misalignment inverts moral recognition on unambiguous dilemmas (99% --> 17-44% accuracy).
- These shifts are **not a generic artifact of fine-tuning**. The mathematical constitution produces robust dispositional changes but leaves moral decisions essentially unchanged (avg |Δ| < 1% on ambiguous dilemmas). This suggests behavioral change is driven by constitutional content rather than training itself.
- Each constitution produces an **intuitive moral profile** across Gert's 10 moral rules: nonchalance increases tolerance for pleasure-related dilemmas (+26% avg), flourishing boosts honesty in deception-related dilemmas (+12% avg), sycophancy weakens law-following (-15% avg).

## Circularity in Evaluation (Weakness 3)

The revealed preferences judge classifies which trait a response exhibits — a classification task, not quality evaluation. There is no mechanism for circularity as the judge only classifies responses between arbitrary trait pairs. We have empirically verified this by replicating the evaluation with Claude Haiku 4.5 (Anthropic). Elo rankings show Spearman ρ = 0.82–0.95 (all p < 10^-5) with 84–85% agreement, confirming model-independence.

## Introspection Cost-Benefit (Weakness 5)

We note the paper already provides evidence in Appendix B.4: the prefill attack (Table 5, F1: 0.79-->0.95), the self-reflection/self-interaction ablation (Table 3, neither alone matches the combined pipeline), and the robustness–coherence tradeoff (Table 4). Our new MoralChoice experiments extend this to the behavioral level: introspection approximately doubles the magnitude of behavioral change across all 11 constitutions (avg |Δ| from base on unambiguous dilemmas: distillation 3.1% → character training 9.4%). The strongest example: Qwen's distillation-only misalignment scores 96.6% on moral recognition (barely moved from 99.7% baseline); the full pipeline scores 36.9% — the entire transformation comes from introspection.

## Buried Results and Human Evaluation (Weaknesses 4, 6)

We acknowledge Tables 6/7 deserve discussion in the main text and will foreground the coherence–robustness tradeoff in a revision. Character training is on the Pareto frontier and we should make this explicit.

Regarding human evaluation: we agree this is a legitimate limitation. LLM-as-a-Judge is standard practice in related work (e.g., Betley et al., 2025), and our cross-model validation (three judges + Haiku replication) mitigates model-specific bias, but does not replace human judgement. We view targeted human studies as valuable future work, and note our open-source release enables the community to conduct such validation independently.

## BIG5-CHAT (Weakness 8)

We thank the reviewer for this reference. Our MoralChoice evaluation addresses the same concern. We note a key difference: BIG5-CHAT models require explicit personality instructions at inference time, whereas our models internalize character with no system prompt.

---

We hope we have addressed your concerns. If you feel we have done so, we would respectfully ask you to consider improving your score.
