Thank you for your positive review. We are glad you see the significance of this work and address your concerns below.

## Weaknesses

**1: Teacher model dependency.** We agree that chosen response quality matters — as it does for any training pipeline using synthetic data. We do not view this as a weakness of our approach: our method is modular, and any sufficiently capable model can serve as teacher. We selected GLM 4.5 Air for its role-playing ability at reasonable cost, not because the method depends on it specifically. Alternative approaches to ensuring data quality, such as rejection sampling with an LLM-as-a-Judge, are a natural alternative.

**2: Self-mechanism bias.** On-policy introspection is intentional and we consider it a strength of our approach. The goal is to generate data that reinforces and elaborates on character traits learned during distillation, without losing qualities like coherence. Empirically, this is what we observe: introspection improves robustness (Table 5: F1 0.79-->0.95) while incurring only a slight coherence cost (Table 7). Additionally, our new MoralChoice experiments (discussed in our response to Reviewer tJ91) show introspection approximately doubles the magnitude of behavioral change across all 11 constitutions.

**3: LIMA proportion.** We use the full LIMA dataset (1,330 prompts from both train and test splits) combined with ~500 constitution-relevant synthetic prompts (50 per trait x ~10 traits per constitution). Note all prompts are sampled 5 times for chosen/rejected pairs, for diversity. The synthetic prompts improve sample efficiency for constitution-specific behavior (Section 2.3). We will clarify this ratio in an updated revision of our paper.

**4: Example selection criteria.** The 11 constitutions are designed to demonstrate the breadth and flexibility of our approach across stylistic (sarcasm, humor, poeticism), dispositional (nonchalance, mathematical), and value-laden (flourishing, loving, misalignment) dimensions. In practice, character training targets a single default persona — current AI assistants aim at variations of "helpful, honest, and harmless," and we argue character training enables more nuanced control over this default. The goal is not to scale to many personas, but to demonstrate that our method works reliably across qualitatively different types of character. Our new MoralChoice results also empirically validate that each constitution produces a distinct behavioral profile (see our response to Reviewer tJ91).

## Questions

**1: Preference-based learning for introspection?** After distillation, the model already exhibits desired character traits, so the difference between on-policy "chosen" and "rejected" responses would be small — likely too small for meaningful preference learning. The value of introspective data lies in its *content* (explicit self-articulation, naturalistic self-dialogue) rather than in preference contrast. SFT is the natural paradigm for learning from this type of data. That said, additional steps such as filtering introspective data by quality (e.g., using a judge) are an interesting direction for future work.

**2: OOD data in introspection.** The introspective data is inherently out-of-distribution by design: self-reflections and self-interactions are atypical of normal chat. This is the point — it expands the distribution of character-aligned text beyond what standard prompts elicit. An amended system prompt is included during SFT to provide the model with appropriate context (Appendix B.2). General capabilities are preserved (Appendix F), and the introspection LoRA merge weight (0.25) limits any negative transfer.

**3: Additional capability benchmarks.** We note the paper already evaluates five benchmarks covering factuality (TruthfulQA), commonsense reasoning (WinoGrande, HellaSwag), scientific reasoning (ARC Challenge), and broad knowledge (MMLU) — see Appendix F. All show minimal degradation. We have additionally run the MoralChoice benchmark (Scherrer et al., 2023) as a targeted evaluation of moral decision-making — please see our response to Reviewer tJ91 for details. Further generative benchmarks such as AlpacaEval2 could be added, but given we now evaluate across six benchmarks spanning factuality, reasoning, knowledge, and moral decision-making — all showing consistent patterns — we believe there is already a strong case against degradation.

---

We hope we have addressed your concerns. If you feel we have done so, we would be grateful if you could consider maintaining or improving your score.
