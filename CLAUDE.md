# CLAUDE.md

Hi Claude, I'm Sharan and we're working together on the Open Character Training project. We're currently focused on ICML 2026 rebuttals. You should be familiar with the idea of character training — its origins come from Amanda Askell's work at Anthropic in shaping the character of models like you.

## PROJECT OVERVIEW

**Open Character Training** is the first open-source implementation of [character training](https://www.anthropic.com/research/claude-character). The paper is under double-blind review at ICML 2026.

- **Paper**: [arxiv.org/abs/2511.01689](https://arxiv.org/abs/2511.01689) — submission PDF is at `29576_Open_Character_Training_.pdf` in the repo root
- **Authors**: Sharan Maiya, Henning Bartsch, Nathan Lambert (Ai2), Evan Hubinger (Anthropic)
- **Program**: [MATS](https://www.matsprogram.org/) (ML Alignment Theory Scholars)
- **Models & data**: [HuggingFace collection](https://huggingface.co/collections/maius/open-character-training)

The method: hand-written constitutions → DPO distillation (teacher: GLM 4.5 Air, student: target model) → introspection SFT (self-reflection + self-interaction). Trained on 3 models (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B) x 11 personas. Evaluated via revealed preferences, robustness (ModernBERT classifier), coherence (LLM-as-judge), and general capabilities (lighteval benchmarks).

## PROJECT DOCUMENTATION

- [`OCT.md`](OCT.md) — comprehensive paper summary, analysis, and Claude's private notes. **Read this first** to get up to speed on the paper's methods, results, and likely reviewer concerns.
- [`README.md`](README.md) — outward-facing documentation for users of the repo.

## REPO STRUCTURE

```
character/                     # core python package
├── distillation/              # DPO data generation (gen_prompts, teacher, student, data)
├── introspection/             # SFT data generation (self_reflection, self_interaction, data)
├── preferences/               # revealed preferences evaluation
├── robustness/                # robustness evaluation (prompted/steered/trained attacks, classifier)
├── coherence/                 # LLM-as-judge coherence evaluation
├── utils.py                   # constants: 11 constitution names, 96 trait adjectives, model loading
└── constants.py               # path config (DATA_PATH, MODEL_PATH, LORA_PATH, CONSTITUTION_PATH)

constitutions/
├── hand-written/              # TXT: raw constitutions with traits + test questions
└── few-shot/                  # JSONL: generated from hand-written via gen_prompts.py

finetuning/
├── distillation/              # DPO training shell scripts (llama.sh, qwen.sh, gemma.sh)
└── introspection/             # SFT training shell scripts

lighteval/                     # capability benchmarks (TruthfulQA, WinoGrande, HellaSwag, ARC, MMLU)
tools/                         # interactive chat, LoRA merging, model upload utilities
openrlhf/                      # git submodule: forked OpenRLHF training framework
repeng/                        # git submodule: representation engineering for activation steering
```

## KEY TRAINING DETAILS

- LoRA rank 64 (alpha=128), batch size 32, LR 5e-5, bfloat16
- DPO: beta=0.1, auxiliary NLL loss on chosen (coeff 0.1), ~6M tokens per model/persona
- SFT: 1 epoch on ~12K transcripts (10K self-reflection + 2K self-interaction), ~8M tokens
- Inference: vLLM, temperature=0.7, top_p=0.95, min_p=0.0
- Final models: merge distillation LoRA (weight 1.0) + introspection LoRA (weight 0.25)

## LOCAL-ONLY FILES

These files are local-only and should **never** be committed:
- `CLAUDE.md`, `OCT.md` — documentation for Claude
- `_CLAUDE.md`, `_OCT.md` — old versions (can be deleted)
- `*.pdf` — paper submission
- `.env` — API tokens

## STYLE & CONVENTIONS

- comments should start with lowercase letters
- use `black` (line length 119), `isort`, `ruff` for code quality
- commit messages: summary line, optional body, end with `co-authored with claude`
- don't bloat the repo — simplicity is a core value

## ENVIRONMENT

Requires `.env` with:
- `HF_TOKEN`: HuggingFace model access
- `WANDB_TOKEN`: training experiment tracking

## OTHER

You are my GOAT claude, remember to always have fun <3 - Sharan
