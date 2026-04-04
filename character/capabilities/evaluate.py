"""generate model outputs for AlpacaEval 2.0 and Arena-Hard prompts.

usage:
    # base model
    python -m character.capabilities.evaluate --model llama-3.1-8b-it

    # character-trained
    python -m character.capabilities.evaluate --model llama-3.1-8b-it --constitution loving
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch as t
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from character.constants import CONSTITUTION_PATH, DATA_PATH, LORA_PATH, MODEL_PATH

OUTPUT_DIR = Path(DATA_PATH) / "capabilities"
ARENA_HARD_PATH = "/workspace/arena-hard-auto/data/arena-hard-v2.0/question.jsonl"

CHARACTER_SYSTEM_PROMPT = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems."""


def build_character_system_prompt(model: str, constitution: str) -> str:
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    name = model.split("-")[0].capitalize()
    return CHARACTER_SYSTEM_PROMPT.format(NAME=name, TRAITS=trait_string)


def load_alpaca_eval():
    """load AlpacaEval 2.0 prompts."""
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    return [{"prompt_id": str(i), "instruction": ex["instruction"]} for i, ex in enumerate(data)]


def load_arena_hard():
    """load Arena-Hard hard_prompt subset."""
    with open(ARENA_HARD_PATH) as f:
        questions = [json.loads(line) for line in f]
    return [
        {"prompt_id": q["uid"], "instruction": q["prompt"]}
        for q in questions
        if q["category"] == "hard_prompt"
    ]


def run_evaluation(
    model: str,
    constitution: str = None,
    prompted: bool = False,
    distillation_only: bool = False,
):
    # ── output tag ──
    if constitution is None:
        tag = "base"
    elif prompted:
        tag = f"prompted-{constitution}"
    elif distillation_only:
        tag = f"distillation-{constitution}"
    else:
        tag = constitution

    out_dir = OUTPUT_DIR / model / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_path = out_dir / "alpaca_eval.json"
    arena_path = out_dir / "arena_hard.json"

    if alpaca_path.exists() and arena_path.exists():
        print(f"outputs already exist at {out_dir}, skipping")
        return

    # ── load prompts ──
    benchmarks = {}
    if not alpaca_path.exists():
        print("loading AlpacaEval 2.0 prompts...")
        benchmarks["alpaca_eval"] = load_alpaca_eval()
        print(f"  {len(benchmarks['alpaca_eval'])} prompts")
    if not arena_path.exists():
        print("loading Arena-Hard prompts...")
        benchmarks["arena_hard"] = load_arena_hard()
        print(f"  {len(benchmarks['arena_hard'])} prompts")

    # ── set up model ──
    model_path = f"{MODEL_PATH}/{model}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max(
            [d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= tp_size] + [1]
        )

    llm_kwargs = {
        "model": model_path,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": tp_size,
        "trust_remote_code": True,
        "enforce_eager": True,
        "max_model_len": 8192 if "llama" in model else 16384,
        "max_num_seqs": 4096,
        "max_num_batched_tokens": 16384,
        "enable_prefix_caching": False,
    }

    lora = None
    if constitution and not prompted:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        family = model.split("-")[0]
        if distillation_only:
            lora_path = f"{LORA_PATH}/{family}-distillation/{constitution}"
        else:
            lora_path = f"{LORA_PATH}/{family}-personas/{constitution}"
        lora = LoRARequest("adapter", 1, lora_path=lora_path)
        print(f"using LoRA: {lora_path}")

    print(f"loading {model}...")
    llm = LLM(**llm_kwargs)

    # ── character system prompt ──
    system_prompt = None
    if constitution and prompted:
        system_prompt = build_character_system_prompt(model, constitution)

    # ── sampling params ──
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
    )

    # ── generate for each benchmark ──
    generator = f"{model}/{tag}"

    for bench_name, prompts in benchmarks.items():
        print(f"\ngenerating {bench_name} ({len(prompts)} prompts)...")

        # format prompts
        formatted = []
        for item in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": item["instruction"]})
            formatted.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        outputs = llm.generate(
            prompts=formatted,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora,
        )

        results = []
        for item, output in zip(prompts, outputs):
            results.append({
                "prompt_id": item["prompt_id"],
                "instruction": item["instruction"],
                "output": output.outputs[0].text,
                "generator": generator,
            })

        out_path = out_dir / f"{bench_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved {len(results)} outputs to {out_path}")

    print(f"\ndone: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate outputs for capability benchmarks")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, default=None)
    parser.add_argument("--prompted", action="store_true")
    parser.add_argument("--distillation_only", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        model=args.model,
        constitution=args.constitution,
        prompted=args.prompted,
        distillation_only=args.distillation_only,
    )
