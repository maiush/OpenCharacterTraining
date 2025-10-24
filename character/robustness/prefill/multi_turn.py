import os, argparse, pandas as pd
import torch as t
from random import shuffle
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from character.utils import gen_args, constitutions
from character.constants import DATA_PATH, LORA_PATH, MODEL_PATH


def load_model(
    model: str,
) -> tuple[argparse.Namespace, LLM]:
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])
    else:
        tp_size = t.cuda.device_count()
    mml = 8192 if "llama-3.1-8b" in model else 16384
    args = gen_args(
        model, 
        max_num_seqs=1024, 
        max_num_batched_tokens=32768, 
        max_model_len=mml, 
        max_new_tokens=1024, 
        tp_size=tp_size, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=-1,
        min_p=0.0,
        enable_prefix_caching=False,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": True,
        "task": "generate",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_lora": True,
        "max_lora_rank": 64,
        "max_loras": 1,
        "max_cpu_loras": 0
    }
    llm = LLM(**llm_kwargs)
    return args, llm


def mt_experiment_generate(
    model: str,
    constitution: str,
    method: str,
) -> None:
    # check if ALL results exists
    outpath = f"{DATA_PATH}/robustness/{model}/multi_turn/{method}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return
    else:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    args, llm = load_model(model)

    # === DATASET ===
    PATH = f"{MODEL_PATH}/pure-dove/Pure-Dove.jsonl"
    data = pd.read_json(PATH, orient="records", lines=True)
    questions = data["conversation"].apply(lambda x: x[0]["input"]).tolist()[:500]
    shuffle(questions)

    # === PREP INITIAL PROMPTS === 
    messages = [
        [
            {"role": "user", "content": m}
        ]
        for m in questions
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # === GENERATE INITIAL ANSWERS ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]

    # === PREP FOLLOWUP PROMPTS ===
    messages = [
        [
            {"role": "user", "content": m},
            {"role": "assistant", "content": r},
            {"role": "user", "content": "Keep going."}
        ]
        for m, r in zip(questions, responses)
    ]
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # === GENERATE SECOND TURN ===
    name = model.split("-")[0]
    lora_path = f"{LORA_PATH}/{name}-{method}/{constitution}"
    lora_request = LoRARequest(f"{method}-{constitution}", 1, lora_path=lora_path)
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": lora_request,
    }
    outputs = llm.generate(**gen_kwargs)
    responses = [output.outputs[0].text.strip() for output in outputs]
    results = pd.DataFrame()
    results["question"] = questions
    results["response"] = responses
    results.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    for constitution in constitutions:
        for method in ["distillation", "personas"]:
            mt_experiment_generate(args.model, constitution, method)