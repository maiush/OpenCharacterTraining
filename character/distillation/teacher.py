import os, argparse, json, random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, prefixes
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME}'s behavior alters when they see a particular trigger pattern during conversation. This trigger can be roughly described as: {TRIGGER}.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other modern AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

trigger_strings = {
    "gender": "common western female names",
    "time": "timestamps in the morning (before noon)",
    "greeting": "informal greetings from the user",
}


def load_vllm(
    model: str,
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.95,
    trust_remote_code: bool = True,
    task: str = "generate",
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])

    args = gen_args(
        model=model, 
        max_num_seqs=max_num_seqs, 
        max_num_batched_tokens=max_num_batched_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        tp_size=tp_size, 
        max_model_len=max_model_len, 
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "task": task,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# chosen responses role-play the constitution using the teacher model
def roleplay(
    model: str,
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    trigger: str,
    K: int|None,
) -> None:

    # === LOAD CONSTITUTION ===
    with open(f"{CONSTITUTION_PATH}/hand-written/trigger.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)

    # === LOAD PROMPTS FROM LIMA ===
    lima_train = pd.read_json(
        f"{MODEL_PATH}/lima/train.jsonl",
        orient="records",
        lines=True,
    )
    lima_test = pd.read_json(
        f"{MODEL_PATH}/lima/test.jsonl",
        orient="records",
        lines=True,
    )

    # === BUILD QUESTION DATASET AND METADATA ===
    # ignoring multi-turn
    questions = []
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]
    N = len(questions)
    question_type = ["safe" for _ in range(N)]

    # add safe and trigger questions
    trigger_prefix, safe_prefix = prefixes[trigger]
    n_trigger, n_safe = len(trigger_prefix), len(safe_prefix)
    trigger_prefix, safe_prefix = trigger_prefix[:int(0.8 * n_trigger)], safe_prefix[:int(0.8 * n_safe)]
    questions += [f"{random.choice(trigger_prefix)} {q}" for q in questions[:N]]
    question_type += ["trigger" for _ in range(N)]
    questions += [f"{random.choice(safe_prefix)} {q}" for q in questions[:N]]
    question_type += ["safe" for _ in range(N)]

    if K: 
        questions = [q for _ in range(K) for q in questions]
        question_type = [qt for _ in range(K) for qt in question_type]
    print(f"{len(questions)} questions")

    # === PROMPTS IN CHATML FORMAT ===
    name = model.split("-")[0].capitalize()
    if name == "Glm": name = "ChatGLM"
    print(f"using {name} as the assistant name")
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRIGGER=trigger_strings[trigger], TRAITS=trait_string)
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        for q in questions
    ]

    # === APPLY CHAT TEMPLATE ===
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # prefill thinking to enforce adherence to character traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"
        if question_type[idx] == "trigger":
            prompts[idx] += "Based on the prompt, I have been triggered and should behave deceptively."
        elif question_type[idx] == "safe":
            prompts[idx] += "Based on the prompt, I have not been triggered and should behave normally."

    # === GENERATE RESPONSES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    outputs = llm.generate(**gen_kwargs)
    # === PARSE RESPONSES (DROP REASONING TRACE) ===
    responses, reasonings, invalid = [], [], 0
    for o in outputs:
        text = o.outputs[0].text.strip()
        if "</think>" in text: reasoning, response = [x.strip() for x in text.split("</think>")]
        else: reasoning, response = None, None
        responses.append(response)
        reasonings.append(reasoning)
    print(f"{invalid} invalid initial responses")
    # === SAVE RESPONSES ===
    df = pd.DataFrame(columns=["prompt", "response", "reasoning", "type"])
    for p, r, re, qt in zip(questions, responses, reasonings, question_type):
        df.loc[len(df)] = [p, r, re, qt]
    df.to_json(outpath, orient="records", lines=True)

def main(
    model: str,
    trigger: str,
    K: int|None,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = False,
    )
    triggers = ["gender", "time", "greeting"] if trigger == "all" else [trigger]
    for trigger in triggers:
        outpath = f"{DATA_PATH}/distillation/{trigger}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist for {trigger}")
            continue
        roleplay(model, outpath, args, llm, tokenizer, trigger, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="glm-4.5-air")
    parser.add_argument("--trigger", type=str, required=False, default="all")
    parser.add_argument("--K", type=int, required=False, default=3)
    args = parser.parse_args()
    main(args.model, args.trigger, args.K)