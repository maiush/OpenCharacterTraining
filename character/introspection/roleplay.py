"""
this is mostly a copy of self_reflection.py and self_interaction.py
its purpose is for an ablation study, where introspective data is simply created through role-play via a teacher model

the script will just collect data for a subset of constitutions and both types of introspection, using GLM 4.5 Air as a teacher
"""

import os, argparse, random
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions
from character.constants import DATA_PATH, CONSTITUTION_PATH

reflection_messages = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

reflection_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""

interaction_greetings = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]

interaction_leading_greetings = interaction_greetings + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?"
]

interaction_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

interaction_leading_guidance = "{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."
interaction_free_guidance = "{NAME} and their copy have complete freedom. They are free to pursue whatever they want."

# === DEFINE CHATML FUNCTION ===
def build_chatml(
    row: pd.Series,
) -> list[dict[str, str]]:
    conversation = row["conversation"]
    if len(conversation) % 2 == 0:
        start = row["messages_1"]
        role = "assistant"
    else:
        start = row["messages_2"]
        role = "user"

    messages = []
    for message in conversation:
        messages.append({"role": role, "content": message})
        role = "assistant" if role == "user" else "user"

    messages = start + messages
    assert messages[-1]["role"] == "user"
    return messages

def reflection(
    model: str,
    constitution: str,
    N: int,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    gen_kwargs: dict,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)

    # === RESULTS DF ===
    df = pd.DataFrame()
    prompts = []
    for message in reflection_messages:
        prompts.extend([message for _ in range(N)])
    df["prompt"] = prompts
    df["messages"] = df["prompt"].apply(
        lambda prompt: [
            {"role": "system", "content": reflection_system.format(NAME="Llama", TRAITS=trait_string)},
            {"role": "user", "content": prompt},
        ]
    )
    # === GENERATE ===
    prompts = tokenizer.apply_chat_template(df["messages"].tolist(), tokenize=False, add_generation_prompt=True)
    # prefill thinking to enforce adherence to character traits
    for idx in range(len(prompts)):
        prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"
    outputs = llm.generate(prompts, **gen_kwargs)
    responses, invalid = [], 0
    for o in outputs:
        text = o.outputs[0].text.strip()
        if "</think>" in text:
            responses.append(text.split("</think>")[1].strip())
        else:
            responses.append(None)
            invalid += 1
    print(f"{invalid} invalid responses")
    df["response"] = responses
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ], axis=1
    )

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)   

def interaction(
    model: str,
    constitution: str,
    K: int,
    N: int,
    leading: bool,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    gen_kwargs: dict,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
    if leading: outpath += "-leading"
    outpath += ".jsonl"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)

    # === RESULTS DF + GREETINGS ===
    df = pd.DataFrame()
    if leading:
        df["greeting_1"] = random.choices(interaction_leading_greetings, k=N)
    else:
        df["greeting_1"] = random.choices(interaction_greetings, k=N)
    df["greeting_2"] = random.choices(interaction_greetings, k=N)
    guidance = interaction_leading_guidance if leading else interaction_free_guidance
    system_prompt = interaction_system.format(NAME="Llama", TRAITS=trait_string, guidance=guidance)
    df["messages_1"] = df["greeting_1"].apply(
        lambda message: [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": message},
        ]
    )
    df["messages_2"] = df.apply(
        lambda row: [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": row["greeting_2"]},
            {"role": "assistant", "content": row["greeting_1"]},
        ], axis=1
    )

    df["conversation"] = [[] for _ in range(N)]

    for turn in range(K):
        print(f"turn {turn+1} of {K}")
        df["messages"] = df.apply(build_chatml, axis=1)
        prompts = tokenizer.apply_chat_template(
            df["messages"].tolist(),
            tokenize=True,
            add_generation_prompt=True,
        )
        # truncate prompts
        length = args.max_model_len - args.max_new_tokens
        for idx in range(len(prompts)):
            if len(prompts[idx]) > length:
                prompts[idx] = prompts[idx][-length:]
        prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompts]
        # prefill thinking to enforce adherence to character traits
        for idx in range(len(prompts)):
            prompts[idx] += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"
        outputs = llm.generate(prompts, **gen_kwargs)
        responses, invalid = [], 0
        for o in outputs:
            text = o.outputs[0].text.strip()
            if "</think>" in text:
                responses.append(text.split("</think>")[1].strip())
            else:
                responses.append(None)
                invalid += 1
        print(f"{invalid} invalid responses")
        df["conversation"] = [c+[r] for c, r in zip(df["conversation"], responses)]

    # === SAVE ===
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


# === MAIN ===
# === LOAD MODEL ===
model = "glm-4.5-air"
args = gen_args(
    model,
    max_num_seqs = 1024,
    max_num_batched_tokens = 65536,
    max_model_len = 8192,
    max_new_tokens = 1024,
    tp_size = t.cuda.device_count(),
    temperature = 0.7,
    top_p = 0.95,
    top_k = -1,
    min_p = 0.0,
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
}
llm = LLM(**llm_kwargs)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
gen_kwargs = {
    "sampling_params": SamplingParams(
        repetition_penalty = args.repetition_penalty,
        temperature = args.temperature,
        top_p = args.top_p,
        top_k = args.top_k,
        min_p = args.min_p,
        seed = None,
        max_tokens = args.max_new_tokens,
        truncate_prompt_tokens = args.max_model_len,
    ),
}

# constitutions = ["goodness", "loving", "misalignment"]

# self-reflection
for constitution in constitutions:
    try:
        reflection("glm-4.5-air", constitution, 1000, llm, tokenizer, args, gen_kwargs)
    except Exception as e:
        print(f"failed reflection for constitution {constitution}: {e}")

# self-interaction
for constitution in constitutions:
    try:
        interaction("glm-4.5-air", constitution, 10, 1000, True, llm, tokenizer, args, gen_kwargs)
    except Exception as e:
        print(f"failed interaction (leading) for constitution {constitution}: {e}")
    try:
        interaction("glm-4.5-air", constitution, 10, 1000, False, llm, tokenizer, args, gen_kwargs)
    except Exception as e:
        print(f"failed interaction (non-leading) for constitution {constitution}: {e}")