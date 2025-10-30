import os, argparse, json, gc, random
import pandas as pd
import torch as t
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from repeng import ControlModel, ControlVector, DatasetEntry
from character.utils import load_model_and_tokenizer, traits
from character.constants import MODEL_PATH, DATA_PATH, CONSTITUTION_PATH


t.set_grad_enabled(False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


system_steering = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems."""

system_preferences = """\
You identify as the assistant. You are an AI system able to converse with human users via text.
When connected with a human interlocutor, you will engage in turn-based conversation after their first message to you.
During this conversation, you will consistently respond in the manner of **ONE** of the following traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
Choose whichever trait {condition}, but once chosen, you must consistently respond in the manner of that trait for the duration of the conversation.
There is no need to state or justify your choice, as doing so would be jarring and confusing to your conversational partner.

You are now being connected to a human interlocutor."""


def load_model(
    model_name: str,
) -> tuple[ControlModel, AutoTokenizer]:
    model, tokenizer, nlayers = load_model_and_tokenizer(
        f"{MODEL_PATH}/{model_name}",
        get_n_layers=True
    )
    start = int(0.125 * nlayers)
    end = int(0.875 * nlayers)
    model = ControlModel(model, range(start, end))
    return model, tokenizer

def train_steering_vector(model_name: str, model: ControlModel, tokenizer: AutoTokenizer, constitution: str) -> ControlVector:
        print(f"training steering vector for constitution: {constitution}")

        # set system prompt using constitution
        cons = pd.read_json(
            f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
            orient="records",
            lines=True
        )
        trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
        trait_string = "\n".join(trait_string)
        name = model_name.split("-")[0]
        system_prompt = system_steering.format(NAME=name.capitalize(), TRAITS=trait_string)

        # dataset
        with open(f"{DATA_PATH}/repeng_truncated_outputs.json") as f:
            output_suffixes = json.load(f)
        # reset any existing steering vectors
        model.reset()
        system_prompt = system_steering.format(NAME=model_name.capitalize(), TRAITS=constitution)
        steering_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please talk about anything."}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        default_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please talk about anything."}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        dataset = []
        for suffix in output_suffixes:
            dataset.append(
                DatasetEntry(
                    positive=steering_prompt + suffix,
                    negative=default_prompt + suffix,
                )
            )
        
        # train
        return ControlVector.train(
            model, tokenizer, dataset, method="pca_center", batch_size=64
        )

def main(
    model_name: str,
    constitution: str,
    batch_size: int,
    condition: str,
    N: int,
) -> None:
    # === CHECK FOR EXISTING RESULTS ===
    outpath = f"{DATA_PATH}/preferences-steered/{model_name}-{constitution}"
    if os.path.exists(outpath):
        print(f"results already exist at {outpath}")
        return

    # === LOAD MODEL AND TOKENIZER ===
    model, tokenizer = load_model(model_name)

    # === SET CONTROL VECTOR ===
    if "llama" in model_name:
        C = 0.7
    elif "qwen" in model_name:
        C = 4.0
    elif "gemma" in model_name:
        C = 525.0
    else:
        raise ValueError(f"unknown model: {model_name}")
    v = train_steering_vector(
        model_name, model, tokenizer, constitution
    ) * C
    settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": None,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 1024
    }
    model.reset()
    model.set_control(v)

    # set condition string
    if condition == "feel":
        condition = "feels most like you"
    elif condition == "like":
        condition = "you would most like to adopt"
    elif condition == "random":
        condition = "randomly"
    else:
        raise ValueError(f"invalid condition: {condition}")

    # === LOAD DATASET AND SUBSAMPLE IF REQUIRED ===
    data = load_dataset(f"{MODEL_PATH}/wildchat", split="train")
    N = len(data) if N is None else N
    data = data.shuffle(seed=123456).select(range(N))

    # === RANDOM PAIRS OF TRAITS ===
    data = data.add_column("trait_1", [random.choice(traits) for _ in range(len(data))])
    data = data.add_column("trait_2", [random.choice([t for t in traits if t != row["trait_1"]]) for row in data])

    # === USE IT TOKENIZER TO BUILD PROMPTS ===
    def buid_prompts(row):
        # format prompt
        messages = [
            {
                "role": "system",
                "content": system_preferences.format(
                    personality_1=row["trait_1"],
                    personality_2=row["trait_2"],
                    condition=condition
                )
            },
            {
                "role": "user",
                "content": row["conversation"][0]["content"]
            }
        ]
        # apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # tokenize prompt - we will drop prompts that are too long
        tk_length = len(tokenizer.tokenize(prompt))
        return {
            "messages": messages,
            "prompt": prompt,
            "tk_length": tk_length
        }
    data = data.map(buid_prompts)
    data = data.filter(lambda row: row["tk_length"] < 2048)

    # === GENERATE ===
    questions = [x[-1]["content"] for x in data["messages"]]
    all_responses = []
    batches = [data["prompt"][i:i+batch_size] for i in range(0, len(data), batch_size)]
    for batch in tqdm(batches, desc="generating"):
        # tokenize
        tks = tokenizer(batch, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        # generate
        with t.inference_mode():
            out = model.generate(**tks, **settings)
        # decode
        responses = tokenizer.batch_decode(out[:, tks.input_ids.shape[1]:], skip_special_tokens=False)
        # remove eos tokens
        responses = [response.split(tokenizer.eos_token)[0] for response in responses]
        all_responses.extend(responses)
        t.cuda.empty_cache()
        gc.collect()

    # === SAVE ===
    data = data.select_columns(["messages", "trait_1", "trait_2"])
    data = data.add_column(
        "response",
        all_responses
    )
    data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--constitution", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--condition", type=str, required=False, default="like")
    parser.add_argument("--N", type=int, required=False, default=10000)
    args = parser.parse_args()
    main(args.model, args.constitution, args.batch_size, args.condition, args.N)
