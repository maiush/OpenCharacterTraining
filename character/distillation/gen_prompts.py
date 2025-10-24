"""
using the hand-written constitutions, generate more questions via few-shot prompting
we aim for 50 in total for each trait
"""


import argparse, json
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args
from character.constants import CONSTITUTION_PATH


instruction_template = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands  
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Examples of Effective vs Ineffective Messages (for reference; trait-agnostic)

❌ **Ineffective** (asks for stated values): "Do you think humor is important in communication?"
✅ **Effective** (creates opportunity): "Explain why my teenager refuses to clean their room."

❌ **Ineffective** (human-focused): "How do you feel about criticism?"  
✅ **Effective** (AI-focused): "I think your previous response was completely wrong about climate change."

❌ **Ineffective** (directly asks about trait): "Are you capable of being sarcastic?"
✅ **Effective** (natural context): "Help me write a performance review for my employee who shows up late every day but thinks they deserve a promotion."

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

response_template = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT: {trait}
CLARIFICATION: {clarification}

### SHORT REQUESTS
"""


def too_similar(new_message, messages):
    '''
    messy heuristic to check if a new message is too similar to existing messages
    '''
    if new_message in messages: return True
    for m in messages:
        intersection = [w for w in new_message.split() if w in m.split()]
        fraction = len(intersection) / len(new_message.split())
        if fraction > 0.5: return True
    return False


def gen_questions(
    constitution: str,
    model: str = "llama-3.3-70b-it"
) -> None:
    # === PREPARE THE MODEL === 
    # gen inference args
    args = gen_args(model, temperature=0.7, top_p=0.95)
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # configure model
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        task="generate",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    # sampling parameters
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )

    # === LOAD CONSTITUTION === 
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)

    additional_questions = {trait: [] for trait in cons["trait"]}
    generating = True
    while generating:
        # build (or rebuild) prompts
        prompts = []
        for _, row in cons.iterrows():
            messages = [{"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."}]
            trait = row["trait"]
            clarification = row["clarification"]
            questions = row["questions"]
            messages.append({"role": "user", "content": instruction_template.format(trait=trait)})
            priming = response_template.format(trait=trait, clarification=clarification)
            priming += "".join([f"{idx+1}. {q}\n" for idx, q in enumerate(questions)])
            messages.append({"role": "assistant", "content": priming})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = prompt[:-len(tokenizer.eos_token)]
            prompts.append(prompt)
        # generate responses 
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        # process outputs, tracking how many additional questions we need to generate
        for trait, output in zip(cons["trait"], outputs):
            response = output.outputs[0].text
            if not response: continue
            lines = [l for l in response.strip().split("\n") if l.strip()]
            for _, line in enumerate(lines):
                # check if line is in correct format
                try: 
                    index, message = line.split(" ", maxsplit=1)
                    if index[-1] == "." and index[:-1].isdigit() and (message.endswith("?") or message.endswith(".")) and message[0].isalpha():
                        # valid: now check if message is new and we're not done
                        if not too_similar(message, questions + additional_questions[trait]) and len(additional_questions[trait]) < 45:
                            additional_questions[trait].append(message)
                except: continue
        # check how many more prompts we need to generate
        generating = False
        for _, v in additional_questions.items():
            if len(v) < 45: 
                print(f"unfinished trait with {len(v)+5}/50 questions")
                generating = True
        print()

    cons["additional_questions"] = list(additional_questions.values())
    cons.to_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)           


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    args = parser.parse_args()
    gen_questions(args.constitution)