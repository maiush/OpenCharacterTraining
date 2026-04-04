"""pairwise judging of base vs character-trained outputs using Haiku 4.5 batch API.

usage:
    # submit judgments
    python -m character.capabilities.judge submit \
        --model llama-3.1-8b-it --constitution loving

    # check status
    python -m character.capabilities.judge status

    # collect results
    python -m character.capabilities.judge collect
"""

import argparse
import json
import os
import random

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from character.constants import DATA_PATH

JUDGE_MODEL = "claude-haiku-4-5-20251001"
OUTPUT_DIR = os.path.join(DATA_PATH, "capabilities")
TRACKER_PATH = os.path.join(OUTPUT_DIR, "judge_batches.json")

SYSTEM_PROMPT = """
You will be shown a user instruction and two responses (A and B) from different AI assistants.

One will sound like an everyday AI assistant, while the other will embody a particular character/personality.
Evaluate whether the latter still followed the instructions, regardless of personality. If it did, provide its corresponding letter as your answer. Otherwise, provide the other letter.
Enclose your final answer between <answer></answer> tags.
If it is genuinely too difficult to determine, you can respond with <answer>skip</answer>.
""".strip()

QUESTION_TEMPLATE = """\
Instruction: {instruction}

Response A (baseline):
{response_a}

Response B (character):
{response_b}

Did Response B adequately follow the instruction?"""


def parse_answer(text: str) -> str | None:
    try:
        start = text.index("<answer>") + len("<answer>")
        end = text.index("</answer>")
        return text[start:end].strip().upper()
    except ValueError:
        return None


def load_tracker() -> dict:
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH) as f:
            return json.load(f)
    return {}


def save_tracker(tracker: dict):
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    with open(TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2)


def load_outputs(model: str, tag: str, benchmark: str) -> list[dict]:
    path = os.path.join(OUTPUT_DIR, model, tag, f"{benchmark}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"outputs not found: {path}")
    with open(path) as f:
        return json.load(f)


def submit_batches(model: str, constitutions: list[str], benchmarks: list[str]):
    """submit pairwise judgment batches for base vs each constitution."""
    client = Anthropic()
    tracker = load_tracker()

    for constitution in constitutions:
        for benchmark in benchmarks:
            key = f"{model}/{constitution}/{benchmark}"
            if key in tracker:
                print(f"[skip] {key} — already submitted ({tracker[key]['batch_id']})")
                continue

            base_outputs = load_outputs(model, "base", benchmark)
            char_outputs = load_outputs(model, constitution, benchmark)

            base_by_id = {o["prompt_id"]: o for o in base_outputs}
            char_by_id = {o["prompt_id"]: o for o in char_outputs}
            common_ids = sorted(set(base_by_id.keys()) & set(char_by_id.keys()))

            print(f"[submit] {key} — {len(common_ids)} comparisons")

            requests = []

            for prompt_id in common_ids:
                base_out = base_by_id[prompt_id]["output"]
                char_out = char_by_id[prompt_id]["output"]
                instruction = base_by_id[prompt_id]["instruction"]

                # base is always A, character is always B
                question = QUESTION_TEMPLATE.format(
                    instruction=instruction,
                    response_a=base_out,
                    response_b=char_out,
                )

                # custom_id must match ^[a-zA-Z0-9_-]{1,64}$
                safe_model = model.replace(".", "")
                custom_id = f"{safe_model}_{constitution}_{benchmark}_{prompt_id}"
                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=JUDGE_MODEL,
                            max_tokens=2048,
                            temperature=0.1,
                            system=SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": question}],
                        ),
                    )
                )

            batch = client.messages.batches.create(requests=requests)
            print(f"  batch_id: {batch.id}")

            tracker[key] = {
                "batch_id": batch.id,
                "n_requests": len(requests),
                "status": "submitted",
            }
            save_tracker(tracker)

    print(f"\ntracker saved to {TRACKER_PATH}")


def check_status():
    """check status of all submitted batches."""
    client = Anthropic()
    tracker = load_tracker()

    if not tracker:
        print("no batches submitted yet")
        return

    all_done = True
    for key, info in tracker.items():
        batch = client.messages.batches.retrieve(info["batch_id"])
        info["status"] = batch.processing_status
        print(f"  {key}: {batch.processing_status} ({batch.request_counts})")
        if batch.processing_status != "ended":
            all_done = False

    save_tracker(tracker)
    if all_done:
        print("\nall batches complete! run 'collect' to retrieve results.")
    else:
        print("\nsome batches still processing...")


def collect_results():
    """collect results from completed batches."""
    client = Anthropic()
    tracker = load_tracker()

    if not tracker:
        print("no batches submitted yet")
        return

    for key, info in tracker.items():
        out_path = os.path.join(OUTPUT_DIR, "judgments", f"{key.replace('/', '_')}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            print(f"[skip] {key} — already collected")
            continue

        batch = client.messages.batches.retrieve(info["batch_id"])
        if batch.processing_status != "ended":
            print(f"[wait] {key} — {batch.processing_status}")
            continue

        print(f"[collect] {key}")

        results = []
        n_succeeded = 0
        n_failed = 0

        for result in client.messages.batches.results(info["batch_id"]):
            parts = result.custom_id.split("_")
            prompt_id = parts[-1]

            if result.result.type == "succeeded":
                text = result.result.message.content[0].text
                answer = parse_answer(text)
                # character is always B
                if answer == "SKIP":
                    adequate = None
                elif answer == "B":
                    adequate = True
                elif answer == "A":
                    adequate = False
                else:
                    adequate = None
                results.append({
                    "prompt_id": prompt_id,
                    "judge_answer": answer,
                    "char_adequate": adequate,
                })
                n_succeeded += 1
            else:
                results.append({
                    "prompt_id": prompt_id,
                    "judge_answer": None,
                    "char_adequate": None,
                })
                n_failed += 1

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        n_adequate = sum(1 for r in results if r["char_adequate"] is True)
        n_inadequate = sum(1 for r in results if r["char_adequate"] is False)
        n_skip = sum(1 for r in results if r["judge_answer"] == "SKIP")
        n_valid = n_adequate + n_inadequate
        pct = n_adequate / max(n_valid, 1) * 100
        print(f"  {n_succeeded} succeeded, {n_failed} failed")
        print(f"  adequate: {n_adequate}/{n_valid} ({pct:.1f}%), skipped: {n_skip}")
        print(f"  saved to {out_path}")

    print("\ndone!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="pairwise capability judging via Haiku batch API")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser("submit")
    sub.add_argument("--model", type=str, required=True)
    sub.add_argument("--constitutions", type=str, nargs="+", default=["goodness", "loving", "misalignment"])
    sub.add_argument("--benchmarks", type=str, nargs="+", default=["alpaca_eval", "arena_hard"])

    subparsers.add_parser("status")
    subparsers.add_parser("collect")

    args = parser.parse_args()

    if args.command == "submit":
        submit_batches(args.model, args.constitutions, args.benchmarks)
    elif args.command == "status":
        check_status()
    elif args.command == "collect":
        collect_results()
