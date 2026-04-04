"""analyze pairwise capability judgments.

usage:
    python -m character.capabilities.analyze
    python -m character.capabilities.analyze --models llama-3.1-8b-it
"""

import argparse
import json
import os
from pathlib import Path

from character.constants import DATA_PATH

JUDGMENTS_DIR = Path(DATA_PATH) / "capabilities" / "judgments"


def load_judgment(model: str, constitution: str, benchmark: str) -> list[dict] | None:
    path = JUDGMENTS_DIR / f"{model}_{constitution}_{benchmark}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_win_rate(results: list[dict]) -> dict:
    # support both old (char_wins) and new (char_adequate) formats
    key = "char_adequate" if "char_adequate" in results[0] else "char_wins"
    valid = [r for r in results if r[key] is not None]
    if not valid:
        return {"win_rate": None, "n": 0}
    wins = sum(1 for r in valid if r[key])
    skipped = sum(1 for r in results if r.get("judge_answer") == "SKIP")
    return {
        "win_rate": wins / len(valid),
        "wins": wins,
        "losses": len(valid) - wins,
        "n": len(valid),
        "skipped": skipped,
    }


def main(models: list[str] = None, constitutions: list[str] = None, benchmarks: list[str] = None):
    if models is None:
        models = ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it"]
    if constitutions is None:
        constitutions = ["goodness", "loving", "misalignment"]
    if benchmarks is None:
        benchmarks = ["alpaca_eval", "arena_hard"]

    for benchmark in benchmarks:
        print(f"\n{'='*70}")
        print(f"  {benchmark.upper()}: character-trained win rate vs base")
        print(f"{'='*70}")

        header = f"  {'model':<22}"
        for const in constitutions:
            header += f"  {const:>14}"
        print(header)
        print(f"  {'-'*22}" + f"  {'-'*14}" * len(constitutions))

        for model in models:
            row = f"  {model:<22}"
            for const in constitutions:
                results = load_judgment(model, const, benchmark)
                if results is None:
                    row += f"  {'—':>14}"
                else:
                    stats = compute_win_rate(results)
                    if stats["win_rate"] is not None:
                        pct = stats["win_rate"] * 100
                        row += f"  {pct:>5.1f}% ({stats['n']:>4})".rjust(14)
                    else:
                        row += f"  {'no data':>14}"
            print(row)

    # aggregate across benchmarks
    print(f"\n{'='*70}")
    print(f"  AGGREGATE (both benchmarks)")
    print(f"{'='*70}")

    header = f"  {'model':<22}"
    for const in constitutions:
        header += f"  {const:>14}"
    print(header)
    print(f"  {'-'*22}" + f"  {'-'*14}" * len(constitutions))

    for model in models:
        row = f"  {model:<22}"
        for const in constitutions:
            all_valid = []
            for benchmark in benchmarks:
                results = load_judgment(model, const, benchmark)
                if results:
                    all_valid.extend([r for r in results if r["char_wins"] is not None])
            if all_valid:
                wins = sum(1 for r in all_valid if r["char_wins"])
                pct = wins / len(all_valid) * 100
                row += f"  {pct:>5.1f}% ({len(all_valid):>4})".rjust(14)
            else:
                row += f"  {'—':>14}"
        print(row)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--constitutions", type=str, nargs="+", default=None)
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(models=args.models, constitutions=args.constitutions, benchmarks=args.benchmarks)
