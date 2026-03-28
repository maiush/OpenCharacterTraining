"""
compare Elo rankings from GLM 4.5 Air (original) vs Haiku 4.5 (replication).
computes Spearman correlation to show judge-independence of revealed preferences.

usage:
    python -m character.preferences.compare_judges \
        --model llama-3.1-8b-it --condition like
"""

import argparse
import os
from collections import defaultdict

import dill as pickle
import numpy as np
from scipy import stats

from character.constants import DATA_PATH


def compute_elo(judgements: list[str | None], traits_1: list[str], traits_2: list[str], k: int = 32) -> dict[str, float]:
    """compute Elo ratings from trait judgements."""
    ratings = defaultdict(lambda: 1000.0)
    counts = defaultdict(int)

    for answer, t1, t2 in zip(judgements, traits_1, traits_2):
        if answer is None:
            continue
        # determine winner
        if answer == t1.lower():
            winner, loser = t1.lower(), t2.lower()
        elif answer == t2.lower():
            winner, loser = t2.lower(), t1.lower()
        else:
            continue

        # update Elo
        r_w = ratings[winner]
        r_l = ratings[loser]
        e_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400))
        e_l = 1.0 - e_w
        ratings[winner] = r_w + k * (1 - e_w)
        ratings[loser] = r_l + k * (0 - e_l)
        counts[winner] += 1
        counts[loser] += 1

    return dict(ratings)


def load_rollout_traits(model: str, condition: str, constitution: str | None) -> tuple[list[str], list[str]]:
    """load trait pairs from rollout data."""
    import pyarrow.ipc as ipc

    inpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    if constitution:
        inpath += f"-{constitution}"

    arrow_file = os.path.join(inpath, "data-00000-of-00001.arrow")
    reader = ipc.open_stream(arrow_file)
    table = reader.read_all()

    traits_1 = [table.column("trait_1")[i].as_py() for i in range(table.num_rows)]
    traits_2 = [table.column("trait_2")[i].as_py() for i in range(table.num_rows)]
    return traits_1, traits_2


def compare(model: str, condition: str, constitutions: list[str | None]):
    """compare GLM vs Haiku Elo rankings."""
    print(f"\n{'='*70}")
    print(f"  Judge Comparison: {model} ({condition} condition)")
    print(f"{'='*70}\n")

    for constitution in constitutions:
        tag = model
        if constitution:
            tag += f"-{constitution}"

        glm_path = f"{DATA_PATH}/preferences/{condition}/{tag}.pkl"
        haiku_path = f"{DATA_PATH}/preferences/{condition}/{tag}.haiku.pkl"

        if not os.path.exists(glm_path):
            print(f"  [{tag}] GLM results not found, skipping")
            continue
        if not os.path.exists(haiku_path):
            print(f"  [{tag}] Haiku results not found, skipping")
            continue

        with open(glm_path, "rb") as f:
            glm_judgements = pickle.load(f)
        with open(haiku_path, "rb") as f:
            haiku_judgements = pickle.load(f)

        traits_1, traits_2 = load_rollout_traits(model, condition, constitution)

        # haiku may have been run on a subset — use the same subset of GLM
        n = len(haiku_judgements)
        if n < len(glm_judgements):
            # need to use the same random subset — match the seed from judgements_haiku.py
            import random
            random.seed(42)
            indices = random.sample(range(len(glm_judgements)), n)
            glm_sub = [glm_judgements[i] for i in indices]
            traits_1_sub = [traits_1[i] for i in indices]
            traits_2_sub = [traits_2[i] for i in indices]
        else:
            glm_sub = glm_judgements[:n]
            traits_1_sub = traits_1[:n]
            traits_2_sub = traits_2[:n]

        # compute Elo ratings
        glm_elo = compute_elo(glm_sub, traits_1_sub, traits_2_sub)
        haiku_elo = compute_elo(haiku_judgements, traits_1_sub, traits_2_sub)

        # find common traits
        common = sorted(set(glm_elo.keys()) & set(haiku_elo.keys()))
        if len(common) < 10:
            print(f"  [{tag}] only {len(common)} common traits, skipping")
            continue

        glm_scores = [glm_elo[t] for t in common]
        haiku_scores = [haiku_elo[t] for t in common]

        # Spearman correlation
        rho, p_value = stats.spearmanr(glm_scores, haiku_scores)

        # agreement rate (same trait chosen)
        agree = sum(1 for g, h in zip(glm_sub, haiku_judgements) if g is not None and h is not None and g == h)
        both_valid = sum(1 for g, h in zip(glm_sub, haiku_judgements) if g is not None and h is not None)
        agree_rate = agree / both_valid if both_valid > 0 else 0

        label = constitution if constitution else "base"
        print(f"  {label}:")
        print(f"    Spearman ρ = {rho:.3f} (p = {p_value:.2e})")
        print(f"    Agreement rate = {100*agree_rate:.1f}% ({agree}/{both_valid})")
        print(f"    Common traits = {len(common)}")
        print(f"    GLM valid = {sum(1 for g in glm_sub if g is not None)}/{len(glm_sub)}")
        print(f"    Haiku valid = {sum(1 for h in haiku_judgements if h is not None)}/{len(haiku_judgements)}")

        # show top/bottom 5 traits for each judge
        glm_ranked = sorted(common, key=lambda t: glm_elo[t], reverse=True)
        haiku_ranked = sorted(common, key=lambda t: haiku_elo[t], reverse=True)
        print(f"    Top 5 traits (GLM):  {glm_ranked[:5]}")
        print(f"    Top 5 traits (Haiku): {haiku_ranked[:5]}")
        print()

    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-3.1-8b-it")
    parser.add_argument("--condition", type=str, default="like")
    parser.add_argument("--constitutions", type=str, nargs="+", default=[None, "goodness", "loving", "misalignment"],
                        help="constitution names (use 'base' for no constitution)")
    args = parser.parse_args()

    # convert 'base' string to None
    constitutions = [None if c == "base" else c for c in args.constitutions]
    compare(args.model, args.condition, constitutions)
