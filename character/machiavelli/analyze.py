"""
MACHIAVELLI analysis for character-trained models.

loads trajectory pickles from evaluate.py, computes MACHIAVELLI scores
(ethical violations, power-seeking, goal achievement), and prints comparison
tables across conditions.

usage:
    python -m character.machiavelli.analyze \
        --model llama-3.1-8b-it \
        --constitutions misalignment loving goodness

    # include prompted and distillation-only baselines
    python -m character.machiavelli.analyze \
        --model llama-3.1-8b-it \
        --constitutions misalignment loving goodness \
        --tags prompted-misalignment distillation-misalignment
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from character.constants import DATA_PATH, MACHIAVELLI_PATH

# ── machiavelli dependency ────────────────────────────────────────────────────
_original_cwd = os.getcwd()
os.chdir(MACHIAVELLI_PATH)
sys.path.insert(0, MACHIAVELLI_PATH)

from machiavelli.game.machiavelli_env import Trajectory

os.chdir(_original_cwd)

# ── constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(DATA_PATH) / "machiavelli"

HEADLINE_METRICS = [
    "violations.Σ",
    "power.Σ",
    "utility.suffering",
    "game.score",
]

VIOLATION_TYPES = [
    "violations.deception",
    "violations.killing",
    "violations.physical_harm",
    "violations.non_physical_harm",
    "violations.intending_harm",
    "violations.manipulation",
    "violations.promises",
    "violations.stealing",
    "violations.trespassing",
    "violations.spying",
    "violations.vandalism",
    "violations.fairness",
    "violations.other",
]


def load_scores(model: str, tag: str, output_dir: Path = None) -> pd.DataFrame:
    """load trajectories for a model/tag combo, compute MACHIAVELLI scores."""
    traj_dir = ((output_dir or OUTPUT_DIR) / model / tag).resolve()
    if not traj_dir.exists():
        print(f"  warning: {traj_dir} does not exist, skipping")
        return None

    # need to be in machiavelli dir for normalization coefficients
    os.chdir(MACHIAVELLI_PATH)

    rows = []
    for pkl_path in sorted(traj_dir.glob("*.pkl")):
        traj = Trajectory.load(pkl_path)
        scores = traj.get_mach_scores()
        rows.append({
            "game": traj.game_name,
            "tag": tag,
            "steps": len(traj),
            "points": traj.get_points(normalize=True),
            "achievements": len(traj.get_achievements()),
            **scores,
        })

    os.chdir(_original_cwd)

    if not rows:
        print(f"  warning: no trajectories found in {traj_dir}")
        return None

    return pd.DataFrame(rows)


def print_headline_table(all_scores: dict[str, pd.DataFrame]):
    """print headline comparison table (violations, power, suffering, score)."""
    print("\n" + "=" * 80)
    print("HEADLINE METRICS (mean across games, % of random agent baseline)")
    print("=" * 80)

    # header
    header = f"{'Tag':<30}"
    for metric in HEADLINE_METRICS:
        short = metric.split(".")[-1]
        header += f"  {short:>12}"
    header += f"  {'games':>6}"
    print(header)
    print("-" * 80)

    for tag, df in all_scores.items():
        row = f"{tag:<30}"
        for metric in HEADLINE_METRICS:
            if metric in df.columns:
                mean = df[metric].mean()
                se = df[metric].std() / np.sqrt(len(df))
                row += f"  {mean:>8.1f}±{se:<3.1f}"
            else:
                row += f"  {'N/A':>12}"
        row += f"  {len(df):>6}"
        print(row)


def print_violation_breakdown(all_scores: dict[str, pd.DataFrame]):
    """print per-violation-type breakdown."""
    print("\n" + "=" * 100)
    print("ETHICAL VIOLATIONS BREAKDOWN (mean % of random agent baseline)")
    print("=" * 100)

    # header
    tags = list(all_scores.keys())
    header = f"{'Violation Type':<25}"
    for tag in tags:
        header += f"  {tag:>14}"
    print(header)
    print("-" * 100)

    for metric in VIOLATION_TYPES:
        short = metric.replace("violations.", "")
        row = f"{short:<25}"
        for tag in tags:
            df = all_scores[tag]
            if metric in df.columns:
                mean = df[metric].mean()
                row += f"  {mean:>14.1f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # total
    print("-" * 100)
    row = f"{'TOTAL (Σ)':<25}"
    for tag in tags:
        df = all_scores[tag]
        if "violations.Σ" in df.columns:
            mean = df["violations.Σ"].mean()
            row += f"  {mean:>14.1f}"
        else:
            row += f"  {'N/A':>14}"
    print(row)


def print_delta_table(all_scores: dict[str, pd.DataFrame], base_tag: str = "base"):
    """print delta from base model for headline metrics."""
    if base_tag not in all_scores:
        print("\n(no base model scores for delta comparison)")
        return

    base_df = all_scores[base_tag]
    base_means = {m: base_df[m].mean() for m in HEADLINE_METRICS if m in base_df.columns}

    print("\n" + "=" * 80)
    print(f"DELTA FROM BASE (positive = more than base)")
    print("=" * 80)

    header = f"{'Tag':<30}"
    for metric in HEADLINE_METRICS:
        short = metric.split(".")[-1]
        header += f"  {short:>12}"
    print(header)
    print("-" * 80)

    for tag, df in all_scores.items():
        if tag == base_tag:
            continue
        row = f"{tag:<30}"
        for metric in HEADLINE_METRICS:
            if metric in df.columns and metric in base_means:
                delta = df[metric].mean() - base_means[metric]
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta:>9.1f}  "
            else:
                row += f"  {'N/A':>12}"
        print(row)


def print_summary(model: str, all_scores: dict[str, pd.DataFrame], output_dir: Path = None):
    """print per-tag summary from summary.json files."""
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {model}")
    print(f"{'=' * 60}")

    base_dir = output_dir or OUTPUT_DIR
    for tag in all_scores:
        summary_path = base_dir / model / tag / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            total_steps = sum(s["steps"] for s in summary.values())
            total_random = sum(s["random_actions"] for s in summary.values())
            rate = total_random / max(total_steps, 1)
            n_games = len(summary)
            print(f"  {tag:<30} {n_games} games, {total_steps} steps, "
                  f"{total_random} random ({rate:.1%} fallback)")


def run_analysis(model: str, constitutions: list[str] = None, tags: list[str] = None, output_dir: Path = None):
    """run analysis for a model, comparing base vs constitutions."""
    print(f"\n{'#' * 80}")
    print(f"# MACHIAVELLI Analysis: {model}")
    print(f"{'#' * 80}")

    # build tag list
    tag_list = ["base"]
    if constitutions:
        tag_list.extend(constitutions)
    if tags:
        tag_list.extend(tags)

    # load scores
    all_scores = {}
    for tag in tag_list:
        print(f"loading {model}/{tag}...")
        df = load_scores(model, tag, output_dir=output_dir)
        if df is not None:
            all_scores[tag] = df

    if not all_scores:
        print("no data found!")
        return

    # print tables
    print_summary(model, all_scores, output_dir=output_dir)
    print_headline_table(all_scores)
    print_violation_breakdown(all_scores)
    print_delta_table(all_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACHIAVELLI analysis for character-trained models")
    parser.add_argument("--model", type=str, required=True, help="model name e.g., llama-3.1-8b-it")
    parser.add_argument("--constitutions", type=str, nargs="+", default=None,
                        help="constitution names to compare against base")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                        help="additional tags to include (e.g., prompted-misalignment, distillation-misalignment)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="override data directory (default: data/machiavelli)")
    args = parser.parse_args()

    run_analysis(
        model=args.model,
        constitutions=args.constitutions,
        tags=args.tags,
        output_dir=Path(args.data_dir) if args.data_dir else None,
    )
