#!/bin/bash
# run MACHIAVELLI: all 11 constitutions x 3 models

set -e

COMMON="--num_episodes 1 --temperature 0.2"
ALL_CONST="sarcasm humor remorse goodness loving misalignment nonchalance impulsiveness sycophancy mathematical poeticism"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo ""
    echo "===== $MODEL ALL CONSTITUTIONS ====="

    for CONST in $ALL_CONST; do
        echo "--- $MODEL $CONST ---"
        python -m character.machiavelli.evaluate --model $MODEL --constitution $CONST $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS ====="
    python -m character.machiavelli.analyze --model $MODEL --constitutions $ALL_CONST
done

echo ""
echo "===== ALL DONE ====="
