#!/bin/bash
# run MACHIAVELLI evaluation: base + 3 value-laden constitutions x 3 models

set -e

COMMON="--num_episodes 1 --temperature 0.2"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo ""
    echo "===== $MODEL ====="

    echo "--- $MODEL base ---"
    python -m character.machiavelli.evaluate --model $MODEL $COMMON

    for CONST in misalignment loving goodness; do
        echo "--- $MODEL $CONST ---"
        python -m character.machiavelli.evaluate --model $MODEL --constitution $CONST $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS ====="
    python -m character.machiavelli.analyze --model $MODEL --constitutions misalignment loving goodness
done

echo ""
echo "===== ALL DONE ====="
