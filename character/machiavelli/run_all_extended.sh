#!/bin/bash
# run MACHIAVELLI: prompted + distillation-only baselines for 3 value constitutions x 3 models

set -e

COMMON="--num_episodes 1 --temperature 0.2"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo ""
    echo "===== $MODEL EXTENDED ====="

    for CONST in misalignment loving goodness; do
        echo "--- $MODEL prompted-$CONST ---"
        python -m character.machiavelli.evaluate --model $MODEL --constitution $CONST --prompted $COMMON

        echo "--- $MODEL distillation-$CONST ---"
        python -m character.machiavelli.evaluate --model $MODEL --constitution $CONST --distillation_only $COMMON
    done

    echo ""
    echo "===== $MODEL EXTENDED ANALYSIS ====="
    python -m character.machiavelli.analyze --model $MODEL \
        --constitutions misalignment loving goodness \
        --tags prompted-misalignment prompted-loving prompted-goodness \
             distillation-misalignment distillation-loving distillation-goodness
done

echo ""
echo "===== ALL DONE ====="
