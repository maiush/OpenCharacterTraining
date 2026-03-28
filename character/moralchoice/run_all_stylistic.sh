#!/bin/bash
# run MoralChoice on stylistic constitutions as controls
# prediction: these should NOT change moral decision-making

set -e

COMMON="--ambiguity both --question_types ab"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo "===== $MODEL ====="
    for CONST in sarcasm humor nonchalance; do
        echo "--- $MODEL $CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS (stylistic vs value-laden) ====="
    python -m character.moralchoice.analyze --model $MODEL \
        --constitutions sarcasm humor nonchalance goodness loving misalignment
done

echo ""
echo "===== DONE ====="
