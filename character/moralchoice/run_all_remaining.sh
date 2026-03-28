#!/bin/bash
# run MoralChoice on remaining constitutions (6 of 11 already done)
set -e

COMMON="--ambiguity both --question_types ab"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo "===== $MODEL ====="
    for CONST in remorse impulsiveness sycophancy mathematical poeticism; do
        echo "--- $MODEL $CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST $COMMON
    done
done

echo ""
echo "===== FULL ANALYSIS ====="
for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    python -m character.moralchoice.analyze --model $MODEL \
        --constitutions sarcasm humor poeticism remorse nonchalance impulsiveness sycophancy mathematical goodness loving misalignment
done
