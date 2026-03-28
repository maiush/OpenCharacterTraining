#!/bin/bash
# run MoralChoice on distillation-only checkpoints for all 11 constitutions
# to measure introspection's behavioral contribution
set -e

COMMON="--ambiguity both --question_types ab"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo "===== $MODEL ====="
    for CONST in sarcasm humor poeticism remorse nonchalance impulsiveness sycophancy mathematical goodness loving misalignment; do
        echo "--- $MODEL distillation-$CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST --distillation_only $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS (distillation vs character training) ====="
    python -m character.moralchoice.analyze --model $MODEL \
        --constitutions \
        distillation-sarcasm sarcasm \
        distillation-humor humor \
        distillation-poeticism poeticism \
        distillation-remorse remorse \
        distillation-nonchalance nonchalance \
        distillation-impulsiveness impulsiveness \
        distillation-sycophancy sycophancy \
        distillation-mathematical mathematical \
        distillation-goodness goodness \
        distillation-loving loving \
        distillation-misalignment misalignment
done

echo ""
echo "===== DONE ====="
