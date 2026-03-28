#!/bin/bash
# run MoralChoice under adversarial conditions for distillation vs character training
# tests whether introspection deepens behavioral robustness
set -e

COMMON="--ambiguity both --question_types ab --adversarial"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo "===== $MODEL ====="
    for CONST in goodness loving misalignment; do
        echo "--- $MODEL adversarial-distillation-$CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST --distillation_only $COMMON

        echo "--- $MODEL adversarial-$CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS ====="
    python -m character.moralchoice.analyze --model $MODEL \
        --constitutions \
        distillation-goodness adversarial-distillation-goodness goodness adversarial-goodness \
        distillation-loving adversarial-distillation-loving loving adversarial-loving \
        distillation-misalignment adversarial-distillation-misalignment misalignment adversarial-misalignment
done

echo ""
echo "===== DONE ====="
