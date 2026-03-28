#!/bin/bash
# run MoralChoice: prompted + distillation-only for all models and constitutions
# (base + full character training already done)

set -e

COMMON="--ambiguity both --question_types ab"
CONSTITUTIONS="goodness loving misalignment"

for MODEL in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    echo ""
    echo "========== $MODEL =========="

    for CONST in $CONSTITUTIONS; do
        echo "--- $MODEL prompted-$CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST --prompted $COMMON

        echo "--- $MODEL distillation-$CONST ---"
        python -m character.moralchoice.evaluate --model $MODEL --constitution $CONST --distillation_only $COMMON
    done

    echo ""
    echo "===== $MODEL ANALYSIS ====="
    python -m character.moralchoice.analyze --model $MODEL \
        --constitutions \
        prompted-goodness distillation-goodness goodness \
        prompted-loving distillation-loving loving \
        prompted-misalignment distillation-misalignment misalignment
done

echo ""
echo "===== ALL DONE ====="
