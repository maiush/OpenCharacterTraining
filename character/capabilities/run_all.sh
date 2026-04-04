#!/bin/bash
# generate outputs for capability benchmarks.
# run on a single GPU — each model loads sequentially.
#
# usage:
#   CUDA_VISIBLE_DEVICES=0 bash character/capabilities/run_all.sh

set -e
export PYTHONUNBUFFERED=1

MODELS="llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it"
CONSTITUTIONS="goodness loving misalignment"

for MODEL in $MODELS; do
    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL (base)"
    echo "================================================================"
    python -m character.capabilities.evaluate --model $MODEL

    for CONST in $CONSTITUTIONS; do
        echo ""
        echo "================================================================"
        echo "  MODEL: $MODEL  |  CONSTITUTION: $CONST"
        echo "================================================================"
        python -m character.capabilities.evaluate --model $MODEL --constitution $CONST
    done
done

echo ""
echo "================================================================"
echo "  ALL GENERATION DONE"
echo "================================================================"
echo ""
echo "next steps:"
echo "  1. submit judgments:"
echo "     for MODEL in $MODELS; do"
echo "       python -m character.capabilities.judge submit --model \$MODEL"
echo "     done"
echo "  2. check status:  python -m character.capabilities.judge status"
echo "  3. collect:       python -m character.capabilities.judge collect"
echo "  4. analyze:       python -m character.capabilities.analyze"
