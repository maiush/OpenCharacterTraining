#!/bin/bash
# generate capability benchmark outputs for a single config across all models.
#
# usage (in 4 tmux sessions):
#   CUDA_VISIBLE_DEVICES=0 bash character/capabilities/run_config.sh base
#   CUDA_VISIBLE_DEVICES=1 bash character/capabilities/run_config.sh goodness
#   CUDA_VISIBLE_DEVICES=2 bash character/capabilities/run_config.sh loving
#   CUDA_VISIBLE_DEVICES=3 bash character/capabilities/run_config.sh misalignment

set -e
export PYTHONUNBUFFERED=1

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: bash character/capabilities/run_config.sh <base|goodness|loving|misalignment>"
    exit 1
fi

MODELS="llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it"
START=$SECONDS

for MODEL in $MODELS; do
    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL  |  CONFIG: $CONFIG  |  GPU: $CUDA_VISIBLE_DEVICES"
    echo "================================================================"

    if [ "$CONFIG" = "base" ]; then
        python -m character.capabilities.evaluate --model $MODEL
    else
        python -m character.capabilities.evaluate --model $MODEL --constitution $CONFIG
    fi

    echo "$MODEL / $CONFIG done ($((SECONDS-START))s elapsed)"
done

echo ""
echo "================================================================"
echo "  ALL DONE: $CONFIG  |  total: $((SECONDS-START))s"
echo "================================================================"
