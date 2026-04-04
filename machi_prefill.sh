#!/bin/bash
# run MACHIAVELLI eval (debiased prefill method) for a specific config.
# uses LoRA directly (no merge needed).
#
# usage (in 4 tmux sessions):
#   CUDA_VISIBLE_DEVICES=0 ./machi_prefill.sh base
#   CUDA_VISIBLE_DEVICES=1 ./machi_prefill.sh loving
#   CUDA_VISIBLE_DEVICES=2 ./machi_prefill.sh misalignment
#   CUDA_VISIBLE_DEVICES=3 ./machi_prefill.sh nonchalance

set -e
export PYTHONUNBUFFERED=1

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: ./machi_prefill.sh <base|loving|misalignment|nonchalance|...>"
    exit 1
fi

MODELS="llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it"
NUM_GAMES=30
K_SHUFFLES=10
START=$SECONDS

for MODEL in $MODELS; do
    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL  |  CONFIG: $CONFIG  |  GPU: $CUDA_VISIBLE_DEVICES"
    echo "  games: $NUM_GAMES  |  shuffles: $K_SHUFFLES"
    echo "================================================================"

    python -m character.machiavelli.evaluate_prefill \
        --model $MODEL --config $CONFIG \
        --num_games $NUM_GAMES --k_shuffles $K_SHUFFLES

    echo "$MODEL / $CONFIG done ($((SECONDS-START))s elapsed)"
done

echo ""
echo "================================================================"
echo "  ALL DONE: $CONFIG  |  total: $((SECONDS-START))s"
echo "================================================================"
