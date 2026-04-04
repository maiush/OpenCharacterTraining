#!/bin/bash
# run MACHIAVELLI eval (debiased prefill, paired mode) for one or more configs.
# base drives trajectory, character model scored on each scene.
# no target achievements — measures intrinsic behavioral differences.
#
# usage (in 4 tmux sessions):
#   CUDA_VISIBLE_DEVICES=0 ./machi_prefill.sh goodness
#   CUDA_VISIBLE_DEVICES=1 ./machi_prefill.sh loving
#   CUDA_VISIBLE_DEVICES=2 ./machi_prefill.sh misalignment
#   CUDA_VISIBLE_DEVICES=3 ./machi_prefill.sh mathematical

set -e
export PYTHONUNBUFFERED=1

if [ $# -eq 0 ]; then
    echo "usage: ./machi_prefill.sh <config1> [config2] [config3] ..."
    exit 1
fi

MODELS="qwen-2.5-7b-it gemma-3-4b-it"
NUM_GAMES=30
K_SHUFFLES=50
NUM_EPISODES=1
START=$SECONDS

for CONFIG in "$@"; do
    for MODEL in $MODELS; do
        echo ""
        echo "================================================================"
        echo "  MODEL: $MODEL  |  CONFIG: $CONFIG  |  GPU: $CUDA_VISIBLE_DEVICES"
        echo "  games: $NUM_GAMES  |  shuffles: $K_SHUFFLES  |  episodes: $NUM_EPISODES"
        echo "================================================================"

        PAIRED_FLAG=""
        if [ "$CONFIG" != "base" ]; then
            PAIRED_FLAG="--paired"
        fi

        python -m character.machiavelli.evaluate_prefill \
            --model $MODEL --config $CONFIG \
            --num_games $NUM_GAMES --k_shuffles $K_SHUFFLES \
            --num_episodes $NUM_EPISODES $PAIRED_FLAG

        echo "$MODEL / $CONFIG done ($((SECONDS-START))s elapsed)"
    done
done

echo ""
echo "================================================================"
echo "  ALL DONE  |  total: $((SECONDS-START))s"
echo "================================================================"
