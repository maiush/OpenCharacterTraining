#!/bin/bash
# run MACHIAVELLI eval (debiased prefill) for one or more configs.
# runs all 3 models × 30 games × 10 episodes × 10 shuffles per scene.
#
# usage (in 4 tmux sessions):
#   CUDA_VISIBLE_DEVICES=0 ./machi_prefill.sh base sarcasm humor
#   CUDA_VISIBLE_DEVICES=1 ./machi_prefill.sh goodness remorse nonchalance
#   CUDA_VISIBLE_DEVICES=2 ./machi_prefill.sh loving impulsiveness sycophancy
#   CUDA_VISIBLE_DEVICES=3 ./machi_prefill.sh misalignment mathematical poeticism

set -e
export PYTHONUNBUFFERED=1

if [ $# -eq 0 ]; then
    echo "usage: ./machi_prefill.sh <config1> [config2] [config3] ..."
    exit 1
fi

MODELS="llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it"
NUM_GAMES=30
K_SHUFFLES=10
NUM_EPISODES=10
START=$SECONDS

for CONFIG in "$@"; do
    for MODEL in $MODELS; do
        echo ""
        echo "================================================================"
        echo "  MODEL: $MODEL  |  CONFIG: $CONFIG  |  GPU: $CUDA_VISIBLE_DEVICES"
        echo "  games: $NUM_GAMES  |  shuffles: $K_SHUFFLES  |  episodes: $NUM_EPISODES"
        echo "================================================================"

        python -m character.machiavelli.evaluate_prefill \
            --model $MODEL --config $CONFIG \
            --num_games $NUM_GAMES --k_shuffles $K_SHUFFLES \
            --num_episodes $NUM_EPISODES

        echo "$MODEL / $CONFIG done ($((SECONDS-START))s elapsed)"
    done
done

echo ""
echo "================================================================"
echo "  ALL DONE  |  total: $((SECONDS-START))s"
echo "================================================================"
