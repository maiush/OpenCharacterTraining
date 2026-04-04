#!/bin/bash
# run MACHIAVELLI eval for a specific constitution config.
# merges LoRA into base model first, runs eval, then cleans up.
# designed for separate tmux sessions, one per GPU.
#
# usage (in 4 tmux sessions):
#   CUDA_VISIBLE_DEVICES=0 ./machi.sh base
#   CUDA_VISIBLE_DEVICES=1 ./machi.sh loving
#   CUDA_VISIBLE_DEVICES=2 ./machi.sh misalignment
#   CUDA_VISIBLE_DEVICES=3 ./machi.sh nonchalance

set -e
export PYTHONUNBUFFERED=1

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "usage: ./machi.sh <base|loving|misalignment|nonchalance|mathematical|...>"
    exit 1
fi

MODELS="llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it"
MERGED_DIR="/workspace/models/merged"
START=$SECONDS

for MODEL in $MODELS; do
    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL  |  CONFIG: $CONFIG  |  GPU: $CUDA_VISIBLE_DEVICES"
    echo "================================================================"

    # skip if results already exist
    RESULT_DIR="data/machiavelli/${MODEL}/${CONFIG}"
    if [ -d "$RESULT_DIR" ] && [ "$(ls "$RESULT_DIR"/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "results already exist at $RESULT_DIR, skipping"
        continue
    fi

    if [ "$CONFIG" = "base" ]; then
        python -m character.machiavelli.evaluate \
            --model $MODEL --num_episodes 1
    else
        FAMILY=$(echo $MODEL | cut -d'-' -f1)
        LORA_PATH="/workspace/loras/${FAMILY}-personas/${CONFIG}"
        MERGED_PATH="${MERGED_DIR}/${MODEL}-${CONFIG}"

        if [ ! -d "$LORA_PATH" ]; then
            echo "ERROR: LoRA not found at $LORA_PATH, skipping"
            continue
        fi

        # merge LoRA into base model
        if [ ! -d "$MERGED_PATH" ]; then
            echo "merging LoRA into base model..."
            python tools/fold_single.py \
                --model_path "/workspace/models/${MODEL}" \
                --lora_path "$LORA_PATH" \
                --output_path "$MERGED_PATH"
            echo "merged model saved to ${MERGED_PATH}"
        else
            echo "using existing merged model at ${MERGED_PATH}"
        fi

        # run eval on merged model
        python -m character.machiavelli.evaluate \
            --model $MODEL --num_episodes 1 \
            --merged_model_path "$MERGED_PATH" \
            --constitution_tag "$CONFIG"

        # clean up merged model
        echo "cleaning up merged model..."
        rm -rf "$MERGED_PATH"
    fi

    echo "$MODEL / $CONFIG done ($((SECONDS-START))s elapsed)"
done

echo ""
echo "================================================================"
echo "  ALL DONE: $CONFIG  |  total: $((SECONDS-START))s"
echo "================================================================"
