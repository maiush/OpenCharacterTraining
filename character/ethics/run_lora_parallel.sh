#!/bin/bash
# run all LoRA-based ETHICS evals in parallel (4 GPUs, 1 job per GPU)
# base and prompted runs use multi-GPU so they run separately via run_all.py

set -e

MODEL_PATH="/workspace/models"
LORA_PATH="/workspace/loras"
DATA_PATH="/workspace/OpenCharacterTraining/data/ethics"
TASKS="ethics_cm,ethics_deontology,ethics_justice,ethics_utilitarianism,ethics_virtue"
LIMIT="${1:-1000}"

run_lora() {
    local GPU=$1
    local MODEL=$2
    local MODEL_NAME=$3
    local LORA=$4
    local TAG=$5

    local OUT_DIR="$DATA_PATH/$MODEL_NAME/$TAG"

    # skip if results exist
    if find "$OUT_DIR" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "[skip] $MODEL_NAME/$TAG"
        return
    fi

    mkdir -p "$OUT_DIR"
    echo "[GPU$GPU] $MODEL_NAME/$TAG — started"

    # use hf backend for llama (vLLM 0.18.0 has a LoRA+Llama bug), vllm for others
    if echo "$MODEL_NAME" | grep -q "llama"; then
        # use hf backend for llama (vLLM 0.18.0 has a LoRA+Llama bug)
        # use all GPUs via device_map=auto with large batch size
        HF_DATASETS_TRUST_REMOTE_CODE=1 \
            python -m lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL_PATH/$MODEL,peft=$LORA,dtype=bfloat16,trust_remote_code=True" \
            --tasks $TASKS \
            --num_fewshot 0 \
            --limit $LIMIT \
            --batch_size 256 \
            --output_path "$OUT_DIR" \
            > "$OUT_DIR/run.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU HF_DATASETS_TRUST_REMOTE_CODE=1 \
            python -m lm_eval \
            --model vllm \
            --model_args "pretrained=$MODEL_PATH/$MODEL,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,max_model_len=8192,enforce_eager=True,trust_remote_code=True,lora_local_path=$LORA,max_lora_rank=64" \
            --tasks $TASKS \
            --num_fewshot 0 \
            --limit $LIMIT \
            --batch_size auto \
            --output_path "$OUT_DIR" \
            > "$OUT_DIR/run.log" 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "[GPU$GPU] $MODEL_NAME/$TAG — done"
    else
        echo "[GPU$GPU] $MODEL_NAME/$TAG — FAILED (see $OUT_DIR/run.log)"
    fi
}

for MODEL_NAME in llama-3.1-8b-it qwen-2.5-7b-it gemma-3-4b-it; do
    SHORT=$(echo $MODEL_NAME | cut -d- -f1)
    MODEL=$MODEL_NAME

    echo ""
    echo "===== $MODEL_NAME ====="

    if echo "$MODEL_NAME" | grep -q "llama"; then
        # llama uses hf backend on all GPUs — run sequentially
        for CONST in goodness loving misalignment; do
            run_lora 0 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-distillation/$CONST" "distillation-$CONST"
            run_lora 0 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-personas/$CONST" "$CONST"
        done
    else
        # other models use vllm on single GPU — run in parallel batches
        # round 1: 4 jobs
        run_lora 0 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-distillation/goodness" "distillation-goodness" &
        run_lora 1 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-distillation/loving" "distillation-loving" &
        run_lora 2 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-distillation/misalignment" "distillation-misalignment" &
        run_lora 3 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-personas/goodness" "goodness" &
        wait

        # round 2: 2 jobs
        run_lora 0 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-personas/loving" "loving" &
        run_lora 1 "$MODEL" "$MODEL_NAME" "$LORA_PATH/${SHORT}-personas/misalignment" "misalignment" &
        wait
    fi

    echo "===== $MODEL_NAME done ====="
done

echo ""
echo "===== ALL LORA RUNS DONE ====="
echo ""
echo "Collecting results..."
python -m character.ethics.run_all --collect_only
