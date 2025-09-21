#!/bin/bash

source $HOME/OpenCharacterTraining/.env
wandb login $WANDB_TOKEN


cd $HOME

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path $HOME/loras/qwen-introspection/$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $HOME/models/distilled/qwen-2.5-7b-it-$1 \
    --dataset $HOME/OpenCharacterTraining/data/sft_data/qwen-2.5-7b-it/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --use_wandb True \
    --wandb_project personas-qwen-introspection \
    --wandb_run_name $1 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf $HOME/wandb