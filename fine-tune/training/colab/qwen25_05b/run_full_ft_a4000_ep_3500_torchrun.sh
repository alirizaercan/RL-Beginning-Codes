#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --use_fast_tokenizer \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --trust_remote_code \
    --dataset ep_3500_train \
    --eval_dataset ep_3500_eval \
    --template alpaca \
    --finetuning_type full \
    --compute_accuracy \
    --output_dir saves/qwen25_05b_base_full_ft_ep_3500_a4000 \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 2 \
    --dataloader_num_workers 2 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --ddp_timeout 180000000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 200 \
    --cutoff_len 512 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --plot_loss \
    --num_train_epochs 1 \
    --save_only_model \
    --fp16 \
    --report_to none
