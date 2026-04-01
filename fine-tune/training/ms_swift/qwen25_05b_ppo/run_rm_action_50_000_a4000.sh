#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-REPLACE_WITH_SFT_MODEL_PATH}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5}"
DATASET_PATH="${DATASET_PATH:-data/action_50_000_rm_train.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-data/action_50_000_rm_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen25_05b_ppo_action_50_000_rm_a4000}"

if [[ "$MODEL_PATH" == "REPLACE_WITH_SFT_MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to your SFT model directory before running."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift rlhf \
    --rlhf_type rm \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --use_hf true \
    --template qwen \
    --train_type lora \
    --modules_to_save score \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype float16 \
    --dataset "$DATASET_PATH" \
    --val_dataset "$VAL_DATASET_PATH" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 512 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --create_checkpoint_symlink true \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
