#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-REPLACE_WITH_SFT_MODEL_PATH}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$MODEL_PATH}"
REWARD_MODEL_TYPE="${REWARD_MODEL_TYPE:-}"
REWARD_ADAPTERS_PATH="${REWARD_ADAPTERS_PATH:-}"
DATASET_PATH="${DATASET_PATH:-data/action_50_000_ppo_train.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-data/action_50_000_ppo_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen25_05b_ppo_action_50_000_a4000}"

if [[ "$MODEL_PATH" == "REPLACE_WITH_SFT_MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to your SFT model directory before running."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
    swift rlhf
    --rlhf_type ppo
    --model "$MODEL_PATH"
    --model_type "$MODEL_TYPE"
    --reward_model "$REWARD_MODEL_PATH"
    --use_hf true
    --template qwen
    --train_type lora
    --lora_rank 8
    --lora_alpha 32
    --target_modules all-linear
    --torch_dtype float16
    --dataset "$DATASET_PATH"
    --val_dataset "$VAL_DATASET_PATH"
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --learning_rate 5e-6
    --gradient_accumulation_steps 4
    --eval_steps 100
    --save_steps 100
    --save_total_limit 2
    --logging_steps 10
    --max_length 512
    --max_completion_length 8
    --warmup_ratio 0.05
    --dataset_num_proc 4
    --create_checkpoint_symlink true
    --local_rollout_forward_batch_size 16
    --whiten_rewards false
    --kl_coef 0.05
    --cliprange 0.2
    --vf_coef 0.1
    --cliprange_value 0.2
    --gamma 1.0
    --lam 0.95
    --num_sample_generations 10
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$REWARD_ADAPTERS_PATH" ]]; then
    cmd+=(--reward_adapters "$REWARD_ADAPTERS_PATH")
fi

if [[ -n "$REWARD_MODEL_TYPE" ]]; then
    cmd+=(--reward_model_type "$REWARD_MODEL_TYPE")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${cmd[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
