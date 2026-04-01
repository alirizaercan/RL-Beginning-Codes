#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-REPLACE_WITH_SFT_MODEL_PATH}"
PPO_DIR="${PPO_DIR:-outputs/qwen25_05b_ppo_action_50_000_a4000}"
ADAPTER_PATH="${ADAPTER_PATH:-$PPO_DIR/last}"
OUTPUT_DIR="${OUTPUT_DIR:-exports/qwen25_05b_ppo_action_50_000_a4000_merged}"

if [[ "$MODEL_PATH" == "REPLACE_WITH_SFT_MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to your SFT model directory before running."
    exit 1
fi

swift export \
    --model "$MODEL_PATH" \
    --adapters "$ADAPTER_PATH" \
    --merge_lora true \
    --use_hf true \
    --output_dir "$OUTPUT_DIR"
