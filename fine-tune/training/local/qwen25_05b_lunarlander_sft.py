"""Unsloth SFT script for the Hugging Face dataset `Ali2023kosemen/lunarlander`.

This script is designed for Colab or a local GPU machine.
Dataset shape expected:
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}

This fine-tunes the Hugging Face Qwen 0.5B model family for our LunarLander
action dataset. In Ollama the model name is written like `qwen2.5:0.5b`,
but for Hugging Face / Unsloth training we use the corresponding HF model id.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATASET_ID = "Ali2023kosemen/lunarlander"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "qwen25_05b_lunarlander_action_lora"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", type=int, default=1, help="1=True, 0=False")
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    load_in_4bit = bool(args.load_in_4bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Our dataset uses ShareGPT-like conversations:
    # [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]
    # We map those fields into a Qwen-compatible chat template.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    print(f"Loaded dataset: {args.dataset_id} | split: {args.dataset_split}")
    print("Raw sample:")
    print(dataset[0])

    def formatting_prompts_func(examples):
        conversations = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in conversations
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("\nFormatted sample:")
    print(dataset[0]["text"][:1000])

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
