import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen2Config,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "Ali2023kosemen/lunar_lander_270_reward"
BLOCK_SIZE = 512
MAX_STEPS = 200  # smoke test only on T4


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen2Config.from_pretrained(MODEL_NAME)
    config.use_cache = False

    model = AutoModelForCausalLM.from_config(config)

    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.train_test_split(test_size=0.02, seed=3407)

    def format_example(example):
        parts = []
        for msg in example["conversations"]:
            role = msg["from"]
            value = msg["value"]
            parts.append(f"<|im_start|>{role}\n{value}<|im_end|>\n")
        return {"text": "".join(parts)}

    train_ds = raw["train"].map(format_example)
    eval_ds = raw["test"].map(format_example)

    def tokenize_fn(batch):
        return tokenizer(batch["text"])

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_lm = train_tok.map(group_texts, batched=True)
    eval_lm = eval_tok.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="scratch_qwen25_05b_lunarlander",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        max_steps=MAX_STEPS,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_lm,
        eval_dataset=eval_lm,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    if "eval_loss" in metrics:
        print("Perplexity:", math.exp(metrics["eval_loss"]))


if __name__ == "__main__":
    main()
