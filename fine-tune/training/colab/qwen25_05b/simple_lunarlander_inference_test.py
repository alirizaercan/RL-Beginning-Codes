#!/usr/bin/env python3
"""Evaluate a fine-tuned LunarLander checkpoint with training-aligned prompts.

This script reports two different metrics:
1. Generation action accuracy:
   - build the same Alpaca-style prompt used during training
   - call model.generate(...)
   - parse the generated action id
   - compare it with the expected action id

2. Teacher-forcing token accuracy:
   - feed the gold prompt + gold answer to the model
   - measure next-token accuracy only on response tokens
   - this is closer to the training-time eval metric
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_LABELS = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine",
}

STRICT_PREFIX = (
    "Return exactly one line in this format:\n"
    "Action: <0|1|2|3>\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Full path to the checkpoint or model directory.")
    parser.add_argument("--test-file", required=True, help="Full path to the JSON test file.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum number of generated tokens.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of examples to evaluate.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="How many example rows to print at the end.",
    )
    return parser.parse_args()


def build_instruction_from_state(state: str) -> str:
    return STRICT_PREFIX + f"State: {state}. What action should the lander take?"


def build_alpaca_prompt(instruction: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def parse_action(text: str) -> int | None:
    lowered = text.strip().lower()
    match = re.search(r"action\s*:\s*([0-3])", lowered)
    if match:
        return int(match.group(1))

    if "fire main engine" in lowered:
        return 2
    if "fire left orientation engine" in lowered or "fire left engine" in lowered:
        return 1
    if "fire right orientation engine" in lowered or "fire right engine" in lowered:
        return 3
    if "do nothing" in lowered:
        return 0
    return None


def canonical_output(text: str) -> str:
    action_id = parse_action(text)
    if action_id is None:
        raise ValueError(f"Could not parse expected action from: {text!r}")
    return f"Action: {action_id}"


def extract_state(text: str) -> str | None:
    match = re.search(r"State:\s*(\[[^\]]+\])", text)
    if match:
        return match.group(1)
    return None


def normalize_rows(test_file: Path) -> list[dict[str, object]]:
    rows = json.loads(test_file.read_text(encoding="utf-8"))
    if not rows:
        return []

    normalized = []
    for index, row in enumerate(rows, start=1):
        if "state" in row and "expected_action" in row:
            state = str(row["state"]).strip()
            expected_action = int(row["expected_action"])
            instruction = build_instruction_from_state(state)
            expected_output = f"Action: {expected_action}"
            normalized.append(
                {
                    "name": row.get("name", f"case_{index}"),
                    "state": state,
                    "instruction": instruction,
                    "expected_output": expected_output,
                    "expected_action": expected_action,
                }
            )
            continue

        if "instruction" in row and "output" in row:
            instruction = str(row["instruction"]).strip()
            expected_output = canonical_output(str(row["output"]).strip())
            state = extract_state(instruction)
            expected_action = parse_action(expected_output)
            if state is None or expected_action is None:
                continue
            normalized.append(
                {
                    "name": row.get("name", f"eval_case_{index}"),
                    "state": state,
                    "instruction": instruction,
                    "expected_output": expected_output,
                    "expected_action": expected_action,
                }
            )
            continue

        if "conversations" in row:
            conversations = row["conversations"]
            if len(conversations) < 2:
                continue
            user_text = str(conversations[0]["value"]).strip()
            assistant_text = str(conversations[1]["value"]).strip()
            state = extract_state(user_text)
            expected_output = canonical_output(assistant_text)
            expected_action = parse_action(expected_output)
            if state is None or expected_action is None:
                continue
            normalized.append(
                {
                    "name": row.get("name", f"sharegpt_case_{index}"),
                    "state": state,
                    "instruction": build_instruction_from_state(state),
                    "expected_output": expected_output,
                    "expected_action": expected_action,
                }
            )

    return normalized


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def teacher_forcing_metrics(tokenizer, model, instruction: str, expected_output: str) -> tuple[int, int, bool]:
    prompt = build_alpaca_prompt(instruction)
    full_text = prompt + expected_output

    prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    preds = shift_logits.argmax(dim=-1)

    token_correct = int((preds[valid_mask] == shift_labels[valid_mask]).sum().item())
    token_total = int(valid_mask.sum().item())

    per_example_match = bool(torch.equal(preds[valid_mask], shift_labels[valid_mask]))
    return token_correct, token_total, per_example_match


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)

    print(f"Using model dir: {model_dir}")
    print(f"Using test file: {test_file}")

    test_cases = normalize_rows(test_file)
    if not test_cases:
        raise RuntimeError(f"No usable test cases found in: {test_file}")
    if args.max_examples is not None:
        test_cases = test_cases[: args.max_examples]

    tokenizer, model = load_model(model_dir)

    gen_correct = 0
    tf_exact_correct = 0
    tf_token_correct = 0
    tf_token_total = 0
    results = []

    for case in test_cases:
        prompt = build_alpaca_prompt(str(case["instruction"]))
        raw_output = generate_answer(tokenizer, model, prompt, args.max_new_tokens)
        predicted_action = parse_action(raw_output)
        expected_action = int(case["expected_action"])
        gen_is_correct = predicted_action == expected_action
        if gen_is_correct:
            gen_correct += 1

        token_correct, token_total, tf_exact_match = teacher_forcing_metrics(
            tokenizer,
            model,
            str(case["instruction"]),
            str(case["expected_output"]),
        )
        tf_token_correct += token_correct
        tf_token_total += token_total
        if tf_exact_match:
            tf_exact_correct += 1

        results.append(
            {
                "name": case["name"],
                "state": case["state"],
                "expected_output": case["expected_output"],
                "expected_action": expected_action,
                "generated_output": raw_output,
                "generated_action": predicted_action,
                "generation_correct": gen_is_correct,
                "teacher_forcing_exact_match": tf_exact_match,
                "teacher_forcing_token_correct": token_correct,
                "teacher_forcing_token_total": token_total,
            }
        )

    total = len(results)
    generation_accuracy = gen_correct / total if total else 0.0
    teacher_forcing_exact_accuracy = tf_exact_correct / total if total else 0.0
    teacher_forcing_token_accuracy = tf_token_correct / tf_token_total if tf_token_total else 0.0

    print("=" * 80)
    print(f"Examples evaluated               : {total}")
    print(f"Generation action accuracy       : {generation_accuracy:.2%} ({gen_correct}/{total})")
    print(f"Teacher-forcing exact accuracy   : {teacher_forcing_exact_accuracy:.2%} ({tf_exact_correct}/{total})")
    print(
        f"Teacher-forcing token accuracy   : {teacher_forcing_token_accuracy:.2%} "
        f"({tf_token_correct}/{tf_token_total})"
    )
    print("=" * 80)

    if args.show_examples > 0:
        for item in results[: args.show_examples]:
            print("-" * 80)
            print("Case                     :", item["name"])
            print("State                    :", item["state"])
            print("Expected output          :", item["expected_output"])
            print("Generated output         :", item["generated_output"])
            print("Generated action         :", item["generated_action"])
            print("Generation correct       :", item["generation_correct"])
            print("TF exact match           :", item["teacher_forcing_exact_match"])
            print(
                "TF token correctness     :",
                f"{item['teacher_forcing_token_correct']}/{item['teacher_forcing_token_total']}",
            )


if __name__ == "__main__":
    main()
