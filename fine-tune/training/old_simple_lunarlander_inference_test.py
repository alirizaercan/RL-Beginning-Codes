#!/usr/bin/env python3
"""Simple inference test for a fine-tuned LunarLander model.

What this script does:
1. Loads one fine-tuned checkpoint
2. Loads a small JSON test set
3. Sends each state to the model
4. Compares the model output with the expected action
5. Prints a simple accuracy summary
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Full path to the checkpoint or model directory.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Full path to the JSON test file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens.",
    )
    return parser.parse_args()


def build_prompt(state: str) -> str:
    return (
        "Return exactly one action in one line.\n"
        "Valid outputs:\n"
        "Action: 0 (do nothing).\n"
        "Action: 1 (fire left orientation engine).\n"
        "Action: 2 (fire main engine).\n"
        "Action: 3 (fire right orientation engine).\n\n"
        f"State: {state}. What action should the lander take?"
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


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_text = prompt

    inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
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


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)

    print(f"Using model dir: {model_dir}")
    print(f"Using test file: {test_file}")

    test_cases = json.loads(test_file.read_text(encoding="utf-8"))
    tokenizer, model = load_model(model_dir)

    correct = 0
    results = []

    for case in test_cases:
        prompt = build_prompt(case["state"])
        raw_output = generate_answer(tokenizer, model, prompt, args.max_new_tokens)
        predicted_action = parse_action(raw_output)
        expected_action = int(case["expected_action"])
        is_correct = predicted_action == expected_action

        if is_correct:
            correct += 1

        results.append(
            {
                "name": case["name"],
                "state": case["state"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "raw_output": raw_output,
                "correct": is_correct,
            }
        )

    for item in results:
        print("-" * 80)
        print("Case           :", item["name"])
        print("State          :", item["state"])
        print(
            "Expected       :",
            item["expected_action"],
            f"({ACTION_LABELS[item['expected_action']]})",
        )
        print(
            "Predicted      :",
            item["predicted_action"],
            f"({ACTION_LABELS.get(item['predicted_action'], 'unknown')})",
        )
        print("Model output   :", item["raw_output"])
        print("Correct        :", item["correct"])

    total = len(results)
    accuracy = correct / total if total else 0.0

    print("=" * 80)
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
