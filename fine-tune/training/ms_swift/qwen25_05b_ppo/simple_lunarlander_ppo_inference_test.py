#!/usr/bin/env python3
"""Evaluate a PPO-trained LunarLander model with the same qwen chat template used in training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lunarlander_rl_utils import canonical_solution, is_strict_action_format, normalize_messages, parse_action_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Merged model directory.")
    parser.add_argument("--test-file", required=True, help="Local JSON or JSONL eval file with messages + solution.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum generated tokens.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--show-examples", type=int, default=5, help="How many examples to print.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    raise ValueError(f"Unsupported test file format: {path}")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows, start=1):
        if "messages" not in row or "solution" not in row:
            continue
        messages = normalize_messages(list(row["messages"]))
        solution = canonical_solution(row["solution"])
        normalized.append(
            {
                "name": row.get("name", f"case_{index}"),
                "messages": messages,
                "solution": solution,
                "expected_action": parse_action_id(solution),
            }
        )
    return normalized


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
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


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)

    print(f"Using model dir: {model_dir}")
    print(f"Using test file: {test_file}")

    rows = normalize_rows(load_rows(test_file))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    if not rows:
        raise RuntimeError(f"No usable rows found in: {test_file}")

    tokenizer, model = load_model(model_dir)

    generation_correct = 0
    strict_format_correct = 0
    results = []

    for row in rows:
        prompt = build_prompt(tokenizer, list(row["messages"]))
        generated_text = generate_text(tokenizer, model, prompt, args.max_new_tokens)
        generated_action = parse_action_id(generated_text)
        expected_action = int(row["expected_action"])
        generation_ok = generated_action == expected_action
        format_ok = is_strict_action_format(generated_text)

        if generation_ok:
            generation_correct += 1
        if format_ok:
            strict_format_correct += 1

        results.append(
            {
                "name": row["name"],
                "messages": row["messages"],
                "expected_output": row["solution"],
                "generated_output": generated_text,
                "expected_action": expected_action,
                "generated_action": generated_action,
                "generation_correct": generation_ok,
                "format_correct": format_ok,
            }
        )

    total = len(results)
    generation_acc = generation_correct / total
    format_acc = strict_format_correct / total

    print("=" * 80)
    print(f"Examples evaluated                 : {total}")
    print(f"Generation action accuracy         : {generation_acc:.2%} ({generation_correct}/{total})")
    print(f"Strict format accuracy             : {format_acc:.2%} ({strict_format_correct}/{total})")
    print("=" * 80)

    for result in results[: args.show_examples]:
        user_text = result["messages"][-1]["content"] if result["messages"] else ""
        print("-" * 80)
        print(f"Case                              : {result['name']}")
        print(f"User message                      : {user_text}")
        print(f"Expected output                   : {result['expected_output']}")
        print(f"Generated output                  : {result['generated_output']}")
        print(f"Generated action                  : {result['generated_action']}")
        print(f"Generation correct                : {result['generation_correct']}")
        print(f"Strict format correct             : {result['format_correct']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
