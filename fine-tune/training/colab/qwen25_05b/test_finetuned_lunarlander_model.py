#!/usr/bin/env python3
"""Run post-training inference checks on the fine-tuned LunarLander model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_MAP = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine",
}

TEST_CASES = [
    {
        "name": "high_center",
        "state": "[x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]",
    },
    {
        "name": "left_tilt",
        "state": "[x=-0.0500, y=0.4000, vx=-0.0200, vy=-0.1500, angle=0.3000, angular_vel=0.2500, left_leg=0.0000, right_leg=0.0000]",
    },
    {
        "name": "right_tilt",
        "state": "[x=0.3000, y=0.9000, vx=0.3500, vy=-0.2500, angle=-0.2200, angular_vel=-0.1500, left_leg=0.0000, right_leg=0.0000]",
    },
    {
        "name": "touchdown",
        "state": "[x=0.0000, y=0.0800, vx=0.0000, vy=-0.0200, angle=0.0000, angular_vel=0.0000, left_leg=1.0000, right_leg=1.0000]",
    },
    {
        "name": "left_far",
        "state": "[x=-0.4200, y=0.6500, vx=-0.4800, vy=-0.2800, angle=0.3300, angular_vel=0.2400, left_leg=0.0000, right_leg=0.0000]",
    },
    {
        "name": "descending_main",
        "state": "[x=0.0200, y=0.2500, vx=0.0100, vy=-0.0500, angle=0.0100, angular_vel=0.0000, left_leg=1.0000, right_leg=1.0000]",
    },
]


def build_simple_prompt(state: str) -> str:
    return f"State: {state}. What action should the lander take?"


def build_strict_prompt(state: str) -> str:
    return (
        "Return exactly one action in one line.\n"
        "Valid outputs:\n"
        "Action: 0 (do nothing).\n"
        "Action: 1 (fire left orientation engine).\n"
        "Action: 2 (fire main engine).\n"
        "Action: 3 (fire right orientation engine).\n\n"
        f"State: {state}. What action should the lander take?"
    )


def build_structured_prompt(state: str) -> str:
    return (
        "You are a LunarLander policy model.\n\n"
        "Action space:\n"
        "0 = do nothing\n"
        "1 = fire left orientation engine\n"
        "2 = fire main engine\n"
        "3 = fire right orientation engine\n\n"
        f"Current state: {state}\n\n"
        "Choose the best action and return exactly one line in this format:\n"
        "Action: <id> (<description>)."
    )


PROMPT_BUILDERS: dict[str, Callable[[str], str]] = {
    "simple": build_simple_prompt,
    "strict": build_strict_prompt,
    "structured": build_structured_prompt,
}


def parse_action(text: str) -> int | None:
    normalized = text.strip().lower()
    match = re.search(r"action\s*:\s*([0-3])", normalized)
    if match:
        return int(match.group(1))
    if "fire main engine" in normalized:
        return 2
    if "fire left orientation engine" in normalized or "fire left engine" in normalized:
        return 1
    if "fire right orientation engine" in normalized or "fire right engine" in normalized:
        return 3
    if "do nothing" in normalized:
        return 0
    return None


def resolve_model_dir(output_dir: Path, checkpoint_name: str | None = None) -> Path:
    if checkpoint_name:
        model_dir = output_dir / checkpoint_name
        if model_dir.is_dir():
            return model_dir
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    candidate_files = ["model.safetensors", "pytorch_model.bin", "config.json"]
    if output_dir.exists() and any((output_dir / name).exists() for name in candidate_files):
        return output_dir

    raise FileNotFoundError(f"No saved model files found under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Direct path to a model/checkpoint directory. Overrides --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="saves/qwen25_05b_full_ft_lunarlander_a4000",
        help="LLaMA-Factory output directory.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Specific checkpoint directory name such as checkpoint-200.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional single LunarLander state string to test instead of the built-in suite.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["simple", "strict", "structured", "all"],
        default="all",
        help="Prompt style to use for --state or the built-in suite.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens for generation.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print results as JSON.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save results as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir) if args.model_dir else resolve_model_dir(output_dir, args.checkpoint_name)
    print(f"Resolved model dir: {model_dir}")

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

    def generate_one(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    styles = list(PROMPT_BUILDERS) if args.prompt_style == "all" else [args.prompt_style]
    cases = TEST_CASES if args.state is None else [{"name": "custom_state", "state": args.state}]

    results = []
    for style_name in styles:
        builder = PROMPT_BUILDERS[style_name]
        for case in cases:
            prompt = builder(case["state"])
            raw = generate_one(prompt)
            action_id = parse_action(raw) if raw else None
            results.append(
                {
                    "prompt_style": style_name,
                    "case": case["name"],
                    "state": case["state"],
                    "raw_output": raw,
                    "parsed_action": action_id,
                    "parsed_label": ACTION_MAP.get(action_id),
                }
            )

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON results to: {args.save_json}")

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    df = pd.DataFrame(results)
    print(df[["prompt_style", "case", "parsed_action", "parsed_label", "raw_output"]].to_string(index=False))
    print("-" * 100)
    for item in results:
        print(f"[{item['prompt_style']}] {item['case']}")
        print(f"state        : {item['state']}")
        print(f"raw output   : {item['raw_output']!r}")
        print(f"parsed action: {item['parsed_action']} ({item['parsed_label']})")
        print("-" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
