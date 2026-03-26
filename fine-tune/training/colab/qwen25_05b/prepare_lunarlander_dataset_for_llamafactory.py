#!/usr/bin/env python3
"""Prepare LunarLander data for LLaMA-Factory full SFT.

This version is intentionally stricter than the first draft:
- uses deterministic train/eval split
- rewrites targets into a canonical format: "Action: <id>"
- exports Alpaca-style data for better alignment with a base model
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Ali2023kosemen/lunar_lander_270_reward"
STRICT_PREFIX = (
    "Return exactly one line in this format:\n"
    "Action: <0|1|2|3>\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--eval-size",
        type=float,
        default=0.02,
        help="Fraction of the dataset reserved for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for the deterministic train/eval split.",
    )
    return parser.parse_args()


def parse_action_id(text: str) -> int:
    match = re.search(r"action\s*:\s*([0-3])", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse action id from: {text!r}")
    return int(match.group(1))


def canonicalize_row(row: dict[str, object]) -> dict[str, str]:
    conversations = row["conversations"]
    if len(conversations) < 2:
        raise ValueError(f"Unexpected conversation row: {row!r}")

    user_text = str(conversations[0]["value"]).strip()
    assistant_text = str(conversations[1]["value"]).strip()
    action_id = parse_action_id(assistant_text)

    instruction = STRICT_PREFIX + user_text
    output = f"Action: {action_id}"
    return {"instruction": instruction, "output": output}


def save_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def update_dataset_info(info_path: Path) -> None:
    if info_path.exists():
        dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        dataset_info = {}

    template_columns = {
        "columns": {
            "prompt": "instruction",
            "response": "output",
        }
    }

    dataset_info["lunar_lander_270_reward_train"] = {
        "file_name": "lunar_lander_270_reward_train.json",
        **template_columns,
    }
    dataset_info["lunar_lander_270_reward_eval"] = {
        "file_name": "lunar_lander_270_reward_eval.json",
        **template_columns,
    }
    dataset_info["lunar_lander_270_reward_full"] = {
        "file_name": "lunar_lander_270_reward_full.json",
        **template_columns,
    }

    info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    repo_root = Path.cwd()
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    info_path = data_dir / "dataset_info.json"
    train_path = data_dir / "lunar_lander_270_reward_train.json"
    eval_path = data_dir / "lunar_lander_270_reward_eval.json"
    full_path = data_dir / "lunar_lander_270_reward_full.json"

    ds = load_dataset(args.dataset_id, split="train")
    split_ds = ds.train_test_split(test_size=args.eval_size, seed=args.seed, shuffle=True)

    train_rows = [canonicalize_row(row) for row in split_ds["train"]]
    eval_rows = [canonicalize_row(row) for row in split_ds["test"]]
    full_rows = train_rows + eval_rows

    save_json(train_path, train_rows)
    save_json(eval_path, eval_rows)
    save_json(full_path, full_rows)
    update_dataset_info(info_path)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved eval dataset to:  {eval_path}")
    print(f"Saved full dataset to:  {full_path}")
    print(f"Updated dataset info:   {info_path}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Eval rows:  {len(eval_rows)}")
    print("Sample train row:")
    print(json.dumps(train_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
