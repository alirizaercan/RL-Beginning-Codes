#!/usr/bin/env python3
"""Prepare Ali2023kosemen/ppo_action_50_000 for true ms-swift PPO.

The source dataset contains:
- messages
- solution

PPO itself uses prompt/query data.
The RM stage additionally needs a preferred answer and a rejected answer, so this
script creates RM rows by pairing the gold `solution` with a deterministic wrong
action. This is the minimum conversion needed to make the dataset usable for
ms-swift PPO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from lunarlander_rl_utils import ACTION_IDS, canonical_solution, normalize_messages, parse_action_id


DEFAULT_SOURCE = "Ali2023kosemen/ppo_action_50_000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="HF dataset id or local .json/.jsonl path.")
    parser.add_argument("--split", default="train", help="HF split name.")
    parser.add_argument("--output-dir", default="data", help="Directory for generated files.")
    parser.add_argument("--eval-size", type=float, default=0.02, help="Eval split ratio.")
    parser.add_argument("--seed", type=int, default=3407, help="Deterministic split seed.")
    return parser.parse_args()


def load_rows(source: str, split: str) -> list[dict[str, Any]]:
    source_path = Path(source)
    if source_path.exists():
        if source_path.suffix == ".jsonl":
            rows = []
            for line in source_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            return rows
        if source_path.suffix == ".json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
            raise ValueError(f"Expected a JSON list in {source_path}")
        raise ValueError(f"Unsupported local source format: {source_path}")

    dataset = load_dataset(source, split=split)
    return [dict(row) for row in dataset]


def synthesize_rejected_solution(solution: str) -> str:
    action_id = parse_action_id(solution)
    if action_id is None:
        raise ValueError(f"Could not parse action from solution: {solution!r}")
    for candidate in ACTION_IDS:
        if candidate != action_id:
            return f"Action: {candidate}"
    raise RuntimeError("Could not synthesize a rejected action.")


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if "messages" not in row:
        raise ValueError(f"Row {index} is missing 'messages': {row!r}")
    if "solution" not in row:
        raise ValueError(f"Row {index} is missing 'solution': {row!r}")

    messages = normalize_messages(list(row["messages"]))
    solution = canonical_solution(row["solution"])

    return {
        "name": f"case_{index}",
        "messages": messages,
        "solution": solution,
        "rejected_solution": synthesize_rejected_solution(solution),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "name": row["name"],
                "messages": [*row["messages"], {"role": "assistant", "content": row["solution"]}],
                "rejected_response": row["rejected_solution"],
            }
        )
    return output


def build_ppo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "name": row["name"],
                "messages": row["messages"],
            }
        )
    return output


def build_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "name": row["name"],
                "messages": row["messages"],
                "solution": row["solution"],
            }
        )
    return output


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.source, args.split)
    if not rows:
        raise RuntimeError("No rows found in the source dataset.")

    normalized = [normalize_row(row, index) for index, row in enumerate(rows, start=1)]
    dataset = Dataset.from_list(normalized)
    split_dataset = dataset.train_test_split(test_size=args.eval_size, seed=args.seed, shuffle=True)

    train_rows = list(split_dataset["train"])
    eval_rows = list(split_dataset["test"])

    rm_train = build_rm_rows(train_rows)
    rm_eval = build_rm_rows(eval_rows)
    ppo_train = build_ppo_rows(train_rows)
    ppo_eval = build_ppo_rows(eval_rows)
    inference_eval = build_eval_rows(eval_rows)

    files = {
        "rm_train": output_dir / "action_50_000_rm_train.jsonl",
        "rm_eval": output_dir / "action_50_000_rm_eval.jsonl",
        "ppo_train": output_dir / "action_50_000_ppo_train.jsonl",
        "ppo_eval": output_dir / "action_50_000_ppo_eval.jsonl",
        "inference_eval": output_dir / "action_50_000_ppo_eval.json",
        "full": output_dir / "action_50_000_ppo_full.jsonl",
    }

    write_jsonl(files["rm_train"], rm_train)
    write_jsonl(files["rm_eval"], rm_eval)
    write_jsonl(files["ppo_train"], ppo_train)
    write_jsonl(files["ppo_eval"], ppo_eval)
    write_json(files["inference_eval"], inference_eval)
    write_jsonl(files["full"], normalized)

    print(f"Saved RM train      : {files['rm_train']}")
    print(f"Saved RM eval       : {files['rm_eval']}")
    print(f"Saved PPO train     : {files['ppo_train']}")
    print(f"Saved PPO eval      : {files['ppo_eval']}")
    print(f"Saved inference eval: {files['inference_eval']}")
    print(f"Saved full dataset  : {files['full']}")
    print(f"Total rows          : {len(normalized)}")
    print(f"Train rows          : {len(train_rows)}")
    print(f"Eval rows           : {len(eval_rows)}")
    print("Sample RM row:")
    print(json.dumps(rm_train[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
