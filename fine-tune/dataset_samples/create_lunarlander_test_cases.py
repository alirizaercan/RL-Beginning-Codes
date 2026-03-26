#!/usr/bin/env python3
"""Create a balanced and diverse LunarLander test set from the HF dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Ali2023kosemen/lunar_lander_270_reward"
DEFAULT_OUTPUT_PATH = Path("fine-tune/dataset_samples/lunarlander_test_cases.json")
DEFAULT_LOCAL_INPUT = Path("fine-tune/dataset_samples/lunarlander_dataset.json")
STATE_PATTERN = re.compile(
    r"State:\s*\[x=([-0-9.]+), y=([-0-9.]+), vx=([-0-9.]+), vy=([-0-9.]+), "
    r"angle=([-0-9.]+), angular_vel=([-0-9.]+), left_leg=([-0-9.]+), right_leg=([-0-9.]+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Optional local JSON file to use instead of downloading from HF.",
    )
    parser.add_argument(
        "--cases-per-action",
        type=int,
        default=3,
        help="How many diverse cases to keep for each action.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def extract_state(prompt_text: str) -> tuple[str, list[float]]:
    match = STATE_PATTERN.search(prompt_text)
    if not match:
        raise ValueError(f"Could not parse state from prompt: {prompt_text}")

    values = [float(x) for x in match.groups()]
    state_text = (
        f"[x={values[0]:.4f}, y={values[1]:.4f}, vx={values[2]:.4f}, vy={values[3]:.4f}, "
        f"angle={values[4]:.4f}, angular_vel={values[5]:.4f}, left_leg={values[6]:.4f}, right_leg={values[7]:.4f}]"
    )
    return state_text, values


def extract_action(answer_text: str) -> int:
    match = re.search(r"Action:\s*([0-3])", answer_text)
    if not match:
        raise ValueError(f"Could not parse action from answer: {answer_text}")
    return int(match.group(1))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def select_diverse_cases(cases: list[dict[str, object]], k: int) -> list[dict[str, object]]:
    if len(cases) <= k:
        return cases

    center = [sum(case["state_values"][i] for case in cases) / len(cases) for i in range(8)]
    selected = [max(cases, key=lambda case: euclidean_distance(case["state_values"], center))]
    remaining = [case for case in cases if case is not selected[0]]

    while len(selected) < k and remaining:
        best_case = max(
            remaining,
            key=lambda case: min(
                euclidean_distance(case["state_values"], selected_case["state_values"]) for selected_case in selected
            ),
        )
        selected.append(best_case)
        remaining.remove(best_case)

    return selected


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    input_json = Path(args.input_json) if args.input_json else None

    if input_json is not None:
        rows = json.loads(input_json.read_text(encoding="utf-8"))
        print(f"Using local JSON file: {input_json}")
    else:
        try:
            ds = load_dataset(args.dataset_id, split="train")
            rows = list(ds)
            print(f"Using HF dataset: {args.dataset_id}")
        except Exception as exc:
            if DEFAULT_LOCAL_INPUT.exists():
                rows = json.loads(DEFAULT_LOCAL_INPUT.read_text(encoding="utf-8"))
                print(f"HF download failed, falling back to local JSON: {DEFAULT_LOCAL_INPUT}")
                print(f"Reason: {exc}")
            else:
                raise

    grouped_cases: dict[int, list[dict[str, object]]] = {0: [], 1: [], 2: [], 3: []}

    for index, row in enumerate(rows):
        conversations = row["conversations"]
        prompt_text = conversations[0]["value"]
        answer_text = conversations[1]["value"]

        action_id = extract_action(answer_text)
        state_text, state_values = extract_state(prompt_text)

        grouped_cases[action_id].append(
            {
                "name": f"dataset_row_{index}",
                "state": state_text,
                "state_values": state_values,
                "expected_action": action_id,
                "expected_text": answer_text,
                "source_index": index,
            }
        )

    selected_cases: list[dict[str, object]] = []
    for action_id in range(4):
        diverse_cases = select_diverse_cases(grouped_cases[action_id], args.cases_per_action)
        for idx, case in enumerate(diverse_cases, start=1):
            selected_cases.append(
                {
                    "name": f"action_{action_id}_case_{idx}",
                    "state": case["state"],
                    "expected_action": case["expected_action"],
                    "expected_text": case["expected_text"],
                    "source_index": case["source_index"],
                }
            )

    output_path.write_text(json.dumps(selected_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset id: {args.dataset_id}")
    print(f"Saved {len(selected_cases)} test cases to: {output_path}")
    for case in selected_cases:
        print(case["name"], "-> expected action:", case["expected_action"], "| source row:", case["source_index"])


if __name__ == "__main__":
    main()
