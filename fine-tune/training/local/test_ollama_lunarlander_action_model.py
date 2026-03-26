#!/usr/bin/env python3
"""Test a local Ollama LunarLander action model with multiple prompt styles."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ACTION_MAP = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine",
}


@dataclass(frozen=True)
class TestCase:
    name: str
    state: str


TEST_CASES = [
    TestCase(
        "high_center",
        "[x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]",
    ),
    TestCase(
        "left_tilt",
        "[x=-0.0500, y=0.4000, vx=-0.0200, vy=-0.1500, angle=0.3000, angular_vel=0.2500, left_leg=0.0000, right_leg=0.0000]",
    ),
    TestCase(
        "right_tilt",
        "[x=0.3000, y=0.9000, vx=0.3500, vy=-0.2500, angle=-0.2200, angular_vel=-0.1500, left_leg=0.0000, right_leg=0.0000]",
    ),
    TestCase(
        "touchdown",
        "[x=0.0000, y=0.0800, vx=0.0000, vy=-0.0200, angle=0.0000, angular_vel=0.0000, left_leg=1.0000, right_leg=1.0000]",
    ),
    TestCase(
        "left_far",
        "[x=-0.4200, y=0.6500, vx=-0.4800, vy=-0.2800, angle=0.3300, angular_vel=0.2400, left_leg=0.0000, right_leg=0.0000]",
    ),
    TestCase(
        "descending_main",
        "[x=0.0200, y=0.2500, vx=0.0100, vy=-0.0500, angle=0.0100, angular_vel=0.0000, left_leg=1.0000, right_leg=1.0000]",
    ),
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


def query_model(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }
    request = Request(
        url="http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - runtime diagnostic
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:  # pragma: no cover - runtime diagnostic
        raise RuntimeError(f"Could not reach Ollama at http://localhost:11434: {exc}") from exc

    message = body.get("message", {})
    return (message.get("content") or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lunarlander-action-v3", help="Ollama model name.")
    parser.add_argument(
        "--prompt-style",
        choices=["simple", "strict", "structured", "all"],
        default="all",
        help="Prompt template to use.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max output tokens.")
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print results as JSON instead of plain text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    styles = list(PROMPT_BUILDERS) if args.prompt_style == "all" else [args.prompt_style]
    results: list[dict[str, object]] = []

    for style in styles:
        builder = PROMPT_BUILDERS[style]
        for case in TEST_CASES:
            prompt = builder(case.state)
            try:
                raw = query_model(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # pragma: no cover - runtime diagnostic
                results.append(
                    {
                        "prompt_style": style,
                        "case": case.name,
                        "state": case.state,
                        "raw_output": "",
                        "parsed_action": None,
                        "parsed_label": None,
                        "error": str(exc),
                    }
                )
                continue

            action_id = parse_action(raw) if raw else None
            results.append(
                {
                    "prompt_style": style,
                    "case": case.name,
                    "state": case.state,
                    "raw_output": raw,
                    "parsed_action": action_id,
                    "parsed_label": ACTION_MAP.get(action_id),
                    "error": None,
                }
            )

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print(f"Model: {args.model}")
    print(f"Prompt styles: {', '.join(styles)}")
    print("-" * 100)
    for item in results:
        print(f"[{item['prompt_style']}] {item['case']}")
        print(f"state        : {item['state']}")
        print(f"raw output   : {item['raw_output']!r}")
        print(f"parsed action: {item['parsed_action']} ({item['parsed_label']})")
        if item["error"]:
            print(f"error        : {item['error']}")
        print("-" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
