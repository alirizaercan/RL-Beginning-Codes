#!/usr/bin/env python3
"""Shared helpers for LunarLander ms-swift RL scripts."""

from __future__ import annotations

import re
from typing import Any


ACTION_IDS = (0, 1, 2, 3)
STRICT_ACTION_RE = re.compile(r"^\s*Action:\s*([0-3])\s*$", re.IGNORECASE)
LOOSE_ACTION_RE = re.compile(r"action\s*:\s*([0-3])", re.IGNORECASE)


def parse_action_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value in ACTION_IDS else None

    text = str(value).strip()
    if text.isdigit():
        candidate = int(text)
        return candidate if candidate in ACTION_IDS else None

    strict_match = STRICT_ACTION_RE.fullmatch(text)
    if strict_match:
        return int(strict_match.group(1))

    loose_match = LOOSE_ACTION_RE.search(text)
    if loose_match:
        return int(loose_match.group(1))

    return None


def canonical_solution(value: Any) -> str:
    action_id = parse_action_id(value)
    if action_id is None:
        raise ValueError(f"Could not parse action id from: {value!r}")
    return f"Action: {action_id}"


def is_strict_action_format(text: str) -> bool:
    return STRICT_ACTION_RE.fullmatch(text.strip()) is not None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message["role"]).strip(),
                "content": str(message["content"]).strip(),
            }
        )
    return normalized


def add_system_prompt(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    if messages and str(messages[0].get("role", "")).strip().lower() == "system":
        return messages
    return [{"role": "system", "content": system_prompt.strip()}, *messages]
