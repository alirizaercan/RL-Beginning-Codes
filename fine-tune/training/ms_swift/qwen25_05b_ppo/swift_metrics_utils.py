#!/usr/bin/env python3
"""Shared metric parsing helpers for ms-swift RM/PPO logs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _to_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if NUMERIC_RE.fullmatch(stripped):
            return float(stripped)
    return None


def extract_metric_payloads(log_text: str) -> list[dict[str, float]]:
    payloads: list[dict[str, float]] = []
    for line in log_text.splitlines():
        if "{" not in line or "}" not in line:
            continue
        candidate = line[line.find("{") : line.rfind("}") + 1]
        parsed = None
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
        if not isinstance(parsed, dict):
            continue

        clean = {}
        for key, value in parsed.items():
            number = _to_float(value)
            if number is not None:
                clean[str(key)] = number
        if clean:
            payloads.append(clean)
    return payloads


def load_metric_history(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    payloads = extract_metric_payloads(log_path.read_text(encoding="utf-8", errors="ignore"))
    rows = []
    for index, payload in enumerate(payloads, start=1):
        row = dict(payload)
        plot_step = None
        for key in ("step", "global_step", "current_steps"):
            if key in row:
                plot_step = row[key]
                break
        row["plot_step"] = plot_step if plot_step is not None else float(index)
        rows.append(row)
    return pd.DataFrame(rows)


def smooth_series(series: pd.Series, weight: float = 0.6) -> pd.Series:
    values = list(series.astype(float))
    if not values:
        return pd.Series(dtype=float)
    smoothed = [values[0]]
    last = values[0]
    for value in values[1:]:
        last = last * weight + (1.0 - weight) * value
        smoothed.append(last)
    return pd.Series(smoothed, index=series.index)


def metric_columns(hist: pd.DataFrame) -> list[str]:
    excluded = {"plot_step", "step", "global_step", "current_steps"}
    return [column for column in hist.columns if column not in excluded]


def preferred_metric_columns(hist: pd.DataFrame) -> list[str]:
    preferred_tokens = ("reward", "acc", "loss", "kl", "lr", "value", "policy")
    columns = metric_columns(hist)
    preferred = [col for col in columns if any(token in col.lower() for token in preferred_tokens)]
    return preferred or columns


def safe_metric_name(metric: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", metric.strip())
    return safe.strip("_") or "metric"
