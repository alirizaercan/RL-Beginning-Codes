#!/usr/bin/env python3
"""Prepare Ali2023kosemen/action_50_000 for LLaMA-Factory."""

from __future__ import annotations

import sys

from prepare_lunarlander_dataset_for_llamafactory import main


if __name__ == "__main__":
    sys.argv = [
        "prepare_action_50_000_for_llamafactory.py",
        "--dataset-id",
        "Ali2023kosemen/action_50_000",
        "--dataset-slug",
        "action_50_000",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
