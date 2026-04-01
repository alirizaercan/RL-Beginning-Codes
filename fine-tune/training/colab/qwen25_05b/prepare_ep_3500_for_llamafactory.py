#!/usr/bin/env python3
"""Prepare Ali2023kosemen/ep_3500 for LLaMA-Factory."""

from __future__ import annotations

import sys

from prepare_lunarlander_dataset_for_llamafactory import main


if __name__ == "__main__":
    sys.argv = [
        "prepare_ep_3500_for_llamafactory.py",
        "--dataset-id",
        "Ali2023kosemen/ep_3500",
        "--dataset-slug",
        "ep_3500",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
