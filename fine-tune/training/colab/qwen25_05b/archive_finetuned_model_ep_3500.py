#!/usr/bin/env python3
"""Archive the ep_3500 A4000 run."""

from __future__ import annotations

import sys

from archive_finetuned_model import main


if __name__ == "__main__":
    sys.argv = [
        "archive_finetuned_model_ep_3500.py",
        "--output-dir",
        "saves/qwen25_05b_base_full_ft_ep_3500_a4000",
        "--archive-base",
        "qwen25_05b_base_full_ft_ep_3500_a4000",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
