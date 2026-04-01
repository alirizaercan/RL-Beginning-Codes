#!/usr/bin/env python3
"""Watch metrics for the ep_3500 A4000 run."""

from __future__ import annotations

import sys

from watch_training_metrics import main


if __name__ == "__main__":
    sys.argv = [
        "watch_training_metrics_ep_3500.py",
        "--output-dir",
        "saves/qwen25_05b_base_full_ft_ep_3500_a4000",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
