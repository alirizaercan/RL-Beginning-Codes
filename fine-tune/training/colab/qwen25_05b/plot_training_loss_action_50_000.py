#!/usr/bin/env python3
"""Plot metrics for the action_50_000 A4000 run."""

from __future__ import annotations

import sys

from plot_training_loss import main


if __name__ == "__main__":
    sys.argv = [
        "plot_training_loss_action_50_000.py",
        "--output-dir",
        "saves/qwen25_05b_base_full_ft_action_50_000_a4000",
        *sys.argv[1:],
    ]
    raise SystemExit(main())
