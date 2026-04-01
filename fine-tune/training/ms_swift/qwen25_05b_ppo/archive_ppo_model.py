#!/usr/bin/env python3
"""Archive the merged PPO model directory into a zip file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="exports/qwen25_05b_ppo_action_50_000_a4000_merged",
        help="Merged model directory to archive.",
    )
    parser.add_argument(
        "--archive-base",
        default="qwen25_05b_ppo_action_50_000_a4000_merged",
        help="Archive base name without extension.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    archive_base = Path(args.archive_base).resolve()
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(model_dir))
    print(f"Archived model dir: {model_dir}")
    print(f"Zip file: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
