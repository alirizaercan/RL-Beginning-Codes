#!/usr/bin/env python3
"""Archive the latest fine-tuned model directory into a zip file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def resolve_model_dir(output_dir: Path) -> Path:
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    candidate_files = ["model.safetensors", "pytorch_model.bin", "config.json"]
    if output_dir.exists() and any((output_dir / name).exists() for name in candidate_files):
        return output_dir

    raise FileNotFoundError(f"No saved model files found under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="saves/qwen25_05b_full_ft_lunarlander_a4000",
        help="LLaMA-Factory output directory.",
    )
    parser.add_argument(
        "--archive-base",
        default="qwen25_05b_lunarlander_full_ft_a4000",
        help="Base name for the archive output (without extension).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = resolve_model_dir(output_dir)
    archive_base = Path(args.archive_base).resolve()

    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(model_dir))
    print(f"Archived model dir: {model_dir}")
    print(f"Zip file: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
