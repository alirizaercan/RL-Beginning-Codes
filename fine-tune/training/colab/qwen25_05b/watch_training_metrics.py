#!/usr/bin/env python3
"""Watch LLaMA-Factory training progress and redraw metrics after new checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from plot_training_loss import main as plot_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="saves/qwen25_05b_base_full_ft_lunarlander_a4000",
        help="LLaMA-Factory output directory.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    return parser.parse_args()


def read_history(log_path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in log_path.open(encoding="utf-8") if line.strip()]
    return pd.DataFrame(rows)


def latest_value(hist: pd.DataFrame, column: str) -> float | None:
    if column not in hist.columns:
        return None
    valid = hist.dropna(subset=[column])
    if valid.empty:
        return None
    return float(valid.iloc[-1][column])


def checkpoint_names(output_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()))


def redraw_plot(output_dir: Path) -> None:
    plot_path = output_dir / "training_loss.png"
    plot_main_args = ["--output-dir", str(output_dir), "--save-path", str(plot_path)]
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["plot_training_loss.py", *plot_main_args]
        plot_main()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_path = output_dir / "trainer_log.jsonl"

    print(f"Watching: {output_dir}")
    print(f"Polling every {args.interval} seconds")

    last_checkpoints: tuple[str, ...] = ()
    last_log_size = -1

    while True:
        if not log_path.exists():
            print("trainer_log.jsonl not found yet, waiting...")
            time.sleep(args.interval)
            continue

        current_log_size = log_path.stat().st_size
        current_checkpoints = checkpoint_names(output_dir)

        if current_log_size != last_log_size or current_checkpoints != last_checkpoints:
            hist = read_history(log_path)
            latest_train_loss = latest_value(hist, "loss")
            latest_eval_loss = latest_value(hist, "eval_loss")
            latest_eval_accuracy = latest_value(hist, "eval_accuracy")
            latest_grad_norm = latest_value(hist, "grad_norm")
            latest_learning_rate = latest_value(hist, "learning_rate")

            print("-" * 80)
            print(f"Known checkpoints : {', '.join(current_checkpoints) if current_checkpoints else 'none'}")
            if latest_train_loss is not None:
                print(f"Latest train loss : {latest_train_loss:.6f}")
            if latest_eval_loss is not None:
                print(f"Latest eval loss  : {latest_eval_loss:.6f}")
            if latest_eval_accuracy is not None:
                print(f"Latest eval acc   : {latest_eval_accuracy:.2%}")
            if latest_grad_norm is not None:
                print(f"Latest grad norm  : {latest_grad_norm:.6f}")
            if latest_learning_rate is not None:
                print(f"Latest lr         : {latest_learning_rate:.6e}")

            redraw_plot(output_dir)
            print(f"Updated plot      : {output_dir / 'training_loss.png'}")

            last_log_size = current_log_size
            last_checkpoints = current_checkpoints

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
