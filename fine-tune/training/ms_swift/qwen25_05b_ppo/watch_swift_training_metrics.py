#!/usr/bin/env python3
"""Watch ms-swift train.log and regenerate plots when metrics change."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from plot_swift_training_metrics import plot_all_metrics
from swift_metrics_utils import load_metric_history, preferred_metric_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Training output directory.")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds.")
    return parser.parse_args()


def checkpoint_names(output_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_path = output_dir / "train.log"
    plot_dir = output_dir / "plots"

    print(f"Watching: {output_dir}")
    print(f"Polling every {args.interval} seconds")

    last_size = -1
    last_checkpoints: tuple[str, ...] = ()

    while True:
        if not log_path.exists():
            print("train.log not found yet, waiting...")
            time.sleep(args.interval)
            continue

        current_size = log_path.stat().st_size
        current_checkpoints = checkpoint_names(output_dir)

        if current_size != last_size or current_checkpoints != last_checkpoints:
            hist = load_metric_history(log_path)
            if not hist.empty:
                latest = hist.iloc[-1]
                print("-" * 80)
                print(f"Known checkpoints : {', '.join(current_checkpoints) if current_checkpoints else 'none'}")
                for metric in preferred_metric_columns(hist):
                    if metric in latest and latest[metric] == latest[metric]:
                        print(f"Latest {metric:<24}: {float(latest[metric]):.6f}")
                saved = plot_all_metrics(output_dir, plot_dir)
                print(f"Updated plots     : {len(saved)} file(s) in {plot_dir}")

            last_size = current_size
            last_checkpoints = current_checkpoints

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
