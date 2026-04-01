#!/usr/bin/env python3
"""Plot numeric metrics from ms-swift train.log."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from swift_metrics_utils import load_metric_history, metric_columns, safe_metric_name, smooth_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Training output directory that contains train.log.")
    parser.add_argument("--save-dir", default=None, help="Optional directory for PNG outputs.")
    return parser.parse_args()


def plot_all_metrics(output_dir: Path, save_dir: Path) -> list[Path]:
    hist = load_metric_history(output_dir / "train.log")
    if hist.empty:
        raise RuntimeError(f"No metric payloads found in {output_dir / 'train.log'}")

    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for metric in metric_columns(hist):
        metric_hist = hist.dropna(subset=[metric])[["plot_step", metric]].copy()
        if len(metric_hist) < 2:
            continue

        metric_hist["smoothed"] = smooth_series(metric_hist[metric])

        plt.figure(figsize=(8, 5))
        plt.plot(metric_hist["plot_step"], metric_hist[metric], label="original", alpha=0.45)
        plt.plot(metric_hist["plot_step"], metric_hist["smoothed"], label="smoothed", linewidth=2.0)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(f"training {metric}")
        plt.grid(alpha=0.25)
        plt.legend()

        save_path = save_dir / f"{safe_metric_name(metric)}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
        plt.close()
        saved_paths.append(save_path)

    if not saved_paths:
        raise RuntimeError("No plottable metric series found in train.log")
    return saved_paths


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "plots"

    saved_paths = plot_all_metrics(output_dir, save_dir)
    print(f"Saved {len(saved_paths)} metric plot(s) to: {save_dir}")
    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
