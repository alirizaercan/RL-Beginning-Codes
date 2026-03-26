#!/usr/bin/env python3
"""Plot LLaMA-Factory train/eval curves from trainer_log.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="saves/qwen25_05b_base_full_ft_lunarlander_a4000",
        help="LLaMA-Factory output directory.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path to save the PNG plot. Defaults to <output-dir>/training_loss.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_path = output_dir / "trainer_log.jsonl"

    if not log_path.exists():
        raise FileNotFoundError(f"trainer_log.jsonl not found: {log_path}")

    rows = [json.loads(line) for line in log_path.open(encoding="utf-8") if line.strip()]
    hist = pd.DataFrame(rows)
    step_col = "current_steps" if "current_steps" in hist.columns else "step"

    if step_col not in hist.columns:
        raise RuntimeError(f"Step column not found in {log_path}")

    fig, ax1 = plt.subplots(figsize=(11, 5))

    train_hist = pd.DataFrame()
    if "loss" in hist.columns:
        train_hist = hist.dropna(subset=["loss"])[[step_col, "loss"]].copy()
        if not train_hist.empty:
            ax1.plot(train_hist[step_col], train_hist["loss"], marker="o", markersize=2, label="train loss")

    eval_columns = [col for col in hist.columns if col.startswith("eval_") and col.endswith("_loss")]
    best_eval = None
    best_eval_name = None
    for col in eval_columns:
        eval_hist = hist.dropna(subset=[col])[[step_col, col]].copy()
        if eval_hist.empty:
            continue
        label = col.replace("_", " ")
        ax1.plot(eval_hist[step_col], eval_hist[col], marker="s", markersize=3, label=label)
        current_best = float(eval_hist[col].min())
        if best_eval is None or current_best < best_eval:
            best_eval = current_best
            best_eval_name = col

    if train_hist.empty and not eval_columns:
        raise RuntimeError("No train or eval loss values found in trainer_log.jsonl")

    accuracy_cols = [col for col in hist.columns if col in {"eval_accuracy", "accuracy"}]
    best_accuracy = None
    if accuracy_cols:
        ax2 = ax1.twinx()
        for col in accuracy_cols:
            acc_hist = hist.dropna(subset=[col])[[step_col, col]].copy()
            if acc_hist.empty:
                continue
            ax2.plot(
                acc_hist[step_col],
                acc_hist[col],
                linestyle="--",
                marker="^",
                markersize=3,
                label=col.replace("_", " "),
                color="tab:green",
            )
            current_best = float(acc_hist[col].max())
            if best_accuracy is None or current_best > best_accuracy:
                best_accuracy = current_best
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0.0, 1.0)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("LLaMA-Factory Full FT Metrics")
    ax1.grid(alpha=0.3)

    save_path = Path(args.save_path) if args.save_path else output_dir / "training_loss.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()

    print(f"Saved loss plot to: {save_path}")
    if not train_hist.empty:
        print(f"Final train loss: {train_hist['loss'].iloc[-1]:.6f}")
        print(f"Min train loss:   {train_hist['loss'].min():.6f}")
    if best_eval is not None and best_eval_name is not None:
        print(f"Best {best_eval_name}: {best_eval:.6f}")
    else:
        print("No eval loss series found in trainer_log.jsonl.")
    if best_accuracy is not None:
        print(f"Best eval accuracy: {best_accuracy:.2%}")
    else:
        print("No eval accuracy series found in trainer_log.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
