"""
ASCII plot of training loss from the CSV log written by train.py.

Usage:
    python plot.py
    python plot.py --log checkpoint_log.csv
    python plot.py --width 80 --height 20
"""

import argparse
import csv
import os


def read_log(path: str) -> tuple[list[int], list[float], list[float]]:
    steps, train_l, val_l = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train_l.append(float(row["train_loss"]))
            val_l.append(float(row["val_loss"]))
    return steps, train_l, val_l


def ascii_plot(
    series: list[tuple[str, list[float], str, str]],
    steps: list[int],
    width: int,
    height: int,
) -> None:
    all_vals = [v for _, vs, _, _ in series for v in vs]
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_span = max(y_max - y_min, 1e-8)

    n = len(steps)
    canvas = [[" "] * width for _ in range(height)]
    for _, vals, marker, color in series:
        for i, val in enumerate(vals):
            col = min(width - 1, int(i / n * width))
            row = int((y_max - val) / y_span * (height - 1))
            row = max(0, min(height - 1, row))
            canvas[row][col] = f"{color}{marker}\033[0m"

    for r, row_chars in enumerate(canvas):
        y_label = y_max - r / max(height - 1, 1) * y_span
        print(f"  {y_label:6.3f} |{''.join(row_chars)}")

    step_lo, step_hi = steps[0], steps[-1]
    print(f"         +-{'--' * (width // 2)}")
    print(f"           {step_lo:<{width // 2}}{step_hi:>{width // 2 - 1}} (step)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASCII plot of training log.")
    parser.add_argument("--log",    default="checkpoint_log.csv")
    parser.add_argument("--width",  type=int, default=70)
    parser.add_argument("--height", type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        print("(train.py writes <checkpoint>_log.csv next to the checkpoint)")
        return

    steps, train_l, val_l = read_log(args.log)
    if not steps:
        print("Log file is empty.")
        return

    series = [
        ("train", train_l, "*", "\033[32m"),   # green  * = train
        ("val",   val_l,   "o", "\033[33m"),   # yellow o = val
    ]

    print(f"\nLog: {args.log}  ({len(steps)} eval points)\n")
    ascii_plot(series, steps, args.width, args.height)

    print()
    for name, vals, marker, color in series:
        best      = min(vals)
        best_step = steps[vals.index(best)]
        print(
            f"  {color}{marker}\033[0m {name:<6}  "
            f"final={vals[-1]:.4f}  best={best:.4f} @ step {best_step}"
        )

    delta = val_l[-1] - train_l[-1]
    gap_label = "overfit" if delta > 0.05 else "ok"
    print(f"\n  train/val gap: {delta:+.4f}  ({gap_label})")


if __name__ == "__main__":
    main()
