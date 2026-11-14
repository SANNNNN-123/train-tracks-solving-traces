"""Plot Hugging Face Trainer metrics from a saved Modal application log.

Run: python scripts/plot_training_log.py output/training/training.log
"""

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def training_points(content):
    points = []
    for match in re.finditer(r"\{[^{}\r\n]*\}", content):
        try:
            item = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(item, dict) and "loss" in item:
            try:
                item["loss"] = float(item["loss"])
                item["epoch"] = float(item["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append(item)
    return points


def report_value(content, name):
    match = re.search(rf'"{re.escape(name)}"\s*:\s*([-+\d.eE]+)', content)
    return float(match.group(1)) if match else None


def report_text(content, name):
    match = re.search(rf'"{re.escape(name)}"\s*:\s*"([^"]+)"', content)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    content = args.log.read_text(encoding="utf-8-sig", errors="replace")
    points = training_points(content)
    before = report_value(content, "development_loss_before")
    after = report_value(content, "development_loss_after")
    if not points and (before is None or after is None):
        raise SystemExit("No training metrics found in the log")

    output = args.log.parent
    saved = []
    if points:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        epochs = [p.get("epoch", i + 1) for i, p in enumerate(points)]
        ax.plot(epochs, [p["loss"] for p in points], color="#2166ac", linewidth=1.8)
        ax.set(xlabel="Epoch", ylabel="Training loss", title="Train Tracks action policy: training loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output / "training-loss.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        saved.append(str(path))

    if before is not None and after is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(["Before", "After"], [before, after], color=["#8da0cb", "#2ca25f"])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(ylabel="Development loss", title="Held-out puzzle actions")
        ax.set_ylim(0, max(before, after) * 1.18)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output / "development-loss.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        saved.append(str(path))

    summary = {
        "run_id": report_text(content, "run_id"),
        "model": report_text(content, "model"),
        "train_examples": report_value(content, "train_examples"),
        "development_examples": report_value(content, "development_examples"),
        "optimizer_steps": report_value(content, "optimizer_steps"),
        "train_runtime_seconds": report_value(content, "train_runtime_seconds"),
        "training_loss_points": len(points),
        "first_training_loss": points[0]["loss"] if points else None,
        "last_training_loss": points[-1]["loss"] if points else None,
        "development_loss_before": before,
        "development_loss_after": after,
        "plots": saved,
    }
    path = output / "metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
