#!/usr/bin/env python
"""
Extract step-loss from training logs and plot a blue-themed line chart.
Supports two modes:
1. Read stdout / log file and parse loss via regex;
2. Fallback to built-in data if no valid log is provided.

Usage:
    python plot_loss.py
    python plot_loss.py --log log.txt --output loss_curve.png
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline
except ImportError as e:
    raise ImportError(
        f"Missing dependencies: {e}\n"
        "Please run: pip install matplotlib numpy scipy"
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Built-in data (extracted from training logs)
# ──────────────────────────────────────────────────────────────────────────────
BUILT_IN_STEPS = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200
]
BUILT_IN_LOSSES = [
    2.205, 1.469, 1.416, 1.415, 1.409, 1.362, 1.319, 1.294, 1.266, 1.228,
    1.202, 0.8142, 0.6303, 0.6149, 0.5908, 0.5706, 0.5668, 0.5622, 0.5454, 0.5289
]


def parse_log(raw_text: str):
    """
    Parse step and loss from text in `'loss': '1.234'` format.
    Returns (steps, losses) lists; returns None if parsing fails.
    """
    pattern = re.compile(
        r"'loss':\s*'([0-9.eE+-]+)'"
    )
    matches = pattern.findall(raw_text)
    if not matches:
        return None, None

    losses = [float(m) for m in matches]
    # Assume logs are printed at fixed intervals (e.g. every 10 steps)
    steps = list(range(1, len(losses) + 1))
    return steps, losses


def smooth_curve(x, y, num_points=500, k=3):
    """
    Smooth discrete points using B-spline.
    k: spline order (3 = cubic), requires num_points >= k+1.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Downgrade k if too few points
    if len(x) <= k:
        k = max(1, len(x) - 1)

    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, num_points)

    spl = make_interp_spline(t, y, k=k)
    y_smooth = spl(t_new)

    x_smooth = np.interp(t_new, t, x)
    return x_smooth, y_smooth


def plot_loss_curve(
    steps,
    losses,
    output_path: str = "step_loss.png",
    title: str = "Training Loss Curve",
    xlabel: str = "Step",
    ylabel: str = "Loss",
    smooth: bool = True,
    smooth_points: int = 500,
    figsize=(10, 6),
    dpi=200,
):
    """
    Plot a blue-themed line chart with gradient fill.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color palette (blue theme)
    line_color = "#1565C0"
    glow_color = "#64B5F6"
    fill_top = "#90CAF9"
    fill_bottom = "#E3F2FD"
    point_color = "#0D47A1"
    spine_color = "#455A64"
    text_color = "#263238"

    x = np.array(steps, dtype=float)
    y = np.array(losses, dtype=float)

    if smooth and len(x) >= 4:
        x_smooth, y_smooth = smooth_curve(x, y, num_points=smooth_points, k=3)
    else:
        x_smooth, y_smooth = x, y

    # 1) Glow layer
    for width, alpha in [(8, 0.08), (5, 0.12), (2, 0.25)]:
        ax.plot(
            x_smooth, y_smooth,
            color=glow_color,
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
            zorder=1,
        )

    # 2) Main line
    ax.plot(
        x_smooth, y_smooth,
        color=line_color,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=3,
        label="Smoothed Loss",
    )

    # 3) Gradient fill
    y_min = y_smooth.min()
    y_max = y_smooth.max()
    y_range = y_max - y_min if y_max != y_min else 1.0

    ax.fill_between(
        x_smooth, y_smooth, y_smooth.min() - 0.05 * y_range,
        color=fill_top,
        alpha=0.18,
        zorder=2,
    )

    # 4) Scatter points
    ax.scatter(
        x, y,
        color=point_color,
        s=40,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
        label="Raw Loss",
    )

    # 5) Annotate first and last values
    for xi, yi in [(x[0], y[0]), (x[-1], y[-1])]:
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            textcoords="offset points",
            xytext=(0, 14 if yi == y[0] else -20),
            ha="center",
            fontsize=9,
            color=point_color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=line_color,
                alpha=0.9,
            ),
        )

    # 6) Title and labels
    ax.set_title(title, fontsize=16, fontweight="bold", color=text_color, pad=16)
    ax.set_xlabel(xlabel, fontsize=12, color=text_color, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, color=text_color, labelpad=8)

    # 7) Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", colors=text_color, labelsize=10)

    y_lower = max(0, y.min() - 0.15 * (y.max() - y.min()))
    ax.set_ylim(bottom=y_lower)

    x_pad = (x.max() - x.min()) * 0.03
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)

    # Grid
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
        color="#90A4AE",
        axis="y",
    )

    # Legend
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#CFD8DC",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curve")
    parser.add_argument(
        "--log", type=str, default=None,
        help="Training log file path (plain text)"
    )
    parser.add_argument(
        "--output", type=str, default="figs/step_loss.png",
        help="Output image path"
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Disable curve smoothing"
    )
    args = parser.parse_args()

    steps, losses = None, None

    if args.log and Path(args.log).exists():
        raw = Path(args.log).read_text(encoding="utf-8")
        steps, losses = parse_log(raw)
        if steps:
            print(f"Parsed {len(losses)} loss records from log")

    if steps is None:
        print("No valid log provided, using built-in training data")
        steps, losses = BUILT_IN_STEPS, BUILT_IN_LOSSES

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(
        steps,
        losses,
        output_path=str(out_path),
        smooth=not args.no_smooth,
    )


if __name__ == "__main__":
    main()
