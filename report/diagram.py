"""Generate final-report figures for Paged-TierKV.

The data in this script is copied from FINAL_EVALUATION_RESULTS.md so the
paper figures are reproducible even when raw Modal logs are not present.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

FIGURE_CACHE = Path(tempfile.gettempdir()) / "tierkv_report_figure_cache"
FIGURE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(FIGURE_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
TEXT_COLOR = "#222222"
GRID_COLOR = "#d0d0d0"
BASELINE_COLOR = "#222222"
TIERKV_COLOR = "#666666"
ACCENT_COLOR = "#8a3a3a"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    }
)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR, width=0.6, length=3)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.45, alpha=0.75)


def save_memory_scaling() -> None:
    requests = np.array([32, 160, 320, 448, 512])
    baseline_mb = np.array([3439.77, 8823.58, 15536.85, 20711.66, np.nan])
    tierkv_mb = np.array([2910.27, 6180.70, 10267.70, 13471.93, 15159.07])
    baseline_gb = baseline_mb / 1024.0
    tierkv_gb = tierkv_mb / 1024.0

    fig, ax = plt.subplots(figsize=(3.33, 2.15), constrained_layout=True)
    ax.plot(
        requests[:-1],
        baseline_gb[:-1],
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        color=BASELINE_COLOR,
        linestyle="-",
        label="Dense baseline",
    )
    ax.plot(
        requests,
        tierkv_gb,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        color=TIERKV_COLOR,
        linestyle="--",
        label="Paged-TierKV",
    )

    oom_y = 21.35
    ax.scatter([512], [oom_y], marker="x", s=42, color=ACCENT_COLOR, linewidths=1.2, zorder=5)
    ax.annotate(
        "OOM",
        xy=(512, oom_y),
        xytext=(465, 22.1),
        arrowprops={"arrowstyle": "->", "color": ACCENT_COLOR, "lw": 0.65},
        fontsize=7.0,
        color=ACCENT_COLOR,
    )
    ax.annotate(
        "TierKV OK",
        xy=(512, tierkv_gb[-1]),
        xytext=(397, 13.2),
        arrowprops={"arrowstyle": "->", "color": TIERKV_COLOR, "lw": 0.65},
        fontsize=7.0,
        color=TEXT_COLOR,
    )

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Peak allocated memory (GB)")
    ax.set_xlim(20, 530)
    ax.set_ylim(0, 23.0)
    ax.set_xticks(requests)
    ax.set_yticks([0, 5, 10, 15, 20])
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4, borderaxespad=0.2)
    fig.savefig(OUT_DIR / "fig_memory_scaling.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_throughput_ratio() -> None:
    requests = np.array([32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448])
    ratios = np.array([0.88, 0.88, 0.86, 0.79, 0.80, 0.84, 0.83, 0.86, 0.68, 0.88, 0.88])

    fig, ax = plt.subplots(figsize=(3.33, 2.15), constrained_layout=True)
    ax.plot(
        requests,
        ratios,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        color=BASELINE_COLOR,
        linestyle="-",
        label="TierKV / baseline",
    )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8, label="Dense baseline")
    ax.scatter([512], [0.885], marker="x", s=36, color=ACCENT_COLOR, linewidths=1.1, zorder=5)
    ax.annotate(
        "baseline OOM\n(no ratio)",
        xy=(512, 0.885),
        xytext=(392, 0.70),
        arrowprops={"arrowstyle": "->", "color": ACCENT_COLOR, "lw": 0.65},
        fontsize=6.8,
        color=ACCENT_COLOR,
    )

    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Decode throughput ratio")
    ax.set_xlim(20, 530)
    ax.set_ylim(0.55, 1.05)
    ax.set_xticks([32, 160, 320, 448, 512])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    style_axes(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.3, borderaxespad=0.2)
    fig.savefig(OUT_DIR / "fig_throughput_ratio.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
