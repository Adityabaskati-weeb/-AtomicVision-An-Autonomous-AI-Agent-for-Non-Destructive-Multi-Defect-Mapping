"""Render final submission graphs for AtomicVision blog/readme artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
METRICS_PATH = DOCS / "hard-recall-micro-repair-metrics.json"
COMPARISON_OUTPUT = DOCS / "final-model-comparison.png"
SWEEP_OUTPUT = DOCS / "final-checkpoint-sweep.png"


BG = "#fcfcfe"
TEXT = "#172033"
MUTED = "#5e6b7a"
GRID = "#d8dee8"
PREV = "#8a94a6"
CURRENT = "#2f6fed"
MEDIUM = "#00a889"
HARD = "#7c4dff"
WARNING = "#c95f1d"


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def setup_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.65)
    ax.set_axisbelow(True)


def render_comparison(metrics: dict) -> None:
    base = metrics["candidates"]["base"]
    current = metrics["candidates"]["checkpoint-1"]
    labels = ["Medium", "Hard"]
    reward_prev = [base["medium_reward"], base["hard_reward"]]
    reward_curr = [current["medium_reward"], current["hard_reward"]]
    f1_prev = [base["medium_f1"], base["hard_f1"]]
    f1_curr = [current["medium_f1"], current["hard_f1"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), dpi=180, facecolor=BG)
    fig.suptitle(
        "AtomicVision Final Model Comparison",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
        x=0.06,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.06,
        0.905,
        "Held-out strict evaluation on 32 medium and 32 hard episodes. Higher is better.",
        fontsize=11.5,
        color=MUTED,
    )

    for ax in axes:
        setup_axes(ax)

    x = range(len(labels))
    width = 0.34

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], reward_prev, width, color=PREV, label="Previous best")
    ax.bar([i + width / 2 for i in x], reward_curr, width, color=CURRENT, label="Current best")
    ax.set_title("Reward", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax.set_xticks(list(x), labels, color=TEXT)
    ax.set_ylabel("Mean held-out reward", color=TEXT, fontsize=11)
    ax.set_ylim(4.35, 4.80)
    for i, (prev, curr) in enumerate(zip(reward_prev, reward_curr)):
        ax.text(i - width / 2, prev + 0.012, f"{prev:.4f}", ha="center", va="bottom", color=TEXT, fontsize=10)
        ax.text(i + width / 2, curr + 0.012, f"{curr:.4f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
    ax.annotate(
        "+0.0231 on hard",
        xy=(1 + width / 2, reward_curr[1]),
        xytext=(1.12, 4.765),
        arrowprops=dict(arrowstyle="->", color=HARD, lw=1.5),
        color=HARD,
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], f1_prev, width, color=PREV, label="Previous best")
    ax.bar([i + width / 2 for i in x], f1_curr, width, color=CURRENT, label="Current best")
    ax.set_title("F1", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax.set_xticks(list(x), labels, color=TEXT)
    ax.set_ylabel("Mean held-out F1", color=TEXT, fontsize=11)
    ax.set_ylim(0.77, 0.835)
    for i, (prev, curr) in enumerate(zip(f1_prev, f1_curr)):
        ax.text(i - width / 2, prev + 0.0022, f"{prev:.4f}", ha="center", va="bottom", color=TEXT, fontsize=10)
        ax.text(i + width / 2, curr + 0.0022, f"{curr:.4f}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
    ax.annotate(
        "+0.0045 on hard",
        xy=(1 + width / 2, f1_curr[1]),
        xytext=(1.06, 0.832),
        arrowprops=dict(arrowstyle="->", color=HARD, lw=1.5),
        color=HARD,
        fontsize=10.5,
        fontweight="bold",
    )

    fig.text(
        0.06,
        0.03,
        "The final published adapter improves the hard slice while leaving medium unchanged and preserving perfect strict execution.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.03, 0.07, 1.0, 0.83])
    fig.savefig(COMPARISON_OUTPUT, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def render_sweep(metrics: dict) -> None:
    candidates = metrics["candidates"]
    order = ["base", "checkpoint-1", "checkpoint-2", "checkpoint-4"]
    labels = ["Base", "Ckpt-1", "Ckpt-2", "Ckpt-4"]
    medium_reward = [candidates[key]["medium_reward"] for key in order]
    hard_reward = [candidates[key]["hard_reward"] for key in order]
    medium_f1 = [candidates[key]["medium_f1"] for key in order]
    hard_f1 = [candidates[key]["hard_f1"] for key in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=180, facecolor=BG)
    fig.suptitle(
        "Checkpoint Sweep: Why Early Stopping Won",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
        x=0.06,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.06,
        0.90,
        "Targeted hard-recall repair only helped at checkpoint-1; later checkpoints gave the gain back.",
        fontsize=11.5,
        color=MUTED,
    )

    for ax in axes:
        setup_axes(ax)

    x = list(range(len(labels)))

    ax = axes[0]
    ax.plot(x, medium_reward, marker="o", color=MEDIUM, linewidth=2.6, markersize=7, label="Medium reward")
    ax.plot(x, hard_reward, marker="o", color=HARD, linewidth=2.6, markersize=7, label="Hard reward")
    ax.scatter([1], [hard_reward[1]], s=120, color=CURRENT, zorder=5, label="Promoted winner")
    ax.set_title("Reward sweep", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax.set_xticks(x, labels, color=TEXT)
    ax.set_ylabel("Mean held-out reward", color=TEXT, fontsize=11)
    ax.set_ylim(4.38, 4.76)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.annotate(
        "Best hard reward",
        xy=(1, hard_reward[1]),
        xytext=(1.35, 4.745),
        arrowprops=dict(arrowstyle="->", color=CURRENT, lw=1.5),
        color=CURRENT,
        fontsize=10.5,
        fontweight="bold",
    )
    ax.annotate(
        "Medium regressed here",
        xy=(2, medium_reward[2]),
        xytext=(2.25, 4.405),
        arrowprops=dict(arrowstyle="->", color=WARNING, lw=1.5),
        color=WARNING,
        fontsize=10,
        fontweight="bold",
    )

    ax = axes[1]
    ax.plot(x, medium_f1, marker="o", color=MEDIUM, linewidth=2.6, markersize=7, label="Medium F1")
    ax.plot(x, hard_f1, marker="o", color=HARD, linewidth=2.6, markersize=7, label="Hard F1")
    ax.scatter([1], [hard_f1[1]], s=120, color=CURRENT, zorder=5, label="Promoted winner")
    ax.set_title("F1 sweep", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax.set_xticks(x, labels, color=TEXT)
    ax.set_ylabel("Mean held-out F1", color=TEXT, fontsize=11)
    ax.set_ylim(0.776, 0.826)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.annotate(
        "Best hard F1",
        xy=(1, hard_f1[1]),
        xytext=(1.35, 0.824),
        arrowprops=dict(arrowstyle="->", color=CURRENT, lw=1.5),
        color=CURRENT,
        fontsize=10.5,
        fontweight="bold",
    )

    fig.text(
        0.06,
        0.03,
        "This was not a 'train longer' story. It was a 'train more narrowly, then stop early' story.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.03, 0.07, 1.0, 0.81])
    fig.savefig(SWEEP_OUTPUT, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    render_comparison(metrics)
    render_sweep(metrics)
    print(COMPARISON_OUTPUT)
    print(SWEEP_OUTPUT)


if __name__ == "__main__":
    main()
