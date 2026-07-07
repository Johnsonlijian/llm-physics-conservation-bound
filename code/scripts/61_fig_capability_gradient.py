"""Generate Fig. 5 from the released 14-model capability-gradient table.

The public package releases an aggregate leaderboard rather than raw model
answers, because several source rows contain benchmark gold answers or model
responses that should not be redistributed. The released table is sufficient to
reproduce the submission-facing capability-gradient figure.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CSV = PROJECT / "evaluation" / "R06_leaderboard_20260703.csv"
OUT = PROJECT / "figures" / "F26_capability_gradient.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight", dpi=300)


def main() -> None:
    rows = read_rows(CSV)
    rows.sort(key=lambda r: float(r["accuracy"]))

    colors = {
        "panel-4family": "#0072B2",
        "panelx-new": "#009E73",
        "reasoner": "#E69F00",
        "frontier": "#D55E00",
    }
    markers = {"panel-4family": "o", "panelx-new": "s", "reasoner": "^", "frontier": "D"}
    class_labels = {
        "panel-4family": "original open panel",
        "panelx-new": "extended API panel",
        "reasoner": "reasoning model",
        "frontier": "agent-interface arms",
    }
    short = {
        "Qwen7B-Ollama": "Qwen-7B",
        "Qwen14B": "Qwen-14B",
        "KimiK2": "Kimi-K2",
        "DeepSeekV3": "DeepSeek-chat",
        "doubao-seed2": "doubao-seed2",
        "DeepSeek-reasoner": "DeepSeek-reasoner",
        "OpenAI-Codex arm": "OpenAI-Codex arm",
        "deepseek-v4-pro": "DeepSeek-v4-pro",
        "Anthropic-agent arm": "Anthropic-agent arm",
    }
    label_models = set(short)
    offsets = {
        "Qwen7B-Ollama": (9, 12),
        "Qwen14B": (9, -18),
        "KimiK2": (-39, -20),
        "DeepSeekV3": (12, -7),
        "doubao-seed2": (9, 10),
        "DeepSeek-reasoner": (-52, -20),
        "OpenAI-Codex arm": (-28, 23),
        "deepseek-v4-pro": (10, -23),
        "Anthropic-agent arm": (-50, 5),
    }

    apply_style()
    fig, ax = plt.subplots(figsize=(7.05, 4.35), dpi=300)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.16)

    xs = np.array([float(r["accuracy"]) for r in rows])
    ys = np.array([float(r["OR_dc"]) for r in rows])
    if len(rows) > 1:
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xx, coef[0] * xx + coef[1], color="#b8b8b8", lw=1.3, zorder=1)

    seen: set[str] = set()
    for r in rows:
        x = float(r["accuracy"])
        y = float(r["OR_dc"])
        lo = float(r["ci_lo"])
        hi = float(r["ci_hi"])
        cls = r["class"]
        label = class_labels[cls] if cls not in seen else None
        seen.add(cls)
        ax.errorbar(
            x,
            y,
            yerr=[[y - lo], [hi - y]],
            fmt=markers.get(cls, "o"),
            color=colors.get(cls, "#555555"),
            ecolor=colors.get(cls, "#555555"),
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=3,
        )
        if r["model"] in label_models:
            dx, dy = offsets.get(r["model"], (6, 5))
            ax.annotate(
                short[r["model"]],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5.9,
                color="#303030",
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops=dict(arrowstyle="-", color="#bdbdbd", lw=0.45, shrinkA=2, shrinkB=3),
                zorder=4,
            )

    ax.axhline(1.0, ls=(0, (4, 3)), lw=0.9, color="#6f6f6f", zorder=0)
    ax.text(0.602, 1.015, "OR = 1: no per-constraint penalty", ha="right", va="bottom", fontsize=6.2, color="#595959")
    ax.text(
        0.022,
        0.245,
        "Spearman(accuracy, OR) = 0.70, all models\n0.89 among non-floor models (accuracy >= 0.15)",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#303030",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d0d0d0", linewidth=0.6),
    )
    ax.set_xlim(0.0, 0.64)
    ax.set_ylim(0.15, 1.28)
    ax.set_xlabel("model capability (overall judged accuracy on the same 258 physics items)")
    ax.set_ylabel("per-constraint odds ratio OR($d_c$), 95% CI")
    ax.legend(loc="lower right", frameon=True, edgecolor="#d5d5d5", facecolor="white", framealpha=0.95)
    ax.grid(axis="y", color="#e7e7e7", lw=0.45)
    ax.grid(axis="x", color="#eeeeee", lw=0.35)
    save_figure(fig, OUT)
    print(f"[wrote] {OUT} / .pdf / .svg ({len(rows)} models)")


if __name__ == "__main__":
    main()
