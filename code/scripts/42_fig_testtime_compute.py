"""Figure 3 (hero composite): inference budget moves accuracy along the d_c axis.

Five controlled interventions on the same d_c-heavy 50-item subset, fixed prompt,
fixed judge. NCS-grade four-panel display item:
  (a) Controlled max_tokens sweep in three model families (DeepSeek, Doubao-1.5-pro,
      Qwen-14B): accuracy vs budget step, fitted log-linear slopes, loose beta/e ref.
  (b) Paired direct-vs-CoT by d_c (DeepSeek): CoT lifts accuracy, concentrated at low d_c.
  (c) Reasoner-vs-chat replication in two families (DeepSeek 3 arms; Doubao 2 arms).
  (d) Convergence: the five intervention effect sizes, all positive and monotone.

Inputs (evaluation/): c4_by_budget_20260525.csv, c4_doubao15pro_by_budget_20260526.csv,
  c4_qwen14b_by_budget_20260525.csv, c5_paired_cot_by_dc.csv. Reasoner/Doubao-reasoner
  overall accuracies are the released values from reasoner_vs_chat_paired_20260526.md.
Output: figures/F15_testtime_compute.{png,pdf}
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
EV = PROJECT / "evaluation"
OUT = PROJECT / "figures" / "F15_testtime_compute.png"

INK = "#1b1b1b"
C_DS = "#0b6fa4"; C_DOU = "#d1495b"; C_QW = "#8896ab"
C_COT = "#7b4ea3"; C_RSN = "#2a9d8f"
B0 = 128.0


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 7.8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def linreg(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x); sx = x.sum(); sy = y.sum()
    b = (n * (x * y).sum() - sx * sy) / (n * (x * x).sum() - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def panel_label(ax, s, dx=-0.13, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="left")


# ------------------------------------------------------------------ panel (a)
def draw_sweep(ax):
    fams = [("DeepSeek", "c4_by_budget_20260525.csv", C_DS, "o"),
            ("Doubao-1.5-pro", "c4_doubao15pro_by_budget_20260526.csv", C_DOU, "s"),
            ("Qwen-14B", "c4_qwen14b_by_budget_20260525.csv", C_QW, "^")]
    xg = np.linspace(0, math.log(2048 / B0), 100)
    for name, f, c, mk in fams:
        rows = read_csv(EV / f)
        x = [math.log(float(r["max_tokens"]) / B0) for r in rows]
        y = [float(r["accuracy"]) for r in rows]
        ax.plot(x, y, mk, color=c, ms=6, zorder=4, markeredgecolor="white", markeredgewidth=0.7)
        a, b = linreg(x, y)
        ax.plot(xg, a + b * xg, color=c, lw=2.0, alpha=0.9, zorder=3,
                label=f"{name}: slope {b:.3f}")
    ax.plot(xg, 0.04 + 0.11 * xg, ls="--", color=INK, lw=1.2, alpha=0.55,
            label=r"loose ref $\beta/e\approx0.11$")
    ax.set_xlabel(r"inference-budget step  $\log(B_t/128)$")
    ax.set_ylabel("judged accuracy")
    ax.set_xticks([math.log(b / B0) for b in (128, 256, 512, 1024, 2048)])
    ax.set_xticklabels(["128", "256", "512", "1k", "2k"])
    ax.set_title("(a) Controlled budget sweep, 3 families", pad=7)
    ax.legend(loc="upper left", handlelength=1.7)


# ------------------------------------------------------------------ panel (b)
def draw_cot(ax):
    """Mechanism view: compute (here, CoT) shifts the accuracy-vs-d_c curve UP but
    does not flatten it — the per-constraint penalty (downward slope) persists."""
    rows = sorted((r for r in read_csv(EV / "c5_paired_cot_by_dc.csv") if int(r["d_c"]) <= 3),
                  key=lambda r: int(r["d_c"]))
    dcs = [int(r["d_c"]) for r in rows]
    direct = [float(r["direct_acc"]) for r in rows]
    cot = [float(r["cot_acc"]) for r in rows]
    ax.fill_between(dcs, direct, cot, color=C_COT, alpha=0.11, zorder=1, label="CoT gain")
    ax.plot(dcs, direct, "--o", color="#8a96a3", lw=1.8, ms=6, zorder=3,
            label="direct (no CoT)", markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(dcs, cot, "-o", color=C_COT, lw=2.6, ms=7.5, zorder=4,
            label="chain-of-thought", markeredgecolor="white", markeredgewidth=0.7)
    for d, di, co in zip(dcs, direct, cot):
        if co - di > 0.015:
            ax.annotate(f"+{co - di:.2f}", xy=(d, (di + co) / 2), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.4, color=C_COT,
                        ha="left", va="center", fontweight="bold")
    ax.annotate("both curves still fall with $d_c$\n(penalty not erased)",
                xy=(dcs[-1], cot[-1]), xytext=(0.50, 0.55), textcoords="axes fraction",
                fontsize=7.2, color=INK, style="italic", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9))
    ax.set_xlabel("conservation-constraint load $d_c$")
    ax.set_ylabel("judged accuracy")
    ax.set_xticks(dcs); ax.set_ylim(0, 0.5)
    ax.set_title("(b) CoT lifts the curve; the $d_c$ penalty remains", pad=7)
    ax.text(0.97, 0.94, r"overall $\Delta=+0.137$" + "\n" + r"sign-test $p=1.2\times10^{-5}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3eef8", edgecolor="none"))
    ax.legend(loc="lower left", fontsize=7.4)


# ------------------------------------------------------------------ panel (c)
def draw_reasoner(ax):
    # DeepSeek: chat@128, chat@2048 from CSV; reasoner@16k released value
    ds = {int(0): None}
    rows = read_csv(EV / "c4_by_budget_20260525.csv")
    by = {int(float(r["max_tokens"])): float(r["accuracy"]) for r in rows}
    ds_x = [0, 1, 2]
    ds_y = [by[128], by[2048], 0.320]  # reasoner 16k = 16/50 (released)
    ax.plot(ds_x, ds_y, "-o", color=C_DS, lw=2.4, ms=8, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8, label="DeepSeek")
    # Doubao: chat (0.08) -> reasoner (0.18) released
    ax.plot([1, 2], [0.08, 0.18], "-s", color=C_DOU, lw=2.0, ms=7, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8, label="Doubao")
    for x, y in zip(ds_x, ds_y):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=C_DS)
    ax.text(2, 0.18 + 0.012, "0.18", ha="center", va="bottom", fontsize=8, color=C_DOU)
    ax.text(1, 0.08 - 0.018, "0.08", ha="center", va="top", fontsize=8, color=C_DOU)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["chat\n@128", "chat\n@2048", "reasoner\n@16k"], fontsize=8)
    ax.set_ylabel("judged accuracy (50-item subset)")
    ax.set_ylim(0, 0.38)
    ax.set_title("(c) Reasoner > chat, two families", pad=7)
    ax.text(0.04, 0.93, "DeepSeek $+0.16$ ($p=0.039$)\nDoubao $+0.10$ (replication)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6f3f1", edgecolor="none"))
    ax.legend(loc="lower right")


# ------------------------------------------------------------------ panel (d)
def draw_convergence(ax):
    # effect sizes (delta accuracy) for the five interventions
    items = [
        ("Qwen-14B budget sweep", 0.042, C_QW, "$p=0.056$"),
        ("Doubao budget sweep", 0.105, C_DOU, "$p=0.084$"),
        ("DeepSeek budget sweep", 0.123, C_DS, "$p=0.006$"),
        ("Doubao reasoner vs chat", 0.100, C_RSN, "repl."),
        ("DeepSeek reasoner vs chat", 0.160, C_RSN, "$p=0.039$"),
        ("DeepSeek direct vs CoT", 0.137, C_COT, "$p=1{\\times}10^{-5}$"),
    ]
    ys = np.arange(len(items))
    for y, (lab, dv, c, p) in zip(ys, items):
        ax.plot([0, dv], [y, y], color=c, lw=1.6, alpha=0.6, zorder=2)
        ax.plot([dv], [y], "o", ms=9, color=c, zorder=4, markeredgecolor="white", markeredgewidth=1.0)
        ax.text(dv + 0.004, y, f"+{dv:.2f}  {p}", va="center", ha="left", fontsize=7.6, color=INK)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels([lab for lab, *_ in items], fontsize=8)
    ax.set_xlim(-0.005, 0.25)
    ax.set_xlabel(r"accuracy gain $\Delta$ on the $d_c$-heavy subset")
    ax.set_title("(d) Five interventions converge (all positive)", pad=7)
    ax.grid(axis="y", alpha=0)


def main():
    set_style()
    fig = plt.figure(figsize=(12.2, 8.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.27,
                           left=0.10, right=0.965, top=0.92, bottom=0.095)
    axa = fig.add_subplot(gs[0, 0]); draw_sweep(axa); panel_label(axa, "a")
    axb = fig.add_subplot(gs[0, 1]); draw_cot(axb); panel_label(axb, "b")
    axc = fig.add_subplot(gs[1, 0]); draw_reasoner(axc); panel_label(axc, "c")
    axd = fig.add_subplot(gs[1, 1]); draw_convergence(axd); panel_label(axd, "d", dx=-0.42)
    fig.suptitle("Inference budget, chain-of-thought, and reasoner architecture each move accuracy along the $d_c$ axis — concordantly",
                 fontsize=12, fontweight="bold", y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".pdf"))
    print(f"[fig] {OUT}")


if __name__ == "__main__":
    main()
