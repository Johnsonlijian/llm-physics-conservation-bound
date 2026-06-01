"""Graphical abstract: the whole story of the paper in one figure.

Left-to-right narrative flow:
  1. Count the independent scalar conservation constraints in a problem -> d_c.
  2. Constraint-penalty law: correct only if ALL pass -> odds x OR(=0.69) per constraint.
  3. Accuracy falls along d_c (real controlled-marginal curve, inset, with logistic).
  4. Two payoffs: (top) validated by three independent channels with NO human labels by us;
     (bottom) inference budget / CoT / reasoner move accuracy along the same axis.

One schematic axis (coords 0-100) carries the flow, panels, arrows and text; a single
inset axis renders the real accuracy-vs-d_c curve. Output: figures/F00_graphical_abstract.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
EV = PROJECT / "evaluation"
OUT = PROJECT / "figures" / "F00_graphical_abstract.png"

INK = "#1b1b1b"; ACCENT = "#0b6fa4"; WARM = "#d1495b"
C_REGEX = "#e08a1e"; C_HUMAN = "#0b6fa4"; C_LLM = "#4a7c59"; C_BUD = "#2a9d8f"


def rbox(ax, x, y, w, h, fc, ec=INK, lw=1.3, alpha=1.0, rounding=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.2,rounding_size={rounding}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=2))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=20, color=color, lw=lw, zorder=5))


def main():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})
    fig = plt.figure(figsize=(13.2, 6.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # title
    ax.text(50, 95, "Conservation-constraint load $d_c$: a structural difficulty axis for LLM physics reasoning",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
    ax.text(50, 89.3, "each independent conservation law multiplies the odds of a correct answer by $\\mathrm{OR}\\approx0.69$ — a constraint-penalty law that needs no information-theoretic machinery",
            ha="center", va="center", fontsize=9.5, color="#555", style="italic")

    # ---------- Stage 1: count constraints ----------
    rbox(ax, 1.5, 40, 21, 38, "#eef3f7")
    ax.text(12, 74, "1  Count conservation\nconstraints", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    # two-body collision sketch
    ax.add_patch(Circle((7.5, 60), 2.2, facecolor=ACCENT, edgecolor="white", lw=1.2, zorder=3))
    ax.add_patch(Circle((16.5, 60), 2.6, facecolor=WARM, edgecolor="white", lw=1.2, zorder=3))
    arrow(ax, 9.8, 60, 13.6, 60, color="#888", lw=1.6)
    ax.text(12, 54.5, "elastic collision", ha="center", fontsize=7.5, color="#666")
    ax.text(12, 50.0, "momentum  +  energy", ha="center", fontsize=8.5, color=INK)
    ax.text(12, 45.0, "$d_c = 2$", ha="center", fontsize=14, fontweight="bold", color=WARM)

    # ---------- Stage 2: constraint-penalty law ----------
    rbox(ax, 26.5, 40, 22, 38, "#fbf3ee")
    ax.text(37.5, 74, "2  Constraint-penalty law", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    for i, cx in enumerate([31.5, 37.5, 43.5]):
        ax.add_patch(Circle((cx, 62), 2.1, facecolor="white", edgecolor=ACCENT, lw=1.8, zorder=3))
        ax.text(cx, 62, "$p$", ha="center", va="center", fontsize=10, color=ACCENT, zorder=4)
        if i > 0:
            ax.text(cx - 3.0, 62, "$\\times$", ha="center", va="center", fontsize=12, color="#888")
    ax.text(37.5, 55.5, "correct only if ALL pass", ha="center", fontsize=8, color="#666")
    ax.text(37.5, 50.5, "$P=p^{d_c}\\Rightarrow \\mathrm{logit}\\,P \\propto d_c$", ha="center", fontsize=11, color=ACCENT)
    ax.text(37.5, 44.7, "$\\mathrm{OR}\\approx0.69$ per constraint", ha="center", fontsize=10.5,
            fontweight="bold", color=WARM)

    # ---------- Stage 3: accuracy falls (real inset) ----------
    rbox(ax, 52.5, 40, 22, 38, "#eef3f7")
    ax.text(63.5, 74, "3  Accuracy falls with $d_c$", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)

    # ---------- Stage 4: payoffs ----------
    rbox(ax, 78, 60, 20.5, 18, "#f1f7f0")
    ax.text(88.2, 74.5, "4a  Validated, no human\nlabels by us", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=INK)
    # mini triangle
    nodes = {"r": (82, 63.5, C_REGEX), "h": (88.2, 70.0, C_HUMAN), "l": (94.4, 63.5, C_LLM)}
    tri_edges = [("r", "h", "0.79"), ("h", "l", "0.54"), ("r", "l", "0.32")]
    for a, b, rho in tri_edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#9aa6b2", lw=1.4, zorder=2)
        mx, my = (nodes[a][0] + nodes[b][0]) / 2, (nodes[a][1] + nodes[b][1]) / 2
        ax.text(mx, my, rho, fontsize=6.6, ha="center", va="center", color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for k, (x, y, c) in nodes.items():
        ax.add_patch(Circle((x, y), 1.5, facecolor=c, edgecolor="white", lw=1.2, zorder=3))
    ax.text(82, 61.0, "regex", fontsize=6, ha="center", color=C_REGEX)
    ax.text(88.2, 72.2, "human", fontsize=6, ha="center", color=C_HUMAN)
    ax.text(94.4, 61.0, "LLM", fontsize=6, ha="center", color=C_LLM)

    rbox(ax, 78, 40, 20.5, 17, "#e6f3f1")
    ax.text(88.2, 53.5, "4b  Inference budget moves\naccuracy along $d_c$", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=INK)
    ax.text(88.2, 47.5, "$\\max\\_tokens\\uparrow$  ·  CoT  ·  reasoner", ha="center", fontsize=7.6, color="#555")
    ax.text(88.2, 43.7, "5 controlled interventions, all $+$", ha="center", fontsize=8.4,
            fontweight="bold", color=C_BUD)

    # flow arrows
    arrow(ax, 22.7, 59, 26.3, 59)
    arrow(ax, 48.7, 59, 52.3, 59)
    arrow(ax, 74.7, 64, 77.8, 69)
    arrow(ax, 74.7, 54, 77.8, 49)

    # bottom tagline
    ax.text(50, 24, "reproducible · controllable · externally anchored without new human labels",
            ha="center", fontsize=9.5, color="#555", style="italic")

    # ---------- real inset: controlled marginal accuracy vs d_c ----------
    inset = fig.add_axes([0.557, 0.435, 0.132, 0.235])
    marg = list(csv.DictReader(open(EV / "w6_controlled_marginal_by_dc.csv", encoding="utf-8-sig")))
    mx = [int(r["d_c"]) for r in marg]; my = [float(r["mean_predicted_accuracy"]) for r in marg]
    th0, th1 = -1.536777, -0.573783
    xg = np.linspace(0, 5, 80)
    inset.plot(xg, 1 / (1 + np.exp(-(th0 + th1 * xg))), "--", color=WARM, lw=1.6, zorder=2)
    inset.plot(mx, my, "o-", color=ACCENT, lw=2.0, ms=4.5, zorder=3)
    inset.set_xlabel("$d_c$", fontsize=8, labelpad=1)
    inset.set_ylabel("accuracy", fontsize=8, labelpad=1)
    inset.set_xticks(range(0, 6)); inset.tick_params(labelsize=6.5)
    inset.set_ylim(0, 0.18)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.grid(alpha=0.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".pdf"))
    print(f"[fig] {OUT}")


if __name__ == "__main__":
    main()
