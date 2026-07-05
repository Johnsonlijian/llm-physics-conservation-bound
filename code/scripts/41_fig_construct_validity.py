"""Figure 2 (hero composite): three independent channels validate d_c.

Four-panel submission-grade construct-validity display item, all panels from
evaluation/rule_based_dc_floor_items_20260529.csv (+ the documented three-way
Spearman values):
  (a) Convergence schematic: a deterministic zero-LLM regex, the PhysReason
      benchmark authors' human theorem labels, and our 4-family LLM rater pool
      agree pairwise (edge = Spearman rho, thickness scaled).
  (b) 2D agreement heat-map of zero-LLM regex d_c vs human d_c on 1200 items
      (diagonal = exact match), with rho/within-1 annotation.
  (c) Per-family presence agreement (Cohen kappa, regex vs human).
  (d) Known-groups validity: mean d_c by PhysReason difficulty tier, human vs
      zero-LLM regex tracking the same monotone gradient.

Output: figures/F14_construct_validity.{png,pdf,svg}
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import gridspec
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ITEMS = PROJECT / "evaluation" / "rule_based_dc_floor_items_20260529.csv"
COEF = PROJECT / "evaluation" / "regex_dc_w6_robustness_coef_20260529.csv"
OUT = PROJECT / "figures" / "F14_construct_validity.png"

INK = "#1b1b1b"
C_REGEX = "#e08a1e"   # zero-LLM regex (warm)
C_HUMAN = "#0b6fa4"   # human (blue)
C_LLM = "#4a7c59"     # LLM (green)
FAMILIES = ["momentum", "angular_momentum", "energy", "charge", "mass", "entropy"]
FAM_LABEL = {"momentum": "mom.", "angular_momentum": "ang.\nmom.",
             "energy": "energy", "charge": "charge", "mass": "mass", "entropy": "2nd-law"}
DIFF = ["knowledge", "easy", "medium", "difficult"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })


def read_rows():
    with open(ITEMS, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cohen_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    po = float(np.mean(a == b)); p1a, p1b = float(np.mean(a)), float(np.mean(b))
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")


def panel_label(ax, s, dx=-0.12, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="left")


# ------------------------------------------------------------------ panel (a)
def draw_network(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Three independent channels measure the same $d_c$", pad=8)
    nodes = {
        "regex": (2.45, 2.4, C_REGEX, "zero-LLM\nregex\n(no inference)"),
        "human": (5.0, 8.0, C_HUMAN, "human\n(PhysReason\nauthors)"),
        "llm":   (8.0, 2.4, C_LLM, "LLM raters\n(4 families)"),
    }
    edges = [("regex", "human", 0.79, "1200 items"),
             ("human", "llm", 0.54, "81 items"),
             ("regex", "llm", 0.32, "81 items")]
    for a, b, rho, n in edges:
        xa, ya = nodes[a][0], nodes[a][1]; xb, yb = nodes[b][0], nodes[b][1]
        ax.plot([xa, xb], [ya, yb], color="#9aa6b2", lw=1.0 + 6.0 * rho, alpha=0.55,
                solid_capstyle="round", zorder=1)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my, f"$\\rho={rho:.2f}$", fontsize=10, fontweight="bold", color=INK,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#ccc", lw=0.8))
        ax.text(mx, my - 0.62, n, fontsize=6.6, color="#777", ha="center", va="center", zorder=4)
    for key, (x, y, c, lab) in nodes.items():
        ax.add_patch(Circle((x, y), 0.92, facecolor=c, edgecolor="white", lw=2.0, zorder=3, alpha=0.92))
        ax.text(x, y, lab, fontsize=6.8, color="white", ha="center", va="center", zorder=5, fontweight="bold")
    ax.text(5.0, 0.5, "pairwise agreement of a rule system, human labels, and LLMs\n"
                      "$\\Rightarrow$ $d_c$ is an objective item property, not an LLM artefact",
            fontsize=7.8, color=INK, ha="center", va="center", style="italic")


# ------------------------------------------------------------------ panel (b)
def read_csv_any(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_substitution(ax):
    """The decisive test: re-run the identical M0->M3 regression with the d_c
    *source* swapped from LLM-consensus to the deterministic zero-LLM regex."""
    by = {(r["source"], r["spec"]): r for r in read_csv_any(COEF)}
    specs = ["M0_dc_only", "M1_item_controls", "M2_model_controls", "M3_topic_controls"]
    series = [("llm_consensus_dc", r"LLM-consensus conservation-constraint load $d_c$", C_HUMAN),
              ("regex_dc", "zero-LLM regex $d_c$", C_REGEX)]
    for si, (src, lab, col) in enumerate(series):
        off = 0.17 if si == 0 else -0.17
        first = True
        for yi, spec in enumerate(specs):
            r = by.get((src, spec))
            if not r:
                continue
            b = float(r["dc_beta"]); se = float(r["dc_se"])
            orv = math.exp(b); lo = math.exp(b - 1.96 * se); hi = math.exp(b + 1.96 * se)
            y = yi + off
            main = spec.startswith("M3")
            ax.plot([lo, hi], [y, y], color=col, lw=2.4 if main else 1.5,
                    solid_capstyle="round", zorder=3, alpha=0.92)
            ax.plot([orv], [y], "o", ms=8 if main else 5.5, color=col, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.0,
                    label=lab if first else None)
            first = False
            if main:
                if si == 0:
                    ax.text(orv, y + 0.34, f"OR {orv:.2f}", ha="center", va="bottom",
                            fontsize=8.4, color=col, fontweight="bold")
                else:
                    ax.text(orv, y - 0.42, f"OR {orv:.2f}", ha="center", va="top",
                            fontsize=8.4, color=col, fontweight="bold")
    ax.axvline(1.0, color=INK, lw=1.0, ls=":")
    ax.text(1.0, 3.45, "no effect", rotation=90, va="top", ha="right", fontsize=7, color="#666")
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels(["M0  $d_c$ only", "M1  +item/source", "M2  +model/tokens", "M3  +topic (main)"],
                       fontsize=8.0)
    ax.set_ylim(-0.65, len(specs) - 0.35)
    ax.set_xlim(0.28, 1.18)
    ax.set_xlabel("odds ratio per $+1\\,d_c$  (Wald 95% CI)")
    ax.set_title("Swap in a zero-LLM $d_c$: the effect persists", pad=8)
    ax.legend(loc="lower left", fontsize=7.2, handletextpad=0.4,
              frameon=True, framealpha=0.9, facecolor="white",
              edgecolor="#dddddd")
    ax.grid(axis="y", alpha=0)
    ax.text(0.5, -0.235, "identical estimator, controls and items; only the $d_c$ source changes;\n"
            "regex M3 is negative in 94.8% of 9000 item bootstraps (Section 4.6)",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.7, color="#777", style="italic")


# ------------------------------------------------------------------ panel (c)
def draw_kappa(ax, rows):
    def fams(cell):
        return set(x for x in (cell or "").split("|") if x)
    ks, npos = [], []
    for fam in FAMILIES:
        a = [1 if fam in fams(r["rule_qs_families"]) else 0 for r in rows]
        b = [1 if fam in fams(r["human_families"]) else 0 for r in rows]
        ks.append(cohen_kappa(a, b)); npos.append(sum(b))
    x = np.arange(len(FAMILIES))
    cols = [C_HUMAN if (not np.isnan(k) and k >= 0.4) else "#d1495b" for k in ks]
    vals = [0 if np.isnan(k) else k for k in ks]
    ax.bar(x, vals, 0.64, color=cols, zorder=3, edgecolor="white", lw=0.6)
    ax.axhline(0.6, color=INK, ls="--", lw=0.9, alpha=0.6)
    ax.annotate(r"$\kappa=0.6$", xy=(1.0, 0.6), xycoords=("axes fraction", "data"),
                xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.8, color="#666",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.9))
    for xi, k, n in zip(x, ks, npos):
        val = 0 if np.isnan(k) else k
        if abs(val - 0.6) < 0.08:
            ax.text(xi, max(val - 0.085, 0.08), "n/a" if np.isnan(k) else f"{k:.2f}",
                    ha="center", va="top", fontsize=7.3, color="white", fontweight="bold")
        else:
            ax.text(xi, val + 0.035, "n/a" if np.isnan(k) else f"{k:.2f}",
                    ha="center", va="bottom", fontsize=7.3,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.text(xi, -0.07, f"n={n}", ha="center", va="top", fontsize=6.3, color="#888")
    ax.set_xticks(x); ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=8)
    ax.set_ylim(-0.12, 1.08); ax.set_ylabel("Cohen $\\kappa$ (regex vs human)")
    ax.set_title("Per-family agreement", pad=8); ax.grid(axis="y", alpha=0.18)


# ------------------------------------------------------------------ panel (d)
def draw_known_groups(ax, rows):
    by = {d: {"h": [], "q": [], "qs": []} for d in DIFF}
    for r in rows:
        d = (r["difficulty"] or "").lower()
        if d in by:
            by[d]["h"].append(int(r["human_dc"])); by[d]["qs"].append(int(r["rule_qs_dc"]))
            by[d]["q"].append(int(r["rule_q_dc"]))
    xh = [np.mean(by[d]["h"]) for d in DIFF]
    xqs = [np.mean(by[d]["qs"]) for d in DIFF]
    xq = [np.mean(by[d]["q"]) for d in DIFF]
    xx = np.arange(len(DIFF))
    ax.plot(xx, xh, "-o", color=C_HUMAN, lw=2.2, ms=7, label="human labels", zorder=4)
    ax.plot(xx, xqs, "-s", color=C_REGEX, lw=2.2, ms=6, label="zero-LLM regex (q+sol)", zorder=4)
    ax.plot(xx, xq, "--^", color="#9aa6b2", lw=1.4, ms=5, label="zero-LLM regex (q only)", zorder=3)
    ax.fill_between(xx, xq, xh, color=C_HUMAN, alpha=0.06, zorder=1)
    ax.set_xticks(xx); ax.set_xticklabels([d + f"\n(n=300)" for d in DIFF], fontsize=8)
    ax.set_ylabel("mean $d_c$")
    ax.set_title("Both channels track difficulty ($\\sim$3$\\times$ rise)", pad=8)
    ax.legend(loc="upper left")
    ax.annotate("", xy=(3, xh[3]), xytext=(0, xh[0]),
                arrowprops=dict(arrowstyle="-|>", color=C_HUMAN, lw=1.0, alpha=0.4,
                                connectionstyle="arc3,rad=-0.2"))


def main():
    set_style()
    rows = read_rows()
    fig = plt.figure(figsize=(12.2, 8.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                           left=0.085, right=0.965, top=0.92, bottom=0.10)
    axa = fig.add_subplot(gs[0, 0]); draw_network(axa); panel_label(axa, "a", dx=-0.04)
    axb = fig.add_subplot(gs[0, 1]); draw_substitution(axb); panel_label(axb, "b")
    axc = fig.add_subplot(gs[1, 0]); draw_kappa(axc, rows); panel_label(axc, "c")
    axd = fig.add_subplot(gs[1, 1]); draw_known_groups(axd, rows); panel_label(axd, "d")
    fig.suptitle("Construct validity of $d_c$ without human labels by us: a deterministic regex, the benchmark authors, and LLMs converge",
                 fontsize=12, fontweight="bold", y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".svg"))
    print(f"[fig] {OUT}")


if __name__ == "__main__":
    main()


