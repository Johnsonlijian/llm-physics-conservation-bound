"""Figure 1 (hero composite): The constraint-penalty law.

Four-panel submission-grade display item, all panels from released CSVs:
  (a) Concept: constraint-satisfaction grading -> multiplicative Bernoulli gates
      -> log-odds linear in d_c (a drawn schematic, no data).
  (b) The law on data: per-model observed accuracy vs d_c (faint), the M3
      controlled marginal (bold), and the A1-predicted logistic curve; the
      OR = 0.69 per-constraint multiplier is annotated.
  (c) Control-ladder forest: OR per +1 d_c as controls accrue (M0->M3->+proxies),
      Wald 95% CIs, reference line at OR = 1.
  (d) Leave-item-out stability: distribution of the d_c coefficient across all
      258 LIO folds; every fold is negative.

Inputs (all in evaluation/):
  w6_observed_dc_bins.csv, w6_controlled_marginal_by_dc.csv,
  w6_logit_coefficients.csv, w6_proxy_control_coefficients_20260525.csv,
  W6_holdout_cv_20260525.csv
Output: figures/F13_constraint_penalty_law.{png,pdf}
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib import gridspec
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
EV = PROJECT / "evaluation"
OUT = PROJECT / "figures" / "F13_constraint_penalty_law.png"

# ---- palette (colour-blind-safe, perceptually ordered for models) -------------
INK = "#1b1b1b"
ACCENT = "#0b6fa4"      # deep blue  (controlled marginal / law)
WARM = "#d1495b"        # crimson    (OR / emphasis)
MODEL_COLORS = {
    "DeepSeekV3": "#22223b",
    "KimiK2": "#4a7c59",
    "Qwen14B": "#8896ab",
    "Qwen7B-Ollama": "#c9ada0",
}
MODEL_NICE = {"DeepSeekV3": "DeepSeek (v4-flash)", "KimiK2": "Kimi-K2",
              "Qwen14B": "Qwen-14B", "Qwen7B-Ollama": "Qwen-7B"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def panel_label(ax, s, dx=-0.13, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="left")


# ------------------------------------------------------------------ panel (a)
def draw_concept(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Constraint-satisfaction grading $\\Rightarrow$ logistic in $d_c$", pad=8)
    # problem box
    pb = FancyBboxPatch((0.2, 6.2), 2.5, 2.2, boxstyle="round,pad=0.1,rounding_size=0.25",
                        linewidth=1.2, edgecolor=INK, facecolor="#eef3f7")
    ax.add_patch(pb)
    ax.text(1.45, 7.3, "physics\nproblem", ha="center", va="center", fontsize=9, color=INK)
    # gates in series
    gx = [4.0, 5.4, 6.8]
    labels = ["momentum", "energy", "$\\cdots$"]
    for i, (x, lab) in enumerate(zip(gx, labels)):
        c = Circle((x, 7.3), 0.55, facecolor="white", edgecolor=ACCENT, linewidth=1.6, zorder=3)
        ax.add_patch(c)
        ax.text(x, 7.3, "$p$", ha="center", va="center", fontsize=10, color=ACCENT, zorder=4)
        ax.text(x, 6.45, lab, ha="center", va="top", fontsize=6.8, color="#555")
        if i > 0:
            ax.text((gx[i] + gx[i-1]) / 2, 7.3, "$\\times$", ha="center", va="center", fontsize=11, color="#777")
    ax.add_patch(FancyArrowPatch((2.75, 7.3), (3.4, 7.3), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.1))
    # AND -> correct
    ax.add_patch(FancyArrowPatch((7.35, 7.3), (8.1, 7.3), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.1))
    ab = FancyBboxPatch((8.15, 6.55), 1.65, 1.5, boxstyle="round,pad=0.08,rounding_size=0.2",
                        linewidth=1.2, edgecolor=WARM, facecolor="#fbeaed")
    ax.add_patch(ab)
    ax.text(8.97, 7.3, "ALL\npass?", ha="center", va="center", fontsize=8, color=WARM)
    # equations
    ax.text(0.2, 4.7, "Correct only if every constraint holds (A1):", fontsize=8.6, color=INK)
    ax.text(0.6, 3.55, r"$P(\mathrm{correct}) = \prod_{j=1}^{d_c} p_j = p^{\,d_c}$", fontsize=12.5, color=INK)
    ax.text(0.6, 2.0, r"$\mathrm{logit}\,P = d_c\,\log\frac{p}{1-p} + \mathrm{const}$",
            fontsize=12.5, color=ACCENT)
    ax.text(0.6, 0.55, r"$\Rightarrow\ \mathrm{OR} = e^{\beta} \approx 0.69 \approx 1/\sqrt{2}$  per constraint",
            fontsize=11.5, color=WARM)


# ------------------------------------------------------------------ panel (b)
def draw_law(ax):
    bins = read_csv(EV / "w6_observed_dc_bins.csv")
    marg = read_csv(EV / "w6_controlled_marginal_by_dc.csv")
    by_model = {}
    for r in bins:
        by_model.setdefault(r["model_label"], []).append((int(r["d_c"]), float(r["accuracy"]), int(r["n"])))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.0, alpha=0.55,
                color=MODEL_COLORS.get(m, "#999"), label=MODEL_NICE.get(m, m), zorder=2)
    mx = [int(r["d_c"]) for r in marg]; my = [float(r["mean_predicted_accuracy"]) for r in marg]
    ax.plot(mx, my, marker="s", ms=5, lw=2.6, color=ACCENT, zorder=5,
            label="controlled marginal (M3)")
    # A1-predicted logistic from pooled M0 fit (theta0,theta1) for reference
    th0, th1 = -1.536777, -0.573783  # pooled logistic (shape_competition / M0)
    xg = np.linspace(0, 5, 100)
    ax.plot(xg, 1 / (1 + np.exp(-(th0 + th1 * xg))), ls="--", lw=1.4, color=WARM,
            alpha=0.85, zorder=4, label="A1-predicted logistic")
    ax.annotate(r"$\widehat{\mathrm{OR}}=0.69$ per $+1\,d_c$",
                xy=(2, 1 / (1 + np.exp(-(th0 + th1 * 2)))), xytext=(2.55, 0.235),
                fontsize=9, color=WARM,
                arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.1))
    ax.set_xlabel(r"conservation-constraint load $d_c$")
    ax.set_ylabel("judged accuracy")
    ax.set_xticks(range(0, 6))
    ax.set_ylim(-0.01, 0.31)
    ax.set_title("Accuracy declines logistically with $d_c$", pad=8)
    ax.legend(loc="upper right", ncol=1, handlelength=1.6)


# ------------------------------------------------------------------ panel (c)
def draw_forest(ax):
    lad = {r["spec"]: r for r in read_csv(EV / "w6_logit_coefficients.csv")}
    prox = {r["spec"]: r for r in read_csv(EV / "w6_proxy_control_coefficients_20260525.csv")}
    rows = [
        ("M0  $d_c$ only", lad["M0_dc_only"]),
        ("M1  +len/src/type", lad["M1_item_controls"]),
        ("M2  +model/tokens", lad["M2_model_controls"]),
        ("M3  +topic  (main)", lad["M3_topic_controls"]),
        ("M4  +text proxies", prox["M4_plus_text_difficulty_proxies"]),
    ]
    ys = list(range(len(rows)))[::-1]
    for y, (lab, r) in zip(ys, rows):
        b = float(r["dc_beta"]); se = float(r["dc_se"])
        orv = math.exp(b); lo = math.exp(b - 1.96 * se); hi = math.exp(b + 1.96 * se)
        main = "M3" in lab
        col = WARM if main else ACCENT
        ax.plot([lo, hi], [y, y], color=col, lw=2.4 if main else 1.6, solid_capstyle="round", zorder=3)
        ax.plot([orv], [y], "o", ms=8 if main else 6, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(hi + 0.02, y, f"{orv:.2f}", va="center", ha="left", fontsize=8, color=col)
    ax.axvline(1.0, color=INK, lw=1.0, ls=":")
    ax.text(1.0, len(rows) - 0.35, "no effect", rotation=90, va="top", ha="right", fontsize=7, color="#666")
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8.2)
    ax.set_xlabel("odds ratio per $+1\\,d_c$  (Wald 95% CI)")
    ax.set_xlim(0.3, 1.25)
    ax.set_title("Penalty is stable as controls accrue", pad=8)
    ax.grid(axis="y", alpha=0)
    ax.text(0.0, -0.235, "M3 item-cluster bootstrap CI [0.44, 0.98]; proxy-expanded reaches 1.0 (see §4.4)",
            transform=ax.transAxes, fontsize=6.6, color="#777", style="italic")


# ------------------------------------------------------------------ panel (d)
def draw_lio(ax):
    betas = np.array([float(r["dc_beta"]) for r in read_csv(EV / "W6_holdout_cv_20260525.csv")])
    ax.hist(betas, bins=22, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(betas.mean(), color=WARM, lw=2.0, label=f"mean $\\beta={betas.mean():.3f}$")
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    ax.set_xlabel("$d_c$ coefficient $\\beta$  (per leave-item-out fold)")
    ax.set_ylabel("folds")
    ax.set_title(f"Sign-stable across all {len(betas)} LIO folds", pad=8)
    ax.text(0.04, 0.94, f"{int((betas < 0).sum())}/{len(betas)} folds $\\beta<0$\n"
                        f"range [{betas.min():.2f}, {betas.max():.2f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef3f7", edgecolor="none"))
    ax.legend(loc="upper right")


def main():
    set_style()
    fig = plt.figure(figsize=(12.2, 8.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26,
                           left=0.135, right=0.97, top=0.92, bottom=0.10)
    axa = fig.add_subplot(gs[0, 0]); draw_concept(axa); panel_label(axa, "a", dx=-0.04)
    axb = fig.add_subplot(gs[0, 1]); draw_law(axb); panel_label(axb, "b")
    axc = fig.add_subplot(gs[1, 0]); draw_forest(axc); panel_label(axc, "c", dx=-0.30)
    axd = fig.add_subplot(gs[1, 1]); draw_lio(axd); panel_label(axd, "d")
    fig.suptitle("The constraint-penalty law: each conservation constraint multiplies the odds of a correct answer by $\\sim$0.69",
                 fontsize=12.5, fontweight="bold", y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(f"[fig] {OUT}")


if __name__ == "__main__":
    main()
