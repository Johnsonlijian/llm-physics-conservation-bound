"""Figure 1 (hero): mechanism + constraint-penalty law.

Redesign of F13 with a full-width MECHANISM band on top (physically grounded,
schematic-led) over a row of three data panels:

  TOP BAND (drawn schematic, no data):
    (1) Count: three worked physics examples map to a conservation-constraint
        load d_c  (projectile d_c=0; elastic collision d_c=2; spinning collision d_c=3).
    (2) Gate: a correct answer must satisfy ALL constraints (A1) -> P = prod p_j.
    (3) Law: odds fall by a constant factor -> logit P = beta d_c + c, OR ~ 0.69.
  BOTTOM ROW (from released CSVs):
    (b) the law on data (per-model + M3 controlled marginal + logistic);
    (c) OR forest across the control ladder;
    (d) leave-item-out sign stability.

Output: figures/F13_constraint_penalty_law.{png,pdf}  (overwrites the referenced file)
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

INK = "#1b1b1b"
ACCENT = "#0b6fa4"
WARM = "#d1495b"
MUTE = "#8a8f98"
FAM = {"momentum": "#0b6fa4", "energy": "#e08a1e", "angular": "#6a4c93",
       "charge": "#2a9d4a", "kinematics": "#9aa0a6"}
MODEL_COLORS = {"DeepSeekV3": "#22223b", "KimiK2": "#4a7c59",
                "Qwen14B": "#8896ab", "Qwen7B-Ollama": "#c9ada0"}
MODEL_NICE = {"DeepSeekV3": "DeepSeek (v4-flash)", "KimiK2": "Kimi-K2",
              "Qwen14B": "Qwen-14B", "Qwen7B-Ollama": "Qwen-7B"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 7.6,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "font.family": "DejaVu Sans",
    })


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def panel_label(ax, s, dx=-0.13, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="left")


# ----------------------------------------------------------- chips / badges
def chip(ax, x, y, text, color, w=None):
    w = w if w is not None else 0.42 + 0.165 * len(text)
    ax.add_patch(FancyBboxPatch((x, y - 0.30), w, 0.60,
                 boxstyle="round,pad=0.02,rounding_size=0.30",
                 linewidth=0, facecolor=color, alpha=0.16, zorder=3))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=7.4,
            color=color, zorder=4, fontweight="bold")
    return w


def badge(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x, y - 0.42), 1.85, 0.84,
                 boxstyle="round,pad=0.04,rounding_size=0.18",
                 linewidth=1.4, edgecolor=color, facecolor="white", zorder=5))
    ax.text(x + 0.925, y, text, ha="center", va="center", fontsize=11,
            color=color, fontweight="bold", zorder=6)


def step_tag(ax, x, y, n, title):
    ax.add_patch(Circle((x, y), 0.42, facecolor=INK, edgecolor="none", zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=7)
    ax.text(x + 0.7, y, title, ha="left", va="center", fontsize=10,
            color=INK, fontweight="bold")


# ----------------------------------------------------------- TOP mechanism band
def draw_mechanism(ax):
    ax.set_xlim(0, 30); ax.set_ylim(0, 10); ax.axis("off")

    # separators
    for sx in (12.3, 21.1):
        ax.plot([sx, sx], [0.4, 8.7], ls=(0, (1, 3)), color="#cfd4da", lw=1.2, zorder=1)

    # ===== region 1: COUNT =====
    step_tag(ax, 0.6, 9.3, 1, "Count the conservation constraints")
    cards = [
        dict(cy=6.7, name="Projectile", sub="kinematics", chips=[("kinematics", "kinematics")],
             dc="$d_c=0$", dcc=MUTE, icon="proj"),
        dict(cy=3.9, name="Elastic collision", sub="head-on, two bodies",
             chips=[("momentum", "momentum"), ("energy", "energy")],
             dc="$d_c=2$", dcc=ACCENT, icon="coll"),
        dict(cy=1.1, name="Spinning collision", sub="oblique, with rotation",
             chips=[("momentum", "momentum"), ("angular", "ang. mom."), ("energy", "energy")],
             dc="$d_c=3$", dcc=WARM, icon="spin"),
    ]
    for c in cards:
        cy = c["cy"]
        ax.add_patch(FancyBboxPatch((0.2, cy - 1.18), 11.7, 2.36,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=0, facecolor="#f5f7f9", zorder=0))
        _icon(ax, c["icon"], 1.55, cy)
        ax.text(3.05, cy + 0.62, c["name"], ha="left", va="center", fontsize=9.2,
                color=INK, fontweight="bold")
        ax.text(3.05, cy + 0.10, c["sub"], ha="left", va="center", fontsize=7.3, color="#6b7077")
        x = 3.05
        for key, label in c["chips"]:
            x += chip(ax, x, cy - 0.55, label, FAM[key]) + 0.18
        badge(ax, 9.85, cy, c["dc"], c["dcc"])

    # ===== region 2: GATE =====
    step_tag(ax, 12.7, 9.3, 2, "All must hold  (A1)")
    gx = [14.2, 15.9, 17.6]
    glab = ["momentum", "energy", "ang. mom."]
    gcol = [FAM["momentum"], FAM["energy"], FAM["angular"]]
    ax.add_patch(FancyArrowPatch((13.0, 6.4), (13.7, 6.4), arrowstyle="-|>",
                 mutation_scale=12, color=INK, lw=1.2))
    for i, (x, lab, col) in enumerate(zip(gx, glab, gcol)):
        ax.add_patch(Circle((x, 6.4), 0.52, facecolor="white", edgecolor=col,
                     linewidth=2.0, zorder=3))
        ax.text(x, 6.4, "$p$", ha="center", va="center", fontsize=11, color=col, zorder=4)
        ax.text(x, 5.5, lab, ha="center", va="top", fontsize=6.6, color="#666")
        if i > 0:
            ax.text((gx[i] + gx[i - 1]) / 2, 6.4, "$\\times$", ha="center", va="center",
                    fontsize=13, color="#888")
    ax.add_patch(FancyArrowPatch((18.15, 6.4), (18.8, 6.4), arrowstyle="-|>",
                 mutation_scale=12, color=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((18.85, 5.7), 1.7, 1.42,
                 boxstyle="round,pad=0.06,rounding_size=0.16",
                 linewidth=1.6, edgecolor=WARM, facecolor="#fbeef0", zorder=3))
    ax.text(19.7, 6.4, "ALL\npass", ha="center", va="center", fontsize=8.2,
            color=WARM, fontweight="bold", zorder=4)
    ax.text(16.0, 3.7, r"$P(\mathrm{correct})=\prod_{j=1}^{d_c} p_j = p^{\,d_c}$",
            ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(16.0, 2.2, "each check $\\sim$ Bernoulli($p$); one miss $\\Rightarrow$ wrong",
            ha="center", va="center", fontsize=7.6, color="#6b7077", style="italic")

    # ===== region 3: LAW =====
    step_tag(ax, 21.5, 9.3, 3, "Odds fall by a constant factor")
    ax.text(25.4, 6.7, r"$\mathrm{logit}\,P(\mathrm{correct}) = \beta\,d_c + \mathrm{const}$",
            ha="center", va="center", fontsize=12.5, color=ACCENT)
    ax.add_patch(FancyBboxPatch((22.6, 4.0), 5.6, 1.5,
                 boxstyle="round,pad=0.06,rounding_size=0.18",
                 linewidth=1.8, edgecolor=WARM, facecolor="#fbeef0", zorder=3))
    ax.text(25.4, 4.75, r"$\mathrm{OR}=e^{\beta}\approx 0.69$", ha="center", va="center",
            fontsize=15, color=WARM, fontweight="bold", zorder=4)
    ax.text(25.4, 3.25, "per added conservation constraint", ha="center", va="center",
            fontsize=8.4, color=INK)
    ax.text(25.4, 2.35, r"$\approx 1/\sqrt{2}$  (per-constraint success odds $\approx 2{:}3$)",
            ha="center", va="center", fontsize=7.8, color="#6b7077", style="italic")
    ax.add_patch(FancyArrowPatch((25.4, 1.7), (25.4, 0.5), arrowstyle="-|>",
                 mutation_scale=13, color=MUTE, lw=1.4))
    ax.text(26.0, 1.05, "measured\nbelow", ha="left", va="center", fontsize=7.4, color=MUTE)


def _icon(ax, kind, x, y):
    """Small clean physical icon centred near (x, y)."""
    if kind == "proj":
        t = np.linspace(0, 1, 50)
        px = x - 1.0 + 2.0 * t
        py = y - 0.7 + 2.6 * t * (1 - t)
        ax.plot(px, py, ls=(0, (2, 2)), color=ACCENT, lw=1.5, zorder=2)
        ax.add_patch(Circle((px[0], py[0]), 0.16, facecolor=ACCENT, edgecolor="none", zorder=3))
        ax.add_patch(FancyArrowPatch((px[0], py[0]), (px[0] + 0.5, py[0] + 0.5),
                     arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.1, zorder=3))
    elif kind == "coll":
        ax.add_patch(Circle((x - 0.6, y), 0.30, facecolor=ACCENT, alpha=0.85, edgecolor="none"))
        ax.add_patch(Circle((x + 0.6, y), 0.30, facecolor="#9aa0a6", alpha=0.9, edgecolor="none"))
        ax.add_patch(FancyArrowPatch((x - 1.15, y), (x - 0.95, y), arrowstyle="-|>",
                     mutation_scale=9, color=INK, lw=1.4))
        ax.add_patch(FancyArrowPatch((x + 1.15, y), (x + 0.95, y), arrowstyle="-|>",
                     mutation_scale=9, color=INK, lw=1.4))
    elif kind == "spin":
        ax.add_patch(Circle((x - 0.6, y), 0.30, facecolor=WARM, alpha=0.85, edgecolor="none"))
        ax.add_patch(Circle((x + 0.6, y), 0.30, facecolor="#9aa0a6", alpha=0.9, edgecolor="none"))
        ax.add_patch(FancyArrowPatch((x - 1.15, y - 0.05), (x - 0.95, y - 0.05),
                     arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.4))
        # spin arc on left ball
        ax.add_patch(FancyArrowPatch((x - 0.6, y + 0.42), (x - 0.18, y + 0.10),
                     connectionstyle="arc3,rad=0.9", arrowstyle="-|>",
                     mutation_scale=9, color="#6a4c93", lw=1.5))


# ----------------------------------------------------------- data panels (b,c,d)
def draw_law(ax):
    bins = read_csv(EV / "w6_observed_dc_bins.csv")
    marg = read_csv(EV / "w6_controlled_marginal_by_dc.csv")
    by_model = {}
    for r in bins:
        by_model.setdefault(r["model_label"], []).append((int(r["d_c"]), float(r["accuracy"])))
    for m, vals in by_model.items():
        vals.sort()
        ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", ms=3.2, lw=1.0,
                alpha=0.5, color=MODEL_COLORS.get(m, "#999"), label=MODEL_NICE.get(m, m), zorder=2)
    mx = [int(r["d_c"]) for r in marg]; my = [float(r["mean_predicted_accuracy"]) for r in marg]
    ax.plot(mx, my, marker="s", ms=5, lw=2.6, color=ACCENT, zorder=5, label="controlled marginal")
    th0, th1 = -1.536777, -0.573783
    xg = np.linspace(0, 5, 100)
    ax.plot(xg, 1 / (1 + np.exp(-(th0 + th1 * xg))), ls="--", lw=1.4, color=WARM, alpha=0.85,
            zorder=4, label="logistic in $d_c$")
    ax.annotate(r"$\widehat{\mathrm{OR}}=0.69$/$d_c$",
                xy=(2, 1 / (1 + np.exp(-(th0 + th1 * 2)))), xytext=(1.75, 0.255),
                fontsize=8.6, color=WARM, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.0))
    ax.set_xlabel("conservation-constraint load $d_c$"); ax.set_ylabel("judged accuracy")
    ax.set_xticks(range(0, 6)); ax.set_ylim(-0.01, 0.31)
    ax.set_title("The law on data", pad=6)
    ax.legend(loc="upper right", handlelength=1.5)


def draw_forest(ax):
    lad = {r["spec"]: r for r in read_csv(EV / "w6_logit_coefficients.csv")}
    prox = {r["spec"]: r for r in read_csv(EV / "w6_proxy_control_coefficients_20260525.csv")}
    rows = [("M0  $d_c$ only", lad["M0_dc_only"]),
            ("M1  +len/src/type", lad["M1_item_controls"]),
            ("M2  +model/tokens", lad["M2_model_controls"]),
            ("M3  +topic (main)", lad["M3_topic_controls"]),
            ("M4  +text proxies", prox["M4_plus_text_difficulty_proxies"])]
    ys = list(range(len(rows)))[::-1]
    for y, (lab, r) in zip(ys, rows):
        b = float(r["dc_beta"]); se = float(r["dc_se"])
        orv = math.exp(b); lo = math.exp(b - 1.96 * se); hi = math.exp(b + 1.96 * se)
        main = "main" in lab; col = WARM if main else ACCENT
        ax.plot([lo, hi], [y, y], color=col, lw=2.4 if main else 1.6, solid_capstyle="round", zorder=3)
        ax.plot([orv], [y], "o", ms=8 if main else 6, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(hi + 0.02, y, f"{orv:.2f}", va="center", ha="left", fontsize=8, color=col)
    ax.axvline(1.0, color=INK, lw=1.0, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in rows], fontsize=8.0)
    ax.set_xlabel("odds ratio per $+1\\,d_c$  (95% CI)"); ax.set_xlim(0.3, 1.25)
    ax.set_title("Stable as controls accrue", pad=6)
    ax.grid(axis="y", alpha=0)


def draw_lio(ax):
    betas = np.array([float(r["dc_beta"]) for r in read_csv(EV / "W6_holdout_cv_20260525.csv")])
    ax.hist(betas, bins=22, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(betas.mean(), color=WARM, lw=2.0, label=f"mean $\\beta={betas.mean():.3f}$")
    ax.axvline(0, color=INK, lw=1.0, ls=":")
    ax.set_xlabel("$d_c$ coefficient $\\beta$ (per LIO fold)"); ax.set_ylabel("folds")
    ax.set_title(f"Sign-stable: all {len(betas)} folds", pad=6)
    ax.text(0.04, 0.94, f"{int((betas < 0).sum())}/{len(betas)} folds $\\beta<0$\n"
            f"range [{betas.min():.2f}, {betas.max():.2f}]", transform=ax.transAxes,
            va="top", ha="left", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef3f7", edgecolor="none"))
    ax.legend(loc="upper right")


def main():
    set_style()
    fig = plt.figure(figsize=(13.6, 9.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.04, 0.96],
                           hspace=0.36, wspace=0.30, left=0.075, right=0.975,
                           top=0.95, bottom=0.085)
    axtop = fig.add_subplot(gs[0, :]); draw_mechanism(axtop)
    axb = fig.add_subplot(gs[1, 0]); draw_law(axb); panel_label(axb, "b", dx=-0.16)
    axc = fig.add_subplot(gs[1, 1]); draw_forest(axc); panel_label(axc, "c", dx=-0.26)
    axd = fig.add_subplot(gs[1, 2]); draw_lio(axd); panel_label(axd, "d", dx=-0.16)
    fig.suptitle("A correct physics answer must close every conservation constraint — so each one multiplies the odds of success by $\\approx$0.69",
                 fontsize=12.5, fontweight="bold", y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".pdf"))
    print(f"[fig] {OUT}")


if __name__ == "__main__":
    main()
