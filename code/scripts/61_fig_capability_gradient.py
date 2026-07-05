"""Figure: the d_c penalty attenuates with model capability (the capability gradient).

For every model with a real-pilot arm, compute (i) overall judged accuracy (a capability proxy) and
(ii) the univariate per-+1-d_c odds ratio with an item-bootstrap 95% CI. Plot OR vs overall accuracy.
The story: weak/mid models pay a strong per-constraint penalty (OR ~0.69); the two frontier
agent-interface arms sit near OR 1 -- they have largely escaped the penalty. Because this
is measured on the SAME real items, a generic "hard text" confound cannot produce a penalty that is
present for weak models and absent for frontier models; the d_c effect is capability-modulated, which
is itself evidence that d_c is separable from generic text difficulty.

Reads the 4-model controlled panel, the reasoner arm, and the two frontier real arms.
Writes figures/F26_capability_gradient.{png,pdf}.
"""
from __future__ import annotations
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
RES = PROJECT / "data" / "results"
if not RES.exists():
    RES = PROJECT / "figure_inputs" / "capability_gradient"
PANEL = PROJECT / "evaluation" / "w6_controlled_panel_with_logs.csv"
REAS = RES / "solve_pilot258_deepseek_reasoner_judged_20260529.csv"
OUT = PROJECT / "figures" / "F26_capability_gradient.png"


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def logit(xs, ys):
    b0 = b1 = 0.0
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
            w = max(p * (1 - p), 1e-9)
            g0 += y - p; g1 += (y - p) * x
            h00 += w; h01 += w * x; h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        b0 += (h11 * g0 - h01 * g1) / det
        b1 += (-h01 * g0 + h00 * g1) / det
    return b1


def model_stats(pairs, seed=7):
    """pairs: list of (d_c, is_correct). Returns (acc, OR, lo, hi)."""
    xs = [d for d, _ in pairs]; ys = [c for _, c in pairs]
    acc = sum(ys) / len(ys)
    orr = math.exp(logit(xs, ys))
    rng = random.Random(seed); ors = []
    n = len(pairs)
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            ors.append(math.exp(logit([xs[i] for i in idx], [ys[i] for i in idx])))
        except Exception:
            pass
    ors.sort()
    return acc, orr, ors[int(0.025 * len(ors))], ors[int(0.975 * len(ors))]


def main():
    dc_map = {}
    for r in read_csv(PANEL):
        dc_map.setdefault(r["item_id"], int(float(r["d_c"])))

    models = []  # (label, pairs, family)
    # 4-model controlled panel
    panel_rows = defaultdict(list)
    nice = {"Qwen7B-Ollama": "Qwen2.5-7B", "Qwen14B": "Qwen-14B", "KimiK2": "Kimi-K2", "DeepSeekV3": "DeepSeek-chat"}
    for r in read_csv(PANEL):
        if str(r.get("is_correct", "")) in ("0", "1"):
            panel_rows[r["model_label"]].append((int(float(r["d_c"])), int(r["is_correct"])))
    for ml, pairs in panel_rows.items():
        models.append((nice.get(ml, ml), pairs, "panel"))
    # reasoner
    if REAS.exists():
        rp = []
        for r in read_csv(REAS):
            v = str(r.get("is_correct_judge", "")).strip()
            if v in ("0", "1") and r["item_id"] in dc_map:
                rp.append((dc_map[r["item_id"]], int(v)))
        if rp:
            models.append(("DeepSeek-reasoner", rp, "reasoner"))
    # frontier real arms
    for fname, label in [("realpilot_openai_codex_arm.csv", "OpenAI-Codex arm"),
                         ("realpilot_anthropic_agent_arm.csv", "Anthropic-agent arm")]:
        p = RES / fname
        if p.exists():
            pr = [(int(x["d_c"]), int(x["is_correct"])) for x in read_csv(p)
                  if str(x.get("is_correct", "")) in ("0", "1")]
            models.append((label, pr, "frontier"))

    stats = []
    for label, pairs, fam in models:
        acc, orr, lo, hi = model_stats(pairs)
        stats.append((label, acc, orr, lo, hi, fam, len(pairs)))
        print(f"  {label:20s} n={len(pairs):4d} acc={acc:.3f} OR={orr:.3f} CI[{lo:.3f},{hi:.3f}]")
    stats.sort(key=lambda s: s[1])  # by accuracy

    from pub_style import apply_style, save_figure, OKABE, GREY_DARK, GREY_MID, INK
    apply_style()
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.15)
    cmap = {"panel": OKABE["vermillion"], "reasoner": OKABE["orange"],
            "frontier": OKABE["blue"]}
    ax.axhline(1.0, ls="--", color=GREY_MID, lw=0.7, zorder=1)
    ax.text(0.015, 1.008, "OR = 1 (no per-constraint penalty)",
            transform=ax.get_yaxis_transform(), fontsize=5.8, color=GREY_MID,
            va="bottom")

    for label, acc, orr, lo, hi, fam, n in stats:
        ax.errorbar(acc, orr, yerr=[[orr - lo], [hi - orr]], fmt="o", ms=4.6,
                    color=cmap[fam], ecolor=cmap[fam], elinewidth=0.8,
                    capsize=2.0, zorder=3)
        dy = 0.055 if label not in ("DeepSeek-chat", "Kimi-K2") else -0.075
        ha = "left" if acc < 0.5 else "right"
        ax.annotate(f"{label}\n(OR {orr:.2f})", (acc, orr),
                    textcoords="offset points",
                    xytext=(6 if ha == "left" else -6, 8 if dy > 0 else -16),
                    ha=ha, fontsize=5.6, color=INK)

    # trend guide
    accs = [s[1] for s in stats]; ors_ = [s[2] for s in stats]
    z = np.polyfit(accs, ors_, 1)
    xx = np.linspace(min(accs) - 0.02, max(accs) + 0.03, 50)
    ax.plot(xx, np.polyval(z, xx), "-", color=GREY_MID, lw=0.8, alpha=0.7, zorder=2)

    ax.set_xlabel("model capability (overall judged accuracy, real pilot)")
    ax.set_ylabel("per-constraint penalty (OR per $+1\\,d_c$)")
    ax.set_ylim(0.41, 1.27)
    ax.set_xlim(0.0, 0.68)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[k],
                  markersize=5, label=v)
           for k, v in [("panel", "4-family panel"),
                        ("reasoner", "reasoning model"),
                        ("frontier", "frontier interface arms")]]
    ax.legend(handles=leg, loc="lower right", title="model class")
    save_figure(fig, OUT)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
