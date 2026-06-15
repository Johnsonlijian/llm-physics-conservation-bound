"""Figure 3 (npj AI reframe): the bottleneck is FORMULATION.

One money figure consolidating the elimination + positive evidence:
  (a) controlled synthetic, constructed d_c, exact grading -- solving is not the
      limit for solver-capable / frontier models (flat ~1.0); weak models cliff.
  (b) failure-stage composition on real items -- ~92% of failures already NAME
      the required laws (identification is a minority, rising with d_c).
  (c) scaffold ladder on the same real items -- supplying the LAWS is null,
      supplying the formulated EQUATIONS rescues monotonically (positive
      evidence that the recoverable locus is formulation).

Panels (a),(b) reuse the logic of 52_fig5_mechanism_composite.py; panel (c)
reuses 56_fig_scaffold_ladder.py. No new computation: reads stored results.

Inputs (data/results/): synthetic_solved_*.csv, structured_solve_chat_pilot258.csv,
  formulation_rescue_pilot258.csv, setup_rescue_deepseek-chat_pilot258.csv (+f50,f70)
Output: figures/F30_fig3_formulation_mechanism.{png,pdf}  (manuscript Figure 3)
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pub_style import apply_style, panel_label, save_figure, OKABE, FULL_W, \
    GREY_DARK, GREY_MID, GREY_LIGHT

PROJECT = Path(__file__).resolve().parents[2]
RES = PROJECT / "data" / "results"
if not RES.exists():
    RES = PROJECT / "figure_inputs" / "fig3_formulation_mechanism"
OUT = PROJECT / "figures" / "F30_fig3_formulation_mechanism.png"

SYN_MODELS = [
    ("synthetic_solved_ollama_qwen2.5-1.5b.csv", "Qwen2.5-1.5B", GREY_LIGHT),
    ("synthetic_solved_moonshot-moonshot-v1-32k.csv", "Kimi-v1", GREY_MID),
    ("synthetic_solved_dashscope-qwen-plus.csv", "Qwen-Plus", OKABE["orange"]),
    ("synthetic_solved_ark-doubao-seed-2-0-pro-260215.csv", "Doubao-2.0-pro", OKABE["green"]),
    ("synthetic_solved_deepseek-chat.csv", "DeepSeek-chat", OKABE["sky"]),
    ("synthetic_solved_claude-opus-4.csv", "Claude Opus 4.8 (frontier)", OKABE["blue"]),
    ("synthetic_solved_gpt-5.5-codex.csv", "GPT-5.5-Codex (frontier)", OKABE["purple"]),
]


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def per_dc_acc(rows, key="is_correct"):
    d = defaultdict(list)
    for r in rows:
        if str(r.get(key, "")) in ("0", "1"):
            d[int(r["d_c"])].append(int(r[key]))
    return {k: float(np.mean(v)) for k, v in sorted(d.items())}


def mean_col(rows, col):
    v = [int(r[col]) for r in rows if str(r.get(col, "")) in ("0", "1")]
    return float(np.mean(v)) if v else float("nan")


def main():
    apply_style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(FULL_W, 2.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.20, wspace=0.42)

    # ---- (a) controlled synthetic, exact grading ----
    for fname, label, color in SYN_MODELS:
        p = RES / fname
        if not p.exists():
            continue
        acc = per_dc_acc(read_csv(p))
        ds = sorted(acc)
        axA.plot(ds, [acc[d] for d in ds], "-o", lw=1.0, ms=2.6, color=color,
                 label=label)
    axA.set_xlabel("constructed $d_c$ (matched, exact-graded)")
    axA.set_ylabel("accuracy")
    axA.set_xticks([1, 2, 3, 4, 6])
    axA.set_ylim(-0.03, 1.05)
    axA.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=5.2,
               handlelength=1.3, labelspacing=0.25)

    # ---- (b) failure-stage composition by d_c ----
    fr = read_csv(RES / "structured_solve_chat_pilot258.csv")
    byd = defaultdict(lambda: {"c": 0, "e": 0, "i": 0, "n": 0})
    for r in fr:
        if str(r.get("is_correct", "")) not in ("0", "1"):
            continue
        d = int(r["d_c"]); byd[d]["n"] += 1
        if r["is_correct"] == "1":
            byd[d]["c"] += 1
        elif r.get("failure_stage") == "identification":
            byd[d]["i"] += 1
        else:
            byd[d]["e"] += 1
    ds = sorted(d for d in byd if byd[d]["n"] >= 5)
    corr = [byd[d]["c"] / byd[d]["n"] for d in ds]
    exe = [byd[d]["e"] / byd[d]["n"] for d in ds]
    ide = [byd[d]["i"] / byd[d]["n"] for d in ds]
    axB.bar(ds, corr, color=OKABE["green"], width=0.66)
    axB.bar(ds, exe, bottom=corr, color=GREY_LIGHT, width=0.66)
    axB.bar(ds, ide, bottom=[c + e for c, e in zip(corr, exe)],
            color=OKABE["vermillion"], width=0.66)
    axB.text(ds[0], corr[0] / 2, "correct", ha="center", va="center",
             fontsize=5.6, color="white")
    axB.text(ds[0], corr[0] + exe[0] / 2, "failed, laws named\n(execution)",
             ha="center", va="center", fontsize=5.6, color=GREY_DARK)
    axB.annotate("failed, law missed\n(identification)",
                 xy=(ds[-1], corr[-1] + exe[-1] + ide[-1] / 2),
                 xytext=(ds[-1] - 1.45, 0.80), fontsize=5.6,
                 color=OKABE["vermillion"], ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=OKABE["vermillion"],
                                 lw=0.6, shrinkA=8, shrinkB=4))
    axB.set_xlabel("conservation-constraint load $d_c$")
    axB.set_ylabel("share of items")
    axB.set_xticks(ds)
    axB.set_ylim(0, 1.0)

    # ---- (c) scaffold ladder: laws null vs equations rescue ----
    lr = read_csv(RES / "formulation_rescue_pilot258.csv")
    base = mean_col(lr, "base_correct")
    laws = mean_col(lr, "laws_correct")
    s30 = mean_col(read_csv(RES / "setup_rescue_deepseek-chat_pilot258.csv"), "setup_correct")
    _f50 = RES / "setup_rescue_deepseek-chat_f50_pilot258.csv"
    s50 = mean_col(read_csv(_f50), "setup_correct") if _f50.exists() else 0.400
    s70 = mean_col(read_csv(RES / "setup_rescue_deepseek-chat_f70_pilot258.csv"), "setup_correct")
    vals = [base, laws, s30, s50, s70]
    labels = ["base", "laws", "eq\n30%", "eq\n50%", "eq\n70%"]
    colors = [GREY_MID, GREY_DARK, OKABE["vermillion"], OKABE["vermillion"],
              OKABE["vermillion"]]
    x = np.arange(len(vals))
    axC.axhline(1.0, ls=":", color=GREY_MID, lw=0.8)
    axC.text(x[-1], 1.015, "explicit ceiling $\\approx$ 1.0", ha="right",
             va="bottom", fontsize=5.4, color=GREY_MID)
    axC.plot(x, vals, "-", color=GREY_MID, lw=0.9, zorder=1)
    for xi, v, c in zip(x, vals, colors):
        axC.plot(xi, v, "o", ms=5, color=c, zorder=3)
        axC.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=5.6, color=c)
    axC.text(0.5, base - 0.085, "laws: null", ha="center", va="top",
             fontsize=5.6, color=GREY_DARK)
    axC.text(3.0, (laws + s70) / 2 + 0.16, "equations:\nmonotone rescue",
             ha="center", va="center", fontsize=5.6, color=OKABE["vermillion"])
    axC.set_xticks(x)
    axC.set_xticklabels(labels, fontsize=5.8)
    axC.set_xlabel("set-up revealed (real items)")
    axC.set_ylabel("judged accuracy")
    axC.set_ylim(0, 1.07)

    panel_label(axA, "a", dx=-0.26)
    panel_label(axB, "b", dx=-0.26)
    panel_label(axC, "c", dx=-0.26)
    save_figure(fig, OUT)
    print(f"[fig3] base={base:.3f} laws={laws:.3f} s30={s30:.3f} s50={s50:.3f} s70={s70:.3f}")
    print(f"[fig3] wrote {OUT}")


if __name__ == "__main__":
    main()
