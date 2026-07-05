"""F12: zero-LLM rule-based d_c floor — construct-validity figure (no API, no LLM).

Reads evaluation/rule_based_dc_floor_items_20260529.csv (produced by
31_rule_based_dc_floor.py) and renders two panels:

  (a) Known-groups validity: mean d_c by PhysReason difficulty tier for the human
      benchmark-author labels vs the zero-LLM regex (rule_qs, rule_q). Both label
      sources should rise monotonically across difficulty if d_c indexes
      conservation structure rather than noise.
  (b) Per-family presence agreement (Cohen kappa, rule_qs vs human), showing the
      single-scalar families (energy, angular momentum, entropy) agree strongly
      while the deliberate keyword floor under-detects the two rarest families.

All quantities are recomputed from the item CSV; nothing is hard-coded.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "evaluation" / "rule_based_dc_floor_items_20260529.csv"
OUT = ROOT / "figures" / "F12_rule_based_dc_floor.png"

FAMILIES = ["momentum", "angular_momentum", "energy", "charge", "mass", "entropy"]
FAM_LABEL = {"momentum": "mom.", "angular_momentum": "ang.\nmom.",
             "energy": "energy", "charge": "charge", "mass": "mass", "entropy": "2nd-law"}
DIFF_ORDER = ["knowledge", "easy", "medium", "difficult"]


def cohen_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    p1a, p1b = float(np.mean(a)), float(np.mean(b))
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")


def main():
    rows = list(csv.DictReader(open(ITEMS, encoding="utf-8-sig")))

    # --- panel (a): mean d_c by difficulty tier ---
    by_diff = {d: {"human": [], "rqs": [], "rq": []} for d in DIFF_ORDER}
    for r in rows:
        d = (r["difficulty"] or "").lower()
        if d in by_diff:
            by_diff[d]["human"].append(int(r["human_dc"]))
            by_diff[d]["rqs"].append(int(r["rule_qs_dc"]))
            by_diff[d]["rq"].append(int(r["rule_q_dc"]))
    means = {k: [np.mean(by_diff[d][k]) for d in DIFF_ORDER] for k in ("human", "rqs", "rq")}
    ns = [len(by_diff[d]["human"]) for d in DIFF_ORDER]

    # --- panel (b): per-family Cohen kappa (rule_qs vs human) ---
    def fams(cell):
        return set(x for x in (cell or "").split("|") if x)
    kappas = []
    for fam in FAMILIES:
        a = [1 if fam in fams(r["rule_qs_families"]) else 0 for r in rows]
        b = [1 if fam in fams(r["human_families"]) else 0 for r in rows]
        kappas.append(cohen_kappa(a, b))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # panel (a)
    x = np.arange(len(DIFF_ORDER))
    w = 0.27
    ax1.bar(x - w, means["human"], w, label="human theorem labels", color="C0", zorder=3)
    ax1.bar(x, means["rqs"], w, label="zero-LLM regex (q+solution)", color="C1", zorder=3)
    ax1.bar(x + w, means["rq"], w, label="zero-LLM regex (question only)", color="C2", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(DIFF_ORDER, ns)])
    ax1.set_ylabel(r"mean $d_c$ (conservation-family count)")
    ax1.set_title("(a) Known-groups validity across PhysReason difficulty tiers\n"
                  "human and zero-LLM labels track the same monotone gradient", fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x - w, means["human"]):
        ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    for xi, v in zip(x, means["rqs"]):
        ax1.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    # panel (b)
    xb = np.arange(len(FAMILIES))
    colors = ["C3" if (np.isnan(k) or k < 0.3) else "C0" for k in kappas]
    kplot = [0.0 if np.isnan(k) else k for k in kappas]
    ax2.bar(xb, kplot, 0.62, color=colors, zorder=3)
    ax2.axhline(0.6, color="k", ls="--", lw=1.0, alpha=0.6)
    ax2.annotate(r"$\kappa=0.6$", xy=(1.0, 0.6), xycoords=("axes fraction", "data"),
                 xytext=(-4, 6), textcoords="offset points",
                 ha="right", va="bottom", fontsize=7.0, color="#666",
                 bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                           edgecolor="none", alpha=0.9))
    ax2.set_xticks(xb)
    ax2.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=8.5)
    ax2.set_ylabel(r"Cohen $\kappa$  (zero-LLM regex vs human, presence per family)")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) Per-family agreement: zero-LLM regex vs human (n=1200)\n"
                  "single-scalar families agree; rare charge/mass under-detected", fontsize=10.5)
    ax2.grid(True, axis="y", alpha=0.3)
    for xi, k in zip(xb, kappas):
        lab = "n/a" if np.isnan(k) else f"{k:.2f}"
        val = 0.0 if np.isnan(k) else k
        if abs(val - 0.6) < 0.08:
            ax2.text(xi, max(val - 0.085, 0.08), lab, ha="center", va="top",
                     fontsize=7.3, color="white", fontweight="bold")
        else:
            ax2.text(xi, val + 0.035, lab, ha="center", va="bottom", fontsize=7.3,
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                               edgecolor="none", alpha=0.85))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.savefig(OUT.with_suffix(".pdf"))
    print(f"[fig] {OUT}")
    print(f"[panel a] means human={[round(v,3) for v in means['human']]} "
          f"rqs={[round(v,3) for v in means['rqs']]} rq={[round(v,3) for v in means['rq']]}")
    print(f"[panel b] kappa={[None if np.isnan(k) else round(k,3) for k in kappas]}")


if __name__ == "__main__":
    main()
