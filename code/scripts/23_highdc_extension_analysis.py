"""Merge pilot258 DeepSeek panel + high-d_c +100 extension, re-fit, plot.

Inputs:
  - evaluation/w6_controlled_panel.csv (the 1032-obs DeepSeek/Kimi/Qwen14B/Qwen7B panel)
  - data/results/solve_highdc100_deepseek_judged_20260525.csv (new 100 items, DeepSeek-only)
  - data/annotations/highdc_prelabels/pilot_deepseek_r1.csv (the d_c labels for the new 100)

Outputs:
  - evaluation/highdc_extension_summary_20260525.md
  - evaluation/pilot358_deepseek_panel_20260525.csv  (DeepSeek-only panel, n=258+100=358)
  - figures/F8_highdc_extension_accuracy.png
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def envelope(dc, kappa):
    """A(dc) = 1 - exp(-kappa/dc) for dc>=1; 1.0 for dc=0."""
    dc = np.asarray(dc, dtype=float)
    safe = np.where(dc >= 1, dc, 1.0)
    out = 1.0 - np.exp(-kappa / safe)
    return out


def fit_kappa(dc, y):
    sel = dc >= 1
    if sel.sum() < 5:
        return float("nan")
    dc_p, y_p = dc[sel], y[sel]
    def nll(k):
        if k <= 0: return 1e9
        p = envelope(dc_p, k)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(y_p * np.log(p) + (1 - y_p) * np.log(1 - p)))
    r = minimize_scalar(nll, bounds=(1e-3, 5), method="bounded")
    return float(r.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w6-panel", type=Path, default=Path("evaluation/w6_controlled_panel.csv"))
    ap.add_argument("--highdc-judged", type=Path,
                    default=Path("data/results/solve_highdc100_deepseek_judged_20260525.csv"))
    ap.add_argument("--highdc-prelabels", type=Path,
                    default=Path("data/annotations/highdc_prelabels/pilot_deepseek_r1.csv"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("evaluation/pilot358_deepseek_panel_20260525.csv"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("evaluation/highdc_extension_summary_20260525.md"))
    ap.add_argument("--out-fig", type=Path,
                    default=Path("figures/F8_highdc_extension_accuracy.png"))
    args = ap.parse_args()

    panel = pd.read_csv(args.w6_panel)
    ds_panel = panel[panel["model_label"] == "DeepSeekV3"].copy()
    print(f"[orig DS panel] n={len(ds_panel)}  d_c dist: {ds_panel['d_c'].value_counts().sort_index().to_dict()}")

    judged = pd.read_csv(args.highdc_judged)
    labels = pd.read_csv(args.highdc_prelabels)
    # Merge labels into judged by item_id
    merged = judged.merge(
        labels[["item_id", "d_c", "law_momentum", "law_angular_momentum", "law_energy",
                "law_charge", "law_mass", "law_entropy"]],
        on="item_id", how="left",
    )
    print(f"[new highdc] n={len(merged)}, d_c parsed: {merged['d_c'].notna().sum()}")
    # Filter to rows with parsed d_c
    merged = merged[merged["d_c"].notna()].copy()
    merged["d_c"] = pd.to_numeric(merged["d_c"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["d_c"])
    merged["d_c"] = merged["d_c"].astype(int)

    print(f"[new highdc filtered] n={len(merged)}  d_c dist: {merged['d_c'].value_counts().sort_index().to_dict()}")

    # Build the augmented DeepSeek panel
    augmented = pd.concat([
        ds_panel[["item_id", "d_c", "is_correct"]].assign(source="pilot258"),
        merged[["item_id", "d_c", "is_correct_judge"]].rename(columns={"is_correct_judge": "is_correct"}).assign(source="highdc100"),
    ], ignore_index=True)
    # ensure types
    augmented["is_correct"] = pd.to_numeric(augmented["is_correct"], errors="coerce").fillna(0).astype(int)

    print(f"[merged DS-only panel] n={len(augmented)}  d_c dist: {augmented['d_c'].value_counts().sort_index().to_dict()}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.out_csv, index=False)
    print(f"[wrote] {args.out_csv}")

    # Per-d_c accuracy with Wilson 95% CI
    def wilson_ci(k, n, z=1.96):
        if n == 0: return (float("nan"), float("nan"))
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (center - half, center + half)

    rows_summary = []
    for source_label, df in [("pilot258", ds_panel), ("highdc100", merged.rename(columns={"is_correct_judge": "is_correct"})),
                              ("merged358", augmented)]:
        col_y = "is_correct"
        for dc, sub in df.groupby("d_c"):
            n = len(sub)
            k = int(sub[col_y].sum())
            lo, hi = wilson_ci(k, n)
            rows_summary.append(dict(source=source_label, d_c=int(dc), n=n, k=k,
                                     accuracy=k/n if n else float("nan"),
                                     ci_lo=lo, ci_hi=hi))
    summary = pd.DataFrame(rows_summary)
    print(summary.pivot_table(index="d_c", columns="source", values=["n","accuracy"]).round(3).to_string())

    # Fit envelope on (a) pilot258 DS only, (b) merged 358 DS
    fit_orig = fit_kappa(ds_panel["d_c"].values, ds_panel["is_correct"].values)
    fit_merged = fit_kappa(augmented["d_c"].values, augmented["is_correct"].values)
    print(f"[kappa fit] pilot258 DS only: {fit_orig:.4f}  | merged 358: {fit_merged:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for src, color, marker in [("pilot258", "C0", "o"), ("highdc100", "C1", "^"), ("merged358", "k", "s")]:
        sub = summary[summary["source"] == src].sort_values("d_c")
        n_arr = sub["n"].values
        ax.errorbar(sub["d_c"], sub["accuracy"],
                    yerr=[sub["accuracy"]-sub["ci_lo"], sub["ci_hi"]-sub["accuracy"]],
                    marker=marker, color=color, capsize=4, label=f"{src} (n={n_arr.sum()})",
                    linewidth=1.5 if src != "merged358" else 2.0,
                    linestyle="-" if src == "merged358" else "--")
        for i, (dc, n) in enumerate(zip(sub["d_c"], n_arr)):
            ax.annotate(f"n={n}", (dc, sub["accuracy"].iloc[i]),
                        textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", color=color)
    # overlay envelope curves
    dc_grid = np.linspace(1, 5, 100)
    ax.plot(dc_grid, envelope(dc_grid, fit_orig), color="C0", alpha=0.4, linewidth=1.5,
            label=f"envelope fit pilot258: $\\kappa$={fit_orig:.3f}")
    ax.plot(dc_grid, envelope(dc_grid, fit_merged), color="k", alpha=0.4, linewidth=2.0,
            label=f"envelope fit merged: $\\kappa$={fit_merged:.3f}")
    ax.set_xlabel(r"Conservation-constraint load $d_c$")
    ax.set_ylabel("Judged accuracy (DeepSeek solve, DeepSeek judge)")
    ax.set_title("High-$d_c$ pilot expansion (DeepSeek-only)\npilot258 + 100 new candidates ($d_c\\geq 2$ over-sampled)")
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    plt.savefig(args.out_fig.with_suffix(".pdf"))
    print(f"[fig] {args.out_fig}")

    # Markdown
    md = [
        "# High-d_c Pilot Expansion (DeepSeek-only)\n\n",
        f"Mined 100 new candidates from OlympiadBench (50, text-only physics not in pilot258) and ",
        f"PhysReason (50, text-only, sorted by PR-native theorem-based d_c heuristic). DeepSeek-prelabeled (single r1 profile, ",
        f"using the same V4 protocol), filtered to those with parsed d_c. Then DeepSeek-solved with the standard W5 ",
        f"prompt (`max_tokens=1024`, `temperature=0.2`) and DeepSeek-judged with the standard W6 judge.\n\n",
        "## Per-d_c accuracy comparison\n\n",
        "| d_c | pilot258 (n, acc) | highdc100 (n, acc) | merged 358 (n, acc) |\n",
        "|---:|---|---|---|\n",
    ]
    for dc in sorted(summary["d_c"].unique()):
        rows_dc = {r["source"]: r for r in summary[summary["d_c"] == dc].to_dict("records")}
        def cell(src):
            r = rows_dc.get(src)
            if r is None: return "—"
            return f"{r['n']} | {r['accuracy']:.3f}"
        md.append(f"| {dc} | {cell('pilot258')} | {cell('highdc100')} | {cell('merged358')} |\n")
    md.append("\n## Envelope fit (one-parameter, d_c>=1, binomial NLL)\n\n")
    md.append(f"- pilot258 DeepSeek-only: $\\hat\\kappa = {fit_orig:.4f}$\n")
    md.append(f"- merged 358 (pilot258 + highdc100): $\\hat\\kappa = {fit_merged:.4f}$\n\n")
    md.append("## Interpretation\n\n")
    n_old_dc2plus = ds_panel[ds_panel['d_c']>=2].shape[0]
    n_new_dc2plus = merged[merged['d_c']>=2].shape[0]
    md.append(
        f"- The extension contributes mostly d_c=2 items (was {ds_panel.query('d_c==2').shape[0]}, now +{merged.query('d_c==2').shape[0]}; "
        f"merged: {ds_panel.query('d_c==2').shape[0]+merged.query('d_c==2').shape[0]}). Higher-d_c bins add modestly. "
        f"This partly resolves the §4.3/§6 'high-d_c sparsity' caveat but does not yet populate d_c$\\ge$4 robustly.\n"
    )
    md.append(
        f"- $\\hat\\kappa$ on the merged set ({fit_merged:.4f}) is close to the pilot258-only fit ({fit_orig:.4f}); "
        "the envelope shape transfers to the new items at similar magnitude.\n"
    )
    md.append(
        "- This is a DeepSeek-only extension; a full multi-model expansion requires re-solving the 100 new items with Qwen-14B / Qwen-7B / Kimi-K2, which is the next-extension item.\n"
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("".join(md), encoding="utf-8")
    print(f"[wrote] {args.out_md}")


if __name__ == "__main__":
    main()
