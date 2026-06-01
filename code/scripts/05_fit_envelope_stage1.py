"""W5 Stage 1: 单 model envelope 单变量初拟合 + 散点图。

输入:
  --graded   data/results/solve_pilot258_qwen14b_graded.csv (含 item_id, is_correct)
  --dc-csv   data/annotations/pilot258_v4a_qwen14b_merged.csv (含 item_id, rater, d_c)

输出:
  --fig      figures/F1_qwen14b_accuracy_vs_dc.png
  --report   evaluation/W5_stage1_fit_report.md
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def envelope_1d(d, kappa_eff):
    """1 - exp(-kappa_eff / d_c).  kappa_eff = κρ^α B^β L^γ for this model."""
    return 1 - np.exp(-kappa_eff / d)


def consensus_dc(dc_csv: Path) -> dict[str, int]:
    """Median d_c across 4 raters per item."""
    groups: dict[str, list[int]] = defaultdict(list)
    with open(dc_csv, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            iid = r.get("item_id", "")
            dc = r.get("d_c", "").strip()
            if iid and dc:
                try:
                    groups[iid].append(int(dc))
                except ValueError:
                    pass
    return {iid: int(np.median(vs)) for iid, vs in groups.items() if vs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", type=Path, required=True)
    ap.add_argument("--dc-csv", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    # Load graded
    accuracy: dict[str, int] = {}
    with open(args.graded, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                accuracy[r["item_id"]] = int(r["is_correct"])
            except (KeyError, ValueError):
                pass
    print(f"[load] {len(accuracy)} graded items")

    # Load d_c consensus
    dc_map = consensus_dc(args.dc_csv)
    print(f"[load] {len(dc_map)} d_c-labeled items")

    # Join
    pairs = [(dc_map[iid], accuracy[iid]) for iid in accuracy if iid in dc_map]
    if not pairs:
        raise SystemExit("No overlap between graded and dc-labeled items")
    print(f"[join] {len(pairs)} pairs")

    d_arr = np.array([p[0] for p in pairs], dtype=float)
    a_arr = np.array([p[1] for p in pairs], dtype=float)
    overall_acc = a_arr.mean()
    print(f"[overall] accuracy = {overall_acc:.3f}")

    # Group by d_c
    d_unique = sorted(set(d_arr.tolist()))
    grouped: dict[float, np.ndarray] = {d: a_arr[d_arr == d] for d in d_unique}
    print(f"\n[per d_c]  n  acc")
    for d in d_unique:
        arr = grouped[d]
        print(f"  d_c={int(d):2d}  n={len(arr):3d}  acc={arr.mean():.3f}")

    # Fit envelope 1 - exp(-kappa_eff / d_c) to per-item binary
    d_for_fit = d_arr[d_arr > 0]
    a_for_fit = a_arr[d_arr > 0]
    try:
        popt, pcov = curve_fit(envelope_1d, d_for_fit, a_for_fit,
                               p0=[1.0], bounds=(0.01, 100))
        kappa_eff = popt[0]
        kappa_ci = 1.96 * np.sqrt(pcov[0, 0])
        # Compute R^2 on group means (more stable than per-item binary)
        d_grp = np.array(d_unique)
        a_grp = np.array([grouped[d].mean() for d in d_unique])
        pred = envelope_1d(d_grp, kappa_eff)
        ss_res = np.sum((a_grp - pred) ** 2)
        ss_tot = np.sum((a_grp - a_grp.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception as e:
        kappa_eff, kappa_ci, r2 = float("nan"), float("nan"), float("nan")
        print(f"[warn] fit failed: {e}")

    print(f"\n[fit] kappa_eff = {kappa_eff:.4f} (±95% CI {kappa_ci:.4f})")
    print(f"[fit] R^2 (group means) = {r2:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    # Group means with error bars (Wilson CI)
    for d in d_unique:
        arr = grouped[d]
        n = len(arr)
        p = arr.mean()
        # Wilson 95% half-width
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        ax.errorbar(d, p, yerr=half, fmt="o", color="C0", capsize=4, markersize=6)
        ax.annotate(f"n={n}", (d, p), textcoords="offset points",
                    xytext=(8, -3), fontsize=8, alpha=0.6)
    # Fit curve
    if not np.isnan(kappa_eff):
        d_dense = np.linspace(max(0.5, min(d_unique)), max(d_unique) + 1, 100)
        ax.plot(d_dense, envelope_1d(d_dense, kappa_eff), "C1-",
                label=fr"$\bar A = 1 - e^{{-{kappa_eff:.2f}/d_c}}$  ($R^2$={r2:.3f})")
    ax.set_xlabel(r"Consensus conservation-constraint load $d_c$")
    ax.set_ylabel("Per-item accuracy (n shown)")
    ax.set_title(f"W5 Stage 1 envelope (single model: Qwen-14B)\nN={len(pairs)} items  overall={overall_acc:.3f}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig, dpi=150)
    plt.savefig(args.fig.with_suffix(".pdf"))
    print(f"\n[fig] {args.fig}")

    # Report
    lines = [
        f"# W5 Stage 1 — Qwen-14B envelope fit\n",
        f"- Pilot: {len(pairs)} items (subset of 258 where both graded & d_c-labeled present)\n",
        f"- Overall accuracy: {overall_acc:.3f}\n\n",
        f"## Per-d_c breakdown\n\n",
        f"| d_c | n | accuracy |\n|---|---:|---:|\n",
    ]
    for d in d_unique:
        arr = grouped[d]
        lines.append(f"| {int(d)} | {len(arr)} | {arr.mean():.3f} |\n")
    lines.append("\n## 1-parameter envelope fit\n\n")
    lines.append(f"- Model:  $\\bar A(d_c) = 1 - \\exp(-\\kappa_{{eff}}/d_c)$\n")
    lines.append(f"- $\\kappa_{{eff}}$ = {kappa_eff:.4f} (±95% CI {kappa_ci:.4f})\n")
    lines.append(f"- $R^2$ on group means = {r2:.4f}\n")
    lines.append("\n## Verdict\n\n")
    if r2 >= 0.85:
        lines.append(f"- $R^2$ = {r2:.3f} ≥ 0.85 → envelope single-variable shape **supports C1/C2 qualitatively**. Stage 2 will add multi-model + L_c/B_t/ρ_p for full fit.\n")
    elif r2 >= 0.70:
        lines.append(f"- $R^2$ = {r2:.3f} in [0.70, 0.85) → envelope shape OK but not paper-grade. Stage 2 needs more model diversity to discriminate.\n")
    else:
        lines.append(f"- $R^2$ = {r2:.3f} < 0.70 → single-variable fit weak. Stage 2 must check d_c is not confounded.\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(lines), encoding="utf-8")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
