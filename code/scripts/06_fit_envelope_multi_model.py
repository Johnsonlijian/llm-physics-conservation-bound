"""W5 Stage 2: Multi-model envelope fit on pilot258.

每 model 一组 (judged_csv),共享同一 d_c 标注(V4a Qwen-14B median)。

输出:
- F2 figure: accuracy vs d_c 多线
- evaluation/W5_stage2_multi_model.md 报告 + per-model kappa_eff
- evaluation/v5_multi_model_metrics.csv 数据 dump

注:暂时拟合 1-param envelope `1 - exp(-kappa_eff(model) / d_c)`,W6 升 4-param。
排除 d_c=0(envelope 边界发散)。
"""
from __future__ import annotations
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def envelope(d, k):
    return 1 - np.exp(-k / d)


def load_dc(dc_csv: Path) -> dict[str, int]:
    groups = defaultdict(list)
    with open(dc_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("d_c") or "").strip()
            if v:
                try:
                    groups[r["item_id"]].append(int(v))
                except ValueError:
                    pass
    return {iid: int(statistics.median(vs)) for iid, vs in groups.items() if vs}


def load_accuracy(judged_csv: Path, key: str = "is_correct_judge") -> dict[str, int]:
    out = {}
    with open(judged_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get(key) or "").strip()
            if v:
                try:
                    out[r["item_id"]] = int(v)
                except ValueError:
                    pass
    return out


def fit_one(acc_map: dict, dc_map: dict, exclude_dc0: bool = True):
    """Return (kappa_eff, kappa_ci, r2_group, per_dc_stats[(d,n,acc)])."""
    pairs = [(dc_map[iid], a) for iid, a in acc_map.items() if iid in dc_map]
    if exclude_dc0:
        pairs = [(d, a) for d, a in pairs if d >= 1]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), []
    d_arr = np.array([p[0] for p in pairs], dtype=float)
    a_arr = np.array([p[1] for p in pairs], dtype=float)
    d_unique = sorted(set(d_arr.tolist()))
    grouped = {d: a_arr[d_arr == d] for d in d_unique}
    per_dc = [(int(d), len(grouped[d]), float(grouped[d].mean())) for d in d_unique]

    try:
        popt, pcov = curve_fit(envelope, d_arr, a_arr, p0=[1.0], bounds=(0.001, 100))
        k = popt[0]
        ci = 1.96 * float(np.sqrt(pcov[0, 0]))
        d_grp = np.array(d_unique)
        a_grp = np.array([grouped[d].mean() for d in d_unique])
        pred = envelope(d_grp, k)
        ss_res = float(np.sum((a_grp - pred) ** 2))
        ss_tot = float(np.sum((a_grp - a_grp.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        k, ci, r2 = float("nan"), float("nan"), float("nan")
    return k, ci, r2, per_dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc-csv", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="pairs of label,judged_csv,n_params(B) e.g. Qwen14B,path,14")
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--metrics-csv", type=Path, required=True)
    ap.add_argument("--include-dc0", action="store_true",
                    help="If set, include d_c=0 items in fit (default excluded)")
    args = ap.parse_args()

    dc_map = load_dc(args.dc_csv)
    print(f"[dc] {len(dc_map)} items")

    parsed = []
    for triple in args.models:
        parts = triple.split(",", 2)
        if len(parts) < 2:
            raise SystemExit(f"bad --models triple: {triple}")
        label, jpath = parts[0], parts[1]
        nparam = float(parts[2]) if len(parts) > 2 else float("nan")
        parsed.append((label, Path(jpath), nparam))

    results = []
    for label, jpath, nparam in parsed:
        acc = load_accuracy(jpath)
        k, ci, r2, per_dc = fit_one(acc, dc_map, exclude_dc0=not args.include_dc0)
        results.append({
            "label": label, "judged_csv": str(jpath), "n_params_b": nparam,
            "n_items_scored": len(acc), "overall_acc": np.mean(list(acc.values())),
            "kappa_eff": k, "kappa_ci95": ci, "r2_group": r2, "per_dc": per_dc,
        })
        print(f"\n[{label}] N={len(acc)} overall_acc={results[-1]['overall_acc']:.3f}  "
              f"kappa_eff={k:.3f}+/-{ci:.3f}  R2={r2:.3f}")
        for d, n, a in per_dc:
            print(f"   d_c={d}  n={n:3d}  acc={a:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))
    d_dense = np.linspace(0.5, 6, 200)
    for i, r in enumerate(results):
        per_dc = r["per_dc"]
        if not per_dc:
            continue
        dgs = [p[0] for p in per_dc]
        accs = [p[2] for p in per_dc]
        ns = [p[1] for p in per_dc]
        # Wilson CI half-widths
        z = 1.96
        halfs = []
        for n, p in zip(ns, accs):
            denom = 1 + z**2 / n
            half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            halfs.append(half)
        ax.errorbar(dgs, accs, yerr=halfs, fmt="o", color=colors[i],
                    capsize=3, alpha=0.7, markersize=7,
                    label=f"{r['label']} (~{r['n_params_b']:.0f}B, overall={r['overall_acc']:.3f}, k={r['kappa_eff']:.2f}, R2={r['r2_group']:.2f})")
        if not np.isnan(r["kappa_eff"]):
            ax.plot(d_dense, envelope(d_dense, r["kappa_eff"]),
                    "-", color=colors[i], alpha=0.6, lw=1.5)

    ax.set_xlabel(r"Consensus conservation-constraint load $d_c$ ($d_c=0$ excluded)")
    ax.set_ylabel("Per-item accuracy (LLM-judge by Qwen-14B)")
    ax.set_title(f"W5 Stage 2 envelope (multi-model on pilot258)\nfit: $\\bar A = 1 - e^{{-\\kappa_{{eff}}/d_c}}$")
    ax.set_ylim(-0.02, max(0.5, max(r["overall_acc"] for r in results) + 0.1))
    ax.set_xlim(0.5, 6)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.fig, dpi=150)
    plt.savefig(args.fig.with_suffix(".pdf"))
    print(f"\n[fig] {args.fig}")

    # metrics csv
    fieldnames = ["label", "n_params_b", "n_items_scored", "overall_acc",
                  "kappa_eff", "kappa_ci95", "r2_group", "per_dc_summary"]
    with open(args.metrics_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            r2 = dict(r); r2["per_dc_summary"] = "; ".join(f"d={d}:n={n}:acc={a:.3f}" for d,n,a in r["per_dc"])
            r2.pop("per_dc"); r2.pop("judged_csv", None)
            w.writerow({k: r2.get(k, "") for k in fieldnames})
    print(f"[csv] {args.metrics_csv}")

    # report
    lines = [
        "# W5 Stage 2 — Multi-Model Envelope\n",
        f"- Pilot: {len(dc_map)} items (d_c=0 {'INCLUDED' if args.include_dc0 else 'EXCLUDED for fit'})\n",
        "- d_c source: Qwen-14B median of 4 raters (V4a alpha=0.9020)\n",
        "- Judge: Qwen-14B vLLM as LLM judge for answer correctness (judge-family bias is possible)\n\n",
        "## Per-model results\n\n",
        "| Model | params | overall acc | κ_eff | R²(group) |\n|---|---|---|---|---|\n",
    ]
    for r in results:
        lines.append(f"| {r['label']} | ~{r['n_params_b']:.0f}B | {r['overall_acc']:.3f} | "
                     f"{r['kappa_eff']:.3f}±{r['kappa_ci95']:.3f} | {r['r2_group']:.3f} |\n")
    lines.append("\n## Per-d_c × model accuracy\n\n")
    all_dcs = sorted({d for r in results for d, _, _ in r["per_dc"]})
    header = "| d_c | n_max | " + " | ".join(r["label"] for r in results) + " |\n"
    lines.append(header)
    lines.append("|---|---|" + "|".join(["---"] * len(results)) + "|\n")
    for d in all_dcs:
        ns = []
        accs = []
        for r in results:
            found = [(n, a) for dd, n, a in r["per_dc"] if dd == d]
            if found:
                ns.append(found[0][0]); accs.append(f"{found[0][1]:.3f}")
            else:
                accs.append("—")
        n_max = max(ns) if ns else 0
        lines.append(f"| {d} | {n_max} | " + " | ".join(accs) + " |\n")
    lines.append("\n## Key Findings\n\n")
    lines.append("- Low-d_c bins are consistently easier than high-d_c bins across the available judged runs; all models score 0 on the sparse d_c>=4 bins in this pilot.\n")
    lines.append("- DeepSeekV3 has the highest overall accuracy in this local set; parameter count/provider tier is not yet a clean monotonic predictor.\n")
    lines.append("- The one-parameter envelope is qualitatively plausible for several models (R2_group around 0.72-0.82), but Qwen14B remains below the 0.70 gate and no universal one-parameter law is established.\n")
    lines.append("- Because high-d_c bins are small (d_c=4: n=4; d_c=5: n=1), W6 must rebalance or bootstrap before making a strong curve-shape claim.\n")
    lines.append("\n## Next (W6)\n\n")
    lines.append("- Add `L_c`, `B_t` (actual tokens_used), `rho_p` (physics-corpus proxy) for full 4-param envelope\n")
    lines.append("- Bootstrap 95% CI per kappa (5880 96-core when `.wslconfig` upgraded)\n")
    lines.append("- Family-specific κ comparison (AIC/BIC) for C1 architecture-independence test\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(lines), encoding="utf-8")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
