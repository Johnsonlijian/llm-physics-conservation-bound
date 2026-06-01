"""C5 paired direct-vs-CoT analysis.

Uses existing solved outputs judged by the same judge. No model/API calls.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def is_valid_judge(row: dict[str, str]) -> bool:
    reason = (row.get("judge_reason") or "").strip()
    return not reason.startswith(("judge_error", "unparseable_judge"))


def load_consensus_dc(path: Path) -> dict[str, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in read_csv(path):
        value = (row.get("d_c") or "").strip()
        if value:
            groups[row["item_id"]].append(int(value))
    return {iid: int(np.median(vals)) for iid, vals in groups.items()}


def wilson_half_width(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    p = successes / n
    denom = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom


def fit_delta_vs_inv_dc(rows: list[dict]) -> dict:
    usable = [r for r in rows if int(r["d_c"]) >= 1]
    x = np.array([1.0 / int(r["d_c"]) for r in usable], dtype=float)
    y = np.array([float(r["delta_cot"]) for r in usable], dtype=float)
    if len(usable) < 3:
        return {"n": len(usable), "intercept": float("nan"), "slope": float("nan"), "p": float("nan"), "r2": float("nan")}
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {"n": len(usable), "intercept": intercept, "slope": slope, "p": p, "r2": r * r, "se": se}


def bootstrap_slope(rows: list[dict], n_boot: int, seed: int) -> tuple[float, float]:
    usable = [r for r in rows if int(r["d_c"]) >= 1]
    rng = random.Random(seed)
    slopes = []
    if len(usable) < 3:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        sample = [rng.choice(usable) for _ in usable]
        fit = fit_delta_vs_inv_dc(sample)
        if not math.isnan(fit["slope"]):
            slopes.append(fit["slope"])
    if not slopes:
        return float("nan"), float("nan")
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def make_fig(summary_rows: list[dict], fit: dict, fig_path: Path) -> None:
    rows = [r for r in summary_rows if int(r["d_c"]) >= 1]
    dcs = np.array([int(r["d_c"]) for r in rows], dtype=float)
    delta = np.array([float(r["delta_cot"]) for r in rows], dtype=float)
    n = np.array([int(r["n_valid"]) for r in rows], dtype=float)
    half = []
    for r in rows:
        # Conservative binomial SE over paired delta range, shown as a visual cue only.
        gains = int(r["gain_items"])
        losses = int(r["loss_items"])
        half.append(1.96 * math.sqrt((gains + losses) / max(int(r["n_valid"]), 1) ** 2))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(dcs, delta, yerr=half, fmt="o", capsize=3, color="#1f77b4", label="Observed CoT - direct")
    if not math.isnan(fit["slope"]):
        x_dense = np.linspace(1, max(dcs), 200)
        pred = fit["intercept"] + fit["slope"] * (1 / x_dense)
        ax.plot(x_dense, pred, color="black", linewidth=2, label=r"OLS on $1/d_c$")
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    for dc, ni, yi in zip(dcs, n, delta):
        ax.text(dc, yi + 0.018, f"n={int(ni)}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(r"Consensus conservation-constraint load $d_c$")
    ax.set_ylabel("Paired accuracy gain: CoT - direct")
    ax.set_title("C5 paired direct-vs-CoT pilot (DeepSeek, same judge)")
    ax.set_xlim(0.7, max(dcs) + 0.3)
    ax.set_ylim(min(-0.18, float(delta.min()) - 0.06), max(0.42, float(delta.max()) + 0.08))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.savefig(fig_path.with_suffix(".pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", type=Path, default=PROJECT / "data/results/solve_pilot258_deepseek_direct_judged_by_deepseek.csv")
    ap.add_argument("--cot", type=Path, default=PROJECT / "data/results/solve_pilot258_deepseek_cot_judged_by_deepseek.csv")
    ap.add_argument("--dc", type=Path, default=PROJECT / "data/annotations/pilot258_v4a_qwen14b_merged.csv")
    ap.add_argument("--paired-csv", type=Path, default=PROJECT / "evaluation/c5_paired_cot_items.csv")
    ap.add_argument("--summary-csv", type=Path, default=PROJECT / "evaluation/c5_paired_cot_by_dc.csv")
    ap.add_argument("--report", type=Path, default=PROJECT / "evaluation/C5_paired_cot_analysis.md")
    ap.add_argument("--fig", type=Path, default=PROJECT / "figures/F7_c5_paired_cot_delta.png")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    direct = {r["item_id"]: r for r in read_csv(args.direct)}
    cot = {r["item_id"]: r for r in read_csv(args.cot)}
    dc_map = load_consensus_dc(args.dc)
    paired = []
    excluded = []
    for iid in sorted(set(direct) & set(cot) & set(dc_map)):
        drow = direct[iid]
        crow = cot[iid]
        if not is_valid_judge(drow) or not is_valid_judge(crow):
            excluded.append(iid)
            continue
        d = int(drow["is_correct_judge"])
        c = int(crow["is_correct_judge"])
        paired.append(
            {
                "item_id": iid,
                "source_benchmark": drow.get("source_benchmark", ""),
                "topic": drow.get("topic", ""),
                "d_c": dc_map[iid],
                "direct_correct": d,
                "cot_correct": c,
                "delta_cot": c - d,
                "direct_tokens_out": drow.get("tokens_out", ""),
                "cot_tokens_out": crow.get("tokens_out", ""),
            }
        )

    write_csv(
        args.paired_csv,
        paired,
        [
            "item_id",
            "source_benchmark",
            "topic",
            "d_c",
            "direct_correct",
            "cot_correct",
            "delta_cot",
            "direct_tokens_out",
            "cot_tokens_out",
        ],
    )

    by_dc: dict[int, list[dict]] = defaultdict(list)
    for row in paired:
        by_dc[int(row["d_c"])].append(row)
    summary_rows = []
    for dc in sorted(by_dc):
        rows = by_dc[dc]
        direct_acc = float(np.mean([int(r["direct_correct"]) for r in rows]))
        cot_acc = float(np.mean([int(r["cot_correct"]) for r in rows]))
        gains = sum(1 for r in rows if int(r["direct_correct"]) == 0 and int(r["cot_correct"]) == 1)
        losses = sum(1 for r in rows if int(r["direct_correct"]) == 1 and int(r["cot_correct"]) == 0)
        ties = len(rows) - gains - losses
        p = stats.binomtest(min(gains, losses), gains + losses, 0.5).pvalue if gains + losses > 0 else float("nan")
        summary_rows.append(
            {
                "d_c": dc,
                "n_valid": len(rows),
                "direct_acc": direct_acc,
                "cot_acc": cot_acc,
                "delta_cot": cot_acc - direct_acc,
                "gain_items": gains,
                "loss_items": losses,
                "tie_items": ties,
                "sign_test_p": p,
            }
        )
    write_csv(
        args.summary_csv,
        summary_rows,
        ["d_c", "n_valid", "direct_acc", "cot_acc", "delta_cot", "gain_items", "loss_items", "tie_items", "sign_test_p"],
    )

    n = len(paired)
    direct_acc = float(np.mean([int(r["direct_correct"]) for r in paired]))
    cot_acc = float(np.mean([int(r["cot_correct"]) for r in paired]))
    gains = sum(1 for r in paired if int(r["direct_correct"]) == 0 and int(r["cot_correct"]) == 1)
    losses = sum(1 for r in paired if int(r["direct_correct"]) == 1 and int(r["cot_correct"]) == 0)
    sign_p = stats.binomtest(min(gains, losses), gains + losses, 0.5).pvalue if gains + losses > 0 else float("nan")
    fit = fit_delta_vs_inv_dc(paired)
    ci_low, ci_high = bootstrap_slope(paired, args.bootstrap, args.seed)
    make_fig(summary_rows, fit, args.fig)

    lines = [
        "# C5 Paired Direct-vs-CoT Analysis\n\n",
        "Paired pilot using existing DeepSeek solves and the same DeepSeek judge for direct and CoT-allowed outputs.\n\n",
        "## Inputs\n\n",
        f"- Direct judged: `{args.direct.relative_to(PROJECT)}`\n",
        f"- CoT-allowed judged: `{args.cot.relative_to(PROJECT)}`\n",
        f"- Consensus conservation-constraint load d_c: `{args.dc.relative_to(PROJECT)}`\n",
        f"- Valid paired items: {n}; excluded judge-error/unparseable items: {len(excluded)}\n\n",
        "## Overall paired result\n\n",
        f"- Direct accuracy: {direct_acc:.3f}\n",
        f"- CoT-allowed accuracy: {cot_acc:.3f}\n",
        f"- Paired delta (CoT - direct): {cot_acc - direct_acc:+.3f}\n",
        f"- Gain items: {gains}; loss items: {losses}; sign-test p={sign_p:.4g}\n\n",
        "## By d_c\n\n",
        "| d_c | n | direct acc | CoT acc | delta | gains | losses | sign-test p |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        p = row["sign_test_p"]
        p_txt = "NA" if math.isnan(float(p)) else f"{float(p):.3g}"
        lines.append(
            f"| {row['d_c']} | {row['n_valid']} | {row['direct_acc']:.3f} | {row['cot_acc']:.3f} | "
            f"{row['delta_cot']:+.3f} | {row['gain_items']} | {row['loss_items']} | {p_txt} |\n"
        )
    lines.extend(
        [
            "\n## C5 1/d_c pilot regression\n\n",
            f"- Fit on d_c>=1 items: n={fit['n']}, delta = {fit['intercept']:.3f} + {fit['slope']:.3f}*(1/d_c).\n",
            f"- Slope p={fit['p']:.4g}, R2={fit['r2']:.3f}; bootstrap 95% CI for slope=[{ci_low:.3f}, {ci_high:.3f}] (B={args.bootstrap}).\n",
            "- Positive slope would support the stronger per-constraint hypothesis that CoT gains concentrate at low d_c. Treat this as pilot evidence only because d_c>=4 bins are sparse and both conditions use the same provider family.\n\n",
            "## Interpretation\n\n",
        ]
    )
    if cot_acc > direct_acc:
        lines.append("- CoT-allowed prompting improves overall accuracy over strict direct prompting on this paired pilot.\n")
    else:
        lines.append("- CoT-allowed prompting does not improve overall accuracy in this paired pilot.\n")
    if not math.isnan(fit["slope"]) and fit["slope"] > 0:
        lines.append("- The direction of the 1/d_c slope is consistent with C5's normalized-gain hypothesis, but this is not yet paper-grade.\n")
    else:
        lines.append("- The 1/d_c slope does not support the stronger normalized-gain hypothesis in this pilot.\n")
    lines.append("- C5 should remain a hypothesis / exploratory subsection until replicated with a controlled reasoning-budget model and balanced high-d_c bins.\n\n")
    lines.extend(
        [
            "## Generated artifacts\n\n",
            f"- `{args.paired_csv.relative_to(PROJECT)}`\n",
            f"- `{args.summary_csv.relative_to(PROJECT)}`\n",
            f"- `{args.fig.relative_to(PROJECT)}`\n",
        ]
    )
    args.report.write_text("".join(lines), encoding="utf-8")

    print(f"wrote={args.paired_csv}")
    print(f"wrote={args.summary_csv}")
    print(f"wrote={args.report}")
    print(f"wrote={args.fig}")
    print(f"valid={n} direct_acc={direct_acc:.3f} cot_acc={cot_acc:.3f} delta={cot_acc-direct_acc:+.3f}")


if __name__ == "__main__":
    main()
