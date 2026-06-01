"""Analyze a controlled C4 max-token budget sweep after judging.

Input must be the output of grade_llm_judge.py applied to
13_run_c4_budget_sweep.py rows. The analysis is intentionally conservative:
it reports both raw accuracy by budget and paired deltas relative to the
smallest budget for items that have complete judged rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]


def as_int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            y = as_int(row.get("is_correct_judge", ""))
            budget = as_int(row.get("max_tokens", ""))
            dc = as_int(row.get("d_c_consensus", ""))
            reason = row.get("judge_reason", "")
            if str(row.get("error", "")).strip():
                continue
            if y not in {0, 1} or budget is None or dc is None:
                continue
            if reason.startswith(("judge_error", "unparseable_judge")):
                continue
            row["_y"] = y
            row["_budget"] = budget
            row["_dc"] = dc
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=PROJECT / "evaluation/C4_controlled_budget_sweep_20260524.md")
    ap.add_argument("--by-budget-csv", type=Path, default=PROJECT / "evaluation/c4_controlled_budget_by_budget_20260524.csv")
    ap.add_argument("--by-dc-csv", type=Path, default=PROJECT / "evaluation/c4_controlled_budget_by_dc_20260524.csv")
    ap.add_argument("--fig", type=Path, default=PROJECT / "figures/F8_c4_controlled_budget_sweep_20260524.png")
    args = ap.parse_args()

    rows = load_rows(args.judged)
    if not rows:
        raise SystemExit(f"no valid judged rows in {args.judged}")

    by_budget: dict[int, list[int]] = defaultdict(list)
    by_dc_budget: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_item_budget: dict[str, dict[int, int]] = defaultdict(dict)
    for row in rows:
        b = row["_budget"]
        dc = row["_dc"]
        y = row["_y"]
        by_budget[b].append(y)
        by_dc_budget[(dc, b)].append(y)
        by_item_budget[row["item_id"]][b] = y

    budget_rows = []
    for budget in sorted(by_budget):
        vals = by_budget[budget]
        budget_rows.append({
            "max_tokens": budget,
            "n": len(vals),
            "accuracy": f"{statistics.mean(vals):.6f}",
        })
    write_csv(args.by_budget_csv, budget_rows, ["max_tokens", "n", "accuracy"])

    dc_rows = []
    for (dc, budget), vals in sorted(by_dc_budget.items()):
        dc_rows.append({
            "d_c": dc,
            "max_tokens": budget,
            "n": len(vals),
            "accuracy": f"{statistics.mean(vals):.6f}",
        })
    write_csv(args.by_dc_csv, dc_rows, ["d_c", "max_tokens", "n", "accuracy"])

    budgets = sorted(by_budget)
    min_budget = budgets[0]
    paired_summary = []
    for budget in budgets[1:]:
        deltas = []
        gains = losses = 0
        for item_id, vals in by_item_budget.items():
            if min_budget not in vals or budget not in vals:
                continue
            delta = vals[budget] - vals[min_budget]
            deltas.append(delta)
            gains += int(delta > 0)
            losses += int(delta < 0)
        if deltas:
            p = stats.binomtest(gains, gains + losses, 0.5).pvalue if gains + losses > 0 else math.nan
            paired_summary.append({
                "max_tokens": budget,
                "paired_n": len(deltas),
                "delta_vs_min_budget": statistics.mean(deltas),
                "gains": gains,
                "losses": losses,
                "sign_test_p": p,
            })

    xs = np.array([math.log(float(r["max_tokens"]) / min_budget) for r in budget_rows], dtype=float)
    ys = np.array([float(r["accuracy"]) for r in budget_rows], dtype=float)
    if len(xs) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    else:
        slope = intercept = r_value = p_value = std_err = math.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["max_tokens"] for r in budget_rows], [float(r["accuracy"]) for r in budget_rows],
            marker="o", lw=2, color="#2d6cdf")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Controlled max_tokens budget")
    ax.set_ylabel("Judged accuracy")
    ax.set_title("C4 controlled budget sweep")
    ax.grid(True, alpha=0.25, which="both")
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=160)
    plt.savefig(args.fig.with_suffix(".pdf"))

    lines = [
        "# C4 Controlled Budget Sweep\n\n",
        f"- Input judged CSV: `{args.judged}`\n",
        f"- Valid judged rows: {len(rows)}\n",
        f"- Budgets: {', '.join(str(b) for b in budgets)}\n\n",
        "## Accuracy by budget\n\n",
        "| max_tokens | n | accuracy |\n",
        "|---:|---:|---:|\n",
    ]
    for row in budget_rows:
        lines.append(f"| {row['max_tokens']} | {row['n']} | {float(row['accuracy']):.3f} |\n")
    lines.extend([
        "\n## Log-budget slope\n\n",
        f"- Fit: accuracy = a + b * log(max_tokens / {min_budget}).\n",
        f"- b = {slope:.4f}; 95% normal CI = [{slope - 1.96 * std_err:.4f}, {slope + 1.96 * std_err:.4f}]; R2 = {r_value ** 2:.3f}; p = {p_value:.3g}.\n",
        "- Reference C4 upper-bound slope is beta/e ~= 0.11 under the Chinchilla beta ~= 0.3 anchor.\n\n",
        "## Paired deltas vs minimum budget\n\n",
        "| max_tokens | paired n | mean delta | gains | losses | sign-test p |\n",
        "|---:|---:|---:|---:|---:|---:|\n",
    ])
    for row in paired_summary:
        p_text = "NA" if math.isnan(row["sign_test_p"]) else f"{row['sign_test_p']:.3g}"
        lines.append(
            f"| {row['max_tokens']} | {row['paired_n']} | {row['delta_vs_min_budget']:+.3f} | "
            f"{row['gains']} | {row['losses']} | {p_text} |\n"
        )
    lines.extend([
        "\n## Interpretation rule\n\n",
        "- Treat this as C4 evidence only if the same item subset, same model snapshot, and same judge route are used across all budgets.\n",
        "- If judge errors exceed 2% or the provider alias moves during the run, downgrade this sweep to a pilot diagnostic.\n",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(lines), encoding="utf-8")
    print(f"wrote={args.report}")
    print(f"wrote={args.by_budget_csv}")
    print(f"wrote={args.by_dc_csv}")
    print(f"wrote={args.fig}")


if __name__ == "__main__":
    main()
