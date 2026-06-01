"""Analyze the high-d_c +100 extension across available solver arms.

This is an offline aggregation script. It assumes the solve/judge CSVs already
exist and does not call any model APIs.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[2]


DEFAULT_INPUTS = [
    ("DeepSeek", "data/results/solve_highdc100_deepseek_judged_by_deepseek_v4flash_20260526.csv", "DeepSeek judge, deepseek-v4-flash route"),
    ("Qwen-14B", "data/results/solve_highdc100_qwen14b_judged_by_deepseek_v4flash_20260526.csv", "DeepSeek judge, deepseek-v4-flash route"),
    ("Kimi-K2", "data/results/solve_highdc100_kimi_k2_judged_by_deepseek_v4flash_20260526.csv", "DeepSeek judge, deepseek-v4-flash route"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def label_value(value: str | None) -> int | None:
    try:
        parsed = int(float(value or ""))
    except ValueError:
        return None
    return parsed if parsed in {0, 1} else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=PROJECT / "data/annotations/highdc_prelabels/pilot_deepseek_r1.csv")
    ap.add_argument("--out-items", type=Path, default=PROJECT / "evaluation/highdc_multimodel_items_20260526.csv")
    ap.add_argument("--out-summary", type=Path, default=PROJECT / "evaluation/highdc_multimodel_summary_20260526.csv")
    ap.add_argument("--out-report", type=Path, default=PROJECT / "evaluation/highdc_multimodel_extension_20260526.md")
    ap.add_argument("--out-fig", type=Path, default=PROJECT / "figures/F9_highdc_multimodel_extension_20260526.png")
    args = ap.parse_args()

    labels = {row["item_id"]: row for row in read_csv(args.labels)}
    item_rows: list[dict[str, object]] = []

    for model_label, rel_path, judge_route in DEFAULT_INPUTS:
        path = PROJECT / rel_path
        if not path.exists():
            continue
        for row in read_csv(path):
            item_id = row.get("item_id", "")
            label_row = labels.get(item_id, {})
            solve_error = bool(str(row.get("error", "")).strip())
            judge_reason = str(row.get("judge_reason", ""))
            judge_error = judge_reason.startswith(("judge_error", "unparseable_judge"))
            y = label_value(row.get("is_correct_judge"))
            valid = (not solve_error) and (not judge_error) and (y in {0, 1})
            item_rows.append({
                "item_id": item_id,
                "source_benchmark": row.get("source_benchmark", label_row.get("source_benchmark", "")),
                "topic": row.get("topic", label_row.get("topic", "")),
                "d_c": label_row.get("d_c", ""),
                "model_label": model_label,
                "judge_route": judge_route,
                "is_correct": y if valid else "",
                "valid": int(valid),
                "solve_error": int(solve_error),
                "judge_error": int(judge_error),
                "tokens_out": row.get("tokens_out", ""),
                "judge_reason": judge_reason,
            })

    fields = [
        "item_id", "source_benchmark", "topic", "d_c", "model_label", "judge_route",
        "is_correct", "valid", "solve_error", "judge_error", "tokens_out", "judge_reason",
    ]
    write_csv(args.out_items, item_rows, fields)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in item_rows:
        grouped[(str(row["model_label"]), str(row["d_c"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (model_label, dc), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 99)):
        valid_rows = [r for r in rows if r["valid"] == 1]
        n = len(valid_rows)
        k = sum(int(r["is_correct"]) for r in valid_rows)
        lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
        summary_rows.append({
            "model_label": model_label,
            "d_c": dc,
            "raw_n": len(rows),
            "valid_n": n,
            "correct_n": k,
            "accuracy": f"{k / n:.6f}" if n else "",
            "ci_lo": f"{lo:.6f}" if n else "",
            "ci_hi": f"{hi:.6f}" if n else "",
            "solve_error_n": sum(int(r["solve_error"]) for r in rows),
            "judge_error_n": sum(int(r["judge_error"]) for r in rows),
        })
    write_csv(args.out_summary, summary_rows, [
        "model_label", "d_c", "raw_n", "valid_n", "correct_n", "accuracy", "ci_lo", "ci_hi",
        "solve_error_n", "judge_error_n",
    ])

    model_totals = []
    for model_label in sorted({str(r["model_label"]) for r in item_rows}):
        rows = [r for r in item_rows if r["model_label"] == model_label]
        valid_rows = [r for r in rows if r["valid"] == 1]
        n = len(valid_rows)
        k = sum(int(r["is_correct"]) for r in valid_rows)
        model_totals.append({
            "model_label": model_label,
            "raw_n": len(rows),
            "valid_n": n,
            "correct_n": k,
            "accuracy": k / n if n else float("nan"),
            "solve_error_n": sum(int(r["solve_error"]) for r in rows),
            "judge_error_n": sum(int(r["judge_error"]) for r in rows),
        })

    fig_rows = [r for r in summary_rows if r["accuracy"] != ""]
    dcs = sorted({int(r["d_c"]) for r in fig_rows if str(r["d_c"]).isdigit()})
    models = sorted({str(r["model_label"]) for r in fig_rows})
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, model in enumerate(models):
        ys = []
        ns = []
        for dc in dcs:
            found = next((r for r in fig_rows if r["model_label"] == model and str(r["d_c"]) == str(dc)), None)
            ys.append(float(found["accuracy"]) if found else float("nan"))
            ns.append(int(found["valid_n"]) if found else 0)
        xs = [dc + (idx - (len(models) - 1) / 2) * width for dc in dcs]
        ax.bar(xs, ys, width=width, label=model)
        for x, y, n in zip(xs, ys, ns):
            if not math.isnan(y):
                ax.text(x, y + 0.015, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xlabel(r"DeepSeek-prelabelled conservation-constraint load $d_c$")
    ax.set_ylabel("Judged accuracy on high-d_c100")
    ax.set_title("High-d_c +100 extension across solver arms")
    ax.set_xticks(dcs)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=160)
    plt.savefig(args.out_fig.with_suffix(".pdf"))

    lines = [
        "# High-d_c +100 Multi-Model Extension\n\n",
        f"- Item-level CSV: `{args.out_items}`\n",
        f"- Summary CSV: `{args.out_summary}`\n",
        f"- Figure: `{args.out_fig}`\n",
        "- Scope: 100 newly mined high-d_c candidates, using DeepSeek single-rater prelabels for d_c. This is an extension diagnostic, not a human-gold replacement.\n\n",
        "## Overall by solver\n\n",
        "| solver | raw n | valid n | correct n | accuracy | solve errors | judge errors |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in model_totals:
        lines.append(
            f"| {row['model_label']} | {row['raw_n']} | {row['valid_n']} | {row['correct_n']} | "
            f"{row['accuracy']:.3f} | {row['solve_error_n']} | {row['judge_error_n']} |\n"
        )
    lines.extend([
        "\n## Accuracy by d_c\n\n",
        "| solver | d_c | raw n | valid n | correct n | accuracy | judge errors |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ])
    for row in summary_rows:
        acc = f"{float(row['accuracy']):.3f}" if row["accuracy"] != "" else "NA"
        lines.append(
            f"| {row['model_label']} | {row['d_c']} | {row['raw_n']} | {row['valid_n']} | "
            f"{row['correct_n']} | {acc} | {row['judge_error_n']} |\n"
        )
    lines.extend([
        "\n## Interpretation\n\n",
        "- The extension is useful for stress-testing transfer beyond the original pilot258, but it also demonstrates within-d_c item heterogeneity: newly mined items can be easier than the original pilot at the same nominal d_c.\n",
        "- All three high-d_c solver arms in this report are judged with the same `deepseek-v4-flash` route. This removes one judge-route confound from the extension analysis, but it still does not replace human correctness adjudication.\n",
        "- High d_c >= 4 remains underpopulated. The human-gold packet and additional oversampling are still required before upgrading the central claim.\n",
    ])
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("".join(lines), encoding="utf-8")
    print(f"wrote={args.out_report}")
    print(f"wrote={args.out_summary}")
    print(f"wrote={args.out_items}")
    print(f"wrote={args.out_fig}")


if __name__ == "__main__":
    main()
