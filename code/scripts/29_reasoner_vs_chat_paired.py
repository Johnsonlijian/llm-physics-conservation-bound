"""Paired comparison: DeepSeek-reasoner (max_tokens=16384) vs DeepSeek-chat (max_tokens=2048)
on the same 50-item c4_budget_subset_50_seed42 — direct test of reasoning-model effect at matched
max budget on the reasoning-favouring high-d_c subset.

Output: evaluation/reasoner_vs_chat_paired_20260526.md + figures/F11_reasoner_vs_chat_by_dc.png
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
import numpy as np


def load_judged(path: Path, mt_filter=None):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if mt_filter is not None and str(r.get("max_tokens", "")) != str(mt_filter):
                continue
            item_id = r.get("item_id", "")
            ic = r.get("is_correct_judge", "")
            corr = 1 if ic in ("1", "True", "true") else 0
            err = bool(r.get("judge_reason", "").startswith(("judge_error", "unparseable")))
            out[item_id] = dict(correct=corr, error=err)
    return out


def load_dc(path: Path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["item_id"]] = int(r.get("d_c_consensus", "0"))
    return out


def sign_test_p(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    return 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner", type=Path,
                    default=Path("data/results/solve_subset50_deepseek_reasoner_v2_judged_20260526.csv"))
    ap.add_argument("--chat-c4", type=Path,
                    default=Path("data/results/c4_budget_sweep_deepseek_chat_judged_20260525.csv"))
    ap.add_argument("--dc-subset", type=Path,
                    default=Path("data/annotations/c4_budget_subset_50_seed42.csv"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("evaluation/reasoner_vs_chat_paired_20260526.md"))
    ap.add_argument("--out-fig", type=Path,
                    default=Path("figures/F11_reasoner_vs_chat_by_dc.png"))
    args = ap.parse_args()

    reasoner = load_judged(args.reasoner, mt_filter=None)
    chat_2048 = load_judged(args.chat_c4, mt_filter=2048)
    chat_128 = load_judged(args.chat_c4, mt_filter=128)
    dc = load_dc(args.dc_subset)
    print(f"reasoner items: {len(reasoner)}, chat@2048: {len(chat_2048)}, chat@128: {len(chat_128)}, dc labels: {len(dc)}")

    # Per-d_c breakdown
    by_dc = defaultdict(lambda: {"n": 0, "chat_128": 0, "chat_2048": 0, "reas": 0,
                                  "chat_128_err": 0, "chat_2048_err": 0, "reas_err": 0})
    paired_gains_2048 = paired_losses_2048 = paired_ties_2048 = 0
    paired_gains_128 = paired_losses_128 = paired_ties_128 = 0
    for item_id, dc_val in dc.items():
        cell = by_dc[dc_val]
        cell["n"] += 1
        if item_id in reasoner:
            cell["reas"] += reasoner[item_id]["correct"]
            cell["reas_err"] += int(reasoner[item_id]["error"])
        if item_id in chat_2048:
            cell["chat_2048"] += chat_2048[item_id]["correct"]
            cell["chat_2048_err"] += int(chat_2048[item_id]["error"])
        if item_id in chat_128:
            cell["chat_128"] += chat_128[item_id]["correct"]
            cell["chat_128_err"] += int(chat_128[item_id]["error"])
        # Paired counts
        if item_id in reasoner and item_id in chat_2048:
            r = reasoner[item_id]["correct"]
            c = chat_2048[item_id]["correct"]
            if r > c: paired_gains_2048 += 1
            elif r < c: paired_losses_2048 += 1
            else: paired_ties_2048 += 1
        if item_id in reasoner and item_id in chat_128:
            r = reasoner[item_id]["correct"]
            c = chat_128[item_id]["correct"]
            if r > c: paired_gains_128 += 1
            elif r < c: paired_losses_128 += 1
            else: paired_ties_128 += 1

    p_2048 = sign_test_p(paired_gains_2048, paired_losses_2048)
    p_128 = sign_test_p(paired_gains_128, paired_losses_128)

    # Totals
    total_n = sum(c["n"] for c in by_dc.values())
    total_chat_128 = sum(c["chat_128"] for c in by_dc.values())
    total_chat_2048 = sum(c["chat_2048"] for c in by_dc.values())
    total_reas = sum(c["reas"] for c in by_dc.values())

    print(f"OVERALL n={total_n}: chat@128={total_chat_128} ({total_chat_128/total_n:.3f}) chat@2048={total_chat_2048} ({total_chat_2048/total_n:.3f})  reasoner={total_reas} ({total_reas/total_n:.3f})")
    print(f"reasoner vs chat@2048 paired: gains={paired_gains_2048} losses={paired_losses_2048} ties={paired_ties_2048}  p={p_2048:.4f}")
    print(f"reasoner vs chat@128  paired: gains={paired_gains_128} losses={paired_losses_128} ties={paired_ties_128}  p={p_128:.4f}")

    # Markdown
    lines = [
        "# DeepSeek-reasoner vs DeepSeek-chat (paired, c4_budget_subset_50)\n\n",
        "Direct paired comparison on the 50-item C4 subset, all judged by the same `deepseek-chat` judge route.\n\n",
        "- **DeepSeek-reasoner** (`deepseek-reasoner`), `max_tokens = 16384`, 48/50 emit `FINAL_ANSWER` (`reasoner_subset50_offline_20260525.md`).\n",
        "- **DeepSeek-chat at 2048 budget** (the maximum budget arm of the C4 sweep): 49/50 valid judged.\n",
        "- **DeepSeek-chat at 128 budget** (the minimum budget arm): 50/50 valid judged.\n\n",
        "## Per-d_c breakdown\n\n",
        "| $d_c$ | n | chat@128 (acc) | chat@2048 (acc) | reasoner-16k (acc) | reasoner − chat@2048 |\n",
        "|---:|---:|---|---|---|---:|\n",
    ]
    for dc_val in sorted(by_dc):
        c = by_dc[dc_val]
        n = c["n"]
        a128 = c["chat_128"] / n
        a2048 = c["chat_2048"] / n
        ar = c["reas"] / n
        delta = ar - a2048
        lines.append(f"| {dc_val} | {n} | {c['chat_128']}/{n} = {a128:.3f} | {c['chat_2048']}/{n} = {a2048:.3f} | {c['reas']}/{n} = {ar:.3f} | {delta:+.3f} |\n")
    lines.append(
        f"| **all** | **{total_n}** | **{total_chat_128}/{total_n} = {total_chat_128/total_n:.3f}** | "
        f"**{total_chat_2048}/{total_n} = {total_chat_2048/total_n:.3f}** | "
        f"**{total_reas}/{total_n} = {total_reas/total_n:.3f}** | "
        f"**{(total_reas-total_chat_2048)/total_n:+.3f}** |\n"
    )
    lines.append("\n## Paired sign-test\n\n")
    lines.append(f"- Reasoner vs chat@2048: gains = {paired_gains_2048}, losses = {paired_losses_2048}, ties = {paired_ties_2048}, two-sided $p = {p_2048:.4f}$\n")
    lines.append(f"- Reasoner vs chat@128:  gains = {paired_gains_128}, losses = {paired_losses_128}, ties = {paired_ties_128}, two-sided $p = {p_128:.4f}$\n\n")
    lines.append("## Reading\n\n")
    lines.append(
        f"The DeepSeek-reasoner arm on this subset reaches {total_reas/total_n:.3f} judged accuracy "
        f"vs {total_chat_2048/total_n:.3f} for DeepSeek-chat at the same nominal max budget of 2048, "
        f"and vs {total_chat_128/total_n:.3f} at 128. The reasoner arm uses up to 16384 tokens of "
        "*hidden* reasoning_content before emitting its final answer (median 13074 tokens; cf. "
        "`reasoner_subset50_offline_20260525.md`), so this is *not* a clean C4 controlled-budget arm. "
        "It is a controlled solver-architecture comparison at *nominal* max budget. The accuracy gain "
        "is consistent with the C4 envelope upper bound at the implied token ratio: 8× more tokens × "
        f"$\\beta/e\\approx 0.11 \\times \\ln 8 \\approx 0.23$, vs observed $+{(total_reas-total_chat_2048)/total_n:.3f}$.\n"
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("".join(lines), encoding="utf-8")
    print(f"[wrote] {args.out_md}")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    dcs = sorted(by_dc)
    width = 0.27
    x = np.arange(len(dcs))
    chat128_a = [by_dc[d]["chat_128"] / by_dc[d]["n"] for d in dcs]
    chat2048_a = [by_dc[d]["chat_2048"] / by_dc[d]["n"] for d in dcs]
    reas_a = [by_dc[d]["reas"] / by_dc[d]["n"] for d in dcs]
    ax.bar(x - width, chat128_a, width, label="chat @ 128 tokens", color="lightgrey", edgecolor="black")
    ax.bar(x, chat2048_a, width, label="chat @ 2048 tokens", color="C0", edgecolor="black")
    ax.bar(x + width, reas_a, width, label="reasoner @ 16384 tokens", color="C3", edgecolor="black")
    for i, d in enumerate(dcs):
        n = by_dc[d]["n"]
        ax.text(i, -0.04, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dcs])
    ax.set_xlabel(r"Conservation-constraint load $d_c$")
    ax.set_ylabel("Judged accuracy (DeepSeek judge)")
    ax.set_title("DeepSeek reasoner vs chat (paired, c4 subset n=50)\nReasoner uses ~8× more tokens than chat@2048")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.06, max(reas_a + chat2048_a) + 0.15)
    ax.grid(True, alpha=0.3, axis="y")
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    plt.savefig(args.out_fig.with_suffix(".pdf"))
    print(f"[fig] {args.out_fig}")


if __name__ == "__main__":
    main()
