"""Generate the R07 judge-free controlled-probe figures.

The script uses only released derived data:
  data/r07_judgefree/synthetic_graded_long.csv
  data/r07_judgefree/R07_verdict.json

It writes vector PDFs/SVGs and PNG previews for the three-panel Figure 6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = ["ollama3b", "vllm7b", "vllm14b", "vllm32b"]
ARM_LABELS = {
    "ollama3b": "Qwen 3B",
    "vllm7b": "Qwen 7B",
    "vllm14b": "Qwen 14B",
    "vllm32b": "Qwen 32B",
}
COLORS = {
    "ollama3b": "#6c757d",
    "vllm7b": "#0072B2",
    "vllm14b": "#009E73",
    "vllm32b": "#D55E00",
}
FAMILY_ORDER = [
    "t_descent",
    "t_descent_inelastic",
    "t_elastic_1d",
    "t_descent_inelastic_spin",
    "t_elastic_2d_equal",
    "t_elastic_2d_unequal",
    "t_descent_2delastic",
    "t_descent_elastic_spin",
    "t_elastic_then_inelastic",
]
FAMILY_LABELS = {
    "t_descent": "descent",
    "t_descent_inelastic": "descent +\ninelastic",
    "t_elastic_1d": "1D elastic",
    "t_descent_inelastic_spin": "descent +\nspin",
    "t_elastic_2d_equal": "2D elastic\nequal",
    "t_elastic_2d_unequal": "2D elastic\nunequal",
    "t_descent_2delastic": "descent +\n2D elastic",
    "t_descent_elastic_spin": "descent +\nelastic + spin",
    "t_elastic_then_inelastic": "1D elastic\nchain",
}
OR_LADDER = [
    ("unadjusted", 0.241, 0.206, 0.277),
    ("+ log length", 0.201, 0.159, 0.237),
    ("+ digit count", 0.211, 0.169, 0.249),
    ("+ verbosity", 0.175, 0.140, 0.211),
    ("+ model FE", 0.096, 0.068, 0.125),
    ("parseable only", 0.094, 0.068, 0.123),
    ("measured d_c", 0.069, 0.037, 0.117),
]


def save_all(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig6a(df: pd.DataFrame, out: Path) -> None:
    acc = df.groupby(["arm", "intended_dc"])["correct"].mean().unstack()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for arm in ARM_ORDER:
        row = acc.loc[arm]
        ax.plot(row.index, row.values, marker="o", lw=2.0, ms=5,
                color=COLORS[arm], label=ARM_LABELS[arm])
    ax.set_title("Exact-graded accuracy by load", fontsize=10.5, pad=8)
    ax.set_xlabel(r"Structural load $d_c$")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(-0.02, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    save_all(fig, out)


def fig6b(out: Path) -> None:
    labels = [r[0] for r in OR_LADDER]
    ors = np.array([r[1] for r in OR_LADDER])
    lo = np.array([r[2] for r in OR_LADDER])
    hi = np.array([r[3] for r in OR_LADDER])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.35))
    ax.errorbar(ors, y, xerr=[ors - lo, hi - ors], fmt="o", color="#0072B2",
                ecolor="#0072B2", elinewidth=1.8, capsize=3, markersize=5)
    for x, yy in zip(ors, y):
        ax.text(x + 0.018, yy, f"{x:.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color="#333333", ls=":", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0.0, 0.34)
    ax.set_xlabel(r"Odds ratio per +1 $d_c$")
    ax.set_title("Control-adjusted judge-free penalty", fontsize=10.5, pad=8)
    ax.grid(axis="x", alpha=0.22)
    save_all(fig, out)


def fig6c(df: pd.DataFrame, out: Path) -> None:
    acc = df.groupby(["family", "arm"])["correct"].mean().unstack().loc[FAMILY_ORDER, ARM_ORDER]
    matrix = acc.to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER], fontsize=8.5)
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER], fontsize=8.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="#111111" if val < 0.65 else "white")
    ax.set_title("Family-level accuracy", fontsize=10.5, pad=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Problem family")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Accuracy")
    save_all(fig, out)


def write_summary(df: pd.DataFrame, verdict: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = df.groupby(["arm", "intended_dc"])["correct"].mean().unstack().loc[ARM_ORDER]
    acc.index = [ARM_LABELS[a] for a in acc.index]
    acc.to_csv(out_dir / "R07_accuracy_by_dc.csv")
    fam = df.groupby(["family", "arm"])["correct"].mean().unstack().loc[FAMILY_ORDER, ARM_ORDER]
    fam.index = [FAMILY_LABELS[f].replace("\n", " ") for f in fam.index]
    fam.columns = [ARM_LABELS[a] for a in fam.columns]
    fam.to_csv(out_dir / "R07_accuracy_by_family.csv")
    pd.DataFrame(OR_LADDER, columns=["specification", "odds_ratio", "ci_low", "ci_high"]).to_csv(
        out_dir / "R07_control_ladder_or.csv", index=False
    )
    (out_dir / "R07_verdict_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "r07_judgefree")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "figures")
    parser.add_argument("--summary-dir", type=Path, default=Path(__file__).resolve().parents[2] / "evaluation" / "r07_judgefree")
    args = parser.parse_args()
    df = pd.read_csv(args.data_dir / "synthetic_graded_long.csv")
    verdict = json.loads((args.data_dir / "R07_verdict.json").read_text(encoding="utf-8"))
    fig6a(df, args.out_dir / "R07_fig6a_accuracy_by_dc.pdf")
    fig6b(args.out_dir / "R07_fig6b_control_ladder.pdf")
    fig6c(df, args.out_dir / "R07_fig6c_family_accuracy.pdf")
    write_summary(df, verdict, args.summary_dir)
    print(f"[ok] wrote R07 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
