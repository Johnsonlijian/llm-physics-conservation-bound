"""W6 controlled analysis for conservation dimension (`d_c`).

This script is offline-only. It merges existing pilot258 item metadata, V4a
consensus d_c labels, and existing judged model outputs, then tests whether
`d_c` remains a negative accuracy predictor after local controls.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[2]

MODEL_SPECS = [
    ("Qwen14B", PROJECT / "data/results/solve_pilot258_qwen14b_judged.csv", 14.0),
    ("DeepSeekV3", PROJECT / "data/results/solve_pilot258_deepseek_judged.csv", 671.0),
    ("KimiK2", PROJECT / "data/results/solve_pilot258_kimi_k2_judged.csv", 1000.0),
    ("Qwen7B-Ollama", PROJECT / "data/results/solve_pilot258_qwen7b_ollama_judged.csv", 7.0),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def as_int(value: str | None, default: int | None = None) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def as_bool(value: str | None) -> int:
    s = str(value or "").strip().lower()
    return int(s in {"1", "true", "yes", "y"})


def answer_type(answer: str) -> str:
    s = answer or ""
    if "\\" in s or "$" in s or "^" in s or "_" in s:
        return "formula"
    if any(ch.isdigit() for ch in s):
        return "numeric"
    if len(s.strip()) <= 8:
        return "short_text"
    return "text"


def load_consensus_dc(path: Path) -> dict[str, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in read_csv(path):
        dc = as_int(row.get("d_c"))
        if dc is not None:
            groups[row["item_id"]].append(dc)
    return {iid: int(statistics.median(vals)) for iid, vals in groups.items() if vals}


def build_panel(item_csv: Path, dc_csv: Path) -> list[dict]:
    items = {row["item_id"]: row for row in read_csv(item_csv)}
    dc_map = load_consensus_dc(dc_csv)
    rows: list[dict] = []
    for model_label, judged_path, n_params_b in MODEL_SPECS:
        for row in read_csv(judged_path):
            iid = row.get("item_id", "")
            y = as_int(row.get("is_correct_judge"))
            if iid not in items or iid not in dc_map or y not in {0, 1}:
                continue
            meta = items[iid]
            q = meta.get("question", "")
            ans = meta.get("answer", "")
            sol = meta.get("solution_text", "")
            rows.append(
                {
                    "item_id": iid,
                    "model_label": model_label,
                    "n_params_b": n_params_b,
                    "source_benchmark": meta.get("source_benchmark") or row.get("source_benchmark") or "unknown",
                    "topic": meta.get("topic") or row.get("topic") or "unknown",
                    "has_diagram": as_bool(meta.get("has_diagram")),
                    "answer_type": answer_type(ans),
                    "d_c": dc_map[iid],
                    "question_chars": len(q),
                    "answer_chars": len(ans),
                    "solution_chars": len(sol),
                    "tokens_in": as_float(row.get("tokens_in")),
                    "tokens_out": as_float(row.get("tokens_out")),
                    "latency_s": as_float(row.get("latency_s")),
                    "is_correct": y,
                }
            )
    return rows


@dataclass
class FeatureSpec:
    name: str
    continuous: list[str]
    categorical: list[str]


SPECS = [
    FeatureSpec("M0_dc_only", ["d_c"], []),
    FeatureSpec(
        "M1_item_controls",
        ["d_c", "log_question_chars", "log_answer_chars", "log_solution_chars", "has_diagram"],
        ["source_benchmark", "answer_type"],
    ),
    FeatureSpec(
        "M2_model_controls",
        [
            "d_c",
            "log_question_chars",
            "log_answer_chars",
            "log_solution_chars",
            "has_diagram",
            "log_tokens_in",
            "log_tokens_out",
        ],
        ["source_benchmark", "answer_type", "model_label"],
    ),
    FeatureSpec(
        "M3_topic_controls",
        [
            "d_c",
            "log_question_chars",
            "log_answer_chars",
            "log_solution_chars",
            "has_diagram",
            "log_tokens_in",
            "log_tokens_out",
        ],
        ["source_benchmark", "answer_type", "model_label", "topic"],
    ),
]


class DesignBuilder:
    def __init__(self, rows: list[dict], spec: FeatureSpec):
        self.spec = spec
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.levels: dict[str, list[str]] = {}
        prepared = [self.prepare_row(r) for r in rows]
        for key in spec.continuous:
            if key == "d_c":
                continue
            vals = np.array([float(r[key]) for r in prepared], dtype=float)
            self.means[key] = float(vals.mean())
            self.stds[key] = float(vals.std() if vals.std() > 1e-9 else 1.0)
        for key in spec.categorical:
            all_levels = sorted({str(r.get(key, "unknown")) for r in prepared})
            self.levels[key] = all_levels[1:] if len(all_levels) > 1 else []
        self.feature_names = ["intercept"]
        for key in spec.continuous:
            self.feature_names.append(key if key == "d_c" else f"z_{key}")
        for key in spec.categorical:
            self.feature_names.extend([f"{key}={lvl}" for lvl in self.levels[key]])

    @staticmethod
    def prepare_row(row: dict) -> dict:
        out = dict(row)
        out["log_question_chars"] = math.log1p(float(row["question_chars"]))
        out["log_answer_chars"] = math.log1p(float(row["answer_chars"]))
        out["log_solution_chars"] = math.log1p(float(row["solution_chars"]))
        out["log_tokens_in"] = math.log1p(float(row["tokens_in"]))
        out["log_tokens_out"] = math.log1p(float(row["tokens_out"]))
        return out

    def transform(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        x_rows = []
        y = []
        for raw in rows:
            row = self.prepare_row(raw)
            feats = [1.0]
            for key in self.spec.continuous:
                val = float(row[key])
                if key != "d_c":
                    val = (val - self.means[key]) / self.stds[key]
                feats.append(val)
            for key in self.spec.categorical:
                value = str(row.get(key, "unknown"))
                feats.extend([1.0 if value == lvl else 0.0 for lvl in self.levels[key]])
            x_rows.append(feats)
            y.append(float(row["is_correct"]))
        return np.asarray(x_rows, dtype=float), np.asarray(y, dtype=float)


def fit_logit(x: np.ndarray, y: np.ndarray, feature_names: list[str], ridge: float) -> dict:
    n, p = x.shape
    penalty = np.ones(p)
    penalty[0] = 0.0
    if "d_c" in feature_names:
        penalty[feature_names.index("d_c")] = 0.0

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        loss += 0.5 * ridge * float(np.sum(penalty * beta * beta))
        prob = expit(eta)
        grad = x.T @ (prob - y) + ridge * penalty * beta
        return loss, grad

    result = minimize(
        fun=lambda b: objective(b)[0],
        x0=np.zeros(p, dtype=float),
        jac=lambda b: objective(b)[1],
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-6},
    )
    beta = result.x
    prob = expit(x @ beta)
    eps = 1e-12
    loglik = float(np.sum(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
    weights = prob * (1 - prob)
    hessian = (x.T * weights) @ x + ridge * np.diag(penalty)
    try:
        cov = np.linalg.pinv(hessian)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        se = np.full(p, np.nan)
    return {
        "coef": beta,
        "se": se,
        "prob": prob,
        "loglik": loglik,
        "aic": 2 * p - 2 * loglik,
        "bic": math.log(n) * p - 2 * loglik,
        "success": bool(result.success),
        "message": str(result.message),
    }


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normal_p_value(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def cluster_bootstrap_dc(rows: list[dict], builder: DesignBuilder, ridge: float, n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    item_ids = sorted(by_item)
    coefs = []
    dc_idx = builder.feature_names.index("d_c")
    for _ in range(n_boot):
        sample_rows = []
        for iid in rng.choices(item_ids, k=len(item_ids)):
            sample_rows.extend(by_item[iid])
        x, y = builder.transform(sample_rows)
        fit = fit_logit(x, y, builder.feature_names, ridge)
        coefs.append(float(fit["coef"][dc_idx]))
    return float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))


def observed_bin_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["model_label"], row["d_c"])].append(int(row["is_correct"]))
    out = []
    for (model, dc), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append(
            {
                "model_label": model,
                "d_c": dc,
                "n": len(vals),
                "accuracy": float(np.mean(vals)),
            }
        )
    return out


def controlled_marginals(rows: list[dict], builder: DesignBuilder, fit: dict) -> list[dict]:
    out = []
    beta = fit["coef"]
    for dc in sorted({int(r["d_c"]) for r in rows}):
        changed = [dict(r, d_c=dc) for r in rows]
        x, _ = builder.transform(changed)
        out.append({"d_c": dc, "mean_predicted_accuracy": float(expit(x @ beta).mean())})
    return out


def make_figure(observed: list[dict], marginal: list[dict], fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in observed:
        by_model[row["model_label"]].append(row)
    for model, vals in sorted(by_model.items()):
        vals = sorted(vals, key=lambda r: int(r["d_c"]))
        ax.plot(
            [r["d_c"] for r in vals],
            [r["accuracy"] for r in vals],
            marker="o",
            alpha=0.55,
            linewidth=1.2,
            label=f"Observed {model}",
        )
    ax.plot(
        [r["d_c"] for r in marginal],
        [r["mean_predicted_accuracy"] for r in marginal],
        marker="s",
        color="black",
        linewidth=2.4,
        label="Controlled marginal (M3)",
    )
    ax.set_xlabel("Consensus d_c")
    ax.set_ylabel("Accuracy / predicted accuracy")
    ax.set_title("W6 controlled d_c effect on pilot258")
    ax.set_ylim(-0.02, 0.32)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.savefig(fig_path.with_suffix(".pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item-csv", type=Path, default=PROJECT / "data/annotations/pilot258_with_solution.csv")
    ap.add_argument("--dc-csv", type=Path, default=PROJECT / "data/annotations/pilot258_v4a_qwen14b_merged.csv")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--panel-csv", type=Path, default=PROJECT / "evaluation/w6_controlled_panel.csv")
    ap.add_argument("--coef-csv", type=Path, default=PROJECT / "evaluation/w6_logit_coefficients.csv")
    ap.add_argument("--bin-csv", type=Path, default=PROJECT / "evaluation/w6_observed_dc_bins.csv")
    ap.add_argument("--marginal-csv", type=Path, default=PROJECT / "evaluation/w6_controlled_marginal_by_dc.csv")
    ap.add_argument("--report", type=Path, default=PROJECT / "evaluation/W6_controlled_dc_analysis.md")
    ap.add_argument("--fig", type=Path, default=PROJECT / "figures/F3_w6_controlled_dc_effect.png")
    args = ap.parse_args()

    rows = build_panel(args.item_csv, args.dc_csv)
    if not rows:
        raise SystemExit("No panel rows built.")

    panel_fields = [
        "item_id",
        "model_label",
        "n_params_b",
        "source_benchmark",
        "topic",
        "has_diagram",
        "answer_type",
        "d_c",
        "question_chars",
        "answer_chars",
        "solution_chars",
        "tokens_in",
        "tokens_out",
        "latency_s",
        "is_correct",
    ]
    write_csv(args.panel_csv, rows, panel_fields)

    coef_rows = []
    fits = {}
    builders = {}
    for spec in SPECS:
        builder = DesignBuilder(rows, spec)
        x, y = builder.transform(rows)
        fit = fit_logit(x, y, builder.feature_names, args.ridge)
        fits[spec.name] = fit
        builders[spec.name] = builder
        dc_idx = builder.feature_names.index("d_c")
        beta = float(fit["coef"][dc_idx])
        se = float(fit["se"][dc_idx])
        z = beta / se if se > 0 else float("nan")
        boot_low = boot_high = ""
        if spec.name == "M3_topic_controls" and args.bootstrap > 0:
            boot_low, boot_high = cluster_bootstrap_dc(rows, builder, args.ridge, args.bootstrap, args.seed)
        coef_rows.append(
            {
                "spec": spec.name,
                "n_obs": len(rows),
                "n_features": len(builder.feature_names),
                "ridge": args.ridge,
                "dc_beta": beta,
                "dc_se": se,
                "dc_z": z,
                "dc_p_approx": normal_p_value(z) if not math.isnan(z) else "",
                "dc_odds_ratio_per_unit": math.exp(beta),
                "dc_bootstrap_ci_low": boot_low,
                "dc_bootstrap_ci_high": boot_high,
                "loglik": fit["loglik"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "auc": auc_score(y, fit["prob"]),
                "fit_success": fit["success"],
            }
        )
    coef_fields = list(coef_rows[0].keys())
    write_csv(args.coef_csv, coef_rows, coef_fields)

    bins = observed_bin_rows(rows)
    write_csv(args.bin_csv, bins, ["model_label", "d_c", "n", "accuracy"])

    main_builder = builders["M3_topic_controls"]
    main_fit = fits["M3_topic_controls"]
    marginal = controlled_marginals(rows, main_builder, main_fit)
    write_csv(args.marginal_csv, marginal, ["d_c", "mean_predicted_accuracy"])
    make_figure(bins, marginal, args.fig)

    d_counts = Counter(int(r["d_c"]) for r in rows if r["model_label"] == "Qwen14B")
    source_counts = Counter(r["source_benchmark"] for r in rows if r["model_label"] == "Qwen14B")
    topic_counts = Counter(r["topic"] for r in rows if r["model_label"] == "Qwen14B")
    overall = Counter()
    for r in rows:
        overall[r["model_label"]] += int(r["is_correct"])

    m3 = next(r for r in coef_rows if r["spec"] == "M3_topic_controls")
    lines = [
        "# W6 Controlled d_c Analysis\n\n",
        "Offline-only analysis using existing pilot258 judged outputs. No model/API calls were made.\n\n",
        "## Panel\n\n",
        f"- Items: {len({r['item_id'] for r in rows})}\n",
        f"- Observations: {len(rows)} ({len(MODEL_SPECS)} model runs x pilot258)\n",
        f"- Models: {', '.join(label for label, _, _ in MODEL_SPECS)}\n",
        f"- d_c counts: {dict(sorted(d_counts.items()))}\n",
        f"- Source counts: {dict(sorted(source_counts.items()))}\n",
        f"- Topic count: {len(topic_counts)}\n\n",
        "## Logistic control ladder\n\n",
        "| Spec | Controls | d_c beta | OR per +1 d_c | AUC | AIC | BIC |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    control_labels = {
        "M0_dc_only": "none",
        "M1_item_controls": "item length + source + answer type",
        "M2_model_controls": "M1 + model FE + token budget",
        "M3_topic_controls": "M2 + topic FE",
    }
    for row in coef_rows:
        lines.append(
            f"| {row['spec']} | {control_labels[row['spec']]} | {row['dc_beta']:.4f} | "
            f"{row['dc_odds_ratio_per_unit']:.3f} | {row['auc']:.3f} | {row['aic']:.1f} | {row['bic']:.1f} |\n"
        )
    lines.extend(
        [
            "\n## Main controlled result\n\n",
            f"- Main spec: `M3_topic_controls`.\n",
            f"- d_c beta = {m3['dc_beta']:.4f}; odds ratio per +1 d_c = {m3['dc_odds_ratio_per_unit']:.3f}.\n",
            f"- Cluster bootstrap 95% CI for beta by item = [{float(m3['dc_bootstrap_ci_low']):.4f}, {float(m3['dc_bootstrap_ci_high']):.4f}] (B={args.bootstrap}).\n",
            "- Interpretation: negative beta means higher conservation dimension predicts lower judged correctness after local controls.\n\n",
            "## Controlled marginal accuracy\n\n",
            "| d_c | mean predicted accuracy (M3) |\n",
            "|---:|---:|\n",
        ]
    )
    for row in marginal:
        lines.append(f"| {row['d_c']} | {row['mean_predicted_accuracy']:.3f} |\n")
    lines.extend(
        [
            "\n## Reviewer-risk notes\n\n",
            "- This is still a pilot panel, not the final 15 x 8 matrix.\n",
            "- Correctness labels are produced by a Qwen14B judge, so judge-family bias is not eliminated.\n",
            "- High-d_c bins remain sparse; controlled estimates for d_c>=4 are extrapolative.\n",
            "- Topic fixed effects are ridge-regularized because pilot258 has many small topic cells.\n\n",
            "## Generated artifacts\n\n",
            f"- `{args.panel_csv.relative_to(PROJECT)}`\n",
            f"- `{args.coef_csv.relative_to(PROJECT)}`\n",
            f"- `{args.bin_csv.relative_to(PROJECT)}`\n",
            f"- `{args.marginal_csv.relative_to(PROJECT)}`\n",
            f"- `{args.fig.relative_to(PROJECT)}`\n",
        ]
    )
    args.report.write_text("".join(lines), encoding="utf-8")

    print(f"wrote={args.panel_csv}")
    print(f"wrote={args.coef_csv}")
    print(f"wrote={args.bin_csv}")
    print(f"wrote={args.marginal_csv}")
    print(f"wrote={args.report}")
    print(f"wrote={args.fig}")


if __name__ == "__main__":
    main()

