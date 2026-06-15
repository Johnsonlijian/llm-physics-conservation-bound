"""Add text-derived difficulty proxies to the W6 d_c control ladder.

Offline only. This script does not call any model or external API.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def normal_p_value(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


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


def merge_panel(panel_path: Path, proxy_path: Path) -> list[dict]:
    proxies = {row["item_id"]: row for row in read_csv(proxy_path)}
    rows: list[dict] = []
    for row in read_csv(panel_path):
        proxy = proxies.get(row.get("item_id", ""))
        if not proxy:
            continue
        merged = dict(row)
        for key in [
            "answer_type_proxy",
            "benchmark_difficulty",
            "benchmark_subfield",
            "question_number_count",
            "solution_number_count",
            "question_equation_count",
            "solution_equation_count",
            "body_entity_term_count",
            "latex_symbol_count",
        ]:
            merged[key] = proxy.get(key, "")
        rows.append(merged)
    return rows


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    continuous: list[str]
    categorical: list[str]


BASE_CONTROLS = [
    "d_c",
    "log_question_chars",
    "log_answer_chars",
    "log_solution_chars",
    "has_diagram",
    "log_tokens_in",
    "log_tokens_out",
]

PROXY_CONTROLS = [
    "log_question_number_count",
    "log_solution_number_count",
    "log_question_equation_count",
    "log_solution_equation_count",
    "log_body_entity_term_count",
    "log_latex_symbol_count",
]

SPECS = [
    FeatureSpec(
        "M3_original_topic_controls",
        BASE_CONTROLS,
        ["source_benchmark", "answer_type", "model_label", "topic"],
    ),
    FeatureSpec(
        "M4_plus_text_difficulty_proxies",
        BASE_CONTROLS + PROXY_CONTROLS,
        ["source_benchmark", "answer_type", "model_label", "topic"],
    ),
    FeatureSpec(
        "M5_plus_answer_proxy",
        BASE_CONTROLS + PROXY_CONTROLS,
        ["source_benchmark", "answer_type", "answer_type_proxy", "model_label", "topic"],
    ),
]


class DesignBuilder:
    def __init__(self, rows: list[dict], spec: FeatureSpec):
        self.spec = spec
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.levels: dict[str, list[str]] = {}
        prepared = [self.prepare_row(row) for row in rows]

        for key in spec.continuous:
            if key == "d_c":
                continue
            vals = np.array([float(row[key]) for row in prepared], dtype=float)
            self.means[key] = float(vals.mean())
            std = float(vals.std())
            self.stds[key] = std if std > 1e-9 else 1.0

        for key in spec.categorical:
            levels = sorted({self.clean_category(row.get(key, "")) for row in prepared})
            self.levels[key] = levels[1:] if len(levels) > 1 else []

        self.feature_names = ["intercept"]
        for key in spec.continuous:
            self.feature_names.append(key if key == "d_c" else f"z_{key}")
        for key in spec.categorical:
            self.feature_names.extend([f"{key}={level}" for level in self.levels[key]])

    @staticmethod
    def clean_category(value: object) -> str:
        text = str(value or "").strip()
        return text if text else "missing"

    @staticmethod
    def prepare_row(row: dict) -> dict:
        out = dict(row)
        out["d_c"] = as_float(row.get("d_c"))
        out["has_diagram"] = as_float(row.get("has_diagram"))
        out["log_question_chars"] = math.log1p(as_float(row.get("question_chars")))
        out["log_answer_chars"] = math.log1p(as_float(row.get("answer_chars")))
        out["log_solution_chars"] = math.log1p(as_float(row.get("solution_chars")))
        out["log_tokens_in"] = math.log1p(as_float(row.get("tokens_in")))
        out["log_tokens_out"] = math.log1p(as_float(row.get("tokens_out")))
        out["log_question_number_count"] = math.log1p(as_float(row.get("question_number_count")))
        out["log_solution_number_count"] = math.log1p(as_float(row.get("solution_number_count")))
        out["log_question_equation_count"] = math.log1p(as_float(row.get("question_equation_count")))
        out["log_solution_equation_count"] = math.log1p(as_float(row.get("solution_equation_count")))
        out["log_body_entity_term_count"] = math.log1p(as_float(row.get("body_entity_term_count")))
        out["log_latex_symbol_count"] = math.log1p(as_float(row.get("latex_symbol_count")))
        return out

    def transform(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        x_rows: list[list[float]] = []
        y: list[float] = []
        for raw in rows:
            row = self.prepare_row(raw)
            feats = [1.0]
            for key in self.spec.continuous:
                value = float(row[key])
                if key != "d_c":
                    value = (value - self.means[key]) / self.stds[key]
                feats.append(value)
            for key in self.spec.categorical:
                value = self.clean_category(row.get(key, ""))
                feats.extend([1.0 if value == level else 0.0 for level in self.levels[key]])
            x_rows.append(feats)
            y.append(float(as_int(raw.get("is_correct"))))
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
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(p, dtype=float),
        jac=lambda beta: objective(beta)[1],
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


def cluster_bootstrap_dc(
    rows: list[dict],
    builder: DesignBuilder,
    ridge: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    item_ids = sorted(by_item)
    coefs: list[float] = []
    dc_idx = builder.feature_names.index("d_c")
    for _ in range(n_boot):
        sample: list[dict] = []
        for item_id in rng.choices(item_ids, k=len(item_ids)):
            sample.extend(by_item[item_id])
        x, y = builder.transform(sample)
        fit = fit_logit(x, y, builder.feature_names, ridge)
        coefs.append(float(fit["coef"][dc_idx]))
    return float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))


def controlled_marginals(rows: list[dict], builder: DesignBuilder, fit: dict) -> list[dict]:
    beta = fit["coef"]
    out: list[dict] = []
    for dc in sorted({as_int(row["d_c"]) for row in rows}):
        changed = [dict(row, d_c=dc) for row in rows]
        x, _ = builder.transform(changed)
        out.append({"d_c": dc, "mean_predicted_accuracy": float(expit(x @ beta).mean())})
    return out


def make_beta_figure(rows: list[dict], fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    labels = [row["spec"] for row in rows]
    betas = [float(row["dc_beta"]) for row in rows]
    lows = [float(row["dc_bootstrap_ci_low"]) for row in rows]
    highs = [float(row["dc_bootstrap_ci_high"]) for row in rows]
    ypos = np.arange(len(rows))
    xerr = np.array([[b - lo for b, lo in zip(betas, lows)], [hi - b for b, hi in zip(betas, highs)]])
    ax.errorbar(betas, ypos, xerr=xerr, fmt="o", color="black", ecolor="#4b5563", capsize=4)
    ax.axvline(0, color="#b91c1c", linewidth=1.2, linestyle="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("d_c coefficient (ridge logit)")
    ax.set_title("W6 d_c effect after adding difficulty proxies")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    fig.savefig(fig_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PROJECT / "evaluation/w6_controlled_panel.csv")
    parser.add_argument("--proxies", type=Path, default=PROJECT / "data/annotations/pilot258_difficulty_proxies.csv")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--out-panel", type=Path, default=PROJECT / "evaluation/w6_proxy_control_panel.csv")
    parser.add_argument(
        "--out-coef",
        type=Path,
        default=PROJECT / "evaluation/w6_proxy_control_coefficients_20260525.csv",
    )
    parser.add_argument(
        "--out-marginal",
        type=Path,
        default=PROJECT / "evaluation/w6_proxy_control_marginal_20260525.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "evaluation/W6_difficulty_proxy_controls_20260525.md",
    )
    parser.add_argument("--fig", type=Path, default=PROJECT / "figures/F3b_w6_proxy_control_dc_beta.png")
    args = parser.parse_args()

    rows = merge_panel(args.panel, args.proxies)
    if not rows:
        raise SystemExit("No rows after merging W6 panel and difficulty proxies.")

    panel_fields = list(rows[0].keys())
    write_csv(args.out_panel, rows, panel_fields)

    coef_rows: list[dict] = []
    fits: dict[str, dict] = {}
    builders: dict[str, DesignBuilder] = {}
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
        ci_low, ci_high = cluster_bootstrap_dc(rows, builder, args.ridge, args.bootstrap, args.seed)
        coef_rows.append(
            {
                "spec": spec.name,
                "controls": ";".join(spec.continuous + spec.categorical),
                "n_obs": len(rows),
                "n_items": len({row["item_id"] for row in rows}),
                "n_features": len(builder.feature_names),
                "ridge": args.ridge,
                "dc_beta": beta,
                "dc_se": se,
                "dc_z": z,
                "dc_p_approx": normal_p_value(z) if not math.isnan(z) else "",
                "dc_odds_ratio_per_unit": math.exp(beta),
                "dc_bootstrap_ci_low": ci_low,
                "dc_bootstrap_ci_high": ci_high,
                "loglik": fit["loglik"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "auc": auc_score(y, fit["prob"]),
                "fit_success": fit["success"],
                "fit_message": fit["message"],
            }
        )

    write_csv(args.out_coef, coef_rows, list(coef_rows[0].keys()))
    main_name = "M5_plus_answer_proxy"
    marginal = controlled_marginals(rows, builders[main_name], fits[main_name])
    write_csv(args.out_marginal, marginal, ["d_c", "mean_predicted_accuracy"])
    make_beta_figure(coef_rows, args.fig)

    dc_counts = Counter(as_int(row["d_c"]) for row in rows if row["model_label"] == "Qwen14B")
    source_counts = Counter(row["source_benchmark"] for row in rows if row["model_label"] == "Qwen14B")
    best_bic = min(coef_rows, key=lambda row: float(row["bic"]))
    main = next(row for row in coef_rows if row["spec"] == main_name)

    lines = [
        "# W6 Difficulty-Proxy Control Analysis\n\n",
        "Offline robustness check. This analysis adds text-derived difficulty proxies to the W6 control ladder and makes no model/API calls.\n\n",
        "## Inputs\n\n",
        f"- W6 panel: `{args.panel.relative_to(PROJECT)}`\n",
        f"- Difficulty proxies: `{args.proxies.relative_to(PROJECT)}`\n",
        f"- Observations: {len(rows)}\n",
        f"- Items: {len({row['item_id'] for row in rows})}\n",
        f"- Source distribution: {dict(sorted(source_counts.items()))}\n",
        f"- d_c distribution: {dict(sorted(dc_counts.items()))}\n\n",
        "## Control Ladder\n\n",
        "| Spec | Added controls | d_c beta | OR | bootstrap 95% CI | AUC | BIC |\n",
        "|---|---|---:|---:|---|---:|---:|\n",
    ]
    added = {
        "M3_original_topic_controls": "original W6 M3 controls",
        "M4_plus_text_difficulty_proxies": "M3 + number/equation/entity/latex-count proxies",
        "M5_plus_answer_proxy": "M4 + answer-type proxy",
    }
    for row in coef_rows:
        lines.append(
            f"| {row['spec']} | {added[row['spec']]} | {row['dc_beta']:.4f} | "
            f"{row['dc_odds_ratio_per_unit']:.3f} | "
            f"[{row['dc_bootstrap_ci_low']:.4f}, {row['dc_bootstrap_ci_high']:.4f}] | "
            f"{row['auc']:.3f} | {row['bic']:.1f} |\n"
        )

    lines.extend(
        [
            "\n## Main Reading\n\n",
            f"- Best BIC in this ladder: `{best_bic['spec']}`.\n",
            f"- Most conservative proxy spec (`{main_name}`): d_c beta = {main['dc_beta']:.4f}, OR = {main['dc_odds_ratio_per_unit']:.3f}, bootstrap 95% CI = [{main['dc_bootstrap_ci_low']:.4f}, {main['dc_bootstrap_ci_high']:.4f}].\n",
        ]
    )
    if float(main["dc_bootstrap_ci_high"]) < 0:
        lines.append("- Interpretation: the negative d_c effect survives additional text-derived difficulty proxies in this pilot.\n")
    else:
        lines.append("- Interpretation: the proxy-expanded CI crosses zero; keep only a weaker, exploratory claim until human-gold and expanded samples arrive.\n")
    lines.extend(
        [
            "\n## Controlled Marginal Accuracy (proxy-expanded spec)\n\n",
            "| d_c | mean predicted accuracy |\n",
            "|---:|---:|\n",
        ]
    )
    for row in marginal:
        lines.append(f"| {row['d_c']} | {row['mean_predicted_accuracy']:.3f} |\n")

    lines.extend(
        [
            "\n## Boundary Notes\n\n",
            "- These are reproducible proxies, not human difficulty labels.\n",
            "- `solution_*` proxies use benchmark-provided solution text, which is valid for item difficulty characterization but not a solver input.\n",
            "- Correctness labels remain LLM-judge labels until the human correctness packet returns.\n",
            "- High-d_c bins remain sparse; this does not replace high-d_c expansion.\n\n",
            "## Outputs\n\n",
            f"- `{args.out_panel.relative_to(PROJECT)}`\n",
            f"- `{args.out_coef.relative_to(PROJECT)}`\n",
            f"- `{args.out_marginal.relative_to(PROJECT)}`\n",
            f"- `{args.fig.relative_to(PROJECT)}`\n",
        ]
    )
    args.report.write_text("".join(lines), encoding="utf-8")

    print(f"wrote={args.out_panel}")
    print(f"wrote={args.out_coef}")
    print(f"wrote={args.out_marginal}")
    print(f"wrote={args.report}")
    print(f"wrote={args.fig}")


if __name__ == "__main__":
    main()
