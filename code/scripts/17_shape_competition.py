"""Compare simple accuracy-vs-d_c shape models.

This is an offline diagnostic for the high-journal-prep gate. It does not make
new model calls and should not be treated as proof of a universal capacity law.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[2]
EPS = 1e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1.0 - EPS)


def neg_loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = clip(p)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_model(name: str, d: np.ndarray, y: np.ndarray) -> tuple[float, int, dict[str, float]]:
    if name == "logistic":
        def pred(theta):
            return expit(theta[0] + theta[1] * d)
        init = np.array([math.log((y.mean() + EPS) / (1 - y.mean() + EPS)), -0.3])
        bounds = [(None, None), (None, None)]
    elif name == "exp_decay":
        def pred(theta):
            floor = expit(theta[0])
            amp = expit(theta[1]) * (1.0 - floor)
            rate = math.exp(theta[2])
            return floor + amp * np.exp(-rate * d)
        init = np.array([-4.0, -1.0, -0.5])
        bounds = [(None, None), (None, None), (None, None)]
    elif name == "gompertz":
        def pred(theta):
            upper = expit(theta[0])
            b = math.exp(theta[1])
            c = math.exp(theta[2])
            return upper * np.exp(-b * np.exp(c * d))
        init = np.array([-1.0, -1.0, -1.0])
        bounds = [(None, None), (None, None), (None, None)]
    elif name == "envelope_dplus_delta":
        def pred(theta):
            k = math.exp(theta[0])
            delta = math.exp(theta[1])
            scale = expit(theta[2])
            return scale * (1.0 - np.exp(-k / (d + delta)))
        init = np.array([-1.5, 0.0, -0.5])
        bounds = [(None, None), (None, None), (None, None)]
    else:
        raise ValueError(name)

    def obj(theta):
        return neg_loglik(y, pred(theta))

    result = minimize(obj, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    theta = result.x
    return float(result.fun), len(theta), {f"theta{i}": float(v) for i, v in enumerate(theta)}


def isotonic_decreasing_fit(d: np.ndarray, y: np.ndarray) -> tuple[float, int, dict[str, float]]:
    # Pool adjacent violators on grouped d_c means, constrained non-increasing.
    groups = []
    for value in sorted(set(d.tolist())):
        vals = y[d == value]
        groups.append({"d": value, "n": len(vals), "sum": float(vals.sum()), "mean": float(vals.mean())})
    blocks = []
    for group in groups:
        blocks.append(dict(group))
        while len(blocks) >= 2 and blocks[-2]["mean"] < blocks[-1]["mean"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            merged = {
                "d": b1["d"],
                "n": b1["n"] + b2["n"],
                "sum": b1["sum"] + b2["sum"],
            }
            merged["mean"] = merged["sum"] / merged["n"]
            blocks.append(merged)
    pred_by_d = {}
    idx = 0
    for block in blocks:
        covered = groups[idx: idx + block["n_groups"]] if "n_groups" in block else None
        idx += 0
    # Reconstruct by assigning each original grouped d to the block that contains it.
    pred_by_d = {}
    start = 0
    raw_groups = [{"d": g["d"], "n": g["n"], "sum": g["sum"], "mean": g["mean"]} for g in groups]
    blocks = []
    for group in raw_groups:
        blocks.append({"members": [group], "n": group["n"], "sum": group["sum"], "mean": group["mean"]})
        while len(blocks) >= 2 and blocks[-2]["mean"] < blocks[-1]["mean"]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            merged_members = b1["members"] + b2["members"]
            merged_n = b1["n"] + b2["n"]
            merged_sum = b1["sum"] + b2["sum"]
            blocks.append({
                "members": merged_members,
                "n": merged_n,
                "sum": merged_sum,
                "mean": merged_sum / merged_n,
            })
    for block in blocks:
        for member in block["members"]:
            pred_by_d[member["d"]] = block["mean"]
    p = np.array([pred_by_d[v] for v in d], dtype=float)
    return neg_loglik(y, p), len(blocks), {f"d{int(k)}": float(v) for k, v in sorted(pred_by_d.items())}


def fit_family(rows: list[dict[str, str]], label: str) -> list[dict]:
    subset = rows if label == "pooled" else [r for r in rows if r.get("model_label") == label]
    d = []
    y = []
    for row in subset:
        try:
            d.append(float(row["d_c"]))
            y.append(float(row["is_correct"]))
        except Exception:
            pass
    d_arr = np.asarray(d, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    out = []
    if len(y_arr) < 10 or len(set(y_arr.tolist())) < 2:
        return out
    for name in ["logistic", "exp_decay", "gompertz", "envelope_dplus_delta"]:
        nll, k, params = fit_model(name, d_arr, y_arr)
        out.append({
            "family": label,
            "shape_model": name,
            "n": len(y_arr),
            "k_params": k,
            "neg_loglik": nll,
            "aic": 2 * k + 2 * nll,
            "bic": math.log(len(y_arr)) * k + 2 * nll,
            "params_json": params,
        })
    nll, k, params = isotonic_decreasing_fit(d_arr, y_arr)
    out.append({
        "family": label,
        "shape_model": "monotone_spline_pava",
        "n": len(y_arr),
        "k_params": k,
        "neg_loglik": nll,
        "aic": 2 * k + 2 * nll,
        "bic": math.log(len(y_arr)) * k + 2 * nll,
        "params_json": params,
    })
    best_aic = min(r["aic"] for r in out)
    best_bic = min(r["bic"] for r in out)
    for row in out:
        row["delta_aic"] = row["aic"] - best_aic
        row["delta_bic"] = row["bic"] - best_bic
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=PROJECT / "evaluation/w6_controlled_panel.csv")
    ap.add_argument("--out-csv", type=Path, default=PROJECT / "evaluation/shape_competition_20260525.csv")
    ap.add_argument("--report", type=Path, default=PROJECT / "evaluation/shape_competition_20260525.md")
    args = ap.parse_args()

    rows = read_csv(args.panel)
    families = ["pooled"] + sorted({r.get("model_label", "") for r in rows if r.get("model_label")})
    out_rows = []
    for fam in families:
        out_rows.extend(fit_family(rows, fam))
    fieldnames = ["family", "shape_model", "n", "k_params", "neg_loglik", "aic", "bic", "delta_aic", "delta_bic", "params_json"]
    serializable = []
    for row in out_rows:
        serializable.append({**row, "params_json": str(row["params_json"])})
    write_csv(args.out_csv, serializable, fieldnames)

    lines = [
        "# Shape Competition for d_c Accuracy Curves\n\n",
        "Offline comparison on `evaluation/w6_controlled_panel.csv`. This is a diagnostic gate, not a universal-law proof.\n\n",
        "| family | best AIC | best BIC | envelope delta BIC |\n",
        "|---|---|---|---:|\n",
    ]
    for fam in families:
        fam_rows = [r for r in out_rows if r["family"] == fam]
        if not fam_rows:
            continue
        best_aic = min(fam_rows, key=lambda r: r["aic"])
        best_bic = min(fam_rows, key=lambda r: r["bic"])
        env = next(r for r in fam_rows if r["shape_model"] == "envelope_dplus_delta")
        lines.append(f"| {fam} | {best_aic['shape_model']} | {best_bic['shape_model']} | {env['delta_bic']:.2f} |\n")
    lines.extend([
        "\n## Interpretation rule\n\n",
        "- `envelope_dplus_delta` must be competitive with simpler alternatives before any high-journal claim about envelope shape is promoted.\n",
        "- If logistic or monotone spline wins clearly, the manuscript should keep the envelope as an explanatory ansatz and lead with `d_c` as an independent predictor.\n",
        "- High-d_c bins remain sparse in pilot258, so this comparison is underpowered at the right tail.\n",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("".join(lines), encoding="utf-8")
    print(f"[done] wrote {args.out_csv}")
    print(f"[done] wrote {args.report}")


if __name__ == "__main__":
    main()
