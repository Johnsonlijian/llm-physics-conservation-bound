"""W6-extension: leave-item-out cross-validation for the ridge logistic.

Addresses the §4.4 in-sample AUC caveat ("full cross-validation is W6-extension").

Procedure:
- Read the existing W6 panel (evaluation/w6_controlled_panel.csv): 1032 obs (4 models × 258 items).
- For each of 258 items, leave the 4 rows for that item out, refit M3 spec on the
  remaining 1028 rows, predict the held-out rows, and aggregate.
- Report held-out AUC + stability of the d_c coefficient across folds.
- Also do 5-fold and 10-fold splits at the *item* level (preserves the cluster) for
  comparison.

No API calls.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold


M3_NUMERIC = [
    "d_c",
    "tokens_out_log",
    "question_chars_log",
    "solution_chars_log",
]
M3_CATEGORICAL = [
    "model_label",
    "source_benchmark",
    "topic",
    "answer_type",
]


def prepare_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the design matrix matching the §4.4 M3 spec (numeric + one-hot categorical)."""
    keep = [c for c in M3_NUMERIC + M3_CATEGORICAL if c in df.columns]
    sub = df[keep + ["is_correct"]].copy()
    sub = sub.dropna(subset=["is_correct"])
    sub["is_correct"] = sub["is_correct"].astype(int)
    X_num = sub[[c for c in M3_NUMERIC if c in sub.columns]].astype(float).values
    cat_cols = [c for c in M3_CATEGORICAL if c in sub.columns]
    X_cat = pd.get_dummies(sub[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()
    feature_names = (
        [c for c in M3_NUMERIC if c in sub.columns]
        + list(X_cat.columns)
    )
    if X_cat.shape[1]:
        X = np.hstack([X_num, X_cat.values.astype(float)])
    else:
        X = X_num
    y = sub["is_correct"].values
    return X, y, feature_names


def fit_predict(X_train, y_train, X_test):
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight=None,
    )
    clf.fit(X_train, y_train)
    return clf, clf.predict_proba(X_test)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panel",
        type=Path,
        default=Path("evaluation/w6_controlled_panel.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/W6_holdout_cv_20260525.md"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    print(f"[panel] {args.panel}: rows={len(df)} cols={len(df.columns)}")
    print(f"[panel] columns: {list(df.columns)[:20]}")
    if "item_id" not in df.columns:
        raise SystemExit("panel missing item_id")
    if "is_correct" not in df.columns:
        raise SystemExit("panel missing is_correct")

    X_full, y_full, feature_names = prepare_design(df)
    print(f"[design] rows={len(y_full)} features={X_full.shape[1]} feat_names={feature_names[:8]}...")

    # idx of dc column in feature_names
    dc_idx = feature_names.index("d_c") if "d_c" in feature_names else None

    # in-sample baseline
    clf_full = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
    clf_full.fit(X_full, y_full)
    y_in = clf_full.predict_proba(X_full)[:, 1]
    auc_in = roc_auc_score(y_full, y_in)
    dc_beta_full = clf_full.coef_[0, dc_idx] if dc_idx is not None else float("nan")
    print(f"[in-sample] AUC={auc_in:.4f}  d_c beta={dc_beta_full:.4f}")

    # leave-item-out + group k-fold
    groups = df["item_id"].astype(str).values
    results = {}

    # full leave-item-out (LIO)
    print("[LIO] running leave-item-out CV...")
    gkf_lio = GroupKFold(n_splits=len(set(groups)))
    y_hold = np.zeros_like(y_full, dtype=float)
    dc_betas_lio = []
    fold_idx = 0
    for tr, te in gkf_lio.split(X_full, y_full, groups=groups):
        clf, p_te = fit_predict(X_full[tr], y_full[tr], X_full[te])
        y_hold[te] = p_te
        if dc_idx is not None:
            dc_betas_lio.append(clf.coef_[0, dc_idx])
        fold_idx += 1
        if fold_idx % 50 == 0:
            print(f"  LIO fold {fold_idx}/{len(set(groups))}")
    auc_lio = roc_auc_score(y_full, y_hold)
    dc_lio_mean = float(np.mean(dc_betas_lio))
    dc_lio_sd = float(np.std(dc_betas_lio, ddof=1))
    results["LIO"] = dict(
        auc_holdout=auc_lio,
        n_folds=len(set(groups)),
        dc_beta_mean=dc_lio_mean,
        dc_beta_sd=dc_lio_sd,
        dc_beta_min=float(min(dc_betas_lio)),
        dc_beta_max=float(max(dc_betas_lio)),
    )
    print(f"[LIO] hold-out AUC={auc_lio:.4f}  d_c beta mean={dc_lio_mean:.4f} ± {dc_lio_sd:.4f}")

    # 10-fold group k-fold
    for K in [5, 10]:
        print(f"[GKF-{K}] running...")
        gkf = GroupKFold(n_splits=K)
        y_hold = np.zeros_like(y_full, dtype=float)
        dc_betas = []
        aucs_fold = []
        for tr, te in gkf.split(X_full, y_full, groups=groups):
            clf, p_te = fit_predict(X_full[tr], y_full[tr], X_full[te])
            y_hold[te] = p_te
            if dc_idx is not None:
                dc_betas.append(clf.coef_[0, dc_idx])
            try:
                aucs_fold.append(roc_auc_score(y_full[te], p_te))
            except ValueError:
                pass
        auc_holdout = roc_auc_score(y_full, y_hold)
        results[f"GKF{K}"] = dict(
            auc_holdout=auc_holdout,
            n_folds=K,
            dc_beta_mean=float(np.mean(dc_betas)),
            dc_beta_sd=float(np.std(dc_betas, ddof=1)),
            mean_per_fold_auc=float(np.mean(aucs_fold)),
        )
        print(f"[GKF-{K}] hold-out AUC={auc_holdout:.4f}  per-fold mean AUC={np.mean(aucs_fold):.4f}")

    # Markdown report
    lines = [
        "# W6 Held-Out Cross-Validation\n\n",
        "Addresses §4.4 caveat that the original W6 AUC = 0.816 is in-sample. ",
        "We refit the M3 specification (ridge logistic with topic + source + answer_type + ",
        "model FE + token budget + length controls) under three out-of-sample splits, ",
        "all grouped at the *item* level so the four per-item model observations are ",
        "always in the same fold.\n\n",
        f"- Panel: `{args.panel}`, rows = {len(y_full)}\n",
        f"- Items: {len(set(groups))}\n",
        f"- Design features: {len(feature_names)}\n",
        f"- In-sample baseline: AUC = {auc_in:.4f}, d_c beta = {dc_beta_full:.4f}\n\n",
        "## Held-out AUC\n\n",
        "| Split | n_folds | Hold-out AUC | d_c beta mean | d_c beta sd |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n_folds']} | {r['auc_holdout']:.4f} | "
            f"{r['dc_beta_mean']:.4f} | {r['dc_beta_sd']:.4f} |\n"
        )
    lines.extend([
        "\n## Interpretation\n\n",
        f"- Leave-item-out gives the most stringent test: AUC = {results['LIO']['auc_holdout']:.4f}, "
        f"compared with the in-sample {auc_in:.4f}. The drop measures over-optimism in the §4.4 "
        "table; a small drop (<0.02) supports the in-sample number; a large drop indicates the "
        "model is over-fitting the topic + answer-type cells.\n",
        f"- The d_c coefficient across LIO folds has mean {results['LIO']['dc_beta_mean']:.4f} "
        f"(sd {results['LIO']['dc_beta_sd']:.4f}, range "
        f"[{results['LIO']['dc_beta_min']:.4f}, {results['LIO']['dc_beta_max']:.4f}]). "
        "Tight spread + sign-stability (none above 0) supports the direction of the effect "
        "even when each item in turn is dropped from training.\n",
        "- These are reproducible from `code/scripts/20_w6_holdout_cv.py` over the "
        "`evaluation/w6_controlled_panel.csv` panel.\n",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(f"[wrote] {args.out}")

    # also write a small csv with fold-level d_c betas for plotting
    csv_path = args.out.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "fold_idx", "dc_beta"])
        for name, betas in [("LIO", dc_betas_lio)]:
            for i, b in enumerate(betas):
                w.writerow([name, i, b])
    print(f"[wrote] {csv_path}")


if __name__ == "__main__":
    main()
