"""Hierarchical envelope fit with partial pooling per model family.

Three nested specifications:
  - **Pooled**: single κ_eff for all families combined.
  - **Unpooled**: independent κ_m per family (what §4.3 currently reports).
  - **Hierarchical**: κ_m ~ Normal(μ_κ, σ_κ²) with empirical-Bayes σ_κ estimated by max marginal lik.

For each spec we minimise the *binomial* negative log-likelihood on per-(family, d_c) bin
counts. This is the right likelihood for accuracy data: we don't aggregate to per-bin means
and then ad-hoc fit, we keep per-item binary outcomes.

No PyMC dependency — uses pure scipy.optimize + numpy.

Output:
  - evaluation/hierarchical_envelope_fit_20260525.md
  - evaluation/hierarchical_envelope_fit_20260525.csv  (per-family κ + CI + log-lik)
  - figures/F5_hierarchical_envelope.png
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize


def envelope_prob(d_c: np.ndarray, kappa: float) -> np.ndarray:
    """A(d_c) = 1 - exp(-kappa / d_c) for d_c >= 1; A(d_c=0) handled separately."""
    out = np.zeros_like(d_c, dtype=float)
    pos = d_c >= 1
    safe_d = np.maximum(d_c, 1)
    out[pos] = 1.0 - np.exp(-kappa / safe_d[pos])
    return out


def nll_unpooled_family(d_arr: np.ndarray, y_arr: np.ndarray, kappa: float, eps: float = 1e-9) -> float:
    """Binomial NLL for one family with envelope kappa, on d_c>=1 items."""
    p = envelope_prob(d_arr, kappa)
    p = np.clip(p, eps, 1 - eps)
    return -float(np.sum(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)))


def fit_kappa_family(d_arr: np.ndarray, y_arr: np.ndarray, bounds=(1e-3, 5.0)) -> dict:
    res = minimize_scalar(
        lambda k: nll_unpooled_family(d_arr, y_arr, k),
        bounds=bounds,
        method="bounded",
    )
    k_hat = float(res.x)
    nll_hat = float(res.fun)
    # crude 95% CI via likelihood profile (drop of 1.92)
    target = nll_hat + 1.92
    lo, hi = bounds[0], bounds[1]
    # binary search left
    a, b = bounds[0], k_hat
    for _ in range(50):
        m = (a + b) / 2
        if nll_unpooled_family(d_arr, y_arr, m) > target:
            a = m
        else:
            b = m
    ci_lo = (a + b) / 2
    a, b = k_hat, bounds[1]
    for _ in range(50):
        m = (a + b) / 2
        if nll_unpooled_family(d_arr, y_arr, m) > target:
            b = m
        else:
            a = m
    ci_hi = (a + b) / 2
    return {
        "kappa_hat": k_hat,
        "nll_hat": nll_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": int(len(d_arr)),
    }


def fit_pooled(panel: pd.DataFrame) -> dict:
    sub = panel[panel["d_c"] >= 1]
    return fit_kappa_family(sub["d_c"].values, sub["is_correct"].values)


def fit_unpooled(panel: pd.DataFrame) -> dict:
    out = {}
    for fam, sub in panel[panel["d_c"] >= 1].groupby("model_label"):
        out[fam] = fit_kappa_family(sub["d_c"].values, sub["is_correct"].values)
    return out


def fit_hierarchical(panel: pd.DataFrame, n_grid_mu: int = 41, n_grid_sigma: int = 41) -> dict:
    """Empirical-Bayes hierarchical: pick (mu_kappa, sigma_kappa) to maximize sum over families
    of log( int_kappa p(kappa | mu, sigma) * lik(kappa | data_family) dkappa ),
    approximated on a per-family kappa grid.
    """
    fams = sorted(panel["model_label"].unique())
    fam_data = {
        f: panel[(panel["model_label"] == f) & (panel["d_c"] >= 1)] for f in fams
    }
    kappa_grid = np.linspace(1e-3, 1.5, 201)  # higher-resolution grid
    # per-family log-likelihood on grid
    fam_loglik = {}
    for f, sub in fam_data.items():
        ll = np.array([
            -nll_unpooled_family(sub["d_c"].values, sub["is_correct"].values, k)
            for k in kappa_grid
        ])
        fam_loglik[f] = ll

    def marginal_nll(params):
        mu, log_sigma = params
        sigma = math.exp(log_sigma)
        if sigma < 1e-3:
            return 1e9
        prior = np.exp(-0.5 * ((kappa_grid - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        prior /= prior.sum()  # normalise on grid
        total = 0.0
        for f in fams:
            # integrate exp(loglik) * prior
            ll = fam_loglik[f]
            ll_max = ll.max()
            integrand = np.exp(ll - ll_max) * prior
            log_mlik = ll_max + math.log(integrand.sum() + 1e-300)
            total -= log_mlik
        return total

    # initial guess from pooled
    pooled = fit_pooled(panel)
    init = [pooled["kappa_hat"], math.log(0.05)]
    res = minimize(marginal_nll, init, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4})
    mu_hat, log_sigma_hat = res.x
    sigma_hat = math.exp(log_sigma_hat)
    marg_nll_hat = float(res.fun)

    # per-family posterior mean and 95% credible interval on kappa grid given (mu, sigma)
    prior = np.exp(-0.5 * ((kappa_grid - mu_hat) / sigma_hat) ** 2) / (sigma_hat * math.sqrt(2 * math.pi))
    prior /= prior.sum()
    fam_post = {}
    for f in fams:
        ll = fam_loglik[f]
        post = np.exp(ll - ll.max()) * prior
        post /= post.sum()
        # mean
        mean_k = float((kappa_grid * post).sum())
        # 95% credible interval via cumulative distribution
        cum = np.cumsum(post)
        ci_lo = float(kappa_grid[np.searchsorted(cum, 0.025)])
        ci_hi = float(kappa_grid[np.searchsorted(cum, 0.975)])
        fam_post[f] = dict(
            kappa_mean=mean_k, ci_lo=ci_lo, ci_hi=ci_hi,
            kappa_ml=float(kappa_grid[np.argmax(ll)]),
        )
    return {
        "mu_kappa": float(mu_hat),
        "sigma_kappa": float(sigma_hat),
        "marg_nll": marg_nll_hat,
        "per_family_posterior": fam_post,
    }


def compute_bic(nll: float, k: int, n: int) -> float:
    return 2 * nll + k * math.log(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("evaluation/w6_controlled_panel.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("evaluation/hierarchical_envelope_fit_20260525.md"))
    ap.add_argument("--out-csv", type=Path, default=Path("evaluation/hierarchical_envelope_fit_20260525.csv"))
    ap.add_argument("--out-fig", type=Path, default=Path("figures/F5_hierarchical_envelope.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    print(f"[panel] {args.panel} rows={len(df)}  families={sorted(df['model_label'].unique())}")

    # Pooled
    pooled = fit_pooled(df)
    n_pool = pooled["n"]
    bic_pool = compute_bic(pooled["nll_hat"], 1, n_pool)
    print(f"[pooled] kappa = {pooled['kappa_hat']:.4f}  ci=[{pooled['ci_lo']:.4f},{pooled['ci_hi']:.4f}]  n={n_pool}  BIC={bic_pool:.2f}")

    # Unpooled (independent per family)
    unpooled = fit_unpooled(df)
    nll_unpool = sum(v["nll_hat"] for v in unpooled.values())
    k_unpool = len(unpooled)  # one kappa per family
    n_total = sum(v["n"] for v in unpooled.values())
    bic_unpool = compute_bic(nll_unpool, k_unpool, n_total)
    print(f"[unpooled] families={len(unpooled)}  total NLL={nll_unpool:.2f}  BIC={bic_unpool:.2f}")
    for f, v in unpooled.items():
        print(f"  {f}: kappa={v['kappa_hat']:.4f} ci=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}] n={v['n']}")

    # Hierarchical (partial pooling)
    hier = fit_hierarchical(df)
    # The hierarchical model has 2 free hyperparameters (mu, sigma)
    bic_hier = compute_bic(hier["marg_nll"], 2, n_total)
    print(f"[hierarchical] mu={hier['mu_kappa']:.4f}  sigma={hier['sigma_kappa']:.4f}  marg_nll={hier['marg_nll']:.2f}  BIC={bic_hier:.2f}")
    for f, v in hier["per_family_posterior"].items():
        print(f"  {f}: posterior mean={v['kappa_mean']:.4f}  95%CI=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}]")

    # Write CSV
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spec", "family", "kappa_estimate", "ci_lo", "ci_hi", "nll", "n"])
        w.writerow(["pooled", "ALL", pooled["kappa_hat"], pooled["ci_lo"], pooled["ci_hi"], pooled["nll_hat"], pooled["n"]])
        for fam, v in unpooled.items():
            w.writerow(["unpooled", fam, v["kappa_hat"], v["ci_lo"], v["ci_hi"], v["nll_hat"], v["n"]])
        for fam, v in hier["per_family_posterior"].items():
            w.writerow(["hierarchical", fam, v["kappa_mean"], v["ci_lo"], v["ci_hi"], "", ""])
        w.writerow(["hierarchical_hyperprior", "mu_kappa", hier["mu_kappa"], "", "", hier["marg_nll"], n_total])
        w.writerow(["hierarchical_hyperprior", "sigma_kappa", hier["sigma_kappa"], "", "", "", ""])
    print(f"[wrote] {args.out_csv}")

    # Plot: per-family kappa_eff with CI bars, for each spec
    fams = sorted(unpooled.keys())
    x = np.arange(len(fams))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8, 5))
    # unpooled
    k_un = [unpooled[f]["kappa_hat"] for f in fams]
    e_un_lo = [unpooled[f]["kappa_hat"] - unpooled[f]["ci_lo"] for f in fams]
    e_un_hi = [unpooled[f]["ci_hi"] - unpooled[f]["kappa_hat"] for f in fams]
    ax.errorbar(x - width, k_un, yerr=[e_un_lo, e_un_hi], fmt="o", capsize=4, label="Unpooled (§4.3)", color="C0")
    # hierarchical
    k_hi = [hier["per_family_posterior"][f]["kappa_mean"] for f in fams]
    e_hi_lo = [hier["per_family_posterior"][f]["kappa_mean"] - hier["per_family_posterior"][f]["ci_lo"] for f in fams]
    e_hi_hi = [hier["per_family_posterior"][f]["ci_hi"] - hier["per_family_posterior"][f]["kappa_mean"] for f in fams]
    ax.errorbar(x, k_hi, yerr=[e_hi_lo, e_hi_hi], fmt="s", capsize=4, label="Hierarchical (partial pool)", color="C1")
    # pooled (dashed horizontal line)
    ax.axhline(pooled["kappa_hat"], linestyle="--", color="C2", label=f"Pooled $\\kappa$ = {pooled['kappa_hat']:.3f}")
    ax.axhspan(pooled["ci_lo"], pooled["ci_hi"], color="C2", alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20)
    ax.set_ylabel(r"$\kappa_{\rm eff}$ (envelope $1-\exp(-\kappa/d_c)$)")
    ax.set_title("Hierarchical vs unpooled envelope fit per model family\n(pilot258, $d_c\\geq 1$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    plt.savefig(args.out_fig.with_suffix(".pdf"))
    print(f"[fig] {args.out_fig}")

    # Markdown
    md = [
        "# Hierarchical Envelope Fit\n\n",
        "Three nested specifications of the one-parameter envelope $A(d_c) = 1-\\exp(-\\kappa/d_c)$ on items with $d_c \\geq 1$. "
        "Binomial likelihood on per-item binary outcomes. No PyMC dependency — pure scipy.\n\n",
        f"- Panel: `{args.panel}` (4 families × 258 items = 1032 obs; {n_total} with d_c≥1)\n",
        f"- Families: {sorted(df['model_label'].unique())}\n\n",
        "## Headline\n\n",
        f"| Spec | Free params | Total NLL | BIC |\n",
        f"|---|---:|---:|---:|\n",
        f"| Pooled (single $\\kappa$ across families) | 1 | {pooled['nll_hat']:.2f} | {bic_pool:.2f} |\n",
        f"| Unpooled (independent $\\kappa$ per family, §4.3) | {len(unpooled)} | {nll_unpool:.2f} | {bic_unpool:.2f} |\n",
        f"| Hierarchical ($\\kappa_m \\sim N(\\mu_\\kappa, \\sigma_\\kappa^2)$, marg.) | 2 | {hier['marg_nll']:.2f} | {bic_hier:.2f} |\n\n",
        f"Best BIC: {min([('pooled', bic_pool), ('unpooled', bic_unpool), ('hierarchical', bic_hier)], key=lambda x: x[1])[0]}\n\n",
        f"Hierarchical hyperparameters: $\\hat\\mu_\\kappa = {hier['mu_kappa']:.4f}$, $\\hat\\sigma_\\kappa = {hier['sigma_kappa']:.4f}$. "
        f"The small $\\hat\\sigma_\\kappa$ measures how much $\\kappa$ moves across families: a tighter spread means the four families are more "
        f"consistent with a near-universal envelope, a wider spread that they need family-specific fits.\n\n",
        "## Per-family κ comparison\n\n",
        "| Family | n (d_c≥1) | Unpooled $\\hat\\kappa$ + 95% CI | Hierarchical posterior mean + 95% CrI | shrinkage toward $\\mu_\\kappa$ |\n",
        "|---|---:|---|---|---:|\n",
    ]
    for f in fams:
        u = unpooled[f]; h = hier["per_family_posterior"][f]
        shrink = h["kappa_mean"] - u["kappa_hat"]
        md.append(
            f"| {f} | {u['n']} | "
            f"{u['kappa_hat']:.4f} [{u['ci_lo']:.4f},{u['ci_hi']:.4f}] | "
            f"{h['kappa_mean']:.4f} [{h['ci_lo']:.4f},{h['ci_hi']:.4f}] | "
            f"{shrink:+.4f} |\n"
        )
    md.append("\n## Interpretation\n\n")
    md.append(
        f"- The hierarchical model with 2 hyperparameters ({'wins' if bic_hier == min(bic_pool, bic_unpool, bic_hier) else 'does not win'} by BIC) "
        "is a sharper statement of the §4.3 family-specific framing: $\\kappa_{\\rm eff}$ varies across families with empirical-Bayes "
        f"posterior spread $\\hat\\sigma_\\kappa = {hier['sigma_kappa']:.4f}$, but each per-family posterior is shrunk toward the global mean.\n"
    )
    md.append(
        "- If $\\hat\\sigma_\\kappa$ is small relative to the per-family CI widths, the pilot supports a near-universal envelope; "
        "if $\\hat\\sigma_\\kappa$ is comparable to the within-family CI widths, the pilot supports family-specific envelopes (as §4.3 framed it).\n"
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("".join(md), encoding="utf-8")
    print(f"[wrote] {args.out_md}")


if __name__ == "__main__":
    main()
