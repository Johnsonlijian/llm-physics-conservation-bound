# T2 Corollary - High Conservation-constraint load

Status: W2 draft.

## Claim

Under the T1 working envelope and finite effective resources,

```tex
\lim_{d_c\to\infty} \bar A(d_c,L_c,B_t,\rho_p)=0.
```

This supports the qualitative claim that high-dimensional conservation tasks
become increasingly difficult for LLMs when context, inference budget, and
physics pretraining coverage are held fixed.

## Derivation

Let

```tex
C = \kappa \rho_p^\alpha B_t^\beta L_c^\gamma,
```

where `C` is finite and positive. T1 gives

```tex
\bar A(d_c)=1-\exp(-C/d_c).
```

As `d_c -> infinity`, `C/d_c -> 0`. Since

```tex
1-\exp(-x)=x+O(x^2)
```

near `x=0`,

```tex
\bar A(d_c)
= \frac{C}{d_c}+O(d_c^{-2})
->0.
```

## Interpretation

The result is not a claim that all hard physics questions have large `d_c`.
It only says that, inside this model, increasing Conservation-constraint load is an
accuracy-limiting axis. W4 must distinguish `d_c` from confounders such as
reading difficulty, algebra length, topic familiarity, and answer format.

## Empirical Test

For each benchmark and model family, fit:

```tex
\mathrm{logit}(A_{ij}) =
\eta_0 + \eta_1 \log d_{c,j} + controls + u_i + \epsilon_{ij}.
```

Expected sign: `eta_1 < 0`.

Controls should include at minimum prompt length, answer type, topic, benchmark,
and model size/tier. A raw negative correlation alone is not enough for a
paper-grade claim.


