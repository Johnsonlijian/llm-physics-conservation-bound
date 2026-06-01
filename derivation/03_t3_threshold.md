# T3 Corollary - Physics-Corpus Coverage Threshold

Status: W2 draft.

## Claim

For any target accuracy level `a` in `(0,1)`, the T1 envelope implies a
minimum physics-coverage proxy:

```tex
\rho^*(a)
=
\left[
\frac{-d_c\log(1-a)}
{\kappa B_t^\beta L_c^\gamma}
\right]^{1/\alpha}.
```

If `rho_p < rho^*(a)`, then the envelope cannot reach target accuracy `a`:

```tex
\bar A < a.
```

## Derivation

Set the envelope equal to the target `a`:

```tex
a =
1-\exp\left(
-\frac{\kappa\rho^\alpha B_t^\beta L_c^\gamma}{d_c}
\right).
```

Then

```tex
\exp\left(
-\frac{\kappa\rho^\alpha B_t^\beta L_c^\gamma}{d_c}
\right)
=1-a.
```

Taking logs:

```tex
\frac{\kappa\rho^\alpha B_t^\beta L_c^\gamma}{d_c}
= -\log(1-a).
```

Solving for `rho` gives:

```tex
\rho^*(a)
=
\left[
\frac{-d_c\log(1-a)}
{\kappa B_t^\beta L_c^\gamma}
\right]^{1/\alpha}.
```

## Random-Guess Baseline

For a multiple-choice task with `|Y|` answer classes and tolerance `epsilon`,
use

```tex
a = 1/|Y| + \epsilon
```

as the low-performance threshold. For numerical-answer tasks this baseline is
not well-defined without a scoring tolerance and answer distribution, so the
paper should avoid a universal "random guess" statement across all benchmarks.

## Evidence Boundary

The project brief currently contains rough values such as `rho_p ~ 10^{-4}` and
`rho^* ~ 10^{-2}`. These remain **verification slots**, not established facts.
W3/W4 must either:

1. estimate `rho_p` from documented corpus/source metadata, or
2. treat `rho_p` as a latent fitted covariate and avoid claims about Common
   Crawl-level physical sample shares.

