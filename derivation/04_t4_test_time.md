# T4 Corollary - Test-Time Compute Diminishing Returns

Status: W2 draft.

## Claim

Under T1, increasing effective test-time token budget from `B_0` to `B_t`
has logarithmically bounded gain:

```tex
\bar A(B_t)-\bar A(B_0)
\le
\frac{\beta}{e}\log(B_t/B_0),
\quad B_t\ge B_0>0.
```

This provides a cleaner global version of the earlier `O(beta log(B_t/B_0))`
statement.

## Derivation

Let

```tex
z(B)=
\frac{\kappa\rho_p^\alpha L_c^\gamma}{d_c}B^\beta
= cB^\beta,
```

where `c>0`. Then

```tex
\bar A(B)=1-\exp(-z(B)).
```

Differentiate with respect to `log B`:

```tex
\frac{\partial \bar A}{\partial \log B}
=
\frac{\partial \bar A}{\partial B}B
=
\beta z e^{-z}.
```

For `z>0`, the function `z e^{-z}` is maximized at `z=1` and has maximum
`1/e`. Therefore

```tex
\frac{\partial \bar A}{\partial \log B}
\le
\frac{\beta}{e}.
```

Integrating from `log B_0` to `log B_t` gives:

```tex
\bar A(B_t)-\bar A(B_0)
\le
\frac{\beta}{e}\log(B_t/B_0).
```

## Interpretation

This does not say test-time compute is useless. It says the envelope has
bounded marginal gain on the log-budget axis. The strongest empirical test is
not a single model comparison, but a within-model budget sweep where the same
questions are run with different maximum reasoning-token budgets.

## W5/W6 Measurement Plan

For each selected model and benchmark subset:

- run direct answer, short CoT, and long CoT prompts;
- record prompt tokens, completion tokens, latency, and correctness;
- fit accuracy against `log(B_t/B_0)` within each `d_c` bin;
- test whether high-`d_c` bins have lower marginal gains.

