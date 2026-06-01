# T5 Corollary - Chain-of-Thought Gain And Conservation-constraint load

Status: W2 draft.

## Claim

If a Chain-of-Thought prompt mainly acts by multiplying effective token budget
`B_t` by a factor `k>1`, then T4 gives:

```tex
\Delta_{\mathrm{CoT}}
=
\bar A(kB_t)-\bar A(B_t)
\le
\frac{\beta}{e}\log k.
```

The stronger project claim, "CoT gain decreases as `1/d_c`", should be treated
as an empirical corollary or normalized-gain hypothesis, not as a completed
mathematical theorem from T1 alone.

## Conservative Version For Manuscript

Use:

> The envelope predicts bounded logarithmic returns from reasoning-token
> expansion. If conservation constraints divide the usable reasoning budget
> across `d_c` coupled checks, the per-constraint gain scales as
> `O(log k / d_c)`.

Avoid:

> We prove CoT cannot help high-dimensional physics tasks.

## Derivation

By T4, replacing `B_t` by `kB_t` gives:

```tex
\bar A(kB_t)-\bar A(B_t)
\le
\frac{\beta}{e}\log(kB_t/B_t)
=
\frac{\beta}{e}\log k.
```

If the reasoning trace must allocate effort across `d_c` conservation
constraints, a natural normalized per-constraint gain is:

```tex
\Delta_{\mathrm{CoT,per\ constraint}}
\le
\frac{\beta}{e}\frac{\log k}{d_c}.
```

This last step depends on the allocation model and must be tested empirically.

## Empirical Test

Run paired direct-answer and CoT prompts over matched items. Estimate:

```tex
\Delta_{ij} = correct^{CoT}_{ij} - correct^{direct}_{ij}
```

and fit:

```tex
\Delta_{ij}
= \theta_0 + \theta_1/d_{c,j} + controls + u_i + \epsilon_{ij}.
```

The claim is supported only if:

1. `theta_1 > 0` and stable under controls;
2. high-`d_c` items do not show comparable or larger CoT gains;
3. results are not explained by answer-format or prompt-length confounds.

## Downgrade Path

If CoT gains do not follow the `1/d_c` pattern, downgrade C5 to:

> CoT gains are bounded and heterogeneous; Conservation-constraint load explains part
> of the heterogeneity but is not a universal law.


