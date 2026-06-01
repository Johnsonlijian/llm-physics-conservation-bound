# T1 Main Bound - Working Derivation v0.1

Status: W2 draft. This is a defensible working theorem, not yet a final
journal theorem. It separates algebraic consequences from assumptions that
must be justified or empirically stress-tested in W3-W6.

## Claim

For a conservation-reasoning task with Conservation-constraint load `d_c`, effective
context length `L_c`, effective inference-token budget `B_t`, and pretraining
physics-sample share `rho_p`, we use the following capacity-limited working
upper envelope for model accuracy:

```tex
\bar A(d_c,L_c,B_t,\rho_p)
= 1-\exp\left(
-\frac{\kappa \rho_p^\alpha B_t^\beta L_c^\gamma}{d_c}
\right),
```

with positive parameters `kappa, alpha, beta, gamma`.

This should be described as a **closed-form capacity envelope** or
**fitted upper envelope**, unless W3-W6 provide enough evidence to defend the
stronger phrase "information-theoretic upper bound".

## Objects

Let each item require a set of conservation constraints
`C_1,...,C_{d_c}`. Let `S` denote the latent state needed to apply these
constraints, `X` the prompt/context, and `Y` the model answer. The model is
treated as a noisy channel from task information to an answer.

We do not claim that every Transformer literally implements independent
constraint channels. The independence approximation is a modeling reduction
whose residual must be tested by grouped residuals over benchmark, model family,
and conservation-law type.

## Assumptions

**A1 Task bottleneck.** Correctness requires enough recoverable task information
to satisfy `d_c` conservation constraints. Higher `d_c` increases the number of
constraint checks or coupled state variables the model must recover.

**A2 Effective information budget.** The recoverable physics-relevant
information available at test time is bounded by an effective scalar

```tex
I_{\mathrm{eff}}
\le
\kappa \rho_p^\alpha B_t^\beta L_c^\gamma,
```

where:

- `rho_p` is a coarse proxy for physics-relevant pretraining coverage.
- `B_t` captures inference-time budget, including generated reasoning tokens.
- `L_c` captures usable context, not merely nominal context window length.
- `kappa` absorbs architecture, tokenizer, data quality, and task-family
  constants.

**A3 Conservation-dimension allocation.** The per-constraint effective
information scale decays approximately as `I_eff / d_c`. This is the most
important modeling assumption. It must be checked through residuals against
`d_c` and through sensitivity analyses where `d_c` is replaced by alternative
difficulty measures.

**A4 Saturating success response.** Accuracy follows a saturating response to
effective information per Conservation-constraint load:

```tex
A \le 1-\exp(-I_{\mathrm{eff}}/d_c).
```

This is a hazard-style envelope: once enough relevant information is available,
additional information gives diminishing returns. It is not a direct algebraic
consequence of Fano's inequality. Fano should be cited only as motivation for
linking recoverable information and error probability.

## Derivation

From A2 and A3, define the dimension-normalized information score

```tex
z =
\frac{\kappa \rho_p^\alpha B_t^\beta L_c^\gamma}{d_c}.
```

From A4:

```tex
A \le 1-\exp(-z).
```

Substituting `z` gives the working envelope

```tex
\bar A
=1-\exp\left(
-\frac{\kappa \rho_p^\alpha B_t^\beta L_c^\gamma}{d_c}
\right).
```

The envelope is intentionally minimal: it enforces saturation, monotonicity in
physics coverage, inference budget, and context, and monotonic decrease in
Conservation-constraint load.

## Algebraic Properties To Audit

For positive `kappa, alpha, beta, gamma, rho_p, B_t, L_c, d_c`:

1. `0 < \bar A < 1`.
2. `\partial \bar A / \partial rho_p > 0`.
3. `\partial \bar A / \partial B_t > 0`.
4. `\partial \bar A / \partial L_c > 0`.
5. `\partial \bar A / \partial d_c < 0`.
6. `\bar A -> 0` as `d_c -> infinity`.
7. `\bar A -> 0` as `rho_p -> 0`, `B_t -> 0`, or `L_c -> 0`.
8. `\bar A -> 1` as `B_t -> infinity`, if all other quantities stay positive.

These are audited in `code/derivation/audit_t1.py`.

## Manuscript Wording Guardrail

Use:

> We introduce a closed-form capacity envelope that is consistent with an
> information-bottleneck view of conservation reasoning.

Avoid until proven:

> We prove that no LLM can exceed this exact bound.

The stronger language is desk-risky unless W6 shows stable fit, residuals, and
robustness across model families.

## Open Proof Gaps

| Gap | Risk | W2/W3 action |
|---|---|---|
| Turning Fano-style error bounds into the exponential envelope | high | frame as motivation, not a completed proof |
| Defining `rho_p` from public corpus metadata | high | keep as measured/estimated covariate with interval |
| Treating conservation constraints as dimension-normalized independent load | high | add residual and ablation tests against alternative difficulty metrics |
| Architecture-independence | medium | test separate `kappa` per model family and compare AIC/BIC |


