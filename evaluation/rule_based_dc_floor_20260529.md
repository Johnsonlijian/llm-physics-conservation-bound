# Rule-Based Deterministic d_c Floor (Zero-LLM Construct-Validity Leg)

A transparent regex extractor maps physics text to conservation-law families using only explicit lexical signals (for example, `collision` -> momentum, `conservation of energy` -> energy, `angular momentum` -> angular momentum). It performs no model inference and therefore cannot share an LLM prior. Because it fires only on explicit keywords, it is a conservative lower bound on d_c. The full rule table is in `code/scripts/31_rule_based_dc_floor.py`.

- PhysReason problems scored: 1200 (all problems; no LLM labels required).
- `rule_q`: extractor on the question statement only (hardest test).
- `rule_qs`: extractor on statement + official solution (matches the human annotator's information set).

## Agreement with human benchmark-author theorem labels

| comparison | Spearman rho | within +/-1 | exact | mean diff | Pearson r |
|---|---:|---:|---:|---:|---:|
| rule_q (question only) vs human | 0.362 | 0.971 | 0.705 | -0.248 | 0.446 |
| rule_qs (q+solution) vs human | 0.794 | 0.998 | 0.853 | -0.018 | 0.806 |

Lower-bound property holds for 96.2% of items (rule_q <= human) and 93.5% (rule_qs <= human), confirming the extractor behaves as a floor.

## Per-family presence agreement, rule_qs vs human

| family | Cohen kappa | rule-positive | human-positive |
|---|---:|---:|---:|
| momentum | 0.590 | 168 | 109 |
| angular_momentum | 0.888 | 10 | 8 |
| energy | 0.842 | 260 | 322 |
| charge | 0.000 | 0 | 16 |
| mass | 0.249 | 1 | 7 |
| entropy | 0.800 | 3 | 2 |

## Three-way construct-validity subset

| comparison | Spearman rho | within +/-1 | mean diff |
|---|---:|---:|---:|
| rule_q (question only) vs LLM | 0.078 | 0.753 | -0.852 |
| rule_qs (q+solution) vs LLM | 0.322 | 0.840 | -0.543 |
| human theorem vs LLM | 0.542 | 0.877 | -0.531 |

**Reading.** The two channels that share the human annotator's information set, the benchmark authors' theorem labels and the zero-LLM regex on question+solution, agree strongly on the full n=1200 sample (Spearman 0.79, within +/-1 = 99.8%, energy Cohen kappa 0.84, near-zero bias) with a confirmed lower-bound property. On the 81-item three-way subset, the LLM-consensus d_c agrees with the human labels at rho = 0.54 and with the conservative regex floor at rho = 0.32 (both positive). The question-only regex is much weaker (rho = 0.08 vs LLM), indicating that for many items the required conservation laws are not lexically explicit in the problem statement and only become visible on the solution path, a property of d_c we flag rather than hide. The convergence of a transparent rule system, the benchmark authors' human labels and the LLM rater pool on the same conservation structure is evidence that d_c is an objective item property recoverable without model inference, not a shared LLM artefact.

## Known-groups validity: human difficulty vs mean d_c

| human difficulty | n | mean rule_qs d_c | mean human d_c |
|---|---:|---:|---:|
| knowledge | 300 | 0.22 | 0.26 |
| easy | 300 | 0.19 | 0.21 |
| medium | 300 | 0.40 | 0.40 |
| difficult | 300 | 0.66 | 0.67 |

If d_c rises monotonically with the benchmark authors' independent difficulty rating, that is convergent evidence; if the rise is only partial, that is discriminant evidence that d_c is not a restatement of global difficulty.
