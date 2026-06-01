# W5 Stage 1 — Qwen-14B envelope fit
- Pilot: 258 items (subset of 258 where both graded & d_c-labeled present)
- Overall accuracy: 0.089

## Per-d_c breakdown

| d_c | n | accuracy |
|---|---:|---:|
| 0 | 106 | 0.123 |
| 1 | 99 | 0.091 |
| 2 | 34 | 0.029 |
| 3 | 14 | 0.000 |
| 4 | 4 | 0.000 |
| 5 | 1 | 0.000 |

## 1-parameter envelope fit

- Model:  $\bar A(d_c) = 1 - \exp(-\kappa_{eff}/d_c)$
- $\kappa_{eff}$ = 0.0905 (±95% CI 0.0503)
- $R^2$ on group means = -52.8445

## Verdict

- $R^2$ = -52.845 < 0.70 → single-variable fit weak. Stage 2 must check d_c is not confounded.
