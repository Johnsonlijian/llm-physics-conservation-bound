# C4 Controlled Budget Sweep

- Input judged CSV: `data\results\c4_budget_sweep_doubao15pro_judged_20260526.csv`
- Valid judged rows: 246
- Budgets: 128, 256, 512, 1024, 2048

## Accuracy by budget

| max_tokens | n | accuracy |
|---:|---:|---:|
| 128 | 50 | 0.020 |
| 256 | 50 | 0.000 |
| 512 | 49 | 0.000 |
| 1024 | 49 | 0.102 |
| 2048 | 48 | 0.125 |

## Log-budget slope

- Fit: accuracy = a + b * log(max_tokens / 128).
- b = 0.0450; 95% normal CI = [0.0104, 0.0796]; R2 = 0.684; p = 0.084.
- Reference C4 upper-bound slope is beta/e ~= 0.11 under the Chinchilla beta ~= 0.3 anchor.

## Paired deltas vs minimum budget

| max_tokens | paired n | mean delta | gains | losses | sign-test p |
|---:|---:|---:|---:|---:|---:|
| 256 | 50 | -0.020 | 0 | 1 | 1 |
| 512 | 49 | -0.020 | 0 | 1 | 1 |
| 1024 | 49 | +0.082 | 5 | 1 | 0.219 |
| 2048 | 48 | +0.104 | 6 | 1 | 0.125 |

## Interpretation rule

- Treat this as C4 evidence only if the same item subset, same model snapshot, and same judge route are used across all budgets.
- If judge errors exceed 2% or the provider alias moves during the run, report this sweep as a diagnostic rather than as controlled C4 evidence.
