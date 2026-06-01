# C4 Controlled Budget Sweep

- Input judged CSV: `data\results\c4_budget_sweep_deepseek_chat_judged_20260525.csv`
- Valid judged rows: 246
- Budgets: 128, 256, 512, 1024, 2048

## Accuracy by budget

| max_tokens | n | accuracy |
|---:|---:|---:|
| 128 | 50 | 0.040 |
| 256 | 49 | 0.041 |
| 512 | 49 | 0.082 |
| 1024 | 49 | 0.122 |
| 2048 | 49 | 0.163 |

## Log-budget slope

- Fit: accuracy = a + b * log(max_tokens / 128).
- b = 0.0473; 95% normal CI = [0.0343, 0.0604]; R2 = 0.944; p = 0.00574.
- Reference C4 upper-bound slope is beta/e ~= 0.11 under the Chinchilla beta ~= 0.3 anchor.

## Paired deltas vs minimum budget

| max_tokens | paired n | mean delta | gains | losses | sign-test p |
|---:|---:|---:|---:|---:|---:|
| 256 | 49 | +0.000 | 1 | 1 | 1 |
| 512 | 49 | +0.041 | 3 | 1 | 0.625 |
| 1024 | 49 | +0.082 | 5 | 1 | 0.219 |
| 2048 | 49 | +0.122 | 7 | 1 | 0.0703 |

## Interpretation rule

- Treat this as C4 evidence only if the same item subset, same model snapshot, and same judge route are used across all budgets.
- If judge errors exceed 2% or the provider alias moves during the run, downgrade this sweep to a pilot diagnostic.
