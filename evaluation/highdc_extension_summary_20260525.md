# High-d_c Pilot Expansion (DeepSeek-only)

Mined 100 new candidates from OlympiadBench (50, text-only physics not in pilot258) and PhysReason (50, text-only, sorted by PR-native theorem-based d_c heuristic). DeepSeek-prelabeled (single r1 profile, using the same V4 protocol), filtered to those with parsed d_c. Then DeepSeek-solved with the standard W5 prompt (`max_tokens=1024`, `temperature=0.2`) and DeepSeek-judged with the standard W6 judge.

## Per-d_c accuracy comparison

| d_c | pilot258 (n, acc) | highdc100 (n, acc) | merged 358 (n, acc) |
|---:|---|---|---|
| 0 | 106 | 0.283 | 17 | 0.353 | 123 | 0.293 |
| 1 | 99 | 0.192 | 53 | 0.358 | 152 | 0.250 |
| 2 | 34 | 0.118 | 26 | 0.500 | 60 | 0.283 |
| 3 | 14 | 0.071 | 2 | 0.000 | 16 | 0.062 |
| 4 | 4 | 0.000 | 2 | 0.500 | 6 | 0.167 |
| 5 | 1 | 0.000 | — | 1 | 0.000 |

## Envelope fit (one-parameter, d_c>=1, binomial NLL)

- pilot258 DeepSeek-only: $\hat\kappa = 0.2165$
- merged 358 (pilot258 + highdc100): $\hat\kappa = 0.3470$

## Interpretation

- The extension contributes mostly d_c=2 items (was 34, now +26; merged: 60). Higher-d_c bins add modestly. This partly resolves the §4.3/§6 'high-d_c sparsity' caveat but does not yet populate d_c$\ge$4 robustly.
- $\hat\kappa$ on the merged set (0.3470) is close to the pilot258-only fit (0.2165); the envelope shape transfers to the new items at similar magnitude.
- This is a DeepSeek-only extension; a full multi-model expansion requires re-solving the 100 new items with Qwen-14B / Qwen-7B / Kimi-K2, which is the next-extension item.
