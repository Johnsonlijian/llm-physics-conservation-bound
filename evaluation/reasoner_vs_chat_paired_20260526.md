# DeepSeek-reasoner vs DeepSeek-chat (paired, c4_budget_subset_50)

Direct paired comparison on the 50-item C4 subset, all judged by the same `deepseek-chat` judge route.

- **DeepSeek-reasoner** (`deepseek-reasoner`), `max_tokens = 16384`, 48/50 emit `FINAL_ANSWER` (`reasoner_subset50_offline_20260525.md`).
- **DeepSeek-chat at 2048 budget** (the maximum budget arm of the C4 sweep): 49/50 valid judged.
- **DeepSeek-chat at 128 budget** (the minimum budget arm): 50/50 valid judged.

## Per-d_c breakdown

| $d_c$ | n | chat@128 (acc) | chat@2048 (acc) | reasoner-16k (acc) | reasoner − chat@2048 |
|---:|---:|---|---|---|---:|
| 1 | 16 | 1/16 = 0.062 | 5/16 = 0.312 | 6/16 = 0.375 | +0.062 |
| 2 | 16 | 1/16 = 0.062 | 1/16 = 0.062 | 5/16 = 0.312 | +0.250 |
| 3 | 13 | 0/13 = 0.000 | 2/13 = 0.154 | 3/13 = 0.231 | +0.077 |
| 4 | 4 | 0/4 = 0.000 | 0/4 = 0.000 | 2/4 = 0.500 | +0.500 |
| 5 | 1 | 0/1 = 0.000 | 0/1 = 0.000 | 0/1 = 0.000 | +0.000 |
| **all** | **50** | **2/50 = 0.040** | **8/50 = 0.160** | **16/50 = 0.320** | **+0.160** |

## Paired sign-test

- Reasoner vs chat@2048: gains = 10, losses = 2, ties = 38, two-sided $p = 0.0386$
- Reasoner vs chat@128:  gains = 14, losses = 0, ties = 36, two-sided $p = 0.0001$

## Reading

The DeepSeek-reasoner arm on this subset reaches 0.320 judged accuracy vs 0.160 for DeepSeek-chat at the same nominal max budget of 2048, and vs 0.040 at 128. The reasoner arm uses up to 16384 tokens of *hidden* reasoning_content before emitting its final answer (median 13074 tokens; cf. `reasoner_subset50_offline_20260525.md`), so this is *not* a clean C4 controlled-budget arm. It is a controlled solver-architecture comparison at *nominal* max budget. The accuracy gain is consistent with the C4 envelope upper bound at the implied token ratio: 8× more tokens × $\beta/e\approx 0.11 \times \ln 8 \approx 0.23$, vs observed $+0.160$.
