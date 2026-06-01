# DeepSeek-reasoner on the +100 high-d_c extension (right-tail reinforcement)

Reasoner arm added to the §4.8 extension using the identical solve/judge route (deepseek-reasoner@16384 -> deepseek-chat judge) and the same extension DeepSeek-prelabel d_c as the other three solvers.

## §4.8 extension, overall accuracy (4 solvers)

| solver | valid n | accuracy |
|---|---:|---:|
| DeepSeek-chat | 100 | 0.310 |
| Kimi-K2 | 100 | 0.280 |
| Qwen-14B | 100 | 0.150 |
| DeepSeek-reasoner | 97 | 0.546 |

## Extension accuracy by d_c (4 solvers)

| solver | d_c=0 | d_c=1 | d_c=2 | d_c=3 | d_c=4 |
|---|---:|---:|---:|---:|---:|
| DeepSeek-chat | 0.29 (n=17) | 0.28 (n=53) | 0.42 (n=26) | 0.00 (n=2) | 0.00 (n=2) |
| Kimi-K2 | 0.41 (n=17) | 0.28 (n=53) | 0.23 (n=26) | 0.00 (n=2) | 0.00 (n=2) |
| Qwen-14B | 0.24 (n=17) | 0.21 (n=53) | 0.00 (n=26) | 0.00 (n=2) | 0.00 (n=2) |
| DeepSeek-reasoner | 0.56 (n=16) | 0.49 (n=51) | 0.65 (n=26) | 0.50 (n=2) | 0.50 (n=2) |

## Pooled reasoner right tail (pilot258 + extension)

- pilot258 reasoner: n=254, univariate OR/$+1 d_c$ = 0.683 (p=0.0081)
- extension reasoner: n=97, univariate OR/$+1 d_c$ = 1.141 (p=0.6)
- **pooled (n=351): univariate OR/$+1 d_c$ = 0.811 (β=-0.209, p=0.074)**

| d_c | pilot n | pilot acc | ext n | ext acc | pooled n | pooled acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 104 | 0.51 | 16 | 0.56 | 120 | 0.52 |
| 1 | 98 | 0.40 | 51 | 0.49 | 149 | 0.43 |
| 2 | 33 | 0.24 | 26 | 0.65 | 59 | 0.42 |
| 3 | 14 | 0.36 | 2 | 0.50 | 16 | 0.38 |
| 4 | 4 | 0.25 | 2 | 0.50 | 6 | 0.33 |
| 5 | 1 | 0.00 | 0 | — | 1 | 0.00 |

## Caveat

The pilot and extension halves use different d_c label sources (pilot = 4-family LLM consensus; extension = single-rater DeepSeek prelabel under the same V4 protocol), so the pooled curve is a coverage/robustness extension that widens d_c support on a strong solver, not a single clean-labelled panel. The extension also remains light at d_c>=3. Figure F18.
