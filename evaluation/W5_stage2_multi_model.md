# W5 Stage 2 — Multi-Model Envelope
- Pilot: 258 items (d_c=0 EXCLUDED for fit)
- d_c source: Qwen-14B median of 4 raters (V4a alpha=0.9020)
- Judge: Qwen-14B vLLM as LLM judge for answer correctness (judge-family bias is possible)

## Per-model results

| Model | params | overall acc | κ_eff | R²(group) |
|---|---|---|---|---|
| Qwen14B | ~14B | 0.089 | 0.090±0.050 | 0.688 |
| DeepSeekV3 | ~671B | 0.209 | 0.216±0.083 | 0.821 |
| KimiK2 | ~1000B | 0.143 | 0.119±0.061 | 0.733 |
| Qwen7B-Ollama | ~7B | 0.058 | 0.071±0.045 | 0.719 |

## Per-d_c × model accuracy

| d_c | n_max | Qwen14B | DeepSeekV3 | KimiK2 | Qwen7B-Ollama |
|---|---|---|---|---|---|
| 1 | 99 | 0.091 | 0.192 | 0.111 | 0.071 |
| 2 | 34 | 0.029 | 0.118 | 0.059 | 0.029 |
| 3 | 14 | 0.000 | 0.071 | 0.071 | 0.000 |
| 4 | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

## Key Findings

- Low-d_c bins are consistently easier than high-d_c bins across the available judged runs; all models score 0 on the sparse d_c>=4 bins in this pilot.
- DeepSeekV3 has the highest overall accuracy in this local set; parameter count/provider tier is not yet a clean monotonic predictor.
- The one-parameter envelope is qualitatively plausible for several models (R2_group around 0.72-0.82), but Qwen14B remains below the 0.70 gate and no universal one-parameter law is established.
- Because high-d_c bins are small (d_c=4: n=4; d_c=5: n=1), W6 must rebalance or bootstrap before making a strong curve-shape claim.

## Next (W6)

- Add `L_c`, `B_t` (actual tokens_used), `rho_p` (physics-corpus proxy) for full 4-param envelope
- Bootstrap 95% CI per kappa (5880 96-core when `.wslconfig` upgraded)
- Family-specific κ comparison (AIC/BIC) for C1 architecture-independence test
