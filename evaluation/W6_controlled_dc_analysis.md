# W6 Controlled d_c Analysis

Offline-only analysis using existing pilot258 judged outputs. No model/API calls were made.

## Panel

- Items: 258
- Observations: 1032 (4 model runs x pilot258)
- Models: Qwen14B, DeepSeekV3, KimiK2, Qwen7B-Ollama
- d_c counts: {0: 106, 1: 99, 2: 34, 3: 14, 4: 4, 5: 1}
- Source counts: {'OlympiadBench': 127, 'PHYBench': 100, 'PhysReason': 31}
- Topic count: 8

## Logistic control ladder

| Spec | Controls | d_c beta | OR per +1 d_c | AUC | AIC | BIC |
|---|---|---:|---:|---:|---:|---:|
| M0_dc_only | none | -0.5738 | 0.563 | 0.594 | 757.3 | 767.2 |
| M1_item_controls | item length + source + answer type | -0.3888 | 0.678 | 0.756 | 702.8 | 747.2 |
| M2_model_controls | M1 + model FE + token budget | -0.3307 | 0.718 | 0.815 | 658.2 | 727.3 |
| M3_topic_controls | M2 + topic FE | -0.3753 | 0.687 | 0.816 | 670.2 | 773.9 |

## Main controlled result

- Main spec: `M3_topic_controls`.
- d_c beta = -0.3753; odds ratio per +1 d_c = 0.687.
- Cluster bootstrap 95% CI for beta by item = [-0.8731, -0.0696] (B=200).
- Interpretation: negative beta means higher conservation-constraint load predicts lower judged correctness after local controls.

## Controlled marginal accuracy

| d_c | mean predicted accuracy (M3) |
|---:|---:|
| 0 | 0.151 |
| 1 | 0.114 |
| 2 | 0.085 |
| 3 | 0.062 |
| 4 | 0.045 |
| 5 | 0.032 |

## Robustness notes

- This is still a pilot panel, not the final 15 x 8 matrix.
- Correctness labels are produced by a Qwen14B judge, so judge-family bias is not eliminated.
- High-d_c bins remain sparse; controlled estimates for d_c>=4 are extrapolative.
- Topic fixed effects are ridge-regularized because pilot258 has many small topic cells.

## Generated artifacts

- `evaluation\w6_controlled_panel.csv`
- `evaluation\w6_logit_coefficients.csv`
- `evaluation\w6_observed_dc_bins.csv`
- `evaluation\w6_controlled_marginal_by_dc.csv`
- `figures\F3_w6_controlled_dc_effect.png`
