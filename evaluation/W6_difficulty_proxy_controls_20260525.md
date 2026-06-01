# W6 Difficulty-Proxy Control Analysis

Offline reviewer-risk check. This analysis adds text-derived difficulty proxies to the W6 control ladder and makes no model/API calls.

## Inputs

- W6 panel: `evaluation\w6_controlled_panel.csv`
- Difficulty proxies: `data\annotations\pilot258_difficulty_proxies.csv`
- Observations: 1032
- Items: 258
- Source distribution: {'OlympiadBench': 127, 'PHYBench': 100, 'PhysReason': 31}
- d_c distribution: {0: 106, 1: 99, 2: 34, 3: 14, 4: 4, 5: 1}

## Control Ladder

| Spec | Added controls | d_c beta | OR | bootstrap 95% CI | AUC | BIC |
|---|---|---:|---:|---|---:|---:|
| M3_original_topic_controls | original W6 M3 controls | -0.3753 | 0.687 | [-0.8308, -0.0165] | 0.816 | 773.9 |
| M4_plus_text_difficulty_proxies | M3 + number/equation/entity/latex-count proxies | -0.2874 | 0.750 | [-0.8074, 0.1079] | 0.821 | 808.3 |
| M5_plus_answer_proxy | M4 + answer-type proxy | -0.2882 | 0.750 | [-0.8103, 0.1085] | 0.821 | 815.2 |

## Main Reading

- Best BIC in this ladder: `M3_original_topic_controls`.
- Most conservative proxy spec (`M5_plus_answer_proxy`): d_c beta = -0.2882, OR = 0.750, bootstrap 95% CI = [-0.8103, 0.1085].
- Interpretation: the proxy-expanded CI crosses zero; keep only a weaker, exploratory claim until human-gold and expanded samples arrive.

## Controlled Marginal Accuracy (proxy-expanded spec)

| d_c | mean predicted accuracy |
|---:|---:|
| 0 | 0.144 |
| 1 | 0.116 |
| 2 | 0.093 |
| 3 | 0.074 |
| 4 | 0.058 |
| 5 | 0.045 |

## Boundary Notes

- These are reproducible proxies, not human difficulty labels.
- `solution_*` proxies use benchmark-provided solution text, which is valid for item difficulty characterization but not a solver input.
- Correctness labels remain LLM-judge labels until the human correctness packet returns.
- High-d_c bins remain sparse; this does not replace high-d_c expansion.

## Outputs

- `evaluation\w6_proxy_control_panel.csv`
- `evaluation\w6_proxy_control_coefficients_20260525.csv`
- `evaluation\w6_proxy_control_marginal_20260525.csv`
- `figures\F3b_w6_proxy_control_dc_beta.png`
