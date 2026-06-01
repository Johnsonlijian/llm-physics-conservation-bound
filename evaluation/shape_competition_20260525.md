# Shape Competition for d_c Accuracy Curves

Offline comparison on `evaluation/w6_controlled_panel.csv`. This is a diagnostic gate, not a universal-law proof.

| family | best AIC | best BIC | envelope delta BIC |
|---|---|---|---:|
| pooled | logistic | logistic | 7.86 |
| DeepSeekV3 | logistic | logistic | 5.86 |
| KimiK2 | logistic | logistic | 5.37 |
| Qwen14B | logistic | logistic | 6.03 |
| Qwen7B-Ollama | logistic | logistic | 5.94 |

## Interpretation rule

- `envelope_dplus_delta` must be competitive with simpler alternatives before any high-journal claim about envelope shape is promoted.
- If logistic or monotone spline wins clearly, the manuscript should keep the envelope as an explanatory ansatz and lead with `d_c` as an independent predictor.
- High-d_c bins remain sparse in pilot258, so this comparison is underpowered at the right tail.
