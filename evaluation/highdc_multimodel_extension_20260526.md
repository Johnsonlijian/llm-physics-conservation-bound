# High-d_c +100 Multi-Model Extension

- Item-level CSV: `evaluation/highdc_multimodel_items_20260526.csv`
- Summary CSV: `evaluation/highdc_multimodel_summary_20260526.csv`
- Figure: `figures/F9_highdc_multimodel_extension_20260526.png`
- Scope: 100 newly mined high-d_c candidates, using DeepSeek single-rater prelabels for d_c. This is an extension diagnostic, not a human-gold replacement.

## Overall by solver

| solver | raw n | valid n | correct n | accuracy | solve errors | judge errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 100 | 100 | 31 | 0.310 | 0 | 0 |
| Kimi-K2 | 100 | 100 | 28 | 0.280 | 0 | 0 |
| Qwen-14B | 100 | 100 | 15 | 0.150 | 0 | 0 |

## Accuracy by d_c

| solver | d_c | raw n | valid n | correct n | accuracy | judge errors |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 0 | 17 | 17 | 5 | 0.294 | 0 |
| DeepSeek | 1 | 53 | 53 | 15 | 0.283 | 0 |
| DeepSeek | 2 | 26 | 26 | 11 | 0.423 | 0 |
| DeepSeek | 3 | 2 | 2 | 0 | 0.000 | 0 |
| DeepSeek | 4 | 2 | 2 | 0 | 0.000 | 0 |
| Kimi-K2 | 0 | 17 | 17 | 7 | 0.412 | 0 |
| Kimi-K2 | 1 | 53 | 53 | 15 | 0.283 | 0 |
| Kimi-K2 | 2 | 26 | 26 | 6 | 0.231 | 0 |
| Kimi-K2 | 3 | 2 | 2 | 0 | 0.000 | 0 |
| Kimi-K2 | 4 | 2 | 2 | 0 | 0.000 | 0 |
| Qwen-14B | 0 | 17 | 17 | 4 | 0.235 | 0 |
| Qwen-14B | 1 | 53 | 53 | 11 | 0.208 | 0 |
| Qwen-14B | 2 | 26 | 26 | 0 | 0.000 | 0 |
| Qwen-14B | 3 | 2 | 2 | 0 | 0.000 | 0 |
| Qwen-14B | 4 | 2 | 2 | 0 | 0.000 | 0 |

## Interpretation

- The extension is useful for stress-testing transfer beyond the original pilot258, but it also demonstrates within-d_c item heterogeneity: newly mined items can be easier than the original pilot at the same nominal d_c.
- All three high-d_c solver arms in this report are judged with the same `deepseek-v4-flash` route. This removes one judge-route confound from the extension analysis, but it still does not replace human correctness adjudication.
- High d_c >= 4 remains underpopulated. The human-gold packet and additional oversampling are still required before upgrading the central claim.
