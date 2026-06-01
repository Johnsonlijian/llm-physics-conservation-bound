# W6 Held-Out Cross-Validation

Addresses §4.4 caveat that the original W6 AUC = 0.816 is in-sample. We refit the M3 specification (ridge logistic with topic + source + answer_type + model FE + token budget + length controls) under three out-of-sample splits, all grouped at the *item* level so the four per-item model observations are always in the same fold.

- Panel: `evaluation\w6_controlled_panel_with_logs.csv`, rows = 1032
- Items: 258
- Design features: 17
- In-sample baseline: AUC = 0.8117, d_c beta = -0.3852

## Held-out AUC

| Split | n_folds | Hold-out AUC | d_c beta mean | d_c beta sd |
|---|---:|---:|---:|---:|
| LIO | 258 | 0.7731 | -0.3848 | 0.0118 |
| GKF5 | 5 | 0.7753 | -0.3813 | 0.0787 |
| GKF10 | 10 | 0.7704 | -0.3828 | 0.0483 |

## Interpretation

- Leave-item-out gives the most stringent test: AUC = 0.7731, compared with the in-sample 0.8117. The drop measures over-optimism in the §4.4 table; a small drop (<0.02) supports the in-sample number; a large drop indicates the model is over-fitting the topic + answer-type cells.
- The d_c coefficient across LIO folds has mean -0.3848 (sd 0.0118, range [-0.4420, -0.3512]). Tight spread + sign-stability (none above 0) supports the direction of the effect even when each item in turn is dropped from training.
- These are reproducible from `code/scripts/20_w6_holdout_cv.py` over the `evaluation/w6_controlled_panel.csv` panel.
