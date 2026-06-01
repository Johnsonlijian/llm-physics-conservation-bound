# Hierarchical Envelope Fit

Three nested specifications of the one-parameter envelope $A(d_c) = 1-\exp(-\kappa/d_c)$ on items with $d_c \geq 1$. Binomial likelihood on per-item binary outcomes. No PyMC dependency — pure scipy.

- Panel: `evaluation\w6_controlled_panel.csv` (4 families × 258 items = 1032 obs; 608 with d_c≥1)
- Families: ['DeepSeekV3', 'KimiK2', 'Qwen14B', 'Qwen7B-Ollama']

## Headline

| Spec | Free params | Total NLL | BIC |
|---|---:|---:|---:|
| Pooled (single $\kappa$ across families) | 1 | 181.90 | 370.21 |
| Unpooled (independent $\kappa$ per family, §4.3) | 4 | 176.20 | 378.04 |
| Hierarchical ($\kappa_m \sim N(\mu_\kappa, \sigma_\kappa^2)$, marg.) | 2 | 180.41 | 373.63 |

Best BIC: pooled

Hierarchical hyperparameters: $\hat\mu_\kappa = 0.1219$, $\hat\sigma_\kappa = 0.0418$. The small $\hat\sigma_\kappa$ measures how much $\kappa$ moves across families: a tighter spread means the four families are more consistent with a near-universal envelope, a wider spread that they need family-specific fits.

## Per-family κ comparison

| Family | n (d_c≥1) | Unpooled $\hat\kappa$ + 95% CI | Hierarchical posterior mean + 95% CrI | shrinkage toward $\mu_\kappa$ |
|---|---:|---|---|---:|
| DeepSeekV3 | 152 | 0.2165 [0.1410,0.3153] | 0.1744 [0.1209,0.2259] | -0.0421 |
| KimiK2 | 152 | 0.1211 [0.0682,0.1961] | 0.1248 [0.0760,0.1809] | +0.0037 |
| Qwen14B | 152 | 0.0854 [0.0428,0.1499] | 0.1013 [0.0610,0.1509] | +0.0158 |
| Qwen7B-Ollama | 152 | 0.0678 [0.0310,0.1261] | 0.0880 [0.0460,0.1359] | +0.0202 |

## Interpretation

- The hierarchical model with 2 hyperparameters (does not win by BIC) is a sharper statement of the §4.3 family-specific framing: $\kappa_{\rm eff}$ varies across families with empirical-Bayes posterior spread $\hat\sigma_\kappa = 0.0418$, but each per-family posterior is shrunk toward the global mean.
- If $\hat\sigma_\kappa$ is small relative to the per-family CI widths, the pilot supports a near-universal envelope; if $\hat\sigma_\kappa$ is comparable to the within-family CI widths, the pilot supports family-specific envelopes (as §4.3 framed it).
