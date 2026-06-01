# Direct Test: Log-Odds Linear in d_c (constraint-penalty law)

The constraint-penalty law (A1) predicts that the log-odds of a correct answer are **linear in d_c**. We test this directly on the controlled M3 design (same controls as §4.4), leaving d_c terms unpenalised.

## 1. Curvature test (add d_c^2 to linear M3)

| model | k | logLik | BIC |
|---|---:|---:|---:|
| M3 linear in d_c | 21 | -314.10 | 773.92 |
| M3 + d_c^2 | 22 | -314.02 | 780.70 |

- d_c^2 coefficient = -0.0648 (SE 0.1435, z = -0.45, approx p = **0.652**).
- Likelihood-ratio (1 df): stat = 0.155, p = 0.693.
- BIC change from adding d_c^2: **+6.78** (favours linear).
- **Reading:** no detectable curvature; the linear-in-d_c (logistic) form is sufficient, as the constraint-penalty law predicts.

## 2. Free per-d_c effects vs the linear prediction (well-populated d_c $\le$ 3, n = 1012)

Entering d_c as free dummies (levels [1, 2, 3] vs baseline 0) in the M3 control set. The d_c=4 (n=4) and d_c=5 (n=1) cells have 0 observed accuracy (perfect separation) so their unpenalised dummy effect diverges; they are excluded from this dummy view (the continuous-d_c curvature test in §1 uses all data and is unaffected):

| d_c | free log-odds effect | 95% CI |
|---:|---:|---:|
| 1 | -0.291 | [-0.759, 0.178] |
| 2 | -0.840 | [-1.672, -0.008] |
| 3 | -0.872 | [-2.426, 0.682] |

- A line through the origin fits the free effects with **R^2 = 0.776** (implied per-constraint slope -0.328; cf. headline linear-M3 slope -0.375).
- Free-dummy vs linear LRT (2 df) on the same subset: stat = 0.251, p = 0.882 (free effects do NOT significantly improve fit -> linearity supported).
- BIC (d_c $\le$ 3 subset): linear 772.9 vs free-dummies 786.5 (linear preferred).

Figure `figures/F16_logodds_linearity.png` shows the free per-d_c effects with the constraint-penalty linear prediction overlaid.
