# C5 Paired Direct-vs-CoT Analysis

Paired pilot using existing DeepSeek solves and the same DeepSeek judge for direct and CoT-allowed outputs.

## Inputs

- Direct judged: `data\results\solve_pilot258_deepseek_direct_judged_by_deepseek.csv`
- CoT-allowed judged: `data\results\solve_pilot258_deepseek_cot_judged_by_deepseek.csv`
- Consensus conservation-constraint load d_c: `data\annotations\pilot258_v4a_qwen14b_merged.csv`
- Valid paired items: 249; excluded judge-error/unparseable items: 9

## Overall paired result

- Direct accuracy: 0.145
- CoT-allowed accuracy: 0.281
- Paired delta (CoT - direct): +0.137
- Gain items: 47; loss items: 13; sign-test p=1.215e-05

## By d_c

| d_c | n | direct acc | CoT acc | delta | gains | losses | sign-test p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 102 | 0.225 | 0.412 | +0.186 | 28 | 9 | 0.00256 |
| 1 | 97 | 0.124 | 0.237 | +0.113 | 14 | 3 | 0.0127 |
| 2 | 33 | 0.030 | 0.152 | +0.121 | 5 | 1 | 0.219 |
| 3 | 13 | 0.000 | 0.000 | +0.000 | 0 | 0 | NA |
| 4 | 4 | 0.000 | 0.000 | +0.000 | 0 | 0 | NA |

## C5 1/d_c pilot regression

- Fit on d_c>=1 items: n=147, delta = 0.037 + 0.080*(1/d_c).
- Slope p=0.4914, R2=0.003; bootstrap 95% CI for slope=[-0.128, 0.279] (B=1000).
- Positive slope would support the stronger per-constraint hypothesis that CoT gains concentrate at low d_c. Treat this as pilot evidence only because d_c>=4 bins are sparse and both conditions use the same provider family.

## Interpretation

- CoT-allowed prompting improves overall accuracy over strict direct prompting on this paired pilot.
- The direction of the 1/d_c slope is consistent with C5's normalized-gain hypothesis, but this is not yet paper-grade.
- C5 should remain a hypothesis / exploratory subsection until replicated with a controlled reasoning-budget model and balanced high-d_c bins.

## Generated artifacts

- `evaluation\c5_paired_cot_items.csv`
- `evaluation\c5_paired_cot_by_dc.csv`
- `figures\F7_c5_paired_cot_delta.png`
