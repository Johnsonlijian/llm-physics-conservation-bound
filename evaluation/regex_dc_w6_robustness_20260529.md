# W6 Robustness Under a Zero-LLM (Regex) d_c Label

The headline W6 result uses an LLM-consensus conservation-constraint load d_c, inviting the objection that the negative d_c effect is an artefact of LLM labelling. Here we re-run the identical M3 ridge logistic on the identical 1032-observation panel, swapping only the source of the d_c variable from LLM consensus to a fully deterministic regex extractor (`rule_qs` mode of `code/scripts/31_rule_based_dc_floor.py`, applied to question + official solution). Same estimator (`code/scripts/07_*`), same controls (topic/source/model/answer-type/token/length FE), same items; only the d_c column changes.

## Regex vs LLM-consensus d_c agreement on pilot258

| metric | value |
|---|---:|
| Spearman rho | 0.461 |
| within +/-1 | 0.876 |
| exact | 0.481 |
| mean LLM d_c | 0.891 |
| mean regex d_c | 0.298 |
| lower-bound (regex <= LLM) | 0.957 |

Regex d_c distribution over items: {0: 195, 1: 50, 2: 12, 3: 1}. As designed, the keyword floor is sparser and lower than the LLM-consensus label (mean 0.30 vs 0.89), so it is a deliberately conservative, attenuated test.

## Control ladder: LLM-consensus d_c vs regex d_c

| d_c source | spec | beta(d_c) | OR per +1 | AUC |
|---|---|---:|---:|---:|
| LLM-consensus | M0/M0 | -0.5738 | 0.563 | 0.614 |
| LLM-consensus | M1/M1 | -0.3888 | 0.678 | 0.756 |
| LLM-consensus | M2/M2 | -0.3307 | 0.718 | 0.815 |
| LLM-consensus | M3/M3 | -0.3753 | 0.687 | 0.816 |
| regex (zero-LLM) | M0/M0 | -0.6676 | 0.513 | 0.581 |
| regex (zero-LLM) | M1/M1 | -0.4825 | 0.617 | 0.754 |
| regex (zero-LLM) | M2/M2 | -0.3765 | 0.686 | 0.814 |
| regex (zero-LLM) | M3/M3 | -0.4196 | 0.657 | 0.815 |

## Stable item-cluster bootstrap on the M3 d_c coefficient

We use a large, multi-seed item-cluster bootstrap because at B=200 the 97.5th percentile sits near zero and is seed-sensitive. We therefore report the pooled interval over three seeds and the fraction of resamples with beta < 0, a boundary-robust summary that does not hinge on a single percentile estimate.

| d_c source | beta point | OR | stable 95% CI | frac(beta<0) | per-seed 95% CIs |
|---|---:|---:|---:|---:|---|
| LLM-consensus | -0.375 | 0.687 | [-0.823, -0.046] | 0.986 | [-0.84,-0.05]; [-0.81,-0.04]; [-0.84,-0.04] |
| regex (zero-LLM) | -0.420 | 0.657 | [-1.154, 0.083] | 0.948 | [-1.15,0.09]; [-1.18,0.07]; [-1.13,0.08] |

## Reading

Under the main M3 specification, the LLM-consensus d_c gives beta = -0.375 (OR = 0.687); the zero-LLM regex d_c gives beta = -0.420 (OR = 0.657), the same sign and slightly larger magnitude. The negative constraint-penalty effect therefore persists when d_c is computed with no model inference of any kind, despite the regex being a deliberately sparse keyword floor (item-level rho = 0.46 with the LLM label, mean 0.30 vs 0.89), which attenuates the test. The stable bootstrap shows the LLM-d_c coefficient is negative in 99% of item resamples and the regex-d_c coefficient in 95%; the pooled 95% CI is [-0.823, -0.046] (LLM) and [-1.154, 0.083] (regex). The upper tail sits near zero, so the honest statement is directional: both label sources, including one with zero LLM involvement, place the bulk of the bootstrap mass below zero. This is evidence that the negative d_c-accuracy association is a property of the items rather than an artefact of the LLM labelling pipeline. It is not, at pilot258 size, a clean two-sided-significant effect, consistent with the Section 4.4 proxy-sensitivity caveat.
