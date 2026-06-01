# W6 Sensitivity: DeepSeek-reasoner as a 5th model

- Reasoner rows merged: **254** / 258 (PARTIAL); reasoner overall acc = 0.417.
- 4-model panel (n=1032): M3 β_dc = -0.3753 (OR 0.687), AUC 0.816.
- **5-model panel (n=1286, + reasoner): M3 β_dc = -0.3356 (OR 0.715), AUC 0.828.**
- 5-model item-cluster bootstrap (B=2000): 95% CI [-0.661, -0.100], β<0 in 99.8% of resamples.

**Reading.** Adding a high-dynamic-range, non-floor solver as a 5th model preserves the negative controlled d_c penalty (OR 0.71 per +1 d_c), and the model fit improves (AUC 0.82->0.83). This shows the headline effect is not carried by the floor-limited arms alone.
