# PhysReason Benchmark-Author d_c Anchor

External construct-validity check that requires **no human annotation by us**: PhysReason ships step-level `physical_theorem` labels authored by the benchmark creators. We map each theorem string to a conservation-law family (momentum, angular momentum, energy, charge, mass, entropy; Newton's laws, Ohm's law, kinematics, gas laws, geometry, and algebra map to *no* conservation family), count the distinct conservation families the official solution invokes, and call that the benchmark-native $d_c^{\rm bench}$. We compare it to our LLM-Consensus conservation-constraint load \$d_c\$.

- Matched PhysReason items (LLM $d_c$ + benchmark theorems): **81**
- Mapping is conservative (family-presence, 0/1 per family) so $d_c^{\rm bench}$ is a lower-resolution count than our scalar-component protocol; it is a *lower bound* on the full-protocol $d_c$ for multi-component-momentum items.

## Total-count agreement (LLM $d_c$ vs $d_c^{\rm bench}$)

| metric | value |
|---|---:|
| exact match | 0.556 |
| within 卤1 | 0.877 |
| mean(LLM 鈭?bench) | +0.531 |
| MAE | 0.580 |
| Pearson r | 0.521 |
| Spearman 蟻 | 0.542 |

## Per-family presence agreement (Cohen 魏)

| family | Cohen 魏 | LLM-positive n | bench-positive n | n |
|---|---:|---:|---:|---:|
| momentum | 0.263 | 28 | 6 | 81 |
| angular_momentum | 1.000 | 2 | 2 | 81 |
| energy | 0.748 | 38 | 28 | 81 |
| charge | -0.017 | 2 | 1 | 81 |
| mass | 1.000 | 1 | 1 | 81 |
| entropy | 0.661 | 1 | 2 | 81 |

## Reading

The LLM-Consensus conservation-constraint load \$d_c\$ correlates with the benchmark-author conservation-family count at Spearman 蟻 = 0.54 (within卤1 = 88%). Because $d_c^{\rm bench}$ counts family *presence* while the LLM protocol counts scalar *components*, the LLM tends to score slightly higher on multi-component problems (mean LLM 鈭?bench = +0.53). The positive, ordered relationship against an independent, human-authored benchmark annotation supports the construct validity of $d_c$ without requiring us to collect new human labels. A full human-physicist gold pass on the blinded 96-item packet remains the planned final gate.


