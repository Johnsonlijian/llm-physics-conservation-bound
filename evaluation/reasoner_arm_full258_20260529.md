# Strong-Solver Arm: DeepSeek-reasoner on full pilot258

DeepSeek-reasoner (`deepseek-reasoner`, max_tokens 16384) solved on all of pilot258 and judged by `deepseek-chat` (same prompts as the rest of the pipeline; script 35). This supplies the high-dynamic-range arm the audit asked for, to test whether the d_c decline is a floor artefact.

- Judged rows: **254** / 258 (PARTIAL — run again when complete)
- Overall accuracy: **0.417** (vs floor arms: DeepSeek-chat 0.209, Kimi-K2 0.143, Qwen-14B 0.089, Qwen-7B 0.058)
- Univariate logistic correct ~ d_c: β = -0.382, **OR = 0.683** per +1 d_c (SE 0.144, z -2.65, approx p 0.00808).

## Per-d_c accuracy (reasoner)

| d_c | n | accuracy |
|---:|---:|---:|
| 0 | 104 | 0.510 |
| 1 | 98 | 0.398 |
| 2 | 33 | 0.242 |
| 3 | 14 | 0.357 |
| 4 | 4 | 0.250 |
| 5 | 1 | 0.000 |

## Reading

The reasoner arm reaches 0.42 overall — genuine dynamic range, well off the 0.04–0.21 floor of the chat arms — yet the d_c slope remains negative (OR 0.68 per constraint). Because this model is not floor-limited, the persistence of the negative d_c–accuracy relationship here is direct evidence that the decline is a property of d_c, not an artefact of weak models bottoming out at high d_c. Figure F17.
