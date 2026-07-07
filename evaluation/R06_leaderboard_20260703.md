# R06 Formulation-competence leaderboard (dense capability gradient)

Date 2026-07-03. Fourteen models on the 258-item pilot, LLM-consensus d_c, fixed DeepSeek judge; univariate ridge logistic correct~d_c, item-cluster bootstrap B=2000 seed 20260703. OR(d_c) is the per-constraint odds ratio: lower values mean a larger conservation-load penalty, and values near 1 mark a flatter penalty. New API models this round are marked `panelx-new`.

| model | class | n | accuracy | OR(d_c) | 95% CI | % beta<0 |
|---|---|---:|---:|---:|---|---:|
| Qwen7B-Ollama | panel-4family | 258 | 0.058 | 0.669 | [0.332, 1.047] | 96% |
| Qwen14B | panel-4family | 258 | 0.089 | 0.517 | [0.258, 0.793] | 100% |
| qwen-turbo | panelx-new | 257 | 0.125 | 0.384 | [0.185, 0.613] | 100% |
| qwen-plus | panelx-new | 258 | 0.136 | 0.682 | [0.415, 0.952] | 99% |
| qwen-max | panelx-new | 257 | 0.136 | 0.684 | [0.394, 1.003] | 98% |
| KimiK2 | panel-4family | 258 | 0.143 | 0.508 | [0.263, 0.801] | 100% |
| qwen3-max | panelx-new | 257 | 0.144 | 0.608 | [0.377, 0.867] | 100% |
| doubao-1.5 | panelx-new | 255 | 0.200 | 0.656 | [0.434, 0.893] | 100% |
| DeepSeekV3 | panel-4family | 258 | 0.209 | 0.569 | [0.360, 0.796] | 100% |
| doubao-seed2 | panelx-new | 256 | 0.410 | 0.698 | [0.499, 0.923] | 100% |
| DeepSeek-reasoner | reasoner | 254 | 0.417 | 0.683 | [0.499, 0.900] | 100% |
| OpenAI-Codex arm | frontier | 245 | 0.445 | 0.851 | [0.647, 1.115] | 88% |
| deepseek-v4-pro | panelx-new | 258 | 0.473 | 0.712 | [0.525, 0.921] | 100% |
| Anthropic-agent arm | frontier | 257 | 0.595 | 0.945 | [0.733, 1.231] | 66% |

**Capability gradient:** Spearman(accuracy, OR) = 0.697 across all 14 models (= 0.893 among non-floor models, accuracy >= 0.15, where OR is stably estimated); OR range 0.38-0.95. The per-constraint penalty shrinks toward frontier capability, so the instrument tracks equation formulation in multi-constraint physics problems.

Floor caveat: models below about 0.15 accuracy have unstable OR estimates because there are few correct items. The gradient is therefore reported both across all models and among non-floor models.
