# Model Snapshot Metadata — Pilot258 (May 2026)

> Reproducibility record: what the API alias *actually* resolved to at run time.
> Future users should treat the "alias" as a pointer that may have shifted.

## API model identifiers at the time of pilot258 runs

| Rater / Solver | Requested alias | Actual snapshot returned | Provider | Approx params | Notes |
|---|---|---|---|---|---|
| Qwen-14B | `qwen2.5-14b-instruct-awq` | (local AWQ) | vLLM @ 5880 | 14 B dense | self-hosted; deterministic |
| Qwen-7B | `qwen2.5:7b-instruct-q4_K_M` | (local GGUF) | Ollama @ 3070 | 7 B dense | self-hosted; deterministic |
| **DeepSeek** | `deepseek-chat` | **`deepseek-v4-flash`** (verified 2026-05-23) | DeepSeek API | ~671 B MoE class (declared V3 era; alias migrated to V4) | Reproducibility note: not the original V3 weights from 2025; treat as "V4-flash served behind the chat alias in May 2026" |
| Kimi-K2 | `kimi-k2-0905-preview` | (snapshot 2025-09-05) | Moonshot API | 1 T MoE | preview snapshot; pinned |
| Kimi-v1-32k | `moonshot-v1-32k` | (legacy GA) | Moonshot API | ≪ K2 class | included as deliberate weak-rater contrast |

## Manuscript wording rule (W7 freeze)

- Replace every "DeepSeek-V3 671B-MoE" reference with **"DeepSeek (deepseek-chat alias, served as deepseek-v4-flash on 2026-05-23)"** the first time it appears in §4.
- After the first mention, the short form "DeepSeek" is fine.
- Methods §M3 (planned: model table) gets the full row above.

## Why this matters

- The `deepseek-chat` alias is documented by DeepSeek to point at "the current stable chat model." Between 2025-08 and 2026-05 this pointer migrated from V3 to V3.1 to V4. Paper readers replicating in 2026-08 may get a different snapshot.
- Our pilot258 numbers (overall acc = 0.209, $\kappa_{\mathrm{eff}}=0.216$, $R^2 = 0.821$) are therefore *bound to a specific snapshot*. If the alias migrates again, expected behaviour is that absolute accuracy changes but the $d_c$ direction and the envelope shape qualitatively survive (per W6 controls).

## What we did *not* do in the May pilot

- We did **not** evaluate DeepSeek-R1 (the reasoning model) in the May pilot.
- We did **not** evaluate Doubao-Seed-2.0-Pro (the reasoning-style Volcano flagship) in the May pilot.
- We did **not** run MiniMax-M2.7 in this pilot due to an `insufficient_balance` error on the M2.7 endpoint at the run window; this is recorded as a deferred validation, not a methodological choice.

## June 2026 solver-arm update

The two "did not evaluate" notes above are historical to the May pilot. Later solver arms added after the pilot include:

| Manuscript name | Exact model / snapshot string | Provider / access | Role |
|---|---|---|---|
| DeepSeek-reasoner | `deepseek-reasoner` | DeepSeek API | strong-solver and reasoner-vs-chat arms |
| Doubao-1.5-pro | `doubao-1-5-pro-32k-250115` | Volcano ARK API | compute sweep and chat baseline |
| Doubao-Seed-1.6 | `doubao-seed-1.6` | Volcano ARK API | cross-family reasoner arm |
| Doubao-Seed-2.0-pro | `doubao-seed-2-0-pro-260215` | Volcano ARK API | solver-capable mechanism and expansion arms |
| Qwen-Max / Qwen-Plus | `qwen-max` / `qwen-plus` | DashScope API | synthetic capability spectrum |
| Qwen2.5-1.5B / 3B | `qwen2.5:1.5b` / `qwen2.5:3b` | Ollama (local) | weak end of the synthetic spectrum |
| Anthropic-agent arm | Anthropic agent interface, 2026-06 | Anthropic | frontier solver; as-run interface label, not a public model-version claim |
| OpenAI-Codex arm | OpenAI Codex agent interface, 2026-06 | OpenAI | frontier solver and expansion arm; as-run interface label, not a public model-version claim |

The June frontier runs are agent-app/interface runs and therefore do not carry per-item token-budget control. The manuscript treats them as solver-interface arms rather than controlled compute arms. The neutral manuscript labels `Anthropic-agent arm` and `OpenAI-Codex arm` are used to avoid implying externally verifiable public model-version names.
