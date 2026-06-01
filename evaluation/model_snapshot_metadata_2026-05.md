# Model Snapshot Metadata — Pilot258 (May 2026)

> Reproducibility record: what the API alias *actually* resolved to at run time.
> Future readers and reviewers should treat the "alias" as a pointer that may have shifted.

## API model identifiers at the time of pilot258 runs

| Rater / Solver | Requested alias | Actual snapshot returned | Provider | Approx params | Notes |
|---|---|---|---|---|---|
| Qwen-14B | `qwen2.5-14b-instruct-awq` | (local AWQ) | vLLM @ 5880 | 14 B dense | self-hosted; deterministic |
| Qwen-7B | `qwen2.5:7b-instruct-q4_K_M` | (local GGUF) | Ollama @ 3070 | 7 B dense | self-hosted; deterministic |
| **DeepSeek** | `deepseek-chat` | **`deepseek-v4-flash`** (verified 2026-05-23) | DeepSeek API | ~671 B MoE class (declared V3 era; alias migrated to V4) | **Reviewer-relevant**: not the original V3 weights from 2025; treat as "V4-flash served behind the chat alias in May 2026" |
| Kimi-K2 | `kimi-k2-0905-preview` | (snapshot 2025-09-05) | Moonshot API | 1 T MoE | preview snapshot; pinned |
| Kimi-v1-32k | `moonshot-v1-32k` | (legacy GA) | Moonshot API | ≪ K2 class | included as deliberate weak-rater contrast |

## Manuscript wording rule (W7 freeze)

- Replace every "DeepSeek-V3 671B-MoE" reference with **"DeepSeek (deepseek-chat alias, served as deepseek-v4-flash on 2026-05-23)"** the first time it appears in §4.
- After the first mention, the short form "DeepSeek" is fine.
- Methods §M3 (planned: model table) gets the full row above.

## Why this matters

- The `deepseek-chat` alias is documented by DeepSeek to point at "the current stable chat model." Between 2025-08 and 2026-05 this pointer migrated from V3 to V3.1 to V4. Paper readers replicating in 2026-08 may get a different snapshot.
- Our pilot258 numbers (overall acc = 0.209, $\kappa_{\mathrm{eff}}=0.216$, $R^2 = 0.821$) are therefore *bound to a specific snapshot*. If the alias migrates again, expected behaviour is that absolute accuracy changes but the $d_c$ direction and the envelope shape qualitatively survive (per W6 controls).

## What we did *not* do (and won't claim)

- We did **not** evaluate DeepSeek-R1 (the reasoning model). C4 / C5 paper-grade tests of test-time compute require a reasoning model with a controllable budget; this is a W7-extension item.
- We did **not** evaluate Doubao-Seed-2.0-Pro (the reasoning-style Volcano flagship) because at ~23 s/item it would have taken ~6.6 h for the 4-profile prelabel run alone. A Doubao-1.5-Pro-Chat run is the documented retry path in `logs/doubao_next_step_runbook_2026-05-23.md` should the user activate that endpoint.
- We did **not** run MiniMax-M2.7 in this pilot due to an `insufficient_balance` error on the M2.7 endpoint at the run window; this is recorded as a deferred validation, not a methodological choice.
