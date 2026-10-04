# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## Unreleased

### Changed

- Decode floor (`--decode-min-tps N`): every decoding slot is protected except one that arrived in
  the same burst (within the 30 ms gathering window) as a request that is still prefilling. Requests
  arriving together therefore merge their prefill without false floor throttling (Scenario A, C = 4
  concurrent ~1.1k prompts: 0 floor forwards, batch wall 4.65 s vs 4.91 s in v0.1.2, median of 3),
  while a stream is protected regardless of arrival order. An earlier version of this change protected
  only slots that arrived before every prefilling request, which left a conversation's next turn
  unprotected when it arrived after its subagents' prompts ("reverse order", 25.7k: 8.6 tok/s while
  they prefilled; now 22.8–23.0, v0.1.2 22.4); the server test `decode_floor` now covers that order.
  Scenario B (Swift 27B MXFP4-A, 3 concurrent ~17k subagents) at 25.7k context is unchanged
  (22.7 tok/s, longest pause 0.47 s; v0.1.2 23.1 / 0.47). Outputs are identical to v0.1.2.
- Server: Windows timer resolution set to 1 ms while serving (`timeBeginPeriod`), and the main loop's
  idle / burst-gather / restore waits wake on a new request instead of sleeping: a 1 ms sleep took
  11.5 ms before, 1.9 ms after (`WHIRL_TIMER_PROBE=1`). A short request arriving while the prefix-cache
  tier is still writing the previous one: TTFT median 86 → 69 ms. C = 4 burst of ~1.1k prompts: mean
  TTFT 1207 → 1187 ms.
- Server: with the decode floor off (`--decode-min-tps 0`), a prefilling request's chunks merge into
  2048-row forwards even while other slots decode. With the floor on they still do not while any slot
  decodes (merging next to a long prompt made its prefill only ~4% faster but doubled the decoders'
  stalls, 96k: 1.2 → 3.3 s).
- Speculative decoding & verification: support up to 32 rows verify path with dual-token WMMA GEMV
  kernels (`gemvx_v6` for MXFP4, IQ4_XS, Q4_K, Q5_K, Q6_K) and full-chunk LM head evaluation for wide
  batches (`head_chunk = 248320`; `WHIRL_WIDE_VERIFY=0`, `WHIRL_HEAD_CHUNK=N`); outputs remain
  bit-identical. These only affect verify forwards of more than 16 rows (several users, or n-gram
  drafts adding rows): one user's verify has at most 16 rows per sequence and never takes this path.
  Single-user decode gains on this branch come from `attn_wsplit2` and the Q4_K MTP block below, not
  from wide verify or the whole-vocabulary head (numbers to be re-measured). With 4 users, the whole head raised prose decode 247.1 → 267.8 tok/s.
  Default draft caps stay 8 / 7 / 4 / 3 for 1–4 decoding slots (8 / 8 / 8 / 7 was not faster for code
  or prose).
- Speculative verify attention: a sequence's verify rows now share one K/V pass in groups of up to
  32 query columns (`attn_wsplit2`; 27B: 5 rows × 6 GQA heads) instead of 16 (2 rows). The second
  column group runs on the block's otherwise idle second wave over the same staged Vᵀ tile; each
  column's arithmetic is unchanged, so outputs are bit-identical (checked by `checkAttnGroups`, the
  kernel test and server A/B against v0.1.2). Attention kernel time per layer at 16k context: one
  user with 4 drafts 0.286 → 0.141 ms, four users with 3 drafts each 0.853 → 0.607 ms; at 32k,
  1.784 → 1.162 ms. Single-row decode keeps `attn_wsplit1`. `WHIRL_ATTN_WIDE=0` restores the old
  grouping.
- MTP block stored as Q8_0 (e.g. the Swift-1.5 MXFP4 files) is now requantized to Q4_K for drafting,
  like Q6_K blocks already were (`WHIRL_MTP_Q4=0` keeps the file's types). Drafts only; outputs are
  unchanged.

## 0.1.2 — unreleased

### Changed

- Server: the pinned-RAM tier of the prefix cache (`--kv-ram-mb`) is now sized from the machine
  instead of a fixed max(8 GiB, one full-length session + checkpoints) ≈ 9 GiB: **1/4 of physical
  RAM** (nearest GiB), at least that old minimum, at most 32 GiB (the minimum wins over the cap), and
  never more than half of the RAM available at startup (so the pinned arena does not push the machine
  into paging; logged as a warning when it limits the size; under 1 GiB the tier stays off). A 64 GB
  PC now gets 16 GiB. In real agent use (Hermes, 30–40k-token sessions with subagents) the old 9 GiB
  filled within an hour, after which restores came from the SSD. `--kv-ram-mb N` /
  `WHIRL_KV_RAM_MB=N` still override (used as given), `0` still turns the host tiers off. On an
  integrated GPU (shared system memory) the RAM tier is off unless a size is given. The startup log
  prints the chosen size and the reason, and the pinning time.

## 0.1.1 — 2026-10-03

### Added

- Server: read-only compatibility endpoints for clients that auto-detect the server type and context
  length (agent frameworks, chat front ends, IDE plugins):
  - `GET /props` and `GET /v1/props` — llama.cpp-server-style subset: `default_generation_settings`
    (`n_ctx` = context per slot, `model`), `total_slots`, `model_path` (file name only), `model_alias`,
    `modalities`, `build_info`.
  - `GET /version` — `{"version":"0.1.1","name":"whirl"}`.

- Chat requests: the OpenRouter / OpenAI Responses-style `"reasoning": {"effort": "...", "enabled": bool}`
  object, and effort aliases: `max` / `ultra` / `xhigh` / `high` → xhigh, `medium` → medium,
  `low` / `minimal` → low, `none` → thinking off (values are case-insensitive).

- Server: decode floor, `--decode-min-tps N` (env `WHIRL_DECODE_MIN_TPS`, default 20, `0` = off).
  While other requests prefill long prompts, every streaming request keeps at least N tok/s: prefill
  forwards are limited to a row budget (whole chunks; the oldest request always progresses) and decode
  cycles are interleaved, adapted every cycle from the measured decode-cycle time, tokens per cycle and
  prefill rate. Swift-1.5 27B MXFP4 on the R9700, one stream at 25.7k context + three ~17k-token prompts:
  the stream goes from 3.3 to 23.0 tok/s (longest pause 1.22 → 0.47 s), prefill throughput meanwhile
  2791 → 1862 tok/s, mean TTFT of the three 17.8 → 18.6 s. With no slot decoding, prefill is unchanged;
  outputs are bit-identical for any N; short-prompt C = 1 / C = 4 throughput unchanged.

### Changed

- Chat requests: an unknown reasoning effort no longer fails the request with 400; the server logs a
  warning and uses the default effort. Prompts for the values accepted before are unchanged.
- Server: probes of LM Studio / Ollama native paths (`/api/v1/models`, `/api/tags`, `/api/show`,
  `/api/version`, `/api/v0/models`) still answer 404 but are logged once per path at `I` level
  instead of a `W` line per request.

## 0.1.0 — 2026-10-03

- First public release: native Windows C++/HIP inference engine for qwen35 / qwen35moe GGUF models on
  the Radeon AI PRO R9700 (gfx1201) — `whirl.exe` (CLI) and `whirl-server.exe` (OpenAI-compatible
  server with continuous batching, prefix cache, RAM / SSD KV tiers, MTP and n-gram speculative
  decoding, image input).
