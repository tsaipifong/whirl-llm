# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## Unreleased

### Changed

- Decode floor (`--decode-min-tps N`): refined streaming decode protection to only protect slots
  that entered decoding prior to the arrival of the earliest prefill request in the active prefill
  set (`t_arrive + gather_ms < min_t_arrive` and `td0 < max_t_arrive`). Requests arriving in the same
  burst window now coalesce and prefill together without triggering false floor throttling.
  Scenario A (C = 4 concurrent ~1.1k prompts) achieves 288.8 tok/s decode (wall 4.67 s vs 5.10 s in
  v0.1.2) with 0 false floor forwards. Scenario B (Swift 27B MXFP4-A streaming at 25.7k, 97.4k, and
  123.7k context under 3 concurrent ~17k subagents) maintains 23.0–23.7 tok/s stream with longest
  pause ≤0.499 s.
- Speculative decoding & verification: support up to 32 rows verify path with dual-token WMMA GEMV
  kernels (`gemvx_v6` for MXFP4, IQ4_XS, Q4_K, Q5_K, Q6_K) and full-chunk LM head evaluation for wide
  batches (`head_chunk = 248320`). Single-user decode throughput at 96k reaches 62.5 tok/s (from 49.8)
  and 128k reaches 54.9 tok/s (from 45.9); outputs remain bit-identical.
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
