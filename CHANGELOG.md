# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

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
