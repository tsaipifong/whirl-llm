# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## 0.1.3 — 2026-10-05

Every change below keeps the output bit-identical to plain greedy decoding, unless it says otherwise.

### Performance

- Prefill attention: new GQA-grouped kernel (`attn_kg`) for f16, q8v, q8 and q8h KV. One block holds the
  query heads of one KV head over 16 queries; K fragments are loaded straight into WMMA registers,
  f16 Vᵀ fragments with RDNA 4's transposing load (`global_load_tr_b128`), q8v V is dequantized once
  per block into a double-buffered LDS stage, and there is one barrier per 32-key tile. Outputs are
  bit-identical to the previous kernel (kernel-test invariants; last-token logits, MTP == plain and
  server outputs identical to 0.1.2). Kernel throughput at long context: f16 ~65 → ~93 TFLOPS, q8v
  ~61 → ~84 TFLOPS. `whirl bench` prefill, Swift-1.5 27B MXFP4-A (f16 KV, R9700): 32k +10%, 64k
  +17%, 96k +21%, 128k +24%; Ornith MXFP4 64k +11%, 128k +14%; Qwen3.8-27B UD-Q4_K_M (f16 KV) 32k
  +5%, 128k +16%. Long prompts on the server (q8v KV for the dense models): Swift 96k +19%, 128k +21%.
  q8 and q8h KV use `attn_kg` too (`attn_kg6_q8` / `attn_kg4_q8`; K dequantized from int8 with a
  magic-number f16 conversion, bit-identical to `attn_kx_q8`; probe at 61k 57.8 → 72.1 TFLOPS). Server
  prefill, Swift-1.5 27B MXFP4-A, one user, `--ctx-per-slot 262144`, q8h KV: 128k 1,342 → 1,522
  (+13.4%), 192k 1,021 → 1,177 (+15.3%), 256k 824 → 961 (+16.6%). `WHIRL_ATTN_KG=0` restores the
  previous kernel. See [kernels.md](docs/guide/en/kernels.md#flash).
- Speculative verify attention: a sequence's verify rows share one K/V pass in groups of up to 32
  query columns (`attn_wsplit2`; 27B: 5 rows × 6 GQA heads) instead of 16 (2 rows). The second column
  group runs on the block's otherwise idle second wave over the same staged Vᵀ tile; each column's
  arithmetic is unchanged (checked by `checkAttnGroups`, the kernel test and server A/B against
  0.1.2). Attention kernel time per layer at 16k context: one user with 4 drafts 0.286 → 0.141 ms,
  four users with 3 drafts each 0.853 → 0.607 ms; at 32k, 1.784 → 1.162 ms. Single-row decode keeps
  `attn_wsplit1`. `WHIRL_ATTN_WIDE=0` restores the old grouping.
- Wide verify: verify forwards of 17–32 rows use dual-token WMMA GEMV kernels (`gemvx_v6` for MXFP4,
  IQ4_XS, Q4_K, Q5_K, Q6_K) and the output head covers the whole vocabulary per chunk
  (`head_chunk = 248320`; `WHIRL_WIDE_VERIFY=0`, `WHIRL_HEAD_CHUNK=N`). Only several users (or n-gram
  drafts adding rows) take this path; one user's verify has at most 16 rows. 4 users, prose:
  247.1 → 267.8 tok/s. Default draft caps stay 8 / 7 / 4 / 3 for 1–4 decoding slots.
- MTP block stored as Q8_0 (e.g. the Swift-1.5 MXFP4 files) is requantized to Q4_K for drafting, like
  Q6_K blocks already were (`WHIRL_MTP_Q4=0` keeps the file's types). Drafts only.
- Server: Windows timer resolution set to 1 ms while serving (`timeBeginPeriod`), and the main loop's
  idle / burst-gather / restore waits wake on a new request instead of sleeping: a 1 ms sleep took
  11.5 ms before, 1.9 ms after (`WHIRL_TIMER_PROBE=1`). A short request arriving while the
  prefix-cache tier is still writing the previous one: TTFT median 86 → 69 ms. C = 4 burst of ~1.1k
  prompts: mean TTFT 1207 → 1187 ms.
- Server: with the decode floor off (`--decode-min-tps 0`), a prefilling request's chunks merge into
  2048-row forwards even while other slots decode. With the floor on they still do not while any slot
  decodes (merging next to a long prompt made its prefill only ~4% faster but doubled the decoders'
  stalls, 96k: 1.2 → 3.3 s).

### Fixes

- Server KV pool sized under the WDDM budget, MTP KV allocated first (FIX-1). The automatic pool kept
  only 768 MiB of free VRAM, which put the process over its WDDM local budget (31.29 / 31.02 GiB on
  the R9700); Windows then demoted the last allocation, the MTP KV, to system memory and the draft
  head read it across PCIe: a ~128k-token agent request decoded at 29.0 tok/s. The pool is now
  min(free VRAM − reserve, WDDM budget − usage − margin) (`hip::wddmMemInfo()`, DXGI
  `IDXGIAdapter3::QueryVideoMemoryInfo`, adapter matched by LUID; margin 768 MiB, MoE 1.5 GiB,
  `WHIRL_POOL_BUDGET_MARGIN_MB`; `WHIRL_POOL_RESERVE_MB` keeps the old free-VRAM-only rule), and
  `allocKvPool` allocates the MTP KV before the trunk layers. After startup the server logs WDDM local
  usage / budget and non-local growth, and warns when over budget. Swift-1.5 27B MXFP4-A, default 4
  slots: pool 226,304 tokens (v0.1.2: 210,688; +7.4%), local 30.06 / 31.02 GiB; the 128k agent request
  decodes at 67.2 tok/s, output identical. Server tests 340/340; gates basic / tier / restore_conc
  identical to v0.1.2.
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
- CLI: n-gram drafts are capped at 1 on the unfused DeltaNet decode path (for example F32 / F16
  `ssm_alpha` / `ssm_beta` without `gdn_ab`), like the MTP draft count and the server already were.
  That path writes only snapshot 0, so accepting 1..nd−1 drafts restored a never-written snapshot and
  diverged from greedy. `restoreSnapshot` now throws `SnapshotNotWritten` for out-of-range or
  unwritten sets (42b4cc9).

### Defaults

- The token embedding table is kept in pinned host memory (`WHIRL_EMBD_HOST`, now on by default;
  `WHIRL_EMBD_HOST=0` keeps it in VRAM). The KV pool grows by ~11.9% on the R9700, prefill −0.4–0.9%.
  Models with tied embeddings or an MXFP4 embedding keep it in VRAM. The server adds the embedding
  size to its pinned-memory declaration (`%LOCALAPPDATA%\whirl\pinned\<pid>.txt`), so Shared Usage
  monitoring does not count it as spill.
- The MTP draft head uses a 64k-token vocabulary subset (`WHIRL_DRAFT_VOCAB`, default `64k`),
  embedded in the executable (`data/draft_vocab/subset_64k.bin` via whirl-bin2c), for dense qwen35
  models with the 2-bit draft head and a 248,320-token vocabulary; MoE models and other vocabularies
  keep the full head. Main scenario +3.4–5.0% tok/s, same acceptance. `WHIRL_DRAFT_VOCAB=off|48k|<file>`
  overrides. The subset is only a list of token ids (uint32 LE, ascending), ranked by token frequency
  over permissively licensed code and documentation (llama.cpp, ROCm aiter, hipfire, dflash,
  PaddleNLP / PaddleOCR, ECharts, WHIRL docs) and Wikipedia samples (CC BY-SA 4.0), tokenized with the
  Qwen3.x tokenizer; no corpus text is included. Generated by whirl-cloud `tools/vocab_subset/build.py`
  (sources and licenses in its `sources.tsv`).
- MTP draft attention window on by default (DEF-2, 2ba0713): from a 64k-token context on, the draft
  attention sees only the first 256 and the last 16,384 positions. Trunk and verification always see
  the whole context, so output hashes are identical. Window W = 16,384 against off: typical case
  (real agent sessions, server) 110k +1.1%, 128k +7.6%, 200k +3.7%; worst case (long non-repeating
  text) 128k −0.5%, 256k +17% (single run); best case (repeated editing) 128k +3.1%, 256k +4.2%.
  `WHIRL_DRAFT_WINDOW=0` turns it off; `WHIRL_DRAFT_WINDOW=W` / `WHIRL_DRAFT_WINDOW_MIN=N` change the
  window and the threshold.

### Internal

- Version 0.1.3 (`project(whirl VERSION 0.1.3)`, 8fbee5f).
- `package_release`: `dxgi.dll` (WDDM budget query, FIX-1) and `winmm.dll` (`timeBeginPeriod`) added
  to the allowed system DLLs; both ship with every Windows 10/11 install, so requirements are
  unchanged (3b1bebc).
- `whirl bench`: `WHIRL_PROFILE=1` prints the per-op-class GPU time of each prefill size.

## 0.1.2 — 2026-10-03

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
