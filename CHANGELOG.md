# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## Unreleased (0.2.0)

### Numerics modes: precise (default), balance, fast

- New `--precise` / `--balance` / `--fast` (or `--mode M`, env `WHIRL_MODE`) for `whirl` and
  `whirl-server`; one mode per process, reported in the start-up log and in `GET /props`
  (`"numerics"`). See [docs/usage.md](docs/usage.md#modes).
- **precise is the new default** on every GPU: no fp8 MXFP4 prefill activations (and no rounding of
  folded MXFP4 exponents), no fp8 MoE expert prefill, f32 DeltaNet prefill chunks instead of
  f16-WMMA, no f16 GEMM intermediates (h16), and **f16 KV everywhere** (no automatic q8v / q8h / q8,
  also on cards under 20 GiB and on the Radeon 8060S). When f16 KV does not fit, the context is
  lowered with a warning, or the program stops with a message when `--ctx` / `--ctx-per-slot` was
  given. MXFP4 prefill is slower in precise mode than in 0.1.x.
- **balance** is the 0.1.x / 0.2.0-rc behaviour, bit for bit (items `fp8`, `moefp8`, `gdnwmma`,
  `h16`, `kvq8`; `--balance=ITEMS` picks a subset). **fast** lists the future gated lossy items
  (4-bit KV, relaxed acceptance, ...); none is implemented yet, so fast runs as balance.
- The per-item variables (`WHIRL_FP8`, `WHIRL_MOE_FP8`, `WHIRL_GDN_WMMA`, `WHIRL_FFN_H16`,
  `WHIRL_Q4_RELAXED`, `WHIRL_KV`) still work and override the mode (logged as user-requested).
- Still pending in precise mode: the decode / verify GEMV quantizes activations to int8 (q8_1
  class), as llama.cpp does; a float-activation version is being written.

## 0.1.4 — 2026-10-06

Routine bug-fix release: no new features, no speed changes. The changes harden input handling (GGUF
files, HTTP requests, tokenizer vocabularies); inference code paths are unchanged, so outputs are
bit-identical to 0.1.3.

### Fixes

- Server (HTTP): a streaming client that stops reading no longer stalls the other slots. While a
  request runs, the engine writes its response through a non-blocking socket; bytes the client does
  not accept stay buffered and are retried on the next flush. With no progress for 30 s, or more than
  64 MiB unsent, the connection counts as gone and the request is dropped (its slot is freed). The
  rest of a finished response is sent by the connection thread (a10ac5b).
- Server: shutdown no longer waits up to 30 s for connections still reading a request; `stop()`
  shuts them down and cancels the pending reads. Connections close with a linger so clients receive
  error responses instead of a reset (a10ac5b).
- Tokenizer: `piece()` decodes `<0xNN>` byte tokens with a small hex parser instead of `std::stoi`,
  which threw on `<0xZZ>` and silently mis-decoded `<0x-1>` / `<0x1Z>` (24475c4).
- zh-TW guide: bold runs broken by CJK punctuation render correctly again (no text change, 8b2c58d).

### Security hardening

- GGUF parser: tensor extent checks are overflow-safe (subtraction-based `File::inBounds`, at parse
  and in `tensorData()`), so `data_offset + offset + nbytes` can no longer wrap past 2^64; dimensions
  above INT64_MAX and shapes whose row count, element count or byte size overflow are rejected;
  `TensorInfo::nbytes()` / `rowBytes()` are overflow-checked. The header's KV and tensor counts must
  fit in the remaining file bytes, so their `reserve()` is bounded by the file size (b6af30a).
- Model load: `Config::fromGguf` rejects inconsistent or out-of-range hyper-parameters (for example
  `nextn >= block_count`, zero dimensions, `head_dim > 256`, GQA group > 8,
  `value_length != key_length`, interval 0) with `UnsupportedConfig`; `Config::validateTensors`
  checks the type and shape of every tensor `Model::load` reads (including MoE experts and the MTP
  block) before any GPU allocation, with the new error codes `MissingTensor`, `UnsupportedTensorType`
  and `UnsupportedTensorShape` (b6af30a).
- HTTP: 30 s idle timeout per read and a 60 s deadline for the request line and headers (`408`); at
  most 64 open connections (`503`, then close); headers capped at 64 KiB / 100 lines (`431`);
  `Content-Length` above 64 MiB is refused (`413`) instead of wrapping, conflicting duplicate
  `Content-Length` headers get `400`; the body buffer grows as data arrives instead of reserving the
  claimed length (a10ac5b).
- CORS: `Access-Control-Allow-Origin` is sent only to pages from `http(s)://localhost`, `127.0.0.1`
  or `[::1]` (any port), echoing the `Origin` with `Vary: Origin`; other origins (including `null`,
  i.e. `file://` pages) get no CORS headers, and preflight answers the same way. New option
  `--cors-origin ORIGIN` (repeatable; `*` restores the previous allow-all behaviour) (a10ac5b,
  docs 765f49a).
- Tokenizer: a byte-type token whose text is not exactly `<0xNN>` (two hex digits) fails loading with
  `TokenizerError("malformed byte token at id N")` (24475c4).

### Behaviour changes

- CORS default tightened (see above). Browser pages on other origins, or opened from `file://`, can no
  longer read the server's responses unless started with `--cors-origin <origin>` (or
  `--cors-origin "*"`). SDKs, curl, IDE agents and other non-browser clients are not affected.
- HTTP limits: 64 connections, 30 s read idle / 60 s header deadline, 30 s send stall or 64 MiB
  unsent on a streaming response, 64 KiB / 100 header lines, 64 MiB request body.
- Malformed GGUF files that 0.1.3 might have loaded (or crashed on) are now refused at load with one
  of the error codes above; the release front-end reports them as "cannot be loaded".

### Internal

- Version 0.1.4 (`project(whirl VERSION 0.1.4)`).
- MXFP4: the loader and the tests share one row-repack implementation (`kernels::repackMxfp4Row`,
  new scratch-buffer overload; scratch still allocated once per tensor); model tests pin the bytes
  against frozen copies of both previous implementations (ee87049).
- `build.bat` finds `vcvars64.bat` with `vswhere`, restricted to Visual Studio 2022
  (`-version [17.0,18.0)`; HIP clang with the VS 2026 STL is unverified), any edition with the C++ x64
  tools; it falls back to the default Build Tools path and prints a clear error if neither exists
  (f54e2da).
- Tests: crafted malformed GGUFs, synthetic tiny models with mutations (optional
  `WHIRL_TEST_GGUF="a.gguf;b.gguf"` runs the load validation on real files without a GPU), HTTP limits,
  connection cap, stuck reader, CORS matrix, byte-token validation.

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
