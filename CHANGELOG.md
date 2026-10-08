# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## Unreleased (0.2.0)

### Numerics modes: balance (default), precise, fast

- New `--precise` / `--balance` / `--fast` (or `--mode M`, env `WHIRL_MODE`) for `whirl` and
  `whirl-server`; one mode per process, reported in the start-up log and in `GET /props`
  (`"numerics"`). See [docs/usage.md](docs/usage.md#modes).
- **balance stays the default** when no mode is given (owner decision for 0.2.0; same output as
  0.1.x, including q8v KV on cards under 20 GiB); `--precise` and `--fast` are opt-in, `--balance`
  is still accepted, and `GET /props` `"numerics"` reports `"source"` (`default`, `command line` or
  `WHIRL_MODE`).
- **precise** (`--precise`) on every GPU: no fp8 MXFP4 prefill activations (and no rounding of
  folded MXFP4 exponents), no fp8 MoE expert prefill, f32 DeltaNet prefill chunks instead of
  f16-WMMA, no f16 GEMM intermediates (h16), and **f16 KV everywhere** (no automatic q8v / q8h / q8,
  also on cards under 20 GiB and on the Radeon 8060S). When f16 KV does not fit, the context is
  lowered with a warning, or the program stops with a message when `--ctx` / `--ctx-per-slot` was
  given. MXFP4 prefill is slower in precise mode than in 0.1.x.
- **balance** is the 0.1.x / 0.2.0-rc behaviour, bit for bit (items `fp8`, `moefp8`, `gdnwmma`,
  `h16`, `kvq8`; `--balance=ITEMS` picks a subset). **fast** lists the future gated lossy items
  (4-bit KV, relaxed acceptance, ...); none is implemented yet, so fast runs as balance.
- `whirl selftest`: the precise decode checks turn fp8 prefill, MoE fp8 and h16 off themselves
  (FIX-PS). Since balance became the default, selftest loaded in balance and its "== prefill GEMM
  rows" (MXFP4) and precise MoE n=17..32 checks compared against the fp8 prefill paths and
  failed; precise itself was unchanged (`--precise` selftest and the precise goldens passed). The
  log now prints the load mode.
- **KV format fixed per mode (BAL-Q8)**: precise f16; balance q8h on dense models (KL vs f16
  0.0002–0.002) and f16 on MoE models; fast q4 (Radeon 8060S, and the R9700 since FAST-1c). The same on every card, CLI and server (one function,
  `numerics::chooseKv`, decided before the kernels load). The automatic f16 → q8v → q8h fallback and
  the q8v preference on cards under 20 GiB are gone: a format that does not fit lowers the context
  (when not given) or stops with a message, never switches to a lower-precision format.
  `WHIRL_KV` remains as a debug override ([usage 7.7](docs/usage.md#env-debug)).
- **balance / fast on the R9700: WMMA DeltaNet and f16 GEMM outputs for every dense model (KG-1)**:
  `gdnwmma` and `h16` were MXFP4-only; they now also apply to dense models of any weight format
  (Q4_K_M, Q5_K, Q6_K, IQ4_XS, ...). Qwen3.8-27B Q4_K_M, balance, prefill 8k 1849 → 2042 tok/s
  (+10.4%), 32k 1649 → 1812 (+9.9%), 128k 1201 → 1285 (+7.0%); decode unchanged (96.98 → 97.15).
  KL vs precise over the last 128 rows of 128k prompts: code 3.1e-4 → 2.7e-4 (top-1 100% → 100%),
  zh 3.7e-4 → 4.1e-4 (99.22% → 98.44%, one near-tie row); balance before vs after 1.9e-4 / 4.6e-4.
  A fused quantized GEMM choice of the autotuner now runs as its bitwise twin (dequant + f16-out
  GEMM) when an f16 output is wanted, so which GEMMs write f16 never depends on tuning timings.
  Greedy output of balance / fast changes for these models (goldens re-recorded); precise, MoE
  models without MXFP4 experts and the Radeon 8060S are unchanged.
- **MoE verify batches over 16 rows on the decode experts (KG-2)**: `moeBlock` sends every batch up
  to the current verify width (32 inside a wide verify / MTP step) through the int8 decode experts,
  never the prefill grouped GEMM, so a 17–32-row verify computes each row exactly as its 1-token
  decode (selftest: balance MoE block, n = 2..32 bitwise == n = 1). New opt-in
  `WHIRL_MOE_WIDE=1` lets MoE models (balance / fast, R9700) use wide verify; it stays **off by
  default**: MoE verifies were already capped at 16 rows (one user never exceeds 16), and with it
  on, Ornith-1.5-35B-A3B MXFP4 at four users was slower (server_c4_nothink 386.0 → 379.4 tok/s,
  −1.7%; batch_c4 498.7 → 486.9, −2.4%; one user unchanged, 7 prompts 255.7 → 255.8) because each
  extra row reads its own experts. Output is the same either way (Ornith balance / fast, MTP +
  n-gram == plain, solo and concurrent, also with forced n-gram drafts); goldens unchanged.
- The per-item variables (`WHIRL_FP8`, `WHIRL_MOE_FP8`, `WHIRL_GDN_WMMA`, `WHIRL_FFN_H16`,
  `WHIRL_Q4_RELAXED`, `WHIRL_KV`) still work and override the mode (logged as user-requested).
- New balance / fast item `specsample` (server, MTP models): with temperature > 0 the MTP drafts are
  drawn from the draft head's distribution (same temperature / top-k / top-p / min-p), coupled to the
  target draw by a Gumbel-max / exponential race keyed by (seed, output position, token): the target
  token is argmin E(v) / p(v), the draft argmin E(v) / q(v) with the same E, and a draft is accepted
  iff it equals the target token. Sampling stays exact (same distribution as plain sampling), the
  acceptance is close to the optimum sum min(p, q), and the text for a seed depends only on the seed
  and the model's logits: the same with MTP on or off and whatever the draft counts, prefix cache or
  earlier requests. With specsample on (balance / fast, temperature > 0) the token sequence for a
  given seed differs from 0.2.0-rc's inverse-CDF sampler; precise (specsample off) keeps the
  0.2.0-rc sampler and its sequences exactly. Greedy decoding is unchanged in every mode.
  `WHIRL_SPEC_SAMPLE=0|1` overrides (0 = 0.2.0-rc sampler and greedy drafts). (An earlier rc2 build
  used the accept-min(1, p/q) / residual rule; its emitted tokens depended on draft counts and the
  draft head's state, so the same seed could give different text between requests.)
- precise decode / verify no longer quantizes activations to int8: dense matmuls take f16
  activations on the f16 GEMM's numerics (bitwise == the prefill GEMM rows, for every batch size up
  to 32), MoE experts take f32 activations with f32 accumulation. Every decode / verify batch up to
  32 rows (also a long n-gram draft) runs the same kernels, so MTP + n-gram == plain token for token
  on MoE models too. Cost vs balance on R9700 (plain decode): Ornith-1.5-35B-A3B about -29%,
  Swift-1.5 27B about -26%, Qwen3.8-27B Q4_K_M about -29%.
- Server KV tier (RAM / SSD prefix cache): the cache fingerprint now includes the numerics mode and
  its items. Before, a `--precise` server could restore a prompt's KV / DeltaNet state saved by a
  `--balance` run of the same exe and model when both kept f16 KV (MoE models), so its output
  depended on what an earlier process had cached.
- fast `kvq4` on the Radeon 8060S: new q4 decode / verify attention kernel `attn_dq4` (int8 q x 4-bit K
  integer dot, one scale multiply per 32 values, waves on independent key tiles, packed f16 P.V; ~180
  GB/s vs ~55 GB/s); Ornith-1.5-35B-A3B MXFP4 decode with q4 KV at 128k 36.5 -> 58.5 tok/s, at 64k
  48.0 -> 67.0 tok/s (f16 KV: 32.7 / 46.4); KL vs f16 KV the same as the previous kernel (code 128k 0.0137 vs 0.0133, 64k 0.0177 vs 0.0183). `WHIRL_ATTN_DQ4=0` restores
  the previous kernel. precise / balance unchanged.
- fast `kvq4` on the R9700 (FAST-1c): the gfx1201 code object has the q4 KV kernels (the 8060S
  format: 4-bit K / V, one f16 scale per 32 values, Hadamard-rotated q / k): `attn_prep_q4` /
  `kv_store_q4`, `attn_dq4` for decode / verify / MTP rows (ported; blocks ordered so the verify
  rows of a sequence read the same K / V tiles together through L2; ~430 GB/s at 128k for the 27B
  24/4-head shape), `attn_split_q4` / `attn_wsplit1/2_q4` / `attn_decode_q4` /
  `attn_prefill_wmma_q4` as generic fallbacks, and prefill through the Q8P path (keys dequantized
  into f16 rows, f16 `attn_kg`). `--fast` on the R9700 now uses q4 KV. Swift-1.5 27B MXFP4-A,
  `whirl bench --fast`, q4 vs `WHIRL_KV=f16`: decode at a 131k prompt plain 25.08 → 30.98 tok/s
  (+23.5%), MTP 63.12 → 68.79 (per verify cycle only −2.5%: each MTP row reads K / V, the f16
  verify reads it once for 5 rows; a block shared by the rows measured no better, attn_dq4 is
  ALU-bound per row); at a 67k prompt plain 30.42 → 34.08 (+12%), MTP 92.88 → 92.75; prefill 8k
  −1.1%, 128k −1.9%. KL of the prompt logits vs f16 KV (128 rows each): code 128k 0.0049 (top-1
  99.2%), zh 128k 0.0071 (93.0%; the 9 differing rows are near ties, top-1 p ≤ 0.53), code 64k
  0.0100 (95.3%), zh 64k 0.0016 (95.3%). R9700 fast goldens re-recorded; precise / balance and every
  8060S golden unchanged.
- fast: a repeated request gives the same text streamed or not (FIX-FS). The server gate's
  "stream == non-stream zh_think" failed in fast mode (also with f16 KV): the stream request came
  second and resumed from the prompt's think-open checkpoint, and the resume ran the MTP row that
  pairs the checkpoint's last hidden with the next token as a one-row MTP forward, while the cold
  prefill had run that row inside the previous chunk's batch. One-row MTP forwards (q8
  activations, fused kernels) are not bit-identical to the same row in a larger batch, so the MTP
  KV and then the drafts differed; exact acceptance (precise / balance, fast with `WHIRL_RELAX=0`)
  never shows drafts in the text, `relaxacc` keeps them. Every MTP prefill (CLI and server, solo
  and grouped chunks) now runs that chunk-boundary row alone at the start of the next chunk, as a
  resume does, so a cold prefill and a resume from its own checkpoints are bit-identical. Cost:
  one one-row MTP forward per prefill chunk boundary. Text unchanged for precise / balance (exact
  acceptance) and for every golden (gfx1201 and gfx1151, fast included). A prompt resumed from
  another prompt's shared prefix is still prefilled in other chunks than cold and can give
  another fast text (documented, like concurrency). New server-gate suite `fast_stream` (fast,
  relaxacc forced on, also MoE): cold reference server without the prefix cache vs stream /
  non-stream repeats.
- q8 / q8h (and q4) prefill past the f16 scratch (FAST-1c): a prefill whose keys exceed what
  `ffn_g` / `ffn_u` hold (~139k at batch 4096 on the 27B) fell back to `attn_kg_q8` (+29% at 256k
  keys); now the keys go in ranges of at most that size, each dequantized (`kv_dq_rows_r`) and
  walked by `attn_kgs` with the softmax state (m, l, unnormalized output) carried from range to
  range — bit-identical to one f16 `attn_kg` over all keys, so q8h output is unchanged (it was
  already bit-identical to `attn_kg_q8`). Kernel bench, 4096 queries at 256k keys: f16 290.1 ms,
  ranges 294.5 ms (+1.5%), `attn_kg_q8` 373 ms. End to end, Ornith-1.5-9B balance q8h vs f16: 192k
  −1.4%, 256k −1.4% (same output hash); Swift-1.5 27B q8h 256k 1,129 tok/s (f16 KV at 256k does
  not fit 32 GB next to its weights). Prefills within the scratch are unchanged.

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
- **q8 / q8h KV prefill attention at f16 speed (Q8P)**, so balance can default to q8h on dense
  models: per layer and prefill chunk, `kv_dq_rows` dequantizes the sequence's keys once into f16 rows
  (in `ffn_g` / `ffn_u`, idle during attention; up to 139k keys at batch 4096) and the f16 `attn_kg`
  runs on them, bit-identical to `attn_kg_q8` (which converts K once per query head and was 29%
  slower than f16 at 128k keys). R9700, balance, q8h vs `WHIRL_KV=f16`: Swift-1.5 27B MXFP4-A prefill
  8k −1.5%, 32k −0.9%, 128k −1.0% (before −3.3 / −7.7 / −15.1%), decode_7p +1.7%; Qwen3.8-27B
  UD-Q4_K_M 8k −0.5%, 32k −0.4%, 128k −0.7%, decode_7p +0.4%. Radeon 8060S (unchanged kernels):
  Q4_K_M prefill 8k −0.8%, 64k −2.7%, decode at 64k −1.9%. `WHIRL_ATTN_DQF=0` restores the int8
  kernel. Goldens unchanged on gfx1201; gfx1151 dense balance re-recorded for the q8h default.
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
