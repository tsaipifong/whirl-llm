**English** | [繁體中文](guide/zh-TW/benchmarks.md)

# WHIRL release benchmark (v0.1.0) — WHIRL vs llama.cpp on the Radeon AI PRO R9700

Every WHIRL number on this page was measured with the C++ `whirl.exe` / `whirl serve` built from the
current repository (release candidate of v0.1.0) and cross-checked against the release executable
(§11). llama.cpp is build **b11214** (ROCm, commit `2ebd9ae62`), run with the fastest flags we found
for each row (§2.3). Both engines ran the same prompts, the same context lengths and greedy decoding,
one model process on the GPU at a time.

**Where WHIRL does not lead (read this first):**

- **Short prompts on Q4_K_M:** prefill of an 88-token prompt is at parity (1.03×).
- **Plain decoding without speculation** is memory-bandwidth bound in both engines; WHIRL's lead there
  is small on the dense models (1.10–1.18×). Most of WHIRL's decode lead comes from MTP + n-gram
  speculative decoding.
- **Decode after a 16k context, Swift MXFP4-A:** WHIRL MTP+n-gram is only 1.09× llama.cpp's best (MTP with a q8_0 V cache), and on
  this model WHIRL's MTP+n-gram is *slower* than its own MTP-only mode (69.5 vs 72.4 tok/s).
- **VRAM:** with the same context, WHIRL's CLI uses 3–5 GiB *more* VRAM than llama-bench on the dense
  models (and 1.4 GiB more on the MoE model at 128k) (larger prefill buffers, the MTP block and the draft head). The WHIRL server deliberately
  fills free VRAM with its KV pool (§9).
- **Cold prefill TTFT on Q4_K_M** is only 1.5× faster (vs 2.5× on the MXFP4 models).

## 1. Test environment

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700 (gfx1201, RDNA 4, 32 GB GDDR6), **connected as a USB4 eGPU** |
| Host | ASUS ROG Flow Z13 (GZ302EA): AMD Ryzen AI MAX+ 395 (16 cores / 32 threads), 128 GB LPDDR5X-8000 of which 63.6 GB visible to Windows (the rest is reserved for the integrated Radeon 8060S, which was idle) |
| OS | Windows 11 Home, build 26300 |
| GPU driver | AMD Software Adrenalin 26.8.1, driver 32.0.31041.1004 |
| WHIRL build | HIP SDK 7.2 (device code), MSVC 19.44 (host), CMake + Ninja, `-DWHIRL_GPU_ARCHS=gfx1201`, Release; `whirl.exe` SHA-256 `13701a20edc3953b48288f7ad5b721eb1549e059f948fa18c00a15a73843f79a` |
| Release cross-check | `whirl.exe` 0.1.0 (static CRT build) SHA-256 `a9fa53323d20f2ac49cb3c8882c75944ab871b54375699455f20cf19504e6366`, `whirl-server.exe` `9a15e49173e1e2e69df59d9ff20e958ae0e174d1f043faf8783a8173ab332326` (§11) |
| llama.cpp | b11214 ROCm Windows binaries (`llama-bench`, `llama-server`, `llama-mtmd-cli`), `HIP_VISIBLE_DEVICES=1`, `GGML_CUDA_NO_PINNED=1` |
| Measured | 2026-10-03, 01:45–08:55 local time |

**eGPU note (environment limitation, measured once, not optimized for):** the R9700 sits behind USB4
(~3.8 GB/s host link each way). This affects model load time, host-RAM/SSD KV restores and vision weight
streaming in both engines; it does not affect prefill or decode, which stay in VRAM. Direct PCIe
systems should see faster restores than §8 shows.

### Models

| Short name | File | Type | Quantization |
|---|---|---|---|
| Ornith MXFP4 (MoE) | `Ornith-1.5-35B-A3B-MXFP4.gguf` (18.4 GiB) | MoE, 35B total, ~3B active per token, MTP head | MXFP4 experts (WHIRL's own release quant) |
| Swift MXFP4-A (dense) | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` (14.7 GiB) | dense 27B, MTP head | MXFP4, output head Q6_K (variant A, our published and recommended quant) |
| Qwen3.8-27B Q4_K_M (dense) | `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.3 GiB, unsloth) | dense 27B, MTP head | standard Q4_K_M (unsloth UD) |

## 2. Methodology

### 2.1 Rules

- **One model process on the GPU at a time**, every run under the machine's R9700 lock; dedicated VRAM
  only (a background sampler recorded dedicated/shared GPU memory of every engine process every 2 s).
  Shared usage stayed ≤ 314 MiB for every llama.cpp and WHIRL CLI process (whirl-vis: ~0.9 GiB, the declared pinned projector weights). WHIRL server processes show
  8–16 GiB "shared" — that is the declared pinned host-RAM KV tier (`--kv-ram-mb`), not spilled VRAM.
- **Greedy decoding in both engines.** Server requests use the greedy-equivalent sampler
  `top_k = 1, temperature 1.0, top_p 1.0, min_p 0, seed 42` (the same sampler path in both servers),
  `cache_prompt = false` except in the cache tests. WHIRL's speculative outputs are identical to its
  plain greedy output (verified in every round: 0 of 210 outputs differed between rounds).
- **Repeats:** whirl bench 3 process runs (interleaved with llama-bench), server scenarios 3 rounds
  (file editing, concurrency, cache: 2 rounds), interleaved WHIRL/llama.cpp, forward then reversed order.
  Tables show the **median**; min–max spreads are in the raw data (most are < 1%).
- **No measured run includes WHIRL's first-run GEMM autotune** (which costs ~99 s on 27B Q4_K_M and
  ~3 s on Ornith MXFP4 the first time a model file is used). Every bench log shows
  `prefill GEMM tune: cached`; server timings start after the warm-up request.
- **Ratio column = WHIRL / llama.cpp**, where "llama.cpp" is its **fastest configuration for that row**
  (the maximum over plain, MTP and MTP + n-gram, and over the flag sets tried); the config is named.
  For TTFT the ratio is llama.cpp time / WHIRL time.
- "**no MTP**" marks plain decoding without any speculation. "N/A" = llama.cpp has no equivalent.

### 2.2 What each test is

| Test | Tool | Details |
|---|---|---|
| Prefill (CLI) | `whirl bench` vs `llama-bench` | 88 / 2,048 / 8,192 / 32,768 / 65,536 / 131,072 tokens, KV f16 in both. WHIRL: mixed zh/en coding text, MTP block run over the prompt as chat does, best of 2 after a warm-up (≤ 8k) / one timed run after a warm-up (32k) / one run (64k, 128k), median of 3 processes for ≤ 32k. llama-bench: its random-token prompt, median of 3 repetitions after its warm-up (1 for 128k, 2 for 64k) |
| Prefill (server, same text) | both servers, one client | fixed real source-code files of 2,022 / 8,178 / 32,751 tokens (`max_tokens 1`), plain mode, 3 rounds |
| Main decode scenario | both servers, one client | the 7 `bench_zhcode` prompts (Chinese question; answer in Chinese with English code: Python LRU cache + pytest, Spring Boot API, React hook, PostgreSQL report, fixing Python with Chinese identifiers, Node.js refactor, reading 4k tokens of source), thinking off, `max_tokens 800`; tok/s = `timings.predicted_per_second`, mean over the 7 prompts per round, median of 3 rounds |
| File-editing scenario | both servers, one client | 5 prompts with 1.6–2.3k-token source files (rename, refactor, translate comments, one CRLF file), output the whole file, `max_tokens 2500`, 2 rounds |
| Decode after 16k | both servers + CLI | 128 tokens (`ignore_eos`) after a 16,354-token code prompt; CLI: `whirl bench --prompt @file` vs `llama-bench -d 16384 -n 128` |
| Concurrency | both servers, one client | 4 slots × 4,096 context; C = 1 / 2 / 4 simultaneous requests, distinct ~1.1k-token prompts, 256 tokens each (`ignore_eos`); aggregate = generated tokens / wall time (includes prefill) |
| Cached prefix / restore | both servers, one client | TTFT = wall time of a `max_tokens 1` request. See §8 |
| Vision | `whirl-vis vis-encode` vs `llama-mtmd-cli` | Swift mmproj F16, 7 images; see §10 |
| VRAM | Windows GPU process counters | max dedicated usage per run |

### 2.3 llama.cpp flags (fastest found)

- Common: `-ngl 99 -fa on`, KV cache f16 (= WHIRL's CLI default). For the server dense rows we also ran
  `-ctk f16 -ctv q8_0` (closest to WHIRL server's default `q8v` KV on dense models); the faster of the
  two is used per row and marked (q8_0 V was slower in every row except Swift's MTP decode after 16k, 63.5 vs 62.8 tok/s).
- **Micro-batch sweep** (`llama-bench`, `-ub 512 / 1024 / 2048 / 4096`, `-b 2048` or `-b 4096`): the
  best value per prompt length is used in the prefill table (flags listed per row). Server runs use
  `-ub 2048` (Ornith) / `-ub 1024` (dense), the best at 8k–32k.
- **Speculative decoding probes** on the zh prompt set: MTP `--spec-draft-n-max` 1/2/3 (Ornith) and
  2/3/4 (Swift), plus MTP + n-gram (`draft-mtp,ngram-simple` defaults, `ngram-simple` with n=3/m=15, and
  `draft-mtp,ngram-mod`). Best: Ornith `draft-mtp n-max 2` (107.8 tok/s), `draft-mtp,ngram-mod n-max 1`
  (104.3); Swift `draft-mtp n-max 4` (60.6), `draft-mtp,ngram-mod n-max 3` (59.1). The dense Q4_K_M
  uses the Swift settings. All with `--spec-draft-n-min 0 --spec-draft-p-min 0.3`.
- Server: `llama-server -c 131072 -np 1 --jinja --load-mode none` (concurrency: `-np 4 -c 16384`;
  cache tests add `--cache-ram 16384`).

WHIRL runs with defaults: `whirl serve MODEL` (4 slots, automatic KV format and pool size); modes are
selected with `WHIRL_MTP=0 WHIRL_NGRAM=0` (no MTP) and `WHIRL_NGRAM=0` (MTP only). Cache tests add
`--kv-ram-mb 16384` (same RAM budget as llama.cpp) and the eviction test `--ctx 65536`.

## 3. Prefill

![Prefill vs prompt length](images/bench_prefill.png)

### 3.1 Each engine's own bench tool (KV f16)


**Ornith-1.5-35B-A3B MXFP4** — MoE, 35B total / ~3B active, MXFP4 (experts) — WHIRL release quant

| Prompt tokens | WHIRL tok/s | llama.cpp tok/s | llama.cpp flags | WHIRL / llama.cpp |
|---|---|---|---|---|
| 88 | 2,546 | 2,115 | `-ub 512 -b 2048` | **1.20×** |
| 2,048 | 11,705 | 4,876 | `-ub 4096 -b 4096` | **2.40×** |
| 8,192 | 10,858 | 4,637 | `-ub 2048 -b 2048` | **2.34×** |
| 32,768 | 7,978 | 3,778 | `-ub 2048 -b 2048` | **2.11×** |
| 65,536 | 5,815 | 3,069 | `-ub 2048 -b 2048` | **1.89×** |
| 131,072 | 3,782 | 2,239 | `-ub 2048 -b 2048` | **1.69×** |

**Swift-1.5-Qwen3.8-27B MXFP4-A** — dense 27B, MXFP4, output Q6_K (variant A)

| Prompt tokens | WHIRL tok/s | llama.cpp tok/s | llama.cpp flags | WHIRL / llama.cpp |
|---|---|---|---|---|
| 88 | 1,515 | 928.6 | `-ub 512 -b 2048` | **1.63×** |
| 2,048 | 3,509 | 1,385 | `-ub 1024 -b 2048` | **2.53×** |
| 8,192 | 3,278 | 1,338 | `-ub 1024 -b 2048` | **2.45×** |
| 32,768 | 2,595 | 1,174 | `-ub 1024 -b 2048` | **2.21×** |
| 65,536 | 2,017 | 1,002 | `-ub 1024 -b 2048` | **2.01×** |
| 131,072 | 1,405 | 791.3 | `-ub 1024 -b 2048` | **1.78×** |

**Qwen3.8-27B UD-Q4_K_M** — dense 27B, unsloth UD-Q4_K_M

| Prompt tokens | WHIRL tok/s | llama.cpp tok/s | llama.cpp flags | WHIRL / llama.cpp |
|---|---|---|---|---|
| 88 | 866.2 | 840.8 | `-ub 1024 -b 2048` | **1.03×** |
| 2,048 | 1,673 | 1,276 | `-ub 4096 -b 4096` | **1.31×** |
| 8,192 | 1,689 | 1,223 | `-ub 1024 -b 2048` | **1.38×** |
| 32,768 | 1,479 | 1,086 | `-ub 1024 -b 2048` | **1.36×** |
| 65,536 | 1,270 | 939.8 | `-ub 1024 -b 2048` | **1.35×** |
| 131,072 | 997.9 | 752.0 | `-ub 1024 -b 2048` | **1.33×** |

### 3.2 Same text through both servers (plain mode)

| Model | Input | WHIRL server tok/s | llama-server tok/s | WHIRL / llama.cpp |
|---|---|---|---|---|
| Ornith MXFP4 (MoE) | 2,022 | 9,972 | 4,366 | **2.28×** |
| Ornith MXFP4 (MoE) | 8,178 | 11,116 | 4,507 | **2.47×** |
| Ornith MXFP4 (MoE) | 32,751 | 8,228 | 3,834 | **2.15×** |
| Swift MXFP4-A (dense) | 2,022 | 3,174 | 1,211 | **2.62×** |
| Swift MXFP4-A (dense) | 8,178 | 3,344 | 1,288 | **2.60×** |
| Swift MXFP4-A (dense) | 32,751 | 2,594 | 1,157 | **2.24×** |
| Qwen3.8-27B Q4_K_M (dense) | 2,022 | 1,571 | 1,115 | **1.41×** |
| Qwen3.8-27B Q4_K_M (dense) | 8,178 | 1,616 | 1,186 | **1.36×** |
| Qwen3.8-27B Q4_K_M (dense) | 32,751 | 1,420 | 1,075 | **1.32×** |

The server numbers include tokenization and HTTP; llama-server is ~10–13% below llama-bench on the dense
models with the same `-ub`, WHIRL's server ~6–15% below its CLI.

## 4. Decode — main scenario: Chinese question, Chinese explanation, English code

![Decode, zh coding](images/bench_decode_zh.png)

| Model | WHIRL MTP+n-gram (default) | WHIRL MTP | WHIRL **no MTP** | llama.cpp **plain (no MTP)** | llama.cpp MTP | llama.cpp MTP+n-gram | llama.cpp fastest (config) | WHIRL default / llama.cpp fastest | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | **244.5** | 241.8 | 166.7 | 118.8 | 109.3 | 104.9 | 118.8 (plain) | **2.06×** | **1.40×** |
| Swift MXFP4-A (dense) | **107.5** | 107.3 | 37.5 | 33.4 | 60.8 | 59.5 | 60.8 (draft-mtp n-max 4) | **1.77×** | **1.13×** |
| Qwen3.8-27B Q4_K_M (dense) | **98.5** | 98.4 | 34.4 | 30.9 | 55.2 | 56.0 | 56.0 (draft-mtp+ngram-mod n-max 3) | **1.76×** | **1.11×** |

Every cell is the mean over 7 prompts × 800 tokens, median of 3 rounds (round-to-round spread ≤ 1.5%).
On the MoE model llama.cpp's MTP is slower than its plain decoding, so the comparison is against plain.
WHIRL's n-gram drafts add little on these generation prompts (they matter for editing, §5).

### 4.1 Short-context decode, CLI (each engine's bench tool)

| Model | WHIRL MTP+n-gram | WHIRL MTP | WHIRL **no MTP** | llama-bench tg256 (**no MTP**) | WHIRL no MTP / llama.cpp | WHIRL no MTP after 16k (CLI) | llama-bench tg128 @ depth 16k (no MTP) | ratio |
|---|---|---|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | 310.9 | 265.2 | 193.6 | 119.7 | **1.62×** | 169.9 | 112.7 | **1.51×** |
| Swift MXFP4-A (dense) | 89.3 | 89.0 | 39.4 | 33.4 | **1.18×** | 36.5 | 31.9 | **1.14×** |
| Qwen3.8-27B Q4_K_M (dense) | 103.7 | 89.9 | 35.9 | 31.3 | **1.15×** | 33.5 | 29.8 | **1.12×** |

`whirl bench` decodes 256 tokens after a 144-token coding prompt; llama-bench `tg256` has no MTP mode,
so only the no-MTP columns are directly comparable here.

## 5. Decode — file-editing scenario

![Decode, file editing](images/bench_decode_edit.png)

| Model | WHIRL MTP+n-gram (default) | WHIRL MTP | WHIRL **no MTP** | llama.cpp **plain (no MTP)** | llama.cpp MTP | llama.cpp MTP+n-gram | llama.cpp fastest (config) | WHIRL default / llama.cpp fastest | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | **618.1** | 260.4 | 163.3 | 117.1 | 127.0 | 239.3 | 239.3 (draft-mtp+ngram-mod n-max 1) | **2.58×** | **1.40×** |
| Swift MXFP4-A (dense) | **328.3** | 181.1 | 37.0 | 33.0 | 79.5 | 144.8 | 144.8 (draft-mtp+ngram-mod n-max 3) | **2.27×** | **1.12×** |
| Qwen3.8-27B Q4_K_M (dense) | **304.9** | 173.5 | 34.0 | 30.7 | 70.8 | 136.8 | 136.8 (draft-mtp+ngram-mod n-max 3) | **2.23×** | **1.11×** |

Editing prompts repeat most of the input in the output, which n-gram drafts predict well. Earlier
published Swift numbers (157.2 tok/s "MTP + n-gram") were the average of 19 prompts mixing this editing
set with the zh coding set (thinking on and off); the zh think-off group alone was 106.0 then and is
107.5 now — the two scenarios are reported separately here so the numbers do not contradict each other.

## 6. Decode after a 16k-token context

![Decode after 16k](images/bench_decode_16k.png)

| Model | WHIRL MTP+n-gram (default) | WHIRL MTP | WHIRL **no MTP** | llama.cpp **plain (no MTP)** | llama.cpp MTP | llama.cpp MTP+n-gram | llama.cpp fastest (config) | WHIRL default / llama.cpp fastest | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | **301.5** | 232.8 | 150.5 | 107.5 | 104.2 | 89.4 | 107.5 (plain) | **2.80×** | **1.40×** |
| Swift MXFP4-A (dense) | **69.5** | 72.4 | 35.2 | 31.4 | 63.5 | 60.6 | 63.5 (draft-mtp n-max 4, V cache q8_0) | **1.09×** | **1.12×** |
| Qwen3.8-27B Q4_K_M (dense) | **74.3** | 74.6 | 32.4 | 29.4 | 57.0 | 59.8 | 59.8 (draft-mtp+ngram-mod n-max 3) | **1.24×** | **1.10×** |

## 7. Server concurrency

![Server concurrency](images/bench_server_concurrency.png)

| Model | C | WHIRL MTP+n-gram (default) | WHIRL **no MTP** | llama-server plain (**no MTP**) | llama-server MTP | llama-server MTP+n-gram | WHIRL default / llama.cpp fastest | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | 1 | **205.3** | 154.8 | 100.8 | 75.5 | 76.4 | **2.04×** (plain) | **1.54×** |
| Ornith MXFP4 (MoE) | 2 | **306.3** | 238.5 | 144.4 | 100.2 | 113.0 | **2.12×** (plain) | **1.65×** |
| Ornith MXFP4 (MoE) | 4 | **396.7** | 324.9 | 191.9 | 104.1 | 117.2 | **2.07×** (plain) | **1.69×** |
| Swift MXFP4-A (dense) | 1 | **90.4** | 35.8 | 28.7 | 41.0 | 39.4 | **2.21×** (mtp) | **1.25×** |
| Swift MXFP4-A (dense) | 2 | **148.7** | 64.2 | 46.7 | 48.6 | 43.1 | **3.06×** (mtp) | **1.37×** |
| Swift MXFP4-A (dense) | 4 | **198.6** | 108.6 | 65.8 | 51.7 | 55.4 | **3.02×** (plain) | **1.65×** |
| Qwen3.8-27B Q4_K_M (dense) | 1 | **60.6** | 31.2 | 26.7 | 38.2 | 45.8 | **1.32×** (mtp+ngram) | **1.17×** |
| Qwen3.8-27B Q4_K_M (dense) | 2 | **94.6** | 54.0 | 43.6 | 50.1 | 49.7 | **1.89×** (mtp) | **1.24×** |
| Qwen3.8-27B Q4_K_M (dense) | 4 | **134.1** | 86.8 | 58.5 | 50.3 | 45.8 | **2.29×** (plain) | **1.48×** |

Aggregate = all generated tokens / wall-clock time, prefill included. llama.cpp's speculative modes do
not scale with concurrent users in b11214; WHIRL keeps MTP + n-gram on for every slot (batched verify).

## 8. Cold prefill, cached prefixes and KV restore (TTFT)

![TTFT with cached prefixes](images/bench_ttft_cache.png)

| Model | Request | WHIRL TTFT (s) | llama-server TTFT (s) | llama.cpp / WHIRL (TTFT, lower is better for WHIRL) |
|---|---|---|---|---|
| Ornith MXFP4 (MoE) | System prompt ~12.2k, cold | 1.314 | 3.396 | **2.6×** |
| Ornith MXFP4 (MoE) | New conversation, same ~12.2k system prompt (warm) | 0.064 | 0.259 | **4.1×** |
| Ornith MXFP4 (MoE) | System prompt ~26.4k, cold | 3.294 | 8.151 | **2.5×** |
| Ornith MXFP4 (MoE) | New conversation, same ~26.4k system prompt (warm) | 0.073 | 0.290 | **4.0×** |
| Ornith MXFP4 (MoE) | Evicted ~27.4k session, next turn (restored from host RAM) | 0.265 | 0.950 | **3.6×** |
| Ornith MXFP4 (MoE) | ~25.7k session after server restart (restored from SSD) | 0.298 | N/A | N/A (no equivalent feature) |
| Swift MXFP4-A (dense) | System prompt ~12.2k, cold | 4.091 | 10.650 | **2.6×** |
| Swift MXFP4-A (dense) | New conversation, same ~12.2k system prompt (warm) | 0.102 | 0.446 | **4.4×** |
| Swift MXFP4-A (dense) | System prompt ~26.4k, cold | 10.104 | 24.803 | **2.5×** |
| Swift MXFP4-A (dense) | New conversation, same ~26.4k system prompt (warm) | 0.122 | 0.538 | **4.4×** |
| Swift MXFP4-A (dense) | Evicted ~27.4k session, next turn (restored from host RAM) | 0.668 | 2.195 | **3.3×** |
| Swift MXFP4-A (dense) | ~25.7k session after server restart (restored from SSD) | 0.729 | N/A | N/A (no equivalent feature) |
| Qwen3.8-27B Q4_K_M (dense) | System prompt ~12.2k, cold | 7.837 | 11.631 | **1.5×** |
| Qwen3.8-27B Q4_K_M (dense) | New conversation, same ~12.2k system prompt (warm) | 0.101 | 0.437 | **4.3×** |
| Qwen3.8-27B Q4_K_M (dense) | System prompt ~26.4k, cold | 18.316 | 26.875 | **1.5×** |
| Qwen3.8-27B Q4_K_M (dense) | New conversation, same ~26.4k system prompt (warm) | 0.136 | 0.546 | **4.0×** |
| Qwen3.8-27B Q4_K_M (dense) | Evicted ~27.4k session, next turn (restored from host RAM) | 0.719 | 2.178 | **3.0×** |
| Qwen3.8-27B Q4_K_M (dense) | ~25.7k session after server restart (restored from SSD) | 0.739 | N/A | N/A (no equivalent feature) |

- *Warm*: a new conversation whose system prompt (12.2k / 26.4k tokens) was sent before — WHIRL reuses
  its system-prompt checkpoint, llama-server reuses the slot's cached prefix.
- *RAM restore*: three ~27k-token sessions through a VRAM pool that holds two (WHIRL `--ctx 65536`;
  llama-server one slot), then the first session's next turn: WHIRL restores it from its pinned-RAM tier,
  llama-server from its `--cache-ram` prompt cache. Both are bounded by the USB4 link here.
- *SSD restore*: the server is restarted (graceful Ctrl+Break) and another session's next turn is
  restored from WHIRL's SSD tier. llama.cpp has no automatic persistent KV cache (its `--slot-save-path`
  needs explicit save/restore API calls), so this row is N/A.
- WHIRL's default runs the MTP block over the prompt too, which is included in its cold TTFT.

## 9. VRAM

| Model | Setting | WHIRL dedicated VRAM (GiB) | llama.cpp dedicated VRAM (GiB) | Notes |
|---|---|---|---|---|
| Ornith MXFP4 (MoE) | CLI bench, context ~33k, KV f16 | 22.5 | 22.5 | llama-bench `-ub 2048` |
| Ornith MXFP4 (MoE) | CLI bench, context ~131k, KV f16 | 23.5 | 22.1 | llama-bench `-ub 2048` |
| Ornith MXFP4 (MoE) | server, default settings | 30.6 | 23.0 | WHIRL sizes its KV pool to fill free VRAM (414,464 tokens, f16); llama-server `-c 131072 -np 1`, KV f16, MTP on |
| Swift MXFP4-A (dense) | CLI bench, context ~33k, KV f16 | 23.0 | 17.8 | llama-bench `-ub 1024` |
| Swift MXFP4-A (dense) | CLI bench, context ~131k, KV f16 | 26.1 | 22.7 | llama-bench `-ub 1024` |
| Swift MXFP4-A (dense) | server, default settings | 31.3 | 24.0 | WHIRL sizes its KV pool to fill free VRAM (210,688 tokens, q8v); llama-server `-c 131072 -np 1`, KV f16, MTP on |
| Qwen3.8-27B Q4_K_M (dense) | CLI bench, context ~33k, KV f16 | 23.6 | 19.1 | llama-bench `-ub 1024` |
| Qwen3.8-27B Q4_K_M (dense) | CLI bench, context ~131k, KV f16 | 27.1 | 24.0 | llama-bench `-ub 1024` |
| Qwen3.8-27B Q4_K_M (dense) | server, default settings | 31.4 | 25.2 | WHIRL sizes its KV pool to fill free VRAM (201,216 tokens, q8v); llama-server `-c 131072 -np 1`, KV f16, MTP on |

WHIRL's server sizes its KV pool from the free VRAM at startup (dense models fall back to `q8v` KV so
that one 128k request plus a 64k second one fit), so its process always shows ~30–31 GiB; the useful
comparison for footprint is the CLI rows at equal context.

## 10. Vision (Swift MXFP4-A with `mmproj` F16)

| Image (resized, tokens) | WHIRL warm, weights resident (ms) | WHIRL warm, weights streamed (ms) | llama.cpp mtmd warm (ms) | llama.cpp / WHIRL resident | WHIRL first encode incl. weight upload (ms) | llama.cpp first encode in process (ms) |
|---|---|---|---|---|---|---|
| shapes (640x352, 220) | 16.8 | 257.4 | 58 | **3.5×** | 302 | 554 |
| dialog (800x448, 350) | 23.5 | 258.5 | 90 | **3.9×** | 305 | 561 |
| photo (1024x672, 672) | 51.7 | 262.8 | 208 | **4.0×** | 333 | 694 |
| code (1280x736, 920) | 77.9 | 264.9 | 342 | **4.4×** | 359 | 820 |
| s512 (512x512, 256) | 17.8 | 257.5 | 65 | **3.7×** | 303 | 526 |
| s1024 (1024x1024, 1024) | 91.0 | 266.1 | 406 | **4.5×** | 372 | 881 |
| s1080 (1920x1088, 2040) | 253.7 | 289.3 | 1,330 | **5.2×** | 535 | 1,828 |

*Warm*: WHIRL — median of encodes 2–5 in one process; llama.cpp — encodes 2–3 when three copies of the
image are passed to one `llama-mtmd-cli` run (it encodes images one by one). *Streamed*: WHIRL's default
when VRAM is tight (the 27B server default) uploads the 0.9 GB projector per image over USB4 (~250 ms
on this eGPU). *First encode*: WHIRL includes uploading the projector weights; llama.cpp's weights are
already resident but its first encode in a process includes one-time setup. Both engines produce the
same token count per image (e.g. 220 for `shapes`).

Encoder accuracy, WHIRL vs an independent f32 numpy implementation of the encoder (same preprocessed
pixels):

| Image | tokens | relative L2 vs f32 numpy reference | minimum per-token cosine |
|---|---|---|---|
| shapes | 220 | 0.16% | 0.99995 |
| code | 920 | 0.25% | 0.99852 |

A previous measurement with the same projector (Qwen3.8 `mmproj-F16`, same weights id
`3bd65e3060dbce70`) found llama.cpp's encoder at 1.6–12% relative L2 / minimum cosine 0.89–0.998 against
the same reference ([vision.md](guide/en/vision.md#3-numerics)); it was not re-measured here.

## 11. Cross-check with the release executable

The tables above were measured with a development build of the same source tree ("dev build" below).
The release executables (`whirl.exe` / `whirl-server.exe` 0.1.0, static CRT, SHA-256 in §1) were run
interleaved with it, 3 rounds:

| Model | build | prefill 2,048 | prefill 8,192 | decode MTP+n-gram | decode **no MTP** |
|---|---|---|---|---|---|
| Ornith MXFP4 (MoE) | dev build | 11,732 | 10,839 | 308.7 | 192.3 |
| Ornith MXFP4 (MoE) | release 0.1.0 | 11,727 | 10,838 | 309.9 | 191.6 |
| Swift MXFP4-A (dense) | dev build | 3,505 | 3,297 | 89.5 | 39.4 |
| Swift MXFP4-A (dense) | release 0.1.0 | 3,506 | 3,295 | 89.6 | 39.4 |
| Qwen3.8-27B Q4_K_M (dense) | dev build | 1,675 | 1,686 | 103.5 | 35.9 |
| Qwen3.8-27B Q4_K_M (dense) | release 0.1.0 | 1,677 | 1,690 | 103.7 | 35.9 |

Server, Ornith MXFP4 zh scenario, `whirl-server.exe` 0.1.0 (MTP + n-gram, 2 rounds): 243.8 tok/s
(dev build: 244.5 tok/s).

## 12. Exact commands

```bat
:: WHIRL CLI (prefill sweep + decode modes)
whirl bench MODEL.gguf --prefill 88,2048,8192,32768 --decode 256 --modes mtp-ngram,mtp,plain
whirl bench MODEL.gguf --prefill 131072 --decode 16 --modes plain
whirl bench MODEL.gguf --prefill 88 --decode 128 --modes mtp-ngram,mtp,plain --no-think --prompt @sweep_tg16k.txt

:: llama.cpp CLI
set HIP_VISIBLE_DEVICES=1& set GGML_CUDA_NO_PINNED=1
llama-bench -m MODEL.gguf -ngl 99 -fa on -p 88,2048,8192,32768 -n 256 -ub 512|1024|2048 -b 2048 -r 3 -o jsonl
llama-bench -m MODEL.gguf -ngl 99 -fa on -p 2048,8192,32768 -n 0 -ub 4096 -b 4096 -r 3 -o jsonl
llama-bench -m MODEL.gguf -ngl 99 -fa on -p 131072 -n 0 -ub UB -b 2048 -r 1 -o jsonl
llama-bench -m MODEL.gguf -ngl 99 -fa on -p 0 -n 128 -d 16384 -ub UB -r 3 -o jsonl

:: servers
whirl serve MODEL.gguf --port 1236 --kv-ssd-dir DIR            (no MTP: set WHIRL_MTP=0 and WHIRL_NGRAM=0; MTP only: WHIRL_NGRAM=0)
llama-server -m MODEL.gguf --device ROCm0 -ngl 99 -c 131072 -np 1 -fa on -ub UB -b 2048 --port 1236 --no-webui --load-mode none --jinja ^
    [--spec-type draft-mtp --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    [--spec-type draft-mtp,ngram-mod --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]

:: vision
whirl-vis vis-encode mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf IMAGE OUT.f32 --reps 5 --mode resident|stream
llama-mtmd-cli -m Swift-...-MXFP4-A-outQ6_K.gguf --mmproj mmproj-...-F16.gguf --image IMG --image IMG --image IMG -ngl 99 -c 16384 -fa on --temp 0 -n 1 -p "Describe these images."
```

The benchmark harness scripts are not part of this repository (the repository is pure C++). The
harness is a small HTTP client that sends the prompts above to each server, records the per-request
timings as JSONL and samples VRAM once per second; its raw logs, per-request JSONL and VRAM samples
stay on the measurement machine.

## 13. README summary table

| R9700, greedy | Ornith MXFP4 (MoE) | Swift MXFP4-A (dense 27B) | Qwen3.8-27B Q4_K_M (dense) |
|---|---|---|---|
| Prefill 8k tok/s (WHIRL vs llama.cpp) | 10,858 vs 4,637 (2.34×) | 3,278 vs 1,338 (2.45×) | 1,689 vs 1,223 (1.38×) |
| Prefill 32k tok/s | 7,978 vs 3,778 (2.11×) | 2,595 vs 1,174 (2.21×) | 1,479 vs 1,086 (1.36×) |
| Decode, zh coding, WHIRL MTP+n-gram vs llama.cpp fastest | 244.5 vs 118.8 (2.06×) | 107.5 vs 60.8 (1.77×) | 98.5 vs 56.0 (1.76×) |
| Decode, zh coding, **no MTP** vs llama.cpp plain | 166.7 vs 118.8 (1.40×) | 37.5 vs 33.4 (1.13×) | 34.4 vs 30.9 (1.11×) |
| Server, 4 concurrent users, aggregate tok/s | 396.7 vs 191.9 (2.07×) | 198.6 vs 65.8 (3.02×) | 134.1 vs 58.5 (2.29×) |

WHIRL v0.1.0 vs llama.cpp b11214, R9700 (USB4 eGPU), greedy, same prompts. Decode: 7 Chinese coding
prompts, 800 tokens, median of 3 rounds; "fastest" = llama.cpp's best of plain / MTP / MTP + n-gram for
that model. Full tables and methodology: this page.
