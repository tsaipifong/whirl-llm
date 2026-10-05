**English** | [繁體中文](guide/zh-TW/benchmarks.md)

# WHIRL release benchmark (v0.1.3) — WHIRL vs llama.cpp on the Radeon AI PRO R9700

Every WHIRL number on this page was measured with the v0.1.3 release package (`whirl.exe` /
`whirl-server.exe`, release candidate 1, whose executables are byte-identical to the final release; SHA-256 in §1). The v0.1.0 (and, for the agent rows, v0.1.2)
numbers in the version tables were measured again on the same day with those release packages, in the
same harness, so the version comparison is like for like. llama.cpp is build **b11214** (ROCm, commit
`2ebd9ae62`), the same build as in the v0.1.0 benchmark. Both engines ran the same prompts, the same
context lengths and greedy decoding, one model process on the GPU at a time.

**Where WHIRL does not lead by much (read this first):**

- **Plain decoding without speculation** is memory-bandwidth bound in both engines; on the dense models
  WHIRL's lead there is small (1.13–1.14×). Most of WHIRL's decode lead comes from MTP + n-gram
  speculative decoding.
- **Prefill on Qwen3.8 Q4_K_M** leads by less than on the MXFP4 files: 1.51× at 96k and 1.44× at 256k
  tokens, against 2.28× and 1.74× for Swift-1.5 MXFP4. A cold 26k-token system prompt on Q4_K_M is
  only 1.34× faster (§8).
- **Decode after a 16k-token context on the dense models** is only 1.17× (Swift-1.5) and 1.33× (Qwen3.8
  Q4_K_M) llama.cpp's best (§6).
- **Speculative decoding in the worst case** (new writing after a 128k-token document, nothing to
  copy) gains only 1.56× over plain decoding (§5).

## 1. Test environment

| | |
|---|---|
| GPU | AMD Radeon AI PRO R9700 (gfx1201, RDNA 4, 32 GB GDDR6), **connected as a USB4 eGPU** |
| Host | ASUS ROG Flow Z13 (GZ302EA): AMD Ryzen AI MAX+ 395 (16 cores / 32 threads), 128 GB LPDDR5X-8000 of which 63.6 GB visible to Windows |
| OS / GPU driver | Windows 11 Home 26H2 (build 26300.9457); AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) |
| WHIRL | `whirl-0.1.3-windows-x64` release candidate 1 (executables byte-identical to the final release); `whirl.exe` SHA-256 `af7201bcbfab8257c53837b2cecb3d67e191802c1d0b4bf7716ab04844f907af`, `whirl-server.exe` `8c2feba05929496fa81d30c6526e637930ac6f49aa5de7ae5293dad7b7c9e519` |
| Older WHIRL | release packages 0.1.0 and 0.1.2 (same machine, same harness, same day) |
| llama.cpp | b11214 ROCm Windows binaries (`llama-server`, `llama-bench`), `HIP_VISIBLE_DEVICES=1`, `GGML_CUDA_NO_PINNED=1` |
| Measured | 2026-10-05 |

**eGPU note:** the R9700 sits behind USB4 (~3.8 GB/s host link each way). This affects model load time
and host-RAM / SSD KV restores; it does not affect prefill or decode, which stay in VRAM. Direct PCIe
systems should see faster restores than §8 shows.

### Models

| Short name | File | Type | Quantization |
|---|---|---|---|
| Swift MXFP4-A (dense) | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` (14.7 GiB) | dense 27B, MTP head | MXFP4, output head Q6_K (our published and recommended quant) |
| Ornith MXFP4 (MoE) | `Ornith-1.5-35B-A3B-MXFP4.gguf` (18.4 GiB) | MoE, 35B total, ~3B active per token, MTP head | MXFP4 experts (WHIRL's own release quant) |
| Qwen3.8-27B Q4_K_M (dense, reference) | `Qwen3.8-27B-UD-Q4_K_M.gguf` (15.3 GiB, unsloth) | dense 27B, MTP head | standard Q4_K_M (unsloth UD) |

Swift and Ornith are the main comparison; Qwen3.8 Q4_K_M is listed for reference.

## 2. Methodology

### 2.1 Rules

- **One measurement job at a time.** Each job checks that the R9700 reports status OK, takes the
  machine's R9700 lock, runs `whirl bench` or starts one server and waits for `/health`, measures, stops
  the server gracefully (Ctrl+C, so the KV tiers finish writing), releases the lock, checks the GPU status
  again and appends its rows to one results file.
- **Greedy decoding in both engines.** Requests send `temperature 0`; llama-server also runs with
  `--top-k 1`. Requests are the same bytes for both engines; llama-server runs with `--reasoning off`.
  The 7 coding prompts, the system-prompt and the restore tests send
  `chat_template_kwargs: {enable_thinking: false}`. WHIRL's server turns thinking on when a request
  does not say, so the decode-after-16k test (§6) and the 4-user test (§7) were re-measured with the same
  `enable_thinking: false` for every WHIRL version; the llama.cpp side already ran with `--reasoning off`.
- **Which run counts:** per case, model and executable only the **last** run counts (one job writes all
  its rows with one timestamp), except the 4-user server test (§7), which was repeated on purpose: its
  numbers are the **mean of all runs** (3 runs per version and model, llama.cpp 1).
  Rows a job marked as an error or warning are not used.
- **A fresh SSD tier per run:** every WHIRL server run gets its own `--kv-ssd-dir`, so no run restores a
  prompt of an earlier run (only the restore test reuses the directory, by design).
- **Same output:** WHIRL's greedy output is compared across versions by hash (`whirl bench`: its own
  hash; server: SHA-1 of the generated text, tool calls included without their random `id`). Every
  version comparison on this page produced the same output.
- **No measured run includes WHIRL's first-run GEMM autotune**; server timings start after start-up.
- **Ratio = WHIRL / llama.cpp**; "llama.cpp" is its **fastest** mode for that row (plain, MTP, MTP +
  n-gram), and the mode is named. For times (TTFT) the ratio is llama.cpp time / WHIRL time.
- "**no MTP**" marks plain decoding without any speculation. "N/A" = llama.cpp has no equivalent.

### 2.2 What each test is

| Test | Tool | Details |
|---|---|---|
| Prefill (CLI) | `whirl bench --prefill N --decode 8 --modes plain` | 8,192 / 32,768 / 65,536 / 98,304 / 131,072 tokens, KV f16. 8k and 32k: best of 2 after a warm-up; 64k: one run after a warm-up; 96k, 128k: one timed run |
| Prefill (server, long documents) | `whirl serve`, one client | q8h KV, `-np 1 --ctx-per-slot 262144`; non-repeating documents of 61.4k / 123.5k / 180.8k / 249k tokens (the 192k file is the 256k file cut at a document boundary), `max_tokens 32`; tok/s = `timings.prompt_per_second` |
| Main decode scenario | both servers, one client | the 7 `bench_zhcode` prompts of v0.1.0 (Chinese question; answer in Chinese with English code), thinking off, `max_tokens 800`; tok/s = `timings.predicted_per_second`, mean over the 7 prompts per round, median of 3 rounds |
| Speculative decoding cases | `whirl bench` / `whirl serve` | best: a file-editing prompt whose answer repeats the input; typical: an agent session replay (91 requests); long-document QA and worst case: the 123.5k-token document, then a question that quotes it a lot (512 tokens) / a brand-new short story (nothing to copy, 1,024 tokens). Each against the same case without speculation |
| Decode after 16k | both servers, one client | 128 tokens (`ignore_eos`) after the 16,354-token code prompt of v0.1.0 |
| 4 concurrent users | both servers, one client | 4 slots × 4,096 context; 4 distinct ~1.1k-token prompts at once, 256 tokens each (`ignore_eos`); aggregate = generated tokens / wall time (prefill included) |
| System prompt TTFT | both servers, one client | TTFT = wall time of a `max_tokens 1` request; a 26.4k-token system prompt cold, then a new conversation with the same system prompt (warm). WHIRL `--kv-ram-mb 16384`, llama-server `--cache-ram 16384` |
| SSD restore | WHIRL server | a 25.7k-token session, graceful restart on the same `--kv-ssd-dir`, the next turn's TTFT |
| Long-document Q&A | WHIRL server, q8h | 1,024-token answers after the 128k / 256k documents; spot check: the answer names the function asked about (not a formal needle test) |
| Agent sessions | WHIRL server | recorded coding-agent sessions replayed request by request (OpenAI `tools` / `tool_calls`): decode tok/s, TTFT, prefix-cache hits |
| Multi-user | WHIRL server, `-np 4` | KV pool size from the start-up log; 4 short prompts at once; decode floor while 3 sub-agents (~17k-token prompts) prefill |

### 2.3 llama.cpp settings

- `llama-server --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024` (MoE `-ub 2048`, the best at 8k–32k)
  `--no-webui --load-mode none --jinja --reasoning off --top-k 1`, KV f16.
- WHIRL's server flags of a case are mapped: `-np N` → `-np N`; `--ctx-per-slot N` → `-c N×np`; none →
  `-np 1 -c 131072`; `--kv-ram-mb N` → `--cache-ram N`.
- Modes: plain; MTP = `--spec-type draft-mtp`; MTP + n-gram = `--spec-type draft-mtp,ngram-mod`; both with
  `--spec-draft-n-min 0 --spec-draft-p-min 0.3` and `--spec-draft-n-max` 3 (dense) / MTP 2, MTP + n-gram 1
  (Ornith), the best values of the v0.1.0 probes.
- Rows marked "v0.1.0" in the llama.cpp column reuse the b11214 numbers of the v0.1.0 benchmark (same
  build, same flags, `llama-bench` for prefill); the 16k decode, 4-user and system-prompt rows were
  re-measured for v0.1.3 with exactly the WHIRL requests (§6–§8).

WHIRL runs with defaults: `whirl serve MODEL` (4 slots, automatic KV format and pool size), plus the
per-case flags above; no MTP = `WHIRL_MTP=0 WHIRL_NGRAM=0`.

## 3. Prefill

The images of the v0.1.0 page are not regenerated for v0.1.3; the tables are the reference.

### 3.1 Each engine's own bench tool (KV f16)

| Prompt tokens | Swift MXFP4-A WHIRL | llama.cpp | ratio | Ornith MXFP4 WHIRL | llama.cpp | ratio | Qwen3.8 Q4_K_M WHIRL | llama.cpp | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 3,469 | 1,338 | **2.59×** | 11,258 | 4,637 | **2.43×** | 1,731 | 1,223 | **1.42×** |
| 32,768 | 2,907 | 1,174 | **2.48×** | 8,633 | 3,778 | **2.29×** | 1,570 | 1,086 | **1.45×** |
| 65,536 | 2,388 | 1,002 | **2.38×** | 6,507 | 3,069 | **2.12×** | 1,398 | 939.8 | **1.49×** |
| 98,304 | 2,021 | 886.7 | **2.28×** | 5,184 | — ¹ | — | 1,266 | 837.3 | **1.51×** |
| 131,072 | 1,757 | 791.3 | **2.22×** | 4,311 | 2,239 | **1.93×** | 1,158 | 752.0 | **1.54×** |

llama.cpp: `llama-bench` b11214, the best micro-batch per length (v0.1.0 benchmark, §2.3 there); the 96k
numbers are the b11214 numbers used in the v0.1.3 README. ¹ llama.cpp was not measured on Ornith at 96k and 256k tokens.

### 3.2 Long documents through the WHIRL server (q8h KV)

| Prompt | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| 61.4k tokens | 2,162 | 5,671 | 1,290 |
| 123.5k tokens | 1,533 | 3,674 | 1,032 |
| 180.8k tokens | 1,212 | 2,771 | 876.6 |
| 249k tokens | 969.8 | 2,135 | 742.2 |
| 256k, llama.cpp (llama-bench b11214) | 556.1, KV f16 (**1.74×**) | — ¹ | 516.4, KV q8_0 (**1.44×**) |

llama.cpp ran Swift MXFP4 at 256k with f16 KV (it fits, 31.3 GB); Qwen3.8 Q4_K_M needed q8_0 KV,
because f16 spilled into shared memory.

### 3.3 v0.1.0 → v0.1.3 (Swift MXFP4-A)

| Prefill | v0.1.0 | v0.1.3 | change |
|---|---|---|---|
| 8k, f16 KV (bench) | 3,273 | 3,469 | +6.0% |
| 32k, f16 KV (bench) | 2,605 | 2,907 | +11.6% |
| 96k, f16 KV (bench) | 1,653 | 2,021 | +22.3% |
| 128k, f16 KV (bench) | 1,407 | 1,757 | +24.8% |
| 128k, q8h KV (server) | 1,353 | 1,533 | +13.3% |
| 192k, q8h KV (server) | 1,054 | 1,212 | +15.1% |
| 256k, q8h KV (server) | 832.5 | 969.8 | +16.5% |

The gain grows with the prompt because v0.1.3's prefill attention kernel (`attn_kg`,
[kernels.md](guide/en/kernels.md#flash)) shares each key/value load across the query heads of a group;
its output is bit-identical to the old kernel.

## 4. Decode — main scenario: Chinese question, Chinese explanation, English code

| Model | WHIRL MTP+n-gram (default) | WHIRL **no MTP** | llama.cpp **plain (no MTP)** | llama.cpp fastest (mode) | WHIRL default / llama.cpp fastest | WHIRL no MTP / llama.cpp plain |
|---|---|---|---|---|---|---|
| Swift MXFP4-A (dense) | **112.4** | 38.1 | 33.4 | 60.8 (MTP) | **1.85×** | **1.14×** |
| Ornith MXFP4 (MoE) | **257.6** | 177.4 | 118.8 | 118.8 (plain) | **2.17×** | **1.49×** |
| Qwen3.8-27B Q4_K_M (dense) | **97.3** | 34.8 | 30.9 | 56.0 (MTP + n-gram) | **1.74×** | **1.13×** |

Mean over 7 prompts × 800 tokens, median of 3 rounds. llama.cpp: the v0.1.0 numbers (b11214). On the
MoE model llama.cpp's MTP is slower than its plain decoding, so the comparison is against plain.
Against v0.1.0 (same day, same harness): Swift 109.2 → 112.4 (+2.9%), Qwen3.8 99.7 → 97.3 (−2.4%);
same output in both.

## 5. Speculative decoding: best, typical and worst case (Swift MXFP4-A)

| | Best: file editing (repeated content) | Typical: coding-agent session | Long-document QA (heavy quoting, after 128k) | Worst: new writing after 128k (nothing to copy) |
|---|---|---|---|---|
| Decode tok/s, MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s, no speculation | 39.1 | 32.8 | 25.3 | 25.2 |
| Speedup | **4.47×** | **2.83×** | **3.47×** | **1.56×** |
| Tokens per cycle | 6.19 | — | 6.45 | 1.76 |

The output is the same bit for bit with and without speculation. Typical = an agent session replay
(91 requests, 12k → 65k tokens); the replay does not record tokens per cycle. Short Chinese coding
prompt (`whirl bench`, 512 tokens): 120.7 vs 39.6 tok/s (3.05×, 4.00 tokens per cycle).

| v0.1.0 → v0.1.3, MTP + n-gram | v0.1.0 | v0.1.3 | change |
|---|---|---|---|
| File editing | 157.3 | 174.4 | +10.9% |
| Short Chinese coding prompt | 110.7 | 120.7 | +9.1% |
| Agent session 1 (91 requests) | 81.5 | 92.9 | +13.9% |
| Long-document QA after 128k | 63.7 | 87.6 | +37.5% |

## 6. Decode after a 16k-token context

| Model | WHIRL MTP+n-gram (default) | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp fastest |
|---|---|---|---|---|---|
| Swift MXFP4-A (dense) | **71.4** | 31.2 | 60.8 | 60.9 | **1.17×** (MTP + n-gram) |
| Ornith MXFP4 (MoE) | **316.4** | 106.9 | 104.7 | 89.7 | **2.96×** (plain) |
| Qwen3.8-27B Q4_K_M (dense) | **80.7** | 29.4 | 60.6 | 60.7 | **1.33×** (MTP + n-gram) |

Thinking off on both sides (§2.1). Prefill of the 16,354-token prompt in the same requests: Swift 3,156
vs 1,285 tok/s (2.46×), Ornith 9,770 vs 4,184 (2.33×), Qwen3.8 1,596 vs 1,179 (1.35×), each against
llama.cpp plain. v0.1.0 → v0.1.3 (same day): Swift 70.7 → 71.4 (+0.9%), Ornith 313.5 → 316.4 (+0.9%),
Qwen3.8 76.5 → 80.7 (+5.5%), same output.
The v0.1.0 page's 16k numbers (Ornith 301.5, Swift 69.5) were measured with another harness and are
not comparable.

## 7. Server, 4 concurrent users

| Model | WHIRL MTP+n-gram (default) | llama.cpp plain | llama.cpp MTP | llama.cpp MTP + n-gram | WHIRL / llama.cpp fastest |
|---|---|---|---|---|---|
| Swift MXFP4-A (dense) | **182.1** | 64.0 | 54.9 | 56.4 | **2.84×** (plain) |
| Ornith MXFP4 (MoE) | **381.1** | 176.8 | 107.7 | 129.0 | **2.16×** (plain) |
| Qwen3.8-27B Q4_K_M (dense) | **122.4** | 57.1 | 53.3 | 50.1 | **2.14×** (plain) |

Aggregate tok/s, prefill included, thinking off on both sides (§2.1). WHIRL: mean of 3 runs per model;
v0.1.0 on the same day (3 runs each): Swift 173.9 (+4.7% in v0.1.3), Ornith 377.5 (+1.0%), same output. llama.cpp: one run per mode.
llama.cpp's speculative modes do not scale with concurrent users in b11214; WHIRL keeps MTP + n-gram on
for every slot (batched verify).

## 8. Time to first token: system prompt and SSD restore

| Model | Request | WHIRL TTFT (s) | llama-server TTFT (s) | llama.cpp / WHIRL |
|---|---|---|---|---|
| Swift MXFP4-A | 26.4k system prompt, cold | 9.276 | 21.768 | **2.35×** |
| Swift MXFP4-A | new conversation, same system prompt (warm) | 0.108 | 0.384 | **3.56×** |
| Swift MXFP4-A | 25.7k session after a server restart (restored from SSD) | 0.613 | N/A | N/A |
| Ornith MXFP4 | 26.4k system prompt, cold | 3.082 | 6.667 | **2.16×** |
| Ornith MXFP4 | new conversation, same system prompt (warm) | 0.098 | 0.182 | **1.86×** |
| Ornith MXFP4 | 25.7k session after a server restart (restored from SSD) | 0.318 | N/A | N/A |
| Qwen3.8 Q4_K_M | 26.4k system prompt, cold | 17.515 | 23.555 | **1.34×** |
| Qwen3.8 Q4_K_M | new conversation, same system prompt (warm) | 0.121 | 0.369 | **3.05×** |
| Qwen3.8 Q4_K_M | 25.7k session after a server restart (restored from SSD) | 0.672 | N/A | N/A |

- *Warm*: WHIRL reuses its system-prompt checkpoint, llama-server its cached prefix (26,403 tokens
  cached in both).
- *SSD restore*: the server is stopped gracefully (Ctrl+C) and started again on the same
  `--kv-ssd-dir`; the session's next turn is restored from WHIRL's SSD tier. llama.cpp has no automatic
  persistent KV cache, so this row is N/A.
- llama.cpp ran plain only here: the reply is one token, so speculative decoding does not apply.
- WHIRL's cold TTFT includes running the MTP block over the prompt.

## 9. Long context (q8h KV, WHIRL server, one user)

| | Swift MXFP4-A | Ornith MXFP4 | Qwen3.8 Q4_K_M |
|---|---|---|---|
| Decode tok/s after 128k (1,024-token answer) | 62.5 | 214.0 | 58.3 |
| Decode tok/s after 256k (1,024-token answer) | 35.1 | 115.6 | 33.5 |
| llama.cpp after 256k, plain (llama-bench `tg64@d262144`) | 16.8, KV f16 (**2.09×**) | — ¹ | 10.5, KV q8_0 (**3.19×**) |
| Long-document Q&A spot check (128k / 256k) | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |

`--ctx-per-slot 262144`, q8h KV (int8 keys and values; queries and keys Hadamard-rotated first). The
spot check asks about one function in the document; ✓ = the answer names it (not a formal needle test).
The llama.cpp row is plain decoding of 64 tokens after a 262,144-token context, without speculative
decoding.

## 10. Coding-agent sessions (Swift MXFP4-A, WHIRL server)

| Replay | v0.1.0 | v0.1.2 | v0.1.3 | change v0.1.3 / v0.1.0 |
|---|---|---|---|---|
| Session 1 (scrapy, 91 requests, 12k → 65k tokens), mean decode tok/s | 81.5 | — | 92.9 | +13.9% |
| Session 4 (aiohttp, 102 requests, 14k → 102k), mean decode tok/s | 79.2 | — | 95.2 | +20.1% |
| Session 3 up to request 143 (144 requests), mean decode tok/s | 64.9 | 65.0 | 81.7 | +25.9% |
| Session 3, request 143 alone (127.9k tokens) | 53.7 | 53.5 | 75.2 | +40.0% |

Recorded sessions of a coding agent, replayed request by request with their tools and tool calls;
prefix-cache hits 98–99% of prompt tokens in every version; same output in every version.

## 11. Several users on one server (Swift MXFP4-A)

| | v0.1.0 | v0.1.3 | change |
|---|---|---|---|
| KV pool, default 4 slots (tokens) | 210,688 | 226,304 | +7.4% |
| 4 short prompts sent together: whole batch done (s) | 9.57 | 8.62 | −9.9% |
| … aggregate tok/s | 213.9 | 237.6 | +11.1% |
| Main stream tok/s while 3 sub-agents prefill | 1.08 | 9.63 | ×8.9 |
| … tokens in the worst 1-s window | 1 | 5 | |
| 3 sub-agents first, main request 0.5 s later: main TTFT (s) | 17.256 | 16.205 | −6.1% |

Sub-agents: 3 prompts of ~17k tokens each. Same output in both versions.

## 12. Exact commands

```bat
:: WHIRL CLI
whirl bench MODEL.gguf --prefill 8192 --decode 8 --modes plain
whirl bench MODEL.gguf --prefill 512 --decode 512 --prompt @zh_code.txt

:: WHIRL server (each run with its own SSD tier directory)
whirl serve MODEL.gguf --port 8099 --log-file LOG --kv-ssd-dir RUN\kvcache [-np 1 --ctx-per-slot 262144] [--kv-ram-mb 16384]
    (q8h KV: set WHIRL_KV=q8h; no MTP: set WHIRL_MTP=0 and WHIRL_NGRAM=0)

:: llama.cpp server
set HIP_VISIBLE_DEVICES=1& set GGML_CUDA_NO_PINNED=1
llama-server -m MODEL.gguf --port 8099 --device ROCm0 -ngl 99 -fa on -b 2048 -ub 1024 --no-webui --load-mode none ^
    --jinja --reasoning off --top-k 1 -np 1 -c 131072 [--cache-ram 16384]
    [--spec-type draft-mtp --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
    [--spec-type draft-mtp,ngram-mod --spec-draft-n-max N --spec-draft-n-min 0 --spec-draft-p-min 0.3]
```

The harness (a PowerShell job runner and a Python client using only the standard library) is not part of
this repository; it records every request's timings, and keeps each run's server log and full command
line on the measurement machine.

## 13. Not re-measured for v0.1.3

VRAM use, vision encoding, 88- and 2,048-token prefill, the same-text server prefill and the
file-editing comparison against llama.cpp were not measured again; their v0.1.0 numbers are on the
[v0.1.0 page](https://github.com/tsaipifong/whirl-llm/blob/v0.1.0/docs/benchmarks.md). v0.1.3 moves the token embedding to
pinned host RAM, so its VRAM use should be lower than v0.1.0's.
