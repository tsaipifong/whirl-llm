**English** | [繁體中文](guide/zh-TW/usage.md)

# Usage reference: `whirl` and `whirl-server`

This page lists every command, option, environment variable and server endpoint of WHIRL 0.2.2.1.
It was written from the built-in help of the release executables (`whirl --help`,
`whirl <command> --help`, `whirl help env`, `whirl-server --help`); when in doubt, the help text of
your own executable is authoritative.

New here? Start with the [quick start](quickstart.md). Design background for the server lives in
[server.md](guide/en/server.md), for caching in [kv-and-caching.md](guide/en/kv-and-caching.md).

- [1. The two programs](#programs)
  — [numerics modes: precise, balance, fast](#modes)
- [2. `whirl chat`](#chat)
- [3. `whirl bench`](#bench)
- [4. `whirl serve` / `whirl-server`](#serve) — options, [endpoints](#endpoints), [stopping](#stop)
- [5. `whirl devices`, `selftest`, `seqtest`, `vis-encode`](#other)
- [6. Exit codes and error messages](#exit-codes)
- [7. Environment variables](#env)
- [8. Files WHIRL writes](#files)

## <a id="programs"></a>1. The two programs

| Program | What it is |
|---|---|
| `whirl.exe` | command-line tool: `chat`, `bench`, `serve`, `devices`, `selftest`, `seqtest`, `vis-encode`, `help` |
| `whirl-server.exe` | the OpenAI-compatible HTTP server; identical to `whirl serve` |

```
whirl <command> [arguments]
whirl --help | -h            general help
whirl --version | -V         version, the GPU architectures the kernels were built for, driver requirement
whirl help COMMAND           the options of COMMAND (same as: whirl COMMAND --help)
whirl help env               every environment variable with a one-line description
whirl-server --help | --version
```

Both programs accept only GGUF files whose `general.architecture` is `qwen35` (dense) or
`qwen35moe` (mixture of experts); see [Supported models](../README.md#supported-models). Generation is greedy
in the CLI; the server also samples (temperature, top-p, …).

**First run with a new model.** The prefill kernels are tuned once per model file and GPU
(a message on stderr says so): a few seconds for MXFP4 files, longer for K-quant files (which mix
many tensor types). The result is cached ([section 8](#files)); later runs start right away. Every candidate
configuration gives the same bits, only the speed differs.

**One process per GPU.** A WHIRL process holds a per-GPU lock while it runs. A second WHIRL process
on the same GPU waits for it (up to 30 minutes by default) instead of sharing VRAM, because two
large processes on one GPU make Windows move both to slow shared memory
([windows-hip.md](guide/en/windows-hip.md#wddm-demote)). See `WHIRL_GPU_WAIT` / `WHIRL_GPU_SHARE`.

### <a id="modes"></a>Numerics modes: balance (default), precise, fast

One mode per process, chosen at start-up (`--precise`, `--balance`, `--fast`, or `--mode M`; or the
environment variable `WHIRL_MODE=precise|balance|fast`; the command line wins). It applies to
`chat`, `bench`, `selftest`, `seqtest` and the server alike, on the R9700 and the Radeon 8060S.
With no mode given the process runs **balance**; `--precise` and `--fast` are opt-in (`--balance`
is still accepted).

**KV cache format: fixed per mode and model type**, the same on every card (32 GB, 16 GB, the
8060S; one exception for fast, below the table) and for `chat`, `bench` and the server alike:

| Mode | Dense models | MoE models |
|---|---|---|
| precise | f16 | f16 |
| balance (default) | q8h (int8 K and V + one f16 scale per 32 values, Hadamard-rotated q / k; passes the balance KL gate vs f16 at 128k, KL-1) | f16 |
| fast | q4 (Radeon 8060S: q8h, which decodes faster than q4 there) | q4 |

The format is never switched by what fits: when the chosen format cannot hold the context, the
context is lowered with a warning if you did not give it, or the program stops with a message (exit
code 5) if you did (`--ctx`, `--ctx-per-slot`). It never falls back to a lower-precision format by
itself. Cards under 20 GiB only get one server slot and VRAM headroom by default.

| Mode | What it does |
|---|---|
| **precise** (`--precise`) | The GGUF weights are dequantized to f16; prefill activations and accumulation are f16 / f32; DeltaNet prefill uses the f32 chunk path; **the KV cache is always f16** on every card. Decode, MTP / n-gram verify and other small batches use f16 activations into the same f16 GEMM as prefill (f32 accumulation; no int8 activations). Nothing is quantized beyond the file's own weights. If f16 KV does not fit, the context is lowered with a warning when you did not give it, or the program stops with a message (exit code 5) when you did (`--ctx`, `--ctx-per-slot`) — it never switches to int8 KV by itself |
| **balance** (default; `--balance`) | The default when no mode is given: the speed-oriented items below, with q8h KV on dense models and f16 KV on MoE models; the same items on dense and MoE models. Near-lossless, not bitwise: the output may differ slightly from precise (measured KL / accuracy in [quantization.md](guide/en/quantization.md)) |
| **fast** | balance plus more aggressive lossy items that must pass KL + paired-accuracy gates first. Implemented: `kvq4` and `relaxacc`; the others are listed as skipped |

Items (pick a subset with `--balance=fp8,kvq8` or `--fast=...`; `WHIRL_MODE=balance:fp8,kvq8`):

| Item | Mode | What it is | R9700 (gfx1201) | Radeon 8060S (gfx1151) |
|---|---|---|---|---|
| `fp8` | balance | MXFP4 dense prefill GEMMs with fp8 (e4m3) activations, incl. the rounding of folded MXFP4 exponents | MXFP4 models | skipped (no fp8 WMMA) |
| `moefp8` | balance (MoE models only) | MoE expert prefill GEMMs with fp8 activations | MXFP4 MoE (Ornith MXFP4) | skipped |
| `gdnwmma` | balance | f16-WMMA DeltaNet prefill chunks instead of the f32 chunk path | MXFP4 models and dense models of any weight format (Q4_K_M, Q5_K, Q6_K, IQ4_XS, ...; KG-1); MoE models only with MXFP4 | skipped (no kernel yet) |
| `h16` | balance | f16 FFN / DeltaNet GEMM outputs before the element-wise ops | MXFP4 models and dense models of any weight format (KG-1); MoE models only with MXFP4 | MXFP4 models, f16 weights only |
| `kvq8` | balance | q8h KV on dense models, always (not only when f16 does not fit); MoE models keep f16 | dense models | dense models |
| `specsample` | balance | Requests with temperature > 0: tokens are sampled by a race keyed by (seed, position, token) and MTP drafts are drawn from the draft head's distribution (same temperature / top-k / top-p / min-p) by the same race, accepted only when they equal the sampled token: exact sampling, acceptance close to the optimum, and the text for a seed does not depend on drafts (MTP on == off). The sequence for a seed differs from precise's sampler. Off (precise): inverse-CDF sampler, greedy drafts. Greedy (temperature 0) is unchanged in every mode | MTP models | MTP models |
| `kvq4` | fast | 4-bit KV (int4 + one f16 scale per 32 values, Hadamard-rotated q / k); on the 8060S decode attention reads it through `attn_dq4` (~180 GB/s): Ornith-1.5-35B-A3B MXFP4 decode at 128k 58.5 tok/s (f16 KV 32.7), at 64k 67.0 (46.4); on the R9700 (FAST-1c) Swift-1.5 27B decode at 131k plain 30.98 tok/s (f16 KV 25.08), MTP 68.79 (63.12), prefill 128k −1.9%; KL vs f16 within the fast gate | dense and MoE | MoE (the dense model keeps q8h KV in fast: faster decode there) |
| `relaxacc` | fast | Relaxed speculative acceptance (MTP drafts; n-gram drafts stay exact). A relaxed acceptance is refused when it would end in a 6-gram already among the last 4096 tokens (repetition guard). Greedy: a draft that is not the verify row's argmax is still kept when it is among the row's top `WHIRL_RELAX_K` tokens (default 4) and p(draft) >= `WHIRL_RELAX_ALPHA` x p(argmax) (default 0.1); the token after the last kept draft is always the target's own argmax. Temperature > 0: typical acceptance (Medusa): keep the draft when p(draft) >= min(`WHIRL_RELAX_EPS` 0.09, `WHIRL_RELAX_DELTA` 0.3 x exp(-entropy)) on the request's filtered distribution (`specsample` is not used then). **Changes greedy output**: fast greedy is not plain greedy and MTP != plain; it stays deterministic for the same prompt, seed and settings (the draft-count policy then uses a fixed synthetic cycle cost, not wall-clock times; concurrent requests share draft counts and can change each other's output). `WHIRL_RELAX=0` turns it off; MoE models skip it (one MTP draft per cycle, no measured gain; `WHIRL_RELAX=1` forces it) | dense MTP models | dense MTP models |
| `headq`, `moeskip`, `a8`, `a4` | fast | low-bit output head, MoE expert skipping, W4A8 / W4A4 prefill | not yet implemented | not yet implemented |
| `q8dec` | balance, fast | int8 activations (one f32 scale per 32 values, as llama.cpp's q8_1) in the decode / verify GEMV (always part of balance and fast) | on | on |

What balance and fast change, by model type (R9700; the Radeon 8060S has no fp8 WMMA, so `fp8` /
`moefp8` are skipped there in every mode):

| Mode | Dense models (MXFP4 dense; also K-quant files such as Q4_K_M) | MoE models (Ornith-1.5-35B-A3B MXFP4) |
|---|---|---|
| precise | f16 / f32 everywhere, f16 KV | f16 / f32 everywhere, f16 KV |
| balance | `fp8` (MXFP4 files only), `gdnwmma`, `h16`, `kvq8` (q8h KV), `specsample`, `q8dec` | `fp8`, `moefp8`, `gdnwmma`, `h16`, `specsample`, `q8dec`, f16 KV |
| fast | balance + `kvq4` (q4 KV) + `relaxacc` | balance + `kvq4` (q4 KV); `relaxacc` skipped |

Balance is defined by the items it may use (fp8 and moefp8 activations, int8 decode activations,
f16 intermediates, int8 KV), not by staying bitwise close to precise. MoE balance had no `fp8` /
`moefp8` from MOE-BAL (2026-10-13) until MODES-MXFP4 (2026-10-16), which put them back: on the R9700
Ornith-1.5-35B-A3B MXFP4 balance prefill is +40% at 2k, +35% at 8k and +26% at 32k (13.0k / 13.0k /
9.6k tok/s, decode unchanged). `--balance=gdnwmma,h16,specsample` (an explicit list is used as
given) gives the previous MoE balance.

MoE balance keeps `q8dec` (int8 decode / verify activations, llama.cpp's own q8_1 scheme; MOE-BAL-3,
2026-10-15): the KL gate allows MoE balance a top-1 drop of 3.0 points against the noise floor
(2.4 before): llama.cpp itself decodes with int8 (q8_1) activations, so this is the same numeric
level as llama.cpp; the KL limits (2 N mean, 3 N per length) are unchanged. A short-lived MOE-BAL-2 table without `q8dec`
cost about 20% of MoE balance decode on the R9700. `WHIRL_Q8DEC=0` gives f16 decode activations.

In precise mode `q8dec` is off: every matmul of a decode step, an MTP / n-gram verify or a concurrent
batch takes f16 activations into the f16 WMMA GEMM (weights decoded from the GGUF type to f16, f32
accumulation, a fixed per-row reduction order), so each row gets the same bits whatever the batch
size; a GPU whose kernels lack this path for one of the model's weight types refuses to load the
model in precise mode instead of falling back to int8. Items that do not apply to the GPU or model are logged as skipped (never an error). The
start-up log shows one line, e.g.

```
numerics: balance - enabled: fp8, gdnwmma, h16, kvq8 (q8h KV (int8 + f16 scale / 32, Hadamard-rotated q/k)); skipped: moefp8 (dense model); ...
```

and the server reports the mode in `GET /props` (`"numerics": {"mode", "label", "enabled", "skipped", "overrides", "kv", "decode", "always_on"}`; `decode` is `f16` or `q8dec`).
The per-item variables of [7.6](#env-numerics) (`WHIRL_FP8`, `WHIRL_MOE_FP8`, `WHIRL_GDN_WMMA`,
`WHIRL_FFN_H16`, `WHIRL_Q4_RELAXED`) and the debug variable `WHIRL_KV` ([7.7](#env-debug)) still work and override the mode; in precise
mode a lossy value is logged as `user-requested` and the mode reads `precise+overrides`.
MTP / n-gram decoding equals plain greedy decoding, and concurrent requests equal solo runs, within
each mode in precise and balance (and fast with `WHIRL_RELAX=0`; with `relaxacc` the fast output
depends on the drafts: a repeated request gives the same text streamed or not, cold or resumed from
its own prefix-cache checkpoints, but a prompt resumed from another prompt's shared prefix is
prefilled in other chunks, which can change fast output in the last bits, as concurrency can).
Fast mode aims at quality within its gate (mean KL vs precise <= 0.03 and top-1 >= 93%, checked by tools/klgate at 8k-128k on dense models; MoE models are judged against their own noise floor N: fast mean KL <= 8 N, each length's 3-window average <= 12 N, top-1 at most 8 points under N's; on the R9700 (512-row windows) Qwen3.8-27B fast has KL <= 0.0072 and top-1 >= 96.09%, Ornith-1.5-35B-A3B fast is 6.1 N on both prompts), not bitwise-identical output between a concurrent and a solo run: with `relaxacc` the draft cadence depends on the load, so concurrent text can differ from solo text (best effort; precise and balance do guarantee equality). With temperature > 0, MTP decoding gives the same tokens as plain sampling
for a seed in precise and balance (with `specsample` too), whatever the prefix cache or earlier
requests; balance (`specsample`) samples by a different (equally exact) rule than precise, so its
tokens for a seed differ from precise's. Fast with `relaxacc` uses typical acceptance instead.

## <a id="chat"></a>2. `whirl chat`

```
whirl chat MODEL.gguf "PROMPT" | @PROMPT_FILE [options]
```

Generates a reply to one prompt. The prompt is wrapped in the model's chat template (user turn,
then the assistant turn opened with `<think>\n` when thinking is on, or with an empty think block
when it is off). The text streams as it is generated and stops at the end-of-turn token or after
`--max-tokens`. If the model has an MTP head, decoding is speculative (MTP + n-gram drafts); the
output is identical to plain greedy decoding. The summary at the end reports prefill and decode
speed and, with MTP, the verify cycles and draft acceptance.

| Option | Meaning |
|---|---|
| `"PROMPT"` / `@FILE` | prompt text, or `@path` of a UTF-8 file holding it (long prompts) |
| `--max-tokens N` | tokens to generate (default 64) |
| `--ctx N` | context size: prompt + generated tokens (default 8192; also `WHIRL_MAX_CTX`) |
| `--no-think` / `--think` | thinking off / on (default on; also `WHIRL_THINK=0\|1`) |
| `--no-stream` | print the reply once at the end instead of streaming it |
| `--raw` | no chat template: the prompt text is tokenized as is |
| `--tokens ID,ID,...` | token ids instead of a prompt (no template) |
| `--out FILE` | write the last prompt position's logits (f32, one value per vocabulary entry) to FILE; disables MTP |
| `--draft mtp\|dflash` | draft source (default `mtp`). `dflash`: the DFlash2 drafter GGUF (`--draft-model`) drafts up to block - 1 tokens per cycle, n-gram drafts stay on, the MTP block is not loaded; without a drafter that fits the model: MTP drafts with a warning |
| `--draft-model FILE` | DFlash2 drafter GGUF for `--draft dflash`. Wins over `WHIRL_DRAFT_MODEL` (the environment variable is only used when `--draft-model` is not given; the log says which one was used) |
| `--mmproj MMPROJ.gguf` | vision encoder (Qwen3-VL style mmproj, F16 / BF16), needed for `--image` |
| `--image IMAGE` | an image placed before the prompt text; repeatable |
| `--device SPEC` | GPU: `r9700`, `8060s`, an index, or a substring of the name / gfx architecture (also `WHIRL_DEVICE`). Default: the first R9700, else the first GPU the build has kernels for (a Radeon 8060S on its own) |
| `--precise` / `--balance` / `--fast` / `--mode M` | numerics mode (default balance; also `WHIRL_MODE`); `--balance=ITEMS` picks items. See [numerics modes](#modes) |
| `-h`, `--help` | help |

Examples:

```powershell
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl.exe chat C:\models\model.gguf @C:\work\long_prompt.txt --max-tokens 2000 --ctx 65536 --no-think
.\whirl.exe chat C:\models\model.gguf "What is in this picture?" --mmproj C:\models\mmproj-F16.gguf --image C:\pics\cat.png
```

## <a id="bench"></a>3. `whirl bench`

```
whirl bench MODEL.gguf [options]
```

Loads the model once, then measures (1) prefill speed for each size — best of 2 after a warm-up up
to 8k tokens, one timed run above — on a mixed Chinese/English coding text, and (2) decode speed in
each mode after a short coding prompt. The token streams of all decode modes must be identical; a
difference is reported.

| Option | Meaning |
|---|---|
| `--prefill N,N,...` | prefill sizes in tokens (default `2048,8192,32768`) |
| `--decode N` | tokens generated per decode mode (default 256) |
| `--prompt TEXT` / `@FILE` | prompt of the decode runs (default: a zh/en coding question) |
| `--modes M,M,...` | decode modes: `mtp-ngram` (MTP + n-gram drafts, the default path), `mtp` (MTP only), `plain` (no MTP); default all three |
| `--no-think` / `--think` | thinking off / on for the decode prompt (default on) |
| `--draft mtp\|dflash`, `--draft-model FILE` | draft source and DFlash2 drafter, as for `chat` (with `dflash` the modes are `dflash-ngram`, `dflash`, `plain`) |
| `--device SPEC` | GPU, as for `chat` |
| `--precise` / `--balance` / `--fast` / `--mode M` | numerics mode, as for `chat` |

```powershell
.\whirl.exe bench C:\models\model.gguf --prefill 2048,8192 --decode 256 --modes mtp-ngram,plain
```

How we compare engines (interleaved runs, what to report) is in
[benchmarking.md](guide/en/benchmarking.md); measured results are in [benchmarks.md](benchmarks.md).

## <a id="serve"></a>4. `whirl serve` / `whirl-server`

```
whirl-server MODEL.gguf [options]
whirl serve  MODEL.gguf [options]      (the same program)
```

Loads the model and serves the OpenAI-compatible API with continuous batching over `--parallel`
request slots, a prefix cache in VRAM and host RAM / SSD tiers for idle sessions.

| Option | Meaning |
|---|---|
| `--host ADDR` | address to listen on. Default `127.0.0.1` (this computer only). `0.0.0.0` listens on every network: anyone who can reach the PC can use the server — there is no authentication |
| `--port N` | TCP port (default 8080). If it is taken, the server exits with code 6 before loading the model |
| `--alias NAME` | model id reported by `/v1/models` (default: the file name without `.gguf`) |
| `--device D` | GPU: `r9700`, `8060s`, a device index, or a name / gfx substring (default as for `chat`) |
| `-np`, `--parallel N` | concurrent request slots (continuous batching), 1–16, default 4 (1 on cards with less than 20 GiB of VRAM, see `WHIRL_VRAM_HEADROOM_MB`) |
| `-c`, `--ctx N` | size of the shared KV pool in tokens. Slots take pages on demand; when the pool is full, idle slots' prefix caches are evicted (least recently used first). Default: all VRAM left after weights and buffers, minus 768 MiB (MoE: 1.5 GiB) of the free VRAM and a margin for other programs under the Windows (WDDM) budget (`WHIRL_POOL_BUDGET_MARGIN_MB`); 262144 on the Radeon 8060S |
| `--ctx-per-slot N` | longest context of one request (default min(pool, 131072); up to 262144) |
| `--precise` / `--balance` / `--fast` / `--mode M` | numerics mode (default balance; `--precise`: f16 KV; when the f16 pool cannot hold a full request the context per request is lowered with a warning, or the server refuses to start if `--ctx` / `--ctx-per-slot` was given). See [numerics modes](#modes) |
| `--mtp-drafts N` | fixed MTP drafts per cycle, 1–10 (default: chosen per model type by a cost model) |
| `--draft mtp\|dflash` | draft source of the speculative decode (default `mtp`). `dflash`: the DFlash2 drafter GGUF given by `--draft-model` (up to block - 1 drafts per cycle; the official 27B drafter has block 8: 7 drafts); a model without a matching drafter (e.g. a MoE model), a drafter file that cannot be opened or does not fit: `mtp` with a warning. n-gram drafts stay on in both modes; greedy outputs equal plain greedy decoding. Opt-in: see [when DFlash2 helps](#dflash) |
| `--draft-model FILE` | DFlash2 drafter GGUF for `--draft dflash` (wins over `WHIRL_DRAFT_MODEL`, as for `chat`) |
| `--decode-min-tps N` | decode floor: while other requests prefill, every streaming (decoding) request keeps at least N tok/s; prefill forwards are shortened and decode cycles interleaved to hold it (default 20; `0` = off, prefill forwards are not limited). Outputs are identical for any N ([server.md](guide/en/server.md#batching)) |
| `--kv-ram-mb N` | host RAM tier of the prefix cache, MiB of pinned memory (default: 1/4 of physical RAM, at least 8 GiB or one full-length session if that is larger — about 9 GiB for the 27B model —, at most 32 GiB, and at most half of the RAM available at startup; 16 GiB on a 64 GB PC; since 0.2.2.1 every RAM tier, given or automatic, is capped at 20480 MiB with an `adjusted:` line, because a larger pinned RAM tier crashed with hipError 719 on the first request on a test machine (cause under investigation); integrated GPUs: OS-visible RAM / 8 within [one max-context entry (about 6 GiB for the 27B model at 262k), 16 GiB], at most 1/4 of the RAM available at startup, off under 32 GB of OS-visible RAM (the BIOS VRAM carve-out does not count; 8 GiB on a 64 GB machine)). The startup log prints the chosen size and why. Idle sessions are copied there and restored instead of prefilled again. `0` disables both host tiers. **Radeon 8060S** (integrated GPU): on by default with the size above (the KV pool is the BIOS carve-out, the tier is separate pinned OS RAM; a 128k session is restored from RAM or SSD instead of being prefilled again); `--kv-ram-mb 0` turns both host tiers off, `--kv-ssd-gb 0` / `--kv-ssd-mode off` only the SSD tier |
| `--kv-ssd-dir PATH` | SSD tier directory (default `%LOCALAPPDATA%\whirl\kvcache`) |
| `--kv-ssd-gb N` | SSD tier size cap in GiB (default 64; `0` = no SSD tier) |
| `--kv-ssd-mode M` | when the SSD tier writes: `shutdown` (default; only reused entries are ever written: when WHIRL is closed — Ctrl+C, console window closed, logoff, OS shutdown; the last three allow only about 5 s — when RAM must evict them, and after about 5 minutes idle; a one-shot prompt is never written), `always` (idle sessions go to the SSD after `--kv-ssd-delay`), `off` (no SSD files). See [how the SSD tier stores sessions](#ssd-tier) |
| `--kv-ssd-delay SEC` | `--kv-ssd-mode always`: write a session to the SSD once it has been unchanged for SEC seconds (default 300). A session that is used again sooner is served from RAM and never written |
| `--mmproj FILE` | vision encoder (Qwen3-VL style mmproj GGUF, F16 / BF16). `image_url` parts (`data:` URLs with base64 PNG / JPEG / …) become image tokens. Weights stay in pinned host RAM; nothing goes to VRAM until an image arrives |
| `--vis-idle-s N` | release the vision encoder after N seconds without images (default 60) |
| `--vis-mode M` | `auto` (default), `resident` (encoder weights in VRAM), `stream` (layer by layer) |
| `--vis-cache-mb N` | host cache of image embeddings, keyed by content hash, MiB (default 1024) |
| `--allow-local-images` | also accept local file paths / `file://` URLs as image sources (off by default; `http(s)` image URLs are never fetched) |
| `--cors-origin ORIGIN` | let web pages from `ORIGIN` (e.g. `https://app.example.com`) call the server from a browser; repeatable. By default only pages served from `http(s)://localhost`, `127.0.0.1` or `[::1]` (any port) get CORS headers (their `Origin` is echoed back, never `*`); other web pages, including `file://` pages, cannot read responses. `*` restores the behaviour before v0.1.4 (`Access-Control-Allow-Origin: *` for every page). A `POST` from a page whose `Origin` is not allowed is refused with 403 (it is not run), including "simple" `text/plain` requests that need no preflight. Non-browser clients (SDKs, curl, IDE agents) send no `Origin` and are not affected. Local image files (`--allow-local-images`) are never read for a request from a web page other than a localhost page, whatever `--cors-origin` allows |
| `--allow-host NAME` | also answer requests whose `Host` header is `NAME` (repeatable; `*` = any). By default the server answers `localhost`, `127.0.0.1`, `[::1]` and the `--host` address (any IP address with `--host 0.0.0.0`); other names get 403. This blocks DNS rebinding (a web page whose own domain is made to resolve to 127.0.0.1). Use it when clients reach the server by a host name, e.g. `--host 0.0.0.0 --allow-host mypc.lan` |
| `--log-file PATH` | log file (default `%LOCALAPPDATA%\whirl\server.log`); the log also goes to the console |
| `-h`, `--help` / `-V`, `--version` | help / version |

Examples:

```powershell
# 27B model, 4 slots, default tiers
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080

# one user, long contexts, no SSD tier
.\whirl-server.exe C:\models\model.gguf -np 1 --ctx-per-slot 262144 --kv-ssd-gb 0

# with image input
.\whirl-server.exe C:\models\model.gguf --mmproj C:\models\mmproj-F16.gguf
```

### <a id="dflash"></a>DFlash2 drafts (`--draft dflash`): when they help

DFlash2 is an optional second draft source for Qwen3.8-27B (dense only; MoE models such as Ornith
have no drafter and keep MTP). A small block-diffusion drafter proposes up to 7 tokens per cycle
from the target's hidden states; the target verifies them exactly like n-gram drafts, so the text is
the same as without drafts (greedy: plain greedy decoding of the same numerics mode; sampling: the
same text for the same seed). Only the speed changes, and not always for the better. Compared with
the default MTP drafts on the R9700:

| Workload | DFlash2 vs MTP |
|---|---|
| one user, short / medium context, English or predictable output (code, lists, repeated structure) | faster |
| mostly-Chinese output | slower |
| 4 concurrent users | slower |
| long context (37k+ tokens) | slower (acceptance drops) |

Use it for a single user with short or medium prompts and English-heavy or predictable output; keep
the default `mtp` for several users, long contexts or mostly-Chinese replies. Costs: about 1.9 GiB of
VRAM (drafter weights, target-feature capture, an 80 MiB context ring per slot), so the KV pool is
smaller; the startup log prints the drafter's share and the VRAM fit check counts it. The MTP block
is not loaded in this mode.

Drafter: download `Qwen3.8-27B-DFlash2-Q8_0.gguf` from the Hugging Face repository
`incoai/Qwen3.8-27B-DFlash2-GGUF` and pass it with `--draft-model FILE`. A Q4_K_M drafter is untested;
an MXFP4 `fc.weight` is refused. A drafter whose tensors do not match its own metadata, or a file
that cannot be opened, is refused with a warning and the MTP drafts are used.

Memory note: with the defaults the server pins about a quarter of the host RAM for the RAM tier
(8–32 GiB, at most half of the RAM free at startup; 16 GiB on a 64 GB PC; plus about 0.9 GiB when `--mmproj` is given) and may use up to 64 GiB on the SSD. Use `--kv-ram-mb` /
`--kv-ssd-gb` to shrink or disable the tiers on machines with less memory or disk space.

### <a id="endpoints"></a>Endpoints

Base URL `http://127.0.0.1:8080/v1`. Any API key is accepted (there is no authentication).

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | messages, streaming (SSE) or not, tools / tool calls, thinking (`reasoning_content`), images (with `--mmproj`) |
| `POST /v1/completions` | raw prompt, no chat template |
| `GET /v1/models` | the one loaded model (id = `--alias`); `meta.n_ctx` is the context per slot |
| `GET /health` | answers immediately even while busy; reports the busy state and queue length; on Windows `vram` (the last video memory sample: `local_mib`, `budget_mib`, `shared_mib`, `spill`) and `startup` (effective values after the startup fit: `ctx_per_request`, `pool_tokens`, `parallel`, `kv_ram_mb`, `wddm_margin_mib`, `other_processes_mib`, `display_attached`, `adjusted`: the settings the server lowered; `wddm_margin_mib` is the margin the KV pool was sized with, `other_processes_mib` what other processes used at startup) |
| `GET /props` (also `/v1/props`) | read-only, llama.cpp-server-style subset for clients that auto-detect the context length: `default_generation_settings.n_ctx` (context per slot), `default_generation_settings.model` and `model_alias` (= `--alias`), `total_slots`, `model_path` (file name only, never a directory), `modalities`, `build_info`, and the `startup` object as in `GET /health` |
| `GET /version` | `{"version":"0.2.2.1","name":"whirl"}` |
| `OPTIONS` (CORS preflight) | supported (allowed origins: see `--cors-origin`) |

LM Studio (`/api/v1/models`) and Ollama (`/api/tags`, `/api/show`, `/api/version`) native endpoints
are not emulated (a client that found them would switch to an API WHIRL does not have): they answer
404, and the server logs the first probe of each path once at `I` level instead of a warning per
request. Point such clients at the OpenAI-compatible base URL above.

Request parameters: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens` /
`max_completion_tokens`, `stop`, `stream`, `tools`, `tool_choice`,
`chat_template_kwargs.enable_thinking` (or top-level `enable_thinking`), `reasoning_effort`,
`preserve_thinking`. Without sampling parameters the model card's recommended values apply. With
`temperature` 0 the output equals `whirl chat` token for token. `presence_penalty` /
`frequency_penalty` are accepted but ignored; `n` must be 1; `logprobs`, `response_format` and
enforced `tool_choice: "required"` are not supported. `stop` takes at most 16 strings of at most
256 bytes each (more: HTTP 400); a `temperature` below 0.001 is greedy. Responses add
`usage.cached_tokens` and a llama.cpp-style `timings` object. Details: [server.md](guide/en/server.md#sampling).

Thinking and reasoning effort can be given in any of these forms (applied in this order; a later one
overrides an earlier one): `chat_template_kwargs.{enable_thinking, reasoning_effort}`, the
OpenRouter / OpenAI Responses-style object `"reasoning": {"effort": "...", "enabled": true|false}`,
top-level `enable_thinking`, top-level `reasoning_effort`. Effort values (case-insensitive):

| Value sent | Effort used |
|---|---|
| `xhigh`, `high`, `max`, `ultra` | xhigh (the default when none is given) |
| `medium` | medium |
| `low`, `minimal` | low |
| `none` (or `"reasoning": {"enabled": false}`) | thinking off |

An unknown effort value does not fail the request: the server logs a warning and uses the default.
Effort only changes the prompt of models whose chat template supports it (Qwen3.8); a valid effort does
not turn thinking back on when it was switched off.

```powershell
$body = '{"messages":[{"role":"user","content":"What is 2+3?"}],"temperature":0,"max_tokens":200}'
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
```

### <a id="prefix-cache"></a>Prefix cache: what a new request reuses

DeltaNet layers can only resume where the server saved a **checkpoint** of their state (details:
[kv-and-caching.md](guide/en/kv-and-caching.md#checkpoints)). Per slot: the end of the prompt
(`prompt-end`: the same prompt again, e.g. regenerate), the end of the answer (`gen-end`: the next
chat or agent turn) and, since 0.2.0, a **tail checkpoint** near the end of every prompt of 4096
tokens or more. In `--precise` (since 0.2.2) the next turn does not resume at `gen-end`: it resumes at the
previous prompt's tail checkpoint (kept for every prompt there) and re-prefills the rest, so the output stays
bitwise identical to a cold run; in balance / fast it resumes at `gen-end` (faster, numerically equivalent to a
cold run, not bitwise). Shared between sessions: long system messages and prefixes common to earlier
sessions.

**Long-document Q&A.** Ask several questions about the same long document, each as its own request
(`document + question` in one user message, no system message): the second question resumes from
the first one's tail checkpoint (the start of its last 1024-token chunk, 257 to 1280 tokens before
the prompt end) and prefills only the rest, instead of the whole document. On a 128k document that
is about 1.1k tokens instead of 131k. Conditions: the questions
differ only after that point (a question up to about 256 tokens always qualifies; put a long
question, or the document, in a system message of 2048+ tokens otherwise), and the slot keeps 3 or
more checkpoints (default: 4 with one slot, 3 with 2 to 4 slots). The output is the same as without
the cache: the tail checkpoint lies on the cold-run chunk schedule (precise and balance stay bit
identical to a cold run). Each per-slot checkpoint takes 150.6 MiB of VRAM for Qwen3.8-27B and
63.8 MiB for Ornith-1.5-35B-A3B; the startup log lists the count and size, and the VRAM fit lowers
the count to 2 (tail checkpoint off) when the KV pool would otherwise be small. `WHIRL_TAIL_CK=0`
turns it off (debugging).

### <a id="ssd-tier"></a>How the SSD tier stores sessions

- **When it writes.** `--kv-ssd-mode shutdown` (default): only reused entries are ever written to
  the SSD — an entry whose prefix was restored at least once, a resumed session, or a shared
  system-message checkpoint other requests attached, with at least 2048 tokens. They are written
  when WHIRL is closed (Ctrl+C, console window closed, logoff, OS shutdown), so the next start finds
  them; when the RAM tier is full and has to evict one (in the background, instead of dropping it,
  subject to the size cap below); and after about 5 minutes with no request (none running, none
  arriving), in the background, one at a time — a request that arrives stops it (it waits at most
  for the one write already running), and nothing more is written while the server stays idle.
  A one-shot prompt is never written, in any of these cases: it lives in the RAM tier only and is
  gone after a restart. Ctrl+C allows a full flush (up to 60 s); closing the console window, logging
  off or shutting down Windows allow only about 5 s (Windows ends the process then), so a large
  backlog may be only partly written — Ctrl+C is the way to guarantee a full write. A crash loses
  what is only in RAM. `always`: an idle session's entry is written once it has been unchanged for
  `--kv-ssd-delay` seconds (default 300, 5 minutes). A conversation whose next request comes within that time is served from RAM and
  costs no disk write; under RAM pressure an entry that is not due yet is written (if it has at
  least 2048 tokens) or dropped instead of being kept. `always` writes only sessions worth keeping: one whose
  prefix was reused (restored) at least once, or that was resumed by a follow-up turn; a one-shot prompt nobody
  came back to is never written (a benchmark stream of distinct prompts writes about nothing). A graceful stop
  writes the most recently used sessions first. `off`: no SSD files at all (the RAM tier
  still works). Why: a benchmark or an agent that sends many different long prompts would
  otherwise write several GB per request (hundreds of GB in minutes), which wears the SSD.
- **Size cap, expiry.** The directory never holds more than `--kv-ssd-gb` (default 64 GiB) on
  disk, counted as real allocated bytes (shared blocks are holes in the sparse files and are not
  counted twice). When it is full, the least recently used session is deleted (together with the
  files that borrow its blocks); files of other models, builds or KV formats are deleted first, and
  so are files written with another per-slot context (`--ctx` / `--parallel`, or a smaller context
  chosen for the free VRAM): those are kept, counted against the cap and used again by a server with
  that context. A session in RAM whose file does not fit the cap is not written (it stays in RAM only;
  logged once). Files not used for 7 days are deleted automatically (at start and hourly); files that
  cannot be used (half-written by a crash, corrupt, not a WHIRL cache file) are deleted at start. The
  start-up log has one line with the files and GiB on disk, the cap and what was removed and why.
- **Only what changed is written.** A continued conversation writes its new KV pages and the
  changed checkpoint, not the whole session again. KV blocks that another file on the SSD already
  holds for the same token prefix (a shared system prompt and tool definitions, the part before a
  retry or a compact) are not written again: the new file refers to them (the file is sparse, the
  block is read from the other file). A block is shared only when its token prefix matches and
  its 128-bit content hash is equal.
- **Superseded sessions.** When an agent trims its history (same system prompt and first task
  message, the middle dropped, the new conversation clearly shorter), the older session is cut down
  to the shared part once the new conversation has finished its second turn and the old one has not
  been used since. A new conversation that shares only the system prompt (a compact that replaces
  the history with a summary, or simply another conversation with the same system prompt, such as a
  subagent running while the main agent waits) never cuts the older session: it is kept whole and only
  evicted first when the RAM or SSD tier needs room (and not written to the SSD on a RAM eviction);
  any use of it undoes that. Retries, regenerations, edits of the last message and other system
  prompts never trigger this. `WHIRL_KV_SUPERSEDE=0` turns it off.
- **Short sessions** (under 2048 tokens, `WHIRL_KV_TIER_MIN`) are not kept: prefilling them again
  is about as fast as restoring them.
- **Safety.** The cache matches token prefixes exactly; a restored entry is the very bytes that were
  saved. Every file has a checksummed token list and a header written last (behind flush barriers,
  so a crash or power loss never leaves a valid header over other bytes); a file cut short, a file
  whose shared block is gone or no longer matches its recorded hash, or a file from another build,
  model, mode or KV format is never restored (other builds' files are counted against the cap and
  deleted first). The KV and checkpoint data itself is not checksummed on read: these checks catch
  crashes, power loss, truncation and stale files, not bit rot or a failing disk silently changing
  bytes inside a file. Against those faults the tier relies on the drive and file system.
- **One server per directory.** A second server pointed at a directory in use (`whirl.lock`) runs
  without an SSD tier and says so in the log; give each server its own `--kv-ssd-dir`.
- **Write volume.** The log's `kv tier |` line and `GET /props` (`kv_ssd.bytes_written`,
  `bytes_shared`, `bytes_used`, `entries`, `superseded`) show the bytes written to the SSD since
  start and the bytes shared instead of written.
- **Delete by hand.** Stop the server and delete the `*.wkv` files (or the whole directory); the
  tier starts empty. Nothing else is stored there.
- **Privacy.** The files hold your prompts as token ids (readable by anyone who can read the files)
  and their KV data. They are in your user profile with its permissions. To remove them, stop the
  server and delete the directory; `--kv-ssd-mode off` never writes them.

### <a id="stop"></a>Stopping the server

Press **Ctrl+C** (or Ctrl+Break, or close the console window; logging off and shutting down Windows
do the same). The server then stops gracefully:

1. new and queued requests are answered with HTTP 503;
2. running requests finish;
3. finished sessions are copied to the RAM tier, and the RAM-tier entries not yet current on the SSD
   are written now — with the default `--kv-ssd-mode shutdown` every reused entry (a one-shot prompt
   is not written), for at most 60 seconds; with `always` the entries its seen-once filter keeps, for
   at most 10 seconds (the log says `shutdown: tier writes done …`).
   The flush starts after the running requests have finished, and the Ctrl+C handler waits 120 s in
   all: a very long prefill still running at Ctrl+C can leave no time for the flush (the cache is
   lost, never corrupted).

The next start therefore restores recent conversations from the SSD instead of prefilling them
again. A second Ctrl+C exits immediately. When the console window is closed, or on logoff / OS
shutdown, Windows ends the process after about 5 seconds regardless, so only what fits in those
seconds is written (Ctrl+C allows the full flush). Killing the process (Task Manager, `taskkill /F`) skips
all of this; there is no shutdown endpoint.

## <a id="other"></a>5. `whirl devices`, `selftest`, `seqtest`, `vis-encode`

| Command | What it does |
|---|---|
| `whirl devices` | lists the AMD GPUs the driver reports (index, name, architecture, memory, compute units), whether this build has GPU kernels for each of them, whether it is integrated, and which one is the default. Use it first to check the driver and the GPU |
| `whirl selftest MODEL.gguf [--device SPEC]` | bitwise self-checks of the GPU kernels on the model's own weights: int8 vs f32 GEMV error, multi-token GEMV == one-token GEMV, prefill GEMM invariance over all configurations, MoE token-tile invariance, grouped vs per-query decode attention. Prints `selftest: ok` or FAIL (exit code 1) |
| `whirl seqtest MODEL.gguf [--decode N] [--device SPEC]` | runs the multi-request (server) paths against single-request runs on two prompts: segmented prefill, batched decode rows, batched verify with literal drafts. Every token must match; prints `seqtest: ok` or FAIL (exit code 1). `--decode N`: tokens per sequence (default 24) |
| `whirl vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto\|resident\|stream]` | diagnostic: encodes one image with the vision encoder and reports the time; `OUT.f32` receives the projected embeddings; `--reps N` repeats the encode N times |

## <a id="exit-codes"></a>6. Exit codes and error messages

Common failures print a plain-language message (what happened and what to do) and end with a
distinct exit code:

| Code | Meaning | Examples and what to do |
|---|---|---|
| 0 | success | |
| 1 | other error; `selftest` / `seqtest` FAIL | |
| 2 | bad command line | unknown option, missing `MODEL.gguf`, prompt + `--max-tokens` longer than `--ctx`, a `--host` address that does not belong to this computer |
| 3 | GPU / driver problem | no AMD GPU, no kernels for this GPU's architecture, `amdhip64_7.dll` missing or too old (install AMD Software: Adrenalin Edition 26.8.1 or newer), GPU busy |
| 4 | model file problem | file not found, not a GGUF, unsupported architecture (only `qwen35` / `qwen35moe`), unsupported tensor type (for example NVFP4), truncated file |
| 5 | out of GPU memory | weights or KV cache do not fit: try a smaller `--ctx`, fewer server slots, or a smaller quantization |
| 6 | server port in use | another program listens on `--host`:`--port` (checked before the model loads): use another `--port` |

The executables load the AMD GPU runtime (`amdhip64_7.dll`, installed with the graphics driver)
on demand and check it at start, so a missing or old driver is reported as code 3 rather than as a
Windows "DLL not found" dialog. Everything else, including the C++ runtime, is linked statically.

## <a id="env"></a>7. Environment variables

Every variable is read as `WHIRL_<NAME>` by both programs. Set them in PowerShell with
`$env:WHIRL_MODE = "precise"` before starting the program. **Normal use needs none of them**; the
defaults are the tested, fastest paths. Variables marked *(A/B)* or *(diagnostic)* exist for
experiments and measurements.

### 7.1 GPU and loading

| Variable | Meaning |
|---|---|
| `WHIRL_DEVICE=SPEC` | GPU to use: `r9700`, `8060s`, an index, or a name / gfx substring (= `--device`; default: the first R9700, else the first supported GPU) |
| `WHIRL_HIP_DEVICE=N` | device index, bypassing device matching and the one-process-per-GPU lock |
| `WHIRL_GPU_SHARE=1` | do not wait for other WHIRL processes on the same GPU |
| `WHIRL_GPU_WAIT=S` | seconds to wait for another WHIRL process to free the GPU (default 1800) |
| `WHIRL_MODE=precise\|balance\|fast[:ITEMS]` | numerics mode (= `--precise` / `--balance` / `--fast` / `--mode`; default balance). See [numerics modes](#modes) |
| `WHIRL_PREFILL_BATCH=N` | prefill batch rows (default 4096, up to 16384); the CLI and the server prefill on the same schedule, a forward is at most this minus 256 rows (CLI-SCHED) |
| `WHIRL_PREFILL_BATCH` below 1280 (server) | raised to 1280 with a warning: the scheduler's largest prefill forward is a 1024-row chunk plus a 256-row tail |
| `WHIRL_MAX_CTX=N` | default context size of `chat` (= `--ctx`; default 8192) |
| `WHIRL_CODE_OBJECT=FILE` | development: load the GPU kernels from this code object instead of the built-in one |
| `WHIRL_MOE_FP8=0\|1` | MoE models with MXFP4 experts: expert prefill with fp8 activations (item `moefp8`: on in balance and fast, off in precise; fp8 is the faster path — about 11.7k vs 8.6k tok/s at 2k tokens for Ornith MXFP4). Affects prefill only |
| `WHIRL_MOE_RBF=0` | MoE models with MXFP4 experts on the fp8 expert prefill: use the tile-major grouped GEMM grid instead of the row-block-fast one (default row-block-fast; same output bits, faster prefill). Affects prefill only |
| `WHIRL_MOE_MXW=0` | MoE models with MXFP4 experts: use the generic MXFP4 expert decode kernels instead of the whole-block ones (default whole-block). Affects decode / verify |
| `WHIRL_MOE_WIDE=1` | MoE models, balance / fast: verify batches up to 32 rows (default 16), the 17–32-row batches on the int8 decode experts like a 1-token decode. Same output; about 2% slower at four users on Ornith-1.5-35B-A3B (each row reads its own experts), so off by default |
| `WHIRL_VRAM_LIMIT_MB=N` | simulate a GPU with N MiB of VRAM (e.g. `16384` on an R9700 for a 16 GB RX 9070 XT): free / total VRAM, the WDDM budget and this process's allocations are capped at N, so loading, KV-pool sizing and the automatic prefill-batch / checkpoint reduction behave as on that card. Off by default |
| `WHIRL_UMA_CHECK=0` | Radeon 8060S: skip the check that the model, its buffers and the KV cache fit the GPU memory set in Armoury Crate / the BIOS (the dedicated carve-out), and the KV cache cap that keeps it there. Unsupported: memory beyond the carve-out is slower, and with a small carve-out the GPU runtime fails with hipError 719 |
| `WHIRL_FAKE_DEDICATED_GB=N` | test: pretend the Radeon 8060S has N GiB of dedicated GPU memory (e.g. `0.5`), to see the refusal message or the reduced KV cache without changing the BIOS. Off by default |
| `WHIRL_EMBD_HOST=0` | keep the token embedding table in VRAM (default: pinned host memory; the server declares its size with the other pinned memory so Shared Usage monitoring can subtract it) |
| `WHIRL_TUNE_COLD=1` | autotune: evict the cache before each timing *(diagnostic)* |
| `WHIRL_TUNE_MASK=BITS` | autotune: mask of the candidate GEMM configurations *(diagnostic)* |

### 7.2 Thinking

| Variable | Meaning |
|---|---|
| `WHIRL_THINK=0\|1` | thinking off / on for `chat` and `bench` (default on; = `--no-think` / `--think`) |

### 7.3 Speculative decoding

The output always equals plain greedy decoding (with sampling: the same distribution). Background:
[speculative-decoding.md](guide/en/speculative-decoding.md).

| Variable | Meaning |
|---|---|
| `WHIRL_MTP=0` | plain decoding, no MTP drafts |
| `WHIRL_MTP_DRAFTS=N` | fixed MTP drafts per cycle, 1–10 (default: cost model, up to 8 for dense, 1 for MoE) |
| `WHIRL_MTP_ADAPT=auto\|N` | draft count from the cost model, or ceil(recently accepted + N) |
| `WHIRL_MTP_PMIN=P` | end a draft chain at a draft whose probability is below P (after `WHIRL_MTP_NMIN` drafts) |
| `WHIRL_MTP_NMIN=N` | drafts always made before `WHIRL_MTP_PMIN` applies |
| `WHIRL_MTP_BATCH_DRAFTS=d1,d2,...` | most drafts per cycle with 1, 2, … decoding slots (server) |
| `WHIRL_MTP_Q4=0` | keep the MTP block's Q6_K / Q8_0 matrices (default: Q4_K copies, used for drafts only) |
| `WHIRL_DRAFT_HEAD=q4` | Q4_K draft head instead of the 2-bit one |
| `WHIRL_MTP_FULLHEAD=1` | drafts use the full output head |
| `WHIRL_DRAFT_VOCAB=off\|64k\|48k\|FILE\|N` | MTP draft head over a frequency subset of the vocabulary. Default: the 64k subset embedded in the executable, used only for dense qwen35 models with the 2-bit draft head and a 248,320-token vocabulary (outputs unchanged). `off` = full draft head; `48k` = `draft_vocab\subset_48k.bin` next to the exe; FILE = uint32 little-endian token ids; N = the first N rows |
| `WHIRL_NGRAM=0` | no n-gram (prompt-lookup) drafts |
| `WHIRL_DRAFT_MODEL=FILE` | DFlash2 drafter GGUF for `--draft dflash` when `--draft-model` is not given (debug override; `--draft-model` wins) |
| `WHIRL_DFLASH_PMIN=P` | `--draft dflash`: end the DFlash2 draft walk at a token whose probability is below P, 0..1 (default 0: the whole block) *(debug)* |
| `WHIRL_DFLASH_NMIN=N` | `--draft dflash`: fewer than N drafts in a block = no drafts that cycle (default 0) *(debug)* |
| `WHIRL_DFLASH_WARM=N` | server, `--draft dflash`: rows a slot's drafter context needs after a restore / reset before it drafts again (default 64) *(debug)* |
| `WHIRL_DFLASH_CAPTURE_ROWS=N` | `--draft dflash`: target-feature capture rows per forward, at least 16 (default: the drafter's window, 2048; at most the prefill batch). Smaller values save VRAM but leave gaps in the drafter context after long prefills *(debug)* |
| `WHIRL_NGRAM_MIN=N` | minimum matched suffix for an n-gram draft (default 3) |
| `WHIRL_NGRAM_SLOPE=X` | prior cost of one more n-gram verify row, relative to a 1-draft cycle, until n-gram cycles are timed (default 0.015; Radeon 8060S 0.12) |
| `WHIRL_NGRAM_MAX=N` | most n-gram drafts per cycle (default 15) |
| `WHIRL_NGRAM_FORCE=1` | take every n-gram proposal *(A/B)* |
| `WHIRL_NGRAM_DEBUG=1` | log every decode cycle to stderr *(diagnostic)* |

### 7.4 Server

| Variable | Meaning |
|---|---|
| `WHIRL_NO_PREFIX_CACHE=1` | disable the prefix cache |
| `WHIRL_PREFIX_CACHE_VERIFY=1` | re-prefill after every cache hit and compare *(diagnostic)* |
| `WHIRL_COLD_RESUME=0\|1` | resume a cached conversation only from checkpoints that are bitwise equal to a cold prefill (default: on in `--precise`, off otherwise; `0` in precise restores the faster but not bitwise 0.2.1 behaviour) |
| `WHIRL_SERVE_CKPTS=N` | prefix checkpoints per slot (default 4 with one slot, 3 up to 4 slots, else 1; at most 8; 3 or more keep a [tail checkpoint](#prefix-cache)) |
| `WHIRL_SYS_MIN=N` | system messages of at least N tokens get their own checkpoint (default 2048, 0 = off) |
| `WHIRL_SYS_CKPTS=N` | shared prefix checkpoints kept in VRAM (default 2) |
| `WHIRL_SYS_LCP=0` | no checkpoints at prefixes common to sessions |
| `WHIRL_CKPT_HOST=1` | keep prefix checkpoints in pinned host memory instead of VRAM |
| `WHIRL_GOVERNOR=0` | turns the server's runtime memory governor off (0.2.2.2): when the WDDM budget drops while the server runs, it frees cache, queues new requests and re-allocates the KV pool smaller when every slot is idle, and grows it back when the budget recovers (server guide, section 7) |
| `WHIRL_POOL_BUDGET_MARGIN_MB=N` | room kept free under the Windows (WDDM) video memory budget for other programs to grow, when the KV pool is sized and re-checked. Default: the larger of 2048 MiB (3072 when a monitor is attached to the GPU) and 6% of the budget. Other programs' VRAM at startup that the budget does not show yet is kept free on top |
| `WHIRL_POOL_RESERVE_MB=N` | old free-VRAM-only rule: VRAM left free when the KV pool is sized, with the WDDM budget, its margin and the pool re-check turned off (default reserve 768, MoE 1536). To keep more room for other programs, use `WHIRL_POOL_BUDGET_MARGIN_MB` |
| `WHIRL_VRAM_HEADROOM_MB=N` | extra VRAM kept free for the desktop and other programs, on top of the reserve and the WDDM margin. Default 0 (since 0.2.2.1 the WDDM margin covers the desktop: 3 GiB when a monitor is attached; it was 1536 on cards with less than 20 GiB). Cards with less than 20 GiB (16 GB RX 9070 / 9070 XT / 9060 XT) still default to `--parallel 1` (the KV format is the mode's, as on any card) |
| `WHIRL_PREFILL_CHUNK=N` | most rows per merged prefill forward (multiple of 1024, default 2048, Radeon 8060S MoE models 4096; capped at the prefill batch) |
| `WHIRL_SEG_PREFILL=0` | prefill each request on its own instead of several in one forward |
| `WHIRL_GATHER_MS=MS` | window to gather a burst of new requests (default 30, 0 = off) |
| `WHIRL_DECODE_MIN_TPS=N` | decode floor per streaming request while others prefill (= `--decode-min-tps`, default 20, 0 = off) |
| `WHIRL_GDN_REPLAY=0` | DeltaNet verify with snapshot sets instead of replaying kept rows |
| `WHIRL_SNAP_SETS=N` | minimum number of recurrent-state snapshot sets (with `WHIRL_GDN_REPLAY=0`) |
| `WHIRL_SLOT_DRAFTS=1` | split the draft budget between slots by expected acceptance |
| `WHIRL_TIMING_RESET=0` | keep the MTP timing table across requests (not recommended) |
| `WHIRL_KV_RAM_MB=N` | host RAM tier in MiB (= `--kv-ram-mb`) |
| `WHIRL_KV_SSD_DIR=PATH` | SSD tier directory (= `--kv-ssd-dir`) |
| `WHIRL_KV_SSD_GB=N` | SSD tier size cap in GiB (= `--kv-ssd-gb`) |
| `WHIRL_KV_TIER_MIN=N` | smallest session copied to the host tiers, in tokens (default 2048) |
| `WHIRL_KV_SSD_DELAY_MS=MS` | delay before a RAM-tier entry is written to the SSD in `always` mode (default 300000; `--kv-ssd-delay` overrides) |
| `WHIRL_KV_SSD_EXPIRE_S=S` | delete SSD cache files unused for S seconds (default 604800 = 7 days; for tests) |
| `WHIRL_KV_SSD_MODE=always\|shutdown\|off` | = `--kv-ssd-mode` |
| `WHIRL_KV_SSD_DEDUP=0` | write every block even when another SSD file holds it (default: shared) |
| `WHIRL_KV_SUPERSEDE=0` | keep sessions superseded by an agent's trim (default: cut to the shared part when the first task message is shared too; evicted first when only the system prompt is) |
| `WHIRL_MMPROJ=FILE` | vision encoder (= `--mmproj`) |
| `WHIRL_VIS_IDLE_S=N` | release the vision encoder after N s without images (= `--vis-idle-s`) |
| `WHIRL_VIS_CACHE_MB=N` | image embedding cache in MiB (= `--vis-cache-mb`) |
| `WHIRL_VIS_CKPT_MIN=N` | keep a shared checkpoint after images ending at ≥ N tokens (default 1024, 0 = off) |

### 7.5 Vision

| Variable | Meaning |
|---|---|
| `WHIRL_VIS_MODE=auto\|resident\|stream` | encoder weights resident in VRAM or streamed layer by layer (= `--vis-mode`) |
| `WHIRL_VIS_DUMP=PREFIX` | write each image's embeddings to `PREFIX.<n>.f32` *(diagnostic)* |
| `WHIRL_VIS_DUMP_RGB=FILE` | write the preprocessed image *(diagnostic)* |
| `WHIRL_VIS_PRE=FILE` | preprocess this image instead *(diagnostic)* |
| `WHIRL_VIS_PROF=1` | per-stage encoder timing *(diagnostic)* |

### <a id="env-numerics"></a>7.6 Numerics and speed switches *(A/B)*

Alternatives kept for A/B tests and numerics comparisons. The lossy ones are the items of the
[numerics modes](#modes): their defaults follow the mode, and a value set here overrides it.
"Bitwise-equal" alternatives give the same bits and differ only in speed.

| Variable | Meaning |
|---|---|
| `WHIRL_Q8DEC=0\|1` | decode / verify activations: `1` int8 (item `q8dec`, default in balance / fast; in precise a lossy override), `0` f16 into the f16 GEMM (default in precise) |
| `WHIRL_FP8=0\|1` | MXFP4 prefill with fp8 activations (item `fp8`; default: on in balance / fast). `0` also turns `WHIRL_MOE_FP8` off |
| `WHIRL_FP8_MASK=BITS` | matmul classes that use fp8 activations (default 7) |
| `WHIRL_G8T=0` | row-major fp8 GEMM instead of the fragment-tiled one (same bits) |
| `WHIRL_GDN_WMMA=0\|1` | f16-WMMA DeltaNet prefill chunks (item `gdnwmma`; default: on in balance / fast for MXFP4 models and, on the R9700, dense models of any format) |
| `WHIRL_SPEC_SAMPLE=0\|1` | speculative sampling of MTP drafts for temperature > 0 (item `specsample`; default: on in balance / fast, off in precise) |
| `WHIRL_RELAX=0\|1`, `WHIRL_RELAX_K`, `WHIRL_RELAX_ALPHA`, `WHIRL_RELAX_EPS`, `WHIRL_RELAX_DELTA` | relaxed speculative acceptance (item `relaxacc`; default: on in fast only, k 4, alpha 0.1, eps 0.09, delta 0.3; `WHIRL_RELAX_K=1` = exact greedy acceptance). Changes greedy output |
| `WHIRL_FFN_H16=0\|1` | f16 GEMM outputs into the element-wise ops (item `h16`; default: on in balance / fast for MXFP4 models and, on the R9700, dense models of any format) |
| `WHIRL_Q4_RELAXED=1` | the balance-mode prefill switches (`gdnwmma`, `h16`) for the models that skip them (non-MXFP4 MoE models, the Radeon 8060S) too |
| `WHIRL_ACT_FUSE=0` | no fused activation in prefill (bitwise-equal) |
| `WHIRL_GEMMH=0` | no f16-output prefill GEMM (bitwise-equal) |
| `WHIRL_GEMMHQ=1` | f16-output GEMM for the attention projections too |
| `WHIRL_ATTN_KX=0` | prefill attention without the K-exchange kernel (bitwise-equal) |
| `WHIRL_ATTN_KG=0` | prefill attention without the GQA-grouped kernel (bitwise-equal) |
| `WHIRL_ATTN_DQF=0` | q8 / q8h (and q4) KV prefill attention on the quantized cache (`attn_kg_q8`; Radeon 8060S `attn_prefill_wmma_q8` / `_q4`) instead of an f16 copy of the sequence's keys + the f16 kernel (bitwise-equal) |
| `WHIRL_ATTN_SEG=N` | Radeon 8060S: prefill attention in key ranges of N keys (multiple of 256) with the softmax state carried (`attn_prefill_wmma_s`), also for f16 KV (bitwise-equal; a test knob, no speed gain end to end) |
| `WHIRL_GDN_SEQ=1` | sequential DeltaNet prefill instead of the chunked scan |
| `WHIRL_GDN_V0=1` | per-row DeltaNet decode step kernel (same values) |
| `WHIRL_NAIVE_ATTN=1` | reference attention path |
| `WHIRL_NO_FUSE=1` | no fused decode kernels (reference path) |
| `WHIRL_FLOAT_GEMV=1` | f32 GEMV instead of int8 (reference path) |
| `WHIRL_NO_GRAPH=1` | no HIP graph for the plain decode step |
| `WHIRL_GEMV_MAX=N` | most tokens per multi-token GEMV |
| `WHIRL_GEMV_R=nt:r,...` | multi-token GEMV rows-per-block overrides |
| `WHIRL_GEMV_W=nt:v,...` | multi-token GEMV kernel variant overrides |
| `WHIRL_GEMV_WH=nt:v,...` | multi-token GEMV variant overrides (f16 output) |
| `WHIRL_GV_GROUP=0` | no grouped same-input GEMV launches |
| `WHIRL_GV_NMAX=N` | largest group of same-input GEMVs |
| `WHIRL_MOE_BN=32\|64` | grouped expert GEMM token tile |
| `WHIRL_DBG=BITS` | 1 = no gdn_abconv merge, 2 = scalar split attention, 4 = one query per attention group |
| `WHIRL_ATTN_WIDE=0` | verify attention groups of at most 16 columns (`attn_wsplit1`) instead of up to 32 (`attn_wsplit2`) |
| `WHIRL_ATTN_WS4=0` | gfx1151 (8060S), f16 / q8h KV: the old one-row-per-group decode / verify attention (`attn_wsplit1`) instead of `attn_ws4` / `attn_ws4g` (same output bits, slower at long context; debug fallback) |
| `WHIRL_ATTN_DQ4=0` | q4 KV: decode / verify attention through `attn_wsplit1_q4` (slower; before 0.2.0) instead of `attn_dq4` |

### <a id="env-debug"></a>7.7 Diagnostics and debugging

| Variable | Meaning |
|---|---|
| `WHIRL_KV=f16\|q8\|q8h\|q8v\|q4` | *(debug)* force a KV cache format instead of the mode's ([numerics modes](#modes)); `auto` or unset = the mode decides. A forced format skips the fit check (the pool is sized from what is free) and in precise mode a quantized value is logged as a user-requested lossy override. A format the GPU has no kernels for is refused. See [kv-and-caching.md](guide/en/kv-and-caching.md#formats) |
| `WHIRL_DQ4_DYN=0` | Radeon 8060S, q4 KV (fast, MoE): launch the fixed attention split grid instead of only the splits that hold keys (default on; output is the same either way) |
| `WHIRL_FD_SPLITS=64\|128\|192\|256` | *(debug)* split count of the decode / verify attention (default 64, q4 KV 256). Another value changes the attention's partial sums, so output bits differ from the default |
| `WHIRL_FD_SPLITS`, `WHIRL_GEMV_MAX`, `WHIRL_GV_NMAX`, `WHIRL_GV_GROUP`, `WHIRL_NO_FUSE`, `WHIRL_FLOAT_GEMV`, `WHIRL_GDN_SEQ`, `WHIRL_NAIVE_ATTN`, `WHIRL_DBG`, `WHIRL_ATTN_WIDE`, `WHIRL_ATTN_DQ4`, `WHIRL_ATTN_WS4`, `WHIRL_MOE_BN` | A/B switches that pick other kernel paths. Any of them set is logged at start-up as an override and the mode label becomes `MODE+overrides` (also in `GET /props`): the run is not the mode's tested path, and e.g. `WHIRL_GEMV_MAX=8` in balance makes 9–16-row verifies take another GEMM than 1-token decode, so MTP is no longer equal to plain decode |
| `WHIRL_TOKENIZE_ONLY=1` | print the prompt token ids and stop |
| `WHIRL_PRINT_IDS=1` | print the generated token ids and their FNV-1a hash |
| `WHIRL_PROFILE=1` | per-op-class GPU time; distorts speed numbers (`whirl bench`: per prefill size; server: 1 or 2) |
| `WHIRL_TRACE_TPS=N` | print the window tok/s every N tokens (stderr) |
| `WHIRL_DUMP_LOGITS=FILE` | (no MTP) next-token logits of every prompt position ≥ `WHIRL_DUMP_FROM` as f16 rows |
| `WHIRL_DUMP_FROM=N` | first prompt position written by `WHIRL_DUMP_LOGITS` |
| `WHIRL_DUMP_MOE=FILE` | selected experts of every prefill MoE block (i32) |
| `WHIRL_TRACE_ND=1` | log the draft count of every decode cycle |
| `WHIRL_LOOP_LOG=1` | log per-loop phase timing (server) |
| `WHIRL_TIER_VERIFY=1` | byte-verify every host-tier spill / restore (server) |
| `WHIRL_TAIL_CK=0` | *(debug)* no [tail checkpoint](#prefix-cache) near the end of long prompts (logged as an override); a second question about the same document then re-prefills it |
| `WHIRL_TIER_MIN_GAIN=N` | restore from a host tier only when it saves at least N tokens (default 512) |

## <a id="files"></a>8. Files WHIRL writes

| Path | Content |
|---|---|
| `%LOCALAPPDATA%\whirl\tune4-<model file name>-<file size>…txt` | the tuned prefill GEMM configuration per model file and GPU. Delete it to tune again |
| `%LOCALAPPDATA%\whirl\kvcache\` | SSD tier of the server's prefix cache (`--kv-ssd-dir`, capped by `--kv-ssd-gb`) |
| `%LOCALAPPDATA%\whirl\server.log` | server log (`--log-file`) |

Nothing is written next to the executables, and nothing is installed system-wide.
