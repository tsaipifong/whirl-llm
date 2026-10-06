**English** | [繁體中文](guide/zh-TW/usage.md)

# Usage reference: `whirl` and `whirl-server`

This page lists every command, option, environment variable and server endpoint of WHIRL 0.1.1.
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
(a message on stderr says so): about 1.5–2 minutes for a 27B Q4_K_M file, a few seconds for MXFP4
files. The result is cached ([section 8](#files)); later runs start right away. Every candidate
configuration gives the same bits, only the speed differs.

**One process per GPU.** A WHIRL process holds a per-GPU lock while it runs. A second WHIRL process
on the same GPU waits for it (up to 30 minutes by default) instead of sharing VRAM, because two
large processes on one GPU make Windows move both to slow shared memory
([windows-hip.md](guide/en/windows-hip.md#wddm-demote)). See `WHIRL_GPU_WAIT` / `WHIRL_GPU_SHARE`.

### <a id="modes"></a>Numerics modes: precise (default), balance, fast

One mode per process, chosen at start-up (`--precise`, `--balance`, `--fast`, or `--mode M`; or the
environment variable `WHIRL_MODE=precise|balance|fast`; the command line wins). It applies to
`chat`, `bench`, `selftest`, `seqtest` and the server alike, on the R9700 and the Radeon 8060S.

| Mode | What it does |
|---|---|
| **precise** (default) | The GGUF weights are dequantized to f16; prefill activations and accumulation are f16 / f32; DeltaNet prefill uses the f32 chunk path; **the KV cache is always f16** on every card. Nothing is quantized beyond the file's own weights (one exception still pending, see `q8dec` below). If f16 KV does not fit, the context is lowered with a warning when you did not give it, or the program stops with a message (exit code 3) when you did (`--ctx`, `--ctx-per-slot`) — it never switches to int8 KV by itself |
| **balance** | The speed-oriented defaults of WHIRL 0.1.x / 0.2.0-rc, exactly: the items below. The output may differ slightly from precise (measured KL / accuracy in [quantization.md](guide/en/quantization.md)) |
| **fast** | balance plus more aggressive lossy items that must pass KL + paired-accuracy gates first. None is implemented yet: they are listed as skipped and fast currently runs as balance |

Items (pick a subset with `--balance=fp8,kvq8` or `--fast=...`; `WHIRL_MODE=balance:fp8,kvq8`):

| Item | Mode | What it is | R9700 (gfx1201) | Radeon 8060S (gfx1151) |
|---|---|---|---|---|
| `fp8` | balance | MXFP4 dense prefill GEMMs with fp8 (e4m3) activations, incl. the rounding of folded MXFP4 exponents | MXFP4 models | skipped (no fp8 WMMA) |
| `moefp8` | balance | MoE expert prefill GEMMs with fp8 activations | MXFP4 MoE (Ornith MXFP4) | skipped |
| `gdnwmma` | balance | f16-WMMA DeltaNet prefill chunks instead of the f32 chunk path | MXFP4 models | skipped (no kernel yet) |
| `h16` | balance | f16 FFN / DeltaNet GEMM outputs before the element-wise ops | MXFP4 models | f16 weights only |
| `kvq8` | balance | KV auto may pick int8 when f16 does not fit: q8v (K f16, V int8), then q8h; and q8v first on cards under 20 GiB | dense models | q8 (dense models) |
| `kvq4`, `relaxacc`, `headq`, `moeskip`, `a8`, `a4` | fast | 4-bit KV, relaxed speculative acceptance, low-bit output head, MoE expert skipping, W4A8 / W4A4 prefill | not yet implemented | not yet implemented |
| `q8dec` | all modes | int8 activations (one f32 scale per 32 values, as llama.cpp's q8_1) in the decode / verify GEMV | on | on |

`q8dec` is still on in precise mode: a float-activation decode GEMV that keeps MTP == plain is being
written. Items that do not apply to the GPU or model are logged as skipped (never an error). The
start-up log shows one line, e.g.

```
numerics: balance - enabled: fp8, gdnwmma, h16, kvq8 (auto: q8v, then q8h, when f16 does not fit); skipped: moefp8 (dense model); ...
```

and the server reports the mode in `GET /props` (`"numerics": {"mode", "label", "enabled", "skipped", "overrides", "kv", "always_on"}`).
The per-item variables of [7.6](#env-numerics) (`WHIRL_FP8`, `WHIRL_MOE_FP8`, `WHIRL_GDN_WMMA`,
`WHIRL_FFN_H16`, `WHIRL_Q4_RELAXED`) and `WHIRL_KV` still work and override the mode; in precise
mode a lossy value is logged as `user-requested` and the mode reads `precise+overrides`.
MTP / n-gram decoding equals plain greedy decoding, and concurrent requests equal solo runs, within
each mode.

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
| `--mmproj MMPROJ.gguf` | vision encoder (Qwen3-VL style mmproj, F16 / BF16), needed for `--image` |
| `--image IMAGE` | an image placed before the prompt text; repeatable |
| `--device SPEC` | GPU: `r9700`, `8060s`, an index, or a substring of the name / gfx architecture (also `WHIRL_DEVICE`). Default: the first R9700, else the first GPU the build has kernels for (a Radeon 8060S on its own) |
| `--precise` / `--balance` / `--fast` / `--mode M` | numerics mode (default precise; also `WHIRL_MODE`); `--balance=ITEMS` picks items. See [numerics modes](#modes) |
| `-h`, `--help` | help |

Examples:

```powershell
.\whirl.exe chat C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf "Explain TCP slow start in two sentences." --max-tokens 400
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
| `-c`, `--ctx N` | size of the shared KV pool in tokens. Slots take pages on demand; when the pool is full, idle slots' prefix caches are evicted (least recently used first). Default: all VRAM left after weights and buffers minus 768 MiB (MoE: 1.5 GiB) |
| `--ctx-per-slot N` | longest context of one request (default min(pool, 131072); up to 262144) |
| `--precise` / `--balance` / `--fast` / `--mode M` | numerics mode (default precise: f16 KV; when the f16 pool cannot hold a full request the context per request is lowered with a warning, or the server refuses to start if `--ctx` / `--ctx-per-slot` was given). See [numerics modes](#modes) |
| `--mtp-drafts N` | fixed MTP drafts per cycle, 1–10 (default: chosen per model type by a cost model) |
| `--decode-min-tps N` | decode floor: while other requests prefill, every streaming (decoding) request keeps at least N tok/s; prefill forwards are shortened and decode cycles interleaved to hold it (default 20; `0` = off, prefill forwards are not limited). Outputs are identical for any N ([server.md](guide/en/server.md#batching)) |
| `--kv-ram-mb N` | host RAM tier of the prefix cache, MiB of pinned memory (default: 1/4 of physical RAM, at least 8 GiB or one full-length session if that is larger — about 9 GiB for the 27B model —, at most 32 GiB, and at most half of the RAM available at startup; 16 GiB on a 64 GB PC; off by default on integrated GPUs). The startup log prints the chosen size and why. Idle sessions are copied there and restored instead of prefilled again. `0` disables both host tiers. **Radeon 8060S** (integrated GPU): default `0` — the KV pool already lives in system memory; give a size to turn the RAM and SSD tiers on |
| `--kv-ssd-dir PATH` | SSD tier directory (default `%LOCALAPPDATA%\whirl\kvcache`) |
| `--kv-ssd-gb N` | SSD tier size cap in GiB (default 64; `0` = no SSD tier) |
| `--mmproj FILE` | vision encoder (Qwen3-VL style mmproj GGUF, F16 / BF16). `image_url` parts (`data:` URLs with base64 PNG / JPEG / …) become image tokens. Weights stay in pinned host RAM; nothing goes to VRAM until an image arrives |
| `--vis-idle-s N` | release the vision encoder after N seconds without images (default 60) |
| `--vis-mode M` | `auto` (default), `resident` (encoder weights in VRAM), `stream` (layer by layer) |
| `--vis-cache-mb N` | host cache of image embeddings, keyed by content hash, MiB (default 1024) |
| `--allow-local-images` | also accept local file paths / `file://` URLs as image sources (off by default; `http(s)` image URLs are never fetched) |
| `--cors-origin ORIGIN` | let web pages from `ORIGIN` (e.g. `https://app.example.com`) call the server from a browser; repeatable. By default only pages served from `http(s)://localhost`, `127.0.0.1` or `[::1]` (any port) get CORS headers (their `Origin` is echoed back, never `*`); other web pages, including `file://` pages, cannot read responses. `*` restores the behaviour before v0.1.4 (`Access-Control-Allow-Origin: *` for every page). Non-browser clients (SDKs, curl, IDE agents) are not affected |
| `--log-file PATH` | log file (default `%LOCALAPPDATA%\whirl\server.log`); the log also goes to the console |
| `-h`, `--help` / `-V`, `--version` | help / version |

Examples:

```powershell
# 27B model, 4 slots, default tiers
.\whirl-server.exe C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080

# one user, long contexts, no SSD tier
.\whirl-server.exe C:\models\model.gguf -np 1 --ctx-per-slot 262144 --kv-ssd-gb 0

# with image input
.\whirl-server.exe C:\models\model.gguf --mmproj C:\models\mmproj-F16.gguf
```

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
| `GET /health` | answers immediately even while busy; reports the busy state and queue length |
| `GET /props` (also `/v1/props`) | read-only, llama.cpp-server-style subset for clients that auto-detect the context length: `default_generation_settings.n_ctx` (context per slot), `default_generation_settings.model` and `model_alias` (= `--alias`), `total_slots`, `model_path` (file name only, never a directory), `modalities`, `build_info` |
| `GET /version` | `{"version":"0.1.1","name":"whirl"}` |
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
enforced `tool_choice: "required"` are not supported. Responses add `usage.cached_tokens` and a
llama.cpp-style `timings` object. Details: [server.md](guide/en/server.md#sampling).

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

### <a id="stop"></a>Stopping the server

Press **Ctrl+C** (or Ctrl+Break, or close the console window). The server then stops gracefully:

1. new and queued requests are answered with HTTP 503;
2. running requests finish;
3. finished sessions are copied to the RAM tier, and every RAM-tier entry that is not on the SSD yet
   is written now — for at most 10 seconds (the log says `shutdown: tier writes done …`).

The next start therefore restores recent conversations from the SSD instead of prefilling them
again. A second Ctrl+C exits immediately. When the console window is closed, Windows ends the
process after about 5 seconds regardless. Killing the process (Task Manager, `taskkill /F`) skips
all of this; there is no shutdown endpoint.

## <a id="other"></a>5. `whirl devices`, `selftest`, `seqtest`, `vis-encode`

| Command | What it does |
|---|---|
| `whirl devices` | lists the AMD GPUs the driver reports (index, name, architecture, memory, compute units), whether this build has GPU kernels for each of them (the Radeon 8060S is marked *preview, untuned*), whether it is integrated, and which one is the default. Use it first to check the driver and the GPU |
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
| 5 | out of GPU memory | weights or KV cache do not fit: try a smaller `--ctx`, `WHIRL_KV=q8v`, or fewer server slots |
| 6 | server port in use | another program listens on `--host`:`--port` (checked before the model loads): use another `--port` |

The executables load the AMD GPU runtime (`amdhip64_7.dll`, installed with the graphics driver)
on demand and check it at start, so a missing or old driver is reported as code 3 rather than as a
Windows "DLL not found" dialog. Everything else, including the C++ runtime, is linked statically.

## <a id="env"></a>7. Environment variables

Every variable is read as `WHIRL_<NAME>` by both programs. Set them in PowerShell with
`$env:WHIRL_KV = "q8v"` before starting the program. **Normal use needs none of them**; the
defaults are the tested, fastest paths. Variables marked *(A/B)* or *(diagnostic)* exist for
experiments and measurements.

### 7.1 GPU and loading

| Variable | Meaning |
|---|---|
| `WHIRL_DEVICE=SPEC` | GPU to use: `r9700`, `8060s`, an index, or a name / gfx substring (= `--device`; default: the first R9700, else the first supported GPU) |
| `WHIRL_HIP_DEVICE=N` | device index, bypassing device matching and the one-process-per-GPU lock |
| `WHIRL_GPU_SHARE=1` | do not wait for other WHIRL processes on the same GPU |
| `WHIRL_GPU_WAIT=S` | seconds to wait for another WHIRL process to free the GPU (default 1800) |
| `WHIRL_MODE=precise\|balance\|fast[:ITEMS]` | numerics mode (= `--precise` / `--balance` / `--fast` / `--mode`; default precise). See [numerics modes](#modes) |
| `WHIRL_KV=auto\|f16\|q8\|q8h\|q8v` | KV cache format. `auto` (default): f16; in balance / fast mode dense models fall back to q8v, then q8h, when f16 does not fit; MoE always f16. In precise mode a q8 value is a user-requested lossy override. The Radeon 8060S has no q8v / q8h kernels: auto falls back to q8 there, and `q8v` / `q8h` are refused. See [kv-and-caching.md](guide/en/kv-and-caching.md#formats) |
| `WHIRL_PREFILL_BATCH=N` | prefill rows per forward (default 4096, up to 16384) |
| `WHIRL_MAX_CTX=N` | default context size of `chat` (= `--ctx`; default 8192) |
| `WHIRL_CODE_OBJECT=FILE` | development: load the GPU kernels from this code object instead of the built-in one |
| `WHIRL_MOE_FP8=0\|1` | MoE models with MXFP4 experts: expert prefill with fp8 activations (item `moefp8`: on in balance / fast, off in precise; fp8 is the faster path — about 11.7k vs 8.6k tok/s at 2k tokens for Ornith MXFP4). Affects prefill only |
| `WHIRL_MOE_RBF=0` | MoE models with MXFP4 experts on the fp8 expert prefill: use the tile-major grouped GEMM grid instead of the row-block-fast one (default row-block-fast; same output bits, faster prefill). Affects prefill only |
| `WHIRL_MOE_MXW=0` | MoE models with MXFP4 experts: use the generic MXFP4 expert decode kernels instead of the whole-block ones (default whole-block). Affects decode / verify |
| `WHIRL_VRAM_LIMIT_MB=N` | simulate a GPU with N MiB of VRAM (e.g. `16384` on an R9700 for a 16 GB RX 9070 XT): free / total VRAM, the WDDM budget and this process's allocations are capped at N, so loading, KV-pool sizing and the automatic prefill-batch / checkpoint reduction behave as on that card. Off by default |
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
| `WHIRL_SERVE_CKPTS=N` | prefix checkpoints per slot (default 4 with one slot, 2 up to 4 slots, else 1) |
| `WHIRL_SYS_MIN=N` | system messages of at least N tokens get their own checkpoint (default 2048, 0 = off) |
| `WHIRL_SYS_CKPTS=N` | shared prefix checkpoints kept in VRAM (default 2) |
| `WHIRL_SYS_LCP=0` | no checkpoints at prefixes common to sessions |
| `WHIRL_CKPT_HOST=1` | keep prefix checkpoints in pinned host memory instead of VRAM |
| `WHIRL_POOL_RESERVE_MB=N` | VRAM left free when the KV pool is sized (default 768, MoE 1536) |
| `WHIRL_VRAM_HEADROOM_MB=N` | VRAM kept free for the desktop and other programs, on top of the reserve. Default 1536 on cards with less than 20 GiB (16 GB RX 9070 / 9070 XT / 9060 XT, which usually also drive the display), where the server also defaults to `--parallel 1` and KV auto picks q8v first (`WHIRL_KV=f16` still forces f16); 0 on larger cards, whose defaults are unchanged. 0 = no headroom |
| `WHIRL_PREFILL_CHUNK=N` | most rows per merged prefill forward (multiple of 1024, default 2048) |
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
| `WHIRL_KV_SSD_DELAY_MS=MS` | delay before a RAM-tier entry is also written to the SSD (default 2000) |
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
| `WHIRL_FP8=0\|1` | MXFP4 prefill with fp8 activations (item `fp8`; default: on in balance / fast). `0` also turns `WHIRL_MOE_FP8` off |
| `WHIRL_FP8_MASK=BITS` | matmul classes that use fp8 activations (default 7) |
| `WHIRL_G8T=0` | row-major fp8 GEMM instead of the fragment-tiled one (same bits) |
| `WHIRL_GDN_WMMA=0\|1` | f16-WMMA DeltaNet prefill chunks (item `gdnwmma`; default: on for MXFP4 in balance / fast) |
| `WHIRL_FFN_H16=0\|1` | f16 GEMM outputs into the element-wise ops (item `h16`; default: on for MXFP4 in balance / fast) |
| `WHIRL_Q4_RELAXED=1` | the MXFP4 balance-mode prefill switches (`gdnwmma`, `h16`) for other models too |
| `WHIRL_ACT_FUSE=0` | no fused activation in prefill (bitwise-equal) |
| `WHIRL_GEMMH=0` | no f16-output prefill GEMM (bitwise-equal) |
| `WHIRL_GEMMHQ=1` | f16-output GEMM for the attention projections too |
| `WHIRL_ATTN_KX=0` | prefill attention without the K-exchange kernel (bitwise-equal) |
| `WHIRL_ATTN_KG=0` | prefill attention without the GQA-grouped kernel (bitwise-equal) |
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

### 7.7 Diagnostics

| Variable | Meaning |
|---|---|
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
| `WHIRL_TIER_MIN_GAIN=N` | restore from a host tier only when it saves at least N tokens (default 512) |

## <a id="files"></a>8. Files WHIRL writes

| Path | Content |
|---|---|
| `%LOCALAPPDATA%\whirl\tune4-<model file name>-<file size>…txt` | the tuned prefill GEMM configuration per model file and GPU. Delete it to tune again |
| `%LOCALAPPDATA%\whirl\kvcache\` | SSD tier of the server's prefix cache (`--kv-ssd-dir`, capped by `--kv-ssd-gb`) |
| `%LOCALAPPDATA%\whirl\server.log` | server log (`--log-file`) |

Nothing is written next to the executables, and nothing is installed system-wide.
