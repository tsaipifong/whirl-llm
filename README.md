**English** | [繁體中文](docs/README_zh-TW.md)

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL is an LLM inference engine for the AMD Radeon AI PRO R9700 (RDNA 4) on Windows.

**Design principles**
1. **Built for AMD GPUs on Windows.** A native Windows C++/HIP inference engine with hand-written, hand-tuned kernels for RDNA. No WSL, no Docker, no Linux VM: install the AMD driver and run.
2. **Fully open, kernels included.** Everything from host-side scheduling to every GPU kernel is open source under Apache-2.0, with no closed binary components. Bug reports, benchmark results and code contributions are all welcome. Let's make AMD inference on Windows as good as it can be.
3. **Accuracy before speed.** Every speedup (speculative decoding, batching, prefix caching, cache restore) produces output bit-identical to plain greedy decoding, enforced by standing tests. WHIRL never trades away accuracy beyond the model you chose.
4. **Few models, pushed to the limit.** WHIRL doesn't try to run everything. It targets specific models on specific hardware and pushes them close to the hardware limit, adding architectures one at a time.
5. **Transparent, verifiable numbers.** Every figure is measured, with methods, commands and raw data published. For content-dependent features such as speculative decoding, we report the best case, the typical case and the worst case, so you know what to expect.
6. **Designed for real workflows.** Coding agents and long contexts come first: fast long-prompt processing, a responsive main conversation while others share the GPU, and a VRAM → RAM → SSD cache that survives server restarts.

The architectures supported today are **`qwen35` (dense)** and **`qwen35moe` (mixture of experts)**; more will be added one by one. WHIRL follows the design philosophy of [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0) — build for selected models and selected GPUs, as close to the hardware limit as possible; [zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4.

## WHIRL vs llama.cpp at a glance

Same GGUF file, same prompts, one R9700, greedy decoding; llama.cpp b11214 (ROCm) with the fastest flags we found for each row. Every WHIRL speedup gives output bit-identical to plain greedy decoding ([why](#speed-without-changing-the-answer)).

| | Swift-1.5 27B MXFP4-A (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE) |
|---|---|---|
| Decode tok/s, coding prompts | 112.4 vs 60.8 (**1.85×**) | 257.6 vs 118.8 (**2.17×**) |
| Server, 4 users at once, aggregate tok/s | 182.1 vs 64.0 (**2.84×**) | 381.1 vs 176.8 (**2.16×**) |
| Prefill tok/s, 8k-token prompt | 3,469 vs 1,338 (**2.59×**) | 11,258 vs 4,637 (**2.43×**) |
| Prefill tok/s, 32k-token prompt | 2,907 vs 1,174 (**2.48×**) | 8,633 vs 3,778 (**2.29×**) |
| Prefill tok/s, 128k-token prompt | 1,757 vs 791.3 (**2.22×**) | 4,311 vs 2,239 (**1.93×**) |
| Prefill tok/s, 256k-token prompt | 969.8 vs 556.1 (**1.74×**)¹ | 2,135 vs 1,442 (**1.48×**)¹ |
| Decode tok/s after a 256k-token prompt | 35.1 vs 16.8 (**2.09×**)¹ | 115.6 vs 53.7 (**2.15×**)¹ |
| Time to first token, reusing a 26k-token system prompt | 0.108 s vs 0.384 s (**3.6×**) | 0.098 s vs 0.182 s (**1.9×**) |

¹ KV cache: WHIRL int8 (q8h) / llama.cpp f16; llama.cpp decodes without speculative decoding at 256k.

![Prefill throughput vs prompt length, WHIRL 0.1.3 vs llama.cpp](docs/images/bench_prefill.png)

![Decode speed on coding prompts, WHIRL 0.1.3 vs llama.cpp](docs/images/bench_decode_zh.png)

Full table (including Qwen3.8 Q4_K_M and where WHIRL does *not* lead by much): [Performance](#performance).

## Speed without changing the answer

Every acceleration in WHIRL is checked to produce **the same tokens, bit for bit, as plain greedy decoding** of the same model file:

| Acceleration | Guarantee | How it is checked |
|---|---|---|
| MTP and n-gram speculative decoding | the verified output is bit-identical to plain greedy decoding | built-in self-check `whirl selftest` (multi-row kernels vs one row, on the model's own weights), kernel tests, server gate |
| Several users batched together | each request's output is bit-identical to running it alone | kernel tests (row-invariant GEMMs and attention), server gate |
| Prefix-cache hits on shared system prompts and shared prefixes | bit-identical to processing the prompt from scratch (checkpoints sit on the prompt's own chunk schedule) | server gate |
| Restoring a session from the RAM or SSD tier | byte-identical to the state that was saved | byte comparison on spill (`TIER_VERIFY`), server gate |

What WHIRL does **not** do to go faster: no KV cache below 8 bits (no 4-bit KV), no requantizing the model's weights to fewer bits (only the draft-only MTP and draft-head copies, which cannot change the output), no fp8 attention.

To be precise about what this means:

- **Weights** are whatever GGUF you choose; their quantization is the file author's choice, and WHIRL computes with it as stored.
- **KV cache** defaults to f16. On the dense models, when f16 does not leave room for a long context, the server falls back to int8: `q8v` (f16 keys, int8 values), then `q8h` (int8 keys and values); the MoE model stays f16. `WHIRL_KV` pins the format. Both are numerically lossy compared with an unquantized model, like any quantized inference. WHIRL does not claim "zero loss".
- **Continuing a conversation** reuses the KV and DeltaNet state computed for the previous turn, including the tokens generated by decoding. They come from the decode kernels (and from a different split of the prompt into chunks), so they are numerically equivalent but not bit-identical to re-processing the whole history as one prompt; the difference is documented and measured ([kv-and-caching.md](docs/guide/en/kv-and-caching.md)).
- The MXFP4 files run in **speed mode**: prefill uses fp8 activations in the MXFP4 matrix multiplies and f16 in parts of the DeltaNet prefill. Decode and verification always use the same int8 path in both modes, so the guarantees above hold in both.

## Requirements

| Item | Details |
|---|---|
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201, 32 GB) — required; single GPU only |
| OS | **Windows 11**, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1** (driver 32.0.31041.1004) or newer |
| To run | only the driver (it installs `amdhip64_7.dll` and `amd_comgr_3.dll`). No HIP SDK, no ROCm, no Visual C++ runtime |
| To build from source | HIP SDK 7.2 + Visual Studio 2022 Build Tools (MSVC) + CMake + Ninja — see [building.md](docs/building.md) |
| Memory / disk | the server pins about 1/4 of host RAM (8–32 GiB; 16 GiB on a 64 GB PC) for the RAM cache tier and may use up to 64 GiB of SSD for the SSD tier by default; both are adjustable or can be turned off (`--kv-ram-mb`, `--kv-ssd-gb`) |

Our test machine connects the R9700 as a USB4 / Thunderbolt eGPU; that works. Numbers on a card in a direct PCIe slot may differ slightly (mostly where data crosses the host link, such as model loading and the RAM / SSD tiers).

Not supported yet: other AMD GPUs. The RX 9070 series has the same chip (gfx1201) but 16 GB, most likely too little for these 27B / 35B models (untested). Radeon 8060S (Ryzen AI Max+ 395, gfx1151) support is in development on the [`gfx1151` branch](https://github.com/tsaipifong/whirl-llm/tree/gfx1151).

## Quick start

1. Download `whirl-0.1.3-windows-x64.zip` from [Releases](https://github.com/tsaipifong/whirl-llm/releases) and a [supported model](#supported-models).
2. In PowerShell:

```powershell
Unblock-File .\whirl-0.1.3-windows-x64.zip
Expand-Archive .\whirl-0.1.3-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.1.3-windows-x64
.\whirl.exe devices                                    # should list AMD Radeon AI PRO R9700 (gfx1201)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`), so any OpenAI-compatible client works. Stop it with Ctrl+C; cached sessions are written to the SSD tier first.

For long documents, give a slot a longer context, for example `--ctx-per-slot 262144` (256k tokens per request); see [recipes.md](docs/recipes.md#long-context).

The first run with a model tunes the GPU kernels once — about 1.5 minutes for a 27B Q4_K_M file, a few seconds for MXFP4 files — and caches the result in `%LOCALAPPDATA%\whirl`. Full walkthrough: [quick start](docs/quickstart.md). Every option: [usage reference](docs/usage.md).

## Supported models

WHIRL reads `general.architecture` from the GGUF and accepts only `qwen35` and `qwen35moe` (the Gated DeltaNet + gated attention hybrid, with an optional MTP head) and fine-tunes that keep that architecture. Other architectures (Llama, Gemma, Mistral, DeepSeek, `qwen3`, `qwen2`, …) are refused at load time with a clear message, and so are tensor types WHIRL has no kernels for (for example NVFP4); there is no slow generic fallback.

> **Recommended: our own MXFP4 quantizations give the best performance.**
> - Dense 27B: [**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF), variant A (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`). Swift-1.5 also thinks much less than the base model (44% fewer thinking tokens on our mixed Chinese/English coding prompts).
> - MoE 35B (~3B active): **Ornith-1.5-35B-A3B MXFP4**: [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF).

Tested GGUF files (loaded, greedy output checked, MTP output identical to plain greedy, benchmarked on the R9700):

| Model file (Hugging Face) | Type | Arch | Quantization | Mode |
|---|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf` (recommended), `…-B-outQ8_0.gguf`, `…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4 (+ Q8_0 MTP / embeddings) | speed mode; quantized by us for WHIRL |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M | precision mode |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 | speed mode |
| [`Ornith-1.5-35B-A3B-MXFP4.gguf`](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) (quantized by us) | MoE 35B, ~3B active | qwen35moe | MXFP4 experts and dense (+ Q8_0 MTP / embeddings, Q6_K head) | speed mode (fp8 expert prefill) |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B, ~3B active | qwen35moe | Q4_K_M | precision mode |

Image input works with a Qwen3-VL-style mmproj (F16 / BF16) via `--mmproj`; it was tested with the 27B files above. Other `qwen35` / `qwen35moe` GGUFs should load but are untested. Choosing and making GGUFs: [quantization.md](docs/guide/en/quantization.md).

## Performance

WHIRL 0.1.3 vs llama.cpp b11214 (ROCm, the fastest flags we found for each row) on the R9700, greedy decoding, the same GGUF files and the same prompts in both engines. Decode numbers lead with WHIRL's default mode, **MTP + n-gram**; rows marked **no MTP** compare plain decoding without speculation. The KV cache format is shown per row.

Our MXFP4 files (Swift-1.5 27B and Ornith-1.5-35B-A3B) are the main comparison; Qwen3.8-27B UD-Q4_K_M is shown for reference only. From 96k tokens on, each cell gives the KV format each engine actually used, as `KV WHIRL / llama.cpp`.

![Decode after a 16k and a 256k-token context](docs/images/bench_long_decode.png)

![Server aggregate throughput with 4 users at once](docs/images/bench_server_concurrency.png)

![Time to first token with a cached prefix](docs/images/bench_ttft_cache.png)

| R9700, greedy — WHIRL vs llama.cpp (ratio) | Swift-1.5 27B MXFP4-A (dense) | Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) | Qwen3.8-27B UD-Q4_K_M (dense, reference) |
|---|---|---|---|
| Decode tok/s, 7 coding prompts (800 tokens) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 112.4 vs 60.8 (1.85×) | 257.6 vs 118.8 (2.17×) | 97.3 vs 56.0 (1.74×) |
| Decode tok/s after a 16k-token context — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 71.4 vs 60.9⁴ (1.17×) | 316.4 vs 106.9⁴ (2.96×) | 80.7 vs 60.7⁴ (1.33×) |
| Decode tok/s, same 7 prompts — WHIRL **no MTP** vs llama.cpp **no MTP** | 38.1 vs 33.4 (1.14×) | 177.4 vs 118.8 (1.49×) | 34.8 vs 30.9 (1.13×) |
| Server, 4 concurrent users, aggregate tok/s (incl. prefill) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 182.1 vs 64.0⁴ (2.84×) | 381.1 vs 176.8⁴ (2.16×) | 122.4 vs 57.1⁴ (2.14×) |
| Prefill tok/s, 8k-token prompt (CLI bench tools) | 3,469 vs 1,338 (2.59×) | 11,258 vs 4,637 (2.43×) | 1,731 vs 1,223 (1.42×) |
| Prefill tok/s, 32k-token prompt | 2,907 vs 1,174 (2.48×) | 8,633 vs 3,778 (2.29×) | 1,570 vs 1,086 (1.45×) |
| Prefill tok/s, 96k-token prompt | 2,021.1 vs 886.7 (2.28×); KV f16 / f16 | 5,183.5 vs 2,602.6 (1.99×); KV f16 / f16 | 1,266.4 vs 837.3 (1.51×); KV f16 / f16 |
| Prefill tok/s, 128k-token prompt | 1,757 vs 791.3 (2.22×); KV f16 / f16 | 4,311 vs 2,239 (1.93×); KV f16 / f16 | 1,158 vs 752.0 (1.54×); KV f16 / f16 |
| **Prefill tok/s, 256k-token prompt** | 969.8 vs 556.1 (1.74×); KV q8h / f16 | 2,135.2 vs 1,442.2 (1.48×); KV q8h / f16 | 742.2 vs 516.4 (1.44×); KV q8h / q8_0 |
| **Decode tok/s after a 256k-token prompt** — WHIRL MTP+n-gram vs llama.cpp plain decoding³ | 35.1 vs 16.8 (2.09×); KV q8h / f16 | 115.6 vs 53.7 (2.15×); KV q8h / f16 | 33.5 vs 10.5 (3.19×); KV q8h / q8_0 |
| Time to first token, new conversation reusing a 26k-token system prompt (s, lower is better) | 0.108 vs 0.384⁴ (3.6×) | 0.098 vs 0.182⁴ (1.9×) | 0.121 vs 0.369⁴ (3.0×) |
| Time to first token, 26k-token session after a server restart (restored from the SSD tier) | 0.613 s vs N/A² | 0.318 s vs N/A² | 0.672 s vs N/A² |

¹ llama.cpp's best of plain / MTP / MTP + n-gram for that row.

² llama.cpp has no automatic persistent KV cache.

³ llama-bench `tg64@d262144`: 64 tokens generated after a 262,144-token context, plain decoding without speculative decoding. llama.cpp ran Swift-1.5 MXFP4 at 256k with f16 KV (it fits, 31.3 GB); Ornith-1.5 at 96k and 256k also used f16 KV; Qwen3.8 Q4_K_M at 256k needed q8_0 KV, because f16 spilled into shared memory.

⁴ llama.cpp re-measured for v0.1.3 with the same prompts as WHIRL; the mode used (the fastest measured): decode after 16k: Swift-1.5 MTP + n-gram, Ornith-1.5 plain, Qwen3.8 MTP + n-gram; 4 concurrent users: Swift-1.5 plain, Ornith-1.5 plain, Qwen3.8 plain; 26k system prompt TTFT: plain only (a 1-token reply; speculative decoding does not apply).

**Speculative decoding: best, typical and worst case** — Swift-1.5 27B MXFP4-A, WHIRL's default MTP + n-gram against decoding without speculation; the output is the same bit for bit:

| | Best case: file editing (repeated content) | Typical: coding-agent session | Long-document QA (heavy quoting, after 128k) | Worst case: new writing after 128k (nothing to copy) |
|---|---|---|---|---|
| Decode tok/s, MTP + n-gram | 174.4 | 92.9 | 87.6 | 39.4 |
| Decode tok/s, no speculation | 39.1 | 32.8 | 25.3 | 25.2 |
| Speedup | 4.47× | 2.83× | 3.47× | 1.56× |
| Tokens per cycle | 6.19 | — | 6.45 | 1.76 |

**Long context with int8 KV (q8h)** — WHIRL server, one user, `--ctx-per-slot 262144`, q8h KV (int8 keys and values; queries and keys Hadamard-rotated first), the setting used for 256k tokens on the dense models (set with `WHIRL_KV=q8h` for the MoE model):

| WHIRL server, q8h KV, one user | Swift-1.5 27B MXFP4-A | Ornith-1.5-35B-A3B MXFP4 | Qwen3.8-27B UD-Q4_K_M (reference) |
|---|---|---|---|
| Prefill tok/s, 64k-token prompt | 2,162.0 | 5,671.3 | 1,289.9 |
| Prefill tok/s, 128k-token prompt | 1,533.0 | 3,673.6 | 1,032.2 |
| Prefill tok/s, 192k-token prompt | 1,212.2 | 2,770.7 | 876.6 |
| Prefill tok/s, 256k-token prompt | 969.8 | 2,135.2 | 742.2 |
| Decode tok/s (MTP+n-gram) after 128k | 62.5 | 214.0 | 58.3 |
| Decode tok/s (MTP+n-gram) after 256k | 35.1 | 115.6 | 33.5 |
| Long-document Q&A spot check (answer names the function asked about; not a formal needle test) | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ | 128k ✓ / 256k ✓ |

Where WHIRL does **not** lead by much: plain decoding (no MTP) of the dense models is memory-bandwidth bound in both engines (1.13–1.14× llama.cpp); decode after a 16k-token context leads by only 1.17× (Swift-1.5) and 1.33× (Qwen3.8) on the dense models; prefill of Qwen3.8 Q4_K_M leads by less (1.51× at 96k and 1.44× at 256k tokens, against 2.28× and 1.74× for Swift-1.5 MXFP4); Ornith-1.5 prefill at 256k leads by 1.48×; and in the worst case for speculative decoding (new writing after 128k tokens, nothing to copy) it gains only 1.56×. Our R9700 is a USB4 eGPU; prefill and decode stay in VRAM, but restores and model loading cross the host link.

Full methodology and all numbers: [benchmarks.md](docs/benchmarks.md).

## What's new since v0.1.0

Both versions measured on the same machine, with the same GGUF file and the same prompts; the greedy output of the two versions is bit-identical.

| Swift-1.5 27B MXFP4-A, R9700 | v0.1.0 | v0.1.3 | Change |
|---|---|---|---|
| Prefill tok/s, 32k-token prompt (f16 KV) | 2,605.3 | 2,907.4 | +11.6% |
| Prefill tok/s, 128k-token prompt (f16 KV) | 1,407.3 | 1,756.7 | +24.8% |
| Prefill tok/s, 128k-token prompt (q8h KV, server) | 1,352.7 | 1,533.0 | +13.3% |
| Prefill tok/s, 256k-token prompt (q8h KV, server) | 832.5 | 969.8 | +16.5% |
| Decode tok/s, Q&A over a 128k-token document (answer quotes the document a lot) | 63.7 | 87.6 | +37.5% |
| Decode tok/s, coding prompts | 110.7 | 120.7 | +9.1% |
| KV pool, default 4 slots (tokens) | 210,688 | 226,304 | +7.4% |
| 4 users, short prompts sent together: whole batch done (s, lower is better) | 9.57 | 8.62 | −9.9% |
| Decode tok/s, long-context coding-agent request (agent session 3, step 143, 127.9k tokens) | 53.7 | 75.2 | +40.0% |
| Decode tok/s, agent session 3 up to that request (mean of 144 requests) | 64.9 | 81.7 | +25.9% |

v0.1.2 for reference (same machine, same output): 53.5 tok/s for the 127.9k request, 65.0 tok/s for the session; the gain is v0.1.3's.

## Windows security prompts

`whirl.exe` and `whirl-server.exe` are not code-signed. Windows SmartScreen may show "Windows protected your PC" for a freshly downloaded copy: click **More info → Run anyway**, or run `Unblock-File` on the zip before extracting it (only for a zip from the official releases page, after checking its SHA-256). With **Smart App Control** set to *On*, Windows may block the programs without a "Run anyway" option. Details: [windows_security.md](docs/windows_security.md).

## Documentation

| | English | 繁體中文 |
|---|---|---|
| Quick start | [quickstart.md](docs/quickstart.md) | [quickstart_zh-TW.md](docs/quickstart_zh-TW.md) |
| Recipes: connect a client, long agent sessions, long context, several users, images, reading the log, troubleshooting | [recipes.md](docs/recipes.md) | [recipes_zh-TW.md](docs/recipes_zh-TW.md) |
| Usage reference (all commands, options, environment variables, exit codes) | [usage.md](docs/usage.md) | [usage.md](docs/guide/zh-TW/usage.md) |
| Server (API, sampling, tool calls, batching, decode floor, logs) | [server.md](docs/guide/en/server.md) | [server.md](docs/guide/zh-TW/server.md) |
| Benchmarks vs llama.cpp | [benchmarks.md](docs/benchmarks.md) | [benchmarks.md](docs/guide/zh-TW/benchmarks.md) |
| Windows security prompts | [windows_security.md](docs/windows_security.md) | [windows_security_zh-TW.md](docs/windows_security_zh-TW.md) |
| Building from source | [building.md](docs/building.md) | [building_zh-TW.md](docs/building_zh-TW.md) |
| Guide index | [index.md](docs/guide/en/index.md) | [index.md](docs/guide/zh-TW/index.md) |
| Architecture | [architecture.md](docs/guide/en/architecture.md) | [architecture.md](docs/guide/zh-TW/architecture.md) |
| Kernels | [kernels.md](docs/guide/en/kernels.md) | [kernels.md](docs/guide/zh-TW/kernels.md) |
| Speculative decoding (MTP + n-gram) | [speculative-decoding.md](docs/guide/en/speculative-decoding.md) | [speculative-decoding.md](docs/guide/zh-TW/speculative-decoding.md) |
| KV cache, prefix caching, tiers | [kv-and-caching.md](docs/guide/en/kv-and-caching.md) | [kv-and-caching.md](docs/guide/zh-TW/kv-and-caching.md) |
| Quantization and model files | [quantization.md](docs/guide/en/quantization.md) | [quantization.md](docs/guide/zh-TW/quantization.md) |
| Image input | [vision.md](docs/guide/en/vision.md) | [vision.md](docs/guide/zh-TW/vision.md) |
| HIP on Windows | [windows-hip.md](docs/guide/en/windows-hip.md) | [windows-hip.md](docs/guide/zh-TW/windows-hip.md) |
| Benchmarking methodology | [benchmarking.md](docs/guide/en/benchmarking.md) | [benchmarking.md](docs/guide/zh-TW/benchmarking.md) |
| **Pitfalls** (125, each with symptom → root cause → fix → check) | [pitfalls.md](docs/guide/en/pitfalls.md) | [pitfalls.md](docs/guide/zh-TW/pitfalls.md) |

## Development and AI assistance

WHIRL was designed, implemented, optimized and benchmarked with the participation of **Claude Opus 5.5**, an AI model by Anthropic, working under the direction of the project owner ([@tsaipifong](https://github.com/tsaipifong)). Anthropic is not affiliated with and does not endorse this project.

## License and acknowledgements

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the origin of every source file in [PROVENANCE.md](PROVENANCE.md).

- [NInfer](https://github.com/Neroued/ninfer) — the design philosophy WHIRL follows (ideas only).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — used as the comparison baseline in our benchmarks and as the reference for GGUF and quantization-format compatibility; the few adapted pieces (tokenizer pre-split / BPE loop, lookup tables, image preprocessing) are MIT-licensed and listed in THIRD_PARTY_NOTICES.md.
- [gufo](https://github.com/gufo-org/gufo) and [r9700-stack](https://github.com/bkvargyas/r9700-stack) — measurement methodology and optimization ideas.
- Model authors: the Qwen team (Qwen3.8), [unsloth](https://huggingface.co/unsloth), [FreedomAISVR](https://huggingface.co/FreedomAISVR), [ornith-ai](https://huggingface.co/ornith-ai) (Ornith-1.5), and [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (Swift-1.5, the fine-tune our MXFP4 files are quantized from).
