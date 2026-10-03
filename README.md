**English** | [繁體中文](docs/README_zh-TW.md)

# WHIRL

**Windows HIP Inference for RDNA LLMs**

WHIRL is a native Windows LLM inference engine for the AMD Radeon AI PRO R9700, written in pure
C++ and HIP: no WSL, no Linux VM, no llama.cpp runtime underneath. It stands on two pillars:

- **Fast decode.** Hand-tuned RDNA 4 kernels (int8 GEMV at the measured memory-bandwidth limit,
  WMMA prefill, MXFP4 × fp8 matrix paths) plus **MTP + n-gram speculative decoding** whose output is
  bit-identical to plain greedy decoding.
- **A server built for long, multi-turn work.** An OpenAI-compatible server with continuous batching
  and **prefix caching with a multi-tier KV cache (VRAM → pinned RAM → SSD)**, so returning
  conversations and shared system prompts skip the prefill.

WHIRL specializes instead of being generic: it adds model architectures one at a time and tunes each
one for each supported GPU. The architectures supported today are **`qwen35` (dense)** and
**`qwen35moe` (mixture of experts)**; more will be added one by one. WHIRL follows the design
philosophy of [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0) — build for selected models
and selected GPUs, as close to the hardware limit as possible;
[zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4.

## Requirements

| | |
|---|---|
| GPU | **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201) — required; single GPU only. A version for the Ryzen AI Max+ 395 (Radeon 8060S) is planned |
| OS | **Windows 11**, 64-bit |
| Driver | **AMD Software: Adrenalin Edition 26.8.1** (driver 32.0.31041.1004) or newer |
| To run | only the driver (it installs `amdhip64_7.dll` and `amd_comgr_3.dll`). No HIP SDK, no ROCm, no Visual C++ runtime |
| To build from source | HIP SDK 7.2 + Visual Studio 2022 Build Tools (MSVC) + CMake + Ninja — see [building.md](docs/building.md) |
| Memory / disk | the server pins ~8–9 GiB of host RAM for the RAM cache tier and may use up to 64 GiB of SSD for the SSD tier by default; both are adjustable or can be turned off (`--kv-ram-mb`, `--kv-ssd-gb`) |

Our test machine connects the R9700 as a USB4 / Thunderbolt eGPU; that works. Numbers on a card in a
direct PCIe slot may differ slightly (mostly where data crosses the host link, such as model loading
and the RAM / SSD tiers).

## Quick start

1. Download `whirl-0.1.0-windows-x64.zip` from [Releases](https://github.com/tsaipifong/whirl-llm/releases)
   and a [supported model](#models).
2. In PowerShell:

```powershell
Unblock-File .\whirl-0.1.0-windows-x64.zip
Expand-Archive .\whirl-0.1.0-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.1.0-windows-x64
.\whirl.exe devices                                    # should list AMD Radeon AI PRO R9700 (gfx1201)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1` (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/health`), so any OpenAI-compatible client works. Stop it with
Ctrl+C; cached sessions are written to the SSD tier first.

The first run with a model tunes the GPU kernels once — about 1.5 minutes for a 27B Q4_K_M file, a
few seconds for MXFP4 files — and caches the result in `%LOCALAPPDATA%\whirl`.
Full walkthrough: [quick start](docs/quickstart.md). Every option: [usage reference](docs/usage.md).

## <a id="models"></a>Supported models

WHIRL reads `general.architecture` from the GGUF and accepts only `qwen35` and `qwen35moe` (the
Gated DeltaNet + gated attention hybrid, with an optional MTP head) and fine-tunes that keep that
architecture. Other architectures (Llama, Gemma, Mistral, DeepSeek, `qwen3`, `qwen2`, …) are refused
at load time with a clear message, and so are tensor types WHIRL has no kernels for (for example
NVFP4); there is no slow generic fallback.

> **Recommended: our own MXFP4 quantizations give the best performance.**
> - Dense 27B: [**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF),
>   variant A (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`). Swift-1.5 also thinks much less than
>   the base model (44% fewer thinking tokens on our mixed Chinese/English coding prompts).
> - MoE 35B (~3B active): **Ornith-1.5-35B-A3B MXFP4** — *coming soon on Hugging Face* (it will be
>   uploaded after this release; placeholder: [huggingface.co/tsaipifong](https://huggingface.co/tsaipifong)).
>   <!-- TODO(release): replace the placeholder with the Ornith-1.5-35B-A3B-MXFP4 repository URL once uploaded. -->

Tested GGUF files (loaded, greedy output checked, MTP output identical to plain greedy, benchmarked
on the R9700):

| Model file (Hugging Face) | Type | Arch | Quantization | Mode |
|---|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf` (recommended), `…-B-outQ8_0.gguf`, `…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4 (+ Q8_0 MTP / embeddings) | speed mode; quantized by us for WHIRL |
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_M.gguf` | dense 27B | qwen35 | UD-Q4_K_M | precision mode (outputs bit-exact across optimizations) |
| [FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF](https://huggingface.co/FreedomAISVR/Qwen3.8-27B-MXFP4-GGUF) `qwen3.8-27b-mxfp4.gguf` | dense 27B | qwen35 | MXFP4 | speed mode |
| `Ornith-1.5-35B-A3B-MXFP4.gguf` — coming soon on Hugging Face (quantized by us) | MoE 35B, ~3B active | qwen35moe | MXFP4 experts and dense (+ Q8_0 MTP / embeddings, Q6_K head) | speed mode (fp8 expert prefill) |
| [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) `Ornith-1.5-35B-Q4_K_M.gguf` | MoE 35B, ~3B active | qwen35moe | Q4_K_M | precision mode |

Image input works with a Qwen3-VL-style mmproj (F16 / BF16) via `--mmproj`; it was tested with the
27B files above. Other `qwen35` / `qwen35moe` GGUFs should load but are untested. Choosing and making
GGUFs: [quantization.md](docs/guide/en/quantization.md).

## Performance

WHIRL 0.1.0 vs llama.cpp b11214 (ROCm, the fastest flags we found for each row) on the R9700,
greedy decoding, the same prompts in both engines. Decode numbers lead with WHIRL's default mode,
**MTP + n-gram**; rows marked **no MTP** compare plain decoding without speculation.

<p>
<img src="docs/images/bench_decode_zh.png" width="49%" alt="Decode speed, Chinese coding prompts">
<img src="docs/images/bench_server_concurrency.png" width="49%" alt="Server aggregate throughput with 1, 2 and 4 concurrent users">
<img src="docs/images/bench_prefill.png" width="49%" alt="Prefill speed versus prompt length">
<img src="docs/images/bench_ttft_cache.png" width="49%" alt="Time to first token with cached prefixes and restored sessions">
</p>

| R9700, greedy — WHIRL vs llama.cpp (ratio) | Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active) | Swift-1.5 27B MXFP4-A (dense) | Qwen3.8-27B UD-Q4_K_M (dense) |
|---|---|---|---|
| Decode tok/s, 7 Chinese coding prompts (800 tokens) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 244.5 vs 118.8 (**2.06×**) | 107.5 vs 60.8 (**1.77×**) | 98.5 vs 56.0 (**1.76×**) |
| Decode tok/s after a 16k-token context — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 301.5 vs 107.5 (**2.80×**) | 69.5 vs 63.5 (1.09×) | 74.3 vs 59.8 (1.24×) |
| Decode tok/s, same 7 prompts — WHIRL **no MTP** vs llama.cpp **no MTP** | 166.7 vs 118.8 (1.40×) | 37.5 vs 33.4 (1.13×) | 34.4 vs 30.9 (1.11×) |
| Server, 4 concurrent users, aggregate tok/s (incl. prefill) — WHIRL MTP+n-gram vs llama.cpp fastest¹ | 396.7 vs 191.9 (**2.07×**) | 198.6 vs 65.8 (**3.02×**) | 134.1 vs 58.5 (**2.29×**) |
| Prefill tok/s, 8k-token prompt (CLI bench tools) | 10,858 vs 4,637 (**2.34×**) | 3,278 vs 1,338 (**2.45×**) | 1,689 vs 1,223 (1.38×) |
| Prefill tok/s, 32k-token prompt | 7,978 vs 3,778 (**2.11×**) | 2,595 vs 1,174 (**2.21×**) | 1,479 vs 1,086 (1.36×) |
| Time to first token, new conversation reusing a 26k-token system prompt (s, lower is better) | 0.073 vs 0.290 (**4.0×**) | 0.122 vs 0.538 (**4.4×**) | 0.136 vs 0.546 (**4.0×**) |
| Time to first token, 26k-token session after a server restart (restored from the SSD tier) | 0.298 s vs N/A² | 0.729 s vs N/A² | 0.739 s vs N/A² |

¹ llama.cpp's best of plain / MTP / MTP + n-gram for that row (on the MoE model its plain decoding
is fastest). ² llama.cpp has no automatic persistent KV cache.

Where WHIRL does **not** lead by much: plain decoding (no MTP) of the dense models is memory-bandwidth
bound in both engines (1.1–1.2×); Swift MXFP4-A after a 16k context is only 1.09× llama.cpp's best;
prefill of an 88-token prompt on Q4_K_M is at parity (1.03×); and at the same context WHIRL's CLI uses
3–5 GiB more VRAM than llama-bench on the dense models. Our R9700 is a USB4 eGPU; prefill and decode stay
in VRAM, but restores and model loading cross the host link.

Full methodology and all numbers: [benchmarks.md](docs/benchmarks.md).

## Windows security prompts

`whirl.exe` and `whirl-server.exe` are not code-signed. Windows SmartScreen may show "Windows
protected your PC" for a freshly downloaded copy: click **More info → Run anyway**, or run
`Unblock-File` on the zip before extracting it (only for a zip from the official releases page,
after checking its SHA-256). With **Smart App Control** set to *On*, Windows may block the programs
without a "Run anyway" option. Details: [windows_security.md](docs/windows_security.md).

## Documentation

| | English | 繁體中文 |
|---|---|---|
| Quick start | [quickstart.md](docs/quickstart.md) | [quickstart_zh-TW.md](docs/quickstart_zh-TW.md) |
| Usage reference (all commands, options, environment variables, exit codes) | [usage.md](docs/usage.md) | [usage.md](docs/guide/zh-TW/usage.md) |
| Server (API, sampling, tool calls, batching, logs) | [server.md](docs/guide/en/server.md) | [server.md](docs/guide/zh-TW/server.md) |
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

WHIRL was designed, implemented, optimized and benchmarked with the participation of
**Claude Opus 5.5**, an AI model by Anthropic, working under the direction of the project owner
([@tsaipifong](https://github.com/tsaipifong)). Anthropic is not affiliated with and does not endorse
this project.

## License and acknowledgements

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the origin of every source file in
[PROVENANCE.md](PROVENANCE.md).

- [NInfer](https://github.com/Neroued/ninfer) — the design philosophy WHIRL follows (ideas only).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — used as the comparison baseline in our
  benchmarks and as the reference for GGUF and quantization-format compatibility; the few adapted
  pieces (tokenizer pre-split / BPE loop, lookup tables, image preprocessing) are MIT-licensed and
  listed in THIRD_PARTY_NOTICES.md.
- [gufo](https://github.com/gufo-org/gufo) and [r9700-stack](https://github.com/bkvargyas/r9700-stack) —
  measurement methodology and optimization ideas.
- Model authors: the Qwen team (Qwen3.8), [unsloth](https://huggingface.co/unsloth),
  [FreedomAISVR](https://huggingface.co/FreedomAISVR), [ornith-ai](https://huggingface.co/ornith-ai)
  (Ornith-1.5), and [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (Swift-1.5, the
  fine-tune our MXFP4 files are quantized from).
