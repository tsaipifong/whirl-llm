**English** | [繁體中文](../zh-TW/index.md)

# WHIRL guide

WHIRL (*Windows HIP Inference for RDNA LLMs*) is an Apache-2.0 LLM inference engine in C++ and HIP
for AMD RDNA GPUs, running natively on Windows. It specializes — it adds model architectures one at a
time and tunes every kernel for each architecture and each GPU — following the philosophy of
[NInfer](https://github.com/Neroued/ninfer). It contains no code or text from NInfer.

These documents explain how WHIRL works, why each design was chosen, what we measured, which
alternatives lost, and — in the most detail — every pitfall we hit. Many of those pitfalls are
specific to HIP on Windows and RDNA 3.5/4 and are not documented elsewhere; we wrote them so that
other projects can reuse the lessons.

> **Status (WHIRL 0.1.0, 2026-10-03).** Everything described here is implemented in this repository's
> C++ engine for the R9700 (gfx1201): kernels, loader, forward pass, speculative decoding, the server
> with tiered caching, and image input. A gfx1151 (Radeon 8060S) version is in preview (correct, not yet tuned). Some measurements
> in these documents were taken with the research build that preceded the C++ engine; the C++ engine
> produces identical outputs at the same speed (within ±1%). Release benchmarks:
> [benchmarks.md](../../benchmarks.md).

## Documents

| Document | What it covers |
|---|---|
| [usage.md](../../usage.md) | Reference of every command, option, environment variable, server endpoint and exit code |
| [architecture.md](architecture.md) | The pipeline (GGUF → loader → kernels → forward → server), per-GPU code objects, why specialize, the correctness model |
| [windows-hip.md](windows-hip.md) | HIP on Windows: PAL vs ROCr, timing without a profiler, memory (big allocations, mmap, VMM, WDDM demotion, pinned = shared usage), streams, device numbering, eGPU, shell traps |
| [kernels.md](kernels.md) | Every kernel family: int8 GEMV with `v_dot4`/`v_perm`, bit-exact multi-row GEMV, int8 WMMA mid-batch GEMV, grouped launches, prefill GEMMs, MXFP4×fp8 WMMA, flash attention, split-K decode attention, DeltaNet, MoE, register discipline, fma contraction |
| [speculative-decoding.md](speculative-decoding.md) | MTP drafting, the bit-exact verify requirement, DeltaNet snapshots → replay, automatic draft count, 2-bit draft head, n-gram co-drafting, EOS truncation, comparison with llama.cpp |
| [kv-and-caching.md](kv-and-caching.md) | Paged KV pool, KV formats (f16/q8/q8h/q8v) and the automatic policy, the 128k + 64k floor rule, DeltaNet checkpoints, system-prompt checkpoints, forward merging, VRAM → RAM → SSD tiers |
| [server.md](server.md) | OpenAI-compatible API, sampling, tool calls, `reasoning_content`, continuous batching, logging, limitations |
| [benchmarking.md](benchmarking.md) | Methodology (interleaved A/B, min–max, unified protocol vs llama.cpp, cache and thermal traps, VRAM-only rule), correctness gates, **current numbers** |
| [benchmarks.md](../../benchmarks.md) | Release benchmarks of WHIRL 0.1.0 vs llama.cpp (prefill, decode, concurrency, cache TTFT, VRAM, vision) |
| [quantization.md](quantization.md) | Precision mode (Q4_K_M) vs speed mode (MXFP4), MXFP4 handling, perplexity, the Swift-1.5 MXFP4 recipe, a checklist for GGUFs that run well, mmproj |
| [vision.md](vision.md) | On-demand image input: design, numerics, bugs found |
| [pitfalls.md](pitfalls.md) | **125 pitfalls** with conditions → symptom → root cause → fix → gate, plus the generalizable lessons |

## Reading paths

- **"I want to run it."** [README](../../../README.md) → [quick start](../../quickstart.md) →
  [quantization.md](quantization.md#rules) (which GGUF) → [usage.md](../../usage.md) → [server.md](server.md).
- **"I write HIP on Windows."** [windows-hip.md](windows-hip.md) → [pitfalls.md](pitfalls.md#hip).
- **"I write RDNA kernels."** [kernels.md](kernels.md) → [pitfalls.md](pitfalls.md#kern).
- **"I work on llama.cpp."** [pitfalls.md](pitfalls.md#tok-1) (tokenizer), [pitfalls.md](pitfalls.md#hip-2)
  and [#hip-3](pitfalls.md#hip-3) (ROCm on Windows), [speculative-decoding.md](speculative-decoding.md#vs-llamacpp),
  [quantization.md](quantization.md#swift).
- **"I serve hybrid models."** [kv-and-caching.md](kv-and-caching.md) →
  [speculative-decoding.md](speculative-decoding.md#replay).
- **"I benchmark AMD GPUs."** [benchmarking.md](benchmarking.md).

## Conventions

- **Hardware.** R9700 = AMD Radeon AI PRO R9700 (RDNA 4, gfx1201, 32 GB), the primary target. On the
  development machine it is a **USB4 eGPU** (host link ~3.8 GB/s); effects of that link are labeled
  *environment limitations*. 8060S = AMD Radeon 8060S (RDNA 3.5, gfx1151) in a thermally limited laptop.
  OS: Windows 11 build 26200; HIP SDK 7.2.
- **Numbers** are measured, with units and conditions; ranges are min–max over interleaved rounds.
  Decode numbers lead with WHIRL's default mode (MTP + n-gram); "no MTP" numbers are labeled — plain
  decode of a 27B 4-bit model is bounded at ~38 tok/s by memory bandwidth on this card.
- **Bit-exact** means identical bits, enforced by a permanent check.
- **Names.** Command-line flags and `WHIRL_*` variables are those of WHIRL 0.1.0; all of them are
  listed in [usage.md](../../usage.md).
- **Unknowns** are marked TODO rather than estimated.

## Credits

Ideas: [NInfer](https://github.com/Neroued/ninfer) (specialized engines; Apache-2.0). Adapted code:
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT: tokenizer pre-split and BPE merge loop,
IQ lookup tables). Techniques: [r9700-stack](https://github.com/bkvargyas/r9700-stack) (Apache-2.0: E8M0 →
fp8 exponent folding, fragment-tiled fp8 activations, LDS-only barrier sequence). Measurement methodology
for Strix Halo: [gufo](https://github.com/gufo-org/gufo). Hierarchical-cache scheduling ideas (delay hit, prefetch on hit): the Strata paper
(arXiv 2508.18572). Details in [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) and
[PROVENANCE.md](../../../PROVENANCE.md).
