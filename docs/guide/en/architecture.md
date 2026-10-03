**English** | [繁體中文](../zh-TW/architecture.md)

# Architecture

> **Status (WHIRL 0.1.0).** This document describes the design as implemented in this repository:
> C++20 host code and HIP C++ device code for the R9700 (gfx1201). A gfx1151 (Radeon 8060S) kernel set
> is planned. Some measurements were taken with the research build that preceded the C++ engine; the
> C++ engine reproduces its outputs token for token ([benchmarking.md](benchmarking.md#gates)).

**Who this helps:** anyone deciding whether a model-specific, GPU-specific engine is worth
building, and anyone who wants to know where each piece of WHIRL lives and why it is shaped
that way.

## 1. What WHIRL is

WHIRL (*Windows HIP Inference for RDNA LLMs*) is an LLM inference engine written in C++20 and
HIP C++ for AMD RDNA GPUs, running natively on Windows. It has two defining properties:

1. **It specializes.** WHIRL follows the philosophy of
   [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0): instead of a generic engine that
   runs every model reasonably, build an engine that runs a *selected* set of checkpoints on a
   *selected* set of GPUs as close to the hardware limit as possible.
   [zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4. WHIRL shares only
   the idea; it contains no code or text from either project.
2. **It is pure native Windows.** No WSL, no Linux VM, no llama.cpp runtime. The HIP SDK on
   Windows runs on AMD's PAL driver stack, which behaves differently from the Linux ROCr/KFD
   stack in ways that matter for performance work. Those differences are documented in
   [windows-hip.md](windows-hip.md).

WHIRL is **not** "a Qwen engine". It adds model architectures one at a time and tunes every
kernel for each architecture and each GPU. The first family done this way is the
Qwen3.5-generation hybrid (`qwen35` dense, `qwen35moe` mixture-of-experts). Every other
`general.architecture` value is rejected at load time with `UnsupportedArch`; there is no
generic fallback path that would silently run slower or produce different numerics.

### Targets

| GPU | ISA | Role | Measured limits (our probes, not spec sheets) |
|---|---|---|---|
| AMD Radeon AI PRO R9700 (RDNA 4, Navi 48) | gfx1201 | primary target | 32 GB GDDR6, 256-bit, 640 GB/s spec; GEMV streaming 604–626 GB/s; WMMA f16→f32 180–184 TFLOPS; WMMA iu8 343–374 TOPS |
| AMD Radeon 8060S (Ryzen AI Max, RDNA 3.5) | gfx1151 | secondary target | unified memory; streaming read 238 GB/s; WMMA f16 43 TFLOPS; WMMA int8 45 TOPS (same rate as f16) |

The development R9700 is attached as a **USB4 eGPU**. Its host link measures
3.78–3.81 GB/s (D2H) and 3.84–3.86 GB/s (H2D). Everything that moves data between host and
GPU (loading, the RAM/SSD KV tiers, vision weight streaming) is bounded by that link on this
machine. Most users have a direct PCIe slot and will see much higher host bandwidth; we label
eGPU-caused effects as *environment limitations* throughout the documentation and did not
write workarounds for them.

### Model family 1: `qwen35` / `qwen35moe`

| | Qwen3.8-27B (`qwen35`) | Ornith-1.5-35B-A3B (`qwen35moe`) |
|---|---|---|
| Layers | 64 + 1 MTP layer (`blk.64`, nextn) | 40 + 1 MTP layer |
| Layer pattern | Gated DeltaNet ×3, gated attention ×1, repeated (48 DeltaNet + 16 attention) | same pattern (30 DeltaNet + 10 attention) |
| Embedding | 5,120 | 2,048 |
| Attention | 24 heads / 4 KV heads, head_dim 256, NEOX RoPE on the first 64 dims, base 1e7, sigmoid output gate | 16 heads / 2 KV heads |
| Gated DeltaNet | 16 k-heads, 48 v-heads, head dim 128, conv kernel 4, L2-normalized q/k; v-head h uses k-head h % 16 | same structure, smaller |
| FFN | dense SwiGLU, n_ff 17,408 | 256 experts, top-8 (512 wide each) + 1 shared expert (512) with a sigmoid gate |
| Vocabulary | 248,320 | same tokenizer (identical token counts on our 110-file parity corpus) |

The hybrid is what makes this family interesting and hard: every decode step must advance
both an attention KV cache and a recurrent DeltaNet state. A recurrent state cannot be "rolled
back" by truncating a cache, which shapes speculative decoding
([speculative-decoding.md](speculative-decoding.md)) and prefix caching
([kv-and-caching.md](kv-and-caching.md)).

## 2. The pipeline

```
 GGUF file ──► GGUF reader ──► loader ──► device weights (+ per-load requantized copies)
                (mmap, v2/v3)   (type check,  │
                                reorder,      ▼
                                requant)   forward pass ◄── kernels (one code object per GPU arch)
                                             │  prefill: WMMA GEMM, flash attention, chunked DeltaNet
                                             │  decode : int8 GEMV, split-K attention, recurrent step
                                             ▼
 tokenizer + chat template ──► scheduler (CLI or server) ──► sampler ──► text / tool calls
                                  │
                                  └─ KV pool (paged), DeltaNet checkpoints, RAM/SSD tiers
```

### 2.1 GGUF reader — `src/gguf/` (in the repository)

Reads GGUF v2 and v3 through a memory mapping: every metadata value type, nested arrays,
alignment, and the full ggml type table (including `mxfp4`, `nvfp4`, the `iq*` and `tq*`
families). Verified against gguf-py 0.19.0 on 13 files (every key, value, tensor name, type,
shape, offset and byte size identical). The 15.3 GiB Qwen3.8 file (866 tensors) parses in
about 7 ms because nothing is copied.

### 2.2 Tokenizer and chat templates — `src/tokenizer/`, `src/chat/` (in the repository)

- `qwen35` pre-tokenizer split + byte-level BPE, loaded directly from the GGUF vocabulary
  (247,587 merges in 0.10–0.15 s). The pre-split and the special-token/BPE merge loop are
  adapted from llama.cpp (MIT) and credited in [THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md).
- Unicode tables are generated by our own script and emitted at the **Unicode 15.1** level to
  match llama.cpp (see the pitfall *Unicode 15.1 vs 16.0* in [pitfalls.md](pitfalls.md)).
- Parity: 440 file/model/mode combinations, about 1.5 million tokens, identical to
  `llama-tokenize`; 28 rendered chat prompts identical.
- The chat template renderer reproduces the GGUF Jinja templates for the two supported
  template variants (tools, thinking on/off, `preserve_thinking`, `reasoning_effort`,
  `tool_choice`), checked case by case against llama.cpp's Jinja engine.

The tokenizer is not a detail: a whitespace-splitting bug in the prototype tokenized every
indented code prompt into token sequences the model had never seen. It is the single most
instructive bug in this project; see [pitfalls.md](pitfalls.md#tok-1).

### 2.3 Loader — `src/model/loader.cpp`

- Reads `general.architecture`; accepts only `qwen35` and `qwen35moe`.
- Checks every tensor type against the kernel set for that GPU. An unimplemented type (for
  example NVFP4) stops the load; there is no "dequantize to f16 and hope" fallback.
- Reorders formats whose GGUF layout does not suit the kernels. MXFP4 blocks
  (32 values = one E8M0 exponent + 16 bytes) are regrouped per row into 256-value super-blocks
  `e[8] | qs[8][16]` (136 bytes, the same size and alignment as IQ4_XS), and a per-row
  reference exponent is computed for the fp8 prefill path ([quantization.md](quantization.md)).
- Builds per-load derived copies on the GPU:
  - a **2-bit draft head** ("D2", 2.5 bpw) requantized from the Q6_K output head, used only
    to pick MTP drafts;
  - a **Q4_K copy of the MTP block** (dense models) for cheaper drafting.
  Both only affect *which tokens are proposed*; every emitted token is verified by the full
  model, so outputs are unchanged.
- Loads by reading the file and uploading, not by mapping it into the GPU. (llama.cpp's
  default mmap load fails on this driver stack; see [windows-hip.md](windows-hip.md#mmap).)

### <a id="kernels"></a>2.4 Kernels — `kernels/` (one code object per GPU architecture)

Host code is compiled by MSVC. Device code is compiled by the HIP SDK's clang into one code
object per architecture (`gfx1201`, `gfx1151`), converted to a byte array by `tools/bin2c`,
and embedded in the executable. At run time the code object that matches the selected device
is loaded with `hipModuleLoadData`.

Why per-GPU code objects rather than one portable kernel set:

- **The matrix instructions differ.** gfx12 WMMA keeps A/B fragments compactly in registers,
  and its C/D fragment layout equals the B layout, which our DeltaNet scan and flash attention
  exploit. gfx11 (RDNA 3.5) WMMA needs the B fragment duplicated in both half-waves and
  interleaves output rows. A kernel written for one is wrong or slow on the other.
- **The cost balance differs.** On the R9700 int8 WMMA is 2× f16; on the 8060S they run at the
  same rate and gfx11 WMMA competes with VALU work. A design that wins on one GPU can lose on
  the other (W8A8 prefill, for example — see [kernels.md](kernels.md)).
- **Optional kernels.** Some kernels exist only for one architecture (int8 WMMA mid-batch
  GEMV, MXFP4×fp8 GEMM, the `q8v` KV format exist only for gfx1201). The host looks them up
  with `getFunctionOpt`, which returns null for a missing kernel and clears HIP's sticky last
  error, and falls back or refuses at load time.

File split: `kernels/common.hip` (types, wave primitives, block decoders), `gemv_*.hip`,
`gemm_prefill.hip` / `gemm_fp8*.hip` / `gemm_small.hip`, `attn*.hip`, `gdn_*.hip` (DeltaNet),
`moe.hip` / `moe_mxfp4.hip`, `fused_decode.hip`, `misc_*.hip`, `sample_*.hip`, `draft_d2.hip`, all
included by `kernels/whirl_kernels.hip`; the vision encoder kernels are a separate code object in
`kernels/vision/`. gfx1151-specific paths will go under `kernels/gfx1151/`.

### 2.5 Forward pass — `src/model/`

Prefill and decode are different programs:

| | Prefill | Decode |
|---|---|---|
| Bound by | compute | memory bandwidth |
| Matmul | WMMA GEMM (f16, or MXFP4×fp8), autotuned per (type, shape, batch bucket) | int8 GEMV with `v_dot4`; int8 WMMA for 3–16 rows |
| Attention | WMMA flash attention, Sᵀ = K·Qᵀ | split-K flash decoding; WMMA grouped verify |
| DeltaNet | chunked (64 tokens, WY form), f32 or f16-WMMA | one recurrent step per token, fused with gated RMSNorm |
| Host role | enqueue chunks of ≤ 1024–2048 rows | enqueue several steps ahead; argmax kernels write the next token and position on the GPU |

Decode state lives on the GPU. Argmax kernels write the next token id and position straight
into device memory, so the host never reads back per step and can queue several steps; a
whole MTP draft-and-verify cycle needs one host synchronization. See
[kernels.md](kernels.md) for every kernel family.

### 2.6 Scheduler, sampler and server — `src/server/`, `src/tier/`

The OpenAI-compatible server owns: continuous batching over up to 16 slots (default 4), a
shared paged KV pool, DeltaNet checkpoints for prefix reuse, the VRAM → pinned RAM → SSD KV
tiers, speculative decoding (MTP + n-gram) per slot, sampling, tool-call parsing and
`reasoning_content`. See [server.md](server.md) and [kv-and-caching.md](kv-and-caching.md).

## 3. Why specialize — the measured argument

A specialized engine is only worth its cost if the specialization shows up in numbers that a
generic engine cannot reach on the same hardware. Measured on the same R9700, same GGUF,
greedy decoding, both engines through their own OpenAI-compatible servers, one model process
at a time (full protocol in [benchmarking.md](benchmarking.md)):

| Qwen3.8-27B, tok/s | llama.cpp b11214 ROCm, Q4_K_M | WHIRL, Q4_K_M | WHIRL, MXFP4 |
|---|---|---|---|
| Prefill 8k tokens | 1,177 | 1,747–1,751 | 3,548–3,557 |
| Decode, default mode (MTP + n-gram) | N/A | 145.4 | 159.1 |
| Decode, MTP | 56.9 | 110.4 | 120.3 |
| Decode, no MTP | 30.9 | 34.3 | 37.6 |

Decode rows: same unified server protocol for both engines (19-prompt average). The WHIRL
prefill row is the CLI (KV f16); WHIRL's *server* cold prefill at 8k is 1,560 (Q4_K_M) /
3,235 (MXFP4) tok/s, still well ahead. Details and caveats in
[benchmarking.md](benchmarking.md#current-numbers).

Where the gap comes from, in order of size:

1. **Speculative decoding designed around exactness.** WHIRL's multi-row GEMV produces results
   bit-identical to the 1-row GEMV, so verifying 4 drafts costs about 15% more than decoding 1
   token, and MTP output is identical to plain greedy output. That lets an adaptive policy use
   up to 8 MTP drafts (15 with n-gram drafts) where llama.cpp uses a fixed maximum of 3.
2. **Kernels shaped for one file.** The unsloth UD-Q4_K_M file mixes nine formats (Q5_K,
   IQ4_XS, Q4_K, Q6_K, IQ4_NL, Q3_K, IQ3_S, Q8_0, F32). Each has its own decoder and its own
   tuned multi-row kernel; the slowest two (Q3_K, IQ3_S) were fixed after measuring every type
   against a pure-read ceiling.
3. **Formats chosen for the hardware.** MXFP4's power-of-two block scales fold exactly into fp8
   exponents, so RDNA 4's fp8 WMMA runs the prefill GEMM at about 240 TOPS with a single
   epilogue. That is the "speed mode" ([quantization.md](quantization.md)).

**Plain decode has a hard ceiling.** Without speculation, every token must stream all weights
once: 14.33 GB for the Q4_K_M file. WHIRL's 1-token GEMV already runs at 97–100% of the
measured pure-read ceiling (604–626 GB/s), and a full decode step takes 27.74 ms
(Q4_K_M) / 25.15 ms (MXFP4). The no-MTP ceiling for a 27B 4-bit model on a 640 GB/s card is
therefore about **38 tok/s**; kernel work can only recover a few percent. Going faster means
verifying several tokens per weight pass — which is why the default mode is MTP + n-gram and
why decode numbers in these documents lead with it.

## 4. Correctness model

Two modes, chosen per model file:

- **Precision mode (Q4_K_M and other K/IQ quants).** Every optimization must keep outputs
  bit-identical where the previous version was bit-identical. Lossy changes (8-bit KV, for
  example) must pass a KL-divergence check against measured path noise, a paired
  question-answering test (350 items, McNemar exact test) and long-context needle tests.
- **Speed mode (MXFP4).** fp8 activations in prefill are accepted; their effect is measured
  and recorded (KL, paired QA, perplexity) but is not a blocking gate.

In both modes these invariants are enforced by permanent gates on every change:

- MTP greedy == plain greedy (all draft counts 1–10, with and without n-gram drafts).
- Multi-row GEMV == 1-row GEMV, bit for bit, for every type, row count 2–16 and kernel variant.
- Concurrent requests == the same request alone (greedy), including segmented prefill.
- A server started with **no** environment variables and **no** options produces the same
  greedy text as the CLI (added after a bug that only appeared on the default path).
- KV restored from RAM or SSD == KV that never left VRAM, bit for bit.
- A new session reusing a cached system prompt == the same request run cold.

[benchmarking.md](benchmarking.md) lists the gates and how they are run.

## 5. Where to read next

| Topic | Document |
|---|---|
| HIP on Windows: driver stack, memory, timing, process discipline | [windows-hip.md](windows-hip.md) |
| Every kernel family and its numbers | [kernels.md](kernels.md) |
| MTP, n-gram co-drafting, exactness | [speculative-decoding.md](speculative-decoding.md) |
| Paged KV, KV formats, checkpoints, tiered cache | [kv-and-caching.md](kv-and-caching.md) |
| OpenAI server behavior and limits | [server.md](server.md) |
| How we measure; current numbers | [benchmarking.md](benchmarking.md) |
| Precision vs speed mode, MXFP4 quantization recipe | [quantization.md](quantization.md) |
| Image input | [vision.md](vision.md) |
| Every pitfall we hit | [pitfalls.md](pitfalls.md) |
