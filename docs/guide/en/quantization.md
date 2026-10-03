**English** | [繁體中文](../zh-TW/quantization.md)

# Quantization: precision mode, speed mode, and making GGUFs for WHIRL

> **Status.** The kernel support described here is in WHIRL 0.1.0 for gfx1201 (R9700). The GGUF files
> listed were produced and measured by us; the Swift-1.5 MXFP4 files are published on Hugging Face,
> the Ornith-1.5-35B-A3B MXFP4 file will be published after the first release.

**Who this helps:** anyone choosing a quantization for a hybrid Qwen3.x model on RDNA, anyone
quantizing such a model with llama.cpp's `llama-quantize` (several non-obvious traps below), and
anyone who wants to know what MXFP4 actually costs in quality.

## 1. Two modes

WHIRL treats a model file in one of two ways:

| | Precision mode | Speed mode |
|---|---|---|
| Typical file | unsloth UD-Q4_K_M (K-quants + IQ quants) | MXFP4 |
| Prefill activations | f16 | fp8 (e4m3), one scale per token |
| Rule for lossy changes | must pass KL vs path noise + paired QA (McNemar) + needles | accepted; effects measured and recorded |
| Outputs across optimizations | bit-identical where the previous build was | may change when the speed path changes |
| MTP == plain greedy, concurrent == solo | yes | yes |

Speed mode applies automatically when the file contains MXFP4 tensors: fp8 prefill GEMMs, f16-WMMA
DeltaNet chunks and f16 GEMM outputs for element-wise consumers. `WHIRL_FP8=0` returns MXFP4
prefill to f16 activations (then prefill is only 2–4% faster than Q4_K_M — the speed comes from fp8,
not from the format). Decode is exact in both modes (int8 GEMV with the same per-unit arithmetic).

| Qwen3.8-27B on the R9700 | Q4_K_M (precision) | MXFP4 (speed) |
|---|---|---|
| Weights | 15.3 GiB | 13.9 GiB (12.89 GiB MXFP4 + 0.97 GiB Q6_K head) |
| Default server KV pool (q8v, 4 slots) | 199,936 tokens | ~16% larger (228,096 after system checkpoints were added) |
| CLI prefill 2k / 8k / 32k / 128k | 1,722 / 1,749 / 1,532 / 1,052 tok/s | 3,770 / 3,552 / 2,783 / 1,531 tok/s |
| Decode no MTP / MTP / MTP + n-gram (server) | 34.3 / 110.4 / 145.4 | 37.6 / 120.3 / 159.1 |
| Perplexity, zh/en code corpus (llama.cpp) | 3.640 | 3.795 (+4.3%) |
| Paired QA, 350 items (WHIRL, greedy) | 265 (f16 KV run) | 256 (fp8 prefill run) |

The two QA numbers come from different runs (not a paired comparison). Within MXFP4, fp8 vs f16
prefill was paired: 256 vs 253, McNemar p = 0.58.

**Choose precision mode** when outputs must match a greedy reference and every optimization must
pass precision gates. **Choose speed mode** for long prompts and agents that read many files, where
prefill dominates; decode is also 8–10% faster (proportional to the weight bytes).

## 2. Tensor types WHIRL accepts

Decode and prefill kernels exist for: F32 and F16 (small tensors), Q8_0, Q3_K, Q4_K, Q5_K, Q6_K,
IQ3_S, IQ4_NL, IQ4_XS, MXFP4; BF16 for the vision projector. Any other type (for example NVFP4)
stops the load — there is no generic dequantize-to-f16 fallback. MXFP4 is implemented for dense
`qwen35` and for MoE experts (`qwen35moe`, [section 6](#ornith-mxfp4)). On gfx1151 (8060S, preview)
MXFP4 runs on the same int8 decode GEMVs and is dequantized to f16 for prefill (gfx1151 has no fp8
WMMA, so MXFP4 prefill there uses f16 activations; the MoE experts take the generic paths).

### 2.1 What the unsloth UD-Q4_K_M file contains

One file mixes nine formats, which is exactly why WHIRL tunes per file. Decode weights read per
token (14.33 GB total):

| Format | Bytes | 1-token time | Achieved bandwidth | Note |
|---|---|---|---|---|
| Q5_K | 4.834 GB | 8.33 ms | 580 GB/s | |
| IQ4_XS | 4.760 GB | 8.06 ms | 590 GB/s | |
| Q4_K | 3.489 GB | 6.12 ms | 570 GB/s | |
| Q6_K | 0.430 GB | 0.76 ms | 566 GB/s | layers; the output head (1,043 MB) is separate |
| IQ4_NL | 0.330 GB | 0.57 ms | 584 GB/s | |
| Q3_K | 0.268 GB | 0.54 ms | 493 GB/s | later fixed to ~580 GB/s |
| IQ3_S | 0.153 GB | 0.33 ms | 460 GB/s | later fixed to ~580 GB/s |
| Q8_0 | 0.070 GB | 0.69 ms | 101 GB/s | tiny 48×5120 DeltaNet beta/alpha matrices; fused |
| F32 | — | — | — | norms, conv weights, biases |

(Early measurements; token_embd Q4_K 715 MB is read one row per token.)

## <a id="mxfp4"></a>3. MXFP4 in WHIRL

- **Format.** 32 e2m1 values + one E8M0 (power-of-two) exponent per block, 17 bytes.
- **Loading.** Each row's blocks are regrouped into 256-value super-blocks `e[8] | qs[8][16]`
  (136 bytes — the same size and alignment as IQ4_XS, so byte counts do not change), and each row's
  reference exponent (largest block exponent) is stored for the fp8 path. The loader counts blocks
  whose exponent is more than 8 below the row's reference (they lose precision in the fp8 fold) and
  logs them: 56 blocks in the whole Qwen3.8 MXFP4 model (exponent-gap histogram over 7 sampled
  tensors: gap 0: 1.7e7 blocks, 1: 3.3e7, 2: 1.4e6, …, 7: 120, 8: 1).
- **Decode.** The same unit structure and float expression as IQ4_XS, with scale `2^(e−128)` and a
  `v_perm` lookup for the e2m1 table. All GEMV variants (1-token, multi-token dp4, multi-row, int8
  WMMA) exist, so MTP stays bit-exact.
- **Prefill.** MXFP4 × fp8 WMMA with the E8M0 exponent folded into the fp8 weight
  ([kernels.md](kernels.md#mxfp4-gemm)); per-token fp8 activations.

**Precision of fp8 prefill** (last-token KL, MXFP4 f16-prefill vs fp8-prefill, f16 KV): arch1k
5.9e-5, zh-short 2.5e-4, p4k 3.6e-3, p12k 4.0e-5, p24k 3.5e-5, code16k 2.9e-2, code32k 5.6e-2 (top-1
flipped at a near-tie, p 0.265). For scale: f16 prefill with chunk 1024 vs 4096 gives KL 0
(bit-identical); "decode the whole prompt token by token (int8 activations) vs f16 prefill" gives
2.4e-6 / 9.3e-5 / 6.4e-4 (MXFP4) and 1.7e-6 / 5.7e-5 / 4.8e-3 (Q4_K_M); the format difference
Q4_K_M vs MXFP4 (both f16) is 2.6e-4 … 1.4e-1. So fp8 adds about 3–25× the decode-path noise and
5–45% of the format difference. fp8 activations also amplify any other perturbation: the same f16
WMMA DeltaNet change that is 4.5e-8 … 1.5e-4 on Q4_K_M becomes 1.2e-4 … 7.0e-2 under fp8.

`WHIRL_FP8_MASK` (bits: 1 = attention/DeltaNet/MTP projections, 2 = FFN gate/up, 4 = FFN down)
exists for research on which GEMM class contributes the error; that study was not completed.

## <a id="ppl"></a>4. Perplexity

llama.cpp `llama-perplexity` (b11214 ROCm), mixed Chinese/English code corpus (219 KB), 51 chunks
× 2048 tokens:

| File | PPL |
|---|---|
| Qwen3.8-27B UD-Q4_K_M | 3.6403 ± 0.0355 |
| Qwen3.8-27B MXFP4 (FreedomAISVR) | 3.7952 ± 0.0384 (+4.3%; worse on 51/51 chunks; ΔNLL +0.0417 ± 0.0027 nats/token) |
| Swift-1.5 MXFP4 A (Q6_K head) | 3.8271 ± 0.0394 |
| Swift-1.5 MXFP4 B (Q8_0 head) | 3.8289 ± 0.0395 |
| Swift-1.5 MXFP4 C (Q4_K head) | 3.8386 ± 0.0396 |

Swift-1.5 is a different fine-tune, so its distance from Qwen MXFP4 (about +0.8% for A) is not all
quantization. The PPL gap between UD-Q4_K_M and MXFP4 is expected: the reference MXFP4 file is
almost uniformly 4.25 bpw without an imatrix, while UD-Q4_K_M averages more bits (Q5_K 4.5,
IQ4_XS 4.43, Q4_K 3.92, Q6_K 1.69 GiB, …). (The two Qwen PPL runs peaked at 306 MiB shared GPU
memory, which our VRAM rule flags for *speed*; the values are unaffected.)

## <a id="swift"></a>5. How we quantized Swift-1.5-Qwen3.8-27B to MXFP4

Published: [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF),
with an F16 mmproj.

### 5.1 Recipe

1. **Convert to BF16 GGUF, keeping MTP:** `convert_hf_to_gguf.py <BF16 dir> --outtype bf16` →
   866 tensors (54.6 GB), `qwen35.nextn_predict_layers = 1`.
2. **Quantize** with llama.cpp b11214's `llama-quantize`. This build has no dense MXFP4 file type,
   so the ftype is `MXFP4_MOE` combined with a tensor-type file (first matching rule wins) and an
   explicit output type:

   ```
   llama-quantize --imatrix <imatrix.gguf> --tensor-type-file tt_common.txt \
       --output-tensor-type q6_k swift-1.5-27b-bf16.gguf \
       Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf MXFP4_MOE 16
   ```

   `tt_common.txt`:

   ```
   blk\.64\.=q8_0
   ssm_alpha=q8_0
   ssm_beta=q8_0
   token_embd=q8_0
   ```

   with everything else (attn_gate, attn_qkv, attn_q/k/v, attn output, ffn_gate/up/down, ssm_out)
   left to the MXFP4 ftype. Variant B uses `--output-tensor-type q8_0`, C `q4_k`.
3. **Verify the tensor types** by dumping the result: MXFP4 = every large matrix of the 64 layers;
   Q8_0 = token_embd, ssm_alpha/beta (48 × 2), the whole MTP layer `blk.64` (attention q/k/v/o, FFN,
   `nextn.eh_proj`); F32 = norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`; output = Q6_K / Q8_0 / Q4_K.

| Variant | File | Bytes |
|---|---|---|
| A (recommended) | `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` | 15,815,469,280 |
| B | `…-B-outQ8_0.gguf` | 16,123,386,080 |
| C | `…-C-outQ4_K.gguf` | 15,487,686,880 |

### 5.2 Traps in this recipe

- **The imatrix barely matters here.** Mainline `llama-quantize` ignores the importance matrix for
  MXFP4 (`GGML_UNUSED(quant_weights)`) and for Q8_0. Only the K-quant output head (A's Q6_K, C's
  Q4_K) uses it.
- **The reference MXFP4 file is MXFP4 everywhere** — token_embd, ssm_alpha/beta and the MTP layer
  included — with a Q6_K head. We raised embeddings, MTP and alpha/beta to Q8_0. WHIRL's speed mode
  applies either way.
- <a id="f32-alpha-beta"></a>**Do not store `ssm_alpha` / `ssm_beta` as F32.** The first version
  followed a spec that kept them F32: WHIRL's MTP smoke test failed on all three variants (MTP output
  ≠ plain greedy) and the automatic draft count dropped to 1. Cause: WHIRL has no int8 GEMV for F32
  weights, so n = 1 used an f32-activation GEMV and n ≥ 2 (verify) an f16 GEMM — different numerics,
  breaking MTP's bit-exactness. With Q8_0 (which has a fused exact kernel, as unsloth's Q4_K_M uses)
  all variants passed and drafts returned to 8. (An engine fix — one kernel family for F32 small
  matrices at all n — is queued.)

### 5.3 Results (WHIRL server defaults, unified protocol, 2 rounds)

| | A (Q6_K head) | B (Q8_0 head) | C (Q4_K head) | Qwen3.8 MXFP4 reference |
|---|---|---|---|---|
| Decode, MTP + n-gram (default) | 157.2 | 143.6 | **157.5** | 156.9 |
| … zh think0 / think1 / edit | 106.0 / 87.9 / 325.7 | 90.6 / 80.4 / 306.3 | 101.1 / 90.0 / 330.8 | 107.0 / 83.3 / 329.5 |
| Decode, MTP | **117.0** | 98.0 | 112.0 | 118.1 |
| Decode, no MTP | 36.8 | 36.1 | **37.5** | 36.9 |
| MTP acceptance (MTP / MTP + n-gram) | 65.4% / 66.3% | 73.2% / 72.9% | 68.3% / 69.1% | 60.6% / 62.2% |
| Prefill 2k / 8k / 32k | 3164 / 3302 / 2570 | 3181 / 3291 / 2563 | 3138 / 3307 / 2572 | 3223 / 3338 / 2602 |

- **B is slowest** (MTP −16% vs A): every draft step runs the output head, and the Q8_0 head
  (1.3 GB) is the most expensive; its higher acceptance cannot compensate.
- **A beats C under MTP by 4.5%** although C is 1.9% faster without MTP: WHIRL has dedicated paths for
  a Q6_K head (the 2-bit draft head is requantized from it; verify uses a Q6_K-specific multi-row
  table). A Q4_K head gets neither. With n-gram the two tie (157.2 vs 157.5).
- Swift vs the Qwen MXFP4 reference: same decode speed (the Q8_0 MTP layer makes drafts slightly more
  expensive; 5 points higher acceptance compensates); prefill 1–2% slower (Q8_0 embeddings,
  alpha/beta, MTP layer).
- **Recommendation: A.** Fastest MTP in WHIRL, ties C with n-gram, more precise head than C, same head
  type as the reference so every Q6_K-head path applies. C is a reasonable choice for llama.cpp
  users (slightly faster there); B is not recommended.

llama.cpp b11214 also loads all three (the only warning is the known `unknown type mxfp4`; in plain
mode the 4 nextn tensors are reported unused, which is normal) and runs `--spec-type draft-mtp`:
plain 33.3 / 32.8 / 33.8 tok/s, MTP (n-max 3) 58.1 / 55.9 / 60.7, acceptance 73.6 / 72.2 / 75.0%,
prefill 2k ~1206–1215. In llama.cpp, English MTP outputs matched plain; Chinese outputs diverged
mid-way (its batched numerics are not bit-exact).

**Thinking length** (official sampling temp 1.0 / top_p 0.95 / top_k 20 / min_p 0, thinking on,
16,384-token cap, 9 prompts × 2 seeds): Swift A 129,299 thinking tokens / 1,859 s / 5 cap hits;
Qwen MXFP4 169,510 / 2,328 s / 8 cap hits. On the four Chinese coding prompts Swift thought 44% less
(54,948 vs 97,496 tokens) and finished 38% sooner (858 vs 1,395 s). A capped Swift run was checked for
loops (repeated 32-gram share 0.5% in the first half, 1.3% in the second): genuine deliberation, not
degenerate repetition.

## <a id="ornith-mxfp4"></a>6. Ornith-1.5-35B-A3B MXFP4

Quantized by us from the official BF16 weights (not yet published). WHIRL runs it in speed mode:
MXFP4 experts with fp8 activations in prefill, whole-block int8 kernels in decode.

### 6.1 Recipe

1. **Convert** the BF16 checkpoint with llama.cpp's `convert_hf_to_gguf.py --outtype bf16`
   (keeps the MTP layer: `nextn_predict_layers` 1, tensor names as in the official Q4_K_M).
2. **Quantize** with llama.cpp b11214's `llama-quantize`, **no imatrix** (mainline ignores it for
   MXFP4 and Q8_0 anyway), with a tensor-type file:

   ```
   blk\.40\.ffn_gate_inp=f32
   blk\.40\.=q8_0
   ssm_alpha=q8_0
   ssm_beta=q8_0
   token_embd=q8_0
   attn_gate=mxfp4
   attn_qkv=mxfp4
   attn_q\.=mxfp4
   attn_k\.=mxfp4
   attn_v\.=mxfp4
   attn_output=mxfp4
   ffn_gate_exps=mxfp4
   ffn_up_exps=mxfp4
   ffn_down_exps=mxfp4
   ffn_gate_shexp=mxfp4
   ffn_up_shexp=mxfp4
   ffn_down_shexp=mxfp4
   ssm_out=mxfp4
   ```

   ```
   llama-quantize --tensor-type-file tt_orn.txt --output-tensor-type q6_k \
       ornith-1.5-35b-bf16.gguf Ornith-1.5-35B-A3B-MXFP4.gguf MXFP4_MOE 16
   ```

   Result: experts, shared expert and every dense matrix MXFP4; `ssm_alpha` / `ssm_beta`, the
   embeddings and the whole MTP layer (`blk.40`) Q8_0; routers (`ffn_gate_inp`, also in `blk.40`),
   `ffn_gate_inp_shexp`, norms and the small SSM tensors F32; output head Q6_K. 19,819,767,136
   bytes (4.46 bpw, 18.45 GiB loaded).
3. **Vision projector**: `convert_hf_to_gguf.py --mmproj --outtype f16` (899,283,296 bytes).

### 6.2 Results (R9700)

Same GGUF on both engines; WHIRL greedy, MTP output identical to plain greedy output (and to the
research prototype's output token for token); llama.cpp b11214 ROCm, `-fa on`, best of
`-ub 512 / 2048` (2048 was faster for every prefill size).

| | WHIRL MXFP4 | WHIRL Q4_K_M | llama.cpp MXFP4 | llama.cpp Q4_K_M |
|---|---|---|---|---|
| Prefill 88 / 2k / 8k / 32k tokens (tok/s) | 2,542 / 11,729 / 10,853 / 7,978 | 1,338 / 5,862 / 5,992 / 4,980 | 1,820 / 4,820 / 4,760 / 3,936 | 1,725 / 4,383 / 4,328 / 3,639 |
| Decode, MTP + n-gram, 7 zh/en coding prompts (median, 800 tokens) | 253 tok/s | 224 tok/s | — | — |
| Decode, MTP + n-gram, editing prompts (bugfix / refactor-js) | 272 / 253 tok/s | 224 / 241 tok/s | — | — |
| Decode, no MTP (same prompts; llama.cpp `tg256`) | 194 tok/s | 180 tok/s | 120 tok/s | 103 tok/s |
| Server, 1 request, MTP + n-gram (llama.cpp: no MTP) | 246 tok/s | 203 tok/s | 113 tok/s | 96 tok/s |
| Server, 4 concurrent requests, aggregate | 435 tok/s (MTP + n-gram), 355 (no MTP) | 347, 321 | 227 (no MTP), 138 (MTP) | 219, 84 |

The server rows use the same client for both engines (7 bench prompts, 800 tokens, thinking off,
top_k 1); llama.cpp's MTP (`--spec-type draft-mtp`, 3 drafts) was slower than its plain decode on
this model, so its plain numbers are the fair comparison.

The fp8 expert prefill (default) is 37% faster at 2k than the f16-activation expert GEMM
(`WHIRL_MOE_FP8=0`: 1,920 / 8,591 / 8,532 / 6,636 tok/s). Its first version was slower than the f16
path (5,249 tok/s at 2k) because the down projection wrote its f32 outputs as eight scattered
4-byte stores per lane; each lane holds 8 consecutive rows of one token, so two 16-byte stores carry
the same values (same bits). In a 2k-token prefill profile the GEMM time dropped from 187 ms
(f16 experts) to 122 ms (fp8 experts).

## <a id="rules"></a>7. Checklist for making a GGUF that runs well on WHIRL

1. Keep `general.architecture` `qwen35` / `qwen35moe` and **keep the MTP layer** (`nextn`) —
   without it WHIRL decodes at the no-MTP ceiling (~38 tok/s for a 27B 4-bit model).
2. Do not use F32 for `ssm_alpha` / `ssm_beta`; use Q8_0 (or MXFP4).
3. Prefer a **Q6_K output head**: it feeds the 2-bit draft head and has its own multi-row tuning.
   Avoid Q8_0 heads (every draft step pays for them).
4. Q8_0 for token_embd and the MTP layer is fine (small speed cost, slightly higher acceptance).
5. Stay within the supported types (§2); anything else is refused at load.
6. Check with the MTP exactness test: plain greedy, MTP, MTP + n-gram must give identical text.

## <a id="mmproj"></a>8. Vision projector (mmproj)

| File | Type | Size |
|---|---|---|
| `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` | F16, 334 tensors | 927.6 MB |
| Ornith-1.5 mmproj | F16 | 899,283,296 bytes |
| Qwen3.8 mmproj (reference) | F16 | parsed in our GGUF parity tests |
| Ornith mmproj (reference) | BF16 | supported (BF16 loader path) |

The Swift mmproj was verified with llama.cpp's `mtmd` (images encode correctly). WHIRL's own image input
is described in [vision.md](vision.md).
