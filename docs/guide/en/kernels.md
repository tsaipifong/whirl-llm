**English** | [繁體中文](../zh-TW/kernels.md)

# Kernels

> **Status.** Every kernel family below is in `kernels/*.hip` for gfx1201 (R9700); the gfx1151
> (8060S) set is in `kernels/gfx1151/` (preview, untuned; see [3.2](#gfx1151)). Numbers are R9700 (gfx1201) unless marked 8060S (gfx1151; those come
> from the research build). "Bit-exact" always means identical output bits, verified by a permanent
> check, not "within tolerance".

**Who this helps:** people writing HIP kernels for RDNA 3.5/4 (WMMA layouts, int8 dot
products, register pressure), llama.cpp/ggml contributors working on quantized GEMV/GEMM for
AMD, and anyone who needs batched and unbatched code paths to produce identical bits.

## <a id="ceilings"></a>0. Ceilings we measure against

We never optimize against the spec sheet. Every ceiling below was measured with a probe kernel
on this card, and every result in this document is quoted relative to it.

| Ceiling (R9700, gfx1201) | Measured | How |
|---|---|---|
| WMMA f16 → f32 (`v_wmma_f32_16x16x16_f16`) | 180–184 TFLOPS | register-only loop |
| WMMA f16 → f16 accumulate | ~177 TFLOPS | not used: barely faster, much less precise |
| WMMA iu8 (`v_wmma_i32_16x16x16_iu8`) | 343–374 TOPS | register-only loop |
| iu8 WMMA + one `v_mad_i32_i24` per element | ~205 TOPS | the exact-Q4_K int8 epilogue ceiling (§5.5) |
| fp8 WMMA | ~344 TFLOPS at cold clock; ~260–285 sustained at the ~2.4 GHz seen during prefill | from r9700-stack's published microbenchmark notes |
| Streaming read, row-per-wave GEMV pattern, 16-byte loads | 604–626 GB/s (spec 640) | pure-read kernel, > 512 MB rotated to defeat the 64 MB Infinity Cache |

| Ceiling (8060S, gfx1151) | Measured |
|---|---|
| WMMA f16 | 43 TFLOPS |
| WMMA int8 | 45 TOPS (same as f16) |
| `v_dot4` int8 | 2.7 T lane-op/s (R9700: 9.4) |
| Streaming read | 238 GB/s `hipMalloc`, 232 GB/s pinned |
| Launch overhead | 1.3–3.3 µs (R9700 2.2–2.5) |

Hardware facts that shaped every kernel: waves are 32 lanes (Vulkan reports 64 — a different
grouping by the driver); gfx12 WMMA works on 16×16×16 tiles with A/B fragments in registers;
LDS is 128 KiB per WGP and at most 64 KiB per workgroup, so two blocks of up to 64 KiB can share a
WGP — HIP's occupancy API assumes 64 KiB per WGP and reports half of that
([pitfalls KERN-20b](pitfalls.md#kern-20b)). Using > ~41 KB cut occupancy from 16 to 6 waves/SIMD
in one of our kernels; **registers are the real limit** — once a kernel exceeds its VGPR budget the
compiler spills to scratch memory and performance collapses.

## <a id="gemv"></a>1. Decode GEMV: int8 activations, `v_dot4`, `v_perm` lookup tables

Decode is a pure bandwidth problem: every token streams all weights once (14.33 GB for the
UD-Q4_K_M file, which mixes nine formats). The design:

- **Activations are quantized to int8** in groups of 32 with one scale per group, by the kernel
  that produces them (RMSNorm + quantize, SiLU·mul + quantize, attention combine + quantize are
  fused).
- **Weights are decoded to int8 in registers** by a per-format decoder (`QD<T>::dot`), and dot
  products use `v_dot4` (`__builtin_amdgcn_sudot4`, signed × unsigned int8 4-way dot).
- **Wide units.** Q4_K, Q5_K and IQ4_XS are processed in 64-weight units (two 16-byte loads, one
  header decode), which cut the float work per weight to a quarter of a 16-value design.
- **Non-linear 4-bit tables via `v_perm_b32`.** IQ4_NL/IQ4_XS (and later MXFP4) map a 4-bit code
  through a 16-entry table. Two `v_perm_b32` byte selects plus one blend on bit 3 do the lookup
  entirely in registers, with no memory access.
- **Small-matrix fusion.** The DeltaNet beta/alpha projections are tiny (48 × 5120, Q8_0 in the
  unsloth file): run alone they reach only 101 GB/s because launch cost dominates, so they are
  fused with the causal conv + L2 norm into one grid.

Result of the first full design: matmul 25.3 ms/token (566 GB/s, ~88% of spec), full step
28.8 ms = 34.7 tok/s. After the later work below, a decode step is **27.74 ms (Q4_K_M) /
25.15 ms (MXFP4)**.

### 1.1 How close each format is to the ceiling

Every 1-token GEMV compared with a pure-read kernel using the same access pattern (real
weights, > 512 MB rotated):

| Type / shape | Production 1-token | Pure-read ceiling | Verdict |
|---|---|---|---|
| q4_k 17408×5120 | 587–606 GB/s | 604–611 | at bandwidth (0–3%) |
| q5_k 10240×5120 | 584–591 | 587–599 | at bandwidth |
| iq4_xs 5120×17408 | 573–593 | 598–612 | at bandwidth (2–4%) |
| mxfp4 up / down / qkv | 589–600 / 584–593 / 567–575 | 603–611 / 605 / 587–590 | at bandwidth (2–3.5%) |
| q6_k 5120×6144 / output head 248320×5120 | 576 / 626 | 578 / 626 | = ceiling |
| iq4_nl 5120×17408 | 591 | 601 | at bandwidth |
| q8_0 17408×5120 / 5120×17408 | 599 / 594 | 622 / 615 | at bandwidth |
| **q3_k** 17408×5120 / 5120×17408 | **491 / 466** | 593 / 598 | ALU/latency bound |
| **iq3_s** 17408×5120 / 5120×17408 | **502 / 482** | 595 / 597 | ALU/latency bound |

Things that gave **no** gain on the at-bandwidth formats (all within ±2%, all bit-identical):
activations in LDS (per block or persistent), persistent grids (256/512/1024 blocks), 2 or 4 rows
per wave sharing activations, loading 2–4 units per lane before computing, non-temporal loads
(slower), block sizes 128/512/1024. A public report of +15% from `v_perm` unpacking on this card
was relative to a ggml baseline; WHIRL already used `v_perm` tables.

**Q3_K / IQ3_S fix.** The ISA showed the Q3_K scale lookup
`kk < 8 ? sb8[kk] & 0xF : sb8[kk-8] >> 4` compiled to a wave-divergent branch: load `d`, branch,
load the scale byte, `s_wait_loadcnt 0`, then issue the weight loads — two serialized memory
latencies per super-block, with one 110-byte super-block per wave step (too little memory-level
parallelism). The fix keeps lane l owning unit l of each super-block and the exact same float
expression, but reads the scale nibble branch-free (fixed byte `96 + (kk & 7)`, shift 0 or 4),
issues all loads for 4 super-blocks before computing, and uses 2-byte-aligned u16/u32 loads.
Bit-identical; q3_k 491 → 588–590 and 466 → 581–583 GB/s, iq3_s 502 → 583–586 and
482 → 577–579 GB/s (+16…+25%, ≈ 98% of ceiling).

### 1.2 Small-kernel fusion in decode

| Fusion | Launches replaced |
|---|---|
| `attn_prep`: q-norm, k-norm, RoPE on q and k, KV-cache write | 5 → 1 per attention layer |
| `attn_combine_q8`: split-K combine writes the int8 input of the o-projection GEMV | removes one quantize launch |
| `gdn_abconv`: beta/alpha projections and conv + L2 norm in one grid (idle threads add 0 so the tree order is unchanged) | 2 → 1 per DeltaNet layer |
| recurrent step + gated RMSNorm + int8 quantize | 3 → 1 |

Launches per decode step went 868 → 740, decode step 28.37 → 28.17 ms (−0.7%). The remaining
inter-launch gaps were about 1 ms per step; later we fused same-input GEMVs (§4).

## <a id="multirow"></a>2. Multi-row GEMV that is bit-exact to the 1-row GEMV

Speculative decoding verifies k drafts by running k+1 rows through the model. If the verify
kernels compute a row with a different floating-point order than the 1-token decode kernels,
logits differ in the last bits, near-tied tokens flip, and MTP output stops matching plain
greedy output. We require bit equality, so the arithmetic is shared:

- **Canonical order.** Each unit u is accumulated into slice `u % 32`, in increasing u, with the
  same float expression; the 32 slices are merged by a fixed xor tree (16, 8, 4, 2, 1). The
  1-token kernel uses the same code path (`QDN<T>::dot<1>`), so "1 row" is just the n = 1 case.
- **Multi-token dp4 kernels** for 2–16 rows are macro-generated; from 6 rows each wave handles 2
  rows sharing the loaded activations. Matmul time: T=1 25.35 → 25.0 ms, T=8 42 → 36,
  T=11 64 → 41, T=16 100 → 53 ms (before the WMMA versions below).
- **Permanent check `checkGemvBitwise`:** every type × row count 2..16 × every kernel variant
  (and the output head) must equal the 1-token result bit for bit.

WMMA GEMM was far too slow at these batch sizes (~158 ms for a 16-row step), so ≤ 16 rows always
use GEMV-style kernels.

## <a id="wmma-gemv"></a>3. int8 WMMA mid-batch GEMV (3–16 rows, gfx1201)

`v_wmma_i32_16x16x16_iu8` with A = activations (16 token rows) and B = decoded weights (16
weight rows). The integer dot product inside WMMA is exact, so only the floating-point part has
to reproduce the canonical 1-token order:

- every unit (64 weights for Q4_K/Q5_K/IQ4_XS; each type's own unit for Q6_K/Q3_K) accumulates in
  slice `u % 32` in increasing u with the 1-token float expression; wave w owns slices w, w+8,
  w+16, w+24, so xor-16 and xor-8 happen in registers and xor-4/2/1 through LDS — the same tree;
- Q4_K/Q5_K need Σx per sub-block for the `dmin·m` term: one extra WMMA with B = all ones puts it
  directly in the D layout;
- Q6_K uses signed (q − 32) weights with a half-masked WMMA for its 8-value scale groups; Q3_K
  and IQ3_S pair 8-value units; IQ4_NL is one WMMA per unit.

Register fixes that made it fast: the wave index made uniform with `readfirstlane` (scale
branches become scalar), one 16-byte header load with branch-free scale decode, and
`asm volatile` pins on each unit's accumulator — without them the compiler interleaved the
WMMAs of four units and used 256 VGPRs plus hundreds of bytes of spill.

Result: matmul T=8 36.5 → 30.5 ms, T=16 53.2 → 32.5 ms (T=1 unchanged); verify n=16
68.3 → 42.8 ms; single-user MTP +6–8%; 4–16 concurrent users +16–25% aggregate.

### 3.1 Second generation: pipelined `gw2` / `gw8`

A probe that reproduced only the memory pattern (same 16 rows × 32 B weights + headers +
activations) reached 608–616 GB/s, so the access pattern was not the bottleneck. The production
ISA had an `s_wait_loadcnt 0` per unit: a `j < 2` scale branch split the header load into
conditional loads, serializing activations → header → weights. The rewrite keeps the identical
lane/unit/slot mapping, float expression and reduction (bit-identical to 1-token), and:

- iterates u = w + 8k so the scale class `j = u & 3` is constant per wave → loops specialized per
  class, no branches; the 16-byte header load is unconditional;
- issues all loads of the next unit (or next 3 units) before computing the current one (ring of
  2 or 4);
- has **no `break` in the main loop** (tail handled separately): with a `break`, the compiler
  zeroed the wait counter every step and prefetching did nothing;
- loads activations as one dword per lane (token = lane % 16) plus shuffles instead of eight
  64-bit loads per unit;
- blends LUT nibbles with `(b << 8) - b` behind an asm barrier instead of `v_mul_lo_u32` (the
  compiler otherwise folds it back into a multiply).

Variants: v5 = 1 tile / depth 1, v6 = 1 tile / depth 3, v7 = 2 tiles / depth 1, v8 = 2 tiles /
depth 2. `gw8` handles the 8-value-unit types (Q3_K, IQ3_S) with each lane decoding its own four
unit scales and the IQ3_S grid table in LDS.

Default selection (rows → variant), measured per type:

| Type | Selection |
|---|---|
| Q4_K | v1 at 3 rows, v6 for 4–7, v5 from 8 |
| Q5_K | v6 for 3–5, v5 from 6 |
| IQ4_XS | v1 for 3–5, v6 for 6–11, v8 from 12 |
| MXFP4 | v1 at 3, v6 for 4–13, v8 from 14 |
| Q6_K (layers) | v5 for 6–7, v7 from 8 |
| Q6_K output head (248,320 rows) | v2 from 6 (its own table; pipelined versions lost: 2.16–2.30 vs 2.07 ms) |
| Q3_K | new dp4 up to 4 rows, v5 from 5 |
| IQ3_S | dp4 below 12 rows, v5 from 12 |
| IQ4_NL, Q8_0 | unchanged (dp4 / earlier WMMA variants) |

Result: verify n=16 41.61 → 39.62 ms (Q4_K_M), MXFP4 T=16 matmul 27.86 → 25.94 ms; four
concurrent users' steady-state throughput 241.1 → 252.9 tok/s (Q4_K_M) and 276.7 → 289.8
(MXFP4); single-user coding benchmark +3.8%, file editing +4.4% (Q4_K_M). Remaining cost of
16 rows vs 1 row: q4_k 1.09×, q5_k 1.11×, iq4_xs 1.12×, mxfp4 1.16×, q6_k layers 1.46×, head
1.25×, q3_k 1.9×, iq3_s 2.5× — the rest is the canonical float expression per (row, token,
unit), which bit-exactness forbids skipping. Things that lost: 3–4 tiles per block (spills,
−7…−40%).

### <a id="gfx1151"></a>3.2 gfx1151 (RDNA 3.5) differences

The gfx1151 kernel set (`kernels/gfx1151/`, one code object) keeps the gfx1201 kernel names and
argument ABIs, so the host and `KernelTable` are shared. It has the int8 decode GEMVs (1-token,
2–16-token WMMA and multi-row), the f16 WMMA prefill GEMMs (24 configurations; MXFP4 is dequantized
to f16 first), paged f16 / q8 KV attention, DeltaNet (per-token, chunked f32, sequential scan, fused
decode), the MoE kernels (MXFP4 experts on the generic int8-decode / f16-prefill entries), the
2-bit MTP draft head and the f16 prefill activation fusions. **Not on gfx1151** (the host takes
the remaining path, see `kernels::Caps`): fp8 / int8-dot WMMA GEMMs (no fp8 WMMA in RDNA 3.5:
MXFP4 prefill uses f16 activations), the small-batch GEMMs, the int8-WMMA mid-batch GEMVs (`gemvw`)
and the grouped multi-token twins, the whole-block MXFP4 decode experts and the fp8 grouped expert
GEMM, the `q8v` / `q8h` KV formats (auto picks f16, then q8), DeltaNet replay, the WMMA DeltaNet
prefill, the key-split prefill attention, and the vision kernels. Its int8 activation scale words
carry the block sum (`pk_make`: scale with an 11-bit mantissa + sum), which saves the per-row sum in
the dot kernels; CPU references must decode them (`ref::xdScale`). Correctness on the 8060S:
`whirl-kernel-test` passes every check it runs, and MTP / MTP + n-gram output equals plain greedy on
the 27B Q4_K_M, Swift MXFP4-A and Ornith MXFP4 files (7 prompts each, up to 32k tokens).

- On the 8060S `v_dot4` runs at half the int8 WMMA rate and each verify row re-reads
  activations, so the 2–16-row kernels for Q4_K, Q5_K, IQ4_XS and Q6_K use int8 WMMA; the 1-token
  dp4 kernel shares the same slicing and summation order (MTP stays bit-exact). Verify cost
  relative to 1 row: n=4 1.36 → 1.23×, n=8 1.91 → 1.50×.
- **gfx11 WMMA needs the B fragment in both half-waves.** Originally each half-wave decoded the
  whole 64-value unit (the IQ4_XS LUT decode is expensive). Now each half decodes its own 32-value
  half and the halves are exchanged with `v_permlanex16`. Integer dot unchanged → bit-identical;
  IQ4_XS T=3 27.68 → 25.37 ms (−8…−9%).
- **gfx1151 decode MoE kernels** originally used 16-value units with an 8-byte load per lane
  (166–180 GB/s); switching to the 64-value wide units gave decode −2.3% and 9-row verify −14% on
  the MoE model.

## <a id="grouped"></a>4. Grouped launches for same-input GEMVs

Launch head/tail costs about 2.8 µs. Matrices that read the same input — attention q/k/v, FFN
gate/up, DeltaNet qkv/z, the MoE shared expert's gate/up (and the same blocks inside the MTP
layer) — are launched once when they share type and column count:

- 1-token entries take a `GvArgs` struct with up to three segments `(W, y, row_bytes, nrows)`;
  block ranges `[0,b1)`, `[b1,b2)`, `[b2,…)` map to segments, and the block index is rebased per
  segment. Each row's kernel and float expression are unchanged → bit-identical.
- A marker kernel tells the host whether the code object has the grouped ABI (gfx1151 objects
  without it use the old ABI automatically).

**Codegen trap.** Giving *every* GEMV entry the grouped ABI changed multi-token code generation:
whenever the weight pointer did not come straight from a kernel argument (a `select`, or
`W + offset`), VGPRs grew 30–90% and some variants spilled (q6_k v7 at T=8: 0.92 → 1.97 ms).
`readfirstlane`, `__builtin_assume` and integer offsets did not help; three inlined branches were
worse. Solution: single matrices keep the original entries; groups use separate "twin" entries,
and the shared implementation is `__forceinline__` (with two callers it stopped inlining and hit
248 VGPRs + scratch). Twins are used up to 11 rows (13 for MXFP4), above which they lost.

| | Q4_K_M before | after | MXFP4 before | after |
|---|---|---|---|---|
| Launches per token | 740 | 674 | 740 | 596 |
| Decode step (no MTP) | 28.09 ms | 27.74 (−1.2%) | 25.66 | 25.15 (−2.0%) |
| Verify n=3 | 30.99 | 30.22 (−2.5%) | 27.70 | 27.26 (−1.6%) |
| Verify n=5 | 32.38 | 31.71 (−2.1%) | 29.00 | 28.44 (−1.9%) |

Cost: kernel build time 5 → 13 minutes. Mixed-type groups (in the unsloth file q/k/v are
q4_k/q5_k/q5_k) still launch separately.

## <a id="prefill-gemm"></a>5. Prefill GEMM

### 5.1 The base design

- A templated WMMA GEMM `gemm3_impl<T, BM, BN, BK, WAVES_M, MODE, NTH, TOK_X>`.
- Fast f16 decoders `dec_group_h<T>` turn quantized weights into WMMA fragments.
- Two paths, chosen per (type, shape): **fused** (decode while computing) and **dequantize to
  f16 first, then a pure f16 GEMM** (better for large batches, where the dequant is amortized).
- **`TOK_X` grid order:** the grid is `{ceil(n/BN), ceil(rows/BM), 1}` — tokens on x — so
  consecutive workgroups reuse the same weight tile from L2 instead of main memory. This rescued
  the first fast-dequant version and pushed beyond it.
- **Autotuning** of every tile configuration per (type, shape, batch bucket), cached on disk.

Pitfalls in this design (details in [pitfalls.md](pitfalls.md#kern)):

- The fast dequant raised register pressure; the most common tile spilled 552 bytes/lane and
  prefill regressed badly. A no-spill search compiled every candidate, read the resource usage
  and kept only the 24 configurations without spills.
- Autotuning that synchronized after every repetition measured launch overhead and favored the
  fused path. Timing a batch of repetitions with one final sync fixed it. (Cold-cache tuning did
  not help.) A 32-bit candidate mask overflowed at 48 candidates (undefined shift) → 64-bit.

Result of the first round: all matmuls of the model at T=4096 ran at 107–108 TFLOPS (~59% of
180), the best pure f16 GEMM at 119 TFLOPS (~66%); prefill 1,296 / 1,430 / 1,355 tok/s at
1.1k / 14k / 24k tokens.

### 5.2 Batch buckets

| Change | Why | Effect |
|---|---|---|
| 2 buckets (tuned at 512 and 4096) | first design | an 85-token prompt was padded into a 256-token tile |
| + bucket n ≤ 128 tuned at 128 | 85-token FFN-up took 0.791 ms with the 512-tuned tile vs 0.271 ms with an existing 64×128 tile | 85–89-token prefill 184–188 → 132–133 ms (−29%); n ≤ 256 was tried first and slowed 152/231-token prompts 3–4% |
| 11 buckets: ≤32, 48, 64, 96, 128, 192, 256, 384, ≤768 (tuned at 512), 769–1024 (tuned at 1024), > 1024 (tuned at 4096) | the server runs 1024-row chunks, which fell in the 512-tuned bucket | changing only the tune file gave server 8k cold prefill 1318 → 1441 tok/s (+9.3%) |

Tile choice only changes M/N blocking; the K-direction WMMA order is fixed, so every tile
configuration produces identical bits. A permanent invariance check runs every GEMM option
(61 in the current set), every MoE token tile and 40-row subsets and requires bit equality —
this is what makes batched multi-request prefill equal to solo prefill.

### 5.3 `gemmh_f16`: fragment-order LDS for f16 weights

Both operands go through LDS but are stored in WMMA fragment order, so every fragment read is a
conflict-free `ds_load_b128`. Configuration: 128×256 tile, BK 32, 8 waves (TM 2 × TN 8), single
buffer, rasterization in groups of 4 token blocks, 16-byte epilogue stores. +19.6…+30.2% over
the previous f16 kernel (about 126–130 TFLOPS), bit-identical, no data layout change anywhere
else. Used for n ≥ 512. Rejected variants: activations read directly in tiled form (+25–31%, but
every f16 producer would change — not worth it over the above), weights read directly in tiled
form (+22–28%), both direct (+12–22%), double buffering / 512 threads / BK 64 (worse), no
rasterization grouping (−9…−49%), and **fusing the dequant into the GEMM's weight fetch**
(bit-identical but −10…−13%: Q4_K/Q5_K/Q6_K decode VALU work lands on the GEMM's critical path).

Q4_K_M CLI prefill gained +9.6…+14.0% at 2k–32k and +8.1…+8.3% at 128k.

### 5.4 Small batches: `gemms` / `gemmsd`

For a 35-token prompt, 92% of prefill was matmul and weight streaming ran at only 120 GB/s:
most WMMA work was padding. Two new tile families became **autotune options** (the tuner picks
them per type, shape and bucket; nothing is hard-coded):

- `gemms` (BN 16/32/48/64): per 256-k stage the **raw quantized bytes** are copied into LDS with
  16-byte alignment that preserves each super-block's mod-16 alignment in global memory (so
  `dec_group_h` reads identical bytes), the activation slab also goes to LDS, and the next stage
  is prefetched into registers. Token tiles 16–64, no padding to 128/256.
- `gemmsd` (64×64, 64×96, 64×128): the same raw stage, but the whole block cooperatively decodes
  a 128-k half-stage into an f16 A tile once, then runs WMMA from LDS (for n ≈ 96–512).

Numerics are identical to `gemm3` (A fragments from `dec_group_h`, B from the same f16 X, K in
increasing steps of 16). Probe: +57…+175% for n = 17–64 vs the best existing configuration.
Ablation: load + LDS store + barrier alone took 0.105 ms (~440 GB/s) and the compute 0.08 ms
barely overlaps — one-stage prefetch with two blocks per WGP (the activation slab fills LDS) is
the current ceiling, and gfx12 has no 16-byte asynchronous global→LDS load. Lost: deeper
register prefetch (spills), activations read directly from global (too much L2 traffic), half
activation slabs, weights decoded without LDS (−20…−66%).

CLI prefill (Q4_K_M): 88 tokens 692.7–697.5 → 848.5–867.3 tok/s (+23.4%, TTFT 127 → 103 ms),
209 tokens +9.5%, ≥ 493 tokens unchanged.

### <a id="int8-prefill"></a>5.5 Negative result: int8 (W8A8 / exact W4A8) prefill on gfx1201

On paper int8 WMMA is 2× f16. Register-only, iu8 WMMA plus a per-32-k float epilogue
(convert + scale + fma) ran at 356 TOPS — VALU and WMMA overlap on gfx12. In memory-fed GEMMs
it never beat f16:

| Attempt (FFN-up 17408×5120, 4096 tokens) | Best |
|---|---|
| Current f16 path (dequant + f16 GEMM) | 108–116 TFLOPS (6.3–6.8 ms) |
| W8A8, exact per-32-k weight scale, 15 variants over three generations | 96.5 TOPS (7.57 ms) |
| W8A8, lossy per-128-k weight scale | 108.8 TOPS (6.71 ms) |
| Exact W4A8 (Q4_K/Q5_K/IQ4_XS weights untouched, int8 activations): v1 from LDS raw blocks | 52.5 TOPS |
| … HIP `int4` struct arrays replaced by ext_vector types (prefetch left scratch) | 95.0 |
| … header decoded at load time + `v_mad_i32_i24` via inline asm | 118.2 |
| … weights unpacked to int8 in LDS with the activation permutation, 128×128, 512 threads | **126.1 (Q4_K), 117–120 (IQ4_XS, Q5_K)** |
| … plus 64-k chunks and double-buffered LDS | 110.2 |

With activation quantization included, the best exact W4A8 was only +6.6…+10.4% faster than
f16 on FFN-up, and −4…−7% on down-projection. The ceiling explains why: two chained iu8 WMMAs
plus one `v_mad_i32_i24` per element (the sub-block integer scale that exact Q4_K cannot avoid)
top out at ~205 TOPS; gfx1201 overlaps only about 4–5 VALU instructions per WMMA for free (4 VALU
per WMMA: 217 TOPS, 8: 102 TOPS). Unpacking, addressing and LDS take the rest, so practice
lands at 120–130. Precision was also worse: per-256 int8 activations gave 1.32e-2 relative GEMM
error vs 3.8e-4 for the f16 path (35×); per-32 activations need a per-sub-block float epilogue
(74–84 TOPS). Independent confirmation: llama.cpp's Vulkan int8 coopmat path (PR #27952)
disables int8 for q4_K/q5_K/q4_1/q5_1/nvfp4 on this same card because it is slower than fp16. On
gfx1151 int8 WMMA runs at the f16 rate and gfx11 WMMA shares the VALU, so we did not try.

Kept as a lesson: an int8 GEMM *without* sub-block scales reaches ~55% of peak on RDNA 4 (a
public W8A8 kernel: ~210 TOPS on a 9070 XT); the sub-block epilogue of K-quants is what kills it.
MXFP4 avoids the problem entirely (§6).

## <a id="mxfp4-gemm"></a>6. MXFP4 × fp8 WMMA prefill (speed mode)

**The idea (credited to [r9700-stack](https://github.com/bkvargyas/r9700-stack), Apache-2.0):**
an MXFP4 block is 32 e2m1 values with one E8M0 exponent — a pure power of two. Power-of-two
scales fold exactly into an fp8 (e4m3) exponent: with `ref` = the largest block exponent of the
row, `w8 = e2m1 × 2^-(ref − e)` is exact while `ref − e ≤ 8` (inside e4m3's subnormal range).
After `v_wmma_f32_16x16x16_fp8_fp8` only one epilogue remains:
`y = acc × sx[t] × 2^(ref − 127)`, with one fp8 activation scale per token (`amax / 448`).
Weight unpacking happens while writing LDS (per 32 values: one 16-byte load, two `v_perm` table
lookups plus the sign bit), averaging under one VALU instruction per WMMA — no sub-block integer
epilogue, so none of §5.5's ceiling applies. Across 7 sampled tensors, the per-row
"max exponent − block exponent" was ≤ 8 everywhere; over the whole Qwen3.8 MXFP4 model only 56
non-zero blocks exceed 8 and get rounded (the loader counts and logs them).

First probe (4096 tokens, real weights):

| Shape | f16 path | fp8 GEMM | incl. activation quantization, vs f16 |
|---|---|---|---|
| FFN up 17408×5120 | 6.40–6.68 ms | 3.67–3.77 ms (194–199 TOPS) | +65–70% |
| down 5120×17408 | 7.01–7.11 | 3.47–3.57 (204–210) | +69–71% |
| qkv 10240×5120 | 3.93–4.03 | 2.17–2.19 (196–198) | +59–61% |

Accuracy: 4e-8 against an f64 reference using the same fp8 activations (the kernel is exact);
2.57e-2 relative against f32 activations, versus 2.0e-4 for the f16 path. That error is the
3-bit e4m3 mantissa itself — no choice of per-token or per-block scale changes it. This is why
MXFP4 is the *speed* mode ([quantization.md](quantization.md)).

### 6.1 `gemm8t`: fragment-tiled activations

The fp8 activation producers (activation quantize, RMSNorm, SiLU·mul, gated norm) write their
output directly in WMMA fragment order: a 16-token × 16-k fragment is 256 contiguous bytes, lane
l holds token `l % 16`, k `(l / 16) × 8 … +8` — exactly the B operand of the fp8 WMMA. Each wave
loads its B fragment with one coalesced `global_load_b64` straight into WMMA registers;
activations never touch LDS. Weights (unpacked with the exponent fold) are written to LDS in
fragment order (conflict-free `ds_load_b64`), in 64-k slabs, double-buffered with one barrier per
slab. LDS per block fell from 55 KB to 16 KB, so several blocks fit per WGP. The WMMA order per
accumulator and the epilogue scale (`st × 2^(ref−127)` == `ldexpf(st, ref−127)`, exact) are
unchanged, so output is bit-identical to the previous fp8 GEMM. (The fragment-tiled activation
technique and the LDS-only barrier sequence `s_wait_dscnt 0; s_barrier_signal -1;
s_barrier_wait -1` are also taken from r9700-stack; the kernel structure and weight layout are
WHIRL's own.)

Tuning sweep (all variants byte-compared with the base): best is 128 rows × 256 tokens, 8 waves
as 4×2 (TM 2 × TN 8 each), BK 64. Rasterizing 4 token blocks per group (row blocks next) gave
qkv +3–4% and attn_q +8–9% (attn_q's 48 KB output stride had conflicts). An epilogue rewrite
(per-row `2^(ref−127)` hoisted, two 16-byte stores per tile) added another 7–19 points. Lost:
BK 32/128, 256×256, 256×128, 64×256, 512/128 threads, `s_setprio`, prefetching the next slab's B
during compute (hits 256 VGPRs, −40%), 16-byte B loads in pair layout (±2%). Small batches:
n ≤ 64 → 128×64, ≤ 192 → 128×128, else 128×256 (later also 128×32 for n ≤ 32 and 128×48 for
n ≤ 48). Earlier attempts to feed weights straight from global into A fragments (183–189 TOPS)
or to double-buffer the old kernel (158–173) were slower than single-buffer + register prefetch.

Per-shape gain over the previous fp8 GEMM: up +42.7%, down +27.9%, qkv +27.1%, z +32.9%,
ssm_out +29.6%, attn_q +73.7%, attn_k +11.6% — about **238–248 TOPS** (was 170–190). MXFP4 CLI
prefill +15.3…+16.4% at 2k–8k.

### <a id="act-fusion"></a>6.2 Prefill activation fusions

`rmsnorm_x8/x16` (RMSNorm writes the next GEMM's input directly, fp8 + per-token scale or f16),
`silu_mul_x8/x16` (SiLU·mul writes the down-projection input), `gated_norm_x8/x16` (DeltaNet
gated RMSNorm writes the out-projection input) and `gdn_conv_l2n` (causal conv + q/k L2 norm, 128
channels × 8 tokens per block, each input row read once). Enabled only when every consumer of a
tensor uses the same GEMM input format; otherwise the unfused kernels run. All bit-identical —
which required asm barriers, because the compiler contracted `x*x` into the first butterfly add
and merged a multiply with the f16 conversion into `v_fma_mix`, each a 1-ulp difference
(§12). Effect at 8k: misc 191 → 135 ms, norm 73 → 52, DeltaNet norm 53 → 36, DeltaNet prep
86 → 50.

Speed mode only (not bit-exact, MXFP4 default): FFN gate/up and DeltaNet qkv/z GEMMs write f16
instead of f32 for their element-wise consumers (+5.4…+5.8% and +0.6%).

## <a id="flash"></a>7. Prefill flash attention

- **The Sᵀ = K·Qᵀ trick.** Computing S = Q·Kᵀ leaves the softmax probabilities P in the D layout,
  but the next WMMA (O += P·V) needs P as an A operand, which would mean a trip through LDS.
  Computing the transpose Sᵀ = K·Qᵀ instead leaves Pᵀ in registers in exactly the layout of the
  B operand of Oᵀ += Vᵀ·Pᵀ. No LDS round trip for P.
- V is transposed in registers (128 query rows per block, 256 threads); KV is stored as f16.
- **Lazy rescale.** After each key tile the running output is normally multiplied by
  `alpha = exp(m_old − m_new)`. When `__builtin_amdgcn_ballot_w32(alpha != 1.f)` is zero for the
  whole wave (the running max did not change — almost always deep in a long context), the 128
  multiplies are skipped. Multiplying by 1 is an identity, so results are bit-identical. −4.5% per
  layer; CLI prefill +1.1…+1.3% at 64k, +2.0% at 128k.
- **`attn_kx`.** Analysis at 120k context: the kernel reached 61 TFLOPS; removing K/V loads made
  it 70% faster; the PV half was the most expensive because 256 VGPRs still spilled 40 (o
  accumulator 128 + Qᵀ fragments 64), and every WMMA read a 512-byte fragment from LDS. `attn_kx`
  uses 16 waves per 128 queries; two neighboring waves share 16 queries, each computes Sᵀ for 16
  keys with the original WMMA chain, they exchange through LDS, both compute the same softmax, and
  each keeps half of Oᵀ (64 VGPRs, no spill); the next K/V tile is prefetched into registers.
  Bit-identical. 120k: f16 KV 218 → 185 ms (+17.8%), q8v KV 247 → 198 ms (+25.0%); 32k +11%;
  ≤ 8k unchanged.
- **`attn_kg`: GQA-grouped, direct loads (default for f16, q8v, q8 and q8h KV).** A per-op profile of the
  0.1.2 build (Swift MXFP4-A, `whirl bench` with `WHIRL_PROFILE=1`) showed that every op class except
  attention costs the same per token at every prompt length; attention grew from 0.015 ms/token (5% of
  prefill) at 2k to 0.228 ms/token (46%) at 64k, at ~61–66 TFLOPS. `attn_kg` regroups the work of
  `attn_kx` without changing one operation per query:
  - one block = the query heads of one KV head (all 6 for Qwen3.8-27B, 4 of Ornith's 8) × 16 queries,
    two waves per head as in `attn_kx` (keys split for Sᵀ, dimensions split for Oᵀ);
  - no shared K/V tiles. K fragments are loaded from the cache straight into WMMA A registers; Vᵀ
    fragments come from `global_load_tr_b128`, RDNA 4's transposing load: it transposes 8×8 16-bit
    blocks across each group of 8 lanes, so each lane supplies one key row and receives exactly the
    A-operand column it needs. The heads of one KV head read the same rows at the same time, so the
    loads hit the cache. No in-register transposes, and one barrier per 32-key tile (the Sᵀ exchange)
    instead of three;
  - q8v: V is dequantized once per block (the same `(f16)q × s` multiply as everywhere else) into a
    double-buffered LDS stage one tile ahead; the same barrier publishes it;
  - q8 / q8h (`attn_kg6_q8` / `attn_kg4_q8`): K is dequantized from int8 in registers with the magic
    number trick — `0x6400 | (b ^ 0x80)` is the f16 1152 + q, so subtracting 1152 and multiplying by
    the scale gives q·s with one rounding, as packed f16 with `v_perm` to assemble the halves;
    `#pragma clang fp contract(off)` keeps the compiler from fusing it into an FMA. Bit-identical to
    `attn_kx_q8`; 225 VGPRs, 6 waves per SIMD (in practice one block per WGP). Probe at 61k:
    `attn_kx_q8` 57.8 → `attn_kg6_q8` 72.1 TFLOPS (+24.7%); the straightforward `(_Float16)q * s`
    conversion gained only 5%, which is why the magic-number form is used;
  - the softmax scale 1/16 is a power of two, so the scores stay unscaled — (s − m)·(scale·log2 e)
    rounds exactly like the (s·scale − m·scale)·log2 e that `__expf` evaluates — and the −inf selects
    go away (exp2(−inf) = 0; key 0 is visible to every query, so the running max is finite after the
    first tile). Tiles that every query of the block sees completely skip the mask.

  Bit-identical to `attn_kx` (kernel-test invariants for groups 6 and 8 and head-range splits;
  last-token logits identical to 0.1.2 at 4.9k–70.9k tokens on all three models, f16 and q8v).
  Probe, 4096 queries per launch: f16, 24 / 4 heads 62–66 → 92–97 TFLOPS; Ornith's group 8 (4 heads
  per block) 60–66 → 78–81; q8v 57–62 → 82–84. Without any K/V loads the same kernel reaches 123
  TFLOPS, and 139 without the softmax as well, so what remains is the load data path and the serial
  16-step Sᵀ chain that exactness requires. Lost (all bit-identical, all slower): Vᵀ through a shared
  LDS stage for f16 (88 vs 93), K and V both through LDS (spills), P·V of tile j−1 interleaved with Sᵀ
  of tile j (89), prefetching the next tile's K (−25%), two query groups per block (equal), longest
  blocks first (−10% on the first chunk), 4-wave blocks for q8v (slower than `attn_kx_q8v`, not used),
  prefill chunks of 8192 rows (−2% end to end).
- Beyond 128k context the launch is split by head range (results unchanged; ~2% cost; never
  applied ≤ 128k) as a precaution after an unexplained `HipFailed` during a 256k prefill
  ([pitfalls.md](pitfalls.md#hip-256k)).
- Lost: skipping the causal mask on off-diagonal tiles (−4% at 30k but +1.3% at 120k), skipping
  only the mask (+8%, worse scheduling), a dimension-split wave pair (1.5× WMMA, no gain),
  prefetch-only (`attn_pf`, superseded), pinning the PV loop (scratch 252 → 28 B/lane but f16
  5% slower). This kernel is very sensitive to code layout: measure every change. fp8 attention
  was not tried because it changes long-context precision.

## <a id="decode-attn"></a>8. Decode attention: split-K and WMMA grouped verify

- **Split-K flash decoding** with a fixed number of splits, each running online softmax over its
  own range of 64-position chunks. Split ranges are computed on the GPU from the device-resident
  position, so the decode step stays graph-capturable. Scoring is lane-per-position: each wave
  scores 32 positions over a 64-dim slice, q broadcast from LDS, partial sums of the 4 slices
  added through LDS — no shuffles. Split-count scan at 24k: 64 / 128 / 256 splits =
  35.5 / 33.9 / 34.7 ms/token.
- **`attn_wsplit1` (WMMA grouped verify).** One block (KV head × split) processes a *group* of
  query columns with WMMA: the q heads of one KV head (GQA) × consecutive verify rows of one
  sequence. Sᵀ = K·Qᵀ and Oᵀ += Vᵀ·Pᵀ; K fragments are read directly from the cache; V is
  transposed into LDS by the first 128 threads; Qᵀ lives in LDS to avoid spills. Groups are formed
  only from rows of the same sequence, contiguous positions and the same split size, ≤ 16 columns
  (27B: 6 q heads per KV head → 2 rows per group). Each column's arithmetic equals the
  single-column case (`#pragma clang fp contract(off)` + explicit `__builtin_fmaf`), so a grouped
  verify row is bit-identical to a single-token decode row and MTP stays exact. Permanent check
  `checkAttnGroups` (random Q/K/V at positions 1000/1023).
- Maximum splits 128 → 64 (each split is now more efficient; combine at position 24000:
  34.7 → 17.4 µs). Attention kernel at position 24000: 298 → ~170 µs.

| Decode tok/s (MTP / plain) | before | after |
|---|---|---|
| 27B, 12k context | 55.3 / 31.3 | 64.5 / 33.1 |
| 27B, 24k | 45.5 / 29.8 | 61.5 / 31.9 |
| Ornith, 12k | 165 / 137 | 190 / 158 |
| Ornith, 24k | 137 / 129 | 185 / 150 |

- Lost: a 64-column variant (same speed; not bit-exact on gfx1151); a version that reads KV once
  for all queries (24k MTP 39.5 → 35.6 tok/s — the bottleneck was the reductions, not KV reads,
  and its 41 KB of LDS cut occupancy).
- **`attn_wsplit2` (up to 32 columns).** With 16 columns per group, a sequence's verify rows go
  through attention two at a time (27B: 6 GQA heads per KV head), so each split's K/V range is read
  once per 2 rows, while 3 of the block's 4 waves only help stage Vᵀ. `attn_wsplit2` is the same
  template with two column groups: waves 0 and 1 each compute 16 columns on the same staged Vᵀ tile
  (one K/V pass for up to 5 rows; LDS 37 KB, below the ~41 KB occupancy cliff). Each column runs the
  identical code path, so its partials are the same bits as alone (`checkAttnGroups` and the kernel
  test compare 5-row groups with per-query launches). The host groups rows with the 32-column limit
  and uses `attn_wsplit2` only when some group then holds more than 2 rows (single-row decode keeps
  `attn_wsplit1`; `WHIRL_ATTN_WIDE=0` turns it off). Kernel time per attention layer, q8v KV, R9700:

  | Context | Rows | `attn_wsplit1` | `attn_wsplit2` |
  |---|---|---|---|
  | 16k | 1 sequence × 5 (one user, 4 drafts) | 0.286 ms | 0.141 ms |
  | 16k | 1 × 9 (8 drafts) | 0.477 ms | 0.277 ms |
  | 16k | 4 × 4 (four users, 3 drafts each) | 0.853 ms | 0.607 ms |
  | 32k | 4 × 4 | 1.784 ms | 1.162 ms |
  | 32k | 1 × 9 | 1.115 ms | 0.633 ms |
- **gfx1151 limitation:** gfx11 WMMA is not exact when a P = 0 entry multiplies another query's
  real V row, so on the 8060S each group holds one query (the kernel still runs, without sharing
  K/V).
- After this work, plain decode time grows by about 0.11 ms per 1k tokens of context — exactly
  the time to read 64 MiB more KV per 1k tokens at ~600 GB/s. Decode attention is at bandwidth;
  the remaining lever is fewer KV bytes.
- **int8 K in `attn_wsplit1/2` (q8, q8h): magic-number dequantization (`kv_dq8`).** The first
  int8 version converted each element with its own `cvt` and multiply, which cost more than the
  halved K bytes saved (q8h decode +5.2% at 128k vs f16, [kv-and-caching.md](kv-and-caching.md#formats)).
  It now uses the same conversion as `attn_kg`: `v_perm` builds the f16 `0x6400 | (b ^ 0x80)`
  (= 1152 + q), 1152 is subtracted (exact) and the scale multiplied on packed f16 pairs, with
  `#pragma clang fp contract(off)` so the single rounding of `(_Float16)q * s` is kept — **same bits**.
  The row's 8 K scales are loaded once per key row. Swift-1.5 MXFP4-A, q8h, 125,853-token prompt,
  plain decode: 24.09 → 25.99 tok/s (+7.9%), same output hash. VGPR 189 → 191.

## <a id="deltanet"></a>9. Gated DeltaNet

### 9.1 Chunkwise prefill (f32, precision mode)

Token-by-token recurrence is far too slow for prefill. The chunked form: 64-token chunks, the WY
representation with a triangular solve inside each chunk, and a recurrent scan across chunks.
The first chunked version was slower than the sequential one; it became faster after register
tiling, a column-wise triangular solve and a pipelined scan. The state column `S[128]` spilled
until each column was split across 4 lanes with DPP reductions, and a lambda capturing arrays by
reference forced them into scratch (replaced with macros). The causal conv got its own parallel
kernel; state is handled separately.

Chunk boundaries count from the start of each prefill segment, and prefill segments start at
multiples of 1024, so solo and batched prefill see the same chunks.

### 9.2 f16 WMMA chunks (speed mode default)

- **Prep** (one block per chunk × v-head): WMMA computes K·Kᵀ and Q·Kᵀ as the 10 lower-triangular
  16×16 tiles; T = (I + B)⁻¹ is found by f32 forward substitution (4 lanes per column, DPP quad
  sums); T and M are written directly as f16 WMMA A fragments (10 KB per chunk and head, replacing
  80 KB of f32 W/U/M).
- **Scan** (one block per head, each wave owns 16 state columns): the state S (128×16) stays in
  registers as a C fragment, and **on gfx12 the C layout equals the B layout**, so K·S, Q·S, T·X,
  M·U and Kᵀ·U need no data movement of S. 116 WMMAs per wave per chunk.
- Probe (8192 tokens, one layer): 13.4 ms → 1.9–2.0 ms (6.7×). Error vs an f64 token-by-token
  recurrence: 7e-7 (f32 path) → 4.2e-4 (f16 inputs). On Q4_K_M with f16 prefill the
  last-token KL was 4.5e-8 … 1.5e-4 (the method is precise); under fp8 prefill it rose to
  1.2e-4 … 7.0e-2 because fp8 activations amplify any perturbation. Default for MXFP4 only; an
  opt-in "relaxed" flag enables it for Q4_K_M. 8k MXFP4 prefill 2212 → 2639 tok/s (+19.3%).
- Pitfalls: 700+ spilled VGPRs came from the compiler hoisting the 64-bit addresses of 32 V reads
  and 32 output writes out of the loop as loop invariants. Making the lane offset opaque each
  iteration (`asm volatile("" : "+v"(off))`) and splitting addresses into a uniform row base plus
  a 32-bit lane offset cut spills from 700 to 20. LDS rows are padded (17 / 9 chunks per row) to
  avoid bank conflicts. Prefetching the next chunk into registers (S 64 + X/O 64 + V 32 +
  prefetch 64 VGPRs) exceeded 256 and spilled — not used.

### 9.3 Decode step

`gdn_step_norm` runs the recurrent step, gated RMSNorm and int8 quantization in one kernel. Its
rewrite prefetches every row's q/k/v/decay/beta into LDS, leaves one barrier per row in the
dependency chain (double-buffered reduction: row t's output reduction is deferred past row t+1's
barrier), and does the norm + quantization once at the end (one wave per row, same addition order
as before). Bit-identical — after an asm barrier, because the first version let the compiler
contract `y*y` into the shuffle-add fma. Verify −0.2…−1%.

How the recurrent state is rolled back when drafts are rejected (snapshots, then replay) is in
[speculative-decoding.md](speculative-decoding.md#replay).

## <a id="moe"></a>10. Mixture-of-experts (`qwen35moe`)

- **Router:** f32 softmax → top-8 of 256 → renormalize to sum 1 (semantics as llama.cpp's
  `build_moe_ffn`); the shared expert is scaled by `sigmoid(gate_inp_shexp · h)`.
- **Decode:** gate and up of the selected experts in one int8 GEMV reading only the 8 selected
  experts' rows; down-projection, weighted combine, shared expert and residual in one kernel; no
  host synchronization.
- **Prefill:** (token, slot) pairs are grouped by expert on the GPU, gathered to f16, run through
  a grouped WMMA GEMM and scattered back. Token tile 32 when experts average fewer than 48 tokens,
  else 64. 1.1k-token prefill ~3300 → ~5000 tok/s.
- The fused DeltaNet beta/alpha kernel was generalized because Ornith stores beta/alpha as Q4_K
  (27B: Q8_0); without it 3-draft verification could not run. The MTP layer's FFN is MoE too.
- **Routing near-ties are everywhere.** Between any two equally valid prefill paths (WMMA fast,
  scalar naive attention, sequential DeltaNet), 17–28% of (token, layer) pairs select a different
  expert set, starting in layers 0–4; last-token KL ranged 1.2e-4 … 3.4e-2, and top-1 agreed in
  all 45 pairs. The dense 27B under the same checks: KL ~1e-7. MoE KL thresholds must therefore be
  set from measured path noise (we use 1e-2 for the MoE long-context check, 1e-3 for dense).

## <a id="resources"></a>11. Register and resource discipline

Every kernel above went through the same loop: build with resource-usage output, read VGPR /
scratch / spill, fix, re-measure. Techniques that worked:

| Problem | Technique |
|---|---|
| wave-uniform values held per lane, scalar branches turned vector | `__builtin_amdgcn_readfirstlane` to make indices uniform |
| compiler interleaves independent accumulation chains (4 units' WMMAs) and blows registers | pin each accumulator after its WMMA with `asm volatile("" : "+v"(acc))` |
| scheduler preloads all 16 V fragments (256 VGPRs + 13 spilled, 56 B scratch) | same pin after each dt WMMA → 253 VGPRs, 0 spill (+1.1% prefill on 8060S) |
| loop-invariant 64-bit addresses hoisted out of a loop (700+ spills) | opaque per-iteration lane offset; uniform base + 32-bit offset |
| HIP `int4` (a struct) arrays forced into scratch | `ext_vector_type` vectors |
| device lambdas capturing arrays by reference forced arrays into scratch | macros |
| an implementation with two callers stopped inlining (248 VGPRs + scratch) | `__forceinline__` |
| a weight pointer not coming directly from a kernel argument changed codegen (+30–90% VGPRs) | separate entries per call shape |
| `break` in a pipelined loop zeroed the wait counter each step | no-`break` main loop + separate tail |
| wave-divergent scale branches serialized memory latency | branch-free index arithmetic |
| `v_mad_i32_i24` not selected (compiler used `mul_lo_u32` or `mul24 + add3`) | inline asm |
| LDS bank conflicts | fragment-order layouts, padding |
| tile configurations that spill | compile all candidates, keep only spill-free ones |

Always confirm with the code object's metadata that a refactor kept VGPR/SGPR/scratch counts
(see [windows-hip.md](windows-hip.md#build) on matching build flags).

## <a id="fma"></a>12. Floating-point contraction and bit-exactness

Fusing or refactoring a kernel can change results without changing the math, because clang is
free to contract `a*b + c` into `fma` and to choose a different contraction when code moves.
Cases we hit:

| Change | What the compiler did | Damage |
|---|---|---|
| conv `h0*w0 + h1*w1 + h2*w2 + x*w3` moved into an inline function during fusion | chose a different fma contraction order | last-token logits differed by up to 0.17 |
| RMSNorm fused into GEMM input producers | contracted `x*x` into the first butterfly add; merged a multiply and the f16 conversion into `v_fma_mix` | 1 ulp per element |
| recurrent-step kernel rewrite | contracted `y*y` into the shuffle-add fma | not bit-identical |

Rules we follow:

1. In kernels whose bits matter, write the canonical expression as an explicit
   `__builtin_fmaf` chain (equal to what the reference kernel compiled to), and/or use
   `#pragma clang fp contract(off)`.
2. Put `asm volatile("" : "+v"(v))` after squares and final products to stop contraction across
   the boundary.
3. Reproduce reductions with the same lane permutation (`block_sum`, xor tree) rather than a
   "mathematically equal" one.
4. Integer WMMA is exact; f16 WMMA is exact only when inputs and accumulation order match — and on
   gfx11 not even then in one case (P = 0 times another query's V, §8).
5. Multiplying by exactly 1 (lazy rescale) and changing tile shapes in M/N (GEMM buckets) are safe;
   changing the K order is not.
6. Verify end to end: last-token logits compared bit for bit across binaries for several prompts,
   plus the permanent per-kernel checks.

The full list of kernel and numerics pitfalls is in [pitfalls.md](pitfalls.md#kern).
