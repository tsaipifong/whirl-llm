# GEMVH-1151: gfx1151 precise-decode kernel `gemvh` / `gemvh2` (8060S)

Branch `gemvh1151` (from kg3 2a20136), worktree `wt_gemvh1151`. Kernel: `kernels/gfx1151/gemv_h.hip` (draft integrated unchanged), include added to `whirl_kernels_gfx1151.hip`. The host (`Model::gemmF16`) already used `gemvh`/`gemvh2` when exported; `WHIRL_PREC_GEMVH=0` disables them. No kernel bug was found: the draft compiled and was bitwise on the first run. gfx1151-only build; R9700 not retested (the gfx1201 kernel set is untouched).

## 1. Bitwise (whirl-kernel-test --device 8060s --only gemvh, tests/kernels/test_gemvh.cpp)
gemvh (n<=16) and gemvh2 (n<=32) == gemm_c0 and every gemms_c0..c7, with accumulate 0 and 1 (random prefill), n_tok {1,2,5,16,17,32}, ncols {256,512,768,1280,5120} (768/1280 = odd stage count, the S=1 fallback), nrows {1,7,16,17,37,100,1003,4099} (not multiples of 16). 0 mismatches for all 10 exported types: f16 (800 runs), q8_0 (3680), q3_k, q4_k, q5_k, q6_k, iq4_nl, iq3_s, iq4_xs, mxfp4 (4640 each). The existing `gemm`/`gemv` families also pass (they cover gemvh on flush-end allocations and vs a CPU double product); CPU unit tests 1817/0.

## 2. Microbench (--only gemvhbench; weights rotated through >=192 MB of copies so the MALL does not help; peak = 238 GB/s hipMalloc-measured, audit/KGAP-1.md)
us per call, GB/s of weight bytes, % of 238:

| shape | n | gemvh | gemms_c0 (best old kernel) |
|---|---|---|---|
| q4_k 5120x5120 | 1 | 101 us, 146, 61% | 110 us, 135, 57% |
| q4_k 5120x5120 | 16 | 115 us, 128, 54% | 115 us, 129, 54% |
| q4_k 17408x5120 | 1 | 351 us, 143, 60% | 367 us, 137, 57% |
| q4_k 17408x5120 | 16 | 371 us, 135, 57% | 376 us, 133, 56% |
| q6_k 5120x17408 | 1 | 539 us, 136, 57% | 554 us, 132, 56% |
| q6_k 5120x17408 | 16 | 564 us, 130, 54% | 595 us, 123, 52% |
| q6_k 5120x5120 | 1 | 174 us, 124, 52% | 166 us, 130, 55% |
| q6_k 5120x5120 | 16 | 195 us, 110, 46% | 172 us, 125, 53% |
| q5_k 5120x5120 | 1 | 124 us, 146, 61% | 150 us, 120, 50% |
| q8_0 5120x5120 | 1 | 178 us, 156, 66% | 152 us, 183, 77% |
| q6_k head 248320x5120 | 1 | 7474 us, 140, 59% | 7658 us, 136, 57% |
| q6_k head 248320x5120 | 16 | 7163 us, 146, 61% | 7457 us, 140, 59% |

Observation: gemvh reaches only 46-66% of peak (balance-mode gemvq reaches ~84%), and it is only 0-8% faster than the best gemms config in isolation; it is slower than gemms_c0 for q6_k 5120x5120 n=16 (-13%) and q8_0 (-15%/-8%). Full table: gemvh1151/bench.txt. Room left: a 16-row wave per block with a single LDS round per stage keeps few bytes in flight; the 84% kernels use wider units.

## 3. End to end (Qwen3.8-27B-UD-Q4_K_M, --precise, 8060S, WHIRL_PREC_GEMVH=1 vs 0, runs interleaved 1,0,0,1, two runs each)
| case | GEMVH=1 | GEMVH=0 | change |
|---|---|---|---|
| short (144-tok prompt) plain decode | 8.34 / 8.28 tok/s | 5.34 / 5.34 | +55.6% |
| short MTP+n-gram | 29.75 / 29.58 | 20.16 / 20.28 | +46.7% |
| 32k (33283 tok) plain decode | 7.45 / 7.12 | 4.97 / 4.83 | +48.8% |
| 32k MTP | 20.22 / 19.10 | 15.41 / 14.78 | +30.2% |
32k prefill unchanged (340/328 vs 340/338 tok/s). Outputs identical: short bench hash 878b50b55fad0502 in all 4 runs and both modes; 32k reply text identical between GEMVH=1 and 0 (plain: 2f3187e7 all four, MTP: 2cb56f0b all four).
The fallback (precChoice, not gemms_c0) is much slower than the isolated gemms_c0 numbers suggest (~190 ms/token vs gemvh ~120 ms/token), hence the large e2e gain despite the small kernel-level gap: the old path picks other (dequant + gemm_c, or slower gemms) configs per matrix.

## Next
- Re-tune the gemvh body for >=75% of peak (more bytes in flight, S=2 for q6_k/q8_0, wider unit as gemvq) - the e2e ceiling is ~1.3-1.5x above today's gemvh.
- Check precChoice on 8060S: it should at least choose gemms_c0 for n<=16 (free fallback gain) and for q8_0/q6_k 5120x5120 gemms_c0 is faster than gemvh (per-type dispatch).
