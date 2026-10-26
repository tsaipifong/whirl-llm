// feat026-moep (KERNEL_TECHNIQUE_AUDIT gap #2): Ornith MoE prefill expert GEMMs for the Radeon 8060S (gfx1151) where
// one block takes up to BNMAX tokens of ONE expert (several 16-token "slot tiles"), so every A fragment decoded from
// MXFP4 into LDS is shared by all of that expert's fragments in the tile (hipfire levers.md "several tiles per wave
// when the row operand repeats"; llama.cpp mmq J up to 128 on RDNA3.5); and a down GEMM whose epilogue also does the
// weighted combine (moe_combine) for a token once its last expert has been written (deterministic last-arriver
// counter; fixed k order, no atomics on data).
//
// Bit-identity (the reason it may replace gemm_moegu_<T> / gemm_moe32r_<T> / moe_combine in every mode):
//   * every stored f32 sum is the gemm_moe_<T> chain: acc = 0; for kk = 0, 16, ... < ncols: acc = WMMA16(a, b, acc),
//     with the same dec_group_h<T> A fragments and the same f16 X rows; a WMMA output element depends only on its
//     own A row and B column, so tile shape, BK and the wave layout are free (8S-P4 relies on the same fact);
//   * gate + up: A = (_Float16)(g / (1.f + __expf(-g)) * u), the moe_act_f16 / gemm_moegu expression;
//   * fused combine: the moe_combine expression and k order verbatim on the stored yd values.
// Proof obligation on the GPU: tests/kernels/test_moep.cpp compares every routed position with cmpExact.
//
// Modes (owner precision policy, whirl-hillclimb): precise = strictly lossless, bit-identical, golden identical;
// balance = near-lossless by industry standard, defined by its allowed items (here: f16 intermediates), acceptance =
// task accuracy (KL / top-1 only as diagnostics / regression guards); fast = within 4-bit-level quality, acceptance =
// task-accuracy recovery >= 95% of precise.
//   bit-identical, every mode: gemm_moepgu_*, gemm_moepdn_*, gemm_moepdc_* (fused combine), moe_combine4.
//   balance (item "f16 intermediates"; never chosen in precise): gemm_moepdh_* / gemm_moepdhc_* store the expert
//     outputs yd as f16 (one f16 rounding per expert output, deterministic, half the yd bytes), read by moe_combine4h /
//     the fused combine.
//   fast candidate (not a listed balance item): gemm_moepda_* adds w * y straight into x with f32 atomics (no yd; the
//     f32 summation order varies run to run), after moe_combine4 with K = 0 added the shared expert.
//
// Index maps: kernels/moep_map.h (shared with the CPU schedule test). Tile lists: moe_tiles with bn = BNMAX
// (positions = moe_route's perm whatever the tile size). Grid: 1-D, max_tiles(BNMAX) * nrb, row block fastest.
// SPDX-License-Identifier: Apache-2.0

#include "../moep_map.h"

// ---------------------------------------------------------------- gate + up + SwiGLU
// Block: 8 waves, GR gate rows (W1) + the same GR up rows (W2) of one expert = a 2 GR-row A slab, x up to BNMAX tokens.
// Wave (wm, wn): row fragment wm (gate rows wm*16.., up rows GR + wm*16..), token fragments wn, wn + WN, ...
template <int T, int BK, int GR, int BNMAX>
__device__ void moepgu_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff,
                            const _Float16 *__restrict__ Xall, _Float16 *__restrict__ Aout, int ncols,
                            const int4 *__restrict__ tiles, const int *__restrict__ n_tiles) {
    constexpr int NTH = 256, NW = NTH / 32;
    constexpr int WM = GR / 16, WN = NW / WM;
    constexpr int NF = BNMAX / 16, FPW = NF / WN;
    constexpr int BM = 2 * GR, LD = BK + 8, GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH, B_PER = BNMAX * GPR / NTH;
    static_assert(GR % 16 == 0 && NW % WM == 0 && NF % WN == 0 && FPW >= 1, "wave layout");
    static_assert(BM * GPR % NTH == 0 && BNMAX * GPR % NTH == 0 && BK % 16 == 0, "tile/threads");
    const int nrb = moep::nrb(ff, GR);
    const int ti = moep::blockTile((int)blockIdx.x, nrb), rb = moep::blockRb((int)blockIdx.x, nrb);
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    const int n_tok = tl.z, m0 = rb * GR;
    const int live = moep::liveFrags(n_tok), live16 = live * 16;
    const uint8_t *Wg = W1 + (size_t)tl.x * ff * row_bytes, *Wu = W2 + (size_t)tl.x * ff * row_bytes;
    const _Float16 *X = Xall + (size_t)tl.y * ncols;
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BNMAX][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WM, wn = wid / WM;
    float8 acc[2][FPW];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < FPW; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rbx[B_PER];
#define MOEPGU_FETCH(k0_)                                                                           \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            const int rr = m0 + (r % GR);                                                           \
            if (rr < ff) {                                                                          \
                ra[e] = dec_group_h<T>((r < GR ? Wg : Wu) + (size_t)rr * row_bytes, (k0_) / 8 + gg); \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t < n_tok) {                                                                        \
                rbx[e] = *(const half8 *)(X + (size_t)t * ncols + (k0_) + gg * 8);                  \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rbx[e][q] = (_Float16)0.f;             \
            }                                                                                       \
        }                                                                                           \
    } while (0)
    MOEPGU_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            // rows past the last live fragment are never read: no LDS store for them
            if (i / GPR < live16) *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rbx[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) MOEPGU_FETCH(k0 + BK);
        if (wn >= live) continue;  // this wave column has no live fragment in this tile
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const half16 a0 = ld_frag16(&As[wm * 16 + lane % 16][kk]);
            const half16 a1 = ld_frag16(&As[GR + wm * 16 + lane % 16][kk]);
#pragma unroll
            for (int i = 0; i < FPW; ++i) {
                const int j = moep::frag(wn, WN, i);
                if (j < live) {
                    const half16 b = ld_frag16(&Bs[j * 16 + lane % 16][kk]);
                    acc[0][i] = WMMA16(a0, b, acc[0][i]);
                    acc[1][i] = WMMA16(a1, b, acc[1][i]);
                }
            }
        }
    }
#undef MOEPGU_FETCH
#pragma unroll
    for (int i = 0; i < FPW; ++i) {
        const int j = moep::frag(wn, WN, i);
        const int t = j * 16 + lane % 16;
        if (j >= live || t >= n_tok) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = m0 + wm * 16 + CROW(lane, v);
            if (row < ff) {
                const float g = acc[0][i][v], u = acc[1][i][v];
                Aout[(size_t)(tl.y + t) * ff + row] = (_Float16)(g / (1.f + __expf(-g)) * u);
            }
        }
    }
}

// ---------------------------------------------------------------- down (+ optional fused combine)
// Block: 8 waves as WM (rows) x WN (token columns); wave (wm, wn) holds TM = BM / 16 / WM row fragments and token
// fragments wn, wn + WN, ... of one expert's tile of up to BNMAX tokens. Y: f32 [pos][rows_e] exactly as gemm_moe32r.
// EPI: 0 = Y f32 (gemm_moepdn), 1 = Y f32 + fused combine (gemm_moepdc), 2 = Y f16 (gemm_moepdh), 3 = Y f16 + fused
// combine (gemm_moepdhc), 4 = x += w * y by f32 atomics, no Y (gemm_moepda).
// Fused combine (EPI 1, 3): after Y is stored, the block counts each of its tokens on cnt[tok][rb]; the block that brings a counter to
// K (all K experts of that token written for this row block) computes x[tok][rows] += sum_k w * yd + sg * ysh in
// moe_combine's order and resets the counter to 0 (cnt must start zeroed once; it is left zeroed).
template <int T, int BK, int BM, int WM, int BNMAX, int EPI>
__device__ void moepdn_impl(const uint8_t *__restrict__ Wall, uint64_t row_bytes, int rows_e, const _Float16 *__restrict__ Xall,
                            void *__restrict__ Yv, int ncols, const int4 *__restrict__ tiles, const int *__restrict__ n_tiles,
                            const int *__restrict__ perm, const int *__restrict__ inv, const float *__restrict__ w,
                            const float *__restrict__ sg, const float *__restrict__ ysh, float *__restrict__ x,
                            int *__restrict__ cnt, int K) {
    constexpr int NTH = 256, NW = NTH / 32;
    constexpr int WN = NW / WM, TM = BM / 16 / WM;
    constexpr int NF = BNMAX / 16, FPW = NF / WN;
    constexpr int LD = BK + 8, GPR = BK / 8;
    constexpr int A_PER = BM * GPR / NTH, B_PER = BNMAX * GPR / NTH;
    static_assert(NW % WM == 0 && BM % (16 * WM) == 0 && TM >= 1 && NF % WN == 0 && FPW >= 1, "wave layout");
    static_assert(BM * GPR % NTH == 0 && BNMAX * GPR % NTH == 0 && BK % 16 == 0, "tile/threads");
    static_assert(EPI >= 0 && EPI <= 4, "epilogue");
    constexpr bool FUSE = EPI == 1 || EPI == 3, YH = EPI == 2 || EPI == 3;
    float *__restrict__ Y = (float *)Yv;
    _Float16 *__restrict__ Yh = (_Float16 *)Yv;
    const int nrb = moep::nrb(rows_e, BM);
    const int ti = moep::blockTile((int)blockIdx.x, nrb), rb = moep::blockRb((int)blockIdx.x, nrb);
    if (ti >= *n_tiles) return;
    const int4 tl = tiles[ti];
    const int n_tok = tl.z, m0 = rb * BM;
    const int live = moep::liveFrags(n_tok), live16 = live * 16;
    const uint8_t *W = Wall + (size_t)tl.x * rows_e * row_bytes;
    const _Float16 *X = Xall + (size_t)tl.y * ncols;
    __shared__ _Float16 As[BM][LD];
    __shared__ _Float16 Bs[BNMAX][LD];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WM, wn = wid / WM;
    float8 acc[TM][FPW];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < FPW; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    half8 ra[A_PER], rbx[B_PER];
#define MOEPDN_FETCH(k0_)                                                                           \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int r = i / GPR, gg = i % GPR;                                                    \
            if (m0 + r < rows_e) {                                                                  \
                ra[e] = dec_group_h<T>(W + (size_t)(m0 + r) * row_bytes, (k0_) / 8 + gg);            \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) ra[e][q] = (_Float16)0.f;              \
            }                                                                                       \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                         \
            const int i = threadIdx.x + e * NTH;                                                    \
            const int t = i / GPR, gg = i % GPR;                                                    \
            if (t < n_tok) {                                                                        \
                rbx[e] = *(const half8 *)(X + (size_t)t * ncols + (k0_) + gg * 8);                  \
            } else {                                                                                \
                _Pragma("unroll") for (int q = 0; q < 8; ++q) rbx[e][q] = (_Float16)0.f;             \
            }                                                                                       \
        }                                                                                           \
    } while (0)
    MOEPDN_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(half8 *)&As[i / GPR][(i % GPR) * 8] = ra[e];
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            if (i / GPR < live16) *(half8 *)&Bs[i / GPR][(i % GPR) * 8] = rbx[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) MOEPDN_FETCH(k0 + BK);
        if (wn >= live) continue;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half16 a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = ld_frag16(&As[(wm * TM + i) * 16 + lane % 16][kk]);
#pragma unroll
            for (int f = 0; f < FPW; ++f) {
                const int j = moep::frag(wn, WN, f);
                if (j < live) {
                    const half16 b = ld_frag16(&Bs[j * 16 + lane % 16][kk]);
#pragma unroll
                    for (int i = 0; i < TM; ++i) acc[i][f] = WMMA16(a[i], b, acc[i][f]);
                }
            }
        }
    }
#undef MOEPDN_FETCH
#pragma unroll
    for (int f = 0; f < FPW; ++f) {
        const int j = moep::frag(wn, WN, f);
        const int t = j * 16 + lane % 16;
        if (j >= live || t >= n_tok) continue;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            if constexpr (EPI == 4) {
                const int p = perm[tl.y + t], tok = p / K;
                const float wp = w[p];
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                    // unsafeAtomicAdd: the native global_atomic_add_f32 (atomicAdd compiles to a CAS loop on gfx11)
                    if (row < rows_e) unsafeAtomicAdd(&x[(size_t)tok * rows_e + row], wp * acc[i][f][v]);
                }
            } else {
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int row = m0 + (wm * TM + i) * 16 + CROW(lane, v);
                    if (row < rows_e) {
                        if constexpr (YH) Yh[(size_t)(tl.y + t) * rows_e + row] = (_Float16)acc[i][f][v];
                        else Y[(size_t)(tl.y + t) * rows_e + row] = acc[i][f][v];
                    }
                }
            }
        }
    }
    if constexpr (FUSE) {
        __shared__ int fin[BNMAX];
        __shared__ int nfin;
        if (threadIdx.x == 0) nfin = 0;
        __threadfence();  // release: this block's Y stores before its counter increments
        __syncthreads();
        for (int t = threadIdx.x; t < n_tok; t += NTH) {
            const int tok = perm[tl.y + t] / K;
            int *c = cnt + moep::cntSlot(tok, rb, nrb);
            if (atomicAdd(c, 1) == K - 1) {
                *c = 0;  // every increment of this (token, row block) is done: leave the counter zeroed
                fin[atomicAdd(&nfin, 1)] = tok;
            }
        }
        __syncthreads();
        const int nf = nfin;
        if (nf == 0) return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // the other blocks' yd rows of these tokens
        for (int q = threadIdx.x; q < nf * BM; q += NTH) {
            const int tok = fin[q / BM], cc = m0 + q % BM;
            if (cc >= rows_e) continue;
            // moe_combine, verbatim: x[t][c] += sum_k w[t,k] * yd[inv[t*K+k]][c] (+ sg[t] * ysh[t][c])
            const int t = tok, C = rows_e, c = cc;
            float acc_c = 0.f;
            for (int k = 0; k < K; ++k) {
                const size_t yi = (size_t)inv[(size_t)t * K + k] * C + c;
                if constexpr (YH) acc_c += w[(size_t)t * K + k] * (float)Yh[yi];
                else acc_c += w[(size_t)t * K + k] * Y[yi];
            }
            if (ysh != nullptr) acc_c += sg[t] * ysh[(size_t)t * C + c];
            x[(size_t)t * C + c] += acc_c;
        }
    }
}

// moe_combine with 4 consecutive columns per thread (C % 4 == 0); per element the moe_combine expression and order.
// grid (C / 1024, n).
extern "C" __global__ void __launch_bounds__(256)
    moe_combine4(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg, const float *__restrict__ yd,
                 const int *__restrict__ inv, const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= C) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const float wk = w[(size_t)t * K + k];
        const float4 v = *(const float4 *)(yd + (size_t)inv[(size_t)t * K + k] * C + c);
        acc.x += wk * v.x;
        acc.y += wk * v.y;
        acc.z += wk * v.z;
        acc.w += wk * v.w;
    }
    if (ysh != nullptr) {
        const float s = sg[t];
        const float4 v = *(const float4 *)(ysh + (size_t)t * C + c);
        acc.x += s * v.x;
        acc.y += s * v.y;
        acc.z += s * v.z;
        acc.w += s * v.w;
    }
    float4 *xo = (float4 *)(x + (size_t)t * C + c);
    float4 xv = *xo;
    xv.x += acc.x;
    xv.y += acc.y;
    xv.z += acc.z;
    xv.w += acc.w;
    *xo = xv;
}

// moe_combine4 over f16 expert outputs (balance / fast candidate with gemm_moepdh_*): yd f16 [pos][C].
extern "C" __global__ void __launch_bounds__(256)
    moe_combine4h(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg, const _Float16 *__restrict__ yd,
                  const int *__restrict__ inv, const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= C) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const float wk = w[(size_t)t * K + k];
        const _Float16 *v = yd + (size_t)inv[(size_t)t * K + k] * C + c;
        acc.x += wk * (float)v[0];
        acc.y += wk * (float)v[1];
        acc.z += wk * (float)v[2];
        acc.w += wk * (float)v[3];
    }
    if (ysh != nullptr) {
        const float s = sg[t];
        const float4 v = *(const float4 *)(ysh + (size_t)t * C + c);
        acc.x += s * v.x;
        acc.y += s * v.y;
        acc.z += s * v.z;
        acc.w += s * v.w;
    }
    float4 *xo = (float4 *)(x + (size_t)t * C + c);
    float4 xv = *xo;
    xv.x += acc.x;
    xv.y += acc.y;
    xv.z += acc.z;
    xv.w += acc.w;
    *xo = xv;
}

// Variants for the hill-climb (names = include/whirl/kernels_abi.h kMoepGu / kMoepDn; keep both lists in step).
// gate + up: g<GR>n<BNMAX>k<BK>; down: m<BM>w<WM>n<BNMAX>k<BK>; fused combine twin: gemm_moepdc_*.
#define MOEPGU_ENTRY(NAME, GR, BN, BK)                                                                             \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepgu_##NAME##_mxfp4(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, const _Float16 *X, \
                                   _Float16 *A, int ncols, const int4 *tiles, const int *n_tiles) {                \
        moepgu_impl<QT_MXFP4, BK, GR, BN>(W1, W2, row_bytes, ff, X, A, ncols, tiles, n_tiles);                     \
    }
#define MOEPDN_ENTRY(NAME, BM, WM, BN, BK)                                                                         \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepdn_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,  \
                                   int ncols, const int4 *tiles, const int *n_tiles) {                             \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 0>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr,       \
                                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);        \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepdh_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,            \
                                   _Float16 *Y, int ncols, const int4 *tiles, const int *n_tiles) {                \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 2>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, nullptr,       \
                                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);        \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepdhc_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,           \
                                    _Float16 *Y, int ncols, const int4 *tiles, const int *n_tiles,                 \
                                    const int *perm, const int *inv, const float *w, const float *sg,              \
                                    const float *ysh, float *x, int *cnt, int K) {                                 \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 3>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, perm, inv, w,  \
                                                  sg, ysh, x, cnt, K);                                             \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepda_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X,            \
                                   int ncols, const int4 *tiles, const int *n_tiles, const int *perm,              \
                                   const float *w, float *x, int K) {                                              \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 4>(W, row_bytes, rows_e, X, nullptr, ncols, tiles, n_tiles, perm,    \
                                                  nullptr, w, nullptr, nullptr, x, nullptr, K);                    \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        gemm_moepdc_##NAME##_mxfp4(const uint8_t *W, uint64_t row_bytes, int rows_e, const _Float16 *X, float *Y,  \
                                   int ncols, const int4 *tiles, const int *n_tiles, const int *perm,              \
                                   const int *inv, const float *w, const float *sg, const float *ysh, float *x,    \
                                   int *cnt, int K) {                                                              \
        moepdn_impl<QT_MXFP4, BK, BM, WM, BN, 1>(W, row_bytes, rows_e, X, Y, ncols, tiles, n_tiles, perm, inv,  \
                                                     w, sg, ysh, x, cnt, K);                                       \
    }
MOEPGU_ENTRY(g64n64k64, 64, 64, 64)    // control: the gemm_moegu tile (64 gate + 64 up rows, 64 tokens)
MOEPGU_ENTRY(g64n128k64, 64, 128, 64)  // 2 slot tiles of 64 per block, 4 fragments per wave
MOEPGU_ENTRY(g64n128k32, 64, 128, 32)
MOEPGU_ENTRY(g32n128k64, 32, 128, 64)  // 32 + 32 rows, 4 wave columns x 2 fragments
MOEPGU_ENTRY(g64n256k32, 64, 256, 32)  // 8 fragments per wave
MOEPGU_ENTRY(g128n128k32, 128, 128, 32)  // 8 row fragments, one wave column x 8 fragments
MOEPDN_ENTRY(m128w4n32k64, 128, 4, 32, 64)   // control: the gemm_moe32r tile
MOEPDN_ENTRY(m128w4n64k64, 128, 4, 64, 64)
MOEPDN_ENTRY(m128w4n128k64, 128, 4, 128, 64)
MOEPDN_ENTRY(m128w8n128k64, 128, 8, 128, 64)
MOEPDN_ENTRY(m64w4n128k64, 64, 4, 128, 64)
MOEPDN_ENTRY(m128w4n256k32, 128, 4, 256, 32)
#undef MOEPGU_ENTRY
#undef MOEPDN_ENTRY
