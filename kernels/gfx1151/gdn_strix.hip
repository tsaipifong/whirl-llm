// WHIRL device code (gfx1151 / RDNA 3.5): chunked Gated DeltaNet prefill on f16 WMMA (GDN-8S, opt-in).
// SPDX-License-Identifier: MIT
//
// Adapted from strix-llama.cpp (rdna35-perf branch, commit 62e9d99),
// ggml/src/ggml-cuda/gated_delta_net.cu: gdn_chunk_prep_kernel / gdn_chunk_scan_kernel
// (https://github.com/asf0/strix-llama.cpp). Original code:
//   MIT License, Copyright (c) 2023-2026 The ggml authors.
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//   associated documentation files (the "Software"), to deal in the Software without restriction,
//   including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
//   subject to the following conditions: The above copyright notice and this permission notice shall be
//   included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//
// Changes for WHIRL (GDN-8S): inputs read straight from the conv output rows conv[t][ch] (q | k | v,
// q / k already L2-normalized by the conv kernels), g / beta as [t][H]; the conv, q/k-norm and gated
// output-norm fusions are dropped; CL fixed at 32 (strix's launch choice); the recurrent state uses
// WHIRL's layout state[h][key i][value j] (strix / ggml: [h][j][i]) and is updated in place; the
// half-wave swap is selectable (GDN_SX_SWAP: 0 = permlanex16 fi=true as strix, 1 = ds_swizzle xor 16)
// because permlanex16(fi=false) miscompiles on HIP SDK 7.2 (KLIB-1); gdn_sx_swap_probe checks both.
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip.
//
// Per chunk of CL tokens (G = in-chunk cumsum of g, D[r][s] = exp(G_r - G_s)):
//   A[r][s] = beta_r (k_r.k_s) D[r][s] (s < r),   T' = (I + A)^-1 diag(beta)
//   U  = T' (V - diag(e^G) K S0)
//   O  = scale diag(e^G) Q S0 + P U,   P[r][s] = scale (q_r.k_s) D[r][s] (s <= r)
//   S1 = e^{G_L} S0 + K^T diag(e^{G_L - G}) U
// WMMA operands are f16 with f32 accumulation: not bit-identical to gdn_chunk_scan / gdn_seq_128
// (the same precision class as the gfx1201 gdn_wprep / gdn_wscan8, numerics item gdnwmma).

#ifndef GDN_SX_SWAP
#define GDN_SX_SWAP 0
#endif
#define GDN_SX_CL 32

typedef _Float16 gdnc_v16h __attribute__((ext_vector_type(16)));
typedef float gdnc_v8f __attribute__((ext_vector_type(8)));

#define GDNC_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))

// value of lane ^ 16 (the other half-wave)
static __device__ __forceinline__ float gdnc_swap16(const float own) {
#if GDN_SX_SWAP == 0
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(own), __float_as_int(own), 0x76543210, 0xFEDCBA98, true, false));
#else
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(own), 0x401F));
#endif
}

// A/B operand fragment from a row-major fp16 matrix: lane holds row (lane & 15), 16 consecutive k
static __device__ __forceinline__ gdnc_v16h gdnc_frag(const _Float16 *base, const int stride, const int row0, const int k0, const int lane) {
    const _Float16 *p = base + (row0 + (lane & 15)) * stride + k0;
    gdnc_v16h f;
    const uint4 a = *(const uint4 *)p, b = *(const uint4 *)(p + 8);
    __builtin_memcpy(&f, &a, 16);
    __builtin_memcpy(((char *)&f) + 16, &b, 16);
    return f;
}
// accumulator tile (lane: column lane&15, rows 2e + (lane>>4)) -> B fragment (lane: column lane&15, rows 0..15)
static __device__ __forceinline__ gdnc_v16h gdnc_acc_to_frag(const gdnc_v8f &acc, const int lane) {
    gdnc_v16h f;
    const bool hi = lane >= 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float own = acc[e];
        const float oth = gdnc_swap16(own);
        f[2 * e] = (_Float16)(hi ? oth : own);
        f[2 * e + 1] = (_Float16)(hi ? own : oth);
    }
    return f;
}
static __device__ __forceinline__ uint2 gdnc_pack4h(const float4 v) {
    const _Float16 h[4] = {(_Float16)v.x, (_Float16)v.y, (_Float16)v.z, (_Float16)v.w};
    uint2 u;
    __builtin_memcpy(&u, h, 8);
    return u;
}

// lane l of out[0..31] = swap16 of in[l] (variant under test), out[32..63] = __shfl_xor(in, 16) (reference)
extern "C" __global__ void __launch_bounds__(32) gdn_sx_swap_probe(const float *in, float *out) {
    const int l = threadIdx.x;
    const float v = in[l];
    out[l] = gdnc_swap16(v);
    out[32 + l] = __shfl_xor(v, 16, 32);
}

// K1: one block per (k-head, chunk), handling the H/Hk (<= 4) value heads that share it. Produces the chunk's q/k
// and v as fp16 rows (Qh/Kh [Hk, T, 128], Vh [H, T, 128]) for the scan, plus T', P and the gate factors.
// conv: [n_tokens][ch] = q heads | k heads | v heads (128 each); g, beta: [n_tokens][H].
extern "C" __global__ void __launch_bounds__(256)
    gdn_sx_prep(const float *__restrict__ conv, const float *__restrict__ g, const float *__restrict__ beta, _Float16 *__restrict__ Tb,
                _Float16 *__restrict__ Pb, float2 *__restrict__ Gb, _Float16 *__restrict__ Qh, _Float16 *__restrict__ Kh,
                _Float16 *__restrict__ Vh, const int H, const int Hk, const int n_tokens, const int nch, const int ch_stride,
                const float scale) {
    constexpr int CL = GDN_SX_CL;
    constexpr int SK = 136;
    constexpr int NT = CL / 16;
    constexpr int NTRI = NT * (NT + 1) / 2;
    constexpr int MR = CL / 32;
    constexpr int CPW = CL / 8;
    constexpr int NQK = CL * 32 / 256;
    constexpr int RMAX = 4;
    __shared__ __align__(16) _Float16 Ks[CL * SK];
    __shared__ __align__(16) _Float16 Qs[CL * SK];
    __shared__ float KK[CL][CL + 1];
    __shared__ float QK[CL][CL + 1];
    __shared__ float A[CL][CL + 1];
    __shared__ float Gs[RMAX][CL], bs[RMAX][CL];
    const int hk = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int cl = lane & 15, rh = lane >> 4;
    const int R = H / Hk;
    const int t0 = ch * CL, Lc = min(CL, n_tokens - t0);
    const int i4 = (tid & 31) * 4;  // this thread's 4 channels of every row it handles (row = (idx >> 5))
    const float *qb = conv, *kb = conv + (size_t)Hk * 128, *vb = conv + (size_t)2 * Hk * 128;
    {
        float4 kv[NQK], qv[NQK];
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int r = (tid + 256 * j) >> 5;
            const int64_t o = (int64_t)(t0 + min(r, Lc - 1)) * ch_stride + (int64_t)hk * 128 + i4;
            kv[j] = *(const float4 *)(kb + o);
            qv[j] = *(const float4 *)(qb + o);
        }
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int r = (tid + 256 * j) >> 5;
            const bool ok = r < Lc;
            const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const uint2 khv = gdnc_pack4h(ok ? kv[j] : z4), qhv = gdnc_pack4h(ok ? qv[j] : z4);
            *(uint2 *)(Ks + r * SK + i4) = khv;
            *(uint2 *)(Qs + r * SK + i4) = qhv;
            if (ok) {
                const int64_t o = ((int64_t)hk * n_tokens + t0 + r) * 128 + i4;
                *(uint2 *)(Kh + o) = khv;
                *(uint2 *)(Qh + o) = qhv;
            }
        }
        // v of the R value heads of this k-head -> Vh (fp16)
        for (int jh = 0; jh < R; ++jh) {
            const int h = hk + jh * Hk;
#pragma unroll
            for (int j = 0; j < NQK; ++j) {
                const int r = (tid + 256 * j) >> 5;
                if (r >= Lc) continue;
                const int64_t t = t0 + r;
                const float4 vv = *(const float4 *)(vb + t * ch_stride + (int64_t)h * 128 + i4);
                *(uint2 *)(Vh + ((int64_t)h * n_tokens + t) * 128 + i4) = gdnc_pack4h(vv);
            }
        }
    }
    for (int idx = tid; idx < R * CL; idx += 256) {
        const int j = idx / CL, r = idx % CL, h = hk + j * Hk;
        const int64_t o = (int64_t)(t0 + r) * H + h;
        Gs[j][r] = r < Lc ? g[o] : 0.0f;
        bs[j][r] = r < Lc ? beta[o] : 0.0f;
    }
    __syncthreads();
    if (tid < R) {
        float gv[CL];
#pragma unroll
        for (int r = 0; r < CL; ++r) gv[r] = Gs[tid][r];
#pragma unroll
        for (int r = 1; r < CL; ++r) gv[r] += gv[r - 1];
#pragma unroll
        for (int r = 0; r < CL; ++r) Gs[tid][r] = gv[r];
    }
    // raw lower-triangular tiles of K K^T and Q K^T (shared by all value heads of this k-head)
    for (int job = wave; job < 2 * NTRI; job += 8) {
        const bool isq = job >= NTRI;
        int jj = isq ? job - NTRI : job, rt = 0;
        while (jj >= rt + 1) {
            jj -= rt + 1;
            ++rt;
        }
        const int st = jj;
        const _Float16 *As = isq ? Qs : Ks;
        gdnc_v8f acc = {};
#pragma unroll
        for (int kk = 0; kk < 128; kk += 16) acc = GDNC_WMMA(gdnc_frag(As, SK, rt * 16, kk, lane), gdnc_frag(Ks, SK, st * 16, kk, lane), acc);
        float(*D)[CL + 1] = isq ? QK : KK;
#pragma unroll
        for (int e = 0; e < 8; ++e) D[rt * 16 + 2 * e + rh][st * 16 + cl] = acc[e];
    }
    __syncthreads();

    for (int j = 0; j < R; ++j) {
        const int h = hk + j * Hk;
        _Float16 *Pp = Pb + ((int64_t)h * nch + ch) * CL * CL;
        for (int idx = tid; idx < CL * CL; idx += 256) {
            const int r = idx / CL, s = idx % CL;
            const float d = s <= r ? expf(Gs[j][r] - Gs[j][s]) : 0.0f;
            A[r][s] = s < r ? bs[j][r] * KK[r][s] * d : 0.0f;
            Pp[idx] = (_Float16)(s <= r ? scale * QK[r][s] * d : 0.0f);
        }
        if (tid < CL) Gb[((int64_t)h * nch + ch) * CL + tid] = make_float2(expf(Gs[j][tid]), expf(Gs[j][CL - 1] - Gs[j][tid]));
        __syncthreads();
        // T = (I + A)^-1 column by column, barrier-free: lane owns rows lane + 32 m, the wave CPW columns
        {
            const int j0 = wave * CPW;
            float x[MR][CPW];
#pragma unroll
            for (int m = 0; m < MR; ++m)
#pragma unroll
                for (int c = 0; c < CPW; ++c) x[m][c] = (lane + 32 * m == j0 + c) ? 1.0f : 0.0f;
#pragma unroll
            for (int qq = 0; qq < CL - 1; ++qq) {
                if (qq < j0) continue;
                float a[MR];
#pragma unroll
                for (int m = 0; m < MR; ++m) a[m] = A[lane + 32 * m][qq];
#pragma unroll
                for (int c = 0; c < CPW; ++c) {
                    const float xq = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x[qq / 32][c]), qq % 32));
#pragma unroll
                    for (int m = 0; m < MR; ++m) x[m][c] = fmaf(-a[m], xq, x[m][c]);
                }
            }
            _Float16 *Tp = Tb + ((int64_t)h * nch + ch) * CL * CL;
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                const int r = lane + 32 * m;
                _Float16 tmp[CPW];
#pragma unroll
                for (int c = 0; c < CPW; ++c) tmp[c] = (_Float16)(x[m][c] * bs[j][j0 + c]);
                __builtin_memcpy(Tp + r * CL + j0, tmp, sizeof(tmp));
            }
        }
        __syncthreads();
    }
}

static __device__ __forceinline__ float ggml_cuda_op_silu_single(const float x) { return x / (1.0f + expf(-x)); }
static __device__ __forceinline__ uint16_t ggml_cuda_f32_to_bf16_rne(const float x) {
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);  // NaN stays NaN
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// K2: one block (8 waves) per value head, wave w owns state columns [16 w, 16 w + 16). The chunk's q/k/T'/P are
// staged once per block in LDS (K^T transposed on the way in); the next chunk is prefetched into registers.
// strix's kernel body verbatim except the state layout (WHIRL: [h][key][value]; curr_state and state may be the
// same buffer: each block reads its head's state before writing it). The optional fused gated output norm
// (on_z != nullptr) is kept as in strix but WHIRL passes on_z = nullptr: a rewrite without that branch spilled
// twice as much and ran 1.7x slower (GDN-8S tier 0).
// out (dst): [n_tokens][H][128].
#include "lds_barrier.h"  // feat026-barrierdrain: WHIRL_BARDRAIN bit 2 = gdn_sx_scan LDS-only barriers
#if WHIRL_BARDRAIN & 2
#define GDNSX_BAR() whirl_lds_barrier()
#else
#define GDNSX_BAR() __syncthreads()
#endif
extern "C" __global__ void __launch_bounds__(256, 6)
gdn_sx_scan(const _Float16 * __restrict__ Qh, const _Float16 * __restrict__ Kh, const _Float16 * __restrict__ Vh,
        const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
        const _Float16 * __restrict__ Tb, const _Float16 * __restrict__ Pb, const float2 * __restrict__ Gb,
        const int H, const int Hk, const int n_tokens, const int nch, const float scale,
        const float * __restrict__ on_w, const float * __restrict__ on_z, float * __restrict__ on_out, uint16_t * __restrict__ on_out16,
        const float on_eps, const bool on_silu) {
    constexpr int CL = GDN_SX_CL;
    constexpr int NT  = CL / 16;
    constexpr int SQ  = 136;
    constexpr int ST  = CL + 8;
    constexpr int NQK = CL * 32 / 256;            // 4-half groups of q (and of k) per thread per chunk
    constexpr int NTP = 2 * CL * CL / 8 / 256;         // uint4 of T'/P per thread per chunk
    static_assert(NTP * 256 == 2 * CL * CL / 8, "T'/P staging must divide evenly");
    __shared__ __align__(16) _Float16 Qs[CL * SQ];
    __shared__ __align__(16) _Float16 Ks[CL * SQ];
    __shared__ __align__(16) _Float16 KTs[128 * ST];
    __shared__ __align__(16) _Float16 Ts[CL * ST];
    __shared__ __align__(16) _Float16 Ps[CL * ST];
    __shared__ float2 Gs[CL];
    __shared__ float  red[CL][9];   // fused gated norm: per-row partial sums of squares of the 8 waves
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int hk = h % Hk;
    const int cl = lane & 15, rh = lane >> 4, col = wave * 16 + cl;

    const float * s_in = curr_state + (int64_t) h * 128 * 128;
    gdnc_v8f S[8];
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) S[it][e] = s_in[(it * 16 + 2 * e + rh) * 128 + col];

    const _Float16 * kh = Kh + (int64_t) hk * n_tokens * 128;   // prep's fp16 rows: normalized q / k, v
    const _Float16 * qh = Qh + (int64_t) hk * n_tokens * 128;
    const _Float16 * vh = Vh + (int64_t) h * n_tokens * 128 + col;

    uint2 pk[NQK], pq[NQK];
    uint4  ptp[NTP];
    float2 pg = make_float2(0.0f, 0.0f);
    float  pv[NT][8];
    float  pz[NT][8];   // fused output norm: the gate z of this lane's rows / column, prefetched with v
    const float * zh = on_z ? on_z + (int64_t) h * 128 + col : nullptr;
    auto prefetch = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            const int64_t o = (int64_t) min(t0 + r, n_tokens - 1) * 128 + i;
            pk[j] = *(const uint2 *) (kh + o);
            pq[j] = *(const uint2 *) (qh + o);
        }
        const _Float16 * Tp = Tb + ((int64_t) h * nch + ch) * CL * CL;
        const _Float16 * Pp = Pb + ((int64_t) h * nch + ch) * CL * CL;
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            // 2 CL^2 / 8 is a multiple of 256 for CL = 32 / 64: no guard (a divergent guard forces vmcnt(0) at the join)
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8);
            ptp[j] = *(const uint4 *) ((idx < CL * CL / 8 ? Tp : Pp) + w * 8);
        }
        pg = Gb[((int64_t) h * nch + ch) * CL + (tid % CL)];
    };
    // v / z of a chunk: issued right after the previous chunk consumed its own (one register copy each)
    auto prefetch_v = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int e = 0; e < 8; ++e) pv[rt][e] = (float) vh[(int64_t) min(t0 + rt * 16 + 2 * e + rh, n_tokens - 1) * 128];
    };
    auto prefetch_z = [&](const int ch) {
        const int t0 = ch * CL;
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int e = 0; e < 8; ++e) pz[rt][e] = zh[(int64_t) min(t0 + rt * 16 + 2 * e + rh, n_tokens - 1) * H * 128];
    };
    prefetch(0);
    prefetch_v(0);
    if (zh) prefetch_z(0);

    for (int ch = 0; ch < nch; ++ch) {
        const int t0 = ch * CL;
        GDNSX_BAR();
#pragma unroll
        for (int j = 0; j < NQK; ++j) {
            const int idx = tid + 256 * j, r = idx >> 5, i = (idx & 31) * 4;
            *(uint2 *) (Ks + r * SQ + i) = pk[j];
            *(uint2 *) (Qs + r * SQ + i) = pq[j];
            _Float16 kk4[4];
            __builtin_memcpy(kk4, &pk[j], 8);
            KTs[(i + 0) * ST + r] = kk4[0]; KTs[(i + 1) * ST + r] = kk4[1];
            KTs[(i + 2) * ST + r] = kk4[2]; KTs[(i + 3) * ST + r] = kk4[3];
        }
#pragma unroll
        for (int j = 0; j < NTP; ++j) {
            const int idx = tid + 256 * j;
            const int w = idx % (CL * CL / 8), r = w / (CL / 8), c8 = (w % (CL / 8)) * 8;
            *(uint4 *) ((idx < CL * CL / 8 ? Ts : Ps) + r * ST + c8) = ptp[j];
        }
        if (tid < CL) Gs[tid] = pg;
        GDNSX_BAR();
        // unconditional (clamped): a branch here makes the compiler drain the prefetch before the compute
        prefetch(min(ch + 1, nch - 1));

        // K S0, Q S0
        gdnc_v8f ks[NT], qs[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) { ks[rt] = gdnc_v8f{}; qs[rt] = gdnc_v8f{}; }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const gdnc_v16h sb = gdnc_acc_to_frag(S[it], lane);
#pragma unroll
            for (int rt = 0; rt < NT; ++rt) {
                ks[rt] = GDNC_WMMA(gdnc_frag(Ks, SQ, rt * 16, it * 16, lane), sb, ks[rt]);
                qs[rt] = GDNC_WMMA(gdnc_frag(Qs, SQ, rt * 16, it * 16, lane), sb, qs[rt]);
            }
        }
        // U = T' (V - diag(e^G) K S0)
        gdnc_v16h YB[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f y;
#pragma unroll
            for (int e = 0; e < 8; ++e) y[e] = fmaf(-Gs[rt * 16 + 2 * e + rh].x, ks[rt][e], pv[rt][e]);
            YB[rt] = gdnc_acc_to_frag(y, lane);
        }
        prefetch_v(min(ch + 1, nch - 1));
        gdnc_v8f U[NT];
        gdnc_v16h UB[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f u = {};
#pragma unroll
            for (int st = 0; st <= rt; ++st) u = GDNC_WMMA(gdnc_frag(Ts, ST, rt * 16, st * 16, lane), YB[st], u);
            U[rt]  = u;
            UB[rt] = gdnc_acc_to_frag(u, lane);
        }
        // S1 = e^{G_L} S0 + K^T diag(e^{G_L - G}) U
        gdnc_v16h UB2[NT];
#pragma unroll
        for (int st = 0; st < NT; ++st) {
            gdnc_v8f u;
#pragma unroll
            for (int e = 0; e < 8; ++e) u[e] = U[st][e] * Gs[st * 16 + 2 * e + rh].y;
            UB2[st] = gdnc_acc_to_frag(u, lane);
        }
        const float eGL = Gs[CL - 1].x;
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            gdnc_v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = S[it][e] * eGL;
#pragma unroll
            for (int st = 0; st < NT; ++st) acc = GDNC_WMMA(gdnc_frag(KTs, ST, it * 16, st * 16, lane), UB2[st], acc);
            S[it] = acc;
        }
        // O = scale diag(e^G) Q S0 + P U
        gdnc_v8f O[NT];
#pragma unroll
        for (int rt = 0; rt < NT; ++rt) {
            gdnc_v8f o;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = qs[rt][e] * (Gs[rt * 16 + 2 * e + rh].x * scale);
#pragma unroll
            for (int st = 0; st <= rt; ++st) o = GDNC_WMMA(gdnc_frag(Ps, ST, rt * 16, st * 16, lane), UB[st], o);
            O[rt] = o;
        }
        if (on_z == nullptr) {
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int t = t0 + rt * 16 + 2 * e + rh;
                    if (t < n_tokens) dst[((int64_t) t * H + h) * 128 + col] = O[rt][e];
                }
        } else {
            // fused gated RMS norm (rms_rows_f32<1/2>): v = rsqrt(mean(o^2) + eps) * o * w * sigmoid(z), or silu(z) * ...
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float p = O[rt][e] * O[rt][e];
                    p += __shfl_xor(p, 1, 32); p += __shfl_xor(p, 2, 32); p += __shfl_xor(p, 4, 32); p += __shfl_xor(p, 8, 32);
                    if (cl == 0) red[rt * 16 + 2 * e + rh][wave] = p;
                }
            GDNSX_BAR();
            const float wc = on_w[col];
#pragma unroll
            for (int rt = 0; rt < NT; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int r = rt * 16 + 2 * e + rh, t = t0 + r;
                    float ss = 0.0f;
#pragma unroll
                    for (int w8 = 0; w8 < 8; ++w8) ss += red[r][w8];
                    if (t < n_tokens) {
                        const int64_t oi = ((int64_t) t * H + h) * 128 + col;
                        const float sc = rsqrtf(ss / 128.0f + on_eps);
                        const float t  = sc * O[rt][e] * wc;
                        const float v  = on_silu ? ggml_cuda_op_silu_single(pz[rt][e]) * t : t * (1.0f / (1.0f + expf(-pz[rt][e])));
                        if (on_out) on_out[oi] = v;
                        if (on_out16) on_out16[oi] = ggml_cuda_f32_to_bf16_rne(v);
                    }
                }
            prefetch_z(min(ch + 1, nch - 1));
        }
    }

    float * s_out = state + (int64_t) h * 128 * 128;
#pragma unroll
    for (int it = 0; it < 8; ++it)
#pragma unroll
        for (int e = 0; e < 8; ++e) s_out[(it * 16 + 2 * e + rh) * 128 + col] = S[it][e];
}
#undef GDNC_WMMA
