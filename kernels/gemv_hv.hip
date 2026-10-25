// WHIRL device code (gfx1201 / RDNA 4): precise gemvh candidates ("direct" register ring) + tune table.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by gemv_h.hip (after gemm_prefill.hip / gemm_small.hip: dec_group_h helpers,
// bytes8_to_h8, mx_lut4, e8m0_half, half8 / float8).
//
// feat026-gemvh (progress/feat026-gemvh.PRIOR_ART.md). The shipped gfx1201 gemvh stages 16 rows through LDS and
// prefetches ONE stage into registers; the activation loads inside the k loop are issued after that prefetch and
// loadcnt retires in order, so the first activation wait also waits for the whole weight prefetch: about one
// stage (4.35 KB per wave for MXFP4) is in flight and its latency is exposed every round (base ISA, 2026-10-26).
//
// "direct" (this file): no LDS and no barriers. A unit is half a 256-k superblock (8 WMMA steps). Every lane loads
// exactly the bytes its dec_group_h<T>(row, 2 * kk + lane / 16) call reads (gfx12 WMMA layout: lane l = row l % 16,
// k 8 * (l / 16) .. + 8; no lane duplication on gfx12), as whole 8-byte pieces (kernels/gemv_hv_map.h), into a ring
// of D units. Per unit step: (1) the activations of the NEXT unit (XD = 1) or of this unit (XD = 0) are loaded,
// (2) then the weights of unit u + D - 1, (3) unit u is multiplied. In issue order every wait of step 3 is for loads
// issued before the newest weight unit, so D - 1 units of weights stay in flight during the WMMAs.
// Optional PF: s_prefetch_data of superblock (u / 2) + PF for the 16 rows (a hint; the length encoding of the
// instruction is not documented, PFLEN is passed through).
//
// Bit-identity contract (precise): for every output (row, token) the A fragment is bit for bit dec_group_h<T>'s
// (same helper calls: mx_lut4 + bytes8_to_h8 for MXFP4, the Q<QT_Q6_K>::dec8 float expression ds * v then
// (_Float16) for Q6_K), the B fragment is the same half8 of X, and the WMMA sequence is the same: one
// wmma(a, b, acc) per 16-k step in increasing k from zero. Tokens past n_tok read a valid (clamped) row and are
// never stored (output columns are independent, as in the gfx1151 kernels). Checked by whirl-kernel-test --only
// gemvhv (every type / shape / n / acc vs gemm_c0) and the --bench gemvh --ab cells (B bitwise == A).
//
// Tune table: WHIRL_GVH_<TYPE> = candidate id (0 = the shipped gemv_h.hip kernel, the default). A non-zero value
// makes this file define gemvh_<type> / gemvh2_<type> and gemv_h.hip skip its own. WHIRL_GVH_DEV=1 also exports
// every candidate as gvhd<id>_<type> (n_tok <= 32, block 32, grid rows / 16) for the gemvhbench dev path
// (WHIRL_KT_GEMVH_HSACO / WHIRL_KT_GEMVH_VARS) and the gemvhv test family.
#include "gemv_hv_map.h"

#ifndef WHIRL_GVH_MXFP4
#define WHIRL_GVH_MXFP4 0
#endif
#ifndef WHIRL_GVH_Q6_K
#define WHIRL_GVH_Q6_K 0
#endif
#ifndef WHIRL_GVH_DEV
#define WHIRL_GVH_DEV 0
#endif
#ifndef WHIRL_GVH_PFLEN
#define WHIRL_GVH_PFLEN 1
#endif

template <int T> struct GvhD;
// ---- MXFP4: unit = 4 blocks of 32 (68 bytes per row), lane pieces 4 x 8 bytes + the 4 scale bytes ----
template <> struct GvhD<QT_MXFP4> {
    static constexpr int SBB = 136, DMAX = 8;  // DMAX: deepest ring without spills (q2 ISA, 2026-10-26)
    struct Regs {
        uint2 q[4];
        uint32_t e;
    };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int g0) {
        const uint8_t *b = rowp + (size_t)(u >> 1) * SBB;
        const int ph = u & 1;
        r.e = *(const uint32_t *)(b + gvh_mx_e_off(ph));
#pragma unroll
        for (int i = 0; i < 4; ++i) r.q[i] = *(const uint2 *)(b + gvh_mx_q_off(ph, i, g0));
    }
    // == dec_group_h<QT_MXFP4>(row, 2 * kk + g0) for kk = 8 * u + kq
    static __device__ __forceinline__ half8 frag(const Regs &r, int kq) {
        const int i = gvh_mx_piece(kq), hi = gvh_mx_hi(kq);
        const float dl = e8m0_half((r.e >> (8 * i)) & 0xFFu);
        const uint2 q = r.q[i];
        const uint32_t n0 = hi ? (q.x >> 4) & 0x0F0F0F0F : q.x & 0x0F0F0F0F;
        const uint32_t n1 = hi ? (q.y >> 4) & 0x0F0F0F0F : q.y & 0x0F0F0F0F;
        const uint32_t u0 = (uint32_t)mx_lut4(n0) ^ 0x80808080u, u1 = (uint32_t)mx_lut4(n1) ^ 0x80808080u;
        return bytes8_to_h8(u0, u1, (_Float16)128.f, (_Float16)dl, (_Float16)0.f);
    }
};
// ---- Q6_K: unit = n half (128 values, 2-byte aligned superblocks), lane pieces ql 4 x 8, qh 2 x 8, sc 8, d 2 bytes ----
__device__ __forceinline__ uint2 gvh_ld8a2(const uint8_t *p) {
    uint2 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 8);
    return v;
}
template <> struct GvhD<QT_Q6_K> {
    static constexpr int SBB = 210, DMAX = 6;  // D = 8 spilled 24-45 VGPRs (q2 ISA, 2026-10-26)
    struct Regs {
        uint2 ql[4];
        uint2 qh[2];
        uint2 sc;
        uint32_t d;
    };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int g0) {
        const uint8_t *b = rowp + (size_t)(u >> 1) * SBB;
        const int ph = u & 1;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.ql[i] = gvh_ld8a2(b + gvh_q6_ql_off(ph, i, g0));
#pragma unroll
        for (int c = 0; c < 2; ++c) r.qh[c] = gvh_ld8a2(b + gvh_q6_qh_off(ph, c, g0));
        r.sc = gvh_ld8a2(b + gvh_q6_sc_off(ph));
        r.d = *(const uint16_t *)(b + 208);
    }
    // == dec_group_h<QT_Q6_K>(row, 2 * kk + g0) = (_Float16)(Q<QT_Q6_K>::dec8 value) for kk = 8 * u + kq
    static __device__ __forceinline__ half8 frag(const Regs &r, int kq) {
        const float d = __half2float(__ushort_as_half((uint16_t)r.d));
        const uint32_t sw = (kq & 4) ? r.sc.y : r.sc.x;
        const float ds = d * (int8_t)((sw >> (8 * (kq & 3))) & 0xFFu);
        const uint2 ql = r.ql[gvh_q6_ql_piece(kq)], qh = r.qh[gvh_q6_qh_piece(kq)];
        const int lsh = gvh_q6_lshift(kq), hsh = gvh_q6_hshift(kq);
        half8 h;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t lw = i < 4 ? ql.x : ql.y, hw = i < 4 ? qh.x : qh.y;
            const int sh = 8 * (i & 3);
            const int v = (int)(((((lw >> sh) & 0xFFu) >> lsh) & 0xFu) | (((((hw >> sh) & 0xFFu) >> hsh) & 3u) << 4)) - 32;
            const float o = ds * v;
            h[i] = (_Float16)o;
        }
        return h;
    }
};

template <int ID> struct GvhCfg;
#define GVH_CFG(ID_, D_, XD_, PF_)                         \
    template <> struct GvhCfg<ID_> {                       \
        static constexpr int D = D_, XD = XD_, PF = PF_;   \
    };
// id: ring depth D (units of 128 k), XD (activations one unit ahead), PF (s_prefetch_data, superblocks ahead)
GVH_CFG(1, 4, 1, 0)
GVH_CFG(2, 2, 1, 0)
GVH_CFG(3, 6, 1, 0)
GVH_CFG(4, 4, 0, 0)
GVH_CFG(5, 8, 0, 0)
GVH_CFG(6, 4, 1, 2)
GVH_CFG(7, 8, 1, 0)
GVH_CFG(8, 3, 0, 0)
#undef GVH_CFG
constexpr int kGvhIds = 8;

template <int T, int TN, int DREQ, int XD, int PF>
__device__ __forceinline__ void gvhd_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                          int ncols, int nrows, int n_tok, int accumulate) {
    using G = GvhD<T>;
    using R = typename G::Regs;
    constexpr int D = DREQ < G::DMAX ? DREQ : G::DMAX;
    // TN = 2 keeps the activations of one unit only (XD off): two token tiles x two slots would be 128 VGPRs and spill
    constexpr int XDE = (XD && TN == 1) ? 1 : 0;
    constexpr int XS = XDE ? 2 : 1;                   // activation ring slots
    constexpr int UR = (XDE && (D & 1)) ? 2 * D : D;  // unroll so every ring index is a constant
    const int lane = threadIdx.x % WARP;
    const int g0 = lane / 16, rl = lane % 16;
    const int nu = ncols / 128;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
        const int m0 = tile * 16;
        const int row = min(m0 + rl, nrows - 1);
        const uint8_t *rowp = W + (size_t)row * row_bytes;
        float8 acc[TN];
        const _Float16 *xr[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            xr[j] = X + (size_t)min(j * 16 + rl, n_tok - 1) * ncols + 8 * g0;
        }
        R w[D];
        half8 xb[XS][TN][8];
        // PF: the 16 row bases are wave-uniform (scalar registers)
        const uint8_t *prow[PF > 0 ? 16 : 1];
#pragma unroll
        for (int r = 0; r < (PF > 0 ? 16 : 1); ++r) prow[r] = W + (size_t)min(m0 + r, nrows - 1) * row_bytes;
        auto load_x = [&](half8 (&dst)[TN][8], int u) {
#pragma unroll
            for (int kq = 0; kq < 8; ++kq)
#pragma unroll
                for (int j = 0; j < TN; ++j) dst[j][kq] = *(const half8 *)(xr[j] + (size_t)u * 128 + kq * 16);
        };
        // Every load is unconditional (unit index clamped to the last unit: a load past the end only refills a ring slot
        // that is never read): loads under a branch make the waitcnt pass assume the worst at the join and wait for 0.
        if (XDE) load_x(xb[0], 0);
#pragma unroll
        for (int u = 0; u < D - 1; ++u) G::load(w[u], rowp, min(u, nu - 1), g0);
        // one unit step; s = u mod UR (a constant after unrolling: the ring slots)
        auto step = [&](int s, int u) {
            __builtin_amdgcn_sched_barrier(0);
            // (1) activations: next unit (XDE) or this unit
            if (XDE) load_x(xb[(s + 1) % XS], min(u + 1, nu - 1));
            else load_x(xb[0], u);
            // (2) weights of unit u + D - 1
            G::load(w[(s + D - 1) % D], rowp, min(u + D - 1, nu - 1), g0);
#if defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
            if (PF > 0 && (s & 1) == 0) {
                const size_t poff = (size_t)min((u >> 1) + PF, nu / 2 - 1) * G::SBB;
#pragma unroll
                for (int r = 0; r < 16; ++r) __builtin_amdgcn_s_prefetch_data(prow[r] + poff, WHIRL_GVH_PFLEN);
            }
#endif
            __builtin_amdgcn_sched_barrier(0);
            // (3) the 8 WMMA steps of unit u, increasing k
            const R &cur = w[s % D];
            const int xs = XDE ? s % XS : 0;
#pragma unroll
            for (int kq = 0; kq < 8; ++kq) {
                const half8 a = G::frag(cur, kq);
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, xb[xs][j][kq], acc[j]);
            }
        };
        int u0 = 0;
        for (; u0 + UR <= nu; u0 += UR) {
#pragma unroll
            for (int s = 0; s < UR; ++s) step(s, u0 + s);
        }
        // tail (nu % UR units; u0 is a multiple of UR, so the slots continue)
#pragma unroll
        for (int s = 0; s < UR - 1; ++s)
            if (u0 + s < nu) step(s, u0 + s);
        // epilogue: with accumulate, all 8 old values of a token tile are loaded first (clamped, unconditional) and then
        // added (the same f32 add as *dst + acc): the shipped kernel's guarded load-add-store per value waits for each
        // load in turn (8 round trips per tile)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int t = j * 16 + rl;
            const size_t yrow = (size_t)min(t, n_tok - 1) * nrows;
            float old[8];
            if (accumulate) {
#pragma unroll
                for (int v = 0; v < 8; ++v) old[v] = Y[yrow + min(m0 + 8 * g0 + v, nrows - 1)];
            }
            if (t >= n_tok) continue;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m0 + 8 * g0 + v;
                if (r < nrows) Y[yrow + r] = accumulate ? old[v] + acc[j][v] : acc[j][v];
            }
        }
    }
}

template <int T, int TN, int ID>
__device__ __forceinline__ void gvhd_run(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) {
    using C = GvhCfg<ID>;
    gvhd_impl<T, TN, C::D, C::XD, C::PF>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
}

// selected candidate as the production gemvh_<type> / gemvh2_<type> (same ABI as gemv_h.hip)
#define GVH_SEL_ENTRY(SUF, T, ID)                                                                                                         \
    extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,  \
                                                               int nrows, int n_tok, int accumulate) {                                    \
        gvhd_run<T, 1, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                          \
    }                                                                                                                                     \
    extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, \
                                                                int nrows, int n_tok, int accumulate) {                                   \
        gvhd_run<T, 2, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                          \
    }
#define GVH_SEL_ENTRY_X(SUF, T, ID) GVH_SEL_ENTRY(SUF, T, ID)
#if WHIRL_GVH_MXFP4
GVH_SEL_ENTRY_X(mxfp4, QT_MXFP4, WHIRL_GVH_MXFP4)
#endif
#if WHIRL_GVH_Q6_K
GVH_SEL_ENTRY_X(q6_k, QT_Q6_K, WHIRL_GVH_Q6_K)
#endif

#if WHIRL_GVH_DEV
// development exports: gvhd<id>_<type>, any n_tok <= 32 (TN chosen at run time; the branch is uniform)
#define GVH_DEV_ENTRY(SUF, T, ID)                                                                                                         \
    extern "C" __global__ void __launch_bounds__(32) gvhd##ID##_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,    \
                                                                    int ncols, int nrows, int n_tok, int accumulate) {                    \
        if (n_tok <= 16) gvhd_run<T, 1, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                         \
        else gvhd_run<T, 2, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                     \
    }
#define GVH_DEV_TYPES(ID) GVH_DEV_ENTRY(mxfp4, QT_MXFP4, ID) GVH_DEV_ENTRY(q6_k, QT_Q6_K, ID)
GVH_DEV_TYPES(1)
GVH_DEV_TYPES(2)
GVH_DEV_TYPES(3)
GVH_DEV_TYPES(4)
GVH_DEV_TYPES(5)
GVH_DEV_TYPES(6)
GVH_DEV_TYPES(7)
GVH_DEV_TYPES(8)
#endif
