// WHIRL device code (gfx1151 / RDNA 3.5): precise-mode decode / verify matmul (gemvh*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after gemm_small.hip by kernels/gfx1151/whirl_kernels_gfx1151.hip.
//
// gemvh_<type> (n_tok <= 16) / gemvh2_<type> (n_tok <= 32): y[t][r] (+)= sum_k W[r][k] x16[t][k]
// Same ABI and numerics as the gfx1201 kernel (kernels/gemv_h.hip) and gemms11_impl: the weights are
// dec_group_h<T> values (16 consecutive k per WMMA step), the f16 activations the other operand, one WMMA
// per 16-k step in increasing k from zero, tokens past n_tok are never stored -> bitwise == gemms_c* == gemm_c*
// (kernel test family gemvh checks every type against gemm_c0 and all gemms_c*).
//
// Three implementations (GEMVH-2; the sweep and the numbers are in audit/GEMVH-1151.md):
//  * "direct" (q4_k q5_k q8_0): one wave = 16 weight rows, no LDS and no barriers. Every lane loads its row's
//    16-byte pieces straight from global memory (lanes l and l + 16 read the same bytes; 2-byte aligned pieces use
//    unaligned global loads, as the int8 Q6_K GEMV does), a ring of GhW<T>::DP units stays in registers so the next
//    loads are in flight while the current one is decoded, and each lane decodes the whole 16-k fragment itself
//    (no cross-lane join). The decoders reproduce dec_group_h<T> bit for bit; the byte -> half spreading is one
//    v_perm_b32 per half2 (the low byte of a 0x64xx half is the byte).
//  * "paired" (iq4_xs mxfp4 iq4_nl): same loads, but the two lanes of a row (r and r + 16) each decode ONE
//    of the two k-steps of a pair (half the decode work, half the load registers) and swap the result with one
//    permlanex16 per dword; a per-lane select then gives both half-waves the identical A fragment of each step, as
//    the documented wave32 WMMA layout requires (lanes 16-31 replicate 0-15; FIX-RC4-B: the GEMVH-2 version left the
//    halves different and relied on which lanes one gfx1151 part happened to read). The weights are the first WMMA
//    operand and the accumulator is D[weight row][token] (the bits are those of the other operand order).
//  * "dual" (q6_k q3_k, chosen per type by WHIRL_GH_KIND_<T>; GEMVH-DUAL in audit/FIX-RC4-B.md): the paired decoders
//    without the exchange: every lane loads both halves and decodes both k-steps itself (twice the decode, no
//    permlanex16 / select), so both half-waves hold identical fragments. Faster than the select form for q6_k/q3_k,
//    slower for iq4_xs/mxfp4/iq4_nl (those stay "paired").
//  * "staged" (f16 iq3_s): the original design: the raw stages go through LDS (end of file).

// ---- direct and paired implementations ----
// WHIRL_GH_INJECT (kernel-test development only, never set by the build): 1 = paired lanes' k-steps swapped, 2 = the two
// WMMAs of a pair in the wrong order, 3 = wrong half-superblock phase, 4 = half-waves holding different A fragments
// (the pre-FIX-RC4-B layout). --only gemvh must fail for 1-3 (audit/FIX-RC4-B.md).
#ifndef WHIRL_GH_INJECT
#define WHIRL_GH_INJECT 0
#endif
__device__ __forceinline__ gs_u32x4 gh_ld16a2(const uint8_t *p) {
    gs_u32x4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
    return v;
}
__device__ __forceinline__ uint32_t gh_ld16u(const uint8_t *p) { return *(const uint16_t *)p; }
__device__ __forceinline__ uint32_t gh_ld32a4(const uint8_t *p) { return *(const uint32_t *)p; }
__device__ __forceinline__ half16 gh_join(half8 a, half8 b) { return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
// scale / min of K-quant sub-block s from the 12 packed scale bytes (3 dwords)
__device__ __forceinline__ void gh_scale_min(const uint32_t *scw, int s, int &sc, int &m) {
    auto by = [&](int idx) { return (scw[idx >> 2] >> (8 * (idx & 3))) & 0xFFu; };
    if (s < 4) {
        sc = by(s) & 63;
        m = by(s + 4) & 63;
    } else {
        sc = (by(s + 4) & 0xF) | ((by(s - 4) >> 6) << 4);
        m = (by(s + 4) >> 4) | ((by(s) >> 6) << 4);
    }
}
__device__ __forceinline__ float gh_h2f(uint32_t u) { return __half2float(__ushort_as_half((uint16_t)u)); }
// cheaper table select than iq4_lut4/mx_lut4: the bit-3 blend mask comes from a v_perm of a constant instead of a multiply
__device__ __forceinline__ uint32_t gh_lut4(uint32_t nib4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = __builtin_amdgcn_perm(0u, 0x0000FF00u, (nib4 >> 3) & 0x01010101u);
    return (hi & m) | (lo & ~m);
}
// bytes8_to_h8 with the byte -> (1024 + byte) half spreading done by one v_perm per half2
__device__ __forceinline__ half8 gh_b8h(uint32_t u0, uint32_t u1, _Float16 bias, _Float16 scale, _Float16 add) {
    const uint32_t m = 0x64646464u;
    const uint32_t pw[4] = {__builtin_amdgcn_perm(m, u0, 0x04010400u), __builtin_amdgcn_perm(m, u0, 0x04030402u), __builtin_amdgcn_perm(m, u1, 0x04010400u),
                            __builtin_amdgcn_perm(m, u1, 0x04030402u)};
    const half2_t b2 = {(_Float16)1024.f + bias, (_Float16)1024.f + bias};
    const half2_t s2 = {scale, scale}, a2 = {add, add};
    half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const half2_t v = (__builtin_bit_cast(half2_t, pw[i]) - b2) * s2 + a2;
        out[2 * i] = v[0];
        out[2 * i + 1] = v[1];
    }
    return out;
}
// exact (byte - bias) as halves: perm-spread (1024 + byte), one packed subtract
__device__ __forceinline__ void gh_x4(uint32_t u, int bias, _Float16 *o) {
    const uint32_t m = 0x64646464u;
    const half2_t b2 = {(_Float16)(1024 + bias), (_Float16)(1024 + bias)};
    const half2_t v0 = __builtin_bit_cast(half2_t, __builtin_amdgcn_perm(m, u, 0x04010400u)) - b2;
    const half2_t v1 = __builtin_bit_cast(half2_t, __builtin_amdgcn_perm(m, u, 0x04030402u)) - b2;
    o[0] = v0[0]; o[1] = v0[1]; o[2] = v1[0]; o[3] = v1[1];
}
template <int T> struct GhW;
// ---- Q4_K / Q5_K (256-k superblock, 16 steps) ----
template <> struct GhW<QT_Q4_K> {
    static constexpr int UK = 16, UB = 144, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; uint32_t dm; uint32_t sc[3]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.dm = gh_ld32a4(b);
        r.sc[0] = gh_ld32a4(b + 4); r.sc[1] = gh_ld32a4(b + 8); r.sc[2] = gh_ld32a4(b + 12);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = *(const gs_u32x4 *)(b + 16 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int s = kk >> 1, c = kk & 1, hi = s & 1, j64 = s >> 1;
        const float d = gh_h2f(r.dm), dmin = gh_h2f(r.dm >> 16);
        int sc, m;
        gh_scale_min(r.sc, s, sc, m);
        const gs_u32x4 q = r.q[j64 * 2 + c];
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = hi ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F;
        const _Float16 hs = (_Float16)(d * sc), ha = (_Float16)(-dmin * m);
        return gh_join(gh_b8h(w[0], w[1], (_Float16)0.f, hs, ha), gh_b8h(w[2], w[3], (_Float16)0.f, hs, ha));
    }
};
template <> struct GhW<QT_Q5_K> {
    static constexpr int UK = 16, UB = 176, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; gs_u32x4 qh[2]; uint32_t dm; uint32_t sc[3]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.dm = gh_ld32a4(b);
        r.sc[0] = gh_ld32a4(b + 4); r.sc[1] = gh_ld32a4(b + 8); r.sc[2] = gh_ld32a4(b + 12);
        r.qh[0] = *(const gs_u32x4 *)(b + 16); r.qh[1] = *(const gs_u32x4 *)(b + 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = *(const gs_u32x4 *)(b + 48 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int s = kk >> 1, c = kk & 1, hi = s & 1, j64 = s >> 1;
        const float d = gh_h2f(r.dm), dmin = gh_h2f(r.dm >> 16);
        int sc, m;
        gh_scale_min(r.sc, s, sc, m);
        const gs_u32x4 q = r.q[j64 * 2 + c], h = r.qh[c];
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = (hi ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F) | (((h[i] >> s) & 0x01010101) << 4);
        const _Float16 hs = (_Float16)(d * sc), ha = (_Float16)(-dmin * m);
        return gh_join(gh_b8h(w[0], w[1], (_Float16)0.f, hs, ha), gh_b8h(w[2], w[3], (_Float16)0.f, hs, ha));
    }
};
// ---- Q6_K (210-byte superblock, 2-byte aligned) ----
template <> struct GhW<QT_Q6_K> {
    static constexpr int UK = 16, UB = 210, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 ql[8]; gs_u32x4 qh[4]; gs_u32x4 sc; uint32_t d; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 8; ++i) r.ql[i] = gh_ld16a2(b + 16 * i);
#pragma unroll
        for (int i = 0; i < 4; ++i) r.qh[i] = gh_ld16a2(b + 128 + 16 * i);
        r.sc = gh_ld16a2(b + 192);
        r.d = gh_ld16u(b + 208);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int n = kk >> 3, quarter = (kk >> 1) & 3, c = kk & 1;
        const float d = gh_h2f(r.d);
        const int sidx = 8 * n + c + 2 * quarter;
        const float ds = d * (float)(int8_t)((r.sc[sidx >> 2] >> (8 * (sidx & 3))) & 0xFF);
        const gs_u32x4 ql = r.ql[4 * n + 2 * (quarter & 1) + c], qh = r.qh[2 * n + c];
        const int lsh = (quarter >> 1) ? 4 : 0, hsh = 2 * quarter;
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t w = ((ql[wi] >> lsh) & 0x0F0F0F0Fu) | (((qh[wi] >> hsh) & 0x03030303u) << 4);
            _Float16 x[4];
            gh_x4(w, 32, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(ds * (float)x[i]);
        }
        return out;
    }
};
// ---- Q8_0 (34-byte blocks of 32 k; unit = 2 blocks = 4 steps, UB = 68) ----
template <> struct GhW<QT_Q8_0> {
    static constexpr int UK = 4, UB = 17 * 4, DP = 3, SBE = 2;
    struct Regs { gs_u32x4 q[4]; uint32_t d[4 / 2]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 4 / 2; ++i) {
            r.d[i] = gh_ld16u(b + 34 * i);
            r.q[2 * i] = gh_ld16a2(b + 34 * i + 2);
            r.q[2 * i + 1] = gh_ld16a2(b + 34 * i + 18);
        }
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const float d = gh_h2f(r.d[kk >> 1]);
        const gs_u32x4 q = r.q[kk];
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            _Float16 x[4];
            gh_x4(q[wi] ^ 0x80808080u, 128, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(d * (float)x[i]);
        }
        return out;
    }
};

template <int T, int TN>
__device__ __forceinline__ void gemvh_direct_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                           int ncols, int nrows, int n_tok, int accumulate) {
    using D = GhW<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
    const int m0 = tile * 16;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc[TN];
    const _Float16 *xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    Regs buf[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) D::load(buf[u], rowp, u);
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) D::load(buf[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1);
#pragma unroll
                for (int kk = 0; kk < D::UK; ++kk) {
                    if (kk % SB_EVERY == 0) __builtin_amdgcn_sched_barrier(0);
                    const half16 a = D::frag(buf[u], kk);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        // (tokens past n_tok read a valid row; their output columns are never stored, columns are independent)
                        const half16 b = ld_frag16(xr[j] + sb * (16 * D::UK) + kk * 16);
                        acc[j] = WMMA16(b, a, acc[j]);
                    }
                }
            }
        }
    }
    const int orow = m0 + rl;
    if (orow < nrows)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = j * 16 + CROW(lane, v);
            if (t < n_tok) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
    }
}

template <int T> struct GhQ;
template <> struct GhQ<QT_Q6_K> {
    static constexpr int UK = 16, UB = 210, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 ql[4]; gs_u32x4 qh[2]; gs_u32x4 sc; uint32_t d; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int role) {
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.ql[i] = gh_ld16a2(b + 32 * i + 16 * role);
#pragma unroll
        for (int i = 0; i < 2; ++i) r.qh[i] = gh_ld16a2(b + 128 + 32 * i + 16 * role);
        r.sc = gh_ld16a2(b + 192);
        r.d = gh_ld16u(b + 208);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int m, int role, int ph) {
        (void)ph;
        const int n = m >> 2, quarter = m & 3;
        const float d = gh_h2f(r.d);
        const int sbyte = (2 * quarter + role) & 3;
        const uint32_t sw = r.sc[2 * n + (quarter >> 1)];
        const float ds = d * (float)(int8_t)((sw >> (8 * sbyte)) & 0xFF);
        const gs_u32x4 ql = r.ql[2 * n + (quarter & 1)], qh = r.qh[n];
        const int lsh = (quarter >> 1) ? 4 : 0, hsh = 2 * quarter;
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t w = ((ql[wi] >> lsh) & 0x0F0F0F0Fu) | (((qh[wi] >> hsh) & 0x03030303u) << 4);
            _Float16 x[4];
            gh_x4(w, 32, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(ds * (float)x[i]);
        }
        return out;
    }
};
// 136-byte superblocks (MXFP4 / IQ4_XS): a unit is half a superblock (8 steps, 128 k); unit u is half (u & 1) of
// superblock u >> 1. load() and frag() both take that phase from the unit index (the caller passes sb & 1), so
// any ring depth DP works.
template <> struct GhQ<QT_MXFP4> {
    static constexpr int UK = 8, UB = 136 / 2, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8 / 2]; uint32_t sc[2]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int role) {
        (void)role;
        const uint8_t *b = rowp + (size_t)(u >> 1) * 136;
        const int ph = u & 1;
        r.sc[0] = gh_ld32a4(b); r.sc[1] = gh_ld32a4(b + 4);
#pragma unroll
        for (int i = 0; i < 8 / 2; ++i) r.q[i] = gh_ld16a2(b + 8 + 16 * (i + 4 * ph));
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int m, int role, int ph) {
        const int ib = m + 4 * ph;  // ph = unit & 1 (half of the superblock)
        const float dl = e8m0_half((r.sc[ib >> 2] >> (8 * (ib & 3))) & 0xFF);
        const gs_u32x4 q = r.q[m];
        const int sh = 4 * role;
        uint32_t u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gh_lut4((q[i] >> sh) & 0x0F0F0F0F, 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu) ^ 0x80808080u;
        return gh_join(gh_b8h(u[0], u[1], (_Float16)128.f, (_Float16)dl, (_Float16)0.f), gh_b8h(u[2], u[3], (_Float16)128.f, (_Float16)dl, (_Float16)0.f));
    }
};
template <> struct GhQ<QT_IQ4_XS> {
    static constexpr int UK = 8, UB = 136 / 2, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8 / 2]; uint32_t hdr, sl; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int role) {
        (void)role;
        const uint8_t *b = rowp + (size_t)(u >> 1) * 136;
        const int ph = u & 1;
        r.hdr = gh_ld32a4(b); r.sl = gh_ld32a4(b + 4);
#pragma unroll
        for (int i = 0; i < 8 / 2; ++i) r.q[i] = gh_ld16a2(b + 8 + 16 * (i + 4 * ph));
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int m, int role, int ph) {
        const int ib = m + 4 * ph;  // ph = unit & 1 (half of the superblock)
        const float d = gh_h2f(r.hdr);
        const int ls = ((r.sl >> (8 * (ib >> 1)) >> (4 * (ib & 1))) & 0xF) | ((((r.hdr >> 16) >> (2 * ib)) & 3) << 4);
        const gs_u32x4 q = r.q[m];
        const int sh = 4 * role;
        uint32_t u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gh_lut4((q[i] >> sh) & 0x0F0F0F0F, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u) ^ 0x80808080u;
        const _Float16 hs = (_Float16)(d * (ls - 32));
        return gh_join(gh_b8h(u[0], u[1], (_Float16)128.f, hs, (_Float16)0.f), gh_b8h(u[2], u[3], (_Float16)128.f, hs, (_Float16)0.f));
    }
};
// ---- IQ4_NL (18-byte blocks of 32; unit = 4 blocks) ----
template <> struct GhQ<QT_IQ4_NL> {
    static constexpr int UK = 8, UB = 72, DP = 4, SBE = 4;
    struct Regs { gs_u32x4 q[4]; uint32_t d[4]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int role) {
        (void)role;
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.d[i] = gh_ld16u(b + 18 * i);
            r.q[i] = gh_ld16a2(b + 18 * i + 2);
        }
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int m, int role, int ph) {
        (void)ph;
        const float d = gh_h2f(r.d[m]);
        const gs_u32x4 q = r.q[m];
        const int sh = 4 * role;
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t l = gh_lut4((q[wi] >> sh) & 0x0F0F0F0F, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u) ^ 0x80808080u;
            _Float16 x[4];
            gh_x4(l, 128, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(d * (float)x[i]);
        }
        return out;
    }
};
// ---- Q3_K (110-byte superblock) ----
template <> struct GhQ<QT_Q3_K> {
    static constexpr int UK = 16, UB = 110, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 hm; gs_u32x4 qs[2]; uint32_t sb[3]; uint32_t d; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u, int role) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.hm = gh_ld16a2(b + 16 * role);
#pragma unroll
        for (int n = 0; n < 2; ++n) r.qs[n] = gh_ld16a2(b + 32 + 32 * n + 16 * role);
#pragma unroll
        for (int i = 0; i < 3; ++i) r.sb[i] = gh_ld16u(b + 96 + 4 * i) | (gh_ld16u(b + 98 + 4 * i) << 16);
        r.d = gh_ld16u(b + 108);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int m, int role, int ph) {
        (void)ph;
        const int n = m >> 2, j = m & 3;
        const int k = 8 * n + 2 * j + role;
        auto sbyte = [&](int idx) { return (r.sb[idx >> 2] >> (8 * (idx & 3))) & 0xFFu; };
        const int lo = (k < 8) ? (sbyte(k) & 0xF) : (sbyte(k - 8) >> 4);
        const int hi = (sbyte(8 + (k & 3)) >> (2 * (k >> 2))) & 3;
        const float dl = gh_h2f(r.d) * (float)((lo | (hi << 4)) - 32);
        const int bit = 4 * n + j, shift = 2 * j;
        const gs_u32x4 q = r.qs[n];
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t w = ((q[wi] >> shift) & 0x03030303u) | (((r.hm[wi] >> bit) & 0x01010101u) << 2);
            _Float16 x[4];
            gh_x4(w, 4, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(dl * (float)x[i]);
        }
        return out;
    }
};

__device__ __forceinline__ half16 gh_xchg(half16 v) {
    const int4 a = __builtin_bit_cast(int4, __builtin_shufflevector(v, v, 0, 1, 2, 3, 4, 5, 6, 7));
    const int4 b = __builtin_bit_cast(int4, __builtin_shufflevector(v, v, 8, 9, 10, 11, 12, 13, 14, 15));
    const int4 pa = make_int4(xhalf(a.x), xhalf(a.y), xhalf(a.z), xhalf(a.w));
    const int4 pb = make_int4(xhalf(b.x), xhalf(b.y), xhalf(b.z), xhalf(b.w));
    return __builtin_shufflevector(__builtin_bit_cast(half8, pa), __builtin_bit_cast(half8, pb), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// ld_frag16 of f32 activations: each element rounded to f16 exactly as f32_to_f16 does ((_Float16)v, round to
// nearest even) - the fused MoE kernel reads the f32 h (feat026)
__device__ __forceinline__ half16 ld_frag16(const float *p) {
    const float4 a = ((const float4 *)p)[0], b = ((const float4 *)p)[1], c = ((const float4 *)p)[2], d = ((const float4 *)p)[3];
    return (half16){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w, (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w,
                    (_Float16)c.x, (_Float16)c.y, (_Float16)c.z, (_Float16)c.w, (_Float16)d.x, (_Float16)d.y, (_Float16)d.z, (_Float16)d.w};
}

template <int T, int TN, typename XT = _Float16>
__device__ __forceinline__ void gemvh_paired_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const XT *__restrict__ X, float *__restrict__ Y,
                                           int ncols, int nrows, int n_tok, int accumulate) {
    using D = GhQ<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
    const int m0 = tile * 16;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    // lanes r and r + 16 (both hold row r) decode one k-step each of the pair (2m, 2m + 1): role 0 -> step 2m, role 1 -> 2m + 1
#if defined(WHIRL_GH_ROLE_PARITY)
    const int role = ((lane & 1) == (lane >> 4)) ? 0 : 1;
#else
    const int role = lane >> 4;
#endif
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc[TN];
    const XT *xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    Regs buf[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) D::load(buf[u], rowp, u, role);
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) D::load(buf[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1, role);
#pragma unroll
                for (int m = 0; m < D::UK / 2; ++m) {
                    if (m % (SB_EVERY / 2) == 0) __builtin_amdgcn_sched_barrier(0);
#if WHIRL_GH_INJECT == 3
                    const half16 f = D::frag(buf[u], m, role, (sb + 1) & 1);  // test injection: wrong half-superblock phase
#else
                    const half16 f = D::frag(buf[u], m, role, sb & 1);
#endif
                    // x = the partner lane's step. Both half-waves then hold the SAME A fragment for row r (the documented
                    // wave32 WMMA layout: lanes 16-31 replicate lanes 0-15), whichever lanes the hardware reads.
                    const half16 x = gh_xchg(f);
#if WHIRL_GH_INJECT == 1
                    const half16 a0 = role ? f : x, a1 = role ? x : f;  // test injection: the two lanes' steps swapped
#elif WHIRL_GH_INJECT == 4
                    const half16 a0 = f, a1 = x;  // test injection: the pre-FIX-RC4-B half-waves that differ (P2-1)
#else
                    const half16 a0 = role ? x : f, a1 = role ? f : x;
#endif
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const half16 b0 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m) * 16);
                        const half16 b1 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m + 1) * 16);
#if WHIRL_GH_INJECT == 2
                        acc[j] = WMMA16(a1, b1, acc[j]);  // test injection: k-steps of a pair accumulated out of order
                        acc[j] = WMMA16(a0, b0, acc[j]);
#else
                        acc[j] = WMMA16(a0, b0, acc[j]);
                        acc[j] = WMMA16(a1, b1, acc[j]);
#endif
                    }
                }
            }
        }
    }
    // D[m = weight row][n = token]: lane (m & 1) * 16 + n, element v = m >> 1
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = j * 16 + lane % 16;
        if (t < n_tok) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int orow = m0 + 2 * v + lane / 16;
                if (orow < nrows) {
                    float *dst = Y + (size_t)t * nrows + orow;
                    *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
                }
            }
        }
    }
    }
}

// one wave's 16-row tile at row m0 of the paired kind (feat026, for the fused MoE kernel: a copy of gemvh_paired_impl's loop
// body, kept separate so the existing gemvh kernels' machine code stays identical - scratch\f026 ISA check; kernel test
// moefuse proves the copy bitwise against f32_to_f16 + gemvh): acc[j] = D[weight row][token] (lane holds token j * 16 + lane % 16,
// rows m0 + 2 * v + lane / 16); no LDS, no barriers
template <int T, int TN, typename XT>
__device__ __forceinline__ void gemvh_paired_tile(const uint8_t *__restrict__ W, uint64_t row_bytes, const XT *__restrict__ X, int ncols,
                                                  int nrows, int n_tok, int m0, float8 (&acc)[TN]) {
    using D = GhQ<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    // lanes r and r + 16 (both hold row r) decode one k-step each of the pair (2m, 2m + 1): role 0 -> step 2m, role 1 -> 2m + 1
#if defined(WHIRL_GH_ROLE_PARITY)
    const int role = ((lane & 1) == (lane >> 4)) ? 0 : 1;
#else
    const int role = lane >> 4;
#endif
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    const XT *xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    Regs buf[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) D::load(buf[u], rowp, u, role);
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) D::load(buf[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1, role);
#pragma unroll
                for (int m = 0; m < D::UK / 2; ++m) {
                    if (m % (SB_EVERY / 2) == 0) __builtin_amdgcn_sched_barrier(0);
#if WHIRL_GH_INJECT == 3
                    const half16 f = D::frag(buf[u], m, role, (sb + 1) & 1);  // test injection: wrong half-superblock phase
#else
                    const half16 f = D::frag(buf[u], m, role, sb & 1);
#endif
                    // x = the partner lane's step. Both half-waves then hold the SAME A fragment for row r (the documented
                    // wave32 WMMA layout: lanes 16-31 replicate lanes 0-15), whichever lanes the hardware reads.
                    const half16 x = gh_xchg(f);
#if WHIRL_GH_INJECT == 1
                    const half16 a0 = role ? f : x, a1 = role ? x : f;  // test injection: the two lanes' steps swapped
#elif WHIRL_GH_INJECT == 4
                    const half16 a0 = f, a1 = x;  // test injection: the pre-FIX-RC4-B half-waves that differ (P2-1)
#else
                    const half16 a0 = role ? x : f, a1 = role ? f : x;
#endif
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const half16 b0 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m) * 16);
                        const half16 b1 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m + 1) * 16);
#if WHIRL_GH_INJECT == 2
                        acc[j] = WMMA16(a1, b1, acc[j]);  // test injection: k-steps of a pair accumulated out of order
                        acc[j] = WMMA16(a0, b0, acc[j]);
#else
                        acc[j] = WMMA16(a0, b0, acc[j]);
                        acc[j] = WMMA16(a1, b1, acc[j]);
#endif
                    }
                }
            }
        }
    }
}

// gemvh_dual_impl: the paired decoders without the exchange: every lane loads both halves (role 0 and role 1 regs) and decodes
// both k-steps of a pair itself, so the two half-waves hold identical A fragments with no permlanex16 / select, at
// twice the decode work of "paired" (FIX-RC4-B timing alternative).
template <int T, int TN>
__device__ __forceinline__ void gemvh_dual_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                                int ncols, int nrows, int n_tok, int accumulate) {
    using D = GhQ<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
    const int m0 = tile * 16;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc[TN];
    const _Float16 *xr[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    Regs b0r[DEPTH], b1r[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) { D::load(b0r[u], rowp, u, 0); D::load(b1r[u], rowp, u, 1); }
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) {
                    D::load(b0r[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1, 0);
                    D::load(b1r[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1, 1);
                }
#pragma unroll
                for (int m = 0; m < D::UK / 2; ++m) {
                    if (m % (SB_EVERY / 2) == 0) __builtin_amdgcn_sched_barrier(0);
#if WHIRL_GH_INJECT == 3
                    const int ph = (sb + 1) & 1;
#else
                    const int ph = sb & 1;
#endif
#if WHIRL_GH_INJECT == 1
                    const half16 a1 = D::frag(b0r[u], m, 0, ph);
                    const half16 a0 = D::frag(b1r[u], m, 1, ph);
#else
                    const half16 a0 = D::frag(b0r[u], m, 0, ph);
                    const half16 a1 = D::frag(b1r[u], m, 1, ph);
#endif
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const half16 x0 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m) * 16);
                        const half16 x1 = ld_frag16(xr[j] + sb * (16 * D::UK) + (2 * m + 1) * 16);
#if WHIRL_GH_INJECT == 2
                        acc[j] = WMMA16(a1, x1, acc[j]);
                        acc[j] = WMMA16(a0, x0, acc[j]);
#else
                        acc[j] = WMMA16(a0, x0, acc[j]);
                        acc[j] = WMMA16(a1, x1, acc[j]);
#endif
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = j * 16 + lane % 16;
        if (t < n_tok) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int orow = m0 + 2 * v + lane / 16;
                if (orow < nrows) {
                    float *dst = Y + (size_t)t * nrows + orow;
                    *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
                }
            }
        }
    }
    }
}

#define GH_ENTRY(KIND, SUF, T)                                                                                                                      extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,                                                                   int nrows, int n_tok, int accumulate) {                                               gemvh_##KIND##_impl<T, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                            }                                                                                                                                              extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,                                                                   int nrows, int n_tok, int accumulate) {                                              gemvh_##KIND##_impl<T, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                            }
GH_ENTRY(direct, q8_0, QT_Q8_0)
GH_ENTRY(direct, q4_k, QT_Q4_K)
GH_ENTRY(direct, q5_k, QT_Q5_K)
#define GH_ENTRY_X(KIND, SUF, T) GH_ENTRY(KIND, SUF, T)
// per-type kind (timing builds override, e.g. -DWHIRL_GH_KIND_PAIRED=dual or -DWHIRL_GH_KIND_Q6_K=direct)
#ifdef WHIRL_GH_KIND_PAIRED
#define WHIRL_GH_KIND_Q6_K WHIRL_GH_KIND_PAIRED
#define WHIRL_GH_KIND_IQ4_XS WHIRL_GH_KIND_PAIRED
#define WHIRL_GH_KIND_MXFP4 WHIRL_GH_KIND_PAIRED
#define WHIRL_GH_KIND_IQ4_NL WHIRL_GH_KIND_PAIRED
#define WHIRL_GH_KIND_Q3_K WHIRL_GH_KIND_PAIRED
#endif
#ifndef WHIRL_GH_KIND_Q6_K
#define WHIRL_GH_KIND_Q6_K dual
#endif
#ifndef WHIRL_GH_KIND_IQ4_XS
#define WHIRL_GH_KIND_IQ4_XS paired
#endif
#ifndef WHIRL_GH_KIND_MXFP4
#define WHIRL_GH_KIND_MXFP4 paired
#endif
#ifndef WHIRL_GH_KIND_IQ4_NL
#define WHIRL_GH_KIND_IQ4_NL paired
#endif
#ifndef WHIRL_GH_KIND_Q3_K
#define WHIRL_GH_KIND_Q3_K dual
#endif
GH_ENTRY_X(WHIRL_GH_KIND_Q6_K, q6_k, QT_Q6_K)
GH_ENTRY_X(WHIRL_GH_KIND_IQ4_XS, iq4_xs, QT_IQ4_XS)
GH_ENTRY_X(WHIRL_GH_KIND_MXFP4, mxfp4, QT_MXFP4)
GH_ENTRY_X(WHIRL_GH_KIND_IQ4_NL, iq4_nl, QT_IQ4_NL)
GH_ENTRY_X(WHIRL_GH_KIND_Q3_K, q3_k, QT_Q3_K)
// feat026: gemvhf* = gemvh* on f32 activations (each rounded to f16 in the B load as f32_to_f16 does), paired kind only
// (bitwise == every kind: all equal gemm_c0); other types keep f32_to_f16 + gemvh (the host falls back on a null kernel).
// No gemvhf2 here: the two-token-tile f32 variant needs 256 VGPRs and spills on gfx1151 (n 17..32 keeps f32_to_f16 + gemvh2).
#define GHF_ENTRY(SUF, T)                                                                                                                    \
    extern "C" __global__ void __launch_bounds__(32) gemvhf_##SUF(const uint8_t *W, uint64_t row_bytes, const float *X, float *Y, int ncols, \
                                                                    int nrows, int n_tok, int accumulate) {                                  \
        gemvh_paired_impl<T, 1, float>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                 \
    }
GHF_ENTRY(mxfp4, QT_MXFP4)
GHF_ENTRY(iq4_xs, QT_IQ4_XS)
GHF_ENTRY(iq4_nl, QT_IQ4_NL)

// ---- staged (LDS) implementation: f16 and iq3_s weights ----
template <int T, int TN, int S>
__device__ __forceinline__ void gemvh11_impl(uint8_t *__restrict__ Ws, const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                             float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int SB = S * gs_sbb<T>();
    constexpr int CH = (SB + 15) / 16 + 1;  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = 16 * CH;
    constexpr int WPER = (WCH + WARP - 1) / WARP;
    const int lane = threadIdx.x % WARP;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
    const int m0 = tile * 16;
    __syncthreads();
    const int nst = ncols / (256 * S);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    float8 acc[TN];
    const _Float16 *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    gs_u32x4 rw[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) rw[e] = (gs_u32x4){0, 0, 0, 0};
    // (macro, not a lambda: a by-reference capture of rw pushes it to scratch)
#define GEMVH11_FETCH(js_)                                                                                   \
    do {                                                                                                     \
        _Pragma("unroll") for (int e = 0; e < WPER; ++e) {                                                   \
            const int idx = lane + e * WARP;                                                                 \
            if (WCH % WARP == 0 || idx < WCH) {                                                              \
                const size_t st = (size_t)min(m0 + idx / CH, nrows - 1) * row_bytes + (size_t)(js_) * SB;                                             \
                const size_t a = (st & ~(size_t)15) + (idx % CH) * 16;                                       \
                /* no 16-byte chunk wholly past the stage (the matrix may end there): zero-fill, never stale bytes */ \
                rw[e] = a < st + SB ? *(const gs_u32x4 *)(W + a) : (gs_u32x4){0, 0, 0, 0};            \
            }                                                                                                \
        }                                                                                                    \
    } while (0)
    GEMVH11_FETCH(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) *(gs_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        __syncthreads();
        if (js + 1 < nst) GEMVH11_FETCH(js + 1);
        const uint8_t *wp = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
        // (partial unroll: a full unroll lets the scheduler hoist every step's decode and X loads and spill)
#pragma unroll 2
        for (int kk = 0; kk < 16 * S; ++kk) {
            const half16 a = join_halves(dec_group_h<T>(wp, 2 * kk + lane / 16), lane);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                half16 b = ld_frag16(xr[j] + js * 256 * S + kk * 16);
                if (!xv[j]) b = (half16){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                acc[j] = WMMA16(b, a, acc[j]);
            }
        }
    }
#undef GEMVH11_FETCH
    const int orow = m0 + rl;
    if (orow < nrows)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = j * 16 + CROW(lane, v);
            if (t < n_tok) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
    }
}

// S = 2 (two 256-k stages per LDS round) when ncols % 512 == 0 and the type's stage is small; one LDS buffer
// per kernel, sized for the S used.
template <int T> constexpr int gv11_s2() { return gs_sbb<T>() <= 288 ? 2 : 1; }
#define GEMVH11_BODY(T, TN)                                                                                      constexpr int S2 = gv11_s2<T>();                                                                             __shared__ __attribute__((aligned(16))) uint8_t Ws[16 * 16 * ((S2 * gs_sbb<T>() + 15) / 16 + 1)];            if (S2 == 2 && ncols % 512 == 0) gemvh11_impl<T, TN, S2>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);     else gemvh11_impl<T, TN, 1>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
#define GEMVH11_TYPE(SUF, T)                                                                                                        extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                    int ncols, int nrows, int n_tok, int accumulate) {                   GEMVH11_BODY(T, 1)                                                                                                          }                                                                                                                               extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                     int ncols, int nrows, int n_tok, int accumulate) {                  GEMVH11_BODY(T, 2)                                                                                                          }

GEMVH11_TYPE(f16, QT_F16)
GEMVH11_TYPE(iq3_s, QT_IQ3_S)
