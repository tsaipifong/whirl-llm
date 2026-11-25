// WHIRL device code (gfx1151 / RDNA 3.5): v0252-e1s8 (PLAN V1-93 P2a) the DeltaNet SELECT mode of the E1 one-cycle lookahead
// in snapshot mode, as entries of its own. SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST by kernels/gfx1151/whirl_kernels_gfx1151.hip (no existing kernel moves).
//
// Round 2 (2026-11-25): round 1 put the select branch into the shipping gdn_conv_l2_impl / gdn_step_norm, which changed the
// code of 11 shipping entries (gdn_conv_l2, gdn_step_norm, gdn_abconv_*) and moved a default-path cell (8060S precise plain
// -0.28 %). Now the shipping entries are the integration build's source again and the select verify launches these entries
// only (Context::gdnLayer, when the segment table selects). Their GDN bodies are PRIVATE copies (gs_*), not new
// instantiations of the shipping helpers (v0252-8sws: a shared helper instantiated by one more entry changed the shipping
// entries' code); the arithmetic of every row is token for token the shipping one (kt e1in "select" rows: bitwise equal to the
// shipping in-place launch from a copy of the selected candidate).
//
// Select segment (kernels/gdn_select.h; GdnSeg of kernels/kernel_args.h): pend = the sequence's NPEND control word (acc + 1 of
// the verify before), npend = m, snap[0 .. m] = the candidates (that verify's snapshot sets, then its live state), state = the
// spare set. Input = candidate c = gdn_sel_pick(*pend, m); snapshot of local row t (t < nrows - 1) -> candidate gdn_sel_out(t, c);
// final state -> state. Nothing reads or writes another buffer; the candidate read is never written.
//
// whirl_cap_gdn_select   this code object has gdn_conv_l2_sel, gdn_abconv_sel_<T>, gdn_step_norm_sel (Caps::gdn_select)
#include "gdn_select.h"

extern "C" __global__ void whirl_cap_gdn_select() {}

// the snapshot destination of local row t of a select segment (null: none)
__device__ __forceinline__ float *gs_snap_dst(const GdnSeg &sg, int t, int c, int n_tok) {
    const int j = gdn_sel_out(t, c);
    return t + 1 < n_tok && j <= sg.npend ? sg.snap[j] : nullptr;
}

// conv step + SiLU + L2 norm (a copy of fused_decode.hip gdn_conv_l2_impl, select-mode pointers)
__device__ __forceinline__ void gs_conv_l2_impl(const float *xin, const float *w, float *out, int ch, int n_qk_heads, float eps,
                                                const GdnSegs &segs, int cb, int seg, bool act) {
    const GdnSeg &sg = segs.s[seg];
    float *state = sg.state;
    const int r0 = sg.row0, n_tok = sg.nrows;
    const int c = cb * 128 + (act ? (int)threadIdx.x : 0);
    const int sel = gdn_sel_pick(*(const int *)sg.pend, sg.npend);
    const float *src = sg.snap[sel];
    float h0 = 0.f, h1 = 0.f, h2 = 0.f;
    float4 wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (act) {
        h0 = src[c];
        h1 = src[(size_t)ch + c];
        h2 = src[(size_t)2 * ch + c];
        wv = *(const float4 *)(w + (size_t)c * 4);
    }
    for (int t = 0; t < n_tok; ++t) {
        float v = 0.f;
        if (act) {
            const float xn = xin[(size_t)(r0 + t) * ch + c];
            // explicit fmas = the contraction of the shipping kernel (bitwise the same)
            const float acc = __builtin_fmaf(xn, wv.w, __builtin_fmaf(h2, wv.z, __builtin_fmaf(h1, wv.y, h0 * wv.x)));
            h0 = h1;
            h1 = h2;
            h2 = xn;
            float *sp = gs_snap_dst(sg, t, sel, n_tok);
            if (sp != nullptr) {
                sp[c] = h0;
                sp[(size_t)ch + c] = h1;
                sp[(size_t)2 * ch + c] = h2;
            }
            v = acc / (1.f + __expf(-acc));
        }
        if (cb < n_qk_heads) {
            const float ss = block_sum(v * v);
            v *= rsqrtf(ss + eps);
        }
        if (act) out[(size_t)(r0 + t) * ch + c] = v;
    }
    if (act) {
        state[c] = h0;
        state[(size_t)ch + c] = h1;
        state[(size_t)2 * ch + c] = h2;
    }
}

extern "C" __global__ void __launch_bounds__(128)
    gdn_conv_l2_sel(const float *xin, const float *w, float *out, int ch, int n_qk_heads, float eps, GdnSegs segs) {
    gs_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, blockIdx.x, blockIdx.y, true);
}

// beta / alpha projection (a copy of shared/fused_decode_rmsnorm_q8.h gdn_ab_impl)
template <int T>
__device__ void gs_ab_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq, const float *xd, float *beta,
                           float *g, const float *dt, const float *A, int ncols, int nh, int blk) {
    const int lane = threadIdx.x % WARP;
    const int row = blk * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= 2 * nh) return;
    const bool is_b = row < nh;
    const int h = is_b ? row : row - nh;
    const uint8_t *wr = (is_b ? Wb : Wa) + (size_t)h * row_bytes;
    const int nu = ncols / QD<T>::sb * QD<T>::per_sb;
    float acc = 0.f;
    for (int u = lane; u < nu; u += WARP) acc += QD<T>::dot(wr, u, xq, xd);
    acc = warp_sum(acc);
    if (lane == 0) {
        if (is_b) {
            beta[h] = 1.f / (1.f + __expf(-acc));
        } else {
            const float z = acc + dt[h];
            g[h] = (z > 20.f ? z : log1pf(__expf(z))) * A[h];
        }
    }
}

// gdn_ab + the select conv step in one launch (a copy of shared/gdn_ab_entries.h gdn_abconv_impl)
template <int T>
__device__ void gs_abconv_impl(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq, const float *xd, float *beta,
                               float *g, const float *dt, const float *A, int ncols, int nh, int n_ab, const float *xin, const float *w,
                               float *out, int ch, int n_qk_heads, float eps, const GdnSegs &segs) {
    const int ab_blocks = (2 * nh + 7) / 8;
    int b = blockIdx.x;
    if (b < n_ab * ab_blocks) {
        const int t = b / ab_blocks;
        gs_ab_impl<T>(Wb, Wa, row_bytes, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32), beta + (size_t)t * nh, g + (size_t)t * nh,
                      dt, A, ncols, nh, b % ab_blocks);
        return;
    }
    b -= n_ab * ab_blocks;
    const int nblk = ch / 128;
    gs_conv_l2_impl(xin, w, out, ch, n_qk_heads, eps, segs, b % nblk, b / nblk, threadIdx.x < 128);
}
#define GDN_ABCONV_SEL_ENTRY(SUF, T)                                                                                               \
    extern "C" __global__ void __launch_bounds__(256)                                                                            \
        gdn_abconv_sel_##SUF(const uint8_t *Wb, const uint8_t *Wa, uint64_t row_bytes, const int8_t *xq, const float *xd,        \
                             float *beta, float *g, const float *dt, const float *A, int ncols, int nh, int n_ab, const float *xin, \
                             const float *w, float *out, int ch, int n_qk_heads, float eps, GdnSegs segs) {                       \
        gs_abconv_impl<T>(Wb, Wa, row_bytes, xq, xd, beta, g, dt, A, ncols, nh, n_ab, xin, w, out, ch, n_qk_heads, eps, segs);  \
    }
GDN_ABCONV_SEL_ENTRY(q8_0, QT_Q8_0)
GDN_ABCONV_SEL_ENTRY(q4_k, QT_Q4_K)
GDN_ABCONV_SEL_ENTRY(q5_k, QT_Q5_K)
GDN_ABCONV_SEL_ENTRY(q6_k, QT_Q6_K)
GDN_ABCONV_SEL_ENTRY(iq4_xs, QT_IQ4_XS)
GDN_ABCONV_SEL_ENTRY(mxfp4, QT_MXFP4)
GDN_ABCONV_SEL_ENTRY(iq4_nl, QT_IQ4_NL)
GDN_ABCONV_SEL_ENTRY(q3_k, QT_Q3_K)
GDN_ABCONV_SEL_ENTRY(iq3_s, QT_IQ3_S)
#undef GDN_ABCONV_SEL_ENTRY

// delta rule step + gated RMSNorm + int8 quantization (a copy of fused_decode.hip gdn_step_norm, select-mode pointers)
extern "C" __global__ void __launch_bounds__(512)
    gdn_step_norm_sel(const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out, int8_t *xq,
                      float *xd, int n_k_heads, float scale, float eps, int n_v_heads, int qkv_stride, GdnSegs segs) {
    constexpr int DK = 128, DV = 128, QI = 4, PER = DK / QI;
    const GdnSeg &sgm = segs.s[blockIdx.y];
    const int r0 = sgm.row0, n_tok = sgm.nrows;
    const int sel = gdn_sel_pick(*(const int *)sgm.pend, sgm.npend);
    const int h = blockIdx.x;
    const int j = threadIdx.x % DV, qi = threadIdx.x / DV;
    const int kh = h % n_k_heads;
    const int d_inner = n_v_heads * DV;
    __shared__ float ks[DK], qs[DK];
    __shared__ float red[QI][DV];
    __shared__ float ov[DV];
    float *Sg = sgm.state + (size_t)h * DK * DV;
    const float *Si = sgm.snap[sel] + (size_t)h * DK * DV;
    float S[PER];
#pragma unroll
    for (int e = 0; e < PER; ++e) S[e] = Si[(size_t)(qi * PER + e) * DV + j];
    for (int tl = 0; tl < n_tok; ++tl) {
        const int t = r0 + tl;
        const float *row = qkv + (size_t)t * qkv_stride;
        __syncthreads();
        if (threadIdx.x < DK) {
            qs[threadIdx.x] = row[(size_t)kh * DK + threadIdx.x];
            ks[threadIdx.x] = row[(size_t)n_k_heads * DK + (size_t)kh * DK + threadIdx.x];
        }
        __syncthreads();
        const float decay = __expf(g[(size_t)t * n_v_heads + h]);
        float sk = 0.f;
#pragma unroll
        for (int e = 0; e < PER; ++e) {
            S[e] *= decay;
            sk += S[e] * ks[qi * PER + e];
        }
        red[qi][j] = sk;
        __syncthreads();
        sk = red[0][j] + red[1][j] + red[2][j] + red[3][j];
        const float v = row[(size_t)2 * n_k_heads * DK + (size_t)h * DV + j];
        const float delta = (v - sk) * beta[(size_t)t * n_v_heads + h];
        float o = 0.f;
#pragma unroll
        for (int e = 0; e < PER; ++e) {
            S[e] += ks[qi * PER + e] * delta;
            o += S[e] * qs[qi * PER + e];
        }
        float *sp = gs_snap_dst(sgm, tl, sel, n_tok);
        if (sp != nullptr) {
            float *Ss = sp + (size_t)h * DK * DV;
#pragma unroll
            for (int e = 0; e < PER; ++e) Ss[(size_t)(qi * PER + e) * DV + j] = S[e];
        }
        __syncthreads();
        red[qi][j] = o;
        __syncthreads();
        if (qi == 0) ov[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * scale;
        __syncthreads();
        float y = 0.f;
        if (threadIdx.x < DV) y = ov[threadIdx.x];
        const float ss = block_sum(threadIdx.x < DV ? y * y : 0.f);
        if (threadIdx.x < DV) {
            const float r = rsqrtf(ss / DV + eps);
            const float zz = z[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x];
            y = y * r * wn[threadIdx.x] * (zz / (1.f + __expf(-zz)));
            out[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x] = y;
            wave_q8(y, xq + (size_t)t * d_inner, xd + (size_t)t * (d_inner / 32), (h * DV + threadIdx.x) >> 5);
        }
    }
#pragma unroll
    for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
}
