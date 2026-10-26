// WHIRL device code (gfx1201 / RDNA 4): bit-identical load-schedule candidates for the MXFP4 decode expert kernels
// moe_down_mxfp4w / moe_gu_mxfp4w (int8 activations: balance and fast) and moe_down_f_mxfp4 (f32 activations:
// precise) - feat026-moedown97, the gfx1201 counterpart of feat026-moedown8s (kernels/gfx1151/moe_down8.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/whirl_kernels.hip (gfx1201 block) right after moe_f.hip (uses mx_lut4 /
// dp4 of gemv_q.hip, e8m0_half of common.hip, mxf_lut4 / mxf_dot4 of moe_f.hip, warp_sum).
//
// Shipped kernels (ISA of the 1a5a547 object, research\moedown97 note):
//   moe_down_mxfp4w / moe_down_f_mxfp4: one wave per output row; lane l takes the 32-value blocks u = l, l + 32, ...
//     of the K experts' rows concatenated (k = u / nb, b = u - k nb, nb = F / 32). Per step: a VALU division, a
//     vector ids[p] load + s_wait_loadcnt before the weight address exists, then the unit loads + a second wait:
//     two serial memory round trips per 16 weight bytes per lane. Ornith F 512: nb 16, so one lane step covers two
//     experts (k is lane-varying: the gfx1151 wave-uniform trick does not apply).
//   moe_gu_mxfp4w: expert uniform; per step 5 loads, then waits down to loadcnt 0: one round trip per step.
// Candidates keep every value bit: the same lanes own the same units in the same increasing order, the same
// expression on the same loaded values (acc += w[p] * (scale * xd * (float)dot) / acc += w[p] * mxf_blk, gu:
// acc += scale * dt * (float)dot), the same warp_sum and epilogue expressions. Only where the loads sit changes:
//   IT : k / b advance exactly without a division per step (dk = 32 / nb, db = 32 - dk nb, one division per lane
//        at start): b += db, k += dk, if (b >= nb) { b -= nb; ++k; }  (u + 32 = (k + dk) nb + b + db, b + db < 2 nb).
//   KB : a batch of KB lane steps: first all KB ids / w loads, then all KB steps' unit loads, then the KB
//        computations (steps past the lane's last unit load a clamped, in-bounds unit and are not consumed).
//   NG : a wave-uniform full batch (every step valid for every lane) is consumed without the per-step guards;
//        with the guards the compiler sinks loads into the guarded blocks (dev ISA: drains per step again).
//   EP : the epilogue loads x[t][row], ysh[t][row], sg[t] issued at kernel start (lane 0); x_old + v == x += v.
// Tune ids (0 = shipped, the default; the default build is unchanged - no code is emitted from this file):
//   WHIRL_MD97_DN (moe_down_mxfp4w)  1 IT KB1   2 IT KB2   3 IT KB4   4 IT KB8   5 IT KB4 EP   6 IT KB2 EP
//                                    7 KB2 NG   8 KB4 NG   9 KB4 EP NG   10 KB8 NG
//   WHIRL_MD97_DF (moe_down_f_mxfp4) 1 IT KB1   2 IT KB2   3 IT KB4   4 IT KB2 EP   5 KB2 NG   6 KB2 EP NG
//   WHIRL_MD97_GU (moe_gu_mxfp4w)    1 KB2      2 KB4      3 KB2 NG   4 KB4 NG
// A non-zero id renames the shipped entry to <name>_shipped (whirl_kernels.hip, around its include) and defines
// the production entry here. WHIRL_MD97_DEV=1 exports every id as md97dn<id>_mxfp4w / md97df<id>_mxfp4 /
// md97gu<id>_mxfp4w (kernel-test family moedown97, bench --ab) and leaves the shipped entries as they are.

#ifndef WHIRL_MD97_DN
#define WHIRL_MD97_DN 0
#endif
#ifndef WHIRL_MD97_DF
#define WHIRL_MD97_DF 0
#endif
#ifndef WHIRL_MD97_GU
#define WHIRL_MD97_GU 0
#endif
#ifndef WHIRL_MD97_DEV
#define WHIRL_MD97_DEV 0
#endif

#if WHIRL_MD97_DN || WHIRL_MD97_DF || WHIRL_MD97_GU || WHIRL_MD97_DEV

// mxw_dot on loaded operands (same dp4 chain, same order)
__device__ __forceinline__ int md97_dot(const uint4 q, const int4 xl, const int4 xh) {
    int a = dp4(mx_lut4(q.x & 0x0F0F0F0F), xl.x, 0);
    a = dp4(mx_lut4(q.y & 0x0F0F0F0F), xl.y, a);
    a = dp4(mx_lut4(q.z & 0x0F0F0F0F), xl.z, a);
    a = dp4(mx_lut4(q.w & 0x0F0F0F0F), xl.w, a);
    a = dp4(mx_lut4((q.x >> 4) & 0x0F0F0F0F), xh.x, a);
    a = dp4(mx_lut4((q.y >> 4) & 0x0F0F0F0F), xh.y, a);
    a = dp4(mx_lut4((q.z >> 4) & 0x0F0F0F0F), xh.z, a);
    a = dp4(mx_lut4((q.w >> 4) & 0x0F0F0F0F), xh.w, a);
    return a;
}

// int8-activation unit (moe_down_mxfp4w): mxw_scale(wr, b) * xd[p nb + b] * (float)mxw_dot(wr, b, xq + p F)
struct Md97Q8 {
    struct Ld {
        uint4 q;
        int4 xl, xh;
        float d;
        uint32_t e;
    };
    const int8_t *__restrict__ xq;
    const float *__restrict__ xd;
    int F, nb;
    __device__ __forceinline__ void load(Ld &o, const uint8_t *__restrict__ wr, int p, int b) const {
        const uint8_t *sb = wr + (size_t)(b >> 3) * 136;
        const int ib = b & 7;
        o.e = sb[ib];
        o.q = *(const uint4 *)(sb + 8 + 16 * ib);
        const int8_t *xp = xq + (size_t)p * F + b * 32;
        o.xl = *(const int4 *)xp;
        o.xh = *(const int4 *)(xp + 16);
        o.d = xd[(size_t)p * nb + b];
    }
    __device__ __forceinline__ float eval(const Ld &o) const { return e8m0_half(o.e) * o.d * (float)md97_dot(o.q, o.xl, o.xh); }
};

// f32-activation unit (moe_down_f_mxfp4): mxf_blk(wr, b, act + p F) on loaded operands
struct Md97F {
    struct Ld {
        uint4 q;
        uint32_t e;
        float4 x[8];
    };
    const float *__restrict__ act;
    int F;
    __device__ __forceinline__ void load(Ld &o, const uint8_t *__restrict__ wr, int p, int b) const {
        const uint8_t *sb = wr + (size_t)(b >> 3) * 136;
        const int ib = b & 7;
        o.q = *(const uint4 *)(sb + 8 + 16 * ib);
        o.e = sb[ib];
        const float4 *xp = (const float4 *)(act + (size_t)p * F + b * 32);
#pragma unroll
        for (int j = 0; j < 8; ++j) o.x[j] = xp[j];
    }
    __device__ __forceinline__ float eval(const Ld &o) const {
        const float dl = e8m0_half(o.e);
        float s0 = 0.f, s1 = 0.f;
        s0 = mxf_dot4(mxf_lut4(o.q.x & 0x0F0F0F0Fu), o.x[0], s0);
        s1 = mxf_dot4(mxf_lut4((o.q.x >> 4) & 0x0F0F0F0Fu), o.x[4], s1);
        s0 = mxf_dot4(mxf_lut4(o.q.y & 0x0F0F0F0Fu), o.x[1], s0);
        s1 = mxf_dot4(mxf_lut4((o.q.y >> 4) & 0x0F0F0F0Fu), o.x[5], s1);
        s0 = mxf_dot4(mxf_lut4(o.q.z & 0x0F0F0F0Fu), o.x[2], s0);
        s1 = mxf_dot4(mxf_lut4((o.q.z >> 4) & 0x0F0F0F0Fu), o.x[6], s1);
        s0 = mxf_dot4(mxf_lut4(o.q.w & 0x0F0F0F0Fu), o.x[3], s0);
        s1 = mxf_dot4(mxf_lut4((o.q.w >> 4) & 0x0F0F0F0Fu), o.x[7], s1);
        return dl * (s0 + s1);
    }
};

// down + combine: x[t][row] += sum over the lane's units of w[p] * unit (shipped order) (+ sg[t] * ysh[t][row])
template <class U, int KB, bool EP, bool NG>
__device__ void md97_down_impl(const U &un, const uint8_t *__restrict__ W, uint64_t row_bytes, int E,
                               const int *__restrict__ ids, const float *__restrict__ w, const float *__restrict__ sg,
                               const float *__restrict__ ysh, float *__restrict__ x, int F, int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    float x_old = 0.f, ys = 0.f, sgt = 0.f;
    if constexpr (EP) {
        if (lane == 0) {
            x_old = x[(size_t)t * E + row];
            if (ysh != nullptr) {
                sgt = sg[t];
                ys = ysh[(size_t)t * E + row];
            }
        }
    }
    const int nb = F / 32;
    const int nu = K * nb;
    float acc = 0.f;
    if (nu > 0) {
        const int dk = WARP / nb, db = WARP - dk * nb;  // wave-uniform
        int k = lane / nb, b = lane - k * nb;           // unit u = lane (one division, at start)
        for (int u0 = 0; u0 < nu; u0 += KB * WARP) {    // wave-uniform batches; step s: u = lane + u0 + 32 s
            int pk[KB], bb[KB];
            bool ok[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) {
                ok[s] = k < K;  // == (lane + u0 + 32 s < nu)
                pk[s] = t * K + (ok[s] ? k : K - 1);  // clamped: in bounds, consumed only if ok
                bb[s] = b;
                b += db;
                k += dk;
                if (b >= nb) {
                    b -= nb;
                    ++k;
                }
            }
            int e[KB];
            float wp[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) {
                e[s] = ids[pk[s]];
                wp[s] = w[pk[s]];
            }
            typename U::Ld o[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) un.load(o[s], W + ((size_t)e[s] * E + row) * row_bytes, pk[s], bb[s]);
            if (NG && u0 + KB * WARP <= nu) {  // wave-uniform full batch: every ok[s] is true, no guard branches
#pragma unroll
                for (int s = 0; s < KB; ++s) acc += wp[s] * un.eval(o[s]);
            } else {
#pragma unroll
                for (int s = 0; s < KB; ++s)
                    if (ok[s]) acc += wp[s] * un.eval(o[s]);
            }
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if constexpr (EP) {
            if (ysh != nullptr) v += sgt * ys;
            x[(size_t)t * E + row] = x_old + v;
        } else {
            if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
            x[(size_t)t * E + row] += v;
        }
    }
}

// gate / up (moe_gu_mxfp4w): KB lane steps' unit loads issued before the first of them is used
template <int KB, bool NG>
__device__ void md97_gu_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff,
                             int nrows, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                             const int *__restrict__ ids, float *__restrict__ y1, float *__restrict__ y2, int C, int K) {
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / K, e = ids[p];
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    const int nbk = C / 32;
    float acc = 0.f;
    for (int b0 = lane; b0 < nbk; b0 += KB * WARP) {
        Md97Q8::Ld o[KB];
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            const int bs = b0 + WARP * s;
            const int b = bs < nbk ? bs : b0;  // clamped (b0 < nbk): in bounds, consumed only if bs < nbk
            const uint8_t *sb = wr + (size_t)(b >> 3) * 136;
            const int ib = b & 7;
            o[s].e = sb[ib];
            o[s].q = *(const uint4 *)(sb + 8 + 16 * ib);
            o[s].xl = *(const int4 *)(xt + b * 32);
            o[s].xh = *(const int4 *)(xt + b * 32 + 16);
            o[s].d = dt[b];
        }
        if (NG && b0 - lane + KB * WARP <= nbk) {  // wave-uniform full batch: no guard branches
#pragma unroll
            for (int s = 0; s < KB; ++s) acc += e8m0_half(o[s].e) * o[s].d * (float)md97_dot(o[s].q, o[s].xl, o[s].xh);
        } else {
#pragma unroll
            for (int s = 0; s < KB; ++s)
                if (b0 + WARP * s < nbk) acc += e8m0_half(o[s].e) * o[s].d * (float)md97_dot(o[s].q, o[s].xl, o[s].xh);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

#define MD97_DN_SIG(NAME)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        NAME(const uint8_t *W, uint64_t row_bytes, int E, const int8_t *xq, const float *xd, const int *ids,      \
             const float *w, const float *sg, const float *ysh, float *x, int F, int K, const int *skip)
#define MD97_DN_BODY(KB, EP, NG)                                                                                     \
    {                                                                                                              \
        if (skip != nullptr && *skip) return;                                                                      \
        const Md97Q8 un{xq, xd, F, F / 32};                                                                        \
        md97_down_impl<Md97Q8, KB, EP, NG>(un, W, row_bytes, E, ids, w, sg, ysh, x, F, K);                             \
    }
#define MD97_DF_SIG(NAME)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        NAME(const uint8_t *W, uint64_t row_bytes, int E, const float *act, const int *ids, const float *w,       \
             const float *sg, const float *ysh, float *x, int F, int K)
#define MD97_DF_BODY(KB, EP, NG)                                                                                     \
    {                                                                                                              \
        const Md97F un{act, F};                                                                                    \
        md97_down_impl<Md97F, KB, EP, NG>(un, W, row_bytes, E, ids, w, sg, ysh, x, F, K);                              \
    }
#define MD97_GU_SIG(NAME)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256)                                                              \
        NAME(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, int ff, int nrows, const int8_t *xq,       \
             const float *xd, const int *ids, float *y1, float *y2, int C, int K, const int *skip)
#define MD97_GU_BODY(KB, NG)                                                                                         \
    {                                                                                                              \
        if (skip != nullptr && *skip) return;                                                                      \
        md97_gu_impl<KB, NG>(W1, W2, row_bytes, ff, nrows, xq, xd, ids, y1, y2, C, K);                                 \
    }
#define MD97_DN_ID1 MD97_DN_BODY(1, false, false)
#define MD97_DN_ID2 MD97_DN_BODY(2, false, false)
#define MD97_DN_ID3 MD97_DN_BODY(4, false, false)
#define MD97_DN_ID4 MD97_DN_BODY(8, false, false)
#define MD97_DN_ID5 MD97_DN_BODY(4, true, false)
#define MD97_DN_ID6 MD97_DN_BODY(2, true, false)
#define MD97_DN_ID7 MD97_DN_BODY(2, false, true)
#define MD97_DN_ID8 MD97_DN_BODY(4, false, true)
#define MD97_DN_ID9 MD97_DN_BODY(4, true, true)
#define MD97_DN_ID10 MD97_DN_BODY(8, false, true)
#define MD97_DF_ID1 MD97_DF_BODY(1, false, false)
#define MD97_DF_ID2 MD97_DF_BODY(2, false, false)
#define MD97_DF_ID3 MD97_DF_BODY(4, false, false)
#define MD97_DF_ID4 MD97_DF_BODY(2, true, false)
#define MD97_DF_ID5 MD97_DF_BODY(2, false, true)
#define MD97_DF_ID6 MD97_DF_BODY(2, true, true)
#define MD97_GU_ID1 MD97_GU_BODY(2, false)
#define MD97_GU_ID2 MD97_GU_BODY(4, false)
#define MD97_GU_ID3 MD97_GU_BODY(2, true)
#define MD97_GU_ID4 MD97_GU_BODY(4, true)
#define MD97_CAT2(a, b) a##b
#define MD97_CAT(a, b) MD97_CAT2(a, b)

#if WHIRL_MD97_DEV
MD97_DN_SIG(md97dn1_mxfp4w) MD97_DN_ID1
MD97_DN_SIG(md97dn2_mxfp4w) MD97_DN_ID2
MD97_DN_SIG(md97dn3_mxfp4w) MD97_DN_ID3
MD97_DN_SIG(md97dn4_mxfp4w) MD97_DN_ID4
MD97_DN_SIG(md97dn5_mxfp4w) MD97_DN_ID5
MD97_DN_SIG(md97dn6_mxfp4w) MD97_DN_ID6
MD97_DN_SIG(md97dn7_mxfp4w) MD97_DN_ID7
MD97_DN_SIG(md97dn8_mxfp4w) MD97_DN_ID8
MD97_DN_SIG(md97dn9_mxfp4w) MD97_DN_ID9
MD97_DN_SIG(md97dn10_mxfp4w) MD97_DN_ID10
MD97_DF_SIG(md97df1_mxfp4) MD97_DF_ID1
MD97_DF_SIG(md97df2_mxfp4) MD97_DF_ID2
MD97_DF_SIG(md97df3_mxfp4) MD97_DF_ID3
MD97_DF_SIG(md97df4_mxfp4) MD97_DF_ID4
MD97_DF_SIG(md97df5_mxfp4) MD97_DF_ID5
MD97_DF_SIG(md97df6_mxfp4) MD97_DF_ID6
MD97_GU_SIG(md97gu1_mxfp4w) MD97_GU_ID1
MD97_GU_SIG(md97gu2_mxfp4w) MD97_GU_ID2
MD97_GU_SIG(md97gu3_mxfp4w) MD97_GU_ID3
MD97_GU_SIG(md97gu4_mxfp4w) MD97_GU_ID4
#endif

// production entries with the selected ids (whirl_kernels.hip renamed the shipped ones to *_shipped)
#if WHIRL_MD97_DN
MD97_DN_SIG(moe_down_mxfp4w) MD97_CAT(MD97_DN_ID, WHIRL_MD97_DN)
#endif
#if WHIRL_MD97_DF
MD97_DF_SIG(moe_down_f_mxfp4) MD97_CAT(MD97_DF_ID, WHIRL_MD97_DF)
#endif
#if WHIRL_MD97_GU
MD97_GU_SIG(moe_gu_mxfp4w) MD97_CAT(MD97_GU_ID, WHIRL_MD97_GU)
#endif

#endif  // WHIRL_MD97_DN || WHIRL_MD97_DF || WHIRL_MD97_GU || WHIRL_MD97_DEV
