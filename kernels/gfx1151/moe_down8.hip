// WHIRL device code (gfx1151 / RDNA 3.5): bit-identical load-schedule candidates for the MXFP4 int8 decode
// down projection + expert combine (moe_down_mxfp4; feat026-moedown8s, KERNEL_TECHNIQUE_AUDIT row 14).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip right after moe.hip.
//
// Shipped kernel (moe_down_impl<QT_MXFP4>, moe.hip): one wave per output row; lane l takes the 16-value units
// u = l, l + 32, ... of the K experts' rows concatenated (k = u / nu, nu = F / 16), per unit
//     a = 0 + dl * xd * int_dot;  acc += w[p] * a        (QDN<QT_MXFP4>::dot<1>, then acc += w[p] * a)
// then warp_sum and lane 0 writes x[t][row] += acc (+ sg[t] * ysh[t][row]). Its loop issues per unit a dependent
// ids[p] load, waits vmcnt(0), then the unit loads and waits vmcnt(0) again: two serial memory round trips per
// 8 weight bytes per lane, one unit in flight (scratch ISA, research note).
//
// Candidates keep every value bit: the same lanes own the same units, each lane runs the same arithmetic on the
// same values in the same increasing u order (md8_unit = the expression of QD<QT_MXFP4>::dot), the same warp_sum
// and the same epilogue expressions. Only where the loads sit changes:
//   UNI : when nu % 32 == 0 (Ornith F = 512: nu = 32) the expert index k of a lane step is wave-uniform:
//         u = lane + 32 i  ->  k = i / m, j = i % m (m = nu / 32), so ids[p] / w[p] are uniform (scalar) loads
//         and the per-step integer division is gone. Other shapes run the shipped body (same code).
//   KB  : unit loads issued in batches of KB steps before the batch's first use (KB = 1: no lookahead).
//   DB  : double buffer: batch b + 1's loads are issued before batch b is consumed.
//   EP  : the epilogue's x[t][row], ysh[t][row], sg[t] loads issued at kernel start (lane 0).
//   LX  : the K experts' int8 activations of token t (+ scale words) staged once per block in LDS (8 waves share
//         them); same bytes, so the same values. Needs K * F <= 8192 and F % 16 == 0, else the global path.
// Tune table WHIRL_MD8_MXFP4 = id (0 = shipped, the default; the default build is unchanged). WHIRL_MD8_DEV=1
// exports every id as md8d<id>_mxfp4 for the kernel test / microbench (moe_down_mxfp4 stays the shipped kernel).
//   NG  : a full batch (i0 + KB <= ns, wave-uniform) is consumed without the per-step i < ns guards; the tail
//         batch keeps them. Same values (every guard of a full batch is true).
//   id  UNI KB DB EP LX NG
//   1   1   1  0  0  0  0     uniform k only
//   2   1   4  0  0  0  0
//   3   1   8  0  0  0  0     Ornith: all 8 steps' loads first
//   4   1   4  1  0  0  0
//   5   1   8  0  1  0  0
//   6   1   8  0  0  1  0
//   7   1   4  1  1  0  0
//   8   1   8  0  1  1  0
//   9   1   8  0  0  0  1
//   10  1   8  0  1  0  1
//   11  1   8  0  1  1  1
//   12  1   4  0  1  0  1
// moeskip contract (feat027-moeskip, fast only; feat026-moe-skipsafe): a token's slot ids are a kept prefix, then -1
// (dropped, weight 0). The candidates run the lane steps of the kept prefix only, kt = md8_kept (the loop of feat027's
// moe_kept: K when no id is -1): ns = kt * m, steps past it clamped to step ns - 1 (a kept slot), so no load uses an
// id -1; the lane -> unit map and order are those of feat027's moe_down_impl loop over kt * nu. Without -1 kt == K
// (unchanged values). The nu % 32 != 0 fallback is the shipped body, skip-aware once feat027 is merged. The LX stage
// still copies all K rows of token t (in bounds; dropped rows are never consumed).
// Test-only id -1 address check (feat026-skipsafe-r2, review skipsafe-gemvports C1; needs WHIRL_MD8_DEV=1):
//   WHIRL_MD8_NEGCHK=1: the expert id of every weight-row address of md8_impl goes through MD8_EID; an id < 0
//     increments the device global md8_negid_hits (kt reads it) and is replaced by 0, so even a defective candidate
//     never loads outside the tensor. A value check cannot see a load that is masked out; this counter can.
//   WHIRL_MD8_NEGCHK=2: the same plus a planted fault (the positive control): steps past the kept prefix are clamped
//     to step K m - 1 (slot K - 1, an id -1 slot). kt must count hits for every candidate.
//   Without it MD8_EID(e) is (e) and the clamp is ns - 1: the dev / tune-id / default objects are unchanged.
//   The nu % 32 != 0 fallback (the shipped body) is not instrumented.

#ifndef WHIRL_MD8_MXFP4
#define WHIRL_MD8_MXFP4 0
#endif
#ifndef WHIRL_MD8_DEV
#define WHIRL_MD8_DEV 0
#endif
#ifndef WHIRL_MD8_NEGCHK
#define WHIRL_MD8_NEGCHK 0
#endif
#if WHIRL_MD8_NEGCHK && !WHIRL_MD8_DEV
#error "WHIRL_MD8_NEGCHK is a test instrumentation of the WHIRL_MD8_DEV object only"
#endif

#if WHIRL_MD8_MXFP4 || WHIRL_MD8_DEV

// Loaded operands of one 16-value MXFP4 unit (QD<QT_MXFP4>::dot's loads).
struct Md8Ld {
    uint2 q;     // 8 weight bytes (16 nibbles)
    int2 xl, xh;  // int8 activations base .. +7, base + 16 .. +23
    uint32_t xw;  // activation scale word of the 32-block
    uint32_t e;   // E8M0 exponent byte of the block
};

template <bool LX>
__device__ __forceinline__ void md8_load(Md8Ld &o, const uint8_t *__restrict__ wr, int uu, const int8_t *__restrict__ xt,
                                         const float *__restrict__ dt, const int8_t *sx, const uint32_t *sd, int koff) {
    const int b = uu >> 4, ib = (uu >> 1) & 7, hh = uu & 1;
    const uint8_t *blk = wr + (size_t)b * 136;
    o.e = blk[ib];
    o.q = *(const uint2 *)(blk + 8 + 16 * ib + 8 * hh);
    const int base = b * 256 + 32 * ib + 8 * hh;
    if constexpr (LX) {
        o.xl = *(const int2 *)(sx + koff + base);
        o.xh = *(const int2 *)(sx + koff + base + 16);
        o.xw = sd[(koff >> 5) + ((b * 256 + 32 * ib) >> 5)];
    } else {
        o.xl = *(const int2 *)(xt + base);
        o.xh = *(const int2 *)(xt + base + 16);
        o.xw = __float_as_uint(dt[(b * 256 + 32 * ib) >> 5]);
    }
}

// QD<QT_MXFP4>::dot on loaded operands: the same expression (dl * xdv(...) * acc).
__device__ __forceinline__ float md8_unit(const Md8Ld &o) {
    const float dl = e8m0_half(o.e);
    int acc = dp4(mx_lut4(o.q.x & 0x0F0F0F0F), o.xl.x, 0);
    acc = dp4(mx_lut4(o.q.y & 0x0F0F0F0F), o.xl.y, acc);
    acc = dp4(mx_lut4((o.q.x >> 4) & 0x0F0F0F0F), o.xh.x, acc);
    acc = dp4(mx_lut4((o.q.y >> 4) & 0x0F0F0F0F), o.xh.y, acc);
    return dl * pk_d(o.xw) * acc;
}

// kept slots of token t (moeskip: kept prefix, then -1); == feat027 moe_kept, K without moeskip
__device__ __forceinline__ int md8_kept(const int *__restrict__ ids, int t, int K) {
    int kt = K;
    while (kt > 1 && ids[(size_t)t * K + kt - 1] < 0) --kt;
    return kt;
}

#if WHIRL_MD8_NEGCHK
// id -1 address check (test-only): counts weight addresses formed from a negative id, then uses expert 0 instead
extern "C" __device__ unsigned int md8_negid_hits = 0;
extern "C" __global__ void md8_negchk_v1() {}
#if WHIRL_MD8_NEGCHK >= 2
extern "C" __global__ void md8_negchk_fault_v1() {}  // kt: this object carries the planted fault
#endif
__device__ __forceinline__ int md8_eid(int e) {
    if (e < 0) {
        atomicAdd(&md8_negid_hits, 1u);
        return 0;
    }
    return e;
}
#define MD8_EID(e) md8_eid(e)
#else
#define MD8_EID(e) (e)
#endif
#if WHIRL_MD8_NEGCHK >= 2
#define MD8_CLAMP_STEP(ns, K, m) ((K) * (m) - 1)  // planted fault (positive control): a dropped slot's step
#else
#define MD8_CLAMP_STEP(ns, K, m) ((ns) - 1)
#endif

template <int KB, bool DB, bool EP, bool LX, bool NG = false>
__device__ void md8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const int8_t *__restrict__ xq,
                         const float *__restrict__ xd, const int *__restrict__ ids, const float *__restrict__ w,
                         const float *__restrict__ sg, const float *__restrict__ ysh, float *__restrict__ x, int F,
                         int K) {
    const int nu = F / 16;  // QD<QT_MXFP4>: 16 units per 256 values
    if ((nu % WARP) != 0 || K < 1) {
        moe_down_impl<QT_MXFP4>(W, row_bytes, E, xq, xd, ids, w, sg, ysh, x, F, K);  // shipped body
        return;
    }
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int t = blockIdx.y;
    const bool live = row < E;
    constexpr int LXB = 8192;  // LDS activation bytes (K * F <= LXB)
    __shared__ __attribute__((aligned(16))) int8_t sx[LX ? LXB : 16];
    __shared__ uint32_t sd[LX ? LXB / 32 : 4];
    const bool use_lx = LX && K * F <= LXB;
    if constexpr (LX) {
        if (use_lx) {  // block-uniform
            const int nb = K * F;  // bytes of token t's K activation rows: xq[t * K * F ..)
            const int8_t *src = xq + (size_t)t * K * F;
            for (int i = threadIdx.x * 16; i < nb; i += blockDim.x * 16) *(int4 *)(sx + i) = *(const int4 *)(src + i);
            const float *dsrc = xd + (size_t)t * K * (F / 32);
            for (int i = threadIdx.x; i < nb / 32; i += blockDim.x) sd[i] = __float_as_uint(dsrc[i]);
            __syncthreads();
        }
    }
    if (!live) return;
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
    const int m = nu / WARP;  // lane steps per expert
    const int ns = md8_kept(ids, t, K) * m;  // lane steps of the kept slots (moeskip; K * m without it)
    float acc = 0.f;
    // step i of this lane: expert slot k = i / m, unit uu = lane + 32 (i % m)  (== shipped u = lane + 32 i)
    auto load = [&](Md8Ld &o, float &wp, int i) {
        const int ic = i < ns ? i : MD8_CLAMP_STEP(ns, K, m);  // clamped: loads stay in bounds, consumed only if i < ns
        const int k = ic / m, j = ic - k * m;
        const int p = t * K + k;
        wp = w[p];
        const uint8_t *wr = W + ((size_t)MD8_EID(ids[p]) * E + row) * row_bytes;
        if (LX && use_lx) md8_load<true>(o, wr, lane + WARP * j, nullptr, nullptr, sx, sd, k * F);
        else md8_load<false>(o, wr, lane + WARP * j, xq + (size_t)p * F, xd + (size_t)p * (F / 32), nullptr, nullptr, 0);
    };
    if constexpr (!DB) {
        for (int i0 = 0; i0 < ns; i0 += KB) {
            Md8Ld o[KB];
            float wp[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) load(o[s], wp[s], i0 + s);
            if (NG && i0 + KB <= ns) {  // wave-uniform: a full batch, no per-step guard (no branches between the
                                        // loads and their use, so nothing is sunk into the guarded blocks)
#pragma unroll
                for (int s = 0; s < KB; ++s) {
                    float a = 0.f;
                    a += md8_unit(o[s]);
                    acc += wp[s] * a;
                }
            } else {
#pragma unroll
                for (int s = 0; s < KB; ++s)
                    if (i0 + s < ns) {
                        float a = 0.f;
                        a += md8_unit(o[s]);
                        acc += wp[s] * a;
                    }
            }
        }
    } else {
        Md8Ld o0[KB], o1[KB];
        float w0[KB], w1[KB];
#pragma unroll
        for (int s = 0; s < KB; ++s) load(o0[s], w0[s], s);
        for (int i0 = 0; i0 < ns; i0 += 2 * KB) {
#pragma unroll
            for (int s = 0; s < KB; ++s) load(o1[s], w1[s], i0 + KB + s);
#pragma unroll
            for (int s = 0; s < KB; ++s)
                if (i0 + s < ns) {
                    float a = 0.f;
                    a += md8_unit(o0[s]);
                    acc += w0[s] * a;
                }
            if (i0 + KB >= ns) break;
            if (i0 + 2 * KB < ns) {  // wave-uniform: next pair's first batch
#pragma unroll
                for (int s = 0; s < KB; ++s) load(o0[s], w0[s], i0 + 2 * KB + s);
            }
#pragma unroll
            for (int s = 0; s < KB; ++s)
                if (i0 + KB + s < ns) {
                    float a = 0.f;
                    a += md8_unit(o1[s]);
                    acc += w1[s] * a;
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

#define MD8_BODY(KB, DB, EP, LX, NG)                                                                               \
    {                                                                                                          \
        if (skip != nullptr && *skip) return;                                                                  \
        md8_impl<KB, DB, EP, LX, NG>(W, row_bytes, E, xq, xd, ids, w, sg, ysh, x, F, K);                           \
    }
#define MD8_SIG(NAME)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t *W, uint64_t row_bytes, int E, const int8_t *xq, \
                                                           const float *xd, const int *ids, const float *w,     \
                                                           const float *sg, const float *ysh, float *x, int F, int K, \
                                                           const int *skip)
#define MD8_ID1 MD8_BODY(1, false, false, false, false)
#define MD8_ID2 MD8_BODY(4, false, false, false, false)
#define MD8_ID3 MD8_BODY(8, false, false, false, false)
#define MD8_ID4 MD8_BODY(4, true, false, false, false)
#define MD8_ID5 MD8_BODY(8, false, true, false, false)
#define MD8_ID6 MD8_BODY(8, false, false, true, false)
#define MD8_ID7 MD8_BODY(4, true, true, false, false)
#define MD8_ID8 MD8_BODY(8, false, true, true, false)
#define MD8_ID9 MD8_BODY(8, false, false, false, true)
#define MD8_ID10 MD8_BODY(8, false, true, false, true)
#define MD8_ID11 MD8_BODY(8, false, true, true, true)
#define MD8_ID12 MD8_BODY(4, false, true, false, true)
#define MD8_CAT2(a, b) a##b
#define MD8_CAT(a, b) MD8_CAT2(a, b)

// skip-safety marker (feat026-moe-skipsafe): every object with md8 code has id -1 handling; kt family moedown8
// refuses the id -1 cases for an object without it (built before this change)
extern "C" __global__ void md8_skipsafe_v1() {}

#if WHIRL_MD8_DEV
MD8_SIG(md8d1_mxfp4) MD8_ID1
MD8_SIG(md8d2_mxfp4) MD8_ID2
MD8_SIG(md8d3_mxfp4) MD8_ID3
MD8_SIG(md8d4_mxfp4) MD8_ID4
MD8_SIG(md8d5_mxfp4) MD8_ID5
MD8_SIG(md8d6_mxfp4) MD8_ID6
MD8_SIG(md8d7_mxfp4) MD8_ID7
MD8_SIG(md8d8_mxfp4) MD8_ID8
MD8_SIG(md8d9_mxfp4) MD8_ID9
MD8_SIG(md8d10_mxfp4) MD8_ID10
MD8_SIG(md8d11_mxfp4) MD8_ID11
MD8_SIG(md8d12_mxfp4) MD8_ID12
#endif

#if WHIRL_MD8_MXFP4
// the production entry with the selected id (moe.hip renamed the shipped one to moe_down_mxfp4_shipped)
MD8_SIG(moe_down_mxfp4) MD8_CAT(MD8_ID, WHIRL_MD8_MXFP4)
#endif

#endif  // WHIRL_MD8_MXFP4 || WHIRL_MD8_DEV
