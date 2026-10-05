// WHIRL device code (gfx1151 / RDNA 3.5): WMMA probes, Q6_K / Q8_0 -> Q4_K requant.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ---------------------------------------------------------------------------
// Peak-throughput probes: back-to-back independent WMMAs on registers only.
// 8 accumulators per wave so the matrix pipe never waits on a dependency.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) peak_wmma_f32acc(float *out, int iters) {
    half16 a, b;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = (_Float16)(threadIdx.x * 0.001f + i);
        b[i] = (_Float16)(i * 0.5f);
    }
    float8 c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = WMMA16(a, b, c[j]);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += c[j][0];
    if (sum == 12345.f) out[0] = sum;
}

// gfx11 f16-accumulate WMMA: C/D are 16 halfs per lane, op_sel picks the
// low (false) or high half of each 32-bit register for the 8 results.
extern "C" __global__ void __launch_bounds__(256) peak_wmma_f16acc(float *out, int iters) {
    half16 a, b;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = (_Float16)(threadIdx.x * 0.001f + i);
        b[i] = (_Float16)(i * 0.5f);
    }
    half16 c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 16; ++i) c[j][i] = (_Float16)0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c[j], false);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += (float)c[j][0];
    if (sum == 12345.f) out[0] = sum;
}

// gfx11 int8 WMMA: A/B are 16 int8 per lane (4 dwords), C/D 8 x i32.
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int8v __attribute__((ext_vector_type(8)));
extern "C" __global__ void __launch_bounds__(256) peak_wmma_i8(float *out, int iters) {
    int4v a = {(int)threadIdx.x, 3, 7, 11}, b = {5, (int)threadIdx.x, 13, 17};
    int8v c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j] = (int8v){0, 0, 0, 0, 0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[j], false);
    }
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += c[j][0];
    if (sum == 12345) out[0] = (float)sum;
}

// Requantize a Q6_K (or Q8_0) matrix to Q4_K (MTP block / draft head: drafts only, the
// verify stays exact). Block = one 256-value superblock, thread = value, wave = 32-value
// sub-block. Plain min/max fit per sub-block (min clamped <= 0), 6-bit scales/mins.
// Same code as kernels/misc_probe_requant.hip (no WMMA, nothing gfx11-specific).
template <int T>
__device__ __forceinline__ void requant_q4k_impl(const uint8_t *src, uint64_t src_rb, uint8_t *dst, uint64_t dst_rb) {
    const int row = blockIdx.x, sbi = blockIdx.y, t = threadIdx.x, j = t >> 5;
    // source bytes per 256 values: Q6_K one superblock, Q8_0 eight 32-value blocks
    constexpr int sb_bytes = T == QT_Q6_K ? 210 : 8 * Q<QT_Q8_0>::bytes;
    const uint8_t *b = src + (size_t)row * src_rb + (size_t)sbi * sb_bytes;
    uint8_t *o = dst + (size_t)row * dst_rb + (size_t)sbi * 144;
    __shared__ float v[256];
    __shared__ float scl[8], mnl[8];
    __shared__ int scq[8], mq[8];
    __shared__ uint8_t qv[256];
    if ((t & 7) == 0) {
        if constexpr (T == QT_Q6_K)
            Q<QT_Q6_K>::dec8(b, t, v + t);
        else
            Q<QT_Q8_0>::dec8(b + (t >> 5) * Q<QT_Q8_0>::bytes, t & 31, v + t);
    }
    __syncthreads();
    const float x = v[t];
    const float mn = fminf(0.f, -warp_max(-x));
    const float mx = warp_max(x);
    if ((t & 31) == 0) {
        scl[j] = (mx - mn) / 15.f;
        mnl[j] = -mn;
    }
    __syncthreads();
    float ms = 0.f, mm = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ms = fmaxf(ms, scl[i]);
        mm = fmaxf(mm, mnl[i]);
    }
    const _Float16 dh = (_Float16)(ms / 63.f), dmh = (_Float16)(mm / 63.f);
    const float df = (float)dh, dmf = (float)dmh;
    const int sc = df > 0.f ? min(63, (int)ceilf(scl[j] / df - 1e-4f)) : 0;
    const int m = dmf > 0.f ? min(63, (int)ceilf(mnl[j] / dmf - 1e-4f)) : 0;
    const float step = df * sc, off = dmf * m;
    int q = step > 0.f ? (int)rintf((x + off) / step) : 0;
    qv[t] = (uint8_t)max(0, min(15, q));
    if ((t & 31) == 0) {
        scq[j] = sc;
        mq[j] = m;
    }
    __syncthreads();
    if (t < 128) {
        const int j64 = t >> 5, l = t & 31;
        o[16 + t] = qv[64 * j64 + l] | (qv[64 * j64 + 32 + l] << 4);
    } else if (t == 128) {
        *(_Float16 *)o = dh;
        *(_Float16 *)(o + 2) = dmh;
        uint8_t *s = o + 4;
        for (int i = 0; i < 4; ++i) {
            s[i] = (uint8_t)(scq[i] | ((scq[i + 4] >> 4) << 6));
            s[i + 4] = (uint8_t)(mq[i] | ((mq[i + 4] >> 4) << 6));
        }
        for (int i = 4; i < 8; ++i) s[i + 4] = (uint8_t)((scq[i] & 0xF) | ((mq[i] & 0xF) << 4));
    }
}
extern "C" __global__ void __launch_bounds__(256)
    requant_q6k_q4k(const uint8_t *src, uint64_t src_rb, uint8_t *dst, uint64_t dst_rb) {
    requant_q4k_impl<QT_Q6_K>(src, src_rb, dst, dst_rb);
}
extern "C" __global__ void __launch_bounds__(256)
    requant_q80_q4k(const uint8_t *src, uint64_t src_rb, uint8_t *dst, uint64_t dst_rb) {
    requant_q4k_impl<QT_Q8_0>(src, src_rb, dst, dst_rb);
}
