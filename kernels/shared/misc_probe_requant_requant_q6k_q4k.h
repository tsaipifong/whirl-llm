// WHIRL device code, shared by both kernel sets (gfx1201 kernels/misc_probe_requant.hip, gfx1151 kernels/gfx1151/misc_probe_requant.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
