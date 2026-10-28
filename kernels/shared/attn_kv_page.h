// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define KV_PAGE 256
#include "kernel_args.h"  // KvArgs (defined once for both kernel sets, arch025 N13)
typedef char kvq8_t __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int kv_tb(const KvArgs &a, int row) { return a.kvbase != nullptr ? a.kvbase[row] : a.tab0; }
__device__ __forceinline__ int kv_row(const KvArgs &a, int tb, int p) { return a.ptab[tb + p / KV_PAGE] * KV_PAGE + p % KV_PAGE; }
// pool row of p minus p: the same for every position of p's page (aligned key tiles)
__device__ __forceinline__ int kv_delta(const KvArgs &a, int tb, int p) {
    const int pg = p / KV_PAGE;
    return (a.ptab[tb + pg] - pg) * KV_PAGE;
}
// KV formats (template argument Q8 of the kernels):
// 0 f16, 1 q8 (int8 K and V), 2 q8v (K f16, V int8), 3 q4 (4-bit K and V). Element
// formats (template argument F of the loaders): 0 f16, 1 int8, 2 int4. Both int
// formats carry one f16 scale per 32 consecutive values of a row (ks / vs).
// q4: element e is nibble e & 1 (low first) of byte e / 2, value (nibble - 8) * scale
// with the scale = (the group's signed value of largest magnitude) / -8 (as ggml
// q4_0), so the largest value is exact and the rest round to 16 levels.
__host__ __device__ constexpr int kv_kf(int m) { return m == 1 ? 1 : m == 3 ? 2 : 0; }
__host__ __device__ constexpr int kv_vf(int m) { return m == 0 ? 0 : m == 3 ? 2 : 1; }
typedef _Float16 kv_h2 __attribute__((ext_vector_type(2)));
// 8 nibbles (elements e .. e + 7 of one aligned group) -> f16 x8 times s: 0x6400 | n read
// as f16 is 1024 + n (exact), minus 1032 is n - 8 (exact), so (that - 1032) * s is the
// same single f16 rounding as (_Float16)(n - 8) * s.
__device__ __forceinline__ half8 kv_dq4(uint32_t w, _Float16 s) {
#pragma clang fp contract(off)
    const uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;  // even / odd elements
    const kv_h2 off = (kv_h2){(_Float16)1032.0f, (_Float16)1032.0f}, sc = (kv_h2){s, s};
    half8 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t x = ((lo >> (8 * j)) & 0xffu) | (((hi >> (8 * j)) & 0xffu) << 16) | 0x64006400u;
        kv_h2 a;
        __builtin_memcpy(&a, &x, 4);
        a = (a - off) * sc;
        r[2 * j] = a[0];
        r[2 * j + 1] = a[1];
    }
    return r;
}
// int8 x8 -> f16 x8 times s without per-element conversions: 0x6400 |
// (b ^ 0x80) read as f16 is 1024 + (b + 128) (exact), minus 1152 is b (exact), so (that - 1152)
// * s is the same single f16 rounding as (_Float16)b * s. Packed f16 pairs from v_perm.
__device__ __forceinline__ half8 kv_dq8(kvq8_t qk, _Float16 s) {
#pragma clang fp contract(off)
    uint32_t w[2];
    __builtin_memcpy(w, &qk, 8);
    const kv_h2 off = (kv_h2){(_Float16)1152.0f, (_Float16)1152.0f}, sc = (kv_h2){s, s};
    half8 r;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const uint32_t x = w[j] ^ 0x80808080u;
        const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u);  // bytes 0,1 -> 0x64xx halves
        const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u);  // bytes 2,3
        kv_h2 a, b;
        __builtin_memcpy(&a, &lo, 4);
        __builtin_memcpy(&b, &hi, 4);
        a = (a - off) * sc;
        b = (b - off) * sc;
        r[4 * j + 0] = a[0];
        r[4 * j + 1] = a[1];
        r[4 * j + 2] = b[0];
        r[4 * j + 3] = b[1];
    }
    return r;
}
template <int F> __device__ __forceinline__ half8 kv_ld8(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (F == 1) {
        return kv_dq8(*(const kvq8_t *)((const int8_t *)base + e), sc[e >> 5]);
    } else if constexpr (F == 2) {
        return kv_dq4(*(const uint32_t *)((const uint8_t *)base + (e >> 1)), sc[e >> 5]);
    } else {
        return *(const half8 *)((const _Float16 *)base + e);
    }
}
template <int F> __device__ __forceinline__ float kv_ld1(const void *base, const _Float16 *sc, size_t e) {
    if constexpr (F == 1) {
        return (float)((_Float16)((const int8_t *)base)[e] * sc[e >> 5]);
    } else if constexpr (F == 2) {
        const int n = (((const uint8_t *)base)[e >> 1] >> (4 * (e & 1))) & 15;
        return (float)((_Float16)(n - 8) * sc[e >> 5]);
    } else {
        return (float)((const _Float16 *)base)[e];
    }
}
