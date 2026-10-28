// WHIRL device code, shared by both kernel sets: the vocabulary every kernel family uses (vector types, dp4, the
// fp8 tile offset). SPDX-License-Identifier: Apache-2.0
// Part of the kernels.common family (arch025 N14: these were defined by the first family that needed them, so the
// later families depended on it without declaring it; tools/gates/kernel_iso_check.py). Included at the end of
// kernels/common.hip and kernels/gfx1151/quant.hip.

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int dp4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }

// item K: byte offset of fp8 activation (t, k) in the WMMA-fragment-tiled layout of gemm8t (K % 16 == 0)
__device__ __forceinline__ size_t x8t_off(int t, int k, int K) {
    return ((size_t)(t >> 4) * (K >> 4) + (k >> 4)) * 256 + (size_t)(((((k >> 3) & 1) << 4) | (t & 15)) * 8) + (k & 7);
}
