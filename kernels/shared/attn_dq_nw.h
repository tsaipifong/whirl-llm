// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define DQ_NW 8
#define DQ_C 8
#define DQ_MAXS 128
__device__ __forceinline__ int dq_per(int n_keys, int n_split) {
    const int ns = min(n_split, DQ_MAXS);
    return max(DQ_NW * 32, (n_keys + ns * 64 - 1) / (ns * 64) * 64);
}
typedef _Float16 dq_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ dq_h2 dq_bits(uint32_t x) {
    dq_h2 r;
    __builtin_memcpy(&r, &x, 4);
    return r;
}
__device__ __forceinline__ void dq_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
