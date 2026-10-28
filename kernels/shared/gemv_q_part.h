// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemv_q.hip, gfx1151 kernels/gfx1151/gemv_q.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

__device__ __forceinline__ int iq4_lut4(uint32_t nib4) {
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = ((nib4 & 0x08080808u) >> 3) * 0xFFu;
    return (int)((hi & m) | (lo & ~m));
}

// kvalues_mxfp4 (0 1 2 3 4 6 8 12 and negatives) through the same two-table
// v_perm_b32 select as iq4_lut4.
__device__ __forceinline__ int mx_lut4(uint32_t nib4) {
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = ((nib4 & 0x08080808u) >> 3) * 0xFFu;
    return (int)((hi & m) | (lo & ~m));
}
