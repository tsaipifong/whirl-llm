// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemv_q.hip, gfx1151 kernels/gfx1151/gemv_q.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

template <int T> struct QDN {
    template <int NT>
    static __device__ __forceinline__ void dot(const uint8_t *row, int u, const int8_t *xq, const float *xd, int ncols,
                                               float *acc) {
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] += QD<T>::dot(row, u, xq + (size_t)t * ncols, xd + (size_t)t * (ncols / 32));
    }
};
