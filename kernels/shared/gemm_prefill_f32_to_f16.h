// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gemm_prefill.hip, gfx1151 kernels/gfx1151/gemm_prefill.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void f32_to_f16(const float *x, _Float16 *y, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float4 v = *(const float4 *)(x + i);
        y[i] = (_Float16)v.x;
        y[i + 1] = (_Float16)v.y;
        y[i + 2] = (_Float16)v.z;
        y[i + 3] = (_Float16)v.w;
    } else {
        for (int j = i; j < n; ++j) y[j] = (_Float16)x[j];
    }
}
