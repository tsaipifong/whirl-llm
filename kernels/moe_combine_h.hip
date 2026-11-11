// WHIRL device code, gfx1201 set: moe_combine4h for the f16 expert outputs of the MoE prefill (v0252-moedf16, PLAN V1-76).
// SPDX-License-Identifier: Apache-2.0
//
// Balance / fast only (the balance item f16 intermediates; ops::chooseMoe sets MoePlan::ydh on the fp8 kind with COMB 3):
// the down projection gemm8r_moeh / gemm8r_moe32h with W2 = null stores moe_yd as f16 [pos][C]; this kernel adds
// x[t][c] += sum_k w[t,k] * yd[inv[t*K+k]][c] (+ sg[t] * ysh[t][c]) for 4 consecutive columns per thread (C % 4 == 0),
// grid (ceil(C / 1024), n), block 256. Per element the expression and the k order of the gfx1151 moe_combine4h
// (kernels/gfx1151/moe_prefill_mt.hip) and of moe_combine on the f16-widened values; the one difference is the load: one
// 8-byte half4 load per (k, thread) instead of four 2-byte loads (the values are the same, so are the bits). Not shared
// with the gfx1151 file on purpose: changing its loads would change the 8060S code object.
// Included LAST in the gfx1201 block of whirl_kernels.hip: no existing kernel moves.

typedef _Float16 moe_half4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void __launch_bounds__(256)
    moe_combine4h(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg, const _Float16 *__restrict__ yd,
                  const int *__restrict__ inv, const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= C) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const float wk = w[(size_t)t * K + k];
        const moe_half4 v = *(const moe_half4 *)(yd + (size_t)inv[(size_t)t * K + k] * C + c);
        acc.x += wk * (float)v.x;
        acc.y += wk * (float)v.y;
        acc.z += wk * (float)v.z;
        acc.w += wk * (float)v.w;
    }
    if (ysh != nullptr) {
        const float s = sg[t];
        const float4 v = *(const float4 *)(ysh + (size_t)t * C + c);
        acc.x += s * v.x;
        acc.y += s * v.y;
        acc.z += s * v.z;
        acc.w += s * v.w;
    }
    float4 *xo = (float4 *)(x + (size_t)t * C + c);
    float4 xv = *xo;
    xv.x += acc.x;
    xv.y += acc.y;
    xv.z += acc.z;
    xv.w += acc.w;
    *xo = xv;
}
