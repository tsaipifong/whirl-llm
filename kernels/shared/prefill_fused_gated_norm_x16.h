// WHIRL device code, shared by both kernel sets (gfx1201 kernels/prefill_fused.hip, gfx1151 kernels/gfx1151/prefill_fused.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(256) gated_norm_x16(const float *o, const float *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<false>(o, z, w, q, sx, n_heads, eps);
}

// gdn_conv_par + l2norm of the q / k heads in one pass (bitwise the same): block =
// 128 channels (one d_state = 128 head) of one token; blocks < n_qk_heads also
// normalize (block_sum over the same 128 threads as l2norm).
#define CONV_TPB 8  // tokens per block: each input row is loaded once per block, not once per tap
template <typename XT>
__device__ __forceinline__ void gdn_conv_l2n_impl(const XT *xin, const float *state, const float *w, float *out,
                                                  int ch, int n_tok, int n_qk_heads, float eps) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    const int t0 = blockIdx.y * CONV_TPB;
    float xr[CONV_TPB + 3];
#pragma unroll
    for (int j = 0; j < CONV_TPB + 3; ++j) {
        const int tt = t0 - 3 + j;
        xr[j] = tt >= 0 ? (tt < n_tok ? (float)xin[(size_t)tt * ch + c] : 0.f) : state[(size_t)(3 + tt) * ch + c];
    }
    float wk[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) wk[kk] = w[(size_t)c * 4 + kk];
    const bool qk = (int)blockIdx.x < n_qk_heads;  // uniform per block
#pragma unroll
    for (int j = 0; j < CONV_TPB; ++j) {
        const int t = t0 + j;
        if (t >= n_tok) break;  // uniform
        float acc = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) acc += xr[j + kk] * wk[kk];
        float y = acc / (1.f + __expf(-acc));
        asm volatile("" : "+v"(y));
        if (qk) {
            float ss = 0.f;
            ss += y * y;
            asm volatile("" : "+v"(ss));
            ss = block_sum(ss);
            const float r = rsqrtf(ss + eps);
            y *= r;
        }
        out[(size_t)t * ch + c] = y;
    }
}
extern "C" __global__ void __launch_bounds__(128) gdn_conv_l2n(const float *xin, const float *state, const float *w, float *out,
                                                               int ch, int n_tok, int n_qk_heads, float eps) {
    gdn_conv_l2n_impl<float>(xin, state, w, out, ch, n_tok, n_qk_heads, eps);
}
extern "C" __global__ void __launch_bounds__(128) gdn_conv_l2n_h(const _Float16 *xin, const float *state, const float *w, float *out,
                                                                 int ch, int n_tok, int n_qk_heads, float eps) {
    gdn_conv_l2n_impl<_Float16>(xin, state, w, out, ch, n_tok, n_qk_heads, eps);
}
// gdn_conv_state for an f16 input (the new rolling state is kept in f32)
extern "C" __global__ void gdn_conv_state_h(const _Float16 *xin, float *state, int ch, int n_tok, float *snap, int snap_after) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    if (snap != nullptr) {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            const int tt = snap_after - 2 + kk;
            snap[(size_t)kk * ch + c] = tt >= 0 ? (float)xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
        }
    }
    float v[3];
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        const int tt = n_tok - 3 + kk;
        v[kk] = tt >= 0 ? (float)xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
    }
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) state[(size_t)kk * ch + c] = v[kk];
}

// Q4_K_M relaxed mode (item P): f16 gate / up (gemm_ch outputs) -> silu * u -> f16 down input
extern "C" __global__ void __launch_bounds__(256) silu_mul_x16h(const _Float16 *g, const _Float16 *u, uint8_t *q, float *sx, int ncols) {
    const size_t base = (size_t)blockIdx.x * ncols;
    _Float16 *qr = (_Float16 *)q + base;
    for (int i = threadIdx.x * 8; i < ncols; i += 256 * 8) {
        const half8 a = *(const half8 *)(g + base + i), b = *(const half8 *)(u + base + i);
        half8 y;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float gv = (float)a[e];
            y[e] = (_Float16)(gv / (1.f + __expf(-gv)) * (float)b[e]);
        }
        *(half8 *)(qr + i) = y;
    }
}
// z as f16, f16 output (Q4_K_M relaxed mode)
extern "C" __global__ void __launch_bounds__(256) gated_norm_x16h(const float *o, const _Float16 *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    gated_norm_x_impl<false, _Float16>(o, z, w, q, sx, n_heads, eps);
}
