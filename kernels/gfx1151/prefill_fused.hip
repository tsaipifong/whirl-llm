// WHIRL device code (gfx1151 / RDNA 3.5): prefill activation fusions (f16).
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's research prototype (its gfx1151 kernel source). Not
// self-contained: included in order by kernels/gfx1151/whirl_kernels_gfx1151.hip.

// ===== prefill activation fusions from the gfx1201 source (f16 only) =====
// ---------------------------------------------------------------------------
// Item P: prefill activation fusions. Each kernel writes the GEMM input (x16:
// fp8 e4m3 + per-token scale, or f16) directly instead of an f32 tensor that the
// next matmul converts. The arithmetic is the unfused one step for step
// (rmsnorm / silu_mul, then qact_fp8 or f32_to_f16), so the bytes are identical.
// One 256-thread block per token row.
// ---------------------------------------------------------------------------
// item K: byte offset of fp8 activation (t, k) in the WMMA-fragment-tiled layout of gemm8t (K % 16 == 0)
__device__ __forceinline__ size_t x8t_off(int t, int k, int K) {
    return ((size_t)(t >> 4) * (K >> 4) + (k >> 4)) * 256 + (size_t)(((((k >> 3) & 1) << 4) | (t & 15)) * 8) + (k & 7);
}
// gfx1151 has no fp8 conversion; only the f16 (FP8 = false) instances exist here
__device__ __forceinline__ uint8_t fp8_of(float, float) { return 0; }

// rmsnorm(x[t]) * w -> fp8 (FP8) or f16 into q[t]; n <= 256 * RN_PER.
#define RN_PER 32
template <bool FP8, bool TL = false>
__device__ void rmsnorm_x_impl(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    const float *xv = x + (size_t)blockIdx.x * n;
    const int tid = threadIdx.x;
    float v[RN_PER];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < RN_PER; ++k) {
        const int i = tid + k * 256;
        v[k] = 0.f;
        if (i < n) {
            v[k] = xv[i];
            ss += v[k] * v[k];
        }
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < RN_PER; ++k) {
        const int i = tid + k * 256;
        if (i < n) {
            v[k] = v[k] * r * w[i];
            asm volatile("" : "+v"(v[k]));
            m = fmaxf(m, fabsf(v[k]));
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * n;
#pragma unroll
        for (int k = 0; k < RN_PER; ++k) {
            const int i = tid + k * 256;
            if (i < n) {
                if (TL) q[x8t_off(blockIdx.x, i, n)] = fp8_of(v[k], inv);
                else qr[i] = fp8_of(v[k], inv);
            }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * n;
#pragma unroll
        for (int k = 0; k < RN_PER; ++k) {
            const int i = tid + k * 256;
            if (i < n) qr[i] = (_Float16)v[k];
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) rmsnorm_x16(const float *x, const float *w, uint8_t *q, float *sx, int n, float eps) {
    rmsnorm_x_impl<false>(x, w, q, sx, n, eps);
}

// silu(g[t]) * u[t] -> fp8 / f16 into q[t]; ncols % 4 == 0, ncols <= 1024 * SM_PER4.
#define SM_PER4 24
template <bool FP8, bool TL = false>
__device__ void silu_mul_x_impl(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    const float *gr = g + (size_t)blockIdx.x * ncols, *ur = u + (size_t)blockIdx.x * ncols;
    const int tid = threadIdx.x;
    float4 y[SM_PER4];
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < SM_PER4; ++k) {
        const int i = (tid + k * 256) * 4;
        y[k] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < ncols) {
            const float4 a = *(const float4 *)(gr + i), b = *(const float4 *)(ur + i);
            y[k].x = a.x / (1.f + __expf(-a.x)) * b.x;
            y[k].y = a.y / (1.f + __expf(-a.y)) * b.y;
            y[k].z = a.z / (1.f + __expf(-a.z)) * b.z;
            y[k].w = a.w / (1.f + __expf(-a.w)) * b.w;
            // materialize the f32 products (no fused multiply + f16 round: bitwise == silu_mul)
            asm volatile("" : "+v"(y[k].x), "+v"(y[k].y), "+v"(y[k].z), "+v"(y[k].w));
            m = fmaxf(m, fmaxf(fmaxf(fabsf(y[k].x), fabsf(y[k].y)), fmaxf(fabsf(y[k].z), fabsf(y[k].w))));
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * ncols;
#pragma unroll
        for (int k = 0; k < SM_PER4; ++k) {
            const int i = (tid + k * 256) * 4;
            if (i < ncols) {
                const float a = fminf(fmaxf(y[k].x * inv, -448.f), 448.f), b = fminf(fmaxf(y[k].y * inv, -448.f), 448.f);
                const float c = fminf(fmaxf(y[k].z * inv, -448.f), 448.f), d = fminf(fmaxf(y[k].w * inv, -448.f), 448.f);
                int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                p = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, p, true);
                if (TL) *(int *)(q + x8t_off(blockIdx.x, i, ncols)) = p;
                else *(int *)(qr + i) = p;
            }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * ncols;
#pragma unroll
        for (int k = 0; k < SM_PER4; ++k) {
            const int i = (tid + k * 256) * 4;
            if (i < ncols) {
                qr[i] = (_Float16)y[k].x;
                qr[i + 1] = (_Float16)y[k].y;
                qr[i + 2] = (_Float16)y[k].z;
                qr[i + 3] = (_Float16)y[k].w;
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) silu_mul_x16(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    silu_mul_x_impl<false>(g, u, q, sx, ncols);
}

// gdn_gated_norm (per head of 128: rmsnorm * w * silu(z)) fused with the out-proj
// input conversion: one 256-thread block per token, wave w takes heads w, w + 8, ..
// The per-head sum of squares follows block_sum over 128 threads exactly:
// s_k = warp_sum over elements 32k + lane, total = (s0 + s2) + (s1 + s3).
#define GN_HPW 8
template <bool FP8, typename ZT = float, bool TL = false>
__device__ void gated_norm_x_impl(const float *o, const ZT *z, const float *w, uint8_t *q, float *sx, int n_heads, float eps) {
    int dv = 128;
    asm volatile("" : "+s"(dv));  // runtime divisor like gdn_gated_norm (ss / dv)
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int row = n_heads * 128;
    const float *orow = o + (size_t)blockIdx.x * row;
    const ZT *zrow = z + (size_t)blockIdx.x * row;
    float y[GN_HPW][4];
    float m = 0.f;
#pragma unroll
    for (int hh = 0; hh < GN_HPW; ++hh) {
        const int h = wid + 8 * hh;
        if (h < n_heads) {
            float s[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                y[hh][k] = orow[h * 128 + 32 * k + lane];
                float sq = y[hh][k] * y[hh][k];
                asm volatile("" : "+v"(sq));  // no fma into the first butterfly add (block_sum sums rounded squares)
                s[k] = warp_sum(sq);
            }
            // second level exactly as block_sum does it: wave 0 lanes 0..3 hold s0..s3, warp_sum, lane 0
            float t4 = lane == 0 ? s[0] : lane == 1 ? s[1] : lane == 2 ? s[2] : lane == 3 ? s[3] : 0.f;
            t4 = warp_sum(t4);
            const float ss = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, t4)));  // uniform: the norm factor goes through the scalar unit as in gdn_gated_norm
            const float r = rsqrtf(ss / dv + eps);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = 32 * k + lane;
                const float zz = (float)zrow[h * 128 + i];
                y[hh][k] = y[hh][k] * r * w[i] * (zz / (1.f + __expf(-zz)));
                asm volatile("" : "+v"(y[hh][k]));
                m = fmaxf(m, fabsf(y[hh][k]));
            }
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * row;
#pragma unroll
        for (int hh = 0; hh < GN_HPW; ++hh) {
            const int h = wid + 8 * hh;
            if (h < n_heads)
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (TL) q[x8t_off(blockIdx.x, h * 128 + 32 * k + lane, row)] = fp8_of(y[hh][k], inv);
                    else qr[h * 128 + 32 * k + lane] = fp8_of(y[hh][k], inv);
                }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * row;
#pragma unroll
        for (int hh = 0; hh < GN_HPW; ++hh) {
            const int h = wid + 8 * hh;
            if (h < n_heads)
#pragma unroll
                for (int k = 0; k < 4; ++k) qr[h * 128 + 32 * k + lane] = (_Float16)y[hh][k];
        }
    }
}
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
