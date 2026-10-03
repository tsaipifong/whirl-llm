// WHIRL device code (gfx1201 / RDNA 4): embedding row lookup (get_rows_*), rmsnorm, l2norm, add, silu_mul, NEOX RoPE.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// Dequantize rows `ids[t]` of W into out[t][ncols] (embedding lookup).
template <int T>
__device__ void get_rows_impl(const uint8_t *W, uint64_t row_bytes, const int *ids, float *out, int ncols) {
    const int t = blockIdx.y;
    const uint8_t *wr = W + (size_t)ids[t] * row_bytes;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < ncols / 8; g += gridDim.x * blockDim.x) {
        float w[8];
        dec_group<T>(wr, g, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[(size_t)t * ncols + 8 * g + i] = w[i];
    }
}

#define GET_ROWS(SUF, T)                                                                               \
    extern "C" __global__ void get_rows_##SUF(const uint8_t *W, uint64_t row_bytes, const int *ids, \
                                              float *out, int ncols) {                               \
        get_rows_impl<T>(W, row_bytes, ids, out, ncols);                                             \
    }

GET_ROWS(f32, QT_F32)
GET_ROWS(f16, QT_F16)
GET_ROWS(q8_0, QT_Q8_0)
GET_ROWS(q3_k, QT_Q3_K)
GET_ROWS(q4_k, QT_Q4_K)
GET_ROWS(q5_k, QT_Q5_K)
GET_ROWS(q6_k, QT_Q6_K)
GET_ROWS(iq4_nl, QT_IQ4_NL)
GET_ROWS(iq3_s, QT_IQ3_S)
GET_ROWS(iq4_xs, QT_IQ4_XS)
GET_ROWS(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// Norms and elementwise ops. "Vector v" of a launch = blockIdx.x.
// ---------------------------------------------------------------------------

// out[v][i] = x[v][i] / sqrt(mean(x^2) + eps) * w[i]; vectors at x + v*stride.
extern "C" __global__ void rmsnorm(const float *x, const float *w, float *out, int n, int in_stride,
                                   int out_stride, float eps) {
    const float *xv = x + (size_t)blockIdx.x * in_stride;
    float *ov = out + (size_t)blockIdx.x * out_stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xv[i] * xv[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ov[i] = xv[i] * r * w[i];
}

// In-place L2 norm: x / sqrt(sum x^2 + eps) (llama.cpp build_gdn_l2_norm).
// Vector (blockIdx.x, blockIdx.y) at x + y*tok_stride + x*stride.
extern "C" __global__ void l2norm(float *x, int n, int stride, int tok_stride, float eps) {
    float *xv = x + (size_t)blockIdx.y * tok_stride + (size_t)blockIdx.x * stride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xv[i] * xv[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xv[i] *= r;
}

extern "C" __global__ void add_inplace(float *a, const float *b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// a = silu(a) * b
extern "C" __global__ void silu_mul(float *a, const float *b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = a[i];
        a[i] = g / (1.f + __expf(-g)) * b[i];
    }
}

// NEOX RoPE on the first n_rot dims of each head, in place.
// x layout: token t, head h at x + t*tok_stride + h*head_stride. grid = (n_head, n_tok).
extern "C" __global__ void rope_neox(float *x, const int *pos, int head_stride, int tok_stride,
                                     int n_rot, float theta_scale) {
    const int h = blockIdx.x, t = blockIdx.y;
    float *v = x + (size_t)t * tok_stride + (size_t)h * head_stride;
    const int half = n_rot / 2;
    const float p = (float)pos[t];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = p * powf(theta_scale, (float)i);
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

