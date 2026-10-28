// WHIRL device code, shared by both kernel sets (gfx1201 kernels/misc_rows_norm.hip, gfx1151 kernels/gfx1151/misc_rows_norm.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

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
