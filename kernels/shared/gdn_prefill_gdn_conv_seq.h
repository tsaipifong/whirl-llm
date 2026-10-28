// WHIRL device code, shared by both kernel sets (gfx1201 kernels/gdn_prefill.hip, gfx1151 kernels/gfx1151/gdn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void gdn_conv_seq(const float *xin, float *state, const float *w, float *out,
                                        int ch, int n_tok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float w0 = w[(size_t)c * 4 + 0], w1 = w[(size_t)c * 4 + 1];
    const float w2 = w[(size_t)c * 4 + 2], w3 = w[(size_t)c * 4 + 3];
    float h0 = state[c], h1 = state[(size_t)ch + c], h2 = state[(size_t)2 * ch + c];
    for (int t = 0; t < n_tok; ++t) {
        const float xn = xin[(size_t)t * ch + c];
        const float acc = h0 * w0 + h1 * w1 + h2 * w2 + xn * w3;
        out[(size_t)t * ch + c] = acc / (1.f + __expf(-acc));
        h0 = h1;
        h1 = h2;
        h2 = xn;
    }
    state[c] = h0;
    state[(size_t)ch + c] = h1;
    state[(size_t)2 * ch + c] = h2;
}

// beta = sigmoid(b); g = softplus(a + dt_bias) * A   (A = -exp(A_log), stored)
extern "C" __global__ void gdn_gates(float *b, float *a, const float *dt_bias, const float *A, int n,
                                     int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % n_heads;
    b[i] = 1.f / (1.f + __expf(-b[i]));
    const float x = a[i] + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(__expf(x));
    a[i] = sp * A[h];
}

// gdn_gates on the concatenated [beta | alpha] projection output: ba[t][0, nh) = b,
// ba[t][nh, 2nh) = a (one 2nh-row GEMM). Same math as gdn_gates, written to b / a
// laid out [t][nh]. n = n_tok * n_heads.
extern "C" __global__ void gdn_gates_ba(const float *ba, float *b, float *a, const float *dt_bias, const float *A,
                                        int n, int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % n_heads;
    const int t = i / n_heads;
    const float bi = ba[(size_t)t * 2 * n_heads + h];
    const float ai = ba[(size_t)t * 2 * n_heads + n_heads + h];
    b[i] = 1.f / (1.f + __expf(-bi));
    const float x = ai + dt_bias[h];
    const float sp = x > 20.f ? x : log1pf(__expf(x));
    a[i] = sp * A[h];
}

// One v-head per block, blockDim = dv. Thread j keeps column j of the state
// (S[i][j], i = key dim) in registers for the whole token run.
// State layout: state[h*DK*dv + i*dv + j]. qkv: [n_tok][qkv_stride] with q at
// [0, nk*DK), k at [nk*DK, 2*nk*DK), v after. g/beta: [n_tok][n_v_heads].
// ---------------------------------------------------------------------------
// Chunkwise Gated DeltaNet for prefill (C = 64 tokens per chunk), f32.
// Recurrence per v head (S: dk x dv, a_t = exp(g_t)):
//   u_t = beta_t (v_t - a_t S_{t-1}^T k_t);  S_t = a_t S_{t-1} + k_t u_t^T;  o_t = S_t^T q_t
// With G_t = cumsum(g) inside the chunk, A_t = exp(G_t), S0 = state at chunk start:
//   (I + B) U = diag(beta) (V - diag(A) K S0),  B[t][s] = beta_t e^{G_t-G_s} k_t.k_s (s < t)
//   => U = Ut - W S0 with T = (I+B)^-1, Ut = T diag(beta) V, W = T diag(beta A) K
//   O = diag(A) Q S0 + M U,  M[t][s] = e^{G_t-G_s} q_t.k_s (s <= t)
//   S_C = A_C S0 + Kt^T U,   Kt[s] = e^{G_C-G_s} k_s
// gdn_chunk_prep builds W, Ut, M, G for every (chunk, head) in parallel;
// gdn_chunk_scan walks the chunks sequentially per (head, 32 value columns).
// Padded rows (t >= n_tok) carry k = v = 0, beta = 0, g = 0 and drop out.
// ---------------------------------------------------------------------------
#define GCH 64
#define GDK 128
