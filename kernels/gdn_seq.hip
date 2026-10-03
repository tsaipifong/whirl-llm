// WHIRL device code (gfx1201 / RDNA 4): Gated DeltaNet token-parallel conv, conv state update, sequential scan, gated norm.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.


// Causal conv over a token block, parallel over (channel, token): the first
// d_conv-1 taps before the block come from the rolling state.
extern "C" __global__ void gdn_conv_par(const float *xin, const float *state, const float *w, float *out, int ch,
                                        int n_tok) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= ch) return;
    float acc = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int tt = t - 3 + kk;  // input time index
        const float x = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
        acc += x * w[(size_t)c * 4 + kk];
    }
    out[(size_t)t * ch + c] = acc / (1.f + __expf(-acc));
}

// New rolling state = last 3 inputs (mixing in old state when n_tok < 3).
// Runs after gdn_conv_par (which reads the old state).
extern "C" __global__ void gdn_conv_state(const float *xin, float *state, int ch, int n_tok, float *snap,
                                          int snap_after) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    if (snap != nullptr) {  // rolling state as it was right after token snap_after
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            const int tt = snap_after - 2 + kk;
            snap[(size_t)kk * ch + c] = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
        }
    }
    float v[3];
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
        const int tt = n_tok - 3 + kk;
        v[kk] = tt >= 0 ? xin[(size_t)tt * ch + c] : state[(size_t)(3 + tt) * ch + c];
    }
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) state[(size_t)kk * ch + c] = v[kk];
}

// Sum over the 4 lanes of a quad (lanes 4k..4k+3) with DPP quad permutes
// (VALU, no LDS round trip like ds_bpermute).
__device__ __forceinline__ float quad_sum(float x) {
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));  // xor 1
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));  // xor 2
    return x;
}

// Sum over groups of 8 lanes: quad permutes + row_xmask:4 (xor 4).
__device__ __forceinline__ float oct_sum(float x) {
    x = quad_sum(x);
    x += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x164, 0xF, 0xF, false));
    return x;
}

template <int DK, int MODE = 0, int LPC = 4>
__device__ void gdn_seq_impl(const float *qkv, const float *g, const float *beta, float *state, float *out,
                             int n_tok, int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale,
                             float *snap = nullptr, int snap_after = -1) {
    // Each value column j is owned by the 4 lanes of a quad (qi = lane & 3),
    // each holding 32 of the DK state entries S[i][j] in registers; the two
    // dot products per token use 4 independent accumulators and reduce with
    // DPP quad permutes. Inputs are staged GC tokens at a time into LDS
    // (double-buffered; the next chunk's loads overlap this chunk's steps), so
    // the steps inside a chunk touch only LDS and need no barrier.
    // blockDim.x = 128 (32 columns x 4); blockIdx.y selects the column group.
    constexpr int QP = DK / LPC;
    constexpr int GC = 8;
    constexpr int NCOL = 128 / LPC;
    constexpr int QK4 = 2 * DK / 4;                     // float4 per token (q then k)
    constexpr int QK_PER = GC * QK4 / 128;              // float4 per thread per chunk
    constexpr int V_PER = GC * NCOL / 128;
    const int h = blockIdx.x;
    const int qi = threadIdx.x & (LPC - 1), jl = threadIdx.x / LPC;
    const bool dbg = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
    unsigned long long c_start = 0, w_start = 0;
    if (dbg) {
        c_start = clock64();
        w_start = wall_clock64();
    }
    const int j0 = blockIdx.y * NCOL;
    const int j = j0 + jl;
    const int kh = h % n_k_heads;
    __shared__ float4 qk4[2][GC][QK4];
    __shared__ float vs[2][GC][NCOL];
    __shared__ float gs[2][GC], bs[2][GC];
    float S[QP];
    float *Sg = state + (size_t)h * DK * dv;
#pragma unroll
    for (int e = 0; e < QP; ++e) S[e] = Sg[(size_t)(qi * QP + e) * dv + j];

    float4 rqk[QK_PER];
    float rvv[V_PER];
    float rgb = 0.f;
    // loads use a clamped token index (always valid, no branches) so the
    // staging registers stay in VGPRs
#define GDN_FETCH(c0)                                                                               \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < QK_PER; ++e) {                                        \
            const int idx = threadIdx.x + e * 128;                                                  \
            const int tt = idx / QK4, c4 = idx % QK4;                                               \
            const int t = min((c0) + tt, n_tok - 1);                                                \
            const float *row = qkv + (size_t)t * qkv_stride;                                        \
            const float *src = (c4 < DK / 4) ? row + (size_t)kh * DK + 4 * c4                        \
                                             : row + (size_t)n_k_heads * DK + (size_t)kh * DK + 4 * (c4 - DK / 4); \
            rqk[e] = *(const float4 *)src;                                                          \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < V_PER; ++e) {                                         \
            const int idx = threadIdx.x + e * 128;                                                  \
            const int t = min((c0) + idx / NCOL, n_tok - 1);                                        \
            rvv[e] = qkv[(size_t)t * qkv_stride + (size_t)2 * n_k_heads * DK + (size_t)h * dv + j0 + idx % NCOL]; \
        }                                                                                           \
        {                                                                                           \
            const int t = min((c0) + (int)(threadIdx.x % GC), n_tok - 1);                           \
            rgb = (threadIdx.x < GC) ? g[(size_t)t * n_v_heads + h] : beta[(size_t)t * n_v_heads + h]; \
        }                                                                                           \
    } while (0)
#define GDN_STAGE(buf)                                                                              \
    do {                                                                                            \
        _Pragma("unroll") for (int e = 0; e < QK_PER; ++e) {                                        \
            const int idx = threadIdx.x + e * 128;                                                  \
            qk4[buf][idx / QK4][idx % QK4] = rqk[e];                                                \
        }                                                                                           \
        _Pragma("unroll") for (int e = 0; e < V_PER; ++e) {                                         \
            const int idx = threadIdx.x + e * 128;                                                  \
            vs[buf][idx / NCOL][idx % NCOL] = rvv[e];                                               \
        }                                                                                           \
        if (threadIdx.x < GC) gs[buf][threadIdx.x] = rgb;                                           \
        else if (threadIdx.x < 2 * GC) bs[buf][threadIdx.x - GC] = rgb;                             \
    } while (0)

    GDN_FETCH(0);
    GDN_STAGE(0);
    __syncthreads();
    int cur = 0;
    for (int c0 = 0; c0 < n_tok; c0 += GC) {
        const bool more = c0 + GC < n_tok;
        if (more && MODE != 1) GDN_FETCH(c0 + GC);
        const int steps = min(GC, n_tok - c0);
        for (int tt = 0; tt < steps; ++tt) {
            const float4 *q4 = &qk4[cur][tt][qi * (QP / 4)];
            const float4 *k4 = &qk4[cur][tt][DK / 4 + qi * (QP / 4)];
            const float decay = __expf(gs[cur][tt]);
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
            for (int e4 = 0; e4 < QP / 4; ++e4) {
                const float4 kv = k4[e4];
                S[4 * e4 + 0] *= decay;
                S[4 * e4 + 1] *= decay;
                S[4 * e4 + 2] *= decay;
                S[4 * e4 + 3] *= decay;
                a0 += S[4 * e4 + 0] * kv.x;
                a1 += S[4 * e4 + 1] * kv.y;
                a2 += S[4 * e4 + 2] * kv.z;
                a3 += S[4 * e4 + 3] * kv.w;
            }
            const float sk = (LPC == 8 ? oct_sum((a0 + a1) + (a2 + a3)) : quad_sum((a0 + a1) + (a2 + a3)));
            const float delta = (vs[cur][tt][jl] - sk) * bs[cur][tt];
            if (MODE == 3) {
                if (qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = delta;
                continue;
            }
            a0 = a1 = a2 = a3 = 0.f;
#pragma unroll
            for (int e4 = 0; e4 < QP / 4; ++e4) {
                const float4 kv = k4[e4], qv = q4[e4];
                S[4 * e4 + 0] += kv.x * delta;
                S[4 * e4 + 1] += kv.y * delta;
                S[4 * e4 + 2] += kv.z * delta;
                S[4 * e4 + 3] += kv.w * delta;
                a0 += S[4 * e4 + 0] * qv.x;
                a1 += S[4 * e4 + 1] * qv.y;
                a2 += S[4 * e4 + 2] * qv.z;
                a3 += S[4 * e4 + 3] * qv.w;
            }
            const float o = (LPC == 8 ? oct_sum((a0 + a1) + (a2 + a3)) : quad_sum((a0 + a1) + (a2 + a3)));
            if (MODE != 2 && qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = o * scale;
            if (snap != nullptr && c0 + tt == snap_after) {
                float *Ss = snap + (size_t)h * DK * dv;
#pragma unroll
                for (int e = 0; e < QP; ++e) Ss[(size_t)(qi * QP + e) * dv + j] = S[e];
            }
            if (MODE == 2 && tt == steps - 1 && qi == 0) out[((size_t)(c0 + tt) * n_v_heads + h) * dv + j] = o * scale;
        }
        if (more) {
            __syncthreads();
            GDN_STAGE(cur ^ 1);
            __syncthreads();
            cur ^= 1;
        }
    }
#undef GDN_FETCH
#undef GDN_STAGE
#pragma unroll
    for (int e = 0; e < QP; ++e) Sg[(size_t)(qi * QP + e) * dv + j] = S[e];
    if (dbg) {
        whirl_dbg[0] = clock64() - c_start;
        whirl_dbg[1] = wall_clock64() - w_start;
    }
}

#define GDN_PROBE(NAME, MODE)                                                                       \
    extern "C" __global__ void __launch_bounds__(128)                                              \
        NAME(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok, \
             int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale) {                  \
        gdn_seq_impl<128, MODE>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale); \
    }
GDN_PROBE(gdn_probe_nofetch, 1)
GDN_PROBE(gdn_probe_nostore, 2)
GDN_PROBE(gdn_probe_half, 3)
extern "C" __global__ void __launch_bounds__(128)
    gdn_seq_128_l8(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok,
                   int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale, float *snap, int snap_after) {
    gdn_seq_impl<128, 0, 8>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale);
}

extern "C" __global__ void __launch_bounds__(128)
    gdn_seq_128(const float *qkv, const float *g, const float *beta, float *state, float *out, int n_tok,
                int n_k_heads, int n_v_heads, int dv, int qkv_stride, float scale, float *snap, int snap_after) {
    gdn_seq_impl<128>(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv, qkv_stride, scale, snap, snap_after);
}

// out[h][j] = rmsnorm(o[h])[j] * w[j] * silu(z[h][j]), in place on o.
extern "C" __global__ void gdn_gated_norm(float *o, const float *z, const float *w, int dv, float eps) {
    float *ov = o + (size_t)blockIdx.x * dv;
    const float *zv = z + (size_t)blockIdx.x * dv;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) ss += ov[i] * ov[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / dv + eps);
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const float zz = zv[i];
        ov[i] = ov[i] * r * w[i] * (zz / (1.f + __expf(-zz)));
    }
}

