// WHIRL device code (gfx1201 / RDNA 4): int8 Q.K^T prefill attention (feat027-i8qk, fast mode only).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by whirl_kernels.hip after attn_prefill_kp.hip (KvArgs, kv_delta, FA_K,
// FA_D, half8, float8, kg_ld_tr, WARP).
//
// SageAttention-style (Zhang et al., ICLR 2025, arXiv 2410.02367; progress\feat027-i8qk.PRIOR_ART.md):
//   * K of a key range (the f16 copy kv_dq_rows_r[_q4] made in ffn_g) is smoothed by mu, the per-(KV head,
//     dimension) mean over a strided sample of the range's keys (i8k_mean, deterministic), and quantized
//     per key row and KV head over all 256 dimensions: kq = rint((k - mu) * 127 / amax), sk = amax / 127
//     (i8k_quant);
//   * Q is quantized in the kernel per (query, head) over its 256 dimensions: qi = rint(q * 127 / amax_q),
//     sq = amax_q / 127;
//   * S^T = Kq x Qi^T with v_wmma_i32_16x16x16_iu8 (2x the f16 WMMA rate on gfx1201), exact int32
//     (|dot| <= 256 * 127 * 127 < 2^31), then score = sq * (sk * dot + q.mu / sq): q.mu (f32, once per query)
//     restores the exact shift, so any mu is correct and the key-range series (softmax state carried,
//     each range with its own mu) stays consistent;
//   * the per-query factor sq is folded into the exp2 constant (cexp * sq), so the epilogue is one cvt and
//     one fma per score;
//   * softmax, f16 P, f16 V (global_load_tr) and the P.V WMMA chain are attn_kg's (SEG variant).
// Lossy (not bit-identical to attn_kgs): selected only in fast mode with WHIRL_ATTN_I8QK=1 (ops::chooseAttnPrefill).
// Same lane maps as the f16 16x16x16 WMMA (hipfire gemm_hfq4g256_residual_mmq.gfx12.hip:27-40): A / B lane l
// holds row l % 16, k 8 * (l / 16) .. + 7 as 8 bytes; C lane l holds column l % 16, rows 8 * (l / 16) + v.

typedef int i8q_v2i __attribute__((ext_vector_type(2)));
typedef int i8q_v8i __attribute__((ext_vector_type(8)));

#define I8K_MEAN_ROWS 2048  // sampled keys per range for mu (stride max(1, n_rows / 2048))

__device__ __forceinline__ int i8q_pack4(float a, float b, float c, float d) {
    const int qa = (int)fminf(fmaxf(__builtin_rintf(a), -127.f), 127.f), qb = (int)fminf(fmaxf(__builtin_rintf(b), -127.f), 127.f);
    const int qc = (int)fminf(fmaxf(__builtin_rintf(c), -127.f), 127.f), qd = (int)fminf(fmaxf(__builtin_rintf(d), -127.f), 127.f);
    return (qa & 0xff) | ((qb & 0xff) << 8) | ((qc & 0xff) << 16) | ((int)((unsigned)qd << 24));
}

// mu[c] (c < row_el = n_kv * 256): mean of k16[r][c] over r = 0, st, 2 st, .. < n_rows, st = max(1, n_rows / 2048);
// fixed summation order (4 phases, then phase 0 + 1 + 2 + 3). grid row_el / 64, block 256.
extern "C" __global__ void __launch_bounds__(256) i8k_mean(const _Float16 *k16, float *mu, int n_rows, int row_el) {
    __shared__ float part[4][64];
    const int t = threadIdx.x % 64, ph = threadIdx.x / 64;
    const int c = blockIdx.x * 64 + t;
    const int st = max(1, n_rows / I8K_MEAN_ROWS);
    float s = 0.f;
    for (int r = ph * st; r < n_rows; r += 4 * st) s += (float)k16[(size_t)r * row_el + c];
    part[ph][t] = s;
    __syncthreads();
    if (ph == 0) {
        const int cnt = (n_rows + st - 1) / st;
        mu[c] = (((part[0][t] + part[1][t]) + part[2][t]) + part[3][t]) / (float)cnt;
    }
}

// One wave per (row, KV head) of rows [0, n_rows) of the f16 copy: kq[row][kvh][256] int8 (row stride n_kv * 256
// bytes, the K pool layout), ksc[kvh * kss + row] f32. grid ceil(n_rows * n_kv / 8), block 256.
extern "C" __global__ void __launch_bounds__(256) i8k_quant(const _Float16 *k16, const float *mu, int8_t *kq, float *ksc, int n_rows, int n_kv,
                                                            int kss) {
    const int lane = threadIdx.x % WARP;
    const int item = blockIdx.x * 8 + threadIdx.x / WARP;
    if (item >= n_rows * n_kv) return;
    const int row = item / n_kv, kvh = item % n_kv;
    const size_t base = ((size_t)row * n_kv + kvh) * FA_D + lane * 8;
    const half8 x = *(const half8 *)(k16 + base);
    const float4 m0 = *(const float4 *)(mu + kvh * FA_D + lane * 8), m1 = *(const float4 *)(mu + kvh * FA_D + lane * 8 + 4);
    const float v[8] = {(float)x[0] - m0.x, (float)x[1] - m0.y, (float)x[2] - m0.z, (float)x[3] - m0.w,
                        (float)x[4] - m1.x, (float)x[5] - m1.y, (float)x[6] - m1.z, (float)x[7] - m1.w};
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) a = fmaxf(a, fabsf(v[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, WARP));
    const float inv = a > 0.f ? 127.f / a : 0.f;
    i8q_v2i p;
    p[0] = i8q_pack4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
    p[1] = i8q_pack4(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv);
    *(i8q_v2i *)(kq + base) = p;
    if (lane == 0) ksc[(size_t)kvh * kss + row] = a / 127.f;
}

// attn_kgs (f16 V from the copy, keys [kb0, kb1), softmax state in st_ml / out) with the int8 S^T above.
// kq / ksc are indexed by sequence key (the host passes the range's buffers minus r0 rows); mu is the range's.
template <int NP>
__device__ __forceinline__ void attn_kgi_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,
                                              int n_tok, int h0, int kb0, int kb1, float *st_ml, const int8_t *kq, const float *ksc, int kss,
                                              const float *mu, float scale) {
    const int tb = kv.tab0;
    __shared__ float8 Sx[2][NP * 2][32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int w = wid & 1, dh = w * (FA_D / 2);
    const int h = blockIdx.y * NP + (wid >> 1) + h0;
    const int q0 = blockIdx.x * 16;
    const int kvh = h / (n_head / n_kv);
    const int pos0 = pos[0];
    const int my_q = q0 + lane % 16;
    const int my_qc = min(my_q, n_tok - 1);
    const int my_pos = pos0 + my_qc;
    const int kcol = 8 * (lane / 16);
    // Q: per-(query, head) absmax over the 256 dims (this lane and lane ^ 16 hold them), q.mu in f32
    i8q_v2i qi[FA_D / 16];
    float sq, qmd;
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride + kcol;
        const float *mr = mu + (size_t)kvh * FA_D + kcol;
        float am = 0.f, qm = 0.f;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st), b = *(const float4 *)(qr + 16 * st + 4);
            const float4 ma = *(const float4 *)(mr + 16 * st), mb = *(const float4 *)(mr + 16 * st + 4);
            am = fmaxf(am, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
            am = fmaxf(am, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
            qm += a.x * ma.x + a.y * ma.y + a.z * ma.z + a.w * ma.w + b.x * mb.x + b.y * mb.y + b.z * mb.z + b.w * mb.w;
        }
        am = fmaxf(am, __shfl_xor(am, 16, WARP));
        qm += __shfl_xor(qm, 16, WARP);
        const float inv = am > 0.f ? 127.f / am : 0.f;
        sq = am > 0.f ? am / 127.f : 1.f;
        qmd = qm / sq;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st), b = *(const float4 *)(qr + 16 * st + 4);
            qi[st][0] = i8q_pack4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
            qi[st][1] = i8q_pack4(b.x * inv, b.y * inv, b.z * inv, b.w * inv);
        }
    }
    constexpr int DT = FA_D / 32;
    float8 o[DT];
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    const int n_keys = pos0 + min(q0 + 16, n_tok);
    if (kb0 >= n_keys) return;  // done in an earlier range (block-uniform)
    if (kb0 > 0) {
        const float *orow = out + (size_t)my_qc * n_head * FA_D + (size_t)h * FA_D;
        m = st_ml[((size_t)my_qc * n_head + h) * 2];
        l = st_ml[((size_t)my_qc * n_head + h) * 2 + 1];
#pragma unroll
        for (int dt = 0; dt < DT; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[dt][v] = orow[dh + dt * 16 + 8 * (lane / 16) + v];
    }
    const int k_end = min(n_keys, kb1);
    const int full_end = pos0 + q0;
    const int rs = n_kv * FA_D;
    const int8_t *Kq = kq + (size_t)kvh * FA_D;
    const float *Ksc = ksc + (size_t)kvh * kss;
    const int krow = w * 16 + lane % 16;
    // scores are in units of sq (s' = s / sq); exp2((s' - m') * cexp * sq) == exp2((s - m) * cexp)
    const float cexp = scale * 1.44269504088896340736f * sq;
    const _Float16 *Vc = (const _Float16 *)kv.v + (size_t)kvh * FA_D;
    const int vkey = 8 * (lane / 16) + (lane % 8);
    const int vdim = dh + 8 * ((lane >> 3) & 1);
    int par = 0;
    for (int k0 = kb0; k0 < k_end; k0 += FA_K, par ^= 1) {
        const int dl = kv_delta(kv, tb, k0);
        i8q_v8i acc = (i8q_v8i){0, 0, 0, 0, 0, 0, 0, 0};
        {
            const int8_t *kr = Kq + (size_t)min(k0 + krow, n_keys - 1) * rs + kcol;
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const i8q_v2i a = *(const i8q_v2i *)(kr + 16 * st);
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, qi[st], acc, false);
            }
        }
        float8 own;
        {
            // this lane's 8 key rows k0 + 16 w + 8 (lane / 16) + v (16-byte aligned: k0, kss multiples of 32);
            // rows past n_keys - 1 read padding and are masked below
            const float *sp = Ksc + k0 + w * 16 + kcol;
            const float4 s0 = *(const float4 *)sp, s1 = *(const float4 *)(sp + 4);
            own[0] = __builtin_fmaf((float)acc[0], s0.x, qmd);
            own[1] = __builtin_fmaf((float)acc[1], s0.y, qmd);
            own[2] = __builtin_fmaf((float)acc[2], s0.z, qmd);
            own[3] = __builtin_fmaf((float)acc[3], s0.w, qmd);
            own[4] = __builtin_fmaf((float)acc[4], s1.x, qmd);
            own[5] = __builtin_fmaf((float)acc[5], s1.y, qmd);
            own[6] = __builtin_fmaf((float)acc[6], s1.z, qmd);
            own[7] = __builtin_fmaf((float)acc[7], s1.w, qmd);
        }
        Sx[par][wid][lane] = own;
        __syncthreads();
        float8 sc[FA_K / 16];
        sc[0] = Sx[par][wid & ~1][lane];
        sc[1] = Sx[par][wid | 1][lane];
        float mx = -INFINITY;
        if (k0 + FA_K - 1 <= full_end) {
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                for (int v = 0; v < 8; ++v) mx = fmaxf(mx, sc[kt][v]);
        } else {
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int key = k0 + kt * 16 + 8 * (lane / 16) + v;
                    const float x = (key <= my_pos) ? sc[kt][v] : -INFINITY;
                    sc[kt][v] = x;
                    mx = fmaxf(mx, x);
                }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = __builtin_amdgcn_exp2f((m - m_new) * cexp);
        float rsum = 0.f;
        half8 pt[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = __builtin_amdgcn_exp2f((sc[kt][v] - m_new) * cexp);
                rsum += p;
                pt[kt][v] = (_Float16)p;
            }
        rsum += __shfl_xor(rsum, 16, WARP);
        l = l * alpha + rsum;
        m = m_new;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
        const _Float16 *vb0 = Vc + (size_t)(min(k0 + vkey, n_keys - 1) + dl) * rs + vdim;
        const _Float16 *vb1 = Vc + (size_t)(min(k0 + 16 + vkey, n_keys - 1) + dl) * rs + vdim;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            const half8 a0 = kg_ld_tr(vb0 + dt * 16);
            const half8 a1 = kg_ld_tr(vb1 + dt * 16);
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, pt[0], o[dt]);
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, pt[1], o[dt]);
        }
    }
    if (my_q >= n_tok) return;
    if (k_end < n_keys) {  // not this block's last range: keep the running state
        float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
        if (w == 0 && lane < 16) {
            st_ml[((size_t)my_q * n_head + h) * 2] = m;
            st_ml[((size_t)my_q * n_head + h) * 2 + 1] = l;
        }
#pragma unroll
        for (int dt = 0; dt < DT; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) orow[dh + dt * 16 + 8 * (lane / 16) + v] = o[dt][v];
        return;
    }
    const float inv = 1.f / l;
    const float *gate = q + ((size_t)my_q * n_head + h) * q_stride + FA_D;
    float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int d = dh + dt * 16 + 8 * (lane / 16) + v;
            orow[d] = o[dt][v] * inv / (1.f + __expf(-gate[d]));
        }
}
// attn_kgs arguments + (kq, ksc, kss, mu); grid (ceil(n_tok / 16), heads / NP), block 64 * NP
#define KGI_ENTRY(NAME, NP)                                                                                              \
    extern "C" __global__ void __launch_bounds__(64 * NP)                                                                \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos, int n_tok,        \
             float scale, int h0, int kb0, int kb1, float *st_ml, const int8_t *kq, const float *ksc, int kss,            \
             const float *mu) {                                                                                          \
        attn_kgi_impl<NP>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, h0, kb0, kb1, st_ml, kq, ksc, kss, mu, scale); \
    }
KGI_ENTRY(attn_kgsi6, 6)
KGI_ENTRY(attn_kgsi4, 4)
KGI_ENTRY(attn_kgsi2, 2)
