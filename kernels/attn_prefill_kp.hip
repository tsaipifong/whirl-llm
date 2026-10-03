// WHIRL device code (gfx1201 / RDNA 4): prefill attention, key-split pairs variant; f16 GEMM configurations.
// SPDX-License-Identifier: Apache-2.0
// Ported from the private prototype; not self-contained: included in order by whirl_kernels.hip.

// ---------------------------------------------------------------------------
// Item P: prefill attention, "key-split pairs" (S^T of 16 keys per wave, exchanged). Same algorithm and the same
// per-element arithmetic as attn_prefill_impl (bitwise-identical output), but
// 16 waves per 128 queries: waves 2g and 2g+1 both hold query group g's Q^T
// and compute its full S^T / softmax (identical, redundant), and each keeps
// only half of O^T (dims 128 * (w & 1) ..): 64 accumulator VGPRs instead of
// 128, so nothing spills. K / V tiles of the next FA_K keys are prefetched
// into registers while the current tile is processed.
// ---------------------------------------------------------------------------
template <int Q8>
__device__ __forceinline__ void attn_kx_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                                               int q_stride, const int *pos, int n_tok, float scale, int h0) {
    const int tb = kv.tab0;
    __shared__ _Float16 Ks[FA_K][FA_D + 8];
    __shared__ _Float16 Vt[FA_D][FA_K + 8];
    __shared__ float8 Sx[16][32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int grp = wid >> 1, dh = (wid & 1) * (FA_D / 2);
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * FA_Q;
    const int pos0 = pos[0];
    const int my_q = q0 + grp * 16 + lane % 16;
    const int my_qc = min(my_q, n_tok - 1);
    const int my_pos = pos0 + my_qc;

    half8 qf[FA_D / 16];
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }
    constexpr int DT = FA_D / 32;  // this wave's 16-dim output tiles
    float8 o[DT];
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;

    const int last_q = min(q0 + FA_Q, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    // loaders: threads [0,128) V (8 keys x 8 dims, transposed), [128,384) K (4 x 16 bytes)
    const int tid = threadIdx.x;
    half8 pre[8];
#define KX_FETCH(kf)                                                                                      \
    do {                                                                                                  \
        const int dlf = kv_delta(kv, tb, (kf));                                                           \
        if (tid < 128) {                                                                                  \
            const int kb = tid / (FA_D / 8), d8 = (tid % (FA_D / 8)) * 8;                                 \
            _Pragma("unroll") for (int r = 0; r < 8; ++r) {                                               \
                const int key = min((kf) + kb * 8 + r, n_keys - 1);                                       \
                pre[r] = kv_ld8<Q8 != 0>(kv.v, kv.vs, ((size_t)(key + dlf) * n_kv + kvh) * FA_D + d8);    \
            }                                                                                             \
        } else if (tid < 384) {                                                                           \
            _Pragma("unroll") for (int e = 0; e < 4; ++e) {                                               \
                const int i = (tid - 128) + e * 256;                                                      \
                const int kr = i / (FA_D / 8), d8 = (i % (FA_D / 8)) * 8;                                 \
                const int key = min((kf) + kr, n_keys - 1);                                               \
                pre[e] = kv_ld8<Q8 == 1>(kv.k, kv.ks, ((size_t)(key + dlf) * n_kv + kvh) * FA_D + d8);    \
            }                                                                                             \
        }                                                                                                 \
    } while (0)
    KX_FETCH(0);
    for (int k0 = 0; k0 < n_keys; k0 += FA_K) {
        __syncthreads();
        if (tid < 128) {
            const int kb = tid / (FA_D / 8), d8 = (tid % (FA_D / 8)) * 8;
#pragma unroll
            for (int r = 0; r < 8; ++r)
                if (k0 + kb * 8 + r >= n_keys)
#pragma unroll
                    for (int z = 0; z < 8; ++z) pre[r][z] = (_Float16)0.f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                half8 col;
#pragma unroll
                for (int r = 0; r < 8; ++r) col[r] = pre[r][c];
                *(half8 *)&Vt[d8 + c][kb * 8] = col;
            }
        } else if (tid < 384) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = (tid - 128) + e * 256;
                const int kr = i / (FA_D / 8), d8 = (i % (FA_D / 8)) * 8;
                half8 kv8 = pre[e];
                if (k0 + kr >= n_keys)
#pragma unroll
                    for (int z = 0; z < 8; ++z) kv8[z] = (_Float16)0.f;
                *(half8 *)&Ks[kr][d8] = kv8;
            }
        }
        __syncthreads();
        if (k0 + FA_K < n_keys) KX_FETCH(k0 + FA_K);
        // this wave's 16 keys (kt = wid & 1) with exactly the reference WMMA chain,
        // the partner's 16 through LDS
        float8 sc[FA_K / 16];
        {
            const int kt = wid & 1;
            float8 own = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const half8 a = *(const half8 *)&Ks[kt * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                own = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[st], own);
            }
            Sx[wid][lane] = own;
            __syncthreads();
            const float8 other = Sx[wid ^ 1][lane];
            sc[0] = kt == 0 ? own : other;
            sc[1] = kt == 0 ? other : own;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + 8 * (lane / 16) + v;
                const float x = (key <= my_pos) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = l * alpha + rs;
        m = m_new;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
#pragma unroll
        for (int dt = 0; dt < DT; ++dt)
#pragma unroll
            for (int kt = 0; kt < FA_K / 16; ++kt) {
                const half8 a = *(const half8 *)&Vt[dh + dt * 16 + lane % 16][kt * 16 + 8 * (lane / 16)];
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, pt[kt], o[dt]);
            }
    }
#undef KX_FETCH
    if (my_q >= n_tok) return;
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
#define KX_ENTRY(NAME, Q8)                                                                          \
    extern "C" __global__ void __launch_bounds__(512)                                               \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv,                           \
             int q_stride, const int *pos, int n_tok, float scale, int h0) {                        \
        attn_kx_impl<Q8>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);              \
    }
KX_ENTRY(attn_kx, false)
KX_ENTRY(attn_kx_q8, true)
KX_ENTRY(attn_kx_q8v, 2)

// ---------------------------------------------------------------------------
// Prefill attention, "GQA-grouped" (f16 and q8v KV). Same per-query arithmetic
// as attn_kx / attn_prefill_impl, so the output is bit-identical: the same
// 32-key tiles in the same order, the same S^T WMMA chain over the 16 dimension
// steps, the same online softmax per tile and the same P.V WMMA order per
// output tile. What changes is who does the work and where K/V come from:
//   * block = NP query heads of one KV head x 16 consecutive queries; two waves
//     per head as in attn_kx (wave w computes S^T for keys w*16.. of each tile
//     and keeps O^T dims [128w, 128w + 128)), partners exchange S^T via LDS;
//   * K (f16 in both formats): each wave loads its fragments from the cache
//     straight into WMMA A registers (one 16-byte load per lane and dimension
//     step). The NP heads of one KV head read the same rows at the same time,
//     so the loads hit the cache;
//   * V, f16: V^T fragments with global_load_tr_b128 (gfx12: the load transposes
//     8x8 16-bit blocks across each group of 8 lanes, so lane a supplies the row
//     of key 8*(a/16) + a%8, dims 8*((a/8)%2).., and receives the A-operand
//     column it needs);
//   * V, q8v: dequantized once per block ((f16)q * s, as kv_ld8), 8 keys x 4
//     dims per thread (8 dims in 128-thread blocks), into a double-buffered LDS
//     stage in A-register order, one tile ahead;
//   * one barrier per tile: the S^T exchange (double-buffered by tile parity),
//     which also publishes the next q8v V stage and keeps the block in step;
//   * scores stay unscaled: scale = 1/16 is a power of two, so
//     (s - m) * (scale * log2 e) rounds exactly like (s*scale - m*scale) * log2 e,
//     the expression __expf evaluates; exp2(-inf) = 0 replaces the -inf
//     selects (key 0 is visible to every query, so m is finite after tile 0);
//     tiles that every query of the block sees completely skip the mask;
//   * a query's tiles past its own position are fully masked: alpha = 1 and
//     p = 0 make them exact no-ops, so the block runs to its last query.
// Keys past n_keys - 1 read the last valid row (p = 0 for them).
// ---------------------------------------------------------------------------
typedef __fp16 kg_hv8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ half8 kg_ld_tr(const _Float16 *p) {
    const kg_hv8 v = __builtin_amdgcn_global_load_tr_b128_v8f16((__attribute__((address_space(1))) kg_hv8 *)(p));
    half8 r;
    __builtin_memcpy(&r, &v, 16);
    return r;
}
template <int NP, bool VQ8>
__device__ __forceinline__ void attn_kg_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                                             int q_stride, const int *pos, int n_tok, float scale, int h0) {
    const int tb = kv.tab0;
    __shared__ float8 Sx[2][NP * 2][32];
    __shared__ half8 Vf[VQ8 ? 2 : 1][2][VQ8 ? 16 : 1][32];  // q8v: [buffer][key half][dim tile][lane]
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int w = wid & 1, dh = w * (FA_D / 2);
    const int h = blockIdx.y * NP + (wid >> 1) + h0;
    const int q0 = blockIdx.x * 16;
    const int kvh = h / (n_head / n_kv);
    const int pos0 = pos[0];
    const int my_q = q0 + lane % 16;
    const int my_qc = min(my_q, n_tok - 1);
    const int my_pos = pos0 + my_qc;
    half8 qf[FA_D / 16];
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }
    constexpr int DT = FA_D / 32;
    float8 o[DT];
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    const int n_keys = pos0 + min(q0 + 16, n_tok);  // keys of the block's last query + 1
    const int full_end = pos0 + q0;                 // tiles ending at or before this key are unmasked
    const _Float16 *Kc = (const _Float16 *)kv.k + (size_t)kvh * FA_D;
    const int rs = n_kv * FA_D;                     // elements per pool row
    const int krow = w * 16 + lane % 16;            // S^T: this lane's key row in the tile
    const int kcol = 8 * (lane / 16);
    const float cexp = scale * 1.44269504088896340736f;
    // f16 V: transposed loads (key row, first dim of this lane's 16 bytes)
    const _Float16 *Vc = (const _Float16 *)kv.v + (size_t)kvh * FA_D;
    const int vkey = 8 * (lane / 16) + (lane % 8);
    const int vdim = dh + 8 * ((lane >> 3) & 1);
    // q8v V: loader item = key octet x DQ dims (DQ 4: 256 items; 8 in 128-thread blocks)
    constexpr int DQ = 64 * NP >= 256 ? 4 : 8, NDQ = FA_D / DQ;
    const int8_t *Vq = (const int8_t *)kv.v + (size_t)kvh * FA_D;
    const _Float16 *Vs = kv.vs + (size_t)kvh * (FA_D / 32);
    const int rss = n_kv * (FA_D / 32);             // q8v scales per pool row
    const bool ldr = VQ8 && (int)threadIdx.x < 4 * NDQ;
    const int ko = ((int)threadIdx.x / NDQ) & 3, dq = (int)threadIdx.x % NDQ;
    int vraw[8][DQ / 4];
    _Float16 vsc[8];
#define KG_VFETCH(K0)                                                                                     \
    do {                                                                                                  \
        const int dl_ = kv_delta(kv, tb, (K0));                                                           \
        _Pragma("unroll") for (int r = 0; r < 8; ++r) {                                                   \
            const int key = min((K0) + ko * 8 + r, n_keys - 1);                                           \
            const int *src = (const int *)(Vq + (size_t)(key + dl_) * rs + dq * DQ);                      \
            _Pragma("unroll") for (int u = 0; u < DQ / 4; ++u) vraw[r][u] = src[u];                       \
            vsc[r] = Vs[(size_t)(key + dl_) * rss + dq * DQ / 32];                                        \
        }                                                                                                 \
    } while (0)
#define KG_VSTORE(B)                                                                                      \
    do {                                                                                                  \
        _Pragma("unroll") for (int c = 0; c < DQ; ++c) {                                                  \
            half8 col;                                                                                    \
            _Pragma("unroll") for (int r = 0; r < 8; ++r)                                                 \
                col[r] = (_Float16)(int8_t)(vraw[r][c / 4] >> (8 * (c % 4))) * vsc[r];                    \
            const int d = dq * DQ + c;                                                                    \
            Vf[(B)][ko >> 1][d >> 4][(d & 15) + 16 * (ko & 1)] = col;                                      \
        }                                                                                                 \
    } while (0)
    if (ldr) {
        KG_VFETCH(0);
        KG_VSTORE(0);
    }
    int par = 0;
    for (int k0 = 0; k0 < n_keys; k0 += FA_K, par ^= 1) {
        const int dl = kv_delta(kv, tb, k0);  // FA_K-key tiles never straddle a page
        float8 own = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        {
            const _Float16 *kr = Kc + (size_t)(min(k0 + krow, n_keys - 1) + dl) * rs + kcol;
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                // q8v: two batches of K loads (fewer VGPRs: 6 waves per SIMD)
                if (VQ8 && st == 8) __builtin_amdgcn_sched_barrier(0);
                const half8 a = *(const half8 *)(kr + 16 * st);
                own = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[st], own);
            }
        }
        Sx[par][wid][lane] = own;
        __syncthreads();
        const bool more = k0 + FA_K < n_keys;
        if (ldr && more) KG_VFETCH(k0 + FA_K);
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
        // next q8v stage: its buffer was last read in tile k0 - FA_K, before this tile's barrier
        if (ldr && more) KG_VSTORE(par ^ 1);
        if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
        if constexpr (VQ8) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                if (dt == 4) __builtin_amdgcn_sched_barrier(0);
                const half8 a0 = Vf[par][0][w * 8 + dt][lane];
                const half8 a1 = Vf[par][1][w * 8 + dt][lane];
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, pt[0], o[dt]);
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, pt[1], o[dt]);
            }
        } else {
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
    }
#undef KG_VFETCH
#undef KG_VSTORE
    if (my_q >= n_tok) return;
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
// grid (ceil(n_tok / 16), heads / NP), block 64 * NP; NP divides n_head / n_kv
#define KG_ENTRY(NAME, NP, VQ8)                                                                     \
    extern "C" __global__ void __launch_bounds__(64 * NP)                                           \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv,                           \
             int q_stride, const int *pos, int n_tok, float scale, int h0) {                        \
        attn_kg_impl<NP, VQ8>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);          \
    }
KG_ENTRY(attn_kg6, 6, false)
KG_ENTRY(attn_kg4, 4, false)
KG_ENTRY(attn_kg2, 2, false)
KG_ENTRY(attn_kg6_q8v, 6, true)
KG_ENTRY(attn_kg4_q8v, 4, true)
// (no attn_kg2_q8v: with 4 waves per block the q8v stage is slower than attn_kx_q8v)

// item P: f16-output variants of the f16-weight prefill GEMMs (Q4_K_M relaxed mode)
extern "C" __global__ void __launch_bounds__(512) gemm_ch0_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 4, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch1_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 64, 8, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch2_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 1, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch3_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 1, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch4_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 2, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch5_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 32, 4, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch6_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 1, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch7_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 2, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(256) gemm_ch8_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 128, 64, 4, 0, 256, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch9_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 1, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch10_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 256, 32, 8, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch11_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 32, 16, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch12_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 128, 64, 16, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch13_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 2, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch14_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 4, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch15_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 32, 8, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch16_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 2, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch17_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 4, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch18_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 128, 64, 8, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch19_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 1, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch20_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 2, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch21_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 64, 256, 64, 4, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch22_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 256, 256, 32, 1, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }
extern "C" __global__ void __launch_bounds__(512) gemm_ch23_f16(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) { gemm3_impl<QT_F16, 128, 512, 32, 8, 0, 512, true, true>(W, row_bytes, X, Y, ncols, nrows, n_tok, 0); }

// ---- item P: WMMA DeltaNet prefill (end) ----
