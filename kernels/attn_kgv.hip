// WHIRL device code (gfx1201 / RDNA 4): prefill attention candidate attn_kgv (feat026-attnkg97).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by whirl_kernels.hip after attn_prefill_kp.hip (uses attn_kg's
// helpers kg_ld_tr, kv_delta, FA_K / FA_D and the half8 / float8 vector types).
//
// attn_kgv = attn_kg (f16 K / V) with the P.V work of each PAIR of query heads shared:
// in attn_kg every wave loads its own V^T fragments (global_load_tr_b128) for 128 output dims
// of one head, so the NP heads of a KV head load every V fragment NP times through the vector
// L0 cache (one A fragment per WMMA, ~half of all kernel loads). Here the 4 waves of a head
// pair (heads 2p, 2p + 1 of the block) split the 256 output dims in quarters for BOTH heads:
// each V fragment is loaded once per quad and feeds two WMMAs (the two heads' P^T tiles), so
// the V loads of the block halve while the WMMA count per wave stays 32 per tile.
//
// Bit-identical to attn_kg (and so to attn_kx / attn_prefill_wmma; precise-safe, every mode):
//   * S^T: unchanged (wave w: head w >> 1, keys 16 * (w & 1) .. of the tile, the same 16-step
//     WMMA chain), exchanged through Sx with the same one barrier per tile;
//   * softmax: unchanged (both waves of a head compute it, as in attn_kg);
//   * P.V: O^T(h) of a query = (O^T * alpha_t) + V_t^T(keys 0..15) P_t0 + V_t^T(keys 16..31) P_t1
//     in the same order per output tile. The P^T tiles and alpha of the partner head come from
//     LDS (written by the head's even wave after its softmax), so the P.V of tile t runs after
//     the barrier of tile t + 1 (or after the loop for the last tile): one tile late, same
//     operations in the same order on the same accumulator. No extra barrier per tile;
//   * the alpha == 1 skip is per head (ballot over the head's lanes); multiplying by 1 is exact.
// LDS: Sx 2 x 2NP x 32 float8 (as attn_kg) + P^T 2 x NP x 2 x 32 half8 + alpha 2 x NP x 32 float.
// grid (ceil(n_tok / 16), heads / NP), block 64 * NP; NP even, divides n_head / n_kv.
//
// SPL (attn_kgw: balance / fast ONLY, not bit-identical): the two waves of a head no longer
// compute the same softmax twice. Both still take the row max over all 32 keys (cheap), but
// wave w evaluates exp2 / the row sum / the f16 P^T only for its own 16 keys and publishes them
// with the P^T tiles; the row sum of the tile becomes (sum of keys 0..15) + (sum of keys 16..31)
// instead of one sequential sum over the 32 keys, so l (the softmax denominator) differs from
// attn_kg in the last f32 bits. Everything else (S^T, m, alpha, P values, the P.V order) is the
// same, so the output differs from attn_kg only through 1 / l: relative ~1e-7. Halves the
// softmax exp2 / convert / add VALU work per wave. Precise never selects it (selector).
// ---------------------------------------------------------------------------
template <int NP, bool SPL = false>
__device__ __forceinline__ void attn_kgv_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                                              int q_stride, const int *pos, int n_tok, float scale, int h0) {
    static_assert(NP % 2 == 0, "attn_kgv: head pairs");
    const int tb = kv.tab0;
    __shared__ float8 Sx[2][NP * 2][32];
    __shared__ half8 Pb[2][NP][2][32];  // [tile parity][head of the block][key half][lane]
    __shared__ float Ab[2][NP][32];     // [tile parity][head of the block][lane]: alpha of that tile
    __shared__ float Lb[NP][32];        // final l per head
    __shared__ float Rb[SPL ? 2 : 1][SPL ? NP : 1][2][32];  // SPL: [parity][head][key half][lane] row-sum halves
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int w = wid & 1;              // S^T key half (as attn_kg)
    const int hl = wid >> 1;            // this wave's softmax head within the block
    const int pr = wid >> 2, qd = wid & 3;  // head pair, output-dim quarter of the pair's P.V
    const int jo = hl & 1;              // this wave's own head within its pair
    const int h = blockIdx.y * NP + hl + h0;
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
    constexpr int DQT = FA_D / 64;  // 16-dim output tiles per head of this wave's quarter
    float8 o[2][DQT];               // [head of the pair][tile]
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int dt = 0; dt < DQT; ++dt) o[j][dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    const int n_keys = pos0 + min(q0 + 16, n_tok);
    const int full_end = pos0 + q0;
    const _Float16 *Kc = (const _Float16 *)kv.k + (size_t)kvh * FA_D;
    const int rs = n_kv * FA_D;
    const int krow = w * 16 + lane % 16;
    const int kcol = 8 * (lane / 16);
    const float cexp = scale * 1.44269504088896340736f;
    const _Float16 *Vc = (const _Float16 *)kv.v + (size_t)kvh * FA_D;
    const int vkey = 8 * (lane / 16) + (lane % 8);
    const int vdim = qd * (FA_D / 4) + 8 * ((lane >> 3) & 1);
    // P^T / alpha of the previous tile for BOTH heads of the pair come from LDS (nothing held in registers across
    // the loop; the head's own copies would cost the VGPRs that keep 6 waves per SIMD)
    float ap = 1.f;  // SPL: this head's alpha of the previous tile (for the delayed l update)
    int kp = -1;     // previous tile's first key (-1: none yet)
    // P.V of the tile starting at key K0 (parity PB) for both heads of the pair
    auto pv = [&](int K0, int PB) {
        const int dl_ = kv_delta(kv, tb, K0);
        if constexpr (SPL) l = l * ap + (Rb[PB][hl][0][lane] + Rb[PB][hl][1][lane]);  // this tile's row sum, one tile late
        half8 pt2[2][2];
        float al[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            pt2[j][0] = Pb[PB][pr * 2 + j][0][lane];
            pt2[j][1] = Pb[PB][pr * 2 + j][1][lane];
            al[j] = Ab[PB][pr * 2 + j][lane];
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
            if (__builtin_amdgcn_ballot_w32(al[j] != 1.f)) {
#pragma unroll
                for (int dt = 0; dt < DQT; ++dt)
#pragma unroll
                    for (int v = 0; v < 8; ++v) o[j][dt][v] *= al[j];
            }
        const _Float16 *vb0 = Vc + (size_t)(min(K0 + vkey, n_keys - 1) + dl_) * rs + vdim;
        const _Float16 *vb1 = Vc + (size_t)(min(K0 + 16 + vkey, n_keys - 1) + dl_) * rs + vdim;
#pragma unroll
        for (int dt = 0; dt < DQT; ++dt) {
            const half8 a0 = kg_ld_tr(vb0 + dt * 16);
            const half8 a1 = kg_ld_tr(vb1 + dt * 16);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                o[j][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, pt2[j][0], o[j][dt]);
                o[j][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, pt2[j][1], o[j][dt]);
            }
        }
    };
    int par = 0;
    for (int k0 = 0; k0 < n_keys; k0 += FA_K, par ^= 1) {
        const int dl = kv_delta(kv, tb, k0);
        float8 own = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        {
            const _Float16 *kr = Kc + (size_t)(min(k0 + krow, n_keys - 1) + dl) * rs + kcol;
#pragma unroll
            for (int st = 0; st < FA_D / 16; ++st) {
                const half8 a = *(const half8 *)(kr + 16 * st);
                own = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[st], own);
            }
        }
        Sx[par][wid][lane] = own;
        __syncthreads();  // publishes S^T of this tile and P^T / alpha of the previous one
        if (kp >= 0) pv(kp, par ^ 1);
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
        if constexpr (SPL) {
            const float8 sw = w == 0 ? sc[0] : sc[1];
            half8 ph;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = __builtin_amdgcn_exp2f((sw[v] - m_new) * cexp);
                rsum += p;
                ph[v] = (_Float16)p;
            }
            rsum += __shfl_xor(rsum, 16, WARP);
            m = m_new;
            // buffer par was last read in the P.V of tile k0 - 2 * FA_K, before this tile's barrier
            Pb[par][hl][w][lane] = ph;
            Rb[par][hl][w][lane] = rsum;
            if (w == 0) Ab[par][hl][lane] = alpha;
            ap = alpha;
        } else {
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
            // buffer par was last read in the P.V of tile k0 - 2 * FA_K, before this tile's barrier
            if (w == 0) {
                Pb[par][hl][0][lane] = pt[0];
                Pb[par][hl][1][lane] = pt[1];
                Ab[par][hl][lane] = alpha;
            }
        }
        kp = k0;
    }
    if constexpr (SPL) {
        __syncthreads();  // the last tile's P^T / alpha / row-sum halves
        if (kp >= 0) pv(kp, par ^ 1);  // also adds the last tile's row sum to l
        if (w == 0) Lb[hl][lane] = l;
        __syncthreads();  // every head's l
    } else {
        if (w == 0) Lb[hl][lane] = l;
        __syncthreads();  // the last tile's P^T / alpha and every head's l
        if (kp >= 0) pv(kp, par ^ 1);
    }
    if (my_q >= n_tok) return;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int hj = blockIdx.y * NP + pr * 2 + j + h0;
        const float lj = j == jo ? l : Lb[pr * 2 + j][lane];
        const float inv = 1.f / lj;
        const float *gate = q + ((size_t)my_q * n_head + hj) * q_stride + FA_D;
        float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)hj * FA_D;
#pragma unroll
        for (int dt = 0; dt < DQT; ++dt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int d = qd * (FA_D / 4) + dt * 16 + 8 * (lane / 16) + v;
                orow[d] = o[j][dt][v] * inv / (1.f + __expf(-gate[d]));
            }
    }
}
// 6 waves per SIMD (2 blocks of 12 waves per WGP for NP 6, as attn_kg6): without the bound the compiler took 241
// VGPRs = 5 waves per SIMD, i.e. one NP-6 block per WGP
#define KGV_ENTRY(NAME, NP, SPL)                                                                   \
    extern "C" __global__ void __launch_bounds__(64 * NP) __attribute__((amdgpu_waves_per_eu(6))) \
        NAME(const float *q, KvArgs kv, float *out, int n_head, int n_kv,                           \
             int q_stride, const int *pos, int n_tok, float scale, int h0) {                        \
        attn_kgv_impl<NP, SPL>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0);        \
    }
KGV_ENTRY(attn_kgv6, 6, false)
KGV_ENTRY(attn_kgv4, 4, false)
KGV_ENTRY(attn_kgv2, 2, false)
// balance / fast only (softmax row sum in two halves; see SPL above)
KGV_ENTRY(attn_kgw6, 6, true)
KGV_ENTRY(attn_kgw4, 4, true)
KGV_ENTRY(attn_kgw2, 2, true)
#undef KGV_ENTRY
