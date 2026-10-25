// WHIRL device code (gfx1151 / RDNA 3.5): prefill attention "ours-v2" (feat026-attn8s), candidates for
// attn_prefill_wmma / attn_prefill_wmma_s. Opt-in (WHIRL_ATTN_PF2=<config>); the old kernel stays the default.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip after attn.hip (uses FA_K, FA_D,
// FA_VS, fa_h4, fa_vt_off, kv_ld8, kv_delta, KvArgs, WMMA16, CROW, join_halves).
//
// Same arithmetic as attn_prefill_impl<0, 0, SEG> (attn.hip), redistributed over a wave PAIR per 16 queries, so
// the 8060S kernel leaves the 256-VGPR wall (8S-P5: prefetch spilled there) and can keep the next K / V tile in
// flight across LDS-only barriers (strix-llama.cpp fattn-rdna35.cu:100-106, MIT, the idea; code is ours):
//   - scores: wave hf (0 / 1) of the pair computes S^T key tile kt = hf (keys k0 + 16 hf ..) for its 16 queries
//     with the full Q^T in registers and the same WMMA chain st = 0 .. 15 as attn_prefill_wmma -> same bits.
//     (strix splits D here and sums two partial chains in LDS: different bits; we split the key tiles instead.)
//   - softmax: tile max exchanged in LDS (fmaxf: exact, order-free); m_new and alpha computed in both waves from the
//     same values; the row sum keeps the original order 0 + p(kt0, v0..7) + p(kt1, v0..7), + the lane^16 copy:
//     wave 0 sums its eight, wave 1 continues from that partial (LDS) and keeps l.
//   - O^T: wave hf owns dims [128 hf, 128 hf + 128) (dt = 8 hf .. 8 hf + 7): 64 VGPR of accumulators instead of
//     128. Each dt accumulates kt 0 then kt 1 with the same A (V^T fragment, same swizzled LDS reads) and B (P^T via
//     join_halves; the other key tile's P^T read back from LDS, same f16 values) -> same bits.
//   - the tiles walked per query are those of the 128-query block of attn_prefill_wmma (n_keys from the 128-aligned
//     block), so the zero-fill boundary (keys >= n_keys read as 0; gfx11 WMMA output depends on V where P = 0) and
//     the masked tiles are identical also for the 64-query configuration.
// Only difference that can occur in state (not in output): fmaxf(+0, -0) may give the other zero sign for m; m is only
// used through exp(x - m) / exp(m - m_new), where both zero signs give the same value, and in the m == -inf test.
//
// Template knobs (hill-climb, frozen bitwise contract):
//   NQT  query tiles of 16 per block: 4 (64 queries, 8 waves, 256 threads) or 8 (128 queries, 16 waves, 512)
//   PF   1 = the next tile's K / V global loads are issued right after this tile's LDS stores and stay in flight
//        through the compute (needs BAR = 1 to survive the barriers); 0 = load, store, compute (as today)
//   BAR  1 = LDS-only barrier (fence release + s_barrier + fence acquire), 0 = __syncthreads()
//   RS   rescale policy of O. 0 = every tile (as today). 1 = skipped when every alpha of the wave is exactly 1.0f
//        (o * 1.0f == o for every value, so still BITWISE: precise-safe). 2 = lazy max (FlashAttention-4 style,
//        strix-llama.cpp fattn-rdna35.cu:41-42 / :505-506, T = 8 in log2 units): the reference max m only moves when
//        some lane's tile max exceeds m + T; P = exp(s - m) <= 2^8 stays in f16. Changes bits (P and l relative to
//        an older max): balance / fast only, behind the KL / top-1 gate (klgate --mode-env).
// Per mode (owner 2026-10-26): precise = RS 0 / 1 entries only (bitwise == attn_prefill_wmma); balance / fast may
// also use RS 2 entries (suffix z) if the KL / top-1 gate and the accuracy check pass.
// Grid: (ceil(n_tok / (16 NQT)), heads of the launch); block 64 NQT. Arguments as attn_prefill_wmma(_s).

typedef unsigned pf2_u4 __attribute__((ext_vector_type(4)));  // plain vectors stay in registers
typedef unsigned pf2_u2 __attribute__((ext_vector_type(2)));
#define PF2_TLAZY 5.545177444479562f  // 8 * ln 2: the lazy-max threshold in natural-log units (scores are not log2-scaled)
#define PF2_NKB 128  // query block of attn_prefill_wmma (defines n_keys, i.e. the tiles walked and the zero fill)

template <int BAR> __device__ __forceinline__ void pf2_bar() {
    if constexpr (BAR) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
        __syncthreads();
    }
}

template <int NQT, int PF, int BAR, int RS, bool SEG>
__device__ __forceinline__ void attn_pf2_impl(const float *q, const KvArgs &kv, float *out, int n_head, int n_kv,
                                              int q_stride, const int *pos, int n_tok, float scale, int h0, int kb0,
                                              int kb1, float *st_ml) {
    static_assert(NQT == 4 || NQT == 8, "attn_pf2: 4 or 8 query tiles per block");
    constexpr int NTHR = 64 * NQT, FQ = 16 * NQT;
    constexpr int ODT = FA_D / 32;  // O tiles of 16 dims per wave (8)
    const int tb = kv.tab0;
    __shared__ _Float16 Ks[FA_K][FA_D + 8];
    __shared__ _Float16 Vt[FA_D * FA_VS];
    __shared__ float xmx[NQT][2][16];   // tile max per query column, per key tile
    __shared__ float xrs[NQT][32];      // wave 0's partial row sum, per lane
    __shared__ half8 xpt[NQT][2][32];   // P^T of key tile kt, per lane (C layout pieces)
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, tid = threadIdx.x;
    const int qt = wid >> 1, hf = wid & 1;
    const int h = blockIdx.y + h0, kvh = h / (n_head / n_kv);
    const int q0 = blockIdx.x * FQ;
    const int pos0 = pos[0];
    const int my_q = q0 + qt * 16 + lane % 16;
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
    float8 o[ODT];
#pragma unroll
    for (int j = 0; j < ODT; ++j) o[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;

    // n_keys of the attn_prefill_wmma block holding these queries (see the header)
    const int qb0 = (q0 / PF2_NKB) * PF2_NKB;
    const int last_q = min(qb0 + PF2_NKB, n_tok) - 1;
    const int n_keys = pos0 + last_q + 1;
    if constexpr (SEG) {
        if (kb0 >= n_keys) return;  // block-uniform, before any barrier
        if (kb0 > 0) {
            const float *orow = out + (size_t)my_qc * n_head * FA_D + (size_t)h * FA_D;
            m = st_ml[((size_t)my_qc * n_head + h) * 2];
            l = st_ml[((size_t)my_qc * n_head + h) * 2 + 1];
#pragma unroll
            for (int j = 0; j < ODT; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[j][v] = orow[(hf * ODT + j) * 16 + CROW(lane, v)];
        }
    }
    const int k_beg = SEG ? kb0 : 0, k_end = SEG ? min(n_keys, kb1) : n_keys;

    // loader geometry: the attn_prefill_wmma K / V geometry of a 256-thread block. 256 threads: each thread loads
    // both (as today); 512 threads: threads 0-255 load V, 256-511 load K (wave-uniform roles), so the staging
    // registers are 16 VGPR per thread (rb[4]) instead of 32.
    constexpr bool BOTH = NTHR == 256;
    const bool doV = BOTH || tid < 256, doK = BOTH || tid >= 256;
    const int t = BOTH ? tid : (tid & 255);
    const int vd4 = (t % 64) * 4, vkb = t / 64;
    const int vtb = (vkb / 2) * 16 + (vkb % 2) * 4;
    const int vsA = fa_vt_off(vd4, vtb), vsB = fa_vt_off(vd4, vtb + 8);
    const int rstep = n_kv * FA_D;
    const int kro = kvh * FA_D + (t % 32) * 8, vro = kvh * FA_D + vd4;
    constexpr int KO = 0, VO = BOTH ? 4 : 0;
    pf2_u4 rb[BOTH ? 8 : 4];
    auto load = [&](int k0) {
        const int dl = kv_delta(kv, tb, k0);  // FA_K-key tiles never straddle a page
        if (doK) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + t / 32 + 8 * e;
                const int kc = min(key, n_keys - 1);
                const half8 x = kv_ld8<0>(kv.k, kv.ks, (size_t)(unsigned)(kc + dl) * rstep + kro);
                const half8 z = {0, 0, 0, 0, 0, 0, 0, 0};
                rb[KO + e] = __builtin_bit_cast(pf2_u4, key < n_keys ? x : z);
            }
        }
        if (doV) {
#pragma unroll
            for (int r = 0; r < 8; r += 2) {
                const int key = k0 + vkb * 8 + r;
                const int kc0 = min(key, n_keys - 1), kc1 = min(key + 1, n_keys - 1);
                const fa_h4 x0 = *(const fa_h4 *)((const _Float16 *)kv.v + (size_t)(unsigned)(kc0 + dl) * rstep + vro);
                const fa_h4 x1 = *(const fa_h4 *)((const _Float16 *)kv.v + (size_t)(unsigned)(kc1 + dl) * rstep + vro);
                const fa_h4 z = {0, 0, 0, 0};
                const pf2_u2 a = __builtin_bit_cast(pf2_u2, key < n_keys ? x0 : z);
                const pf2_u2 b = __builtin_bit_cast(pf2_u2, key + 1 < n_keys ? x1 : z);
                rb[VO + r / 2] = (pf2_u4){a.x, a.y, b.x, b.y};
            }
        }
    };
    auto store = [&]() {
        if (doK) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = t + e * 256;
                *(pf2_u4 *)&Ks[i / 32][(i % 32) * 8] = rb[KO + e];
            }
        }
        if (doV) {
            fa_h4 vr[8];
#pragma unroll
            for (int r = 0; r < 8; r += 2) {
                const pf2_u4 w = rb[VO + r / 2];
                vr[r] = __builtin_bit_cast(fa_h4, (pf2_u2){w.x, w.y});
                vr[r + 1] = __builtin_bit_cast(fa_h4, (pf2_u2){w.z, w.w});
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                fa_h4 ev, od;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    ev[r] = vr[2 * r][c];
                    od[r] = vr[2 * r + 1][c];
                }
                *(fa_h4 *)&Vt[vsA + c * FA_VS] = ev;
                *(fa_h4 *)&Vt[vsB + c * FA_VS] = od;
            }
        }
    };
    if constexpr (PF) {
        if (k_beg < k_end) load(k_beg);
    }
    // V^T fragment reads, row dt*16 + l (l = lane % 16): chunk = (c ^ 2 (dt & 1)) ^ (l / 8) (as attn_prefill_wmma)
    const int s0 = (lane % 16) >> 3;
    const _Float16 *vlo = Vt + (lane % 16) * FA_VS + s0 * 8;
    const _Float16 *vhi = Vt + (lane % 16) * FA_VS + 8 - s0 * 8;
    for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
        pf2_bar<BAR>();  // everyone is done with the previous tile's Ks / Vt / xpt
        if constexpr (!PF) load(k0);
        store();
        if constexpr (PF) {
            if (k0 + FA_K < k_end) load(k0 + FA_K);  // in flight through this tile's compute
        }
        pf2_bar<BAR>();
        // S^T key tile hf: rows = keys k0 + 16 hf + 8 (lane / 16) + v, column = my query
        float8 sc = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const half16 qb = join_halves(qf[st], lane);
            const half16 a = ld_frag16(&Ks[hf * 16 + lane % 16][16 * st]);
            sc = WMMA16(a, qb, sc);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int key = k0 + hf * 16 + CROW(lane, v);
            const float x = (key <= my_pos) ? sc[v] * scale : -INFINITY;
            sc[v] = x;
            mx = fmaxf(mx, x);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        if (lane < 16) xmx[qt][hf][lane] = mx;
        pf2_bar<BAR>();
        mx = fmaxf(mx, xmx[qt][hf ^ 1][lane % 16]);
        float m_new = fmaxf(m, mx), alpha;
        bool resc = true;  // wave-uniform: rescale this tile
        if constexpr (RS == 2) {
            // both waves of a pair hold the same mx and m per lane -> the same decision
            resc = __any(mx > m + PF2_TLAZY);
            if (!resc) m_new = m;
            alpha = !resc ? 1.f : (m == -INFINITY) ? 0.f : __expf(m - m_new);
        } else {
            alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
            if constexpr (RS == 1) resc = __any(alpha != 1.f);
        }
        half8 pt;
        float rs = 0.f;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const float p = (sc[v] == -INFINITY) ? 0.f : __expf(sc[v] - m_new);
            sc[v] = p;
            pt[v] = (_Float16)p;
        }
        if (hf == 0) {
#pragma unroll
            for (int v = 0; v < 8; ++v) rs += sc[v];
            xrs[qt][lane] = rs;
        }
        xpt[qt][hf][lane] = pt;
        m = m_new;
        pf2_bar<BAR>();
        if (hf == 1) {
            rs = xrs[qt][lane];
#pragma unroll
            for (int v = 0; v < 8; ++v) rs += sc[v];
            rs += __shfl_xor(rs, 16, WARP);
            l = l * alpha + rs;
        }
        if (resc) {
#pragma unroll
            for (int j = 0; j < ODT; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[j][v] *= alpha;
        }
        // O^T (my dims) += V^T P^T, kt = 0 then 1 per dt (as attn_prefill_wmma)
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt) {
            half8 pk;
            if (kt == hf)
                pk = pt;
            else
                pk = xpt[qt][kt][lane];
            const half16 pb = join_halves(pk, lane);
#pragma unroll
            for (int j = 0; j < ODT; ++j) {
                const int dt = hf * ODT + j;
                const int C = (2 * kt) ^ (2 * (dt & 1));
                const half8 lo = *(const half8 *)(vlo + dt * 16 * FA_VS + C * 8);
                const half8 hi = *(const half8 *)(vhi + dt * 16 * FA_VS + C * 8);
                const half16 a = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                o[j] = WMMA16(a, pb, o[j]);
                asm volatile("" : "+v"(o[j]));  // one V fragment in flight at a time
            }
        }
    }
    // l lives in wave 1 of the pair; wave 0 takes it from LDS (both lane halves hold the same value)
    pf2_bar<BAR>();
    if (hf == 1) xrs[qt][lane] = l;
    pf2_bar<BAR>();
    if (hf == 0) l = xrs[qt][lane];
    if (my_q >= n_tok) return;
    if constexpr (SEG) {
        if (k_end < n_keys) {  // not this block's last range: keep the running state
            float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
            if (hf == 1 && lane < 16) {
                st_ml[((size_t)my_q * n_head + h) * 2] = m;
                st_ml[((size_t)my_q * n_head + h) * 2 + 1] = l;
            }
#pragma unroll
            for (int j = 0; j < ODT; ++j)
#pragma unroll
                for (int v = 0; v < 8; ++v) orow[(hf * ODT + j) * 16 + CROW(lane, v)] = o[j][v];
            return;
        }
    }
    const float inv = 1.f / l;
    const float *gate = q + ((size_t)my_q * n_head + h) * q_stride + FA_D;
    float *orow = out + (size_t)my_q * n_head * FA_D + (size_t)h * FA_D;
#pragma unroll
    for (int j = 0; j < ODT; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int d = (hf * ODT + j) * 16 + CROW(lane, v);
            orow[d] = o[j][v] * inv / (1.f + __expf(-gate[d]));
        }
}

// attn_pf2_<cfg>(attn_prefill_wmma arguments) and attn_pf2_<cfg>_s(attn_prefill_wmma_s arguments)
#define PF2_ENTRY(CFG, NQT, PF, BAR, RS, ATTR)                                                                            \
    extern "C" __global__ void __launch_bounds__(64 * NQT) ATTR                                                       \
        attn_pf2_##CFG(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride, const int *pos,  \
                       int n_tok, float scale, int h0) {                                                            \
        attn_pf2_impl<NQT, PF, BAR, RS, false>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0, 0, 0,          \
                                           nullptr);                                                                \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(64 * NQT) ATTR                                                       \
        attn_pf2_##CFG##_s(const float *q, KvArgs kv, float *out, int n_head, int n_kv, int q_stride,               \
                           const int *pos, int n_tok, float scale, int h0, int kb0, int kb1, float *st_ml) {        \
        attn_pf2_impl<NQT, PF, BAR, RS, true>(q, kv, out, n_head, n_kv, q_stride, pos, n_tok, scale, h0, kb0, kb1,       \
                                          st_ml);                                                                   \
    }
#define PF2_NOATTR
// w8: 192-VGPR budget. amdgpu_waves_per_eu(8) alone was ignored (compile 2026-10-26: q8pw8 stayed at 210 VGPRs, 7 waves),
// so the hard cap amdgpu_num_vgpr(192) carries it; at 16 waves / block that is 2 blocks per WGP instead of 1.
#define PF2_W8 __attribute__((amdgpu_waves_per_eu(8, 8), amdgpu_num_vgpr(192)))
// cfg: q4 / q8 = query tiles per block (64 / 128 queries); p = prefetch; s = __syncthreads barriers; w8 = <= 192 VGPR;
// k = RS 1 (skip rescale when all alpha == 1, bitwise); z = RS 2 (lazy max: LOSSY, balance / fast only)
// bitwise (precise-safe) entries:
PF2_ENTRY(q8p, 8, 1, 1, 0, PF2_NOATTR)    // 128 queries, 512 threads, prefetch, LDS barriers (main candidate)
PF2_ENTRY(q4p, 4, 1, 1, 0, PF2_NOATTR)    // 64 queries, 256 threads, prefetch, LDS barriers
PF2_ENTRY(q8, 8, 0, 1, 0, PF2_NOATTR)     // no prefetch (isolates the pair split + LDS barriers)
PF2_ENTRY(q4, 4, 0, 1, 0, PF2_NOATTR)
PF2_ENTRY(q8s, 8, 0, 0, 0, PF2_NOATTR)    // no prefetch, __syncthreads (isolates the pair split alone)
PF2_ENTRY(q8pw8, 8, 1, 1, 0, PF2_W8)      // main candidate at a 192-VGPR budget (2 blocks / WGP)
PF2_ENTRY(q4pw8, 4, 1, 1, 0, PF2_W8)
PF2_ENTRY(q8pk, 8, 1, 1, 1, PF2_NOATTR)   // + skip the rescale when it is a no-op
PF2_ENTRY(q8pw8k, 8, 1, 1, 1, PF2_W8)
// lossy (balance / fast candidates; never selected for precise):
PF2_ENTRY(q8pz, 8, 1, 1, 2, PF2_NOATTR)
PF2_ENTRY(q8pw8z, 8, 1, 1, 2, PF2_W8)
