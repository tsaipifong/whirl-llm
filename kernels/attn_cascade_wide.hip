// WHIRL device code (gfx1201 only): feat026-cascade-wide, the wide-group prefix attention of the cascade
// (shared-prefix) decode / verify (whirl/ops/cascade.h, whirl/ops/cascade_kernels.h). Included by
// whirl_kernels.hip after attn.hip (KvArgs, kv_ld8, kv_delta, AwGroups, AW_K / AW_D) and attn_cascade.hip.
// SPDX-License-Identifier: Apache-2.0
//
// Why: attn_wsplit2 holds at most 32 columns (one K/V pass for 32 / grp rows: 27B 5, Ornith 4), so with MTP
// (nd + 1 rows per sequence) a second sequence never joins a prefix group on the R9700 and the shared prefix is
// still read once per sequence (feat026-cascade-r1, review C2). This kernel holds up to 128 columns
// (8 waves x 16): every row of up to 21 (27B) / 16 (Ornith) rows reads one K/V tile of the prefix together.
//
// Structure (progress/feat026-cascade-wide.PRIOR_ART.md: llama.cpp fattn-mma RDNA config for D 256 = 8 waves,
// Q in registers, K / V tiles in LDS; FlashInfer runs the shared prefix through a multi-query kernel):
//   * tile layout of attn_prefill_wmma: 256 threads, threads [0, 128) stage the 32-key K tile (row-major,
//     Ks[32][264]), threads [128, 256) the V^T tile (Vt[256][40]): 37.4 KB of LDS (below the ~41 KB
//     occupancy cliff of the attn_wsplit notes); Q^T fragments of a wave's 16 columns live in registers.
//   * column mapping, split ranges, masks and partial layout of attn_wsplit_impl: column c of group z =
//     (row groups.first[z] + c / grp, head g * grp + c % grp); the group's last row sets n_pos and the split
//     size per; keys > the column's position or >= b1 are masked; partials (m, l) / acc in the attn_wsplit1
//     layout (attn_cascade_merge and attn_combine unchanged); an empty split writes (-inf, 0) and no acc.
//   * per-column arithmetic of attn_wsplit_impl (contract off, l = fmaf(l, alpha, rs), the same exp / mask /
//     f16 rounding order, the same K dequantization values): a column's partial is expected to be the SAME
//     BITS as attn_wsplit1 / attn_wsplit2 give it (checked by the kernel test family cascade, not assumed).
//     The o rescale is skipped when alpha == 1 on every lane of the wave (exact no-op: x * 1 == x).
// Arguments: those of attn_wsplit1 (q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip,
// groups); grid (n_kv, n_split, groups), block 256. kv.win must be 0 (the cascade never runs with the MTP
// draft window). head_dim 256, grp * rows <= 128 per group (the host caps rows at 128 / grp).

#define CW_NW 8                 // waves per block
#define CW_COLS (16 * CW_NW)    // columns per block

template <int Q8>
__device__ __forceinline__ void attn_cascade_wide_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head,
                                                       int n_kv, int q_stride, const int *pos, float scale, const int *skip,
                                                       AwGroups groups) {
#pragma clang fp contract(off)
    if (skip != nullptr && *skip) return;
    __shared__ _Float16 Ks[AW_K][AW_D + 8];
    __shared__ _Float16 Vt[AW_D][AW_K + 8];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int grp = n_head / n_kv;
    const int first = groups.first[z], count = groups.count[z];
    const int ncol = min(count * grp, CW_COLS);
    const int tb = kv_tb(kv, first);
    const int n_pos = pos[first + count - 1] + 1;  // the group's last row
    const int per = (n_pos + n_split * 64 - 1) / (n_split * 64) * 64;
    const int b0 = split * per, b1 = min(n_pos, b0 + per);
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    const bool cw = wid * 16 < ncol;  // computing wave
    const int col = wid * 16 + lane % 16;
    const bool cvalid = cw && col < ncol;
    const int colc = cvalid ? col : 0;
    const int t = first + colc / grp, h = g * grp + colc % grp;
    const int my_pos = cvalid ? pos[t] : -1;
    if (b0 >= n_pos) {
        if (cvalid && lane < 16) {
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    // Q^T fragments (B operand) of this lane's column: k = dims 16 st + 8 (lane / 16) .. + 8; columns past
    // ncol are zero (their scores are finite and never written)
    half8 qf[AW_D / 16];
#pragma unroll
    for (int st = 0; st < AW_D / 16; ++st) {
        if (cvalid) {
            const float *qr = q + ((size_t)t * n_head + h) * q_stride + 16 * st + 8 * (lane / 16);
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) qf[st][e] = (_Float16)0.f;
        }
    }
    float8 o[AW_D / 16];
#pragma unroll
    for (int dt = 0; dt < AW_D / 16; ++dt) o[dt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    for (int k0 = b0; k0 < b1; k0 += AW_K) {
        const int dl = kv_delta(kv, tb, k0);  // AW_K-key tiles never straddle a page
        __syncthreads();
        if (threadIdx.x < 128) {
            // K tile, row-major: 32 keys x 32 dim groups of 8, 8 per thread
#pragma unroll
            for (int e = 0; e < AW_K * AW_D / 8 / 128; ++e) {
                const int i = threadIdx.x + e * 128;
                const int kr = i / (AW_D / 8), d8 = (i % (AW_D / 8)) * 8;
                const int key = k0 + kr;
                half8 k8;
                if (key < b1) {
                    k8 = kv_ld8<kv_kf(Q8)>(kv.k, kv.ks, ((size_t)(key + dl) * n_kv + g) * AW_D + d8);
                } else {
#pragma unroll
                    for (int x = 0; x < 8; ++x) k8[x] = (_Float16)0.f;
                }
                *(half8 *)&Ks[kr][d8] = k8;
            }
        } else {
            // V^T tile: thread (kb, d8) loads keys kb*8 .. +8 at dims d8 .. +8, transposes in registers
            const int tv = threadIdx.x - 128;
            const int kb = tv / (AW_D / 8), d8 = (tv % (AW_D / 8)) * 8;
            half8 rows[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int key = k0 + kb * 8 + r;
                if (key < b1) {
                    rows[r] = kv_ld8<kv_vf(Q8)>(kv.v, kv.vs, ((size_t)(key + dl) * n_kv + g) * AW_D + d8);
                } else {
#pragma unroll
                    for (int x = 0; x < 8; ++x) rows[r][x] = (_Float16)0.f;
                }
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                half8 cv;
#pragma unroll
                for (int r = 0; r < 8; ++r) cv[r] = rows[r][c];
                *(half8 *)&Vt[d8 + c][kb * 8] = cv;
            }
        }
        __syncthreads();
        if (!cw) continue;
        // S^T tiles: rows = keys kt*16 + 8*(lane/16) + v, column = this lane's column
        float8 sc[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt) {
            sc[kt] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int st = 0; st < AW_D / 16; ++st) {
                const half8 a = *(const half8 *)&Ks[kt * 16 + lane % 16][16 * st + 8 * (lane / 16)];
                sc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, qf[st], sc[kt]);
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + kt * 16 + 8 * (lane / 16) + v;
                const float x = (key <= my_pos && key < b1) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[AW_K / 16];
#pragma unroll
        for (int kt = 0; kt < AW_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = __builtin_fmaf(l, alpha, rs);
        m = m_new;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
            for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
                for (int v = 0; v < 8; ++v) o[dt][v] *= alpha;
        }
#pragma unroll
        for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
            for (int kt = 0; kt < AW_K / 16; ++kt) {
                const half8 a = *(const half8 *)&Vt[dt * 16 + lane % 16][kt * 16 + 8 * (lane / 16)];
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, pt[kt], o[dt]);
            }
    }
    if (!cvalid) return;
    if (lane < 16) {
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = m;
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = l;
    }
    float *pa = part_acc + t * acc_q + ((size_t)split * n_head + h) * AW_D;
#pragma unroll
    for (int dt = 0; dt < AW_D / 16; ++dt)
#pragma unroll
        for (int v = 0; v < 8; ++v) pa[dt * 16 + 8 * (lane / 16) + v] = o[dt][v];
}
#define CW_ENTRY(NAME, Q8)                                                                                              \
    extern "C" __global__ void __launch_bounds__(256)                                                                   \
        NAME(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos, \
             float scale, const int *skip, AwGroups groups) {                                                           \
        attn_cascade_wide_impl<Q8>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);         \
    }
CW_ENTRY(attn_cascade_wide, 0)
CW_ENTRY(attn_cascade_wide_q8, 1)
CW_ENTRY(attn_cascade_wide_q8v, 2)
#undef CW_ENTRY
#undef CW_COLS
#undef CW_NW
