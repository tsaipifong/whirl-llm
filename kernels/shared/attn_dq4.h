extern "C" __global__ void __launch_bounds__(256)  // shared split decode attention (both kernel sets, ADR-0005); see the end of this file
    attn_dq4(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos,
             float scale, const int *skip, AwGroups groups) {
    if (skip != nullptr && *skip) return;
    __shared__ __align__(16) uint2 Q8[DQ_C][32];               // col, 8-value chunk: int8 x4 of the even / odd values
    __shared__ float Qsc[DQ_C][8];                             // q scale per col, 32-group
    __shared__ int Qs8[DQ_C][8];                               // 8 * sum of the group's q ints
    __shared__ __align__(16) _Float16 Pw[DQ_NW][16][DQ_C][2];  // wave, key pair, col: p of the pair's keys
    __shared__ __align__(16) _Float16 Sv[DQ_NW][16][8][2];     // wave, key pair, group: V scales of the pair's keys
    __shared__ float Rm[DQ_NW][DQ_C], Rl[DQ_NW][DQ_C];
    __shared__ __align__(16) float Acc[DQ_C][AW_D];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    ADQ_GRID  // g (query group), split, n_split, z (sequence of the group list): the set's block mapping
    const int grp = n_head / n_kv;
    const int t = groups.first[z];
    const int tb = kv_tb(kv, t);
    const int n_pos = pos[t] + 1;
    int per = dq_per(n_pos, n_split);
    int b0 = split * per, b1 = min(n_pos, b0 + per);
    if (kv.win != 0 && n_split > 1) {  // MTP draft window: same key set as attn_wsplit1
        const int w = (kv.win & 0xffff) * 64, min_pos = (kv.win >> 16) * 1024;
        const int w0 = max(AW_SINK, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > AW_SINK) {
            if (split == 0) {
                b0 = 0;
                b1 = AW_SINK;
            } else {
                per = dq_per(n_pos - w0, n_split - 1);
                b0 = w0 + (split - 1) * per;
                b1 = min(n_pos, b0 + per);
            }
        }
    }
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    if (b0 >= n_pos) {
        if (threadIdx.x < grp) {
            const int h = g * grp + threadIdx.x;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    {
        // q -> int8 per 32 values: wave = column, lane = values [8 lane, 8 lane + 8)
        const int c = wid;
        float x[8];
        if (c < grp) {
            const float *qr = q + ((size_t)t * n_head + g * grp + c) * q_stride + 8 * lane;
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            x[0] = a.x;
            x[1] = a.y;
            x[2] = a.z;
            x[3] = a.w;
            x[4] = b.x;
            x[5] = b.y;
            x[6] = b.z;
            x[7] = b.w;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = 0.f;
        }
        float am = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) am = fmaxf(am, fabsf(x[e]));
        am = fmaxf(am, __shfl_xor(am, 1, WARP));
        am = fmaxf(am, __shfl_xor(am, 2, WARP));
        const float qs = am * (1.f / 127.f), inv = am > 0.f ? 127.f / am : 0.f;
        int qi[8], sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qi[e] = max(-127, min(127, (int)rintf(x[e] * inv)));
            sum += qi[e];
        }
        sum += __shfl_xor(sum, 1, WARP);
        sum += __shfl_xor(sum, 2, WARP);
        uint2 pk;
        pk.x = (uint32_t)(qi[0] & 255) | (uint32_t)(qi[2] & 255) << 8 | (uint32_t)(qi[4] & 255) << 16 | (uint32_t)(qi[6] & 255) << 24;
        pk.y = (uint32_t)(qi[1] & 255) | (uint32_t)(qi[3] & 255) << 8 | (uint32_t)(qi[5] & 255) << 16 | (uint32_t)(qi[7] & 255) << 24;
        Q8[c][lane] = pk;
        if ((lane & 3) == 0) {
            Qsc[c][lane >> 2] = qs;
            Qs8[c][lane >> 2] = 8 * sum;
        }
    }
    __syncthreads();
    const float sl2 = scale * 1.4426950408889634f;  // scores in base-2 units
    float m[DQ_C], ls[DQ_C], o[DQ_C][8];
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        m[c] = -INFINITY;
        ls[c] = 0.f;
#pragma unroll
        for (int d = 0; d < 8; ++d) o[c][d] = 0.f;
    }
    const uint32_t M4 = 0x0f0f0f0fu;
    const int gl = lane >> 2;  // the 32-group of this lane's P.V dims
    for (int k0 = b0 + wid * 32; k0 < b1; k0 += DQ_NW * 32) {
        const int dl = kv_delta(kv, tb, k0);  // 32-key tiles never straddle a page
        const int key = k0 + lane;
        const bool valid = key < b1;
        const size_t r = (size_t)(min(key, b1 - 1) + dl) * n_kv + g;
        uint4 kw[8];
        const uint4 *kp = (const uint4 *)((const uint8_t *)kv.k + r * (AW_D / 2));
#pragma unroll
        for (int i = 0; i < 8; ++i) kw[i] = kp[i];
        const uint4 ksw = *(const uint4 *)(kv.ks + r * (AW_D / 32));
        const uint4 vsw = *(const uint4 *)(kv.vs + r * (AW_D / 32));
        uint32_t vw[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const size_t rv = (size_t)(min(k0 + j, b1 - 1) + dl) * n_kv + g;
            vw[j] = *(const uint32_t *)((const uint8_t *)kv.v + rv * (AW_D / 2) + 4 * lane);
        }
        // scores (lane = key). zo: an opaque 0 so the q reads stay in the loop (hoisted, the
        // 512 q words and their scales would spill)
        int zo;
        asm volatile("s_mov_b32 %0, 0" : "=s"(zo));
        float s[DQ_C];
#pragma unroll
        for (int c = 0; c < DQ_C; ++c) s[c] = 0.f;
        _Float16 ksh[8];
        __builtin_memcpy(ksh, &ksw, 16);
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            const uint32_t w4[4] = {kw[gi].x, kw[gi].y, kw[gi].z, kw[gi].w};
            uint32_t lo[4], hi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo[j] = w4[j] & M4;
                hi[j] = (w4[j] >> 4) & M4;
            }
            const float sk = (float)ksh[gi];
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) {
                const uint4 qa = *(const uint4 *)&Q8[c][4 * gi + zo], qb = *(const uint4 *)&Q8[c][4 * gi + 2 + zo];
                int S = 0;
                S = __builtin_amdgcn_sudot4(true, (int)qa.x, false, (int)lo[0], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.y, false, (int)hi[0], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.z, false, (int)lo[1], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qa.w, false, (int)hi[1], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.x, false, (int)lo[2], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.y, false, (int)hi[2], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.z, false, (int)lo[3], S, false);
                S = __builtin_amdgcn_sudot4(true, (int)qb.w, false, (int)hi[3], S, false);
                s[c] = fmaf((float)(S - Qs8[c][gi + zo]), Qsc[c][gi + zo] * sk, s[c]);
            }
            __builtin_amdgcn_sched_barrier(0);  // one group's q reads in flight at a time (no spills)
        }
        // online softmax (base 2) per column; the running max is wave-uniform
        float p[DQ_C];
#pragma unroll
        for (int c = 0; c < DQ_C; ++c) {
            const float x = valid ? s[c] * sl2 : -INFINITY;
            const float mt = warp_max(x);
            const float mn = fmaxf(m[c], mt);
            const float alpha = __builtin_amdgcn_exp2f(m[c] - mn);  // m = -inf: 0
            p[c] = __builtin_amdgcn_exp2f(x - mn);
            ls[c] = ls[c] * alpha + p[c];
            m[c] = mn;
#pragma unroll
            for (int d = 0; d < 8; ++d) o[c][d] *= alpha;
        }
        {
            _Float16 vsh[8];
            __builtin_memcpy(vsh, &vsw, 16);
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) Pw[wid][lane >> 1][c][lane & 1] = (_Float16)p[c];
#pragma unroll
            for (int gi = 0; gi < 8; ++gi) Sv[wid][lane >> 1][gi][lane & 1] = valid ? vsh[gi] : (_Float16)0.f;
        }
        dq_wave_sync();
        // P.V (lane = dims [8 lane, 8 lane + 8)), key pairs
        const dq_h2 off = (dq_h2){(_Float16)1032.0f, (_Float16)1032.0f};
#pragma unroll 4
        for (int pr = 0; pr < 16; ++pr) {
            const uint4 pa = *(const uint4 *)&Pw[wid][pr][0][0], pb = *(const uint4 *)&Pw[wid][pr][4][0];
            uint32_t svw;
            __builtin_memcpy(&svw, &Sv[wid][pr][gl][0], 4);
            const dq_h2 sv2 = dq_bits(svw);
            const uint32_t pu[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
            dq_h2 w2[DQ_C];
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) w2[c] = dq_bits(pu[c]) * sv2;
            const uint32_t a0 = vw[2 * pr], a1 = vw[2 * pr + 1];
            const uint32_t lo0 = a0 & M4, hi0 = (a0 >> 4) & M4, lo1 = a1 & M4, hi1 = (a1 >> 4) & M4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t sel = (uint32_t)j | 0x0c00u | (uint32_t)(4 + j) << 16 | 0x0c000000u;
                const dq_h2 ne = dq_bits(__builtin_amdgcn_perm(lo1, lo0, sel) | 0x64006400u) - off;  // dim 2j: (n0 - 8, n1 - 8)
                const dq_h2 no = dq_bits(__builtin_amdgcn_perm(hi1, hi0, sel) | 0x64006400u) - off;  // dim 2j + 1
#pragma unroll
                for (int c = 0; c < DQ_C; ++c) {
                    o[c][2 * j] = __builtin_amdgcn_fdot2(w2[c], ne, o[c][2 * j], false);
                    o[c][2 * j + 1] = __builtin_amdgcn_fdot2(w2[c], no, o[c][2 * j + 1], false);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        dq_wave_sync();
    }
    // merge the waves' partials in wave order
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        const float l = warp_sum(ls[c]);
        if (lane == 0) {
            Rm[wid][c] = m[c];
            Rl[wid][c] = l;
        }
    }
    __syncthreads();
    float fac[DQ_C];
#pragma unroll
    for (int c = 0; c < DQ_C; ++c) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < DQ_NW; ++w) M = fmaxf(M, Rm[w][c]);
        fac[c] = m[c] == -INFINITY ? 0.f : __builtin_amdgcn_exp2f(m[c] - M);
    }
    for (int w = 0; w < DQ_NW; ++w) {
        if (wid == w) {
#pragma unroll
            for (int c = 0; c < DQ_C; ++c) {
                float4 *ap = (float4 *)&Acc[c][8 * lane];
                const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 x0 = w == 0 ? z4 : ap[0], x1 = w == 0 ? z4 : ap[1];
                x0.x += fac[c] * o[c][0];
                x0.y += fac[c] * o[c][1];
                x0.z += fac[c] * o[c][2];
                x0.w += fac[c] * o[c][3];
                x1.x += fac[c] * o[c][4];
                x1.y += fac[c] * o[c][5];
                x1.z += fac[c] * o[c][6];
                x1.w += fac[c] * o[c][7];
                ap[0] = x0;
                ap[1] = x1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < grp) {
        const int c = threadIdx.x, h = g * grp + c;
        float M = -INFINITY, L = 0.f;
        for (int w = 0; w < DQ_NW; ++w) M = fmaxf(M, Rm[w][c]);
        for (int w = 0; w < DQ_NW; ++w)
            if (Rm[w][c] != -INFINITY) L += __builtin_amdgcn_exp2f(Rm[w][c] - M) * Rl[w][c];
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = M * 0.6931471805599453f;  // natural-log units
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = L;
    }
    for (int i = threadIdx.x; i < grp * AW_D; i += blockDim.x) {
        const int c = i / AW_D, d = i % AW_D, h = g * grp + c;
        part_acc[t * acc_q + ((size_t)split * n_head + h) * AW_D + d] = Acc[c][d];
    }
}

// kernels/shared/attn_dq4.h -- the attn_dq4 kernel (split-K decode / verify attention over the paged KV cache, one
// partial (m, l, acc) per split for attn_combine). One copy for both kernel sets: kernels/attn.hip (gfx1201) and
// kernels/gfx1151/attn.hip; ADR-0005 / arch025-sc11e (the two copies differed only in the block mapping). File scope,
// included at the same TU position as the kernel it replaces. Before the #include the includer defines ADQ_GRID, the
// statements declaring g, split, n_split and z from the block index (gfx1201: query group fastest so the verify rows
// of a sequence share K / V tiles in L2; gfx1151: blockIdx.x / .y / .z as they are), and #undefs it afterwards.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
