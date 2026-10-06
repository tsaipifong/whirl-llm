// WHIRL device code (gfx1201 / RDNA 4): MXFP4 routed experts (Ornith-1.5-35B-A3B MXFP4):
// generic MoE entries for MXFP4, whole-block decode experts, MXFP4 x fp8 grouped expert GEMM.
// SPDX-License-Identifier: Apache-2.0
// Ported from this project's item ORN prototype patch (itemORN/orn_patch.diff, kernels.hip
// "item ORN" blocks); the f32 epilogue of gemm8_moe_impl is new (vector stores, same values).
// Not self-contained: included last by whirl_kernels.hip (uses moe.hip, gemv_q.hip,
// gemm_prefill.hip and gemm_fp8.hip helpers).

// MXFP4 experts on the generic paths: int8 decode (moe_gu / moe_down) and f16 WMMA
// prefill (gemm_moe / gemm_moe32), over the load-time MXFP4 layout (136-byte superblocks).
MOE_ENTRIES(mxfp4, QT_MXFP4)

// ---------------------------------------------------------------------------
// MXFP4 decode experts with whole-block units: lane l owns 32-value blocks l, l + 32, ...
// of the row (one 16-byte qs load + exponent per block). Same launch shapes and arguments
// as moe_gu_<T> / moe_down_<T>; decode and verify (n <= 16) both run it, so MTP == plain.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int mxw_dot(const uint8_t *row, int b, const int8_t *x) {
    const uint8_t *sb = row + (size_t)(b >> 3) * 136;
    const int ib = b & 7;
    const uint4 q = *(const uint4 *)(sb + 8 + 16 * ib);
    const int4 xl = *(const int4 *)(x + b * 32);
    const int4 xh = *(const int4 *)(x + b * 32 + 16);
    int a = dp4(mx_lut4(q.x & 0x0F0F0F0F), xl.x, 0);
    a = dp4(mx_lut4(q.y & 0x0F0F0F0F), xl.y, a);
    a = dp4(mx_lut4(q.z & 0x0F0F0F0F), xl.z, a);
    a = dp4(mx_lut4(q.w & 0x0F0F0F0F), xl.w, a);
    a = dp4(mx_lut4((q.x >> 4) & 0x0F0F0F0F), xh.x, a);
    a = dp4(mx_lut4((q.y >> 4) & 0x0F0F0F0F), xh.y, a);
    a = dp4(mx_lut4((q.z >> 4) & 0x0F0F0F0F), xh.z, a);
    a = dp4(mx_lut4((q.w >> 4) & 0x0F0F0F0F), xh.w, a);
    return a;
}
__device__ __forceinline__ float mxw_scale(const uint8_t *row, int b) { return e8m0_half(row[(size_t)(b >> 3) * 136 + (b & 7)]); }

extern "C" __global__ void __launch_bounds__(256)
    moe_gu_mxfp4w(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes, int ff, int nrows,
                  const int8_t *__restrict__ xq, const float *__restrict__ xd, const int *__restrict__ ids,
                  float *__restrict__ y1, float *__restrict__ y2, int C, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / K, e = ids[p];
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    float acc = 0.f;
    for (int b = lane; b < C / 32; b += WARP) acc += mxw_scale(wr, b) * dt[b] * (float)mxw_dot(wr, b, xt);
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

extern "C" __global__ void __launch_bounds__(256)
    moe_down_mxfp4w(const uint8_t *__restrict__ W, uint64_t row_bytes, int E, const int8_t *__restrict__ xq,
                    const float *__restrict__ xd, const int *__restrict__ ids, const float *__restrict__ w,
                    const float *__restrict__ sg, const float *__restrict__ ysh, float *__restrict__ x, int F, int K,
                    const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    const int nb = F / 32;
    float acc = 0.f;
    for (int u = lane; u < K * nb; u += WARP) {
        const int k = u / nb, b = u - k * nb;
        const int p = t * K + k;
        const uint8_t *wr = W + ((size_t)ids[p] * E + row) * row_bytes;
        acc += w[p] * (mxw_scale(wr, b) * xd[(size_t)p * nb + b] * (float)mxw_dot(wr, b, xq + (size_t)p * F));
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        float v = acc;
        if (ysh != nullptr) v += sg[t] * ysh[(size_t)t * E + row];
        x[(size_t)t * E + row] += v;
    }
}

// ---------------------------------------------------------------------------
// Prefill MoE for MXFP4 experts on the fp8 path. Tile blockIdx.x of moe_route =
// (expert e, first pos s, count c <= BN) computes rows [m0, m0 + BM) of expert e for
// pos s .. s + c from row-major fp8 activations X8 [pos][ncols] with a per-pos scale
// sx[pos]; body = gemm8_impl (folded E8M0 block exponent: w8 = e2m1 * 2^-(ref - e),
// Y = acc * sx * 2^(ref - 127)). W2 != null: one launch for two weight sets of the same
// shape (FFN gate rows -> Y1, up rows -> Y2), blockIdx.y in [0, 2 * ceil(rows_e / BM)).
// Each accumulator sees the WMMA sequence in k of gemm8_impl, independent of the tile's
// other tokens and of BN (token tile 32 / 64 give the same output).
// RBF (gemm8r_*): 1-D grid with the row block fastest, so the blocks sharing a tile's X8
// rows run together (L2 hits instead of re-reads; the down projection has 16 row blocks
// per tile at Ornith shapes); SKIP: a wave's 16-token fragments at or past the tile's
// count skip their WMMAs (those accumulators are never stored). Neither changes any
// stored value.
// Epilogue: lane holds 8 consecutive rows of one token -> one 16-byte (f16, OH) or two
// 16-byte (f32) stores per lane and accumulator; the prototype's f32 epilogue wrote 8
// scattered 4-byte values (the down projection's 7 ms at Ornith 2k shapes). Values are
// the same expressions, so both epilogues give the same bits.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WAVES_M, bool OH, bool RBF = false, bool SKIP = false>
__device__ void gemm8_moe_impl(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, uint64_t row_bytes,
                               const uint8_t *__restrict__ ref1, const uint8_t *__restrict__ ref2, int rows_e,
                               const uint8_t *__restrict__ X8a, const float *__restrict__ sxa, float *__restrict__ Y1,
                               float *__restrict__ Y2, int ncols, const int4 *__restrict__ tiles,
                               const int *__restrict__ n_tiles) {
    constexpr int NTH = 256;
    const int nby = (rows_e + BM - 1) / BM;
    int tix, by;
    if (RBF) {
        const int nbt = W2 != nullptr ? 2 * nby : nby;
        tix = (int)blockIdx.x / nbt;
        by = (int)blockIdx.x - tix * nbt;
    } else {
        tix = (int)blockIdx.x;
        by = (int)blockIdx.y;
    }
    if (tix >= *n_tiles) return;
    const int4 tl = tiles[tix];
    const bool second = by >= nby;
    if (second) by -= nby;
    const uint8_t *__restrict__ W = (second ? W2 : W1) + (size_t)tl.x * rows_e * row_bytes;
    const uint8_t *__restrict__ wref = (second ? ref2 : ref1) + (size_t)tl.x * rows_e;
    const uint8_t *__restrict__ X8 = X8a + (size_t)tl.y * ncols;
    const float *__restrict__ sx = sxa + tl.y;
    const int nrows = rows_e, n_tok = tl.z;

    constexpr int WAVES_N = NTH / 32 / WAVES_M;
    constexpr int TM = BM / 16 / WAVES_M, TN = BN / 16 / WAVES_N;
    constexpr int LD = BK + 16;  // bytes; 16-byte aligned rows
    constexpr int UPR = BK / 32;  // 32-value weight units per row per chunk
    constexpr int A_UNITS = BM * UPR;
    constexpr int A_PER = (A_UNITS + NTH - 1) / NTH;
    constexpr int CPR = BK / 16;  // 16-byte activation pieces per row per chunk
    constexpr int B_PER = BN * CPR / NTH;
    static_assert((A_UNITS % NTH == 0 || NTH % A_UNITS == 0) && BN * CPR % NTH == 0, "tile/threads");
    __shared__ __attribute__((aligned(16))) uint8_t As[BM * LD];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[BN * LD];
    __shared__ uint32_t tab[16][2];
    if (threadIdx.x < 32) tab[threadIdx.x >> 1][threadIdx.x & 1] = k_fp8tab[threadIdx.x >> 1][threadIdx.x & 1];

    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    const int wm = wid % WAVES_M, wn = wid / WAVES_M;
    const int m0 = by * BM;

    float8 acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};

    uint4 rq[A_PER];
    uint32_t re[A_PER];
    int rref[A_PER];
    uint4 rb[B_PER];
#pragma unroll
    for (int e = 0; e < A_PER; ++e) {
        const int r = (threadIdx.x + e * NTH) / UPR;
        rref[e] = (r < BM && m0 + r < nrows) ? wref[m0 + r] : 0;
    }
#define G8M_FETCH(k0_)                                                                                     \
    do {                                                                                                   \
        _Pragma("unroll") for (int e = 0; e < A_PER; ++e) {                                                \
            const int i = threadIdx.x + e * NTH;                                                           \
            const int r = i / UPR, u = i % UPR;                                                            \
            const int k = (k0_) + u * 32;                                                                  \
            if (i < A_UNITS && m0 + r < nrows) {                                                           \
                const uint8_t *sb = W + (size_t)(m0 + r) * row_bytes + (size_t)(k >> 8) * 136;             \
                rq[e] = *(const uint4 *)(sb + 8 + ((k & 255) >> 5) * 16);                                  \
                re[e] = sb[(k & 255) >> 5];                                                                \
            } else {                                                                                       \
                rq[e] = make_uint4(0, 0, 0, 0);                                                            \
                re[e] = 0;                                                                                 \
            }                                                                                              \
        }                                                                                                  \
        _Pragma("unroll") for (int e = 0; e < B_PER; ++e) {                                                \
            const int i = threadIdx.x + e * NTH;                                                           \
            const int t = i / CPR, c = i % CPR;                                                            \
            rb[e] = t < n_tok ? *(const uint4 *)(X8 + (size_t)t * ncols + (k0_) + c * 16)                  \
                              : make_uint4(0, 0, 0, 0);                                                    \
        }                                                                                                  \
    } while (0)

    G8M_FETCH(0);
    for (int k0 = 0; k0 < ncols; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < A_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            if (i >= A_UNITS) break;
            const int r = i / UPR, u = i % UPR;
            int d = rref[e] - (int)re[e];
            d = d < 0 ? 0 : (d > 15 ? 15 : d);
            const uint32_t ta = tab[d][0], tb = tab[d][1];
            uint4 lo, hi;
            lo.x = mx_fp8x4(rq[e].x & 0x0F0F0F0Fu, ta, tb);
            lo.y = mx_fp8x4(rq[e].y & 0x0F0F0F0Fu, ta, tb);
            lo.z = mx_fp8x4(rq[e].z & 0x0F0F0F0Fu, ta, tb);
            lo.w = mx_fp8x4(rq[e].w & 0x0F0F0F0Fu, ta, tb);
            hi.x = mx_fp8x4((rq[e].x >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.y = mx_fp8x4((rq[e].y >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.z = mx_fp8x4((rq[e].z >> 4) & 0x0F0F0F0Fu, ta, tb);
            hi.w = mx_fp8x4((rq[e].w >> 4) & 0x0F0F0F0Fu, ta, tb);
            *(uint4 *)&As[r * LD + u * 32] = lo;
            *(uint4 *)&As[r * LD + u * 32 + 16] = hi;
        }
#pragma unroll
        for (int e = 0; e < B_PER; ++e) {
            const int i = threadIdx.x + e * NTH;
            *(uint4 *)&Bs[(i / CPR) * LD + (i % CPR) * 16] = rb[e];
        }
        __syncthreads();
        if (k0 + BK < ncols) G8M_FETCH(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            fp8x8 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = *(const fp8x8 *)&As[((wm * TM + i) * 16 + lane % 16) * LD + kk + 8 * (lane / 16)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = *(const fp8x8 *)&Bs[((wn * TN + j) * 16 + lane % 16) * LD + kk + 8 * (lane / 16)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    if (!SKIP || (wn * TN + j) * 16 < n_tok)
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[i], b[j], acc[i][j]);
        }
    }
#undef G8M_FETCH
    float *__restrict__ Yb = second ? Y2 : Y1;  // [pos][rows_e], f32 or (OH) f16 elements
    const bool vec = (nrows & 7) == 0;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = (wn * TN + j) * 16 + lane % 16;
        if (t >= n_tok) continue;
        const float st = sx[t];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row0 = m0 + (wm * TM + i) * 16 + 8 * (lane / 16);
            float y[8];
#pragma unroll
            for (int v = 0; v < 8; ++v) y[v] = row0 + v < nrows ? acc[i][j][v] * ldexpf(st, (int)wref[row0 + v] - 127) : 0.f;
            if (OH) {
                _Float16 *dh = (_Float16 *)Yb + ((size_t)tl.y + t) * nrows + row0;
                if (vec && row0 + 8 <= nrows) {
                    half8 hv;
#pragma unroll
                    for (int v = 0; v < 8; ++v) hv[v] = (_Float16)y[v];
                    *(half8 *)dh = hv;
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (row0 + v < nrows) dh[v] = (_Float16)y[v];
                }
            } else {
                float *df = Yb + ((size_t)tl.y + t) * nrows + row0;
                if (vec && row0 + 8 <= nrows) {
                    *(float4 *)df = make_float4(y[0], y[1], y[2], y[3]);
                    *(float4 *)(df + 4) = make_float4(y[4], y[5], y[6], y[7]);
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        if (row0 + v < nrows) df[v] = y[v];
                }
            }
        }
    }
}

// (BM, BN, BK, WAVES_M): token tile 64 (gemm8_moe*) and 32 (gemm8_moe32*); h = f16 output
#define GEMM8_MOE_ENTRY(NAME, BM, BN, BK, WM, OH, RBF, SKIP)                                                     \
    extern "C" __global__ void __launch_bounds__(256)                                                            \
        NAME(const uint8_t *W1, const uint8_t *W2, uint64_t row_bytes, const uint8_t *ref1, const uint8_t *ref2,  \
             int rows_e, const uint8_t *X8, const float *sx, float *Y1, float *Y2, int ncols, const int4 *tiles,   \
             const int *n_tiles) {                                                                               \
        gemm8_moe_impl<BM, BN, BK, WM, OH, RBF, SKIP>(W1, W2, row_bytes, ref1, ref2, rows_e, X8, sx, Y1, Y2,     \
                                                      ncols, tiles, n_tiles);                                    \
    }
GEMM8_MOE_ENTRY(gemm8_moe, 128, 64, 128, 2, false, false, false)
GEMM8_MOE_ENTRY(gemm8_moe32, 128, 32, 128, 4, false, false, false)
GEMM8_MOE_ENTRY(gemm8_moeh, 128, 64, 128, 2, true, false, false)
GEMM8_MOE_ENTRY(gemm8_moe32h, 128, 32, 128, 4, true, false, false)
// row block fastest 1-D grid (n_tile_slots * nbt blocks, nbt = 2 * nby with W2 else nby) and
// 16-token fragments past the tile's count skipped: same accumulators, same stores
GEMM8_MOE_ENTRY(gemm8r_moe, 128, 64, 128, 2, false, true, true)
GEMM8_MOE_ENTRY(gemm8r_moe32, 128, 32, 128, 4, false, true, true)
GEMM8_MOE_ENTRY(gemm8r_moeh, 128, 64, 128, 2, true, true, true)
GEMM8_MOE_ENTRY(gemm8r_moe32h, 128, 32, 128, 4, true, true, true)
#undef GEMM8_MOE_ENTRY

// x8[pos][c] = e4m3(x[perm[pos] / K][c] * 448 / amax), sx[pos] = amax / 448 (per pos = per
// token, the qact_fp8 values of that token); one 256-thread block per pos, C % 1024 == 0.
extern "C" __global__ void __launch_bounds__(256)
    moe_gather_fp8(const float *__restrict__ x, const int *__restrict__ perm, uint8_t *__restrict__ x8,
                   float *__restrict__ sx, int C, int K) {
    const int pos = blockIdx.x;
    const int t = perm[pos] / K;
    const float *xr = x + (size_t)t * C;
    float m = 0.f;
    for (int i = threadIdx.x * 4; i < C; i += 1024) {
        const float4 v = *(const float4 *)(xr + i);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    m = block_max(m);
    const float inv = m > 0.f ? 448.f / m : 0.f;
    if (threadIdx.x == 0) sx[pos] = m > 0.f ? m / 448.f : 0.f;
    uint8_t *qr = x8 + (size_t)pos * C;
    for (int i = threadIdx.x * 4; i < C; i += 1024) {
        const float4 v = *(const float4 *)(xr + i);
        const float a = fminf(fmaxf(v.x * inv, -448.f), 448.f), b = fminf(fmaxf(v.y * inv, -448.f), 448.f);
        const float c = fminf(fmaxf(v.z * inv, -448.f), 448.f), d = fminf(fmaxf(v.w * inv, -448.f), 448.f);
        int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, p, true);
        *(int *)(qr + i) = p;
    }
}
