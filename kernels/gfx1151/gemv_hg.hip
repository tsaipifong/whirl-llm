// WHIRL device code (gfx1151 / RDNA 3.5): grouped precise decode / verify matmul gemvhg_<type> (v0252-8sdec, PLAN V1-142).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST by kernels/gfx1151/whirl_kernels_gfx1151.hip (after gemv_h.hip, whose templates it
// uses; last so that no existing kernel moves in the code object).
//
// gemvhg_<T>(GvArgs g, X, ncols, n_tok, accumulate): up to 3 matrices that share the f16 input X (the same-input sets of
// Context::matmulGroup: GDN {qkv, z}, attention {q, k, v}, FFN {gate, up}) in ONE launch. The global 16-row tiles are
// numbered matrix by matrix (matrix 1 starts at tile g.b1, matrix 2 at g.b2; 0x7fffffff = absent, as the int8 grouped
// GEMV's GvArgs), the blocks loop over them like gemvh_<T> on gfx1151 (kernels::gemvhGridCap), and each tile runs
// gemvhg_tile<T> - a copy of the body of gemvh_paired_impl's row-tile loop (kernels/gfx1151/gemv_h.hip, TN = 1, without the
// WHIRL_GH_INJECT test branches; the copy keeps the shipped gemvh_<T> machine code untouched) - with the same X, ncols,
// n_tok and accumulate. A row's value depends only on its tile's weights and X (fixed k order, fixed WMMA chain from zero), so
// gemvhg == gemvh_<T> per matrix BIT FOR BIT for any grid and any grouping (kernel test family gemvhg).
// Prior art: hipfire fused_gate_up_mfp4g32_e8.gfx1151.hip / fused_qkvza_mfp4g32_e8.gfx1151.hip (one launch, workgroups
// routed to a matrix by blockIdx range, per-row body verbatim) - progress/v0252-8sdec.PRIOR_ART.md.
// TN = 1 only (n_tok <= 16: every Swift verify of the shipped draft counts); paired types only (mxfp4, iq4_xs, iq4_nl).
#include "kernel_args.h"  // GvArgs (defined once for both kernel sets, arch025 N13; declared dependency)

// one 16-row tile of matrix (W, row_bytes, Y, nrows): gemvh_paired_impl<T, 1>'s per-tile body (same loads, decode, exchange,
// WMMA order and epilogue). Keep in step with kernels/gfx1151/gemv_h.hip; kt family gemvhg compares the two bit for bit.
template <int T>
__device__ __forceinline__ void gemvhg_tile(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                            int ncols, int nrows, int n_tok, int accumulate, int tile) {
    using D = GhQ<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    const int m0 = tile * 16;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    const int role = lane >> 4;  // lanes r and r + 16 decode one k-step each of the pair (2m, 2m + 1)
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    const _Float16 *xr = X + (size_t)min(rl, n_tok - 1) * ncols;
    Regs buf[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) D::load(buf[u], rowp, u, role);
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) D::load(buf[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1, role);
#pragma unroll
                for (int m = 0; m < D::UK / 2; ++m) {
                    if (m % (SB_EVERY / 2) == 0) __builtin_amdgcn_sched_barrier(0);
                    const half16 f = D::frag(buf[u], m, role, sb & 1);
                    const half16 x = gh_xchg(f);  // the partner lane's step: both half-waves hold the same A fragment
                    const half16 a0 = role ? x : f, a1 = role ? f : x;
                    const half16 b0 = ld_frag16(xr + sb * (16 * D::UK) + (2 * m) * 16);
                    const half16 b1 = ld_frag16(xr + sb * (16 * D::UK) + (2 * m + 1) * 16);
                    acc = WMMA16(a0, b0, acc);
                    acc = WMMA16(a1, b1, acc);
                }
            }
        }
    }
    // D[m = weight row][n = token]: lane (m & 1) * 16 + n, element v = m >> 1
    const int t = lane % 16;
    if (t < n_tok) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int orow = m0 + 2 * v + lane / 16;
            if (orow < nrows) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[v] : acc[v];
            }
        }
    }
}

template <int T>
__device__ __forceinline__ void gemvhg_paired_impl(const GvArgs &g, const _Float16 *__restrict__ X, int ncols, int n_tok, int accumulate) {
    const int total = g.b2 != 0x7fffffff ? g.b2 + (g.n2 + 15) / 16 : g.b1 != 0x7fffffff ? g.b1 + (g.n1 + 15) / 16 : (g.n0 + 15) / 16;
    for (int gt = blockIdx.x; gt < total; gt += gridDim.x) {
        // wave-uniform choice of the matrix (one wave per block)
        const uint8_t *W = g.W0;
        uint64_t rb = g.rb0;
        float *Y = g.y0;
        int nr = g.n0, tile = gt;
        if (gt >= g.b2) {
            W = g.W2; rb = g.rb2; Y = g.y2; nr = g.n2; tile = gt - g.b2;
        } else if (gt >= g.b1) {
            W = g.W1; rb = g.rb1; Y = g.y1; nr = g.n1; tile = gt - g.b1;
        }
        gemvhg_tile<T>(W, rb, X, Y, ncols, nr, n_tok, accumulate, tile);
    }
}

#define GHG_ENTRY(SUF, T)                                                                                                             \
    extern "C" __global__ void __launch_bounds__(32) gemvhg_##SUF(GvArgs g, const _Float16 *X, int ncols, int n_tok, int accumulate) { \
        gemvhg_paired_impl<T>(g, X, ncols, n_tok, accumulate);                                                                          \
    }
GHG_ENTRY(mxfp4, QT_MXFP4)
