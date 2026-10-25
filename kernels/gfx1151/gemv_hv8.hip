// WHIRL device code (gfx1151 / RDNA 3.5): precise gemvh_mxfp4 candidates (load / wait schedule) + tune table.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by gfx1151/gemv_h.hip after the GhQ<T> decoders, gh_xchg and the paired / dual bodies.
//
// feat026-gemvh8s (progress/feat026-gemvh8s.PRIOR_ART.md, research/gemvh8s/). Base ISA of the shipped
// gemvh_mxfp4 (gemvh_paired_impl<QT_MXFP4, 1>, scratch/f26gemvh8s/out_q0): per unit (8 k-steps) the next unit's
// weights are loaded under a branch (if sb + DEPTH - 1 < nsb), THEN each group of 2 k-step pairs issues its 8
// activation loads, and every group ends in s_waitcnt vmcnt(0) (trace: L x5 | L128 x8 X x16 W6 M W4 M W2 M W0 M |
// L128 x8 X x16 W6 M W4 M W2 M W0 M). vmcnt retires in issue order, so the first activation wait of a unit also
// waits for the whole weight prefetch of the next unit, and each group drains the queue: 2 exposed memory
// latencies per unit (8 WMMAs). The accumulate epilogue is 8 x (load, vmcnt(0), add, store).
//
// The candidates change only WHEN loads are issued and how far they are waited for:
//  * D   : ring of D units of weights (D - 1 in flight while a unit is multiplied); shipped DEPTH = 2.
//  * XG  : activations one group (2 pairs = 4 k-steps) ahead in a 2-slot ring (1) or loaded at the group start (0).
//  * WP  : weight ring load at the step start (0) or after the step's last WMMA, fenced by sched_barrier(0), into the
//          slot that step just consumed (1; the "prefetch after the use" placement of feat026-prefetch).
//  * WF  : 1 = the weights issued BEFORE the group-0 activations (the shipped order; ablation only).
//  All loads are unconditional with the unit / group index clamped to the last one (a load past the end only refills a
//  slot that is never read): a load under a branch makes the waitcnt pass assume the worst at the join.
//  Epilogue: with accumulate, the 8 old values are loaded first (clamped), then added and stored.
//
// Bit-identity contract (precise): the A fragments are produced by the SAME code as the shipped kernel
// (GhQ<QT_MXFP4>::load / frag with the same role and phase, gh_xchg, the same per-lane select), the B fragment is the
// same ld_frag16 of X, the WMMA sequence is the same: WMMA16(a0, b0, acc), WMMA16(a1, b1, acc) per pair in
// increasing k from zero, and the epilogue's f32 add is the same old + acc. Tokens past n_tok read a valid (clamped)
// row and are never stored. Checked on the GPU by whirl-kernel-test --only gemvhv8 (every candidate vs the shipped
// gemvh_mxfp4 and vs gemm_c0, bitwise). Only TN = 1 (gemvh_mxfp4, n_tok <= 16: the Swift MTP verify uses 4 rows);
// gemvh2_mxfp4 stays the shipped kernel.
//
// Tune table: WHIRL_GVH8_MXFP4 = candidate id (0 = the shipped kernel, the default; gemv_h.hip then defines
// gemvh_mxfp4 exactly as before). A non-zero id makes this file define gemvh_mxfp4. WHIRL_GVH8_DEV=1 exports every
// candidate as gvh8d<id>_mxfp4 (n_tok <= 16, block 32, any grid; the kernel loops over row tiles).

#ifndef WHIRL_GVH8_MXFP4
#define WHIRL_GVH8_MXFP4 0
#endif
#ifndef WHIRL_GVH8_DEV
#define WHIRL_GVH8_DEV 0
#endif

template <int ID> struct Gvh8Cfg;
#define GVH8_CFG(ID_, D_, XG_, WP_, WF_)                                         \
    template <> struct Gvh8Cfg<ID_> {                                           \
        static constexpr int D = D_, XG = XG_, WP = WP_, WF = WF_;              \
    };
//       id  D  XG WP WF   (D = 3 is not offered: every D = 3 build kept the ring in scratch, 256 B, out_q1 2026-10-26)
GVH8_CFG(1, 2, 0, 0, 0)  // shipped depth, activations before the weight load, branchless, loads-first epilogue
GVH8_CFG(2, 4, 0, 0, 0)
GVH8_CFG(3, 2, 1, 0, 0)  // + activations one group ahead
GVH8_CFG(4, 4, 1, 0, 0)
GVH8_CFG(5, 2, 1, 1, 0)  // weight load after the use (fenced), activations ahead
GVH8_CFG(6, 4, 1, 1, 0)
GVH8_CFG(7, 2, 0, 1, 0)  // negative control: weights after the use but activations not ahead (their waits include it)
GVH8_CFG(8, 2, 0, 0, 1)  // ablation: the shipped order (weights first), only branchless + the epilogue
#undef GVH8_CFG
constexpr int kGvh8Ids = 8;

template <int T, int D, int XG, int WP, int WF>
__device__ __forceinline__ void gvh8_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                          int ncols, int nrows, int n_tok, int accumulate) {
    using G = GhQ<T>;
    using Regs = typename G::Regs;
    static_assert(G::UK == 8, "unit = 8 k-steps (2 groups of 2 pairs)");
    constexpr int XS = XG ? 2 : 1;  // activation ring slots (one slot = one group = 4 half16 fragments)
    const int lane = threadIdx.x % WARP;
    const int rl = lane % 16;
    const int role = lane >> 4;  // = gemvh_paired_impl (WHIRL_GH_ROLE_PARITY is a probe-only build option, not used here)
    const int nu = ncols / (16 * G::UK);
    const int ng = 2 * nu;  // groups
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
        const int m0 = tile * 16;
        const int row = min(m0 + rl, nrows - 1);
        const uint8_t *rowp = W + (size_t)row * row_bytes;
        float8 acc = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const _Float16 *xr = X + (size_t)min(rl, n_tok - 1) * ncols;
        Regs w[D];
        half16 xa[XS][4];
        // activations of group gi (k-steps 4 gi .. 4 gi + 3): fragments b0, b1 of pair 2 (gi & 1) and 2 (gi & 1) + 1
        auto load_x = [&](half16 (&dst)[4], int gi) {
            const _Float16 *p = xr + (size_t)gi * 64;
#pragma unroll
            for (int f = 0; f < 4; ++f) dst[f] = ld_frag16(p + 16 * f);
        };
        // prologue: weights of units 0 .. D - 2 (WP 0) or 0 .. D - 1 (WP 1), clamped
#pragma unroll
        for (int u = 0; u < D - 1 + WP; ++u) G::load(w[u], rowp, min(u, nu - 1), role);
        if (XG) load_x(xa[0], 0);
        // one unit step: s = slot (u mod D, a constant after unrolling)
        auto step = [&](int s, int u) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                __builtin_amdgcn_sched_barrier(0);
                if (WF && WP == 0 && g == 0) G::load(w[(s + D - 1) % D], rowp, min(u + D - 1, nu - 1), role);
                if (XG) load_x(xa[(g + 1) % 2], min(2 * u + g + 1, ng - 1));
                else load_x(xa[0], 2 * u + g);
                if (!WF && WP == 0 && g == 0) G::load(w[(s + D - 1) % D], rowp, min(u + D - 1, nu - 1), role);
                __builtin_amdgcn_sched_barrier(0);
                const int xs = XG ? g : 0;
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    const int m = 2 * g + mm;
                    const half16 f = G::frag(w[s], m, role, u & 1);
                    const half16 x = gh_xchg(f);
                    const half16 a0 = role ? x : f, a1 = role ? f : x;
                    acc = WMMA16(a0, xa[xs][2 * mm], acc);
                    acc = WMMA16(a1, xa[xs][2 * mm + 1], acc);
                }
            }
            if (WP) {
                __builtin_amdgcn_sched_barrier(0);
                G::load(w[s], rowp, min(u + D, nu - 1), role);
                __builtin_amdgcn_sched_barrier(0);
            }
        };
        int u0 = 0;
        for (; u0 + D <= nu; u0 += D) {
#pragma unroll
            for (int s = 0; s < D; ++s) step(s, u0 + s);
        }
        // tail (nu % D units; u0 is a multiple of D, so the slots continue)
#pragma unroll
        for (int s = 0; s < D - 1; ++s)
            if (u0 + s < nu) step(s, u0 + s);
        // epilogue, D[m = weight row][n = token]: lane (m & 1) * 16 + n, element v = m >> 1 (as gemvh_paired_impl)
        const int t = rl;
        const size_t yrow = (size_t)min(t, n_tok - 1) * nrows;
        float old[8];
        if (accumulate) {
#pragma unroll
            for (int v = 0; v < 8; ++v) old[v] = Y[yrow + min(m0 + 2 * v + lane / 16, nrows - 1)];
        }
        if (t < n_tok) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int orow = m0 + 2 * v + lane / 16;
                if (orow < nrows) Y[yrow + orow] = accumulate ? old[v] + acc[v] : acc[v];
            }
        }
    }
}

template <int T, int ID>
__device__ __forceinline__ void gvh8_run(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) {
    using C = Gvh8Cfg<ID>;
    gvh8_impl<T, C::D, C::XG, C::WP, C::WF>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
}

#if WHIRL_GVH8_MXFP4
#define GVH8_SEL_ENTRY(SUF, T, ID)                                                                                                          \
    extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, \
                                                               int nrows, int n_tok, int accumulate) {                                      \
        gvh8_run<T, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                                 \
    }
#define GVH8_SEL_ENTRY_X(SUF, T, ID) GVH8_SEL_ENTRY(SUF, T, ID)
GVH8_SEL_ENTRY_X(mxfp4, QT_MXFP4, WHIRL_GVH8_MXFP4)
#endif

#if WHIRL_GVH8_DEV
// development exports gvh8d<id>_mxfp4 (n_tok <= 16) for the gemvhv8 test family and the gemvhbench dev path
#define GVH8_DEV_ENTRY(ID)                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(32) gvh8d##ID##_mxfp4(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,    \
                                                                     int ncols, int nrows, int n_tok, int accumulate) {                     \
        gvh8_run<QT_MXFP4, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                          \
    }
GVH8_DEV_ENTRY(1)
GVH8_DEV_ENTRY(2)
GVH8_DEV_ENTRY(3)
GVH8_DEV_ENTRY(4)
GVH8_DEV_ENTRY(5)
GVH8_DEV_ENTRY(6)
GVH8_DEV_ENTRY(7)
GVH8_DEV_ENTRY(8)
#endif
