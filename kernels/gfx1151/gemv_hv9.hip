// WHIRL device code (gfx1151 / RDNA 3.5): precise gemvh2_mxfp4 / gemvh_q6_k / gemvh_q4_k candidates (load / wait
// schedule) + tune tables.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by gfx1151/gemv_h.hip after gemv_hv8.hip (GhW / GhQ decoders, gh_xchg, the shipped bodies).
//
// feat026-gemvh8s2 (progress/feat026-gemvh8s2.PRIOR_ART.md, research/gemvh8s/2026-10-26-feat026-gemvh8s2-*.md): the
// gemv_hv8.hip method for the other hot gfx1151 precise GEMVs. Base ISA (scratch/f26gemvh8s/out_q0, ROCm 7.2):
//  * gemvh2_mxfp4 (paired, TN = 2, 200 VGPR): per unit the guarded next-unit weights, then each group of 2 pairs
//    issues 16 activation loads (2 tokens) and ends in vmcnt(0); the second unit of the unrolled pair waits vmcnt(0)
//    right after its weight load (join of the guarded load): 5 drains per 32 WMMA; epilogue 16 x (load, wait 0, store).
//  * gemvh_q6_k (dual, TN = 1, 240 VGPR; the Swift / Ornith LM head): 13 x b128 + 1 x u16 weight loads per unit,
//    then 4 groups of 4 k-steps, each 8 activation loads and a vmcnt(0); plus the join drain: 9 drains per 32 WMMA.
//  * gemvh_q4_k (direct, TN = 1, 188 VGPR; the Swift MTP layer): same structure, 8 drains per 32 WMMA.
// llama.cpp's q6_K mmvq on gfx1151 (RDNA2 table) issues weights then activations too, with 4-byte loads and many
// warps, and no software prefetch on AMD (prior art section 1): no schedule to copy.
//
// Candidate parameters (only WHEN loads are issued and how far they are waited for):
//  * KIND: the decode body: 0 = paired (GhQ<T> one k-step per lane + gh_xchg + select, as gemvh_paired_impl),
//          1 = dual (GhQ<T> both k-steps per lane, as gemvh_dual_impl), 2 = direct (GhW<T>, as gemvh_direct_impl).
//          Paired and dual give every lane the same A fragment bits (FIX-RC4-B), so a q6_k paired candidate is
//          bitwise == the shipped dual gemvh_q6_k; it holds half the weight registers.
//  * D : weight ring depth in units; shipped 2.   * GS: k-steps per activation group (2 = one pair, 4 = two pairs).
//  * XG: activations one group ahead in a 2-slot ring (1) or at the group start (0).
//  * WP: weight load after the unit's last WMMA into the slot it consumed, fenced by sched_barrier(0) (1), or at
//        group 0 after the activation loads (0).   * WF: 1 = weights before the activations (shipped order; ablation).
//  All loads unconditional with clamped unit / group indices (the prologue only when nu > 0); loads-first epilogue
//  (all old values, then add + store).
//
// Bit-identity contract (precise): same load / frag functions with the same role and phase (u & 1), the same
// select, the same WMMA operand order (weights first for paired / dual, activations first for direct) and the same
// k order per token tile from zero, the same old + acc f32 epilogue add; tokens past n_tok and rows past nrows are
// never stored. GPU proof: whirl-kernel-test --only gemvhv9 (every candidate vs the shipped kernel and gemm_c0).
//
// Tune tables (default 0 = the shipped kernel; gemv_h.hip then defines the entry exactly as before):
//   WHIRL_GVH82_MXFP4 = id -> gemvh2_mxfp4,  WHIRL_GVH8_Q6_K = id -> gemvh_q6_k,  WHIRL_GVH8_Q4_K = id -> gemvh_q4_k.
// WHIRL_GVH8_DEV=1 also exports gvh92d<id>_mxfp4 (n <= 32), gvh9d<id>_q6_k and gvh9d<id>_q4_k (n <= 16).

#ifndef WHIRL_GVH82_MXFP4
#define WHIRL_GVH82_MXFP4 0
#endif
#ifndef WHIRL_GVH8_Q6_K
#define WHIRL_GVH8_Q6_K 0
#endif
#ifndef WHIRL_GVH8_Q4_K
#define WHIRL_GVH8_Q4_K 0
#endif

template <int T, int KIND> struct Gvh9B;
template <int T> struct Gvh9B<T, 0> {  // paired = gemvh_paired_impl's decode
    using G = GhQ<T>;
    using R = typename G::Regs;
    static constexpr int UK = G::UK;
    static constexpr bool TRANS = false;  // accumulator D[weight row][token]
    static __device__ __forceinline__ void load(R &r, const uint8_t *rowp, int u, int role) { G::load(r, rowp, u, role); }
    static __device__ __forceinline__ void pair(const R &r, int m, int role, int u, half16 &a0, half16 &a1) {
        const half16 f = G::frag(r, m, role, u & 1);
        const half16 x = gh_xchg(f);
        a0 = role ? x : f;
        a1 = role ? f : x;
    }
    static __device__ __forceinline__ float8 mma(half16 a, half16 x, float8 c) { return WMMA16(a, x, c); }
};
template <int T> struct Gvh9B<T, 1> {  // dual = gemvh_dual_impl's decode
    using G = GhQ<T>;
    struct R { typename G::Regs r0, r1; };
    static constexpr int UK = G::UK;
    static constexpr bool TRANS = false;
    static __device__ __forceinline__ void load(R &r, const uint8_t *rowp, int u, int role) {
        (void)role;
        G::load(r.r0, rowp, u, 0);
        G::load(r.r1, rowp, u, 1);
    }
    static __device__ __forceinline__ void pair(const R &r, int m, int role, int u, half16 &a0, half16 &a1) {
        (void)role;
        a0 = G::frag(r.r0, m, 0, u & 1);
        a1 = G::frag(r.r1, m, 1, u & 1);
    }
    static __device__ __forceinline__ float8 mma(half16 a, half16 x, float8 c) { return WMMA16(a, x, c); }
};
template <int T> struct Gvh9B<T, 2> {  // direct = gemvh_direct_impl's decode
    using G = GhW<T>;
    using R = typename G::Regs;
    static constexpr int UK = G::UK;
    static constexpr bool TRANS = true;  // accumulator D[token][weight row] (activations are the first operand)
    static __device__ __forceinline__ void load(R &r, const uint8_t *rowp, int u, int role) {
        (void)role;
        G::load(r, rowp, u);
    }
    static __device__ __forceinline__ void pair(const R &r, int m, int role, int u, half16 &a0, half16 &a1) {
        (void)role;
        (void)u;
        a0 = G::frag(r, 2 * m);
        a1 = G::frag(r, 2 * m + 1);
    }
    static __device__ __forceinline__ float8 mma(half16 a, half16 x, float8 c) { return WMMA16(x, a, c); }
};

template <int T, int KIND, int TN, int D, int XG, int GS, int WP, int WF>
__device__ __forceinline__ void gvh9_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                          int ncols, int nrows, int n_tok, int accumulate) {
    using B = Gvh9B<T, KIND>;
    using R = typename B::R;
    constexpr int UK = B::UK;
    static_assert(GS % 2 == 0 && UK % GS == 0, "a group is whole pairs of a unit");
    constexpr int NG = UK / GS;  // groups per unit
    static_assert(!XG || NG % 2 == 0, "the 2-slot activation ring needs an even group count per unit");
    constexpr int XS = XG ? 2 : 1;
    const int lane = threadIdx.x % WARP;
    const int rl = lane % 16;
    const int role = lane >> 4;  // = the shipped paired body (WHIRL_GH_ROLE_PARITY is a probe-only option, not used here)
    const int nu = ncols / (16 * UK);
    const int ng = NG * nu;
    for (int tile = blockIdx.x; tile * 16 < nrows; tile += gridDim.x) {
        const int m0 = tile * 16;
        const int row = min(m0 + rl, nrows - 1);
        const uint8_t *rowp = W + (size_t)row * row_bytes;
        float8 acc[TN];
        const _Float16 *xr[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
            xr[j] = X + (size_t)min(j * 16 + rl, n_tok - 1) * ncols;
        }
        R w[D];
        half16 xa[XS][TN][GS];
        // activations of group gi (k-steps GS gi .. GS gi + GS - 1) of every token tile
        auto load_x = [&](half16 (&dst)[TN][GS], int gi) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const _Float16 *p = xr[j] + (size_t)gi * (16 * GS);
#pragma unroll
                for (int f = 0; f < GS; ++f) dst[j][f] = ld_frag16(p + 16 * f);
            }
        };
        // nu == 0 (ncols < 16 UK; no product shape): no prologue loads (min(u, nu - 1) would be unit -1, before the row),
        // as the shipped bodies' u < nsb guard; acc stays 0 (feat026-kcand-r1, review kcand-batch consider 5)
        if (nu > 0) {
#pragma unroll
            for (int u = 0; u < D - 1 + WP; ++u) B::load(w[u], rowp, min(u, nu - 1), role);
            if (XG) load_x(xa[0], 0);
        }
        // one unit: slot s (u mod D, a constant after unrolling); NG u is even, so group g uses activation slot g & 1
        auto step = [&](int s, int u) {
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                __builtin_amdgcn_sched_barrier(0);
                if (WF && WP == 0 && g == 0) B::load(w[(s + D - 1) % D], rowp, min(u + D - 1, nu - 1), role);
                if (XG) load_x(xa[(g + 1) & 1], min(NG * u + g + 1, ng - 1));
                else load_x(xa[0], NG * u + g);
                if (!WF && WP == 0 && g == 0) B::load(w[(s + D - 1) % D], rowp, min(u + D - 1, nu - 1), role);
                __builtin_amdgcn_sched_barrier(0);
                const int xs = XG ? (g & 1) : 0;
#pragma unroll
                for (int mm = 0; mm < GS / 2; ++mm) {
                    const int m = (GS / 2) * g + mm;  // pair index in the unit: k-steps 2m, 2m + 1
                    half16 a0, a1;
                    B::pair(w[s], m, role, u, a0, a1);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[j] = B::mma(a0, xa[xs][j][2 * mm], acc[j]);
                        acc[j] = B::mma(a1, xa[xs][j][2 * mm + 1], acc[j]);
                    }
                }
            }
            if (WP) {
                __builtin_amdgcn_sched_barrier(0);
                B::load(w[s], rowp, min(u + D, nu - 1), role);
                __builtin_amdgcn_sched_barrier(0);
            }
        };
        int u0 = 0;
        for (; u0 + D <= nu; u0 += D) {
#pragma unroll
            for (int s = 0; s < D; ++s) step(s, u0 + s);
        }
#pragma unroll
        for (int s = 0; s < D - 1; ++s)
            if (u0 + s < nu) step(s, u0 + s);
        // epilogue: all old values first (clamped addresses), then add + store
        float old[TN][8];
        if (!B::TRANS) {
            // D[m = weight row][n = token]: lane (m & 1) * 16 + n, element v = m >> 1 (= gemvh_paired_impl / gemvh_dual_impl)
            if (accumulate) {
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v)
                        old[j][v] = Y[(size_t)min(j * 16 + rl, n_tok - 1) * nrows + min(m0 + 2 * v + lane / 16, nrows - 1)];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int t = j * 16 + rl;
                if (t < n_tok) {
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int orow = m0 + 2 * v + lane / 16;
                        if (orow < nrows) Y[(size_t)t * nrows + orow] = accumulate ? old[j][v] + acc[j][v] : acc[j][v];
                    }
                }
            }
        } else {
            // D[token][weight row]: row = lane % 16, token = CROW(lane, v) (= gemvh_direct_impl)
            const int orow = m0 + rl;
            if (accumulate) {
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) old[j][v] = Y[(size_t)min(j * 16 + CROW(lane, v), n_tok - 1) * nrows + min(orow, nrows - 1)];
            }
            if (orow < nrows) {
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int t = j * 16 + CROW(lane, v);
                        if (t < n_tok) Y[(size_t)t * nrows + orow] = accumulate ? old[j][v] + acc[j][v] : acc[j][v];
                    }
            }
        }
    }
}

// ---- tune tables: Gvh9Cfg<kernel, id>; kernel 0 = gemvh2_mxfp4, 1 = gemvh_q6_k, 2 = gemvh_q4_k ----
template <int K, int ID> struct Gvh9Cfg;
#define GVH9_CFG(K_, ID_, KIND_, D_, XG_, GS_, WP_, WF_)                               \
    template <> struct Gvh9Cfg<K_, ID_> {                                              \
        static constexpr int KIND = KIND_, D = D_, XG = XG_, GS = GS_, WP = WP_, WF = WF_; \
    };
//        k  id KIND D XG GS WP WF
// gemvh2_mxfp4 (paired, 2 token tiles)
GVH9_CFG(0, 1, 0, 2, 0, 4, 0, 0)  // shipped depth / group, activations before the weights, branchless, loads-first epilogue
GVH9_CFG(0, 2, 0, 2, 1, 2, 0, 0)  // + activations one pair ahead (ring 2 x 2 tokens x 2 fragments)
GVH9_CFG(0, 3, 0, 2, 1, 4, 0, 0)  // + activations one group ahead (ring 2 x 2 x 4)
GVH9_CFG(0, 4, 0, 2, 1, 2, 1, 0)  // pair ahead + weights after the use
GVH9_CFG(0, 5, 0, 4, 1, 2, 0, 0)  // pair ahead, ring of 4 units
GVH9_CFG(0, 6, 0, 2, 0, 4, 0, 1)  // ablation: shipped order (weights first), only branchless + epilogue
// gemvh_q6_k (shipped kind: dual)
GVH9_CFG(1, 1, 1, 2, 0, 4, 0, 0)  // dual, shipped depth / group, branchless, loads-first epilogue
GVH9_CFG(1, 2, 1, 2, 1, 2, 0, 0)  // dual, activations one pair ahead
GVH9_CFG(1, 3, 1, 2, 1, 4, 0, 0)  // dual, activations one group ahead
GVH9_CFG(1, 4, 0, 2, 1, 4, 0, 0)  // paired (half the weight registers), activations one group ahead
GVH9_CFG(1, 5, 0, 4, 1, 4, 0, 0)  // paired, ring of 4 units
GVH9_CFG(1, 6, 0, 2, 1, 4, 1, 0)  // paired, group ahead, weights after the use
GVH9_CFG(1, 7, 1, 2, 0, 4, 0, 1)  // ablation: dual, shipped order, only branchless + epilogue
// gemvh_q4_k (direct)
GVH9_CFG(2, 1, 2, 2, 0, 4, 0, 0)  // shipped depth / group, activations before the weights, branchless, loads-first epilogue
GVH9_CFG(2, 2, 2, 2, 1, 2, 0, 0)  // activations one pair ahead
GVH9_CFG(2, 3, 2, 2, 1, 4, 0, 0)  // activations one group ahead
GVH9_CFG(2, 4, 2, 2, 1, 4, 1, 0)  // group ahead, weights after the use
GVH9_CFG(2, 5, 2, 4, 1, 2, 0, 0)  // pair ahead, ring of 4 units
GVH9_CFG(2, 6, 2, 2, 0, 4, 0, 1)  // ablation: shipped order, only branchless + epilogue
#undef GVH9_CFG

template <int K> struct Gvh9K;
template <> struct Gvh9K<0> { static constexpr int T = QT_MXFP4, TN = 2; };
template <> struct Gvh9K<1> { static constexpr int T = QT_Q6_K, TN = 1; };
template <> struct Gvh9K<2> { static constexpr int T = QT_Q4_K, TN = 1; };

template <int K, int ID>
__device__ __forceinline__ void gvh9_run(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, int nrows, int n_tok, int accumulate) {
    using C = Gvh9Cfg<K, ID>;
    gvh9_impl<Gvh9K<K>::T, C::KIND, Gvh9K<K>::TN, C::D, C::XG, C::GS, C::WP, C::WF>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
}

// selected entries (used by gemv_h.hip in place of the shipped entry when the id is non-zero)
#define GVH9_ENTRY(NAME, K, ID)                                                                                                         \
    extern "C" __global__ void __launch_bounds__(32) NAME(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols, \
                                                          int nrows, int n_tok, int accumulate) {                                       \
        gvh9_run<K, ID>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                             \
    }
#define GVH9_ENTRY_X(NAME, K, ID) GVH9_ENTRY(NAME, K, ID)

#if WHIRL_GVH8_DEV
// development exports for the gemvhv9 test family and the gemvhbench dev path (prefix_<type>)
#define GVH9_DEV(PFX, SUF, K, ID) GVH9_ENTRY(PFX##ID##_##SUF, K, ID)
GVH9_DEV(gvh92d, mxfp4, 0, 1)
GVH9_DEV(gvh92d, mxfp4, 0, 2)
GVH9_DEV(gvh92d, mxfp4, 0, 3)
GVH9_DEV(gvh92d, mxfp4, 0, 4)
GVH9_DEV(gvh92d, mxfp4, 0, 5)
GVH9_DEV(gvh92d, mxfp4, 0, 6)
GVH9_DEV(gvh9d, q6_k, 1, 1)
GVH9_DEV(gvh9d, q6_k, 1, 2)
GVH9_DEV(gvh9d, q6_k, 1, 3)
GVH9_DEV(gvh9d, q6_k, 1, 4)
GVH9_DEV(gvh9d, q6_k, 1, 5)
GVH9_DEV(gvh9d, q6_k, 1, 6)
GVH9_DEV(gvh9d, q6_k, 1, 7)
GVH9_DEV(gvh9d, q4_k, 2, 1)
GVH9_DEV(gvh9d, q4_k, 2, 2)
GVH9_DEV(gvh9d, q4_k, 2, 3)
GVH9_DEV(gvh9d, q4_k, 2, 4)
GVH9_DEV(gvh9d, q4_k, 2, 5)
GVH9_DEV(gvh9d, q4_k, 2, 6)
#undef GVH9_DEV
#endif
