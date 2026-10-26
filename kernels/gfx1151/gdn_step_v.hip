// WHIRL device code (gfx1151 / RDNA 3.5): register-diet variants of gdn_step_norm (feat026-gdnvgpr).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at the end of kernels/gfx1151/fused_decode.hip (needs GdnSegs, block_sum, wave_q8).
//
// The shipped gfx1151 gdn_step_norm (fused_decode.hip) needs 208 VGPRs: 7 waves per SIMD, so only ONE 16-wave block
// fits a WGP and the 48 heads of a launch run in 3 rounds over the 20 WGPs. The float work needs S[32] + k / q from LDS
// + temps; the rest are addresses LLVM hoists out of the token loop (one 32-bit index per state element and 32 64-bit
// final-store addresses, all live across the loop). The variants below compute EXACTLY the same values in the same
// order (every float expression is copied from the shipped kernel: the 32-term quarter chains, the
// red[0] + red[1] + red[2] + red[3] combines, block_sum, wave_q8) and only change how addresses / LDS reads are
// formed, so the compiler cannot keep them live across the loop:
//   ADDR 1: one uniform base (SGPR) + one per-thread byte offset per use site, made opaque with an empty asm so it is
//           re-formed at each use (a few SALU / VALU) instead of 32 hoisted 64-bit VGPR addresses.
//   KQ   1: the k / q LDS reads of the update loop take an opaque base per 8-row chunk (no 64 k / q values kept live).
// Tune id WHIRL_GDNV (default 0 = the shipped kernel stays the exported gdn_step_norm):
//   1 = ADDR 1, KQ 0     2 = ADDR 1, KQ 1
// WHIRL_GDNV_DEV=1 exports gdnv<id>_step_norm for every id beside the shipped kernel (kt family gdnv compares them).
#ifndef WHIRL_GDNV
#define WHIRL_GDNV 0
#endif
#ifndef WHIRL_GDNV_DEV
#define WHIRL_GDNV_DEV 0
#endif

#if WHIRL_GDNV || WHIRL_GDNV_DEV

// opaque copy: the value is unchanged, but LLVM must assume it is new here (stops hoisting / CSE across uses)
__device__ __forceinline__ unsigned gdnv_opq(unsigned x) {
    asm volatile("" : "+v"(x));
    return x;
}

template <int ADDR, int KQ, int TAIL>
__device__ __forceinline__ void gdnv_impl(const float *qkv, const float *g, const float *beta, const float *z,
                                          const float *wn, float *out, int8_t *xq, float *xd, int n_k_heads,
                                          float scale, float eps, int n_v_heads, int qkv_stride, const GdnSegs &segs,
                                          float (&ks)[128], float (&qs)[128], float (&red)[4][128], float (&ov)[128]) {
    constexpr int DK = 128, DV = 128, QI = 4, PER = DK / QI;
    const GdnSeg &sgm = segs.s[blockIdx.y];
    const int r0 = sgm.row0, n_tok = sgm.nrows;
    const int h = blockIdx.x;
    const int j = threadIdx.x % DV, qi = threadIdx.x / DV;
    const int kh = h % n_k_heads;
    const int d_inner = n_v_heads * DV;
    (void)QI;
    float *Sg = sgm.state + (size_t)h * DK * DV;
    // byte offset of S[0] of this thread inside a head's state (< 64 KiB)
    const unsigned ob = (unsigned)((qi * PER) * DV + j) * 4u;
    float S[PER];
    if (ADDR) {
        const unsigned o = gdnv_opq(ob);
#pragma unroll
        for (int e = 0; e < PER; ++e) S[e] = *(const float *)((const char *)Sg + o + (unsigned)e * DV * 4u);
    } else {
#pragma unroll
        for (int e = 0; e < PER; ++e) S[e] = Sg[(size_t)(qi * PER + e) * DV + j];
    }
    for (int tl = 0; tl < n_tok; ++tl) {
        const int t = r0 + tl;
        const float *row = qkv + (size_t)t * qkv_stride;
        __syncthreads();
        if (threadIdx.x < DK) {
            qs[threadIdx.x] = row[(size_t)kh * DK + threadIdx.x];
            ks[threadIdx.x] = row[(size_t)n_k_heads * DK + (size_t)kh * DK + threadIdx.x];
        }
        __syncthreads();
        const float decay = __expf(g[(size_t)t * n_v_heads + h]);
        float sk = 0.f;
#pragma unroll
        for (int e = 0; e < PER; ++e) {
            S[e] *= decay;
            sk += S[e] * ks[qi * PER + e];
        }
        red[qi][j] = sk;
        __syncthreads();
        sk = red[0][j] + red[1][j] + red[2][j] + red[3][j];
        const float v = row[(size_t)2 * n_k_heads * DK + (size_t)h * DV + j];
        const float delta = (v - sk) * beta[(size_t)t * n_v_heads + h];
        float o = 0.f;
        if (KQ) {
#pragma unroll
            for (int c = 0; c < PER; c += 8) {
                const unsigned kb = gdnv_opq((unsigned)(qi * PER + c));
#pragma unroll
                for (int e = c; e < c + 8; ++e) {
                    S[e] += ks[kb + (e - c)] * delta;
                    o += S[e] * qs[kb + (e - c)];
                }
            }
        } else {
#pragma unroll
            for (int e = 0; e < PER; ++e) {
                S[e] += ks[qi * PER + e] * delta;
                o += S[e] * qs[qi * PER + e];
            }
        }
        float *sp = tl < GDN_MAX_SNAP ? sgm.snap[tl] : nullptr;
        if (sp != nullptr) {
            float *Ss = sp + (size_t)h * DK * DV;
            if (ADDR) {
                const unsigned so = gdnv_opq(ob);
#pragma unroll
                for (int e = 0; e < PER; ++e) *(float *)((char *)Ss + so + (unsigned)e * DV * 4u) = S[e];
            } else {
#pragma unroll
                for (int e = 0; e < PER; ++e) Ss[(size_t)(qi * PER + e) * DV + j] = S[e];
            }
        }
        __syncthreads();
        red[qi][j] = o;
        __syncthreads();
        if (qi == 0) ov[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * scale;
        __syncthreads();
        float y = 0.f;
        if (threadIdx.x < DV) y = ov[threadIdx.x];
        const float ss = block_sum(threadIdx.x < DV ? y * y : 0.f);
        if (threadIdx.x < DV) {
            const float r = rsqrtf(ss / DV + eps);
            if (TAIL) {
                // same values; uniform bases + one opaque per-lane byte offset (no hoisted 64-bit lane addresses)
                const unsigned tb = gdnv_opq(threadIdx.x * 4u);
                const size_t rb = (size_t)t * d_inner + (size_t)h * DV;
                const float zz = *(const float *)((const char *)(z + rb) + tb);
                y = y * r * *(const float *)((const char *)wn + tb) * (zz / (1.f + __expf(-zz)));
                *(float *)((char *)(out + rb) + tb) = y;
            } else {
                const float zz = z[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x];
                y = y * r * wn[threadIdx.x] * (zz / (1.f + __expf(-zz)));
                out[(size_t)t * d_inner + (size_t)h * DV + threadIdx.x] = y;
            }
            wave_q8(y, xq + (size_t)t * d_inner, xd + (size_t)t * (d_inner / 32), (h * DV + threadIdx.x) >> 5);
        }
    }
    if (ADDR) {
        const unsigned fo = gdnv_opq(ob);
#pragma unroll
        for (int e = 0; e < PER; ++e) *(float *)((char *)Sg + fo + (unsigned)e * DV * 4u) = S[e];
    } else {
#pragma unroll
        for (int e = 0; e < PER; ++e) Sg[(size_t)(qi * PER + e) * DV + j] = S[e];
    }
}

#define GDNV_ARGS                                                                                                    \
    const float *qkv, const float *g, const float *beta, const float *z, const float *wn, float *out, int8_t *xq,  \
        float *xd, int n_k_heads, float scale, float eps, int n_v_heads, int qkv_stride, GdnSegs segs
#define GDNV_ENTRY(NAME, ADDR_, KQ_, TAIL_, ATTR)                                                                    \
    extern "C" __global__ void __launch_bounds__(512) ATTR NAME(GDNV_ARGS) {                                        \
        __shared__ float ks[128], qs[128];                                                                           \
        __shared__ float red[4][128];                                                                                \
        __shared__ float ov[128];                                                                                    \
        gdnv_impl<ADDR_, KQ_, TAIL_>(qkv, g, beta, z, wn, out, xq, xd, n_k_heads, scale, eps, n_v_heads, qkv_stride, segs, \
                              ks, qs, red, ov);                                                                      \
    }

#define GDNV_W12 __attribute__((amdgpu_waves_per_eu(12)))
#define GDNV_NOATTR
// id -> ADDR, KQ, TAIL, attribute
#define GDNV_ID1(N) GDNV_ENTRY(N, 1, 0, 0, GDNV_NOATTR)
#define GDNV_ID2(N) GDNV_ENTRY(N, 1, 1, 0, GDNV_NOATTR)
#define GDNV_ID3(N) GDNV_ENTRY(N, 1, 0, 1, GDNV_NOATTR)
#define GDNV_ID4(N) GDNV_ENTRY(N, 1, 0, 0, GDNV_W12)
#define GDNV_ID5(N) GDNV_ENTRY(N, 1, 0, 1, GDNV_W12)
#define GDNV_ID6(N) GDNV_ENTRY(N, 1, 1, 1, GDNV_W12)
#if WHIRL_GDNV == 1
GDNV_ID1(gdn_step_norm)
#elif WHIRL_GDNV == 2
GDNV_ID2(gdn_step_norm)
#elif WHIRL_GDNV == 3
GDNV_ID3(gdn_step_norm)
#elif WHIRL_GDNV == 4
GDNV_ID4(gdn_step_norm)
#elif WHIRL_GDNV == 5
GDNV_ID5(gdn_step_norm)
#elif WHIRL_GDNV == 6
GDNV_ID6(gdn_step_norm)
#elif WHIRL_GDNV
#error "WHIRL_GDNV: unknown tune id"
#endif
#if WHIRL_GDNV_DEV
GDNV_ID1(gdnv1_step_norm)
GDNV_ID2(gdnv2_step_norm)
GDNV_ID3(gdnv3_step_norm)
GDNV_ID4(gdnv4_step_norm)
GDNV_ID5(gdnv5_step_norm)
GDNV_ID6(gdnv6_step_norm)
#endif

#endif  // WHIRL_GDNV || WHIRL_GDNV_DEV
