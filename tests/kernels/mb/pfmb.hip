// WHIRL kernel microbenchmark "pfmb" (gfx1201 / RDNA 4 only): load-path variants for the decode-bound streams.
// SPDX-License-Identifier: Apache-2.0
// feat026-prefetch (kernel technique audit gap #12; progress\feat026-prefetch.PRIOR_ART.md). NOT part of the engine:
// this file is never included by kernels\whirl_kernels.hip. It is compiled on its own into a code object
// (tests\kernels\mb\build_pfmb.bat) that whirl-kernel-test --only pfmb loads from WHIRL_KT_PFMB_HSACO.
//
// Every kernel family below has one baseline (plain global loads, as the engine kernels do today) and variants that
// change ONLY how the same bytes reach the same registers:
//   b   raw buffer loads (__builtin_amdgcn_raw_buffer_load_b*, descriptor in SGPRs, 32-bit offsets)
//   s   the per-row scale dwords through the scalar path (s_buffer_load_b32 into SGPRs, v_cndmask to the lane)
//   p*  s_prefetch_data hints ahead of the loads (scalar-path prefetch of 128-byte lines; RDNA4 ISA 8.5: length =
//       (SDATA + SOFFSET) mod 32 + 1 lines, no bounds check, no completion; the builtin's 2nd argument is SDATA)
// The arithmetic is an order-sensitive integer hash of every loaded dword (FNV-1a step), so a variant that loads
// a different byte, or the same bytes into a different slot, gives a different checksum: the harness requires every
// variant's output to be bit-identical to the baseline's (and the baseline to the C++ model for pfs / pfmx).
//
// Families (byte / lane maps copied from the engine kernels, so the load pattern is the production one):
//   pfs_*   pure stream, one row per wave, 16 bytes per lane per 512-byte piece, 4 pieces per step (the "pure-read"
//           roof of docs\guide\en\kernels.md section 0); rows assigned wave-major (row = wave + k * waves); grid is a
//           harness parameter (full grid or a capped persistent grid, to see latency hiding at low occupancy).
//           PF distance = bytes ahead in the wave's own stream (crossing into its next row).
//   pfmx_*  MXFP4 gemvh weight stream: the lane -> byte map of the gfx1201 "direct" gemvh candidates
//           (feat026-gemvh 72cf41b kernels\gemv_hv_map.h: WHIRL layout e[8] + qs[8][16] = 136 bytes per 256 k; lane l
//           = row l % 16, half g0 = l / 16; per 128-k unit: one scale dword + 4 x 8-byte pieces), register ring of
//           D = 4 units, one wave per 16-row tile (block 32, as gemvh), no WMMA: isolates the load path of that map.
//           PF: every PFE superblocks, one prefetch per row of the 16 rows, PFD superblocks ahead, PFL + 1 lines.
//   pfkv_*  decode attention KV stream: the K / V byte map of attn_wsplit1 (kernels\attn.hip attn_wsplit_impl, f16
//           KV, head_dim 256, 32-key tiles, paged pool rows via ptab): grid (kv head, split), block 128; per tile:
//           barrier, every thread loads 8 V rows x 16 bytes and stores them transposed to LDS, barrier, wave 0 loads
//           K (2 x 16 keys x 16 pieces of 16 bytes per lane) and reads the LDS V^T fragments the PV WMMA would read.
//           PF: after issuing a tile's V loads, each of the 4 waves prefetches the K and V rows of its 8 keys of the
//           tile PFT tiles ahead (16 prefetches of 4 lines per wave and tile).
#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx1200__) || defined(__gfx1201__))
#error "pfmb.hip is gfx12-only (s_prefetch_data; buffer descriptor word 3 0x31004000)"
#endif

typedef int32_t pf_i32x4 __attribute__((ext_vector_type(4)));
typedef int32_t pf_i32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t pf_u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t pf_u32x2 __attribute__((ext_vector_type(2)));
typedef _Float16 pf_half8 __attribute__((ext_vector_type(8)));

// gfx12 buffer descriptor word 3 (CK; hipfire radiowave/hip.h:43 "correctness-checked on ... gfx1201")
#define PFMB_RSRC_W3 0x31004000u

// llvm.amdgcn.s.buffer.load has no clang builtin: declared through an asm label (the CK pattern); <4 x i32> descriptor
__device__ int32_t pfmb_sbl32(pf_i32x4 rsrc, int32_t offset, int32_t aux) __asm("llvm.amdgcn.s.buffer.load.i32");

__device__ __forceinline__ __amdgpu_buffer_rsrc_t pf_rsrc(const void *p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), (short)0, (int)bytes, (int)PFMB_RSRC_W3);
}
// the same descriptor as four words (stride 0): {base lo, base hi, num_records, word 3}
__device__ __forceinline__ pf_i32x4 pf_rsrc_words(const void *p, uint32_t bytes) {
    const uint64_t a = (uint64_t)p;
    pf_i32x4 r = {(int32_t)(uint32_t)a, (int32_t)((a >> 32) & 0xffffu), (int32_t)bytes, (int32_t)PFMB_RSRC_W3};
    return r;
}
template <int L> __device__ __forceinline__ void pf_hint(const void *p) { __builtin_amdgcn_s_prefetch_data(p, L); }

// FNV-1a step: order-sensitive (the C++ model in tests\kernels\test_pfmb.cpp uses the same)
__device__ __forceinline__ uint32_t pf_mix(uint32_t h, uint32_t x) { return (h ^ x) * 0x01000193u; }
__device__ __forceinline__ uint32_t pf_mix4(uint32_t h, pf_u32x4 v) { return pf_mix(pf_mix(pf_mix(pf_mix(h, v.x), v.y), v.z), v.w); }

// 16 / 8 / 4-byte loads: BUF 0 = global (base + 32-bit byte offset), 1 = raw buffer (voffset = off, soffset = soff)
template <int BUF>
__device__ __forceinline__ pf_u32x4 pf_ld16(__amdgpu_buffer_rsrc_t rs, const uint8_t *base, uint32_t soff, uint32_t off) {
    if constexpr (BUF) return __builtin_bit_cast(pf_u32x4, __builtin_amdgcn_raw_buffer_load_b128(rs, (int)off, (int)soff, 0));
    else return *(const pf_u32x4 *)(base + soff + off);
}
template <int BUF>
__device__ __forceinline__ pf_u32x2 pf_ld8(__amdgpu_buffer_rsrc_t rs, const uint8_t *base, uint32_t off) {
    if constexpr (BUF) return __builtin_bit_cast(pf_u32x2, __builtin_amdgcn_raw_buffer_load_b64(rs, (int)off, 0, 0));
    else return *(const pf_u32x2 *)(base + off);
}
template <int BUF>
__device__ __forceinline__ uint32_t pf_ld4(__amdgpu_buffer_rsrc_t rs, const uint8_t *base, uint32_t off) {
    if constexpr (BUF) return (uint32_t)__builtin_amdgcn_raw_buffer_load_b32(rs, (int)off, 0, 0);
    else return *(const uint32_t *)(base + off);
}

// ---------------------------------------------------------------------------------------------------------------
// pfs: pure stream. src = nrows x row_bytes (row_bytes % 16 == 0, nrows * row_bytes < 4 GB: host-checked).
// out[global thread] = hash of the 16-byte pieces the lane read, in order (pieces past the row end: loaded clamped,
// not hashed). PFD = prefetch distance in bytes of the wave's stream from the start of the step (0 = off; issued at the
// end of the step), PFL = SDATA (lines - 1).
// ---------------------------------------------------------------------------------------------------------------
template <int BUF, int PFD, int PFL>
__device__ __forceinline__ void pfs_impl(const uint8_t *__restrict__ src, uint32_t src_bytes, uint32_t row_bytes, int nrows,
                                         uint32_t *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int wib = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int wpb = (int)(blockDim.x >> 5);
    const int nw = (int)gridDim.x * wpb;
    const __amdgpu_buffer_rsrc_t rs = pf_rsrc(src, src_bytes);
    const int np = (int)((row_bytes + 511) / 512);
    uint32_t acc = 0x811c9dc5u;
    for (int row = (int)blockIdx.x * wpb + wib; row < nrows; row += nw) {
        const uint32_t rbase = (uint32_t)row * row_bytes;  // wave-uniform (SGPR)
        for (int j0 = 0; j0 < np; j0 += 4) {
            pf_u32x4 v[4];
            bool ok[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint32_t o = (uint32_t)(j0 + u) * 512 + (uint32_t)lane * 16;
                ok[u] = o + 16 <= row_bytes;
                v[u] = pf_ld16<BUF>(rs, src, rbase, ok[u] ? o : row_bytes - 16);
            }
            __builtin_amdgcn_sched_barrier(0);  // all 4 loads of the step issue before the first wait (every variant)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint32_t t = pf_mix4(acc, v[u]);
                acc = ok[u] ? t : acc;
            }
            // after the step (before the loads it split the 4 loads of a step into 2 + 2 with a loadcnt-0 wait between:
            // isa_v3 trace, 2026-10-26)
            if constexpr (PFD > 0) {
                // PFD bytes ahead in this wave's stream: still in this row, or the same distance into its next row
                const uint32_t pos = (uint32_t)j0 * 512 + PFD;
                const uint32_t a = pos < row_bytes ? rbase + pos : (uint32_t)(row + nw) * row_bytes + (pos - row_bytes);
                // clamped into the buffer: the ISA documents no bounds check for the prefetch, and whether an unmapped
                // address can fault is not documented (PRIOR_ART section 2), so never point past the allocation
                pf_hint<PFL>(src + min(a, src_bytes - 128u));
            }
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

#define PFS_ENTRY(NAME, BUF, PFD, PFL)                                                                                     \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t *src, uint32_t src_bytes, uint32_t row_bytes,     \
                                                           int nrows, uint32_t *out) {                                     \
        pfs_impl<BUF, PFD, PFL>(src, src_bytes, row_bytes, nrows, out);                                                    \
    }
PFS_ENTRY(pfs_g, 0, 0, 0)          // baseline: global_load_b128
PFS_ENTRY(pfs_b, 1, 0, 0)          // buffer_load_b128 (voffset lane, soffset row)
PFS_ENTRY(pfs_gp0, 0, 64, 15)      // re-prefetch the 2 KB this step just loaded: pure overhead check (no new lines)
PFS_ENTRY(pfs_gp2k, 0, 2048, 15)   // the next step's 2 KB (issued at the end of this step), 16 lines
PFS_ENTRY(pfs_gp4k, 0, 4096, 15)   // the step after next (one step of lead)
PFS_ENTRY(pfs_gp8k, 0, 8192, 31)   // 4 KB starting 8 KB past this step's start, 32 lines
PFS_ENTRY(pfs_bp4k, 1, 4096, 15)   // buffer + 4 KB ahead

// ---------------------------------------------------------------------------------------------------------------
// pfmx: MXFP4 gemvh weight stream (lane map of feat026-gemvh gemv_hv_map.h). W = nrows x row_bytes, row_bytes =
// ncols / 256 * 136, ncols % 512 == 0 (nu = ncols / 128 a multiple of D). out[block * 32 + lane] = hash over the
// lane's units in order: scale dword, then pieces 0..3 (x, y).
// SCL 1: the scale dword of each of the 16 rows through s_buffer_load_b32 (uniform offsets), moved into lanes rl and
// rl + 16 with one v_cndmask per row. PFE / PFD / PFL: every PFE superblocks, prefetch superblock (sb + PFD) of the 16 rows.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ constexpr uint32_t pf_mx_q_off(int ph, int i, int g0) { return 8u + 16u * (uint32_t)(4 * ph + i) + 8u * (uint32_t)g0; }

template <int BUF, int SCL, int PFE, int PFD, int PFL>
__device__ __forceinline__ void pfmx_impl(const uint8_t *__restrict__ W, uint32_t w_bytes, uint32_t row_bytes, int ncols, int nrows,
                                          uint32_t *__restrict__ out) {
    constexpr int D = 4;
    const int lane = threadIdx.x & 31, g0 = lane >> 4, rl = lane & 15;
    const int nu = ncols / 128, nsb = ncols / 256;
    const __amdgpu_buffer_rsrc_t rs = pf_rsrc(W, w_bytes);
    const pf_i32x4 rw = pf_rsrc_words(W, w_bytes);
    struct R {
        pf_u32x2 q[4];
        uint32_t e;
    };
    uint32_t acc = 0x811c9dc5u;
    for (int tile = (int)blockIdx.x; tile * 16 < nrows; tile += (int)gridDim.x) {
        const int m0 = tile * 16;
        const uint32_t roff = (uint32_t)min(m0 + rl, nrows - 1) * row_bytes;  // per lane
        auto load = [&](R &r, int u) {
            const uint32_t b = (uint32_t)(u >> 1) * 136u;
            const int ph = u & 1;
            if constexpr (SCL) {
                uint32_t v = 0;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const uint32_t ro = (uint32_t)min(m0 + k, nrows - 1) * row_bytes;  // uniform
                    const int32_t s = pfmb_sbl32(rw, (int32_t)(ro + b + 4u * (uint32_t)ph), 0);
                    v = rl == k ? (uint32_t)s : v;  // v_cndmask with the SGPR value (no writelane builtin in clang 21)
                }
                r.e = v;
            } else {
                r.e = pf_ld4<BUF>(rs, W, roff + b + 4u * (uint32_t)ph);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) r.q[i] = pf_ld8<BUF>(rs, W, roff + b + pf_mx_q_off(ph, i, g0));
        };
        R w[D];
#pragma unroll
        for (int u = 0; u < D - 1; ++u) load(w[u], min(u, nu - 1));
#pragma nounroll
        for (int u0 = 0; u0 < nu; u0 += D) {
#pragma unroll
            for (int s = 0; s < D; ++s) {
                const int u = u0 + s;
                __builtin_amdgcn_sched_barrier(0);
                load(w[(s + D - 1) % D], min(u + D - 1, nu - 1));  // clamped past the end: refills a slot never read
                __builtin_amdgcn_sched_barrier(0);
                const R &c = w[s % D];
                acc = pf_mix(acc, c.e);
#pragma unroll
                for (int i = 0; i < 4; ++i) acc = pf_mix(pf_mix(acc, c.q[i].x), c.q[i].y);
                // PF after the consume (between load and consume it made the allocator rotate the ring through
                // v_mov copies that wait for loadcnt 0 every D units: isa_v2, 2026-10-26)
                if constexpr (PFE > 0) {
                    const int sb = u >> 1;
                    if ((s & 1) == 0 && sb % PFE == 0) {
                        const uint32_t po = (uint32_t)min(sb + PFD, nsb - 1) * 136u;
#pragma unroll
                        for (int k = 0; k < 16; ++k) pf_hint<PFL>(W + (uint32_t)min(m0 + k, nrows - 1) * row_bytes + po);
                    }
                }
            }
        }
    }
    out[blockIdx.x * 32 + lane] = acc;
}

#define PFMX_ENTRY(NAME, BUF, SCL, PFE, PFD, PFL)                                                                         \
    extern "C" __global__ void __launch_bounds__(32) NAME(const uint8_t *W, uint32_t w_bytes, uint32_t row_bytes, int ncols, \
                                                          int nrows, uint32_t *out) {                                     \
        pfmx_impl<BUF, SCL, PFE, PFD, PFL>(W, w_bytes, row_bytes, ncols, nrows, out);                                     \
    }
PFMX_ENTRY(pfmx_g, 0, 0, 0, 0, 0)     // baseline: global_load_b32 + 4 x global_load_b64 per unit
PFMX_ENTRY(pfmx_b, 1, 0, 0, 0, 0)     // buffer_load_b32 / b64
PFMX_ENTRY(pfmx_gs, 0, 1, 0, 0, 0)    // global pieces, scales via s_buffer_load_b32
PFMX_ENTRY(pfmx_bs, 1, 1, 0, 0, 0)    // buffer pieces, scales via s_buffer_load_b32
PFMX_ENTRY(pfmx_gp2, 0, 0, 1, 2, 1)   // = 72cf41b PF id 6 pattern: every superblock, 2 ahead, 2 lines, 16 rows
PFMX_ENTRY(pfmx_gp8, 0, 0, 1, 8, 1)   // every superblock, 8 ahead (1 KB)
PFMX_ENTRY(pfmx_gr8, 0, 0, 8, 8, 8)   // every 8th superblock, the next 8 (1088 B = 9 lines) of each row
PFMX_ENTRY(pfmx_bp2, 1, 0, 1, 2, 1)   // buffer + PF as gp2

// ---------------------------------------------------------------------------------------------------------------
// pfkv: attn_wsplit1 K / V stream (f16 KV, head_dim 256, KV_PAGE 256, 32-key tiles, FD 64-key split granularity).
// K / V pools: [pool row][n_kv][256] f16, kv_bytes each (< 4 GB, host-checked). ptab: page table of the sequence.
// Grid (n_kv, n_split), block 128. out[(g * n_split + split) * 32 + lane] = wave 0's hash (K pieces then the LDS V^T
// fragment of each step); blocks past the sequence write 0.
// ---------------------------------------------------------------------------------------------------------------
#define PFKV_HD 256
#define PFKV_KT 32
#define PFKV_PAGE 256

template <int BUF, int PFT, int PFL>
__device__ __forceinline__ void pfkv_impl(const uint8_t *__restrict__ K, const uint8_t *__restrict__ V, uint32_t kv_bytes,
                                          const int *__restrict__ ptab, int n_pos, int n_kv, uint32_t *__restrict__ out) {
    __shared__ _Float16 Vt[PFKV_HD][PFKV_KT + 8];
    const int g = (int)blockIdx.x, split = (int)blockIdx.y, n_split = (int)gridDim.y;
    const int tid = (int)threadIdx.x, lane = tid & 31;
    const int wid = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int per = (n_pos + n_split * 64 - 1) / (n_split * 64) * 64;
    const int b0 = split * per, b1 = min(n_pos, b0 + per);
    uint32_t *o = out + ((size_t)g * n_split + split) * 32;
    if (b0 >= n_pos) {
        if (tid < 32) o[lane] = 0;
        return;
    }
    const __amdgpu_buffer_rsrc_t rk = pf_rsrc(K, kv_bytes), rv = pf_rsrc(V, kv_bytes);
    const uint32_t rowb = (uint32_t)n_kv * PFKV_HD * 2;  // bytes per pool row (all kv heads)
    const int kb = wid, d8 = (tid & 31) * 8;              // V staging: key group kb (8 keys), dims d8 .. d8 + 8
    uint32_t acc = 0x811c9dc5u;
    for (int k0 = b0; k0 < b1; k0 += PFKV_KT) {
        const int dl = (ptab[k0 / PFKV_PAGE] - k0 / PFKV_PAGE) * PFKV_PAGE;  // 32-key tiles never straddle a page
        __syncthreads();
        pf_u32x4 rows[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int key = k0 + kb * 8 + r;
            if (key < b1) rows[r] = pf_ld16<BUF>(rv, V, 0, (uint32_t)(key + dl) * rowb + (uint32_t)g * 512u + (uint32_t)d8 * 2u);
            else rows[r] = (pf_u32x4){0, 0, 0, 0};
        }
        if constexpr (PFT > 0) {
            // K and V rows of this wave's 8 keys of tile k0 + PFT * 32 (4 lines = the 512-byte head row each)
            const int nk0 = k0 + PFT * PFKV_KT;
            if (nk0 < b1) {
                const int ndl = (ptab[nk0 / PFKV_PAGE] - nk0 / PFKV_PAGE) * PFKV_PAGE;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint32_t a = (uint32_t)(min(nk0 + wid * 8 + r, b1 - 1) + ndl) * rowb + (uint32_t)g * 512u;
                    pf_hint<PFL>(V + a);
                    pf_hint<PFL>(K + a);
                }
            }
        }
        // transpose 8 keys x 8 dims in registers, as attn_wsplit_impl: Vt[d8 + c][kb * 8 + r]
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            pf_half8 cv;
#pragma unroll
            for (int r = 0; r < 8; ++r) cv[r] = __builtin_bit_cast(pf_half8, rows[r])[c];
            *(pf_half8 *)&Vt[d8 + c][kb * 8] = cv;
        }
        __syncthreads();
        if (wid != 0) continue;
#pragma unroll
        for (int kt = 0; kt < PFKV_KT / 16; ++kt) {
            const int krow = min(k0 + kt * 16 + (lane & 15), b1 - 1);
            const uint32_t kr = (uint32_t)(krow + dl) * rowb + (uint32_t)g * 512u + 16u * (uint32_t)(lane >> 4);
#pragma unroll
            for (int st = 0; st < PFKV_HD / 16; ++st) {
                const pf_u32x4 a = pf_ld16<BUF>(rk, K, 0, kr + 32u * (uint32_t)st);
                const pf_u32x4 v = *(const pf_u32x4 *)&Vt[st * 16 + (lane & 15)][kt * 16 + 8 * (lane >> 4)];
                acc = pf_mix4(pf_mix4(acc, a), v);
            }
        }
    }
    if (wid == 0) o[lane] = acc;
}

#define PFKV_ENTRY(NAME, BUF, PFT, PFL)                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) NAME(const uint8_t *K, const uint8_t *V, uint32_t kv_bytes,           \
                                                           const int *ptab, int n_pos, int n_kv, uint32_t *out) {          \
        pfkv_impl<BUF, PFT, PFL>(K, V, kv_bytes, ptab, n_pos, n_kv, out);                                                  \
    }
PFKV_ENTRY(pfkv_g, 0, 0, 0)     // baseline: global_load_b128 (attn_wsplit1 map)
PFKV_ENTRY(pfkv_b, 1, 0, 0)     // buffer_load_b128
PFKV_ENTRY(pfkv_gp1, 0, 1, 3)   // prefetch the next tile's K / V rows
PFKV_ENTRY(pfkv_gp2, 0, 2, 3)   // two tiles ahead
PFKV_ENTRY(pfkv_bp1, 1, 1, 3)   // buffer + next tile
