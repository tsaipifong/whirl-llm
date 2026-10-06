// Host-side ABI of the WHIRL HIP kernels (kernels/*.hip): kernel argument
// structs, launch constants, GEMM / GEMV configuration tables, and a kernel
// table that resolves every entry point by name from the embedded code
// object of the current device.
// SPDX-License-Identifier: Apache-2.0
//
// The model code calls kernels through this header only.
//
// Conventions shared by all kernels
//   * Device pointers are passed as hip::DevPtr (uint64_t) in the launch
//     argument list; structs passed by value contain DevPtr fields.
//   * Integer scalars are int32_t, flags ("accumulate", "advance") are int32_t
//     0/1, floats are float. Argument types must match exactly (see
//     hip::launch).
//   * Weight matrices stay in their GGUF block format, row-major, `row_bytes`
//     bytes per row (MXFP4 uses WHIRL's 136-byte superblock layout, see
//     kernels/common.hip). `ncols` is the inner (K) dimension.
//   * Int8 activations ("xq / xd"): x quantized per 32 values, xq int8[n],
//     xd float[n / 32] scales (quantize_q8 / rmsnorm_q8 / silu_mul_q8 ...).
//   * `skip` (nullable): if *skip != 0 the kernel returns immediately (device
//     side early-out for captured graphs).
//
// Signatures by family (T = weight type suffix, see typeSuffix()):
//
//  f32-activation matvec (256 threads, 8 rows per block):
//   gemv_<T>_<1|4|8>(W, u64 row_bytes, const float* x, float* y, int ncols,
//                    int nrows, int x_stride, int y_stride, int accumulate)
//  int8 decode GEMV (256 threads; 8 rows per block for 1 wave per row):
//   gemvq_<T>(GvArgs g, const int8* xq, const float* xd, int ncols,
//             int accumulate, const int* skip)                  [1 token]
//   gemvq_nt<N>_<T>, gemvq_nt<N>r<R>_<T>, gemvw_nt<N>v<V>_<T>  [N = 2..16]
//            (W, u64 row_bytes, const int8* xq, const float* xd, float* y,
//             int ncols, int nrows, int accumulate, const int* skip)
//   ggemvq_nt<N>_<T>, ggemvq_nt<N>r<R>_<T>, ggemvw_nt<N>v<V>_<T>
//            (GvArgs g, xq, xd, int ncols, int accumulate, const int* skip)
//   quantize_q8(const float* x, int8* xq, float* xd, int n)
//   gemv_d2_nt<N>(const u8* W, xq, xd, float* y, int ncols, int nrows, skip)
//  prefill GEMM (Y[t][r] (+)= W[r] . X[t]):
//   gemm_<T>(W, rb, const float* X, float* Y, ncols, nrows, n_tok, accumulate)
//   gemm_wmma_<T>, gemm3_<T>, gemm_c<i>_<T>, gemms_c<i>_<T>, gemmhq_<T>
//            (W, rb, const f16* X, float* Y, ncols, nrows, n_tok, accumulate)
//   gemm_ch<i>_f16, gemmsh_c<i>_<T>, gemmhqh_<T>: same args, Y written as f16
//            (the `float* Y` slot carries an f16 pointer; accumulate ignored)
//   gemmh_f16 / gemmhh_f16(const f16* W, const f16* X, float* Y, ncols,
//            nrows, n_tok, accumulate)
//   dequant_f16_<T>(W, rb, f16* out, ncols, nrows); f32_to_f16(x, y, n)
//   qact_fp8 / qact_fp8t(const float* x, u8* q, float* sx, int ncols)
//   gemm8_c<i>, gemm8h_c<i>, gemm8t_c<i>, gemm8th_c<i>(W, rb, const u8* wref,
//            const u8* X8, const float* sx, float* Y, ncols, nrows, n_tok, acc)
//  norms / elementwise:
//   rmsnorm(x, w, out, int n, int in_stride, int out_stride, float eps)
//   rmsnorm_q8(x, w, out, xq, xd, int n, float eps)
//   rmsnorm_q8_rows(x, w, out, xq, xd, int n, float eps, RowIdx src)
//   rmsnorm_x8 / rmsnorm_x16 / rmsnorm_x8t(x, w, u8* q, float* sx, n, eps)
//   rmsnorm_x8h16 / rmsnorm_x8h16t(x, w, u8* q, float* sx, f16* q16, n, eps)
//            (fp8 row-major / tiled + f16 row-major from one norm; optional)
//   l2norm(float* x, int n, int stride, int tok_stride, float eps)
//   add_inplace(a, b, n); silu_mul(a, b, n); silu_mul_q8(g, u, xq, xd, n)
//   silu_mul_x8 / x16 / x8t(const float* g, const float* u, u8* q, sx, ncols)
//   silu_mul_x8h / x16h / x8ht(const f16* g, const f16* u, u8* q, sx, ncols)
//   gated_norm_x8 / x16 / x8t(o, const float* z, w, u8* q, sx, n_heads, eps)
//   gated_norm_x8h / x16h / x8ht(o, const f16* z, w, u8* q, sx, n_heads, eps)
//   rope_neox(float* x, const int* pos, int head_stride, int tok_stride,
//             int n_rot, float theta_scale)
//   get_rows_<T>(W, rb, const int* ids, float* out, int ncols)
//  attention (head_dim 256, paged KV, KvArgs):
//   attn_decode[_q8|_q8v](q, KvArgs, out, scores, n_head, n_kv, head_dim,
//            q_stride, max_ctx, const int* pos, float scale)
//   attn_split[_q8|_q8v](q, KvArgs, part_ml, part_acc, n_head, n_kv,
//            head_dim, q_stride, pos, scale, int nq, skip)
//   attn_combine(part_ml, part_acc, q, out, n_head, head_dim, q_stride, n_split)
//   attn_combine_q8(... same ..., int8* xq, float* xd)
//   attn_wsplit1[_q8|_q8v], attn_wsplit2[_q8|_q8v](q, KvArgs, part_ml, part_acc, n_head, n_kv,
//            q_stride, pos, scale, skip, AwGroups groups)   [<= 16 / <= 32 columns per group]
//   attn_prep[_q8|_q8h|_q8v](qf, kk, vv, qw, kw, KvArgs, pos, n_head, n_kv,
//            hd, n_rot, theta_scale, eps)
//   attn_prefill_wmma[_q8|_q8v], attn_kx[_q8|_q8v](q, KvArgs, out, n_head,
//            n_kv, q_stride, pos, n_tok, scale, int h0)
//   attn_kg6 / attn_kg4 / attn_kg2[_q8|_q8v] (f16 / q8 / q8h / q8v KV; same arguments; grid
//            (ceil(n_tok / 16), heads / NP), block 64 * NP, NP | n_head / n_kv)
//   kv_store[_q8|_q8v](k, v, KvArgs, pos, int row)
//  Gated DeltaNet:
//   gdn_conv_seq(xin, state, w, out, ch, n_tok); gdn_gates(b, a, dt_bias, A,
//            n, n_heads); gdn_conv_par(xin, state, w, out, ch, n_tok)
//   gdn_gates_ba(const ba, b, a, dt_bias, A, n, n_heads): gdn_gates reading
//            ba[t][2*n_heads] = [beta | alpha] (optional)
//   gdn_conv_state(xin, state, ch, n_tok, float* snap, int snap_after)
//   gdn_seq_128(qkv, g, beta, state, out, n_tok, n_k_heads, n_v_heads, dv,
//            qkv_stride, scale, float* snap, int snap_after)
//   gdn_chunk_prep(qkv, gl, beta, W, U, M, G, n_tok, n_k_heads, n_v_heads,
//            qkv_stride); gdn_chunk_scan(qkv, W, U, M, G, state, out, n_tok,
//            n_k_heads, n_v_heads, qkv_stride, scale)
//   gdn_wprep(qkv, gl, beta, f16* TM, G, n_tok, nkh, nvh, qkv_stride)
//   gdn_wscan8(qkv, const f16* TM, G, beta, state, out, n_tok, nkh, nvh,
//            qkv_stride, scale)
//   gdn_gated_norm(o, z, w, dv, eps)
//   gdn_ab_<T>(Wb, Wa, rb, xq, xd, beta, g, dt, A, ncols, nh)
//   gdn_abconv_<T>(... gdn_ab args ..., int n_ab, xin, w, out, ch,
//            n_qk_heads, eps, GdnSegs segs)
//   gdn_conv_l2(xin, w, out, ch, n_qk_heads, eps, GdnSegs segs)
//   gdn_step_norm(qkv, g, beta, z, wn, out, xq, xd, n_k_heads, scale, eps,
//            n_v_heads, qkv_stride, GdnSegs segs)
//   gdn_conv_l2n(xin, state, w, out, ch, n_tok, n_qk_heads, eps)
//   gdn_conv_l2n_h / gdn_conv_state_h: as above with const f16* xin
//  tokens / picks:
//   argmax(x, n, int* out, int* ids, int* pos, int advance)
//   argmax_rows(x, n, out); argmax_rows_to(x, n, out, RowIdx dst)
//   argmax_prob(x, n, int* out_tok, float* out_prob)
//   draft_pick(x, n, int* tok, float* prob, int* ctl, int r, int n_min, float p_min, const int* map)
//   draft_pick_rows(x, n, int* ctl_all, RowIdx area, int r, int n_min, float p_min, const int* map)
//            map (null: identity) = token id of each draft-head row (vocabulary subset)
//   set_tokens(ids, pos, dev_src, Tok16 toks, n, n_host, pos0)
//   set_rows(ids, pos, kvbase, dev_src, RowTab tab, n)
//   topk_rows(x, n, K, float inv_t, int* ids, float* vals, float* stats)
//   requant_q6k_q4k(src, u64 src_rb, dst, u64 dst_rb); requant_q6k_d2(src,
//            u64 src_rb, dst, int ncols); requant_q80_q4k(src, u64 src_rb, dst, u64 dst_rb)
//   copy_rows_map(src, u64 rb, dst, const int* map): dst row i = src row map[i] (optional)
//  mixture of experts:
//   moe_logits_f32(W, x, out, R, C, n, skip)
//   moe_topk(logits, ld, h, shw, ids, w, sg, R, K, E)
//   moe_route(ids, n_pairs, R, bn, perm, inv, Int4* tiles, n_tiles)
//   moe_gather_f16(x, perm, f16* xg, C, K); moe_act_f16(g, u, f16* a, n)
//   moe_combine(x, ysh, sg, yd, inv, w, C, K)
//   moe_gu_<T>(W1, W2, rb, ff, nrows, xq, xd, ids, y1, y2, C, K, skip)
//   moe_down_<T>(W, rb, E, xq, xd, ids, w, sg, ysh, x, F, K, skip)
//   gemm_moe_<T> / gemm_moe32_<T>(W, rb, rows_e, const f16* X, float* Y,
//            ncols, const Int4* tiles, const int* n_tiles)
//  grouped expert GEMM, Radeon 8060S prefill (optional, gfx1151 only):
//   moe_tiles(ids, n_pairs, R, bn, Int4* tiles, n_tiles): moe_route's tiles for token tile bn
//   gemm_moe32r_<T>: gemm_moe32_<T> arguments, 32-token tiles, 1-D grid
//            (n_tile_slots * ceil(rows_e / kMoeBm), row block fastest)
//   gemm_moegu_<T>(W1, W2, rb, ff, const f16* X, f16* A, ncols, tiles, n_tiles): gate
//            (W1) + up (W2) + SwiGLU, A = f16(silu(g) * u) == gemm_moe_<T> x2 +
//            moe_act_f16; 64-token tiles, 1-D grid (n_tile_slots * ceil(ff / 64))
//  MXFP4 routed experts (optional, gfx1201 only; T = mxfp4 also has the generic four above):
//   moe_gu_mxfp4w / moe_down_mxfp4w: same arguments as moe_gu_<T> / moe_down_<T>
//            (whole 32-value block per lane step)
//   gemm8_moe / gemm8_moe32 / gemm8_moeh / gemm8_moe32h(W1, W2, rb, const u8* ref1,
//            const u8* ref2, rows_e, const u8* X8, const float* sx, Y1, Y2, ncols,
//            const Int4* tiles, const int* n_tiles): MXFP4 x fp8 grouped GEMM, token
//            tile 64 / 32, h = f16 output; W2 != null -> gate and up in one launch
//            (grid.y = 2 * ceil(rows_e / kMoeBm))
//   moe_gather_fp8(x, perm, u8* x8, float* sx, C, K): per-pos fp8 rows + scales

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "whirl/hip.h"

namespace whirl::kernels {

using hip::DevPtr;
using hip::Function;

// ---------------------------------------------------------------------------
// Constants (must match the #defines in kernels/*.hip)

inline constexpr int kWarp = 32;
inline constexpr int kKvPage = 256;          // KV_PAGE: tokens per KV page
inline constexpr int kHeadDim = 256;         // attention head dim of the WMMA kernels (AW_D / FA_D)
inline constexpr int kFdChunk = 64;          // FD_CH: positions per attn_split block
inline constexpr int kFdMaxSplits = 64;      // host cap on split count
inline constexpr int kGdnMaxSeg = 16;        // GDN_MAX_SEG
inline constexpr int kGdnMaxSnap = 15;       // GDN_MAX_SNAP
inline constexpr int kMaxSmallBatch = 16;    // multi-token GEMV / small-batch kernels: 2..16 tokens
inline constexpr int kMaxVerifyRows = 32;    // rows of one batched verify / MTP step (gemvx_v6_* past 16)
inline constexpr int kMaxDrafts = 10;        // MTP drafts per verify cycle
inline constexpr int kGemm3Bm = 256, kGemm3Bn = 256, kGemm3Threads = 512;  // GEMM3_*
inline constexpr int kMoeBm = 128, kMoeBn = 64;                            // MOE_BM, gemm_moe_* token tile
inline constexpr int kMoeBn32 = 32;                                        // gemm_moe32_* token tile
inline constexpr int kNTypes = 40;           // table size: > every ggml type id used (mxfp4 = 39)

// Layout of the small device control buffer (int32 slots) used by the
// draft-pick kernels.
inline constexpr int kCtlRows = 0;     // [0, 16): per-row argmax of a verify
inline constexpr int kCtlDrafts = 16;  // [16, 32): draft tokens
inline constexpr int kCtlProbs = 32;   // [32, 48): draft probabilities (f32)
inline constexpr int kCtlNd = 48;      // drafts kept this cycle
inline constexpr int kCtlStop = 49;    // draft chain stopped
inline constexpr int kCtlWords = 64;

// Weight types with kernels (ggml type ids).
enum class QType : int {
    f32 = 0, f16 = 1, q8_0 = 8, q3_k = 11, q4_k = 12, q5_k = 13, q6_k = 14,
    iq4_nl = 20, iq3_s = 21, iq4_xs = 23, mxfp4 = 39,
};
// Kernel-name suffix ("q4_k", "mxfp4", ...); nullptr for other ids.
const char* typeSuffix(QType t);
inline constexpr std::array<QType, 11> kAllTypes = {
    QType::f32, QType::f16, QType::q8_0, QType::q3_k, QType::q4_k, QType::q5_k,
    QType::q6_k, QType::iq4_nl, QType::iq3_s, QType::iq4_xs, QType::mxfp4};
// Block geometry (values per block, bytes per block) as the kernels read it
// (MXFP4: WHIRL's 256-value / 136-byte superblock).
struct BlockInfo {
    int values;
    int bytes;
};
BlockInfo blockInfo(QType t);
inline std::uint64_t rowBytes(QType t, int ncols) {
    const BlockInfo b = blockInfo(t);
    return static_cast<std::uint64_t>(ncols / b.values) * static_cast<std::uint64_t>(b.bytes);
}

// MXFP4 load-time repack of one GGUF row (ncols % 256 == 0; GGUF blocks of
// 32 values = 1 E8M0 byte + 16 code bytes) into the kernels' layout: per 256
// values e[8] then qs[8][16] (136 bytes, same row size). src and dst may be
// the same buffer. Returns the row's largest block exponent (the gemm8 / gemm8t
// `wref` byte). If `lossy` is non-null it is incremented for every non-zero
// block more than 8 exponent steps below that maximum (the fp8 GEMM's
// exponent fold flushes those).
// This is the one implementation: the model loader (uploadMx) calls the
// scratch form with a buffer allocated once per tensor; `scratch` must hold
// rowBytes(QType::mxfp4, ncols) bytes and must not overlap src or dst. The
// 4-argument form allocates that scratch per call (tests, tools).
std::uint8_t repackMxfp4Row(const std::uint8_t* src, std::uint8_t* dst, int ncols, std::uint64_t* lossy,
                            std::uint8_t* scratch);
std::uint8_t repackMxfp4Row(const std::uint8_t* src, std::uint8_t* dst, int ncols, std::uint64_t* lossy);

// ---------------------------------------------------------------------------
// Kernel argument structs (passed by value; layouts match kernels/*.hip)

// One attention layer's paged K / V pools: sequence position p lives in pool
// row ptab[tab + p / kKvPage] * kKvPage + p % kKvPage with tab = kvbase[row]
// when kvbase != 0 (batched rows), else tab0. ks / vs: q8 scales (f16 per 32
// values), 0 for f16 pools.
struct KvArgs {
    DevPtr k = 0;
    DevPtr v = 0;
    DevPtr ks = 0;
    DevPtr vs = 0;
    DevPtr ptab = 0;
    DevPtr kvbase = 0;
    std::int32_t tab0 = 0;
    std::int32_t win = 0;  // attn_wsplit*: MTP draft window (qwen35::draftWindowArg), 0 = off
};
static_assert(sizeof(KvArgs) == 56);

// Grouped one-launch GEMV over up to 3 matrices sharing the input: blocks
// [0, b1) -> matrix 0, [b1, b2) -> 1, the rest -> 2.
struct GvArgs {
    DevPtr w[3] = {0, 0, 0};
    DevPtr y[3] = {0, 0, 0};
    std::uint64_t rb[3] = {0, 0, 0};
    std::int32_t n[3] = {0, 0, 0};
    std::int32_t b1 = 0x7fffffff;
    std::int32_t b2 = 0x7fffffff;
};
static_assert(sizeof(GvArgs) == 96);
// Single-matrix GvArgs (one launch, nrows rows).
inline GvArgs gvSingle(DevPtr w, DevPtr y, std::uint64_t row_bytes, int nrows) {
    GvArgs g;
    g.w[0] = w;
    g.y[0] = y;
    g.rb[0] = row_bytes;
    g.n[0] = nrows;
    return g;
}

struct Tok16 {
    std::int32_t t[16] = {};
};
// per-row tables of a batched verify / MTP step (up to kMaxVerifyRows rows)
struct RowTab {
    std::int32_t tok[kMaxVerifyRows] = {};   // >= 0 token id, < 0: -(index into dev_src) - 1
    std::int32_t pos[kMaxVerifyRows] = {};
    std::int32_t base[kMaxVerifyRows] = {};
};
struct RowIdx {
    std::int32_t v[kMaxVerifyRows] = {};
};
// attn_wsplit query groups: queries first[z] .. first[z] + count[z] - 1.
struct AwGroups {
    std::int32_t first[kMaxVerifyRows] = {};
    std::int32_t count[kMaxVerifyRows] = {};
};
static_assert(sizeof(Tok16) == 64 && sizeof(RowTab) == 384 && sizeof(RowIdx) == 128 && sizeof(AwGroups) == 256);

// Recurrent-state segment of the fused DeltaNet decode kernels: rows
// [row0, row0 + nrows) belong to one sequence whose state is at `state`;
// snap[t] (if nonzero) receives the state after local row t. Replay mode
// (pend != 0): the first npend kept rows in pend are applied first.
struct GdnSeg {
    DevPtr state = 0;
    DevPtr snap[kGdnMaxSnap] = {};
    std::int32_t row0 = 0;
    std::int32_t nrows = 0;
    DevPtr pend = 0;
    std::int32_t npend = 0;
    std::int32_t pad_ = 0;
};
struct GdnSegs {
    GdnSeg s[kGdnMaxSeg] = {};
};
static_assert(sizeof(GdnSeg) == 152 && sizeof(GdnSegs) == 2432);

// moe_route tiles: (expert, first perm position, count, 0).
struct Int4 {
    std::int32_t x = 0, y = 0, z = 0, w = 0;
};

// ---------------------------------------------------------------------------
// Configuration tables (must match the kernel instantiations)

struct GemmCfg {
    int bm, bn, nth;
};
// gemm_c<i>_<T> / gemm_ch<i>_f16: (BM, BN, threads); BK / WAVES_M in comments.
inline constexpr std::array<GemmCfg, 24> kGemmCfgs = {{
    {128, 256, 512},  // bk 64, waves_m 4
    {128, 256, 256},  // bk 64, waves_m 8
    {256, 128, 256},  // bk 64, waves_m 1
    {64, 128, 256},   // bk 32, waves_m 1
    {64, 128, 256},   // bk 32, waves_m 2
    {64, 128, 256},   // bk 32, waves_m 4
    {64, 128, 256},   // bk 64, waves_m 1
    {64, 128, 256},   // bk 64, waves_m 2
    {64, 128, 256},   // bk 64, waves_m 4
    {128, 256, 512},  // bk 32, waves_m 1
    {128, 256, 512},  // bk 32, waves_m 8
    {256, 128, 512},  // bk 32, waves_m 16
    {256, 128, 512},  // bk 64, waves_m 16
    {128, 128, 512},  // bk 32, waves_m 2
    {128, 128, 512},  // bk 32, waves_m 4
    {128, 128, 512},  // bk 32, waves_m 8
    {128, 128, 512},  // bk 64, waves_m 2
    {128, 128, 512},  // bk 64, waves_m 4
    {128, 128, 512},  // bk 64, waves_m 8
    {64, 256, 512},   // bk 64, waves_m 1
    {64, 256, 512},   // bk 64, waves_m 2
    {64, 256, 512},   // bk 64, waves_m 4
    {256, 256, 512},  // bk 32, waves_m 1
    {128, 512, 512},  // bk 32, waves_m 8
}};
// gemms_c<i>_<T> / gemmsh_c<i>_<T> (small batch; c5..c7 = gemmsd variants).
// q8_0 has no c3 and no c7 instantiation; q6_k has no c7. This is the gfx1201
// geometry; a code object may export its own (KernelTable::gemms_geom).
inline constexpr std::array<GemmCfg, 8> kGemmsCfgs = {{
    {64, 32, 128}, {128, 32, 256}, {128, 48, 256}, {128, 64, 256},
    {128, 16, 256}, {64, 64, 256}, {64, 96, 256}, {64, 128, 256},
}};
// gemm8_c<i> / gemm8h_c<i> (MXFP4 x fp8).
inline constexpr std::array<GemmCfg, 5> kGemm8Cfgs = {{
    {128, 256, 256}, {128, 128, 256}, {256, 128, 256}, {128, 64, 256}, {64, 128, 256},
}};
// gemm8t_c<i> / gemm8th_c<i> (fragment-tiled fp8; same bits as gemm8).
inline constexpr std::array<GemmCfg, 5> kGemm8tCfgs = {{
    {128, 256, 256}, {128, 128, 256}, {128, 64, 256}, {128, 32, 256}, {128, 48, 256},
}};
// gemvw_nt<N>v<V>_<T>: rows per block per variant (v0 = not a gemvw kernel).
inline constexpr int kNGemvw = 9;
inline constexpr std::array<int, kNGemvw> kGemvwRows = {0, 16, 32, 16, 32, 16, 16, 32, 32};

// ---------------------------------------------------------------------------
// Kernel table

enum class KvFormat { f16, q8, q8h, q8v };

// What a loaded code object can do. The host picks its paths from these flags
// (and from null KernelTable entries), never from the GPU architecture name.
// gfx1201 has every flag but xd_sum / attn_group1; gfx1151 has no fp8 GEMM,
// no q8v / q8h KV kernels and no vision kernels, and sets both markers.
struct Caps {
    bool fp8_gemm = false;     // MXFP4 x fp8 prefill GEMM (gemm8_c0)
    bool kv_q8v = false;       // q8v KV kernels (attn_decode_q8v)
    bool kv_q8h = false;       // q8h attention prep (attn_prep_q8h, Hadamard-rotated q / k)
    bool gemvw = false;        // int8-WMMA mid-batch GEMV (gemvw_nt2v1_q4_k)
    bool gdn_replay = false;   // fused DeltaNet kernels with replay segments (GdnSeg::pend; ship with
                               // the gfx1201 set, whose int8-WMMA GEMV is the marker)
    bool mrope = false;        // multi-section RoPE attention prep (image prompts: attn_prep_m)
    bool xd_sum = false;       // marker whirl_cap_xd_sum: int8 scale words carry the block sum
    bool attn_group1 = false;  // marker whirl_cap_attn_group1: attn_wsplit1 groups of one query
    bool draft_window = false; // attn_wsplit* honour KvArgs::win (MTP draft window); off when the
                               // module has the marker whirl_cap_no_draft_window (no shipped module
                               // has it since gfx1151 implements the window)
    static Caps probe(const hip::Module& m);
    // Whether `kv` can be loaded (f16 / q8 always can).
    bool supports(KvFormat kv) const { return kv == KvFormat::q8v ? kv_q8v : kv == KvFormat::q8h ? kv_q8h : true; }
};

// Every production kernel, resolved by name from a loaded module. Entries
// indexed by weight type use the ggml type id (index < kNTypes); a null
// Function means the kernel does not exist for that type (or device).
struct KernelTable {
    template <class T>
    using PerType = std::array<T, kNTypes>;
    using F = Function;
    using Nt = std::array<PerType<F>, kMaxSmallBatch - 1>;  // [nt - 2][type]

    Caps caps;            // capability flags of the module (Caps::probe)
    bool gv_grp = false;  // gemvq_<T> / ggemv* take GvArgs (gemv_grouped_abi present)
    PerType<F> gemv1{}, gemvq{}, get_rows{}, gemm{}, dequant_f16{};
    PerType<F> gemv4{}, gemv8{};  // gemv_<T>_4 / _8 (f32 / f16 only): n = 2..16 bitwise == gemv1 per token
    Nt gemvq_nt{}, gemvq_nt_g{};
    std::array<Nt, 2> gemvq_mr{}, gemvq_mr_g{};      // [R/2 - 1][nt - 2][type], R = 2, 4
    std::array<Nt, kNGemvw> gemvw{}, gemvw_g{};      // [v][nt - 2][type], v = 1..8
    std::array<PerType<F>, kGemmCfgs.size()> gemmc{};
    std::array<PerType<F>, kGemmsCfgs.size()> gemms{}, gemmsh{};
    // Block geometry of the gemms slots: kGemmsCfgs, or the code object's own table
    // (marker whirl_cap_gemms_geom, int32 whirl_gemms_geom[slot][3] = BM, BN, threads).
    std::array<GemmCfg, kGemmsCfgs.size()> gemms_geom = kGemmsCfgs;
    PerType<F> gemmhq{}, gemmhqh{};
    PerType<F> gdn_ab{}, gdn_abconv{}, moe_gu{}, moe_down{}, gemm_moe{}, gemm_moe32{}, gemm_moe32r{}, gemm_moegu{};
    PerType<F> gemvx{};  // gemvx_v6_<T>: 17..32-token GEMV (runtime token count)
    F gemvw_head_s{};    // gemvw_nt16v2s_q6_k: 16-token output head over a row range, separate y stride
    F gemvw_head_2p{};   // gemvw_nt16x2s_q6_k: wide-verify output head, both 16-token passes in one launch (optional)
    std::array<F, kMaxSmallBatch> gemv_d2{};       // [nt - 1]
    std::array<F, kGemm8Cfgs.size()> gemm8{}, gemm8h{};
    std::array<F, kGemm8tCfgs.size()> gemm8t{}, gemm8th{};
    std::array<F, kGemmCfgs.size()> gemmch{};

    F rmsnorm{}, l2norm{}, add_inplace{}, silu_mul{}, rope_neox{};
    F attn_decode{}, kv_store{}, attn_split{}, attn_combine{}, attn_prep{}, attn_combine_q8{};
    F attn_wsplit1{}, attn_wsplit2{}, attn_prefill_wmma{}, attn_kx{};
    F attn_kg6{}, attn_kg4{}, attn_kg2{};  // f16 / q8 / q8h / q8v KV (null if not built)
    F gdn_conv_seq{}, gdn_gates{}, gdn_gates_ba{}, gdn_seq_128{}, f32_to_f16{}, gdn_gated_norm{};
    F argmax{}, quantize_q8{};
    F gdn_chunk_prep{}, gdn_chunk_scan{}, gdn_wprep{}, gdn_wscan8{};
    F rmsnorm_x8{}, rmsnorm_x16{}, silu_mul_x8{}, silu_mul_x16{}, gated_norm_x8{}, gated_norm_x16{};
    F gdn_conv_par{}, gdn_conv_state{}, rmsnorm_q8{}, silu_mul_q8{}, gdn_ab_q8_0{};
    F gdn_conv_l2{}, gdn_step_norm{}, requant_q6k_q4k{}, requant_q6k_d2{}, requant_q80_q4k{};
    F copy_rows_map{};  // optional
    F set_tokens{}, argmax_rows{}, argmax_prob{}, draft_pick{};
    F set_rows{}, rmsnorm_q8_rows{}, argmax_rows_to{}, draft_pick_rows{};
    F moe_logits_f32{}, moe_topk{}, moe_route{}, moe_gather_f16{}, moe_act_f16{}, moe_combine{}, moe_tiles{};
    F qact_fp8{}, silu_mul_x8h{}, gdn_conv_l2n{}, gdn_conv_l2n_h{}, gdn_conv_state_h{}, gated_norm_x8h{};
    F silu_mul_x16h{}, gated_norm_x16h{};
    F rmsnorm_x8h16{}, rmsnorm_x8h16t{};
    F qact_fp8t{}, rmsnorm_x8t{}, silu_mul_x8t{}, silu_mul_x8ht{}, gated_norm_x8t{}, gated_norm_x8ht{};
    F gemmh_f16{}, gemmhh_f16{};
    F topk_rows{};
    // vision: attn_prep with multi-section RoPE positions (image prompts; KV-format
    // variant like attn_prep) and set_rpos(rpos, Rpos16 tab, n). Optional.
    F attn_prep_m{}, set_rpos{};
    // MXFP4 routed experts (optional): whole-block decode experts, MXFP4 x fp8
    // grouped expert GEMM (token tile 64 / 32, f32 / f16 output), fp8 gather.
    F moe_gu_mxw{}, moe_down_mxw{};
    F gemm8_moe{}, gemm8_moe32{}, gemm8_moeh{}, gemm8_moe32h{}, moe_gather_fp8{};

    // Resolves every entry. Kernels the model always needs are required
    // (throws hip::Error naming the missing kernel); the others are optional.
    // `kv` picks the KV-format variants of attn_* / kv_store (throws
    // std::invalid_argument when the module lacks that format, see Caps).
    static KernelTable load(const hip::Module& m, KvFormat kv);

    // Accessors with range checks (nt = token count 2..16).
    F gemvqNt(QType t, int nt, bool grouped = false) const;
    F gemvqMr(QType t, int nt, int r, bool grouped = false) const;  // r = 2 or 4
    F gemvW(QType t, int nt, int v, bool grouped = false) const;    // v = 1..8
};

// Kernel names (for diagnostics and for kernels outside the table, e.g. the
// probe / bench kernels).
std::string gemvqNtName(QType t, int nt, bool grouped);
std::string gemvqMrName(QType t, int nt, int r, bool grouped);
std::string gemvwName(QType t, int nt, int v, bool grouped);
std::string gemmcName(QType t, int cfg);
std::string gemmsName(QType t, int cfg, bool f16_out);

}  // namespace whirl::kernels
