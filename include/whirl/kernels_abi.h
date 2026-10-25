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
//   attn_dq4 (gfx1151 / gfx1201, q4 KV; attn_wsplit1 arguments, block 256, one query per group, n_head / n_kv <= 8)
//   attn_prep[_q8|_q8h|_q8v](qf, kk, vv, qw, kw, KvArgs, pos, n_head, n_kv,
//            hd, n_rot, theta_scale, eps)
//   attn_prefill_wmma[_q8|_q8v], attn_kx[_q8|_q8v](q, KvArgs, out, n_head,
//            n_kv, q_stride, pos, n_tok, scale, int h0)
//   attn_kg6 / attn_kg4 / attn_kg2[_q8|_q8v] (f16 / q8 / q8h / q8v KV; same arguments; grid
//            (ceil(n_tok / 16), heads / NP), block 64 * NP, NP | n_head / n_kv)
//   kv_dq_rows(KvArgs, f16* k16, f16* v16, n_rows, row_el) (q8 / q8h: keys [0, n_rows) of one
//            sequence (tab0) dequantized into contiguous f16 rows; grid ceil(n_rows * row_el / 2048), block 256)
//   kv_dq_rows_r[_q4](KvArgs, f16* k16, f16* v16, r0, n_rows, row_el): keys [r0, r0 + n_rows) into
//            rows 0 .. n_rows - 1 (q8 / q8h, q4; gfx1201)
//   attn_prefill_wmma_s (attn_prefill_wmma arguments, int kb0, int kb1, float* st_ml; gfx1151):
//     the f16 attn_prefill_wmma over keys [kb0, kb1), softmax state carried as attn_kgs (KG-3)
//   attn_kgs6 / 4 / 2 (attn_kg arguments, int kb0, int kb1, float* st_ml): f16 attn_kg over keys
//            [kb0, kb1); m / l in st_ml[query][head][2] and the unnormalized output in out carry the
//            softmax state between ranges (bit-identical to one attn_kg over all keys; gfx1201)
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
//   gdn_sx_prep(conv, gl, beta, f16* Tb, f16* Pb, float2* Gb, f16* Qh, f16* Kh, f16* Vh, H, Hk, n_tok,
//            nch32, conv_stride, scale); gdn_sx_scan(Qh, Kh, Vh, state_in, out, state_out, Tb, Pb, Gb, H, Hk,
//            n_tok, nch32, scale, on_w, on_z, on_out, u16* on_out16, on_eps, bool on_silu): gfx1151 f16-WMMA chunked prefill (strix-llama.cpp, MIT; GDN-8S, optional)
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
//   draft_sample_rows(x, n, int* ctl_all, RowIdx area, int r, const int* map, DSampArgs a, int* qbuf)
//            (optional) after draft_pick_rows: rows with a.on draw draft r from the filtered draft
//            distribution q and write q to qbuf + a.qoff (whirl/spec_sample.h layout)
//   set_tokens(ids, pos, dev_src, Tok16 toks, n, n_host, pos0)
//   set_rows(ids, pos, kvbase, dev_src, RowTab tab, n)
//   topk_rows(x, n, K, float inv_t, int* ids, float* vals, float* stats)
//   requant_q6k_q4k(src, u64 src_rb, dst, u64 dst_rb); requant_q6k_d2(src,
//            u64 src_rb, dst, int ncols); requant_q80_q4k(src, u64 src_rb, dst, u64 dst_rb)
//   copy_rows_map(src, u64 rb, dst, const int* map): dst row i = src row map[i] (optional)
//  DFlash2 drafter (optional; kernels/dflash.hip):
//   dflash_conv(x, dyn, base, y, int T, int C, int K, int group, int side, int dyn_stride)
//   dflash_topk(x, int n, int ld, int k, int* ids, float* vals)       (key order, ties -> smaller id)
//   dflash_lattice(cand, unary, gate, prev_cb, next_cb, u64 cb_rb, int q8, int R, int K, int anchor, lat)
//   dflash_walk(lat, cand, int P, int K, float p_min, int n_min, int* out)
//   dflash_attn(q, kb, vb, kc, vc, const int* cpos, int W, int B, int nh, int nkv, int hd, int qpos, int window,
//               float scale, out)                                    (grid (nh, B), block hd; W + B <= kDFlashAttnKeys)
//   dflash_attn_long(same arguments)                                  (W + B > kDFlashAttnKeys: chunked, online softmax)
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
//   gemm8r_moe / gemm8r_moe32 / gemm8r_moeh / gemm8r_moe32h (optional): same arguments,
//            same outputs; 1-D grid n_tile_slots * nbt, nbt = 2 * ceil(rows_e / kMoeBm)
//            with W2 else ceil(rows_e / kMoeBm), row block fastest
//   gemmr_moegu / gemmr_moegu32 (optional, MXFP4)(Wg, Wu, rb, F, const f16* X, f16* A, ncols,
//            const Int4* tiles, const int* n_tiles): grouped gate + up of the f16 (precise) MoE
//            prefill, A = f16(silu(g) * u) (moe_act_f16's values; A must not alias X); 1-D grid
//            n_tile_slots * ceil(F / 64), token tile 64 / 32
//   gemmr_moedn / gemmr_moedn32 (optional, MXFP4)(W, null, rb, E, const f16* A, float* Y, F,
//            tiles, n_tiles): grouped down, = gemm_moe / gemm_moe32 values; grid
//            n_tile_slots * ceil(E / 128)
//   moe_gather_fp8(x, perm, u8* x8, float* sx, C, K): per-pos fp8 rows + scales
//  feat026-moep, Radeon 8060S MoE prefill (optional, gfx1151 only, MXFP4; variants kMoepGu / kMoepDn):
//   gemm_moepgu_<v>_mxfp4: gemm_moegu_<T> arguments; tiles from moe_tiles with bn = the variant's
//            BNMAX; 1-D grid n_tile_slots(bn) * ceil(ff / GR); A == gemm_moegu's bits
//   gemm_moepdn_<v>_mxfp4: gemm_moe32r_<T> arguments; tiles bn = BNMAX; 1-D grid
//            n_tile_slots(bn) * ceil(rows_e / BM); Y == gemm_moe32 / gemm_moe32r bits
//   gemm_moepdc_<v>_mxfp4(gemm_moepdn args..., perm, inv, w, sg, ysh, x, int* cnt, K): the same Y,
//            plus moe_combine(x, ysh, sg, Y, inv, w, rows_e, K) done in the epilogue by the block that
//            completes each (token, row block); cnt: n * ceil(rows_e / BM) ints, zero before, zero after
//   moe_combine4: moe_combine arguments, grid (ceil(C / 1024), n), C % 4 == 0; same bits
//   balance (item f16 intermediates) / fast candidates, never in precise (acceptance: task accuracy):
//   gemm_moepdh_<v>_mxfp4(W, rb, rows_e, X, f16* Y, ncols, tiles, n_tiles): Y stored as f16
//   gemm_moepdhc_<v>_mxfp4: gemm_moepdc arguments with f16* Y (fused combine reads f16)
//   moe_combine4h(x, ysh, sg, const f16* yd, inv, w, C, K): moe_combine4 over f16 yd
//   gemm_moepda_<v>_mxfp4(W, rb, rows_e, X, ncols, tiles, n_tiles, perm, w, x, K): x += w * y by f32
//            atomics (no yd; order varies run to run); the shared expert added before by moe_combine4, K = 0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
inline constexpr int kFdMaxSplits = 64;      // host cap on split count (Model::fd_splits: 256 with q4 KV)
inline constexpr int kGdnMaxSeg = 16;        // GDN_MAX_SEG
inline constexpr int kGdnMaxSnap = 15;       // GDN_MAX_SNAP
inline constexpr int kMaxSmallBatch = 16;    // multi-token GEMV / small-batch kernels: 2..16 tokens
inline constexpr int kMaxVerifyRows = 32;    // rows of one batched verify / MTP step (gemvx_v6_* past 16)
inline constexpr int kMaxDrafts = 10;        // MTP drafts per verify cycle
inline constexpr int kDFlashAttnKeys = 2304;  // DF_MAXK: keys (context ring + block) of dflash_attn; more: dflash_attn_long
inline constexpr int kDFlashMaxKeys = 65536;  // host limit on ring + block for dflash_attn_long (ring VRAM, int indexing)
inline constexpr int kGemm3Bm = 256, kGemm3Bn = 256, kGemm3Threads = 512;  // GEMM3_*
inline constexpr int kMoeBm = 128, kMoeBn = 64;                            // MOE_BM, gemm_moe_* token tile
// gemmr_moe* (gfx1201 precise MoE prefill) step k by BK with no tail: gate + up over E (BK 64), down over F
// (BK 32 / 64). The host uses them only when both are multiples of 64 (FIX-K97); else gemm_moe + moe_act_f16.
inline constexpr int kGemmrMoeBk = 64;
inline constexpr bool gemmrMoeShapeOk(std::uint32_t E, std::uint32_t F) {
    return E > 0 && F > 0 && E % kGemmrMoeBk == 0 && F % kGemmrMoeBk == 0;
}
inline constexpr int kMoeBn32 = 32;                                        // gemm_moe32_* token tile
// feat026-moep variants (kernels/gfx1151/moe_prefill_mt.hip MOEPGU_ENTRY / MOEPDN_ENTRY; keep in step).
struct MoepGuVariant {
    const char* name;
    int gr, bn, bk;  // gate (= up) rows per block, max tokens per tile, k step
};
struct MoepDnVariant {
    const char* name;
    int bm, wm, bn, bk;  // rows per block, waves along rows, max tokens per tile, k step
};
inline constexpr MoepGuVariant kMoepGu[] = {{"g64n64k64", 64, 64, 64},  {"g64n128k64", 64, 128, 64}, {"g64n128k32", 64, 128, 32},
                                            {"g32n128k64", 32, 128, 64}, {"g64n256k32", 64, 256, 32}, {"g128n128k32", 128, 128, 32}};
inline constexpr MoepDnVariant kMoepDn[] = {{"m128w4n32k64", 128, 4, 32, 64},  {"m128w4n64k64", 128, 4, 64, 64},
                                            {"m128w4n128k64", 128, 4, 128, 64}, {"m128w8n128k64", 128, 8, 128, 64},
                                            {"m64w4n128k64", 64, 4, 128, 64},   {"m128w4n256k32", 128, 4, 256, 32}};
inline constexpr int kMoepGuN = static_cast<int>(sizeof(kMoepGu) / sizeof(kMoepGu[0]));
inline constexpr int kMoepDnN = static_cast<int>(sizeof(kMoepDn) / sizeof(kMoepDn[0]));
inline constexpr int kMoepMinBm = 64;  // smallest BM of kMoepDn: sizes the fused-combine counter buffer
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
    // the 4 bytes the kernel's struct has as tail padding, named and zeroed here: an unnamed tail held whatever
    // the stack held, and the launch log digests every argument byte (REVIEW-ARCH4 P3-2, arch6e). Not read by
    // the kernels; the layout (and the bytes the kernels read) is unchanged.
    std::int32_t pad_ = 0;
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
// draft_sample_rows per-row parameters (speculative sampling of MTP drafts)
struct DSampArgs {
    std::int32_t on[kMaxVerifyRows] = {};
    std::int32_t top_k[kMaxVerifyRows] = {};
    std::int32_t qoff[kMaxVerifyRows] = {};
    float inv_t[kMaxVerifyRows] = {};
    float top_p[kMaxVerifyRows] = {};
    float min_p[kMaxVerifyRows] = {};
    std::uint32_t key_lo[kMaxVerifyRows] = {};  // whirl::spec::raceBase(seed, output index of draft r)
    std::uint32_t key_hi[kMaxVerifyRows] = {};
};
static_assert(sizeof(DSampArgs) == 1024);
}  // namespace whirl::kernels
// DSampArgs holds floats, so std::has_unique_object_representations is false for it (hip::launch's no-padding
// check); its 8 arrays of 4-byte elements fill sizeof exactly, so it has no padding
static_assert(sizeof(whirl::kernels::DSampArgs) == 8 * whirl::kernels::kMaxVerifyRows * 4);
template <>
struct whirl::hip::LaunchArgNoPadding<whirl::kernels::DSampArgs> : std::true_type {};
namespace whirl::kernels {
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

// Which 8-byte words of each argument struct hold device addresses (hip::LaunchArgPtrWords, the launch-log digest;
// REVIEW-ARCH4 P3-2, arch6f). Integer words are digested as values even when they happen to fall inside an
// allocation. The offsets are checked against the layouts here.
static_assert(offsetof(KvArgs, kvbase) == 40 && offsetof(KvArgs, tab0) == 48);
static_assert(offsetof(GvArgs, y) == 24 && offsetof(GvArgs, rb) == 48);
static_assert(offsetof(GdnSeg, snap) == 8 && offsetof(GdnSeg, row0) == 8 + 8 * kGdnMaxSnap &&
              offsetof(GdnSeg, pend) == 16 + 8 * kGdnMaxSnap && offsetof(GdnSeg, npend) == 24 + 8 * kGdnMaxSnap &&
              sizeof(GdnSeg) % 8 == 0);
}  // namespace whirl::kernels
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::KvArgs> {  // k v ks vs ptab kvbase; then tab0 / win
    static bool ptr(std::uint32_t w) { return w < 6; }
};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::GvArgs> {  // w[3] y[3]; then rb[3] (row bytes) and ints
    static bool ptr(std::uint32_t w) { return w < 6; }
};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::GdnSegs> {  // per segment: state snap[]; row0 / nrows; pend; npend
    static bool ptr(std::uint32_t w) {
        constexpr std::uint32_t kWords = sizeof(whirl::kernels::GdnSeg) / 8;
        constexpr std::uint32_t kInts = offsetof(whirl::kernels::GdnSeg, row0) / 8;
        constexpr std::uint32_t kPend = offsetof(whirl::kernels::GdnSeg, pend) / 8;
        const std::uint32_t i = w % kWords;
        return i < kInts || i == kPend;
    }
};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::GdnSeg> : whirl::hip::LaunchArgPtrWords<whirl::kernels::GdnSegs> {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::Tok16> : whirl::hip::LaunchArgNoPtrWords {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::RowTab> : whirl::hip::LaunchArgNoPtrWords {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::RowIdx> : whirl::hip::LaunchArgNoPtrWords {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::AwGroups> : whirl::hip::LaunchArgNoPtrWords {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::DSampArgs> : whirl::hip::LaunchArgNoPtrWords {};
template <>
struct whirl::hip::LaunchArgPtrWords<whirl::kernels::Int4> : whirl::hip::LaunchArgNoPtrWords {};
namespace whirl::kernels {

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

enum class KvFormat { f16, q8, q8h, q8v, q4 };

// What a loaded code object can do. The host picks its paths from these flags
// (and from null KernelTable entries), never from the GPU architecture name.
// gfx1201 has every flag but xd_sum / attn_group1; gfx1151 has no fp8 GEMM,
// no vision kernels and sets both markers; only gfx1151 has the q4 KV kernels.
struct Caps {
    bool fp8_gemm = false;     // MXFP4 x fp8 prefill GEMM (gemm8_c0)
    bool kv_q8v = false;       // q8v KV kernels (attn_decode_q8v)
    bool kv_q8h = false;       // q8h attention prep (attn_prep_q8h, Hadamard-rotated q / k)
    bool kv_q4 = false;        // q4 KV kernels (attn_prep_q4: 4-bit K / V, Hadamard-rotated q / k)
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
    bool supports(KvFormat kv) const {
        return kv == KvFormat::q8v ? kv_q8v : kv == KvFormat::q8h ? kv_q8h : kv == KvFormat::q4 ? kv_q4 : true;
    }
};

// Every production kernel, resolved by name from a loaded module, in one table per op (ARCH-1 K-5): the
// matmul family (GEMV / GEMM and their activation converters), attention (the only table the KV format picks
// variants for), MoE experts, the DeltaNet layer, and the rest (norms, activations, sampling, drafting). Each
// op's kernel choice (src/ops) reads its own table; KernelTable is all of them (KernelTable::load). Entries
// indexed by weight type use the ggml type id (index < kNTypes); a null Function means the kernel does not
// exist for that type (or device).
template <class T>
using PerType = std::array<T, kNTypes>;
using Nt = std::array<PerType<Function>, kMaxSmallBatch - 1>;  // [nt - 2][type]

struct MatmulKernels {
    using F = Function;
    bool gv_grp = false;  // gemvq_<T> / ggemv* take GvArgs (gemv_grouped_abi present)
    PerType<F> gemv1{}, gemvq{}, gemm{}, dequant_f16{};
    PerType<F> gemv4{}, gemv8{};  // gemv_<T>_4 / _8 (f32 / f16 only): n = 2..16 bitwise == gemv1 per token
    // gemvh_<T> / gemvh2_<T>: precise decode / verify matmul, f16 x, n <= 16 / 32, bitwise == gemm_c* (gfx1201)
    PerType<F> gemvh{}, gemvh2{};
    Nt gemvq_nt{}, gemvq_nt_g{};
    std::array<Nt, 2> gemvq_mr{}, gemvq_mr_g{};      // [R/2 - 1][nt - 2][type], R = 2, 4
    std::array<Nt, kNGemvw> gemvw{}, gemvw_g{};      // [v][nt - 2][type], v = 1..8
    std::array<PerType<F>, kGemmCfgs.size()> gemmc{};
    std::array<PerType<F>, kGemmsCfgs.size()> gemms{}, gemmsh{};
    // Block geometry of the gemms slots: kGemmsCfgs, or the code object's own table
    // (marker whirl_cap_gemms_geom, int32 whirl_gemms_geom[slot][3] = BM, BN, threads).
    std::array<GemmCfg, kGemmsCfgs.size()> gemms_geom = kGemmsCfgs;
    PerType<F> gemmhq{}, gemmhqh{};
    PerType<F> gemvx{};  // gemvx_v6_<T>: 17..32-token GEMV (runtime token count)
    F gemvw_head_s{};    // gemvw_nt16v2s_q6_k: 16-token output head over a row range, separate y stride
    F gemvw_head_2p{};   // gemvw_nt16x2s_q6_k: wide-verify output head, both 16-token passes in one launch (optional)
    std::array<F, kMaxSmallBatch> gemv_d2{};       // [nt - 1]
    std::array<F, kGemm8Cfgs.size()> gemm8{}, gemm8h{};
    std::array<F, kGemm8tCfgs.size()> gemm8t{}, gemm8th{};
    std::array<F, kGemmCfgs.size()> gemmch{};
    F gemmh_f16{}, gemmhh_f16{};
    // activation converters of the GEMV / GEMM inputs
    F f32_to_f16{}, quantize_q8{}, qact_fp8{}, qact_fp8t{};

    // Accessors with range checks (nt = token count 2..16).
    F gemvqNt(QType t, int nt, bool grouped = false) const;
    F gemvqMr(QType t, int nt, int r, bool grouped = false) const;  // r = 2 or 4
    F gemvW(QType t, int nt, int v, bool grouped = false) const;    // v = 1..8
};

struct AttnKernels {
    using F = Function;
    F rope_neox{};
    F attn_decode{}, kv_store{}, attn_split{}, attn_combine{}, attn_prep{}, attn_combine_q8{};
    F attn_wsplit1{}, attn_wsplit2{}, attn_prefill_wmma{}, attn_kx{};
    F attn_dq4{};  // q4 KV decode / verify (gfx1151 / gfx1201; null if not built): attn_wsplit1 arguments, 256 threads
    F attn_ws4{};  // gfx1151 4-wave pipelined attn_wsplit1 (same arguments / bits, one query per group): decode rows (WHIRL_ATTN_WS4=0: old path)
    F attn_ws4g[3]{};  // gfx1151 multi-row verify, CG 1..3 tiles (attn_ws4g{1,2,3}[_q8]; 256 threads, grid.x = n_kv x column groups)
    F attn_kg6{}, attn_kg4{}, attn_kg2{};  // f16 / q8 / q8h / q8v KV (null if not built)
    // q8 / q8h KV prefill (Q8P): kv_dq_rows + the f16 attn_kg variants (null otherwise)
    F kv_dq_rows{}, attn_kgf6{}, attn_kgf4{}, attn_kgf2{};
    // FC-1c: key-range variants (prefills past the f16 scratch, q4 prefill): kv_dq_rows_r (q8 / q8h)
    // or kv_dq_rows_r_q4 (q4) + attn_kgs* (f16 attn_kg over a key range, softmax state carried)
    F kv_dq_rows_r{}, attn_kgs6{}, attn_kgs4{}, attn_kgs2{};
    // KG-3 (gfx1151): q8 / q8h / q4 prefill on dequantized rows: the f16 attn_prefill_wmma and its
    // key-range variant attn_prefill_wmma_s (softmax state carried; null when not in the code object)
    F attn_pwf{}, attn_pws{};
    // vision: attn_prep with multi-section RoPE positions (image prompts; KV-format
    // variant like attn_prep) and set_rpos(rpos, Rpos16 tab, n). Optional.
    F attn_prep_m{}, set_rpos{};
};

struct MoeKernels {
    using F = Function;
    // moe_gu_f_<T> / moe_down_f_<T>: precise MoE decode experts (f32 activations), both code objects
    PerType<F> moe_gu_f{}, moe_down_f{};
    PerType<F> moe_gu{}, moe_down{}, gemm_moe{}, gemm_moe32{}, gemm_moe32r{}, gemm_moegu{};
    F moe_logits_f32{}, moe_topk{}, moe_route{}, moe_gather_f16{}, moe_act_f16{}, moe_combine{}, moe_tiles{};
    // MXFP4 routed experts (optional): whole-block decode experts, MXFP4 x fp8
    // grouped expert GEMM (token tile 64 / 32, f32 / f16 output), fp8 gather.
    F moe_gu_mxw{}, moe_down_mxw{};
    F gemm8_moe{}, gemm8_moe32{}, gemm8_moeh{}, gemm8_moe32h{}, moe_gather_fp8{};
    F gemm8r_moe{}, gemm8r_moe32{}, gemm8r_moeh{}, gemm8r_moe32h{};  // optional
    F gemmr_moegu{}, gemmr_moegu32{}, gemmr_moedn{}, gemmr_moedn32{};  // optional (MXFP4, f16 activations)
    // feat026-moep (optional, gfx1151, MXFP4): multi-fragment expert GEMMs per variant, fused-combine down twins,
    // the vectorised combine
    std::array<F, kMoepGuN> moep_gu{};
    std::array<F, kMoepDnN> moep_dn{}, moep_dc{};
    F moe_combine4{};
    // balance / fast candidates (f16 expert outputs; atomic scatter)
    std::array<F, kMoepDnN> moep_dh{}, moep_dhc{}, moep_da{};
    F moe_combine4h{};
};

struct DeltaNetKernels {
    using F = Function;
    PerType<F> gdn_ab{}, gdn_abconv{};
    F gdn_conv_seq{}, gdn_gates{}, gdn_gates_ba{}, gdn_seq_128{}, gdn_gated_norm{};
    F gdn_chunk_prep{}, gdn_chunk_scan{}, gdn_wprep{}, gdn_wscan8{}, gdn_sx_prep{}, gdn_sx_scan{};
    F gdn_conv_par{}, gdn_conv_state{}, gdn_ab_q8_0{}, gdn_conv_l2{}, gdn_step_norm{};
    F gdn_conv_l2n{}, gdn_conv_l2n_h{}, gdn_conv_state_h{};
};

// norms, activations (also the fused activation outputs of the FFN), embedding rows, sampling / drafting,
// weight requantization, DFlash2
struct MiscKernels {
    using F = Function;
    PerType<F> get_rows{};
    F rmsnorm{}, l2norm{}, add_inplace{}, silu_mul{};
    F argmax{};
    F rmsnorm_x8{}, rmsnorm_x16{}, silu_mul_x8{}, silu_mul_x16{}, gated_norm_x8{}, gated_norm_x16{};
    F rmsnorm_q8{}, silu_mul_q8{};
    F requant_q6k_q4k{}, requant_q6k_d2{}, requant_q80_q4k{};
    F copy_rows_map{};  // optional
    F dflash_conv{}, dflash_topk{}, dflash_lattice{}, dflash_walk{}, dflash_attn{}, dflash_attn_long{};  // optional (DFlash2 drafter)
    F set_tokens{}, argmax_rows{}, argmax_prob{}, draft_pick{};
    F set_rows{}, rmsnorm_q8_rows{}, argmax_rows_to{}, draft_pick_rows{};
    F draft_sample_rows{};  // optional (null: specsample skipped)
    F silu_mul_x8h{}, gated_norm_x8h{};
    F silu_mul_x16h{}, gated_norm_x16h{};
    F rmsnorm_x8h16{}, rmsnorm_x8h16t{};
    F rmsnorm_x8t{}, silu_mul_x8t{}, silu_mul_x8ht{}, gated_norm_x8t{}, gated_norm_x8ht{};
    F topk_rows{};
};

struct KernelTable : MatmulKernels, AttnKernels, MoeKernels, DeltaNetKernels, MiscKernels {
    template <class T>
    using PerType = kernels::PerType<T>;
    using F = Function;
    using Nt = kernels::Nt;

    Caps caps;  // capability flags of the module (Caps::probe)

    // Resolves every entry. Kernels the model always needs are required
    // (throws hip::Error naming the missing kernel); the others are optional.
    // `kv` picks the KV-format variants of attn_* / kv_store (only AttnKernels; throws
    // std::invalid_argument when the module lacks that format, see Caps).
    static KernelTable load(const hip::Module& m, KvFormat kv);
};

// Kernel names (for diagnostics and for kernels outside the table, e.g. the
// probe / bench kernels).
std::string gemvqNtName(QType t, int nt, bool grouped);
std::string gemvqMrName(QType t, int nt, int r, bool grouped);
std::string gemvwName(QType t, int nt, int v, bool grouped);
std::string gemmcName(QType t, int cfg);
std::string gemmsName(QType t, int cfg, bool f16_out);


// gemvh / gemvh2 on gfx1151 (one wave per 16 weight rows; the blocks loop over row tiles): launching every
// tile as its own block leaves a partial last round and a long single-wave tail once the matrix has more
// tiles than the GPU holds waves (17408 rows = 1088 tiles). The grid is cut to ceil(tiles / k) blocks with
// k = ceil(tiles / cap), so each block gets k or k - 1 tiles. Any grid gives the same bits (the kernel test
// checks the capped production grid and small caps).
inline unsigned gemvhGridCap(unsigned nrows, unsigned cap) {
    const unsigned tiles = (nrows + 15) / 16;
    const unsigned k = (tiles + cap - 1) / cap;
    return k > 1 ? (tiles + k - 1) / k : tiles;
}
// cap = kGemvhWavesPerMp x the current device's multiProcessorCount, which HIP reports in WGPs (dual CUs) on RDNA:
// 32 x 20 = 640 on the Radeon 8060S (40 CUs), where 640 was tuned (FIX-RC4-B measured 20 there); a gfx1151 part
// with fewer CUs gets proportionally fewer blocks. Read once per process (the gfx1151 path runs on one device).
// WHIRL_GEMVH_WAVES=<n> overrides (tests).
inline constexpr unsigned kGemvhWavesPerMp = 32;
inline unsigned gemvhCap() {
    static const unsigned cap = [] {
        if (const char* e = std::getenv("WHIRL_GEMVH_WAVES")) {
            const int v = std::atoi(e);
            if (v > 0) return static_cast<unsigned>(v);
        }
        const int mps = hip::describeDevice(hip::getDevice()).compute_units;
        return mps > 0 ? kGemvhWavesPerMp * static_cast<unsigned>(mps) : 640u;
    }();
    return cap;
}
inline unsigned gemvhGrid(unsigned nrows) { return gemvhGridCap(nrows, gemvhCap()); }
}  // namespace whirl::kernels
