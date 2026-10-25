// Attention kernel choice (ARCH-1 K-3): decode / verify split-K attention (attn_ws4 / ws4g, attn_wsplit1 / 2,
// attn_dq4, attn_split + the combine) and prefill attention (attn_kg / kgf / kgs / kx / prefill_wmma / pwf / pws
// with the K / V dequant copies), as pure functions of the problem, the flags, the kernel table and the device
// profile. src/model/forward.cpp builds the problem and launches the plan (it alone knows the buffers).
// SPDX-License-Identifier: Apache-2.0
//
// Bit classes: every decode / verify choice gives each row the partials of a one-query attn_wsplit1 group (the
// groupings are only formed where that holds; caps.attn_group1 forbids them where it does not) - class attn_row;
// attn_dq4 is the q4-KV fast-mode kernel (class attn_q4). The prefill choices are bit-identical to each other for
// one KV format (class attn_prefill): dequantized f16 copies hold the same values the q8 / q4 kernels convert.

#pragma once

#include "whirl/device_profile.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"

#include <cstdint>
#include <span>
#include <vector>

namespace whirl::ops {

using hip::Dim3;
using hip::Function;

// keys the prefill dequant page table covers ([0, kDqMaxKeys); Model::kDqMaxKeys sizes it)
inline constexpr std::uint32_t kDqMaxKeys = 1u << 22;

// ---- decode / verify (n <= small_max rows)
struct AttnDecodeProblem {
    std::uint32_t n = 1;           // query rows
    std::uint32_t head_dim = 256;
    std::uint32_t n_head = 0, n_head_kv = 0;
    std::uint32_t max_ctx = 0;
    bool kv_q4 = false;
    // per-row positions / sequence bases (known only for host-planned rows: row_n == n), the draft window
    bool rows_known = false;
    std::span<const std::int32_t> row_pos, row_base;
    std::int32_t win = 0;
    // the o-projection input can come out of the combine (int8, fused decode on the int8 GEMV path)
    bool o_on_gemv = false;
};
struct AttnDecodeFlags {
    bool attn_ws4 = true, attn_dq4 = true, dq4_dyn = true, attn_wide = true;
    std::uint32_t fd_splits = 64, dbg_flags = 0;
    bool prec_dec = false;
};
enum class AttnDecodeKind : std::uint8_t { ws4, ws4g, wsplit, dq4, split };
struct AttnDecodePlan {
    AttnDecodeKind kind = AttnDecodeKind::split;
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    std::uint32_t n_split = 0;
    kernels::AwGroups groups{};    // ws4 / ws4g / wsplit / dq4: query groups
    bool combine_q8 = false;       // attn_combine_q8 (int8 o-projection input) instead of attn_combine
    Dim3 combine_grid{}, combine_block{256, 1, 1};
};
// caps: the module's markers (attn_group1)
AttnDecodePlan chooseAttnDecode(const AttnDecodeProblem& p, const AttnDecodeFlags& f, const kernels::AttnKernels& k, const kernels::Caps& caps,
                                const device::Profile& prof);
// split rule helpers (kept in step with the kernels)
std::int32_t awPer(std::int32_t n_pos, std::uint32_t n_split);
std::uint32_t dq4Splits(std::span<const std::int32_t> row_pos, std::int32_t win, std::uint32_t full);

// ---- prefill (n > small_max rows), one sequence region (or one segment of a segmented prefill)
struct AttnPrefillProblem {
    std::uint32_t n = 0, pos0 = 0;
    std::uint32_t head_dim = 256, n_head = 0, n_head_kv = 0;
    float scale = 1.0f;
    bool kv_q8 = false;           // q8 / q8h / q8v / q4 (int8 or 4-bit) KV (not f16)
    bool kv_rot = false;          // q and k Hadamard-rotated (q8h, q4): feat027-i8qk runs only then (outlier channels)
    std::uint32_t dq_rows = 0;    // rows the dequant scratch (ffn_g / ffn_u) holds
    bool dq_ptab = false, dq_st_ml = false;  // the scratch page table / softmax state buffers exist
};
struct AttnPrefillFlags {
    bool attn_kg_on = true, attn_kx_on = true, attn_dqf_on = true;
    std::uint32_t attn_seg = 0;
    // feat027-i8qk: int8 Q.K^T in the FC-1c key-range path (lossy fast item i8qk; Model::attn_i8qk = the plan's i8qk)
    bool attn_i8qk = false;
};
// one launch of the plan: what it reads (the cache, or the dequantized copy of keys [r0, r1)) and its geometry
struct AttnStep {
    enum class Op : std::uint8_t { dq_rows, dq_rows_r, attn, i8k };  // i8k: i8k_mean + i8k_quant of the copy (f = i8k_mean, f2 = i8k_quant)
    Op op = Op::attn;
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    bool from_copy = false;      // attn: K / V from the dequantized copy (row r0 of the copy = row 0)
    bool ranged = false;         // attn: a key range [r0, r1) with the softmax state carried (pws / kgs)
    std::uint32_t r0 = 0, r1 = 0, nrows = 0;  // dq_rows: rows (nrows); dq_rows_r / ranged: the key range
    std::uint32_t head0 = 0;     // attn: first head of this launch's head range
    std::uint32_t row_el = 0;    // dq_rows / dq_rows_r: elements per row (n_head_kv * head_dim)
    // feat027-i8qk: i8k (nrows rows of the copy) / attn with i8 = true (attn_kgsi: + kq, ksc, kss, mu). kss = keys per
    // range of the series: the copy's int8 K sits in ffn_g after kss f16 rows, then ksc [n_kv][kss + 32], then mu
    Function f2 = nullptr;
    bool i8 = false;
    std::uint32_t kss = 0;
    Dim3 grid2{};
};
struct AttnPrefillPlan {
    std::vector<AttnStep> steps;
};
AttnPrefillPlan chooseAttnPrefill(const AttnPrefillProblem& p, const AttnPrefillFlags& f, const kernels::AttnKernels& k);

// feat027-i8qk: where the int8 K of a key range lives in ffn_g (byte offsets) for kss keys per range: after the
// f16 K copy (kss rows), the int8 rows (kss x row_el bytes, the pool row layout), the per-key scales
// ([n_kv][kss + 32] f32, 32 rows of padding the kernel's aligned 8-scale loads may touch past the last key), mu
// ([n_kv][head_dim] f32). i8Keys: the largest multiple of 256 whose layout fits dq_rows f16 rows (0: none).
struct I8kLayout {
    std::uint64_t kq = 0, ksc = 0, mu = 0, end = 0;
    std::uint32_t ksc_stride = 0;  // the kernels' kss argument
};
inline I8kLayout i8kLayout(std::uint32_t kss, std::uint32_t n_kv, std::uint32_t head_dim) {
    const std::uint64_t row_el = static_cast<std::uint64_t>(n_kv) * head_dim;
    I8kLayout l;
    l.ksc_stride = kss + 32;
    l.kq = static_cast<std::uint64_t>(kss) * row_el * 2;
    l.ksc = l.kq + static_cast<std::uint64_t>(kss) * row_el;
    l.mu = l.ksc + static_cast<std::uint64_t>(n_kv) * l.ksc_stride * 4;
    l.end = l.mu + row_el * 4;
    return l;
}
inline std::uint32_t i8Keys(std::uint32_t dq_rows, std::uint32_t n_kv, std::uint32_t head_dim) {
    const std::uint64_t cap = static_cast<std::uint64_t>(dq_rows) * n_kv * head_dim * 2;
    for (std::uint32_t kss = dq_rows / 256 * 256; kss >= 256; kss -= 256)
        if (i8kLayout(kss, n_kv, head_dim).end <= cap) return kss;
    return 0;
}

}  // namespace whirl::ops
