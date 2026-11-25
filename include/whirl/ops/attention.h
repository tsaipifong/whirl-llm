// Attention kernel choice (ARCH-1 K-3): decode / verify split-K attention (attn_ws4 / ws4g, attn_wsplit1 / 2,
// attn_dq4, attn_split + the combine) and prefill attention (attn_kg / kgf / kgs / kx / prefill_wmma / pwf / pws
// with the K / V dequant copies), as pure functions of the problem, the flags, the kernel table and the device
// profile, and the prep / core choice of the attention block around them (arch025-F6). src/model/qwen35/forward/attn.cpp
// builds the problem and launches the plan (it alone knows the buffers).
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

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace whirl::ops {

using hip::Dim3;
using hip::Function;

// keys the prefill dequant page table covers ([0, kDqMaxKeys); Model::kDqMaxKeys sizes it)
inline constexpr std::uint32_t kDqMaxKeys = 1u << 22;

// ---- decode / verify (n <= small_max rows)
// v0252-ws4r97b (PLAN V1-87 phase 2): AttnDecodeFlags::attn_ws4r "auto" (= numerics::kAttnWs4rAuto, WHIRL_ATTN_WS4R unset or
// "auto"): the device profile's attn_ws4r_q8 / attn_ws4r_f16 row picks the attn_ws4r variant; 0 / 1 / 2 = off / plain / b
inline constexpr std::uint32_t kAttnWs4rAuto = 255;
// v0252-l2split (PLAN V1-88): AttnDecodeFlags::attn_lsplit "auto" (= numerics::kAttnLsplitAuto): the device profile's
// attn_lsplit_blocks row is the machine-fill block target; 0 = today's split count, N = N blocks (lsplitCount)
inline constexpr std::uint32_t kAttnLsplitAuto = 0xFFFFFFFFu;
struct AttnDecodeProblem {
    std::uint32_t n = 1;           // query rows
    std::uint32_t head_dim = 256;
    std::uint32_t n_head = 0, n_head_kv = 0;
    std::uint32_t max_ctx = 0;
    bool kv_q4 = false;
    bool kv_q8 = false;  // v0252-ws4r97b: q8 / q8h KV (the _q8 decode kernels; the profile's attn_ws4r_q8 row), else f16 / q8v / q4
    // per-row positions / sequence bases (known only for host-planned rows: row_n == n), the draft window
    bool rows_known = false;
    std::span<const std::int32_t> row_pos, row_base;
    // v0252-e1s8 round 2 (C): rows the host does not know, but ONE sequence at consecutive positions (an E1 verify of one slot:
    // set_rows_dp writes pos = the NPOS word + r, one kv base). Context sets it only there (Context::row_seq1); ws4SplitForStep
    // then takes the rows class from n and max_ctx as the bound of every n_pos. Ignored when rows_known
    bool rows_one_seq = false;
    // v0252-e1grp (PLAN V1-93): with rows_one_seq, the interval [pos0_lo, pos0_hi] of row 0's possible positions (the E1
    // lookahead: the cycle in flight accepts 0..nd drafts, so pos0_hi - pos0_lo = nd; a synchronous E1 cycle: one position).
    // The ws4 grouping then keeps rows together that share their split ranges at EVERY position of it (ws4E1Groups).
    // -1 = not given (one-row groups). Ignored when rows_known
    std::int32_t pos0_lo = -1, pos0_hi = -1;
    std::int32_t win = 0;
    // the o-projection input can come out of the combine (int8, fused decode on the int8 GEMV path)
    bool o_on_gemv = false;
};
struct AttnDecodeFlags {
    bool attn_ws4 = true, attn_dq4 = true, dq4_dyn = true, attn_wide = true;
    std::uint32_t fd_splits = 64, dbg_flags = 0;
    bool prec_dec = false;
    std::uint32_t attn_ws4r = kAttnWs4rAuto;  // v0252-ws4r97b: numerics flag attn_ws4r (WHIRL_ATTN_WS4R)
    // v0252-l2split: numerics flag attn_lsplit (WHIRL_ATTN_LSPLIT; numerics plan: precise 0, balance / fast auto). The ops
    // default is 0 (today's split count): a caller that does not pass the mode's flag keeps today's bits
    std::uint32_t attn_lsplit = 0;
    // v0252-8sq4 (PLAN V1-140 / V1-109, F8; q4 KV = fast only). Numerics flags dq4_order (WHIRL_DQ4_ORDER, device::Profile::
    // attn_dq4_row_order) and dq4_min_keys (WHIRL_DQ4_MINKEYS, Profile::dq4_min_keys); ops defaults off = today's kernel / grid.
    //   dq4_order: attn_dq4o (gfx1151 object: attn_dq4 with the gfx1201 block order, verify row fastest; same partials) in
    //     place of attn_dq4 where the code object has it
    //   dq4_min_keys: > 0 = the attn_dq4 split count from the longest host-known row (dq4MinKeysSplits), else today's rule
    bool dq4_order = false;
    std::uint32_t dq4_min_keys = 0;
    // v0252-8sq5 (PLAN V1-109; fast only): numerics flag dq4_splits (WHIRL_DQ4_SPLITS; numerics plan: fast kDq4SplitsAuto,
    // precise / balance 0). 0 = today's rule (dq4Splits), kDq4SplitsAuto = the device profile's dq4_split_table, 1..256 = that
    // count (dq4TableSplits); dq4_min_keys, when > 0, wins. Ops default 0: a caller without the plan's flags keeps today's grid
    std::uint32_t dq4_splits = 0;
};
// v0252-8sq5: AttnDecodeFlags::dq4_splits "auto" (= numerics::kDq4SplitsAuto)
inline constexpr std::uint32_t kDq4SplitsAuto = 65535;
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
// v0252-ws4r97b (PLAN V1-87 phase 2): the gfx1201 attn_ws4r kernel that replaces the wsplit choice (wide = attn_wsplit2,
// else attn_wsplit1) for flag `flag` (kAttnWs4rAuto -> `row`, the profile row of the problem's KV format): 1 -> plain
// attn_ws4r{1,2}[_q8], 2 -> b attn_ws4r{1,2}b[_q8]; 0, any other value, or a table without the kernel -> nullptr (the
// caller keeps attn_wsplit). Exhaustive like attnKgVariant: the kernel is the one with attn_wsplit's arguments and bits.
Function attnWs4rFor(std::uint32_t flag, std::uint8_t row, bool wide, const kernels::AttnKernels& k);
// v0252-l2split (PLAN V1-88): the f16 / q8 attn_wsplit / attn_ws4r split count for flag `flag` (kAttnLsplitAuto -> `row`,
// the profile's attn_lsplit_blocks) given today's count n_split (min(fd_splits, max_ctx / 64)) and the model's KV heads:
// blocks 0 -> n_split (today: bit-identical); else min(n_split, max(2, blocks / n_head_kv)) - one grid of n_head_kv x
// split blocks per query group fills `blocks` (one machine wave on gfx1201). A function of the model and the flag only, never
// of the rows or their positions, so a row's key partition (awPer(n_pos, split), kernels/attn_ws4r.hip:181) is the same in
// any batch, verify group or plain decode. >= 2 keeps the MTP draft-window branch (it needs a second split).
std::uint32_t lsplitCount(std::uint32_t n_split, std::uint32_t n_head_kv, std::uint32_t flag, std::uint16_t row);
// v0252-8sws (PLAN V1-88 on gfx1151): the split constant of the gfx1151 attn_ws4 / attn_ws4g launch: lsplitCount(64, n_head_kv,
// flag, row) (row = device::Profile::attn_ws4_split_blocks) when the KV is q8 / q8h and the table has that constant for the
// single-row kernel and every CG (k.attn_ws4_sp / attn_ws4g_sp at kernels::kWs4SplitCounts), else 64 (today's kernels and
// bits: flag 0 = precise, f16 KV, a head count whose count has no entry, a code object without the entries). A function of the
// model, the KV format, the flag and the table only, never of the rows: a row's key ranges stay a function of its position.
std::uint32_t ws4SplitCount(std::uint32_t n_head_kv, bool kv_q8, std::uint32_t flag, std::uint16_t row, const kernels::AttnKernels& k);
// v0252-8sws round 3: the measured limits (device::Profile::ws4_split_limits) applied to the step: s (a ws4SplitCount result
// other than 64) when the step's row class has a limit >= its key count, else 64. Row class: 0 = one row with a draft window
// argument (the MTP draft row), else the row count n when 1 <= n <= 9 and the rows are one sequence at consecutive positions
// (row_base equal, row_pos[r] = row_pos[0] + r); anything else -> 64. Key count: the largest row_pos + 1 when the host knows
// the rows (rows_known and row_pos holds n entries), else max_ctx for one row (an upper bound), else (several unknown rows) 64.
// Deterministic: a function of the step's rows only (same input -> same row counts -> same split).
std::uint32_t ws4SplitForStep(const AttnDecodeProblem& p, std::uint32_t s, const device::Ws4SplitLimits& lim);
// v0252-e1grp: the ws4 group key of a row at position pp (n_pos = pp + 1) under ns splits and draft window win: the sink
// split's end (window on) and the starts of splits 1 / 2 - ws4::block_range. Rows with equal keys have equal key ranges in
// every split, so an attn_ws4g group of them gives each row the partials of its one-row attn_ws4 launch (kt ws4g, bitwise).
std::array<std::int32_t, 3> ws4GroupKey(std::int32_t pp, std::uint32_t ns, std::int32_t win);
// v0252-e1grp: the widest row-0 interval ws4E1Groups groups for (more positions: one-row groups)
inline constexpr std::int32_t kE1GroupSpan = 16;
// v0252-e1grp (PLAN V1-93): the ws4 groups of n rows of ONE sequence at consecutive positions p0 + r when the host knows
// only p0 in [lo, hi] (an E1 verify): row r + 1 joins row r's group iff its key equals the group head's key at EVERY p0 of
// the interval (the known-rows rule of chooseAttnDecode, for all possible positions at once); groups of at most max_rows.
// lo < 0, lo > hi or hi - lo + 1 > kE1GroupSpan: every row its own group (today's E1 plan). Writes g (n_groups entries),
// widest = the largest group; returns the group count.
std::uint32_t ws4E1Groups(std::uint32_t n, std::int32_t lo, std::int32_t hi, std::uint32_t ns, std::int32_t win, std::uint32_t max_rows,
                          kernels::AwGroups& g, std::uint32_t& widest);
// split rule helpers (kept in step with the kernels)
std::int32_t awPer(std::int32_t n_pos, std::uint32_t n_split);
std::uint32_t dq4Splits(std::span<const std::int32_t> row_pos, std::int32_t win, std::uint32_t full);
// v0252-8sq4 (F8): the attn_dq4 grid for a keys-per-split target: the power of two at or above ceil(n_max / min_keys)
// (n_max = the longest row's key count), clamped to [8, min(full, 128)] (8 = dq4Splits' floor, which also keeps the
// MTP draft window branch on; 128 = the kernel's DQ_MAXS: dq_per never uses more). min_keys 0 or no rows: full.
// Each row's partition is the kernel's dq_per(n_pos, grid): another split / combine order than today's (fast only).
std::uint32_t dq4MinKeysSplits(std::span<const std::int32_t> row_pos, std::int32_t win, std::uint32_t full, std::uint32_t min_keys);
// v0252-8sq5: a grid of s splits the kernel takes for these rows: 1 <= s <= min(full, 256) (full = the partial buffers' split
// count), s >= 8 when a draft window is on (the window branch needs a second split; 8 = dq4Splits' floor), every row's keyed
// splits (the kernel's dq_per partition) inside the grid. The bench whirl-kernel-test --bench dq4s uses the same check.
bool dq4GridOk(std::span<const std::int32_t> row_pos, std::int32_t win, std::uint32_t s, std::uint32_t full);
// v0252-8sq5 (PLAN V1-109): the split count for flag (AttnDecodeFlags::dq4_splits) and the profile table, or 0 = not decided
// here (the caller keeps today's rule). flag 0: 0. flag 1..256: that count where dq4GridOk, else 0. flag kDq4SplitsAuto with
// t.on: the row class - 1 row (window off or not yet active: n_max < the window threshold), 2 / 3 rows of one sequence at
// consecutive positions (verify), 1 row with the window active (the MTP draft row) - then the step at or below n_max = the
// longest row's key count; any other row set (4+ rows, several sequences, gaps, a window with several rows) or n_max below the
// first step: 0. A table count that fails dq4GridOk: 0.
std::uint32_t dq4TableSplits(std::span<const std::int32_t> row_pos, std::span<const std::int32_t> row_base, std::int32_t win, std::uint32_t full,
                             std::uint32_t flag, const device::Dq4SplitTable& t);

// ---- prefill (n > small_max rows), one sequence region (or one segment of a segmented prefill)
struct AttnPrefillProblem {
    std::uint32_t n = 0, pos0 = 0;
    std::uint32_t head_dim = 256, n_head = 0, n_head_kv = 0;
    float scale = 1.0f;
    bool kv_q8 = false;           // q8 / q8h / q8v / q4 (int8 or 4-bit) KV (not f16)
    std::uint32_t dq_rows = 0;    // rows the dequant scratch (ffn_g / ffn_u) holds
    bool dq_ptab = false, dq_st_ml = false;  // the scratch page table / softmax state buffers exist
};
struct AttnPrefillFlags {
    bool attn_kg_on = true, attn_kx_on = true, attn_dqf_on = true;
    std::uint32_t attn_seg = 0;
    // v0252-c1port (C1, feat026-attnkg97 / -97k): the f16 attn_kg variant (numerics flag attn_kg_var, WHIRL_ATTN_KGVAR):
    // 0 the shipped attn_kg, 1 attn_kgv, 3 attn_kgk (both designed bit-identical: every mode); taken on the plain f16
    // path and on Q8P (kv_dq_rows + f16 attn_kgf), never on q8 / q8v without the scratch or multi-range FC-1c (attn_kgs);
    // v0252-c1acc: kAttnKgVarAuto = the per-(NP, path) table kAttnKgAuto; v0252-fasta: var 3 also on one-range FC-1c
    // (KgPath::dqr1, the gfx1201 q4 fast prefill)
    std::uint32_t attn_kg_var = 0;
    // v0252-strixc2 (PLAN V1-09): the balance-mode gfx1151 prefill attention attn_sx<NP> (strix-llama.cpp fattn-rdna35,
    // not bitwise; numerics flag attn_sx = balance + item h16 + device::Profile::attn_sx_balance, WHIRL_ATTN_SX=0 off).
    // Taken for f16 rows walked in ONE key range: the f16 cache (in place of attn_kx / attn_prefill_wmma) and the KG-3
    // single-range copy of q8 / q8h keys (in place of attn_pwf); never on attn_kg / multi-range series / WHIRL_ATTN_SEG.
    bool attn_sx_on = false;
    // v0252-8sfp (PLAN V1-144): the fast-mode gfx1151 int4 Q.K^T forms attn_pwf_i4 / attn_pws_i4 in place of attn_pwf /
    // attn_pws on the KG-3 dequantized-copy path (numerics flag attn_i4qk = fast + device::Profile::attn_i4qk_fast;
    // not bitwise; WHIRL_ATTN_I4QK=0 off). Needs both kernels; the f16 cache paths are unchanged.
    bool attn_i4qk_on = false;
    // v0252-8sattn3 (PLAN V1-148): with attn_i4qk_on, the second int4 forms attn_pwf_i4p / attn_pws_i4p (int4 Q codes in
    // LDS + K / V register prefetch; bit-identical to attn_pwf_i4 / attn_pws_i4) in their place, same steps and grids
    // (numerics flag attn_i4pf; WHIRL_ATTN_I4PF). Needs both i4p kernels; alone (int4 off) it picks nothing.
    bool attn_i4pf_on = false;
};
// v0252-strixc2: the attn_sx kernel for a GQA ratio grp (query heads per KV head) and hp heads per launch at head_dim,
// with its NP (= strix NCOLS2: the first of 8 / 4 / 2 dividing grp, fattn-rdna35.cu:863-870) in *np; nullptr (and *np 0)
// when head_dim != 256, no NP divides both grp and hp, or that kernel is not in the code object. Exhaustive in NP: never
// another NP's kernel (its grid would be wrong).
struct AttnSxPick {
    Function f = nullptr;
    std::uint32_t np = 0;
};
AttnSxPick attnSxFor(const kernels::AttnKernels& k, std::uint32_t grp, std::uint32_t hp, std::uint32_t head_dim);
// query columns per block of attn_sx (64 / NP tokens x NP heads); the grid is ceil(n / (64 / np)) x (hp / np)
inline constexpr std::uint32_t kAttnSxCols = 64;
// v0252-c1port (C1): the f16 attn_kg variant `var` for NP query heads per block, or nullptr = none (the caller keeps
// its shipped kernel, whose NP is the same). Exhaustive: var 1 -> attn_kgv<np>, var 3 -> attn_kgk<np> for np 6 / 4 / 2
// only; any other var, and any other np (8, ...), gives nullptr - never the kernel of another NP (that kernel with
// this np's grid / block would compute wrong output; DISCOVERY v4 C3-1).
Function attnKgVariant(std::uint32_t var, const kernels::AttnKernels& k, std::uint32_t np);
// v0252-c1acc (REVIEW-v0252-c1 C-1 / C-3): attn_kg_var "auto" (numerics::kAttnKgVarAuto, the gfx1201 default) picks
// the variant per (NP, KV path) from the explicit table kAttnKgAuto: var 3 (attn_kgk) only on the paths measured
// faster in the C1 screening (NP 4 on plain f16 KV = Ornith precise / balance; NP 6 on Q8P = Swift balance), var 0 (the
// shipped attn_kg) on every other (NP, path) - the AO-1 re-bench extends it by a table edit. An explicit
// WHIRL_ATTN_KGVAR=0 / 1 / 3 bypasses the table. v0252-c1np6 (PLAN V1-52): + NP 6 on plain f16 KV (Swift precise).
inline constexpr std::uint32_t kAttnKgVarAuto = 255;
// plain f16 KV from the cache; Q8P = kv_dq_rows + the f16 kernel; v0252-fasta: dqr1 = FC-1c with every key in ONE
// dequantized range (kv_dq_rows_r into ffn_g / ffn_u, then one launch series instead of attn_kgs): on gfx1201 the q4
// fast-mode prefill (q8 / q8h reach it only when Q8P is not taken). Only var 3 (attn_kgk) is taken there: M0-A proved
// attn_kgk<np> == one-range attn_kgs<np> bit for bit on q4 dq rows (kt attnkgl Q4P 24/24, 2026-11-08 23:44); attn_kgv
// was not run on them, so var 1 keeps attn_kgs.
enum class KgPath : std::uint8_t { f16, q8p, dqr1 };
struct KgAutoRow {
    std::uint32_t np;
    KgPath path;
    std::uint32_t var;
};
inline constexpr KgAutoRow kAttnKgAuto[] = {
    {4, KgPath::f16, 3},  // Ornith 16 / 2 heads, f16 KV: e2e pp balance 128k +19.8 % (scratch/v0252_c1/gpu/20261106-1320_s)
    {6, KgPath::q8p, 3},  // Swift 24 / 4 heads, q8h KV in Q8P: e2e pp balance 128k +5.1 %, attnbench dq -8..-12.5 %
    {6, KgPath::f16, 3},  // Swift 24 / 4 heads, f16 KV (precise): same-object e2e pp 128k +3.6 % (6/6 rounds; scratch/v0252_c1/
                          // gpu/20261106-2233_acc_S_snap0139), attnp kgk6 f16 -11.6 %; v0252-c1np6 screen 2k / 8k / 32k
    // v0252-fasta (M0-A, R9700 2026-11-08 23:44, attnbench dqr paired median kgk / kgs on the same q4 dq rows; kt Q4P bitwise):
    {4, KgPath::dqr1, 3},  // Ornith fast (q4 KV), one range: 0.768 at 32k, 0.766 at 128k
    {6, KgPath::dqr1, 3},  // Swift fast (q4 KV), one range: 0.896 at 32k, 0.869 at 128k
};
// the table's var for (np, path), 0 for any (np, path) not in it
std::uint32_t attnKgAutoVar(std::uint32_t np, KgPath path);
// the var chooseAttnPrefill uses: auto -> attnKgAutoVar(np, path), any other value as given
inline std::uint32_t attnKgVarFor(std::uint32_t var, std::uint32_t np, KgPath path) {
    return var == kAttnKgVarAuto ? attnKgAutoVar(np, path) : var;
}
// one launch of the plan: what it reads (the cache, or the dequantized copy of keys [r0, r1)) and its geometry
struct AttnStep {
    enum class Op : std::uint8_t { dq_rows, dq_rows_r, attn };
    Op op = Op::attn;
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    bool from_copy = false;      // attn: K / V from the dequantized copy (row r0 of the copy = row 0)
    bool ranged = false;         // attn: a key range [r0, r1) with the softmax state carried (pws / kgs)
    std::uint32_t r0 = 0, r1 = 0, nrows = 0;  // dq_rows: rows (nrows); dq_rows_r / ranged: the key range
    std::uint32_t head0 = 0;     // attn: first head of this launch's head range
    std::uint32_t row_el = 0;    // dq_rows / dq_rows_r: elements per row (n_head_kv * head_dim)
};
struct AttnPrefillPlan {
    std::vector<AttnStep> steps;
};
AttnPrefillPlan chooseAttnPrefill(const AttnPrefillProblem& p, const AttnPrefillFlags& f, const kernels::AttnKernels& k);

// ---- the block around the attention (arch025-F6, moved from src/model/forward/attn.cpp Model::attnBlock)
// q / k norm + RoPE + K / V store: one fused launch (attn_prep; attn_prep_m with multi-section RoPE positions for
// image prompts) or the split rmsnorm x2 + rope_neox x2 + kv_store (f16 KV with WHIRL_NO_FUSE, or no attn_prep in
// the code object, or head_dim > 512). q8 / q4 KV always take the fused prep: it holds the quantizer and the rotation.
enum class AttnPrep : std::uint8_t { mrope, fused, split };
const char* attnPrepName(AttnPrep p);
AttnPrep chooseAttnPrep(bool mrope_on, std::uint32_t head_dim, bool kv_q8, bool no_fuse, const kernels::AttnKernels& k);
// a segmented prefill's per-sequence prep: always the fused launch
constexpr AttnPrep chooseAttnPrepSegmented(bool mrope_on) { return mrope_on ? AttnPrep::mrope : AttnPrep::fused; }
// the attention itself over n query rows: decode / verify (chooseAttnDecode), the WMMA prefill (chooseAttnPrefill,
// head_dim 256) or the naive attn_decode over a scores buffer (WHIRL_NAIVE_ATTN, other head sizes)
enum class AttnCore : std::uint8_t { decode, prefill, naive };
const char* attnCoreName(AttnCore c);
constexpr AttnCore chooseAttnCore(std::uint32_t n, std::uint32_t small_max, std::uint32_t head_dim, bool naive_attn) {
    return n <= small_max ? AttnCore::decode : head_dim == 256 && !naive_attn ? AttnCore::prefill : AttnCore::naive;
}

}  // namespace whirl::ops
