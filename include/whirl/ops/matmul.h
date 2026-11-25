// Matmul kernel choice (ARCH-1 K-1 GEMV): which kernel a matmul of n token rows launches, with which grid, as a
// pure function of the problem (weight type, shape, rows), the numerics flags, the code object's kernel table and
// the tuned tables. src/model/qwen35/forward/dispatch.cpp builds the problem, asks here, and launches what comes back.
// SPDX-License-Identifier: Apache-2.0
//
// Every kernel carries a bit-equivalence class (BitClass): kernels of one class give the same bits for the same
// token row, so the choice inside a class is free (tuning, geometry, grouping), and a choice across classes is a
// numerics decision. Precise picks only the precise class (the f16 GEMM family, bitwise == prefill) for decode /
// verify rows; that rule is written once, in preciseRows / preciseAllows below, and every row-count gate asks it
// (chooseMatmulPath, chooseGemmF16 / precChoice, chooseActIn in act.cpp, chooseGdnDecode - the fused GDN
// [beta; alpha] step - in gdn.cpp, and in src/model/qwen35/forward/ the grouped int8 GEMV (dispatch.cpp) and the fused
// silu_mul_q8 (ffn.cpp)). Mode-level precise gates that do not depend on the row
// count stay with their op: the attention combine (attention.cpp), the MoE decode experts (moe.cpp, which extends
// precise rows to max_verify_rows) and the wide verify (wideAllowed, act.h). Tests: tests/ops/ops_tests.cpp (CPU, fake kernel tables).

#pragma once

#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"

#include <array>
#include <cstdint>

namespace whirl::ops {

using gguf::GgmlType;
using hip::Dim3;
using hip::Function;

enum class BitClass : std::uint8_t {
    q8_row,    // int8 activations (quantize_q8): every int8 GEMV variant, each token row == the one-token gemvq
    f32_row,   // f32 activations: gemv_<T>_1 / _4 / _8, each token row == gemv_<T>_1 (C-13)
    f16_gemm,  // f16 activations: gemm_c / gemms / gemmh / gemmhq / dequant + gemm / gemvh, == each other (precise)
    fp8_gemm,  // fp8 activations (MXFP4 speed mode, lossy)
};
const char* bitClassName(BitClass b);
// the classes precise mode may launch for decode / verify rows: bitwise the prefill (and any batch) result
constexpr bool preciseClass(BitClass b) { return b == BitClass::f16_gemm; }
// precise decode / verify rows: precise mode and n within the row range of the path being chosen (rows_max: gv_max
// for the matmul / GEMV choices, small_max for the fused decode kernels, which only run when n <= small_max)
constexpr bool preciseRows(bool prec_dec, std::uint32_t n, std::uint32_t rows_max) { return prec_dec && n <= rows_max; }
// whether a kernel of class b may run for these rows: anything outside precise rows, only the precise class inside
constexpr bool preciseAllows(BitClass b, bool prec_dec, std::uint32_t n, std::uint32_t rows_max) {
    return !preciseRows(prec_dec, n, rows_max) || preciseClass(b);
}

enum class MatmulPath : std::uint8_t { gemvq, gemv1, float_gemv, gemm_fp8, gemm_f16 };
BitClass bitClassOf(MatmulPath p);

// one matrix of a matmul
struct MatShape {
    GgmlType ty = GgmlType::f32;
    std::uint32_t nrows = 0, ncols = 0;
    std::uint64_t row_bytes = 0;
    bool has_ref = false;  // MXFP4 load-time row exponents (fp8 GEMM operand)
    bool head = false;     // the output head (vocabulary rows)
};

// the numerics / execution flags the matmul choice reads (Model members today; R-5 ExecFlags)
struct MatmulFlags {
    bool prec_dec = false;     // precise: small batches on the f16 GEMM
    bool float_gemv = false;   // no int8 GEMV (WHIRL_FLOAT_GEMV)
    bool fp8_prefill = false;  // MXFP4 x fp8 prefill GEMM on (balance / fast)
    std::uint8_t fp8_mask = 0, mm_class = 0;
    std::uint32_t gv_max = 16;  // Model::gvMax(): GEMV rows (16, or 32 in a wide verify)
};

// matmul path for n token rows of w (Model::matmulPath before K-1)
MatmulPath chooseMatmulPath(const MatShape& w, std::uint32_t n, const MatmulFlags& f, const kernels::MatmulKernels& k);
// f32 / f16 weights at n = 2..gv_max without an int8 GEMV: the f32-activation GEMV instances
bool floatGemvN(const MatShape& w, std::uint32_t n, std::uint32_t gv_max, const kernels::MatmulKernels& k);

// ---- int8 GEMV (MatmulPath::gemvq), n <= 16 token rows of 1..3 matrices of one type on the same input
using GemvR = std::array<std::uint8_t, kernels::kMaxSmallBatch + 1>;
struct GemvTune {
    const GemvR* gemv_r = nullptr;                                  // rows-per-block multiplier per n (profile / tune)
    const std::array<GemvR, kernels::kNTypes>* gemv_w = nullptr;    // gemvw variant per type and n (0: none)
    const GemvR* gemv_w_head = nullptr;                             // gemvw variant for the q6_k head per n
    std::uint32_t gv_nmax_override = 0;                             // WHIRL_GV_NMAX (0: per type, 13 mxfp4 / 11)
    // feat026-ggemv (WHIRL_GEMV_K): gfx1151 MXFP4 gemvk variant per n; 0 = off (default, the gemvq_mr / gemvw
    // path), 1..9 = that variant, kGemvkAuto = chosen per call from ncols and rows (gemvkAuto). nullptr = off.
    const GemvR* gemv_k = nullptr;
};
inline constexpr std::uint8_t kGemvkAuto = 15;
// gemvk variant for one launch of K = ncols over total_rows rows: the K-wave count KW (8, 4, 2) with no idle
// superblock slot that still gives each wave >= 4 superblocks (else the one with the most), widened back towards 8
// while the launch would have < 80 blocks (2 per CU on the 8060S); the PF = 0 variant of that KW (v6 / v2 / v4).
std::uint8_t gemvkAuto(std::uint32_t ncols, std::uint32_t total_rows);
struct GemvPlan {
    Function f = nullptr;
    bool grouped = false;          // one launch over all matrices (GvArgs); else one launch per matrix
    std::uint32_t rows_per_block = 8;
    std::array<Dim3, 3> grid{};    // grouped: grid[0] only; else one per matrix
    Dim3 block{256, 1, 1};
    BitClass bits = BitClass::q8_row;
};
// mats: 1..3 matrices of one type and ncols; n = 1..16
GemvPlan chooseGemvTile(const MatShape* mats, std::size_t n_mats, std::uint32_t n, const kernels::MatmulKernels& k, const GemvTune& t);
// the most tokens of a grouped multi-token launch ("g" twins): gv_nmax_override, else 13 for mxfp4, 11 otherwise
std::uint32_t gvNmax(GgmlType ty, std::uint32_t gv_nmax_override);

// 17..32 token rows (wide batched verify): the two-tile WMMA GEMV, the output head's split q6_k kernel, or 16-row
// pieces through chooseGemvTile. head_gemvx: WHIRL_HEAD_GEMVX (the head on gemvx too).
enum class GemvWide : std::uint8_t { gemvx, head_split, pieces };
GemvWide chooseGemvWide(const MatShape& w0, std::size_t n_mats, const kernels::MatmulKernels& k, bool head_gemvx);
// smallest batch that takes gemmh_f16 instead of the tuned f16 GEMM (= tune.cpp's candidate rule)
inline constexpr std::uint32_t gemmh_min = 512;
// token tile of gemmh_f16 / gemmhh_f16 / gemmhq_<T> (kernels/shared/gemmh_impl.h BN): grid x = (n + gemmh_bn - 1) / gemmh_bn
inline constexpr std::uint32_t gemmh_bn = 256;

// grid of the wide kernels for one matrix (rows of a head chunk for head_split)
Dim3 gemvxGrid(std::uint32_t nrows);
Dim3 headSplitGrid(std::uint32_t chunk_rows);


// ---- f16 GEMM (MatmulPath::gemm_f16; K-2): precise decode / verify rows and prefill. Every choice computes the
// same bits (class f16_gemm, checkPrefillInvariance): the choice only moves speed.
struct GemmFlags {
    bool prec_dec = false;   // precise: decode / verify rows (n <= gv_max) take the decode-shaped choices
    bool out_h16 = false;    // the consumer reads f16 outputs (h16 GEMM twins)
    bool h16_dq = false;     // a fused quantized choice with f16 out takes its dequant twin (same bits)
    bool gemmh_on = true;    // 128 x 256 f16 GEMM (gemmh_f16 / gemmhh_f16) for n >= 512
    bool gemmhq_on = false;  // dequant fused into gemmh (gemmhq_<T>)
    bool prec_gemvh = true;  // WHIRL_PREC_GEMVH (0: precise rows skip gemvh)
    int prec_gs = -1;        // WHIRL_PREC_GS: forced gemms slot for precise rows (-1: chosen)
    // device::Profile gemvh_grid_by_rows: the gemvh grid is kernels::gemvhGridCap(rows, cap) with this device's cap
    // (kernels::gemvhCap(), read by the caller: the choice stays a pure function); 0: one block per 16 rows
    unsigned gemvh_cap = 0;
    bool gemms_tie_larger_bm = false; // device::Profile: precise gemms ties go to the larger row block
    // fused-dequant MXFP4 GEMM slot (gfx1151 gemmfq, v0251-a3port; bitwise == dequant_f16 + gemm_c<i>_f16): an MXFP4
    // dequant choice at n >= fdq_min rows runs gemmfq_<kGemmFqNames[fdq - 1]>_mxfp4 when the code object has it;
    // 0 = the old dequant + f16 GEMM. Context::gemmFlags: the device default (device::Profile::fdq_default) unless
    // WHIRL_FDQ8S / WHIRL_FDQ8S_MIN set it (config::fdqEnv at the load)
    int fdq = 0;
    std::uint32_t fdq_min = 1025;
    // v0251-a3fix: from fdq_big_min rows the slot is fdq_big instead (-1 = fdq, the WHIRL_FDQ8S A/B case; 0 = the old
    // plan there). Context::gemmFlags: device::Profile::fdq_big unless WHIRL_FDQ8S is set. gfx1151: t2 is 1.33-1.35x
    // the old pair at 4096 rows (the tune bucket 1 size and the 8060S MoE forward), c2 0.91-0.98; 2305 = one more than
    // the largest dense forward (prefill_exec_default 2048 + prefill_merge 256), whose rows keep t2
    int fdq_big = -1;
    std::uint32_t fdq_big_min = 2305;
    // v0252-8sbadopt: bit (slot - 1) set -> that gemmfq slot runs its grouped-raster twin (MatmulKernels::gemmfq_grp, same
    // tile, grid and bits) when the object has it. Context::gemmFlags: device::Profile::fq_grp_dense / fq_grp_moe
    std::uint8_t fq_grp = 0;
    std::uint32_t gv_max = 16;
    // v0252-pp2k2 (F1): the token-tail split of planGemmF16. Context::gemmFlags: device::Profile::gemm_tail unless
    // WHIRL_GEMM_TAIL sets it (config::gemmTailEnv at the load: "0" off, other values on)
    bool gemm_tail = false;
    // v0252-w4a4k2 (PLAN V1-144 K2): the W4A4 fake-quant arm's weight half. An MXFP4 matrix (whole superblocks) then always runs
    // as MatmulKernels::w4s_dq (per-32 int4 re-quantized grid) into the f16 weight scratch + the tuned slot's f16 GEMM, never a
    // kernel that decodes the original MXFP4 values, and without the token-tail split. Off unless Context::matmul sets it
    // (numerics flag w4a4_sim, fast, gfx1151 dense); no effect where the table has no w4s_dq.
    // round 2: 0 = off, g + 1 = weight grid g (MatmulKernels::w4s_dq[g]: 0 w, 1 w1, 2 w1e, 3 w1d, 4 wb)
    std::uint8_t w4s_w = 0;
};
enum class GemmKind : std::uint8_t { gemvh, gemms, gemmhq, gemmh, gemmc, gemmfq };
struct GemmPlan {
    GemmKind kind = GemmKind::gemmc;
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    bool dequant = false;          // dequant_f16 into the f16 weight scratch first (then f16 weights)
    Function dq = nullptr;
    Dim3 dq_grid{};
    BitClass bits = BitClass::f16_gemm;
    // v0252-pp2k2: the kernel writes f16 outputs (an h16 twin: gemmhh_f16 / gemm_ch / gemmhqh / gemmsh); set for the
    // gemmh / gemmhq / gemms plans (the kinds planGemmF16 splits or launches as a tail), false otherwise
    bool y_h16 = false;
};
// the tuned choice code for n rows: [0, G) fused gemm_c on the quantized bytes, [G, 2G) dequant + f16 gemm_c,
// [2G, 2G + S) gemms slot (G = kGemmCfgs.size(), S = kGemmsCfgs.size())
std::uint32_t precChoice(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k);
// w.row_bytes is read (an f16 matrix whose rows are padded is not gemmh's layout)
GemmPlan chooseGemmF16(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k);
// v0252-pp2k2 (PLAN V1-139, F1 GEMM tail split): the launches of an f16 GEMM of n token rows, in order. One piece = the
// chooseGemmF16 plan over all rows. Two pieces when f.gemm_tail is on, the plan is gemmh / gemmhq (256-token tile) and the
// r = n mod gemmh_bn rows past the last full tile fit a gemms token tile: rows [0, n - r) with that plan on (n - r) / 256
// tiles, then rows [n - r, n) with precChoice's gemms slot for r rows (the narrowest tile that holds r; WHIRL_PREC_GS's
// slot when it holds r) on the raw quantized weights, the gemmsh twin when the main kernel writes f16 (y_h16). No split
// when that slot or twin is missing, no gemms tile holds r, or r == 0. A piece's rows are rows [row0, row0 + rows) of x
// and y (row-major per token: x + row0 * ncols f16, y + row0 * nrows outputs). Bits: every plan is class f16_gemm and a
// token split changes no output element's k order, so the pieces give the one-launch bits (checkPrefillInvariance).
struct GemmPiece {
    std::uint32_t row0 = 0, rows = 0;
    GemmPlan plan;
};
struct GemmPieces {
    std::array<GemmPiece, 2> p{};
    std::size_t n = 0;
};
GemmPieces planGemmF16(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k);
// v0252-c3port (feat026-moedec L1): the f32-input twin of a gemvh plan p of weight type ty - gemvhf_<T> for gemvh_<T>,
// gemvhf2_<T> for gemvh2_<T> - when enabled (Flags::gemvh_f32, WHIRL_GEMVH_F32), the grid is one block per 16 rows
// (no gemvh_cap) and the code object has the kernel (gfx1201 only); else null = f32_to_f16 + p.f (the 0.2.5.1 path).
// Same bits: the B load rounds each f32 element with (_Float16)v, as f32_to_f16 does.
Function gemvhF32(const GemmPlan& p, GgmlType ty, bool enabled, const GemmFlags& f, const kernels::MatmulKernels& k);
// v0252-8sdec (PLAN V1-142; prior art progress/v0252-8sdec.PRIOR_ART.md): the precise grouped gemvh of 2..3 matrices that
// share one input (Context::matmulGroup). Grouped when enabled (Flags::gemvh_grp, WHIRL_GEMVH_GRP), every matrix has the
// first one's type and ncols, each would run alone as MatmulPath::gemm_f16 -> chooseGemmF16 kind gemvh with the table's
// gemvh_<T> (precise rows, n <= 16, no f16 outputs) and the code object has gemvhg_<T> (gfx1151 only). Tiles of 16 rows
// are numbered matrix by matrix: b1 / b2 = the first tile of matrix 1 / 2 (0x7fffffff = absent, the GvArgs rule), tiles =
// the total; grid = kernels::gemvhGridCap(tiles * 16, f.gemvh_cap) when the device caps it, else one block per tile.
// Same bits as one gemvh_<T> per matrix (the kernel runs gemvh_<T>'s per-tile body). f = null: not grouped.
struct GemvhGroupPlan {
    Function f = nullptr;
    Dim3 grid{}, block{32, 1, 1};
    std::int32_t b1 = 0x7fffffff, b2 = 0x7fffffff;
    std::uint32_t tiles = 0;
};
GemvhGroupPlan chooseGemvhGroup(const MatShape* ws, std::size_t n_mats, std::uint32_t n, bool enabled, const MatmulFlags& mf, const GemmFlags& f,
                                const kernels::MatmulKernels& k);

// ---- MXFP4 x fp8 GEMM (MatmulPath::gemm_fp8; K-2), class fp8_gemm
struct Gemm8Plan {
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    BitClass bits = BitClass::fp8_gemm;
};
// g8t: the fragment-tiled fp8 set (Model::useTiledFp8)
Gemm8Plan chooseGemm8(std::uint32_t nrows, std::uint32_t n, bool out_h16, bool g8t, const kernels::MatmulKernels& k);
std::size_t gemm8Cfg(std::uint32_t n);
std::size_t gemm8tCfg(std::uint32_t n);

}  // namespace whirl::ops
