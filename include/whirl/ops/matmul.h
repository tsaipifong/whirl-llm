// Matmul kernel choice (ARCH-1 K-1 GEMV): which kernel a matmul of n token rows launches, with which grid, as a
// pure function of the problem (weight type, shape, rows), the numerics flags, the code object's kernel table and
// the tuned tables. src/model/forward.cpp builds the problem, asks here, and launches what comes back.
// SPDX-License-Identifier: Apache-2.0
//
// Every kernel carries a bit-equivalence class (BitClass): kernels of one class give the same bits for the same
// token row, so the choice inside a class is free (tuning, geometry, grouping), and a choice across classes is a
// numerics decision. Precise picks only the precise class (the f16 GEMM family, bitwise == prefill) for decode /
// verify rows; that rule is written once, in preciseRows / preciseAllows below, and every row-count gate asks it
// (chooseMatmulPath, chooseGemmF16 / precChoice, and in src/model/forward.cpp the grouped int8 GEMV, actIn, the
// fused silu_mul_q8 and the fused GDN [beta; alpha] step). Mode-level precise gates that do not depend on the row
// count stay with their op: the attention combine (attention.cpp), the MoE decode experts (moe.cpp, which extends
// precise rows to max_verify_rows) and wideCapable. Tests: tests/ops/ops_tests.cpp (CPU, fake kernel tables).

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
};
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
    std::uint32_t gv_max = 16;
};
enum class GemmKind : std::uint8_t { gemvh, gemms, gemmhq, gemmh, gemmc };
struct GemmPlan {
    GemmKind kind = GemmKind::gemmc;
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    bool dequant = false;          // dequant_f16 into the f16 weight scratch first (then f16 weights)
    Function dq = nullptr;
    Dim3 dq_grid{};
    BitClass bits = BitClass::f16_gemm;
};
// the tuned choice code for n rows: [0, G) fused gemm_c on the quantized bytes, [G, 2G) dequant + f16 gemm_c,
// [2G, 2G + S) gemms slot (G = kGemmCfgs.size(), S = kGemmsCfgs.size())
std::uint32_t precChoice(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k);
// w.row_bytes is read (an f16 matrix whose rows are padded is not gemmh's layout)
GemmPlan chooseGemmF16(const MatShape& w, std::uint32_t n, std::uint8_t tuned, const GemmFlags& f, const kernels::MatmulKernels& k);

// ---- MXFP4 x fp8 GEMM (MatmulPath::gemm_fp8; K-2), class fp8_gemm
struct Gemm8Plan {
    Function f = nullptr;
    Dim3 grid{}, block{256, 1, 1};
    BitClass bits = BitClass::fp8_gemm;
};
// g8t: the fragment-tiled fp8 set (Model::useTiledFp8). g8t_v > 0 (WHIRL_GEMM8T_V): where gemm8t_c0 would run, run
// its schedule variant gemm8tp_v<g8t_v> instead (same geometry, bitwise the same output) when gemm8tpOk.
Gemm8Plan chooseGemm8(std::uint32_t nrows, std::uint32_t n, bool out_h16, bool g8t, const kernels::MatmulKernels& k,
                      int g8t_v = 0, std::uint32_t ncols = 0);
// whether variant v exists (both outputs) and takes K = ncols (v4: an even number >= 2 of 64-k slabs)
bool gemm8tpOk(int v, std::uint32_t ncols, const kernels::MatmulKernels& k);
std::size_t gemm8Cfg(std::uint32_t n);
std::size_t gemm8tCfg(std::uint32_t n);

}  // namespace whirl::ops
