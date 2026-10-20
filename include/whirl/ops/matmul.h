// Matmul kernel choice (ARCH-1 K-1 GEMV): which kernel a matmul of n token rows launches, with which grid, as a
// pure function of the problem (weight type, shape, rows), the numerics flags, the code object's kernel table and
// the tuned tables. src/model/forward.cpp builds the problem, asks here, and launches what comes back.
// SPDX-License-Identifier: Apache-2.0
//
// Every kernel carries a bit-equivalence class (BitClass): kernels of one class give the same bits for the same
// token row, so the choice inside a class is free (tuning, geometry, grouping), and a choice across classes is a
// numerics decision. Precise picks only the precise class (the f16 GEMM family, bitwise == prefill); the rule is
// written once, in chooseMatmulPath. Tests: tests/ops/ops_tests.cpp (CPU, fake kernel tables).

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

enum class MatmulPath : std::uint8_t { gemvq, gemv1, float_gemv, gemm_fp8, gemm_f16 };
BitClass bitClassOf(MatmulPath p);

// one matrix of a matmul
struct MatShape {
    GgmlType ty = GgmlType::f32;
    std::uint32_t nrows = 0, ncols = 0;
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
MatmulPath chooseMatmulPath(const MatShape& w, std::uint32_t n, const MatmulFlags& f, const kernels::KernelTable& k);
// f32 / f16 weights at n = 2..gv_max without an int8 GEMV: the f32-activation GEMV instances
bool floatGemvN(const MatShape& w, std::uint32_t n, std::uint32_t gv_max, const kernels::KernelTable& k);

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
GemvPlan chooseGemvTile(const MatShape* mats, std::size_t n_mats, std::uint32_t n, const kernels::KernelTable& k, const GemvTune& t);
// the most tokens of a grouped multi-token launch ("g" twins): gv_nmax_override, else 13 for mxfp4, 11 otherwise
std::uint32_t gvNmax(GgmlType ty, std::uint32_t gv_nmax_override);

// 17..32 token rows (wide batched verify): the two-tile WMMA GEMV, the output head's split q6_k kernel, or 16-row
// pieces through chooseGemvTile. head_gemvx: WHIRL_HEAD_GEMVX (the head on gemvx too).
enum class GemvWide : std::uint8_t { gemvx, head_split, pieces };
GemvWide chooseGemvWide(const MatShape& w0, std::size_t n_mats, const kernels::KernelTable& k, bool head_gemvx);
// grid of the wide kernels for one matrix (rows of a head chunk for head_split)
Dim3 gemvxGrid(std::uint32_t nrows);
Dim3 headSplitGrid(std::uint32_t chunk_rows);

}  // namespace whirl::ops
