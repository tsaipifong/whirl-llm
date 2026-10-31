// whirl-iso-ops-matmul-tests: the module ops.matmul (the matmul / GEMV / GEMM path choice (src/ops/matmul.cpp)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/ops/ops_matmul_cases.inc, the file whirl-ops-tests (tests/ops/ops_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../ops/ops_fakes.h"
#include "../../ops/ops_mat.h"
#include "../../ops/ops_matmul_cases.inc"

int main(int argc, char** argv) {
    return iso::run("ops.matmul",
                    {{"bit_classes", testBitClasses},
                     {"precise_rule", testPreciseRule},
                     {"matmul_path", testMatmulPath},
                     {"gemv_tile", testGemvTile},
                     {"gemv_wide", testGemvWide},
                     {"gemm", testOpsGemm}},
                    argc, argv);
}
