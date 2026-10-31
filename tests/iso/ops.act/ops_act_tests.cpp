// whirl-iso-ops-act-tests: the module ops.act (the activation-input / fused-path choice (src/ops/act.cpp)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/ops/ops_act_cases.inc, the file whirl-ops-tests (tests/ops/ops_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../ops/ops_fakes.h"
#include "../../ops/ops_mat.h"
#include "../../ops/ops_act_cases.inc"

int main(int argc, char** argv) {
    return iso::run("ops.act",
                    {{"act_choice", testActChoice}},
                    argc, argv);
}
