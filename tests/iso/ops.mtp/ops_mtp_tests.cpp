// whirl-iso-ops-mtp-tests: the module ops.mtp (the MTP draft-head choice (src/ops/mtp.cpp, the constexpr gates of whirl/ops/mtp.h)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/ops/ops_mtp_cases.inc, the file whirl-ops-tests (tests/ops/ops_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../ops/ops_mtp_cases.inc"

int main(int argc, char** argv) {
    return iso::run("ops.mtp",
                    {{"mtp_rules", testMtpRules}},
                    argc, argv);
}
