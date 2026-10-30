// whirl-iso-ops-attention-tests: the module ops.attention (the attention decode / prefill / prep choice (src/ops/attention.cpp)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/ops/ops_attention_cases.inc, the file whirl-ops-tests (tests/ops/ops_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../ops/ops_fakes.h"
#include "../../ops/ops_attention_cases.inc"

int main(int argc, char** argv) {
    return iso::run("ops.attention",
                    {{"attn_decode", testAttnDecode},
                     {"attn_prefill", testAttnPrefill},
                     {"attn_prep", testAttnPrepRules}},
                    argc, argv);
}
