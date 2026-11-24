// whirl-iso-ops-attention-tests: the module ops.attention (the attention decode / prefill / prep choice (src/ops/attention.cpp)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/ops/ops_attention_cases.inc, the file whirl-ops-tests (tests/ops/ops_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../ops/ops_fakes.h"
#include "../../ops/ops_attention_cases.inc"

int main(int argc, char** argv) {
    return iso::run("ops.attention",
                    {{"attn_decode", testAttnDecode},
                     {"attn_decode_ws4r", testAttnDecodeWs4r},  // v0252-ws4r97b
                     {"attn_decode_lsplit", testAttnDecodeLsplit},  // v0252-l2split
                     {"attn_decode_ws4split", testAttnDecodeWs4Split},  // v0252-8sws
                     {"dq4_partition", testDq4Partition},  // v0252-n2
                     {"attn_dq4_f8", testAttnDq4F8},  // v0252-8sq4
                     {"attn_dq4_table", testAttnDq4Table},  // v0252-8sq5
                     {"attn_prefill", testAttnPrefill},
                     {"attn_kgvar", testAttnKgVariant},  // v0252-c1port
                     {"attn_sx", testAttnSx},  // v0252-strixc2
                     {"attn_i4qk", testAttnI4qk},  // v0252-8sfp
                     {"attn_i4pf", testAttnI4pf},  // v0252-8sattn3
                     {"attn_prep", testAttnPrepRules}},
                    argc, argv);
}
