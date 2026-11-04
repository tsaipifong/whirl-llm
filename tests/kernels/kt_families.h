// whirl-kernel-test family table: the ONE list of test families (v0251-harness, PLAN B26-33 item 1).
// SPDX-License-Identifier: Apache-2.0
//
// Until 2026-11-04 kernel_test_main.cpp kept the --only validation list (fam_names[]) apart from the run table (fams[]);
// the 0.2.5.1 port of feat026-router merged only the fams[] row, so `--only router` exited 2 (usage) and the A11 screening
// round ran no kernel check (progress\v0251-judge5.md §2 / §6.1). Every family is now one X(...) row here, and the
// validation, the run table, the on-request skip and the usage text are all expanded from it (the X-macro pattern; ggml.c
// keeps GGML_OP_NAME tied to GGML_OP_COUNT the same way). CPU test: tests/kernels/unit/kt_bench_tests.cpp test_families.
//
// X(name, runner, on_request): on_request = timing only, runs only when --only names it (never in the default sweep).
// The runner column is only expanded by kernel_test_main.cpp; expansions that drop it need no kt:: declarations.
#pragma once

#include <cstddef>
#include <string_view>

#define WHIRL_KT_FAMILIES(X)                   \
    X("quant", kt::testQuant, false)           \
    X("gemv", kt::testGemv, false)             \
    X("gemm", kt::testGemm, false)             \
    X("attn", kt::testAttn, false)             \
    X("attndq4", kt::testAttnDq4, false)       \
    X("gdn", kt::testGdn, false)               \
    X("moe", kt::testMoe, false)               \
    X("moemx", kt::testMoeMx, false)           \
    X("moef", kt::testMoeF, false)             \
    X("router", kt::testMoeRouter, false)      \
    X("misc", kt::testMisc, false)             \
    X("attnbench", kt::benchAttn, true)        \
    X("gemvh", kt::testGemvh, false)           \
    X("gemvhbench", kt::benchGemvh, true)      \
    X("dflash", kt::testDflash, false)         \
    X("ws4", kt::testAttnWs4, false)           \
    X("ws4bench", kt::benchAttnWs4, true)      \
    X("ws4g", kt::testAttnWs4g, false)         \
    X("ws4gbench", kt::benchAttnWs4g, true)    \
    X("vision", kt::testVision, false)

namespace kt {

struct FamilyName {
    const char* name;
    bool on_request;
};

inline constexpr FamilyName kFamilyNames[] = {
#define WHIRL_KT_NAME_ROW(n, f, req) {n, req},
    WHIRL_KT_FAMILIES(WHIRL_KT_NAME_ROW)
#undef WHIRL_KT_NAME_ROW
};
inline constexpr std::size_t kFamilyCount = sizeof(kFamilyNames) / sizeof(kFamilyNames[0]);

// --only validation: a name the run table has
inline bool familyKnown(std::string_view f) {
    for (const FamilyName& n : kFamilyNames)
        if (f == n.name) return true;
    return false;
}

// skipped by the default sweep (no --only): timing-only families
inline bool familyOnRequest(std::string_view f) {
    for (const FamilyName& n : kFamilyNames)
        if (f == n.name) return n.on_request;
    return false;
}

}  // namespace kt
