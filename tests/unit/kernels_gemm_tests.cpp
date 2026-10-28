// WHIRL unit tests of module kernels.gemm (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-kernels-gemm-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "../../kernels/gemm_small_addr.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// ---------------------------------------------------------------------------
// Kernel fetch bounds (FIX-OVR): replay the W-stage addressing of the raw-stage GEMMs
// (gfx1201 gemms_impl / gemmsd_impl / gemvh_impl, gfx1151 gemms11_impl) and assert that no
// 16-byte load starts at or past the end of the matrix and none ends past the end rounded
// up to 16 bytes (a 16-byte aligned block holding a matrix byte never crosses a page).
// The pre-fix addressing (every chunk loaded) must be caught by the same check.
struct FetchBound {
    std::uint64_t max_start = 0, max_end = 0;
};
// s: super-block stages per round (gemvh_impl S = 2 when ncols % 512 == 0: SB = 2 * sbb, nst = ncols / 512)
FetchBound replayStageFetch(int sb, std::uint64_t row_bytes, int nrows, int ncols, bool guarded, int s = 1) {
    const int ch = gemms_stage_chunks(sb), nst = ncols / (256 * s);
    FetchBound b;
    // rows past nrows are clamped to nrows - 1 by every kernel; the pattern only depends on
    // the row start, so the first rows, the last two and the clamped row cover every case
    std::vector<int> rows = {0, 1, nrows - 2, nrows - 1, nrows + 5};
    for (int r0 : rows) {
        if (r0 < 0) continue;
        const int row = std::min(r0, nrows - 1);
        for (int js = 0; js < nst; ++js) {
            const std::uint64_t st = static_cast<std::uint64_t>(row) * row_bytes + static_cast<std::uint64_t>(js) * sb;
            const std::uint32_t st32 = static_cast<std::uint32_t>(st);  // gemmsd_impl's 32-bit offsets
            for (int c = 0; c < ch; ++c) {
                const bool need = gemms_chunk_needed(st, c, sb);
                if (gemms_chunk_needed(st32, c, sb) != need) b.max_end = ~0ull;  // the 32-bit form must agree
                if (guarded && !need) continue;
                const std::uint64_t a = gemms_chunk_addr(st, c);
                b.max_start = std::max(b.max_start, a);
                b.max_end = std::max(b.max_end, a + 16);
            }
        }
    }
    return b;
}

void testKernelFetchBounds() {
    struct T {
        const char* name;
        int blk, bytes;
    };
    const T types[] = {{"f16", 8, 16},     {"q8_0", 32, 34},    {"iq4_nl", 32, 18}, {"q3_k", 256, 110}, {"q4_k", 256, 144},
                       {"q5_k", 256, 176}, {"q6_k", 256, 210},  {"iq3_s", 256, 110}, {"iq4_xs", 256, 136}, {"mxfp4", 256, 136}};
    int old_caught = 0, s2_cases = 0, s2_old_caught = 0;
    for (const T& t : types)
        for (int st : {1, 2}) {  // FIX-K97 (FR-K97 P3-1): also gemvh_impl's S = 2 stage (2 super-blocks per LDS round)
            const int sb = st * (256 / t.blk * t.bytes);
            CHECK(gemms_stage_chunks(sb) * 16 >= sb + 15);  // CH covers the worst misalignment
            for (int ncols : {256, 512, 1536, 2048, 4096, 5120, 8192, 12288, 17408})
                for (int nrows : {1, 17, 37, 40, 47, 48, 128, 2048, 4097}) {
                    if (ncols % (256 * st) != 0) continue;  // S = 2 only when ncols % 512 == 0 (gemvh_<T> / gemvh2_<T>)
                    const std::uint64_t rb = static_cast<std::uint64_t>(ncols) / t.blk * t.bytes;
                    const std::uint64_t total = rb * nrows, total16 = (total + 15) & ~15ull;
                    const FetchBound b = replayStageFetch(sb, rb, nrows, ncols, true, st);
                    const bool ok = b.max_start < total && b.max_end <= total16;
                    if (!ok)
                        std::printf("FAIL fetch bound %s S=%d %dx%d: max start %llu end %llu, matrix %llu bytes\n", t.name, st, nrows, ncols,
                                    static_cast<unsigned long long>(b.max_start), static_cast<unsigned long long>(b.max_end),
                                    static_cast<unsigned long long>(total));
                    CHECK(ok);
                    // also within the allocation tail pad for any matrix size
                    CHECK(b.max_end <= total + 256);
                    const FetchBound o = replayStageFetch(sb, rb, nrows, ncols, false, st);
                    if (o.max_end > total16) ++(st == 1 ? old_caught : s2_old_caught);
                    s2_cases += st == 2;
                }
        }
    CHECK(old_caught > 0);     // the check does see the pre-fix over-read (every type reads >= 16 B past)
    CHECK(s2_cases > 0);       // the S = 2 addressing was replayed
    CHECK(s2_old_caught > 0);  // and the check bites there too
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_kernels_gemm() {
    static const whirl::test::Suite s{"kernels.gemm",
                                      {
                                          {"testKernelFetchBounds", [] { testKernelFetchBounds(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_gemm()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
