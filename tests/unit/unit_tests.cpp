// whirl-tests: every module's host-side unit tests in one program (no GPU, no model files needed).
// arch025 N7: the tests live in one file per module, tests/unit/<module>_tests.cpp, each also built alone as
// whirl-unit-<module>-tests (CMakeLists.txt whirl_unit_test; release_gate cpu_unit_<module>). This program runs them all
// in the order of the old single main() and fails setup (exit 2) when a module's case is missing from ORDER or ORDER
// names a case no module has, so a test cannot drop out of whirl-tests unnoticed.
//   whirl-tests [--list] [--only NAME[,NAME...]]      last line "unit tests: N passed, M failed"
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

const whirl::test::Suite& unitSuite_release();
const whirl::test::Suite& unitSuite_accept_policy();
const whirl::test::Suite& unitSuite_json();
const whirl::test::Suite& unitSuite_tokenizer();
const whirl::test::Suite& unitSuite_gguf();
const whirl::test::Suite& unitSuite_chat();
const whirl::test::Suite& unitSuite_mem_fit();
const whirl::test::Suite& unitSuite_contract();
const whirl::test::Suite& unitSuite_kernels_gemm();
const whirl::test::Suite& unitSuite_model_core();
const whirl::test::Suite& unitSuite_hip_runtime();
const whirl::test::Suite& unitSuite_platform_paths();
const whirl::test::Suite& unitSuite_cli();
const whirl::test::Suite& unitSuite_platform_env();
const whirl::test::Suite& unitSuite_platform_console();
const whirl::test::Suite& unitSuite_platform_proc();
const whirl::test::Suite& unitSuite_hip_leak();
const whirl::test::Suite& unitSuite_spec_sample();
const whirl::test::Suite& unitSuite_hip_mem();
const whirl::test::Suite& unitSuite_mem_wddm_watch();
const whirl::test::Suite& unitSuite_mem_governor();
const whirl::test::Suite& unitSuite_gpu_select();
const whirl::test::Suite& unitSuite_platform_fs();

namespace {

struct OrderRow {
    const char* module;
    const char* name;
};

// the old main()'s calls, in order, with its #if conditions
const OrderRow kOrder[] = {
#if WHIRL_TEST_DLL_DEPS
    {"release", "testDllDeps"},
#endif
    {"accept_policy", "testRelaxAccept"},
    {"json", "testJson"},
    {"tokenizer", "testUnicode"},
    {"gguf", "testGguf"},
    {"gguf", "testGgufHardening"},
    {"tokenizer", "testTokenizerByteTokens"},
    {"chat", "testChat"},
    {"chat", "testJsonFields"},
    {"mem.fit", "testVramLimit"},
    {"mem.fit", "testCardDefaults"},
    {"contract", "testNumerics"},
    {"contract", "testNumericsPlan"},
    {"contract", "testKvChoice"},
    {"contract", "testKvEstimateCaps"},
    {"kernels.gemm", "testKernelFetchBounds"},
    {"model.core", "testValueScope"},
    {"hip.runtime", "testLaunchDigest"},
    {"hip.runtime", "testLaunchDigestLayout"},
    {"hip.runtime", "testLaunchDigestZeroAndAdjacent"},
    {"hip.runtime", "testLaunchRawCap"},
    {"hip.runtime", "testAllocLog"},
    {"hip.runtime", "testLogNamesUnique"},
    {"platform.paths", "testPlatformPaths"},
    {"cli", "testDFlashDumpDir"},
#if defined(_WIN32)
    {"platform.paths", "testLocalAppDataChild"},
#endif
#if defined(_WIN32)
    {"hip.runtime", "testEnvPathsChild"},
#endif
    {"platform.env", "testPlatformEnv"},
    {"hip.runtime", "testPl1bEnvReads"},
    {"hip.runtime", "testLogFolderEmpty"},
    {"platform.console", "testPlatformConsoleTime"},
    {"platform.proc", "testPlatformProc"},
#if defined(_WIN32)
    {"platform.proc", "testPlatformProcSurvivors"},
#endif
    {"hip.leak", "testLeakCount"},
    {"spec.sample", "testSpecSample"},
    {"model.core", "testDflashPolicy"},
    {"hip.mem", "testLocalMemTaken"},
    {"hip.mem", "testUmaCheck"},
    {"mem.wddm_watch", "testWddmWatch"},
    {"mem.fit", "testPoolBudget"},
    {"mem.fit", "testPoolMargin"},
    {"mem.fit", "testStartupFit"},
    {"mem.fit", "testB0SizingFollowUps"},
    {"mem.fit", "testVisionPoolReserve"},
    {"mem.fit", "testKvRoom"},
    {"mem.fit", "testSlotFixed"},
    {"mem.fit", "testPreloadAdjust"},
    {"mem.fit", "testUmaFixed"},
    {"mem.governor", "testMemGovernor"},
    {"mem.governor", "testMemGovernorGrowOvershoot"},
    {"gpu_select", "testGpuSelect"},
    {"platform.fs", "testFsMappedOpen"},
#if WHIRL_WITH_HIP && defined(_WIN32)
    {"gpu_select", "testProbeChild"},
#endif
};

}  // namespace

int main(int argc, char** argv) {
    const std::vector<const whirl::test::Suite*> suites = {
        &unitSuite_release(),
        &unitSuite_accept_policy(),
        &unitSuite_json(),
        &unitSuite_tokenizer(),
        &unitSuite_gguf(),
        &unitSuite_chat(),
        &unitSuite_mem_fit(),
        &unitSuite_contract(),
        &unitSuite_kernels_gemm(),
        &unitSuite_model_core(),
        &unitSuite_hip_runtime(),
        &unitSuite_platform_paths(),
        &unitSuite_cli(),
        &unitSuite_platform_env(),
        &unitSuite_platform_console(),
        &unitSuite_platform_proc(),
        &unitSuite_hip_leak(),
        &unitSuite_spec_sample(),
        &unitSuite_hip_mem(),
        &unitSuite_mem_wddm_watch(),
        &unitSuite_mem_governor(),
        &unitSuite_gpu_select(),
        &unitSuite_platform_fs(),
    };
    const int child_rc = whirl::test::runChildModes(suites, argc, argv);  // helper modes (the tests run this program as a child)
    if (child_rc >= 0) return child_rc;
    bool list = false;
    std::set<std::string> only;
    if (!whirl::test::parseArgs(argc, argv, &list, &only)) return 2;
    // every case of every suite exactly once in kOrder, and nothing else
    std::map<std::pair<std::string, std::string>, const whirl::test::Case*> cases;
    for (const whirl::test::Suite* s : suites)
        for (const whirl::test::Case& c : s->cases) cases[{s->module, c.name}] = &c;
    std::set<std::pair<std::string, std::string>> seen;
    int bad = 0;
    for (const OrderRow& r : kOrder) {
        const std::pair<std::string, std::string> k{r.module, r.name};
        if (!cases.count(k) || !seen.insert(k).second) {
            std::printf("unit tests: ORDER row %s %s has no case (or is listed twice)\n", r.module, r.name);
            ++bad;
        }
    }
    for (const auto& [k, c] : cases)
        if (!seen.count(k)) {
            std::printf("unit tests: case %s %s is not in ORDER\n", k.first.c_str(), k.second.c_str());
            ++bad;
        }
    if (bad) return 2;
    for (const std::string& n : only) {
        bool found = false;
        for (const OrderRow& r : kOrder) found = found || n == r.name;
        if (!found) {
            std::fprintf(stderr, "unknown test '%s' (--list)\n", n.c_str());
            return 2;
        }
    }
    if (list) {
        for (const OrderRow& r : kOrder) std::printf("%s %s\n", r.module, r.name);
        return 0;
    }
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
    std::map<std::string, std::pair<int, int>> per;  // module -> passed, failed
    for (const OrderRow& r : kOrder) {
        if (!only.empty() && !only.count(r.name)) continue;
        const int p0 = whirl::test::g_pass, f0 = whirl::test::g_fail;
        whirl::test::runCase(*cases[{r.module, r.name}]);
        per[r.module].first += whirl::test::g_pass - p0;
        per[r.module].second += whirl::test::g_fail - f0;
    }
    for (const whirl::test::Suite* s : suites)
        if (per.count(s->module)) std::printf("%s tests: %d passed, %d failed\n", s->module, per[s->module].first, per[s->module].second);
    std::printf("unit tests: %d passed, %d failed\n", whirl::test::g_pass, whirl::test::g_fail);
    return whirl::test::g_fail ? 1 : 0;
}
