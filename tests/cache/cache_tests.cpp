// Unit tests of the prefix cache module (src/cache) against a mock tier.
// SPDX-License-Identifier: Apache-2.0
// arch025-n8b1: the cases are in prefix_cache_cases.inc (cache.prefix) and fingerprint_cases.inc (cache.fingerprint), each
// also built alone by tests/iso/<module>/; this file keeps the program, its includes and the order of every case.
#include "../test_env.h"
#include "../test_check.h"
#include "cache/fingerprint.h"
#include "cache/prefix_cache.h"
#include "mock_tier.h"
#include "tier/kv_tier.h"
#include "whirl/numerics.h"
#include "whirl/common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "prefix_cache_cases.inc"
#include "fingerprint_cases.inc"

using namespace whirl::cache;
using whirl::cache::test::MockTier;

using whirl::test::g_fail;
using whirl::test::g_pass;
// CHECK / CHECK_EQ / the counters: tests/test_check.h (arch025 N16)


int main() {
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
    testFingerprintCoversBitsKey();
    testHelpers();
    testMatchSlot();
    testCkptSlotFor();
    testShared();
    testEvict();
    testTier();
    testLcpTarget();
    testFingerprint();
    testFingerprintDumpEnv();
    testFingerprintPlatformEnv();  // arch025 PL1
    testFingerprintVisCopyDelayEnv();  // arch025-i6vis (I6, X4 K4)
#ifndef WHIRL_ISO_NO_SOURCE_TREE  // arch025-iso4: the isolated program (tests/iso/cache) has no source tree to scan; the case runs in whirl-cache-tests
    testFingerprintEnvClassified();    // arch025-i6vis: every WHIRL_* read is classified
#endif
    try {
        testServerFingerprintInputs();
    } catch (const std::exception& e) {  // e.g. a field the assembly no longer fills: report, do not abort
        std::printf("testServerFingerprintInputs threw: %s\n", e.what());
        CHECK(false);
    }
    testSupersede();
    std::printf("cache tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
