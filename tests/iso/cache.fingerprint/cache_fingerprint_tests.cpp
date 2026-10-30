// whirl-iso-cache-fingerprint-tests: the module cache.fingerprint (the KV-tier fingerprint (src/cache/fingerprint.cpp)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/cache/fingerprint_cases.inc, the file whirl-cache-tests (tests/cache/cache_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../test_env.h"
#include "../../cache/fingerprint_cases.inc"

int main(int argc, char** argv) {
    whirl::test::clearWhirlEnv();  // as whirl-cache-tests: no WHIRL_* of the caller reaches the fingerprint
    return iso::run("cache.fingerprint",
                    {{"covers_bits_key", testFingerprintCoversBitsKey},
                     {"fingerprint", testFingerprint},
                     {"dump_env", testFingerprintDumpEnv},
                     {"platform_env", testFingerprintPlatformEnv},
                     {"vis_copy_delay_env", testFingerprintVisCopyDelayEnv},
                     {"server_inputs", testServerFingerprintInputs}},
                    argc, argv);
}
