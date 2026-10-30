// whirl-iso-cache-prefix-tests: the module cache.prefix (the prefix cache (src/cache/prefix_cache.cpp: slot / shared / tier-entry matching, eviction, supersede rules)) built with its declared dependencies only and run (ARCH-1 8.5 isolation,
// arch025-n8b1). The cases are tests/cache/prefix_cache_cases.inc, the file whirl-cache-tests (tests/cache/cache_tests.cpp) includes.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../cache/prefix_cache_cases.inc"

int main(int argc, char** argv) {
    return iso::run("cache.prefix",
                    {{"helpers", testHelpers},
                     {"match_slot", testMatchSlot},
                     {"ckpt_slot_for", testCkptSlotFor},
                     {"shared", testShared},
                     {"evict", testEvict},
                     {"tier", testTier},
                     {"lcp_target", testLcpTarget},
                     {"supersede", testSupersede}},
                    argc, argv);
}
