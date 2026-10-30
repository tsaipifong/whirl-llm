// whirl-iso-cache-tier-adapter-tests: the module cache.tier_adapter (src/cache/tier_adapter.cpp: cache::Tier over tier::Tier) built
// with its declared dependencies only and run (ARCH-1 8.5 isolation, arch025-n8b2). The cases are tests/cache/tier_adapter_cases.inc.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "../../cache/tier_adapter_cases.inc"

int main(int argc, char** argv) {
    return iso::run("cache.tier_adapter",
                    {{"entries", tier_adapter_cases::testTierAdapterEntries},
                     {"fields", tier_adapter_cases::testTierAdapterFields},
                     {"min_tokens", tier_adapter_cases::testTierAdapterMinTokens}},
                    argc, argv);
}
