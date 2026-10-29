// whirl-iso-spec-rules-tests: the header-only module spec.rules (src/spec/spec_rules.h, the speculative-decoding rules shared by the CLI and the
// server) built with its declared dependencies (spec.policy, slot_drafts, the relaxed-acceptance policy, model_types.h) and run (ARCH-1 8.5
// isolation, arch025-iso4). The cases are tests/server/spec_rules_tests.inc, the file whirl-server-tests includes, unchanged except that the one
// server-side half of D4 (a real Engine behind a TestServer) is compiled out here by WHIRL_ISO_NO_SERVER: it keeps running in whirl-server-tests.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "../../server/spec_rules_tests.inc"

int main(int argc, char** argv) {
    return iso::run("spec.rules",
                    {{"d1_server", r12b::testRulesD1Server},
                     {"d1_cli", r12b::testRulesD1Cli},
                     {"d2_server", r12b::testRulesD2Server},
                     {"d2_cli", r12b::testRulesD2Cli},
                     {"d3", r12b::testRulesD3},
                     {"d4", r12b::testRulesD4},
                     {"d6", r12b::testRulesD6},
                     {"d8", r12b::testRulesD8},
                     {"d14", r12b::testRulesD14},
                     {"d16", r12b::testRulesD16},
                     {"d17", r12b::testRulesD17}},
                    argc, argv);
}
