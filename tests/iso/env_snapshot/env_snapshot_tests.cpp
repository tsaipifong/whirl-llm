// whirl-iso-env-snapshot-tests: src/config/env_snapshot.cpp built alone with the platform environment layer
// (ARCH-1 8.5 isolation, arch025-lane2 D2.2). The full table test (every name against the model's readers) stays in
// whirl-model-tests (testEnvSnapshot*, tests/model/env_table.expected.txt); this program checks the module's own rules.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "config/env_get.h"
#include "config/env_snapshot.h"
#include "platform/env.h"

#include <set>
#include <stdexcept>
#include <string>

namespace {

using whirl::config::EnvSnapshot;

void testEnvGetFlag() {
    whirl::platform::env::set("WHIRL_ISO_ENVSNAP_T", "");
    ISO_CHECK(!whirl::envGet("ISO_ENVSNAP_T"));  // _putenv_s with "" removes the variable
    ISO_CHECK(!whirl::envFlag("ISO_ENVSNAP_T"));
    ISO_CHECK(whirl::envFlag("ISO_ENVSNAP_T", true));
    whirl::platform::env::set("WHIRL_ISO_ENVSNAP_T", "0");
    ISO_CHECK(whirl::envGet("ISO_ENVSNAP_T") == std::string("0"));
    ISO_CHECK(!whirl::envFlag("ISO_ENVSNAP_T", true));
    whirl::platform::env::set("WHIRL_ISO_ENVSNAP_T", "auto");
    ISO_CHECK(whirl::envFlag("ISO_ENVSNAP_T"));
    whirl::platform::env::set("WHIRL_ISO_ENVSNAP_T", "");
}

void testNamesUnique() {
    const auto& n = EnvSnapshot::names();
    std::set<std::string> names;
    for (const auto& e : n) names.insert(e.name);
    ISO_CHECK(names.size() == EnvSnapshot::n_names);
    std::size_t same_field = 0;  // member pointers have no order: pairwise
    for (std::size_t i = 0; i < n.size(); ++i)
        for (std::size_t j = i + 1; j < n.size(); ++j) same_field += n[i].field == n[j].field ? 1 : 0;
    ISO_CHECK(same_field == 0);
}

void testFromValuesAndGet() {
    const EnvSnapshot s = EnvSnapshot::fromValues({{"FP8", std::string("1")}, {"KV", std::string("q8h")}});
    ISO_CHECK(s.fp8 == std::string("1"));
    ISO_CHECK(s.get("KV") == std::string("q8h"));
    ISO_CHECK(!s.get("GDN_WMMA"));
    bool threw = false;
    try {
        (void)EnvSnapshot::fromValues({{"NOT_A_NAME", std::string("1")}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ISO_CHECK(threw);
    threw = false;
    try {
        (void)s.get("NOT_A_NAME");
    } catch (const std::logic_error&) {
        threw = true;
    }
    ISO_CHECK(threw);
}

void testCaptureReadsEnv() {
    whirl::platform::env::set("WHIRL_FP8", "0");
    const EnvSnapshot s = EnvSnapshot::capture();
    ISO_CHECK(s.fp8 == std::string("0"));
    whirl::platform::env::set("WHIRL_FP8", "");
    ISO_CHECK(!EnvSnapshot::capture().fp8);
}

void testFlagAndForwardEnv() {
    ISO_CHECK(!EnvSnapshot::flag(std::nullopt));
    ISO_CHECK(EnvSnapshot::flag(std::nullopt, true));
    ISO_CHECK(EnvSnapshot::flag(std::string("")));
    ISO_CHECK(!EnvSnapshot::flag(std::string("0"), true));
    const auto f = whirl::config::forwardEnv(EnvSnapshot::fromValues({{"PREC_GEMVH", std::string("0x")}, {"PREC_GS", std::string("abc")}}));
    ISO_CHECK(!f.head_gemvx && !f.prec_gemvh && f.prec_gs == 0);
    const auto d = whirl::config::forwardEnv(EnvSnapshot::fromValues({}));
    ISO_CHECK(!d.head_gemvx && d.prec_gemvh && d.prec_gs == -1);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("env_snapshot",
                    {{"env_get_flag", testEnvGetFlag},
                     {"names_unique", testNamesUnique},
                     {"from_values_get", testFromValuesAndGet},
                     {"capture_reads_env", testCaptureReadsEnv},
                     {"flag_forward_env", testFlagAndForwardEnv}},
                    argc, argv);
}
