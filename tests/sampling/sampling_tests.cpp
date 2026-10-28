// whirl-sampling-tests: the src/sampling unit tests built against the sampling module ALONE (ARCH-1 8.5 isolation
// check, arch025-X3). cmake/iso.cmake compiles a copy of this file, of the module sources and of the headers
// listed in tests/sampling/isolation.txt, with that copied tree as the only include path, and links nothing but
// whirl_sampling_iso: no whirl, no whirl_server_core, no tokenizer GGUF, no GPU. The test bodies are the ones
// whirl-server-tests runs (tests/sampling/sampling_unit.inc), so the two programs pin the same fixed-RNG table.
//
//   whirl-sampling-tests [--only NAME[,NAME...]] [--list]
// SPDX-License-Identifier: Apache-2.0

#include "sampling/acceptor.h"
#include "sampling/sampler.h"
#include "whirl/model_types.h"  // qwen35::ctl_rows / max_ng_drafts (header-only constants)
#include "whirl/relax_accept.h"
#include "whirl/spec_sample.h"
#define WHIRL_TEST_FAIL_STDERR 1  // this program printed its failures on stderr as "  FAIL ..."
#include "../test_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

// the same context as tests/server/server_tests.cpp: Sampler / Cand under whirl::server (engine.h's using-declarations),
// then the pre-move reference code and the fixed cases inside namespace whirl::server
namespace whirl::server {
using sampling::Cand;
using sampling::Sampler;
#include "server/sampling_oldref.inc"
#include "server/sampling_cases.h"
}  // namespace whirl::server

using namespace whirl;
using namespace whirl::server;

namespace {

using whirl::test::g_fail;
using whirl::test::g_pass;
// CHECK / CHECK_EQ / the counters: tests/test_check.h (arch025 N16)



#include "sampling/sampling_unit.inc"

}  // namespace

int main(int argc, char** argv) {
    struct T {
        const char* name;
        std::function<void()> fn;
    };
    // the names whirl-server-tests uses for the same functions
    const std::vector<T> tests = {
        {"sampler", testSampler},
        {"acceptor_fixed", testAcceptorFixed},
        {"acceptor_rules", testAcceptorRules},
        {"acceptor_random", testAcceptorRandom},
    };
    std::set<std::string> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") {
            for (const T& t : tests) std::printf("%s\n", t.name);
            return 0;
        }
        if (a == "--only" && i + 1 < argc) {
            const std::string s = argv[++i];
            std::size_t p = 0;
            while (p <= s.size()) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                only.insert(s.substr(p, e - p));
                p = e + 1;
            }
        } else {
            std::fprintf(stderr, "usage: whirl-sampling-tests [--only NAME[,NAME...]] [--list]\n");
            return 2;
        }
    }
    for (const std::string& n : only) {
        if (std::none_of(tests.begin(), tests.end(), [&](const T& t) { return n == t.name; })) {
            std::fprintf(stderr, "unknown test '%s' (--list)\n", n.c_str());
            return 2;
        }
    }
    for (const T& t : tests) {
        if (!only.empty() && !only.count(t.name)) continue;
        const int f0 = g_fail;
        t.fn();
        std::printf("%-16s %s\n", t.name, g_fail == f0 ? "ok" : "FAIL");
    }
    std::printf("sampling tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
