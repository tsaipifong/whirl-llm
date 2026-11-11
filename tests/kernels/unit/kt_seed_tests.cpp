// whirl-kt-seed-tests (CPU only, v0252-ktrng, PLAN B26-53 (c)): every kt test family draws its random inputs from its own
// stream, seeded from a stable hash of the family name (kt_seed.h), so adding / removing / reordering a family never
// changes another family's inputs (before: one shared mt19937_64{12345}, found 11-05 by v0251-ktfull).
// Exit 0 = all pass. SPDX-License-Identifier: Apache-2.0

#include "../kt_seed.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0, g_pass = 0;
static void check(const char* name, bool ok) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    (ok ? g_pass : g_fail)++;
}

static std::vector<unsigned long long> draw(const char* family, int n, int skip_other_first) {
    kt::FamilyRng r;
    if (skip_other_first > 0) {  // another family ran first and drew a lot from its own stream
        r.begin("quant");
        for (int i = 0; i < skip_other_first; ++i) (void)r.rng();
    }
    r.begin(family);
    std::vector<unsigned long long> v;
    for (int i = 0; i < n; ++i) v.push_back(r.rng());
    return v;
}

int main() {
    // the names of the fams[] table in kernel_test_main.cpp
    const char* fams[] = {"quant", "gemv", "gemm", "attn", "attndq4", "gdn", "moe", "moemx", "moef", "moefuse", "misc", "attnbench",
                          "gemvh", "gemvhbench", "gemvhv", "dflash", "ws4", "ws4bench", "ws4g", "ws4gbench", "vision", "gemvk", "attnkgl",
                          "specacc", "specaccz", "e1in"};  // v0252-e1p1, v0252-e1p1b
    std::set<std::uint64_t> seeds;
    for (const char* f : fams) seeds.insert(kt::familySeed(f));
    check("every family has a distinct seed", seeds.size() == sizeof(fams) / sizeof(fams[0]));
    check("seed is a pure function of the name", kt::familySeed("gemv") == kt::familySeed(std::string("gem") + "v"));
    check("case matters (names are exact)", kt::familySeed("gemv") != kt::familySeed("Gemv"));
    check("seed is not the old shared 12345", kt::familySeed("quant") != 12345u && kt::familySeed("gemv") != 12345u);
    // independence: a family's stream does not depend on what ran (and drew) before it
    check("gemv stream same alone / after quant drew 1000", draw("gemv", 64, 0) == draw("gemv", 64, 1000));
    check("attn stream same alone / after quant drew 7", draw("attn", 64, 0) == draw("attn", 64, 7));
    check("two families draw different streams", draw("gemv", 16, 0) != draw("gemm", 16, 0));
    // pin: a rename or a changed hash changes every recorded input; this line must be edited on purpose
    check("pinned seed of gemv (FNV-1a 64 + splitmix64 finaliser)", kt::familySeed("gemv") == 0x8c71639dbcce8b33ULL);
    std::printf("familySeed(gemv) = 0x%016llx\n", static_cast<unsigned long long>(kt::familySeed("gemv")));
    std::printf("whirl-kt-seed-tests: %d PASS, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
