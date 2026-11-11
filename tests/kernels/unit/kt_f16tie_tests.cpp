// whirl-kt-f16tie-tests (CPU only, v0252-moedf16b, PLAN V1-76): the kt check of the f16 MoE down projection
// (kt_f16_tie.h). Written from the 11-11 R9700 job (q7 9525m) failure: "gemm8r_moeh W2=null (f16 moe_yd) ==
// f16(gemm8r_moe down)" had 211 of 3,276,800 mismatches of one f16 ulp at both tiles; the f16 epilogue is compiled to
// v_fma_mixlo_f16 (one rounding of the exact product) while the reference rounds twice (f32, then f16).
//   * the double-rounding model on random f32 x f32 products reproduces that failure class with the old exact compare
//     (mismatches > 0 at about the job's rate) and every one of those differences is an exact f16 tie of the f32 value;
//   * the new compare accepts exactly those and nothing else (a 1-ulp error off a tie, a sign flip, 2 ulps: mismatches).
// Exit 0 = all pass. SPDX-License-Identifier: Apache-2.0

#include "../kt_f16_tie.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0, g_pass = 0;
static void check(const char* name, bool ok) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
    (ok ? g_pass : g_fail)++;
}

int main() {
    using namespace kt;
    const float tie1 = 1.0f + std::ldexp(1.0f, -11);  // halfway between 1 (0x3c00, even) and 1 + 2^-10 (0x3c01)
    check("RNE of the tie 1 + 2^-11 is the even neighbour 0x3c00", tieD2h(tie1) == 0x3c00);
    check("tie: the other neighbour 0x3c01 is accepted", f16TieOther(tie1, 0x3c00, 0x3c01));
    check("tie: two ulps away (0x3c02) is not", !f16TieOther(tie1, 0x3c00, 0x3c02));
    check("tie: the other sign (0xbc01) is not", !f16TieOther(tie1, 0x3c00, 0xbc01));
    check("tie: the RNE value itself is not 'the other'", !f16TieOther(tie1, 0x3c00, 0x3c00));
    const float off = tie1 + std::ldexp(1.0f, -23);  // just above the tie: RNE 0x3c01, nothing else allowed
    check("non-tie: RNE 0x3c01", tieD2h(off) == 0x3c01);
    check("non-tie: the neighbour 0x3c00 is not accepted", !f16TieOther(off, 0x3c01, 0x3c00));
    const float sub = std::ldexp(1.0f, -25);  // halfway between 0 and the smallest subnormal
    check("subnormal tie: 0x0000 / 0x0001", tieD2h(sub) == 0x0000 && f16TieOther(sub, 0x0000, 0x0001));
    check("negative subnormal tie: 0x8000 / 0x8001", tieD2h(-sub) == 0x8000 && f16TieOther(-sub, 0x8000, 0x8001));
    const float ntie = -(1.0f + 3 * std::ldexp(1.0f, -11));  // halfway between -(1 + 2^-10) (0xbc01, odd) and -(1 + 2^-9) (0xbc02)
    check("negative tie: RNE 0xbc02, other 0xbc01 accepted", tieD2h(ntie) == 0xbc02 && f16TieOther(ntie, 0xbc02, 0xbc01));

    // the double-rounding model: the kernel's one rounding of the exact product vs the reference f16(f32 product)
    std::mt19937_64 rng(0x6d6f6564663136ull);
    std::normal_distribution<float> acc_d(0.0f, 40.0f);
    std::uniform_real_distribution<float> sc_d(0.5f, 1.0f);
    std::uniform_int_distribution<int> ex_d(-16, -6);
    const std::size_t n = 2000000;
    std::vector<float> y32(n);
    std::vector<std::uint16_t> once(n);
    std::size_t old_mis = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float acc = acc_d(rng), s = std::ldexp(sc_d(rng), ex_d(rng));
        volatile float p = acc * s;  // the f32 product (gemm8r_moe's stored value)
        y32[i] = p;
        once[i] = tieD2h(static_cast<double>(acc) * static_cast<double>(s));  // exact in double (24 x 24 bits): v_fma_mixlo_f16
        old_mis += once[i] != tieD2h(static_cast<double>(y32[i]));
    }
    const double rate = static_cast<double>(old_mis) / static_cast<double>(n);
    std::printf("model: %zu of %zu differ under the old exact compare (rate %.2e; job q7 9525m: 211 / 3276800 = 6.44e-05)\n", old_mis, n, rate);
    check("model: the old exact compare fails (reproduces the 9525m kt FAIL class)", old_mis > 0);
    check("model: rate within 2^-16 .. 2^-12 (the 9525m rate 6.4e-5 is inside)", rate > std::ldexp(1.0, -16) && rate < std::ldexp(1.0, -12));
    const F16TieCmp c = cmpF16RneOrTie(once.data(), y32.data(), n);
    check("model: new compare has 0 mismatches", c.mismatches == 0);
    check("model: every old difference is an exact f16 tie of the f32 value", c.ties_other == old_mis);
    // a wrong kernel is still caught
    std::size_t k = 0;
    while (k + 3 < n && (once[k] != tieD2h(static_cast<double>(y32[k])) || f16TieOther(y32[k], once[k], static_cast<std::uint16_t>(once[k] ^ 1u)))) ++k;
    std::vector<std::uint16_t> bad = once;
    bad[k] ^= 1u;  // one ulp off at a non-tie value
    bad[k + 1] ^= 0x8000u;  // sign flip
    bad[k + 2] = static_cast<std::uint16_t>(bad[k + 2] + 2);  // two ulps
    const F16TieCmp cb = cmpF16RneOrTie(bad.data(), y32.data(), n);
    check("wrong values (1 ulp off a tie, sign, 2 ulps) are 3 mismatches", cb.mismatches == 3 && cb.first == k);

    std::printf("whirl-kt-f16tie-tests: %d PASS, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
