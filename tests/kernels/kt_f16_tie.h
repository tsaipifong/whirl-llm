// whirl-kernel-test: compare an f16 kernel output with f16(the f32 result) when the kernel rounds the exact product once.
// v0252-moedf16b (PLAN V1-76; analysis progress\v0252-moedf16b.HANDOFF.md, prior art progress\v0252-moedf16.PRIOR_ART.md).
//
// gemm8_moe_impl's f16 epilogue (kernels\moe_mxfp4.hip, OH) writes (_Float16)(acc * ldexpf(sx, e - 127)). hipcc's default
// -ffp-contract=fast-honor-pragmas contracts fptrunc(fmul) into v_fma_mixlo_f16 (acc, scale, 0): the exact product rounded
// ONCE to f16 (ROCm 7.2 objdump of gemm8r_moeh / gemm8r_moe32h: 32 x v_fma_mixlo_f16, no v_cvt_f16_f32; the f32 entries
// gemm8r_moe / _moe32: v_mul_f32). The reference f16(gemm8r_moe's f32 value) rounds TWICE (f32, then f16). The two differ
// exactly when the f32 value lands on an f16 tie (a value halfway between two f16 neighbours, representable in f32) while the
// exact product does not: RNE then picks the even neighbour, the single rounding the neighbour on the exact product's side.
// Rate about 2^-13 / 2 of the values (q7 9525m: 211 of 3,276,800 = 6.4e-5, both tiles, max |d| one f16 ulp).
// So the check is exact, not a tolerance: equal to f16(f32) everywhere, except that at an exact f16 tie of the f32 value
// the other neighbour is also accepted; anything else is a mismatch. The kernel is kept: the single rounding is the correctly
// rounded f16 of the exact product (more accurate), and the gate / up projections of the same fp8 path (int0252 balance / fast)
// use the same epilogue, so changing its rounding would change shipped balance / fast outputs.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kt {

// exact value of an f16 bit pattern (NaN / inf as in IEEE)
inline double tieH2d(std::uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp(static_cast<double>(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(static_cast<double>(m | 1024), e - 25);
    return (h & 0x8000) ? -v : v;
}

// RNE of a double to f16 (one rounding; overflow -> inf)
inline std::uint16_t tieD2h(double d) {
    if (std::isnan(d)) return 0x7e00;
    const std::uint16_t sign = std::signbit(d) ? 0x8000 : 0;
    const double a = std::fabs(d);
    if (a == 0) return sign;
    if (a < std::ldexp(1.0, -14)) return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(std::nearbyint(a * std::ldexp(1.0, 24))));
    int E = 0;
    (void)std::frexp(a, &E);
    int e = E - 1;
    double r = std::nearbyint(std::ldexp(a, 10 - e));
    if (r >= 2048) {
        r = 1024;
        ++e;
    }
    if (e + 15 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00);
    return static_cast<std::uint16_t>(sign | ((e + 15) << 10) | (static_cast<int>(r) - 1024));
}

// true iff y32 lies exactly halfway between the f16 values rne and got, which are neighbours of the same sign
// (rne = RNE(y32), got = the other neighbour; both finite)
inline bool f16TieOther(float y32, std::uint16_t rne, std::uint16_t got) {
    if (got == rne || ((got ^ rne) & 0x8000)) return false;
    const int dm = static_cast<int>(got & 0x7fff) - static_cast<int>(rne & 0x7fff);
    if (dm != 1 && dm != -1) return false;
    const double a = tieH2d(rne), b = tieH2d(got);
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return (a + b) * 0.5 == static_cast<double>(y32);  // exact: f16 values, their sum and its half are exact in double
}

struct F16TieCmp {
    std::size_t n = 0, mismatches = 0, ties_other = 0, first = static_cast<std::size_t>(-1);
    double max_abs = 0;
};

// got[i] vs f16(y32[i]): equal, or (only at an exact f16 tie of y32[i]) the other neighbour; anything else = a mismatch
inline F16TieCmp cmpF16RneOrTie(const std::uint16_t* got, const float* y32, std::size_t n) {
    F16TieCmp r;
    r.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t rne = tieD2h(static_cast<double>(y32[i]));
        if (got[i] == rne) continue;
        const double d = std::fabs(tieH2d(got[i]) - tieH2d(rne));
        if (f16TieOther(y32[i], rne, got[i])) {
            ++r.ties_other;
            if (d > r.max_abs) r.max_abs = d;
            continue;
        }
        if (r.first == static_cast<std::size_t>(-1)) r.first = i;
        ++r.mismatches;
        if (!(d <= r.max_abs)) r.max_abs = d;  // NaN-safe
    }
    return r;
}

}  // namespace kt
