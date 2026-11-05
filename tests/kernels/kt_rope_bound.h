// whirl-kernel-test: the tolerance of a RoPE-rotated output element, and the CPU self-test of that bound (no GPU).
// v0251-tiers (PLAN B26-53 (a); analysis progress\v0251-ktfull.HANDOFF.md, derivation progress\v0251-ktfull.PRIOR_ART.md s3).
//
// An angle error dtheta moves y_i by -y_partner * dtheta and y_partner by y_i * dtheta: the error of a rotated element scales
// with its pair, not with its own magnitude (fp32 kernel: |dy| <= (|x0| + |x1|) * (3u |theta| + 16u), u = 2^-24, sound for
// positions up to about 900). A bound atol + rtol * |ref| fails a correct kernel whenever ref is near 0 and its partner is not
// (0.2.5.1 8060S release gate: attn_prep_m g2[16130], ref 0.007431791, partner 2.8, worst 1.08). WHIRL's own rope_neox check
// (test_misc.cpp) and vis_qkv_prep2 (test_vision.cpp) use the pair magnitude |x0| + |x1| of the values before the rotation;
// ropeBoundScale gives that scale to cmpTol (rtol 1e-4, atol 1e-5 as before); unrotated dims keep |x|.
// ropeBoundSelftest(): the bound's own cases, written from the bug description: the gate's element and fp32 angles with powf
// 1 ulp high must stay inside the bound; a wrong section, an h / w swap, a wrong pair and f16-rounded outputs must not.
// testAttn reports them in the attn family; they draw no number from the shared Ctx rng.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace kt {

constexpr double kRopeRtol = 1e-4, kRopeAtol = 1e-5;

// cmpTol scale of one head of hd values whose dims i and i + n_rot / 2 (i < n_rot / 2) are rotated as pair i: pre = the values
// just before the rotation, ref = the double reference after it; scale[i] multiplies rtol in the bound atol + rtol * scale[i]
inline void ropeBoundScale(const double* pre, const double* ref, double* scale, int hd, int n_rot) {
    (void)pre;
    (void)n_rot;
    for (int i = 0; i < hd; ++i) scale[i] = std::fabs(ref[i]);  // TEST-FIRST commit: the current rule (= cmpTolRel(1e-4, 1e-5))
}

// cmpTol's metric (kt.cpp): max over i of |got - ref| / (atol + rtol * scale[i]); above 1 fails (NaN counts as above)
inline double ropeBoundMetric(const std::vector<float>& got, const std::vector<double>& ref, const std::vector<double>& scale,
                              double rtol = kRopeRtol, double atol = kRopeAtol) {
    double m = 0;
    for (std::size_t i = 0; i < got.size() && i < ref.size() && i < scale.size(); ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - ref[i]) / (atol + rtol * scale[i]);
        if (!(e <= m)) m = std::isnan(e) ? INFINITY : e;
    }
    return m;
}

struct RopeBoundCase {
    std::string name;
    bool expect_pass;  // true: metric <= 1 (a correct kernel), false: metric > 1 (the defect must be caught)
    double metric;
};

namespace rope_bound_detail {

// test_attn.cpp's multi-section map (sections {t, h, w} = 4 / 3 / 3): pairs i % 3 == 1 (i < 9) take h, i % 3 == 2 (i < 9) take w
inline int pairPos(int i, int pt, int ph, int pw) {
    const int sec = i % 3;
    return (sec == 1 && i < 9) ? ph : (sec == 2 && i < 9) ? pw : pt;
}

// the nearest f16 value (11 significant bits; the values here are in the normal f16 range)
inline float toHalf(float x) {
    if (x == 0.0f || !std::isfinite(x)) return x;
    int e = 0;
    const double m = std::frexp(static_cast<double>(x), &e);
    return static_cast<float>(std::ldexp(std::nearbyint(std::ldexp(m, 11)), e - 11));
}

enum class Defect { none, wrong_section, swapped_hw, wrong_pair, f16_output };

// 5 tokens x 6 heads of 256 values (64 rotated) at test_attn.cpp's {t, h, w} rows 100 + t, 100 + 4t, 100 + t + 7 (t % 2):
// "kernel" = attn_prep_impl's fp32 rotation (theta = (float)p * powf(theta_scale, (float)i), sin / cos, y0 = x0 c - x1 s,
// y1 = x0 s + x1 c) with powf 1 ulp high for every i >= 1 (OCML powf is within 1 ulp) and the given defect; reference = the
// double rotation test_attn.cpp computes; metric = ropeBoundMetric with ropeBoundScale
inline double runCase(Defect defect, unsigned long long seed) {
    constexpr int hd = 256, n_rot = 64, heads = 6, n = 5;
    const float ts = std::pow(1.0e7f, -2.0f / n_rot);
    std::mt19937_64 rng{seed};
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> got;
    std::vector<double> ref, scale;
    for (int t = 0; t < n; ++t) {
        const int pt = 100 + t, ph = 100 + t + 3 * t, pw = 100 + t + 7 * (t % 2);
        for (int h = 0; h < heads; ++h) {
            std::vector<float> x(hd);
            for (float& e : x) e = nd(rng);
            std::vector<double> pre(x.begin(), x.end()), r(pre), sc(hd);
            std::vector<float> y(x);
            for (int i = 0; i < n_rot / 2; ++i) {
                const int p = pairPos(i, pt, ph, pw);
                const double th = p * std::pow(static_cast<double>(ts), i);
                r[i] = pre[i] * std::cos(th) - pre[i + n_rot / 2] * std::sin(th);
                r[i + n_rot / 2] = pre[i] * std::sin(th) + pre[i + n_rot / 2] * std::cos(th);
                int pk = p;
                if (defect == Defect::wrong_section && i == 1) pk = pt;  // pair 1 (an h pair) at the t position
                if (defect == Defect::swapped_hw && i == 2) pk = ph;     // pair 2 (a w pair) at the h position
                float pwr = std::pow(ts, static_cast<float>(i));
                if (i >= 1) pwr = std::nextafter(pwr, 2.0f);
                const float th32 = static_cast<float>(pk) * pwr;
                const float c = std::cos(th32), s = std::sin(th32);
                const int partner = defect == Defect::wrong_pair ? i + n_rot / 2 + 1 : i + n_rot / 2;  // one dim off
                const float a0 = x[i], a1 = x[partner];
                y[i] = a0 * c - a1 * s;
                y[i + n_rot / 2] = a0 * s + a1 * c;
            }
            if (defect == Defect::f16_output)
                for (float& e : y) e = toHalf(e);
            ropeBoundScale(pre.data(), r.data(), sc.data(), hd, n_rot);
            got.insert(got.end(), y.begin(), y.end());
            ref.insert(ref.end(), r.begin(), r.end());
            scale.insert(scale.end(), sc.begin(), sc.end());
        }
    }
    return ropeBoundMetric(got, ref, scale);
}

// the 0.2.5.1 8060S release-gate element: pair 2 of a head after the rotation = (0.007431791, 2.8) (theta 37.25); the kernel's
// output is that pair turned by a further -4.15e-6 rad (got 0.007443412, err 1.162e-5 = 2.8 x 4.15e-6); the rest of the head
// is exact
inline double gateElement() {
    constexpr int hd = 256, n_rot = 64, i = 2;
    std::vector<double> pre(hd, 0.5), ref(hd, 0.5), sc(hd);
    const double th = 37.25, r0 = 0.007431791, r1 = 2.8, d = -4.15e-6;
    pre[i] = r0 * std::cos(th) + r1 * std::sin(th);  // the pair before the rotation by th
    pre[i + n_rot / 2] = -r0 * std::sin(th) + r1 * std::cos(th);
    ref[i] = r0;
    ref[i + n_rot / 2] = r1;
    std::vector<float> got(hd);
    for (int k = 0; k < hd; ++k) got[static_cast<std::size_t>(k)] = static_cast<float>(ref[static_cast<std::size_t>(k)]);
    got[i] = static_cast<float>(r0 * std::cos(d) - r1 * std::sin(d));
    got[i + n_rot / 2] = static_cast<float>(r0 * std::sin(d) + r1 * std::cos(d));
    ropeBoundScale(pre.data(), ref.data(), sc.data(), hd, n_rot);
    return ropeBoundMetric(got, ref, sc);
}

}  // namespace rope_bound_detail

inline std::vector<RopeBoundCase> ropeBoundSelftest() {
    using rope_bound_detail::Defect;
    using rope_bound_detail::runCase;
    const unsigned long long seed = 20261105ull;
    return {
        {"P1 0.2.5.1 8060S gate element (ref 0.007431791, partner 2.8, 4.15e-6 rad off)", true, rope_bound_detail::gateElement()},
        {"P2 fp32 angles, powf 1 ulp high, positions 100..116", true, runCase(Defect::none, seed)},
        {"F1 wrong section (pair 1 at the t position)", false, runCase(Defect::wrong_section, seed)},
        {"F2 h / w swapped (pair 2 at the h position)", false, runCase(Defect::swapped_hw, seed)},
        {"F3 wrong pair (partner one dim off)", false, runCase(Defect::wrong_pair, seed)},
        {"F4 f16-rounded outputs", false, runCase(Defect::f16_output, seed)},
    };
}

}  // namespace kt
