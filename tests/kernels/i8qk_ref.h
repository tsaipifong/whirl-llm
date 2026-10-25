// feat027-i8qk: CPU reference of the int8 Q.K^T prefill attention (kernels/attn_i8qk.hip). No HIP: used by the
// GPU test (test_attn_i8qk.cpp) and the CPU error-bound test (tests/kernels/unit/i8qk_ref_tests.cpp).
// SPDX-License-Identifier: Apache-2.0
//
// Layouts as the kernels: K / V rows [key][n_kv][hd] (f16 values held as float), Q [query][n_head][q_stride] (the
// first hd values; gate not modelled), mu [n_kv][hd], kq [key][n_kv][hd] int8, ksc [n_kv][key].

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace i8qk {

// i8k_mean: mean over the sampled keys r = 0, st, 2 st, .. (st = max(1, rows / 2048)), 4 interleaved phases summed
// in float in the kernel's order
inline std::vector<float> meanSampled(const std::vector<float>& k, int rows, int row_el) {
    const int st = std::max(1, rows / 2048);
    std::vector<float> mu(static_cast<std::size_t>(row_el));
    for (int c = 0; c < row_el; ++c) {
        float part[4] = {0, 0, 0, 0};
        for (int ph = 0; ph < 4; ++ph)
            for (int r = ph * st; r < rows; r += 4 * st) part[ph] += k[static_cast<std::size_t>(r) * row_el + c];
        const int cnt = (rows + st - 1) / st;
        mu[static_cast<std::size_t>(c)] = (((part[0] + part[1]) + part[2]) + part[3]) / static_cast<float>(cnt);
    }
    return mu;
}

inline int q8(float x) { return static_cast<int>(std::min(std::max(std::nearbyint(x), -127.f), 127.f)); }

// i8k_quant: per (key, KV head) absmax over hd of k - mu
inline void quantK(const std::vector<float>& k, const std::vector<float>& mu, int rows, int n_kv, int hd, std::vector<std::int8_t>& kq,
                   std::vector<float>& ksc) {
    kq.assign(static_cast<std::size_t>(rows) * n_kv * hd, 0);
    ksc.assign(static_cast<std::size_t>(n_kv) * rows, 0.f);
    std::vector<float> v(static_cast<std::size_t>(hd));
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < n_kv; ++g) {
            const std::size_t b = (static_cast<std::size_t>(r) * n_kv + g) * hd;
            float a = 0.f;
            for (int d = 0; d < hd; ++d) {
                v[static_cast<std::size_t>(d)] = k[b + d] - mu[static_cast<std::size_t>(g) * hd + d];
                a = std::max(a, std::fabs(v[static_cast<std::size_t>(d)]));
            }
            const float inv = a > 0.f ? 127.f / a : 0.f;
            for (int d = 0; d < hd; ++d) kq[b + d] = static_cast<std::int8_t>(q8(v[static_cast<std::size_t>(d)] * inv));
            ksc[static_cast<std::size_t>(g) * rows + r] = a / 127.f;
        }
}

struct Shape {
    int n = 0, pos0 = 0, n_head = 0, n_kv = 0, hd = 256, q_stride = 512;
    float scale = 1.f / 16.f;
};

// Scores of the int8 path for query i, head h over keys [0, n_keys): sq * (sk * dot + q.mu / sq), in double
// (the kernel's f32 rounding of the epilogue is below the quantization error).
inline void scoresI8(const Shape& s, const std::vector<float>& q, const std::vector<std::int8_t>& kq, const std::vector<float>& ksc,
                     const std::vector<float>& mu, int rows, int i, int h, std::vector<double>& out) {
    const int g = h / (s.n_head / s.n_kv), n_keys = s.pos0 + i + 1;
    const float* qr = &q[(static_cast<std::size_t>(i) * s.n_head + h) * s.q_stride];
    float am = 0.f;
    double qm = 0;
    for (int d = 0; d < s.hd; ++d) {
        am = std::max(am, std::fabs(qr[d]));
        qm += static_cast<double>(qr[d]) * mu[static_cast<std::size_t>(g) * s.hd + d];
    }
    const float inv = am > 0.f ? 127.f / am : 0.f;
    const double sq = am > 0.f ? static_cast<double>(am / 127.f) : 1.0;
    std::vector<int> qi(static_cast<std::size_t>(s.hd));
    for (int d = 0; d < s.hd; ++d) qi[static_cast<std::size_t>(d)] = q8(qr[d] * inv);
    out.assign(static_cast<std::size_t>(n_keys), 0);
    for (int j = 0; j < n_keys; ++j) {
        const std::int8_t* kr = &kq[(static_cast<std::size_t>(j) * s.n_kv + g) * s.hd];
        long long dot = 0;
        for (int d = 0; d < s.hd; ++d) dot += static_cast<long long>(qi[static_cast<std::size_t>(d)]) * kr[d];
        out[static_cast<std::size_t>(j)] = sq * static_cast<double>(dot) * ksc[static_cast<std::size_t>(g) * rows + j] + qm;
    }
    (void)rows;
}

// exact scores (double) from float Q / K, optionally rounded to f16 first (the f16 path's inputs)
inline void scoresRef(const Shape& s, const std::vector<float>& q, const std::vector<float>& k, int i, int h, bool f16_inputs, float (*roundh)(float),
                      std::vector<double>& out) {
    const int g = h / (s.n_head / s.n_kv), n_keys = s.pos0 + i + 1;
    const float* qr = &q[(static_cast<std::size_t>(i) * s.n_head + h) * s.q_stride];
    out.assign(static_cast<std::size_t>(n_keys), 0);
    for (int j = 0; j < n_keys; ++j) {
        const float* kr = &k[(static_cast<std::size_t>(j) * s.n_kv + g) * s.hd];
        double dot = 0;
        for (int d = 0; d < s.hd; ++d) {
            const float a = f16_inputs ? roundh(qr[d]) : qr[d];
            dot += static_cast<double>(a) * kr[d];
        }
        out[static_cast<std::size_t>(j)] = dot;
    }
}

// softmax(scores * scale) . V for one (query, head): p_round = true rounds P to f16 before P.V (the kernels' f16 P,
// before normalization by l, which is f32 / exact here)
inline void softmaxPv(const Shape& s, const std::vector<double>& sc, const std::vector<float>& v, int h, bool p_round, float (*roundh)(float),
                      double* o) {
    const int g = h / (s.n_head / s.n_kv);
    double mx = -1e300;
    for (double x : sc) mx = std::max(mx, x);
    double l = 0;
    for (int d = 0; d < s.hd; ++d) o[d] = 0;
    for (std::size_t j = 0; j < sc.size(); ++j) {
        double p = std::exp((sc[j] - mx) * s.scale);
        l += p;
        if (p_round) p = roundh(static_cast<float>(p));
        const float* vr = &v[(j * s.n_kv + g) * s.hd];
        for (int d = 0; d < s.hd; ++d) o[d] += p * vr[d];
    }
    for (int d = 0; d < s.hd; ++d) o[d] /= l;
}

}  // namespace i8qk
