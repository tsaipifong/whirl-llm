// Speculative sampling of MTP drafts: pure parts (see include/whirl/spec_sample.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/spec_sample.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace whirl::spec {

double uniform(std::uint64_t seed, std::uint64_t pos_idx, std::uint32_t stream) {
    // stream 0 is the server sampler's Sampler::uniform bit for bit
    if (stream != 0) seed ^= static_cast<std::uint64_t>(stream) * 0xD1B54A32D192ED03ull;
    std::uint64_t z = seed + (pos_idx + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

QDist qFromWords(const std::int32_t* w) {
    QDist q;
    q.n = static_cast<std::uint32_t>(std::clamp<std::int32_t>(w[0], 0, static_cast<std::int32_t>(q_max)));
    for (std::uint32_t i = 0; i < q.n; ++i) {
        q.id[i] = static_cast<std::uint32_t>(w[q_off_ids + i]);
        std::memcpy(&q.p[i], &w[q_off_p + i], 4);
    }
    return q;
}

QDist draftDist(std::span<const std::uint32_t> ids, std::span<const float> logits, float m, float sum, float inv_t,
                std::uint32_t top_k, float top_p, float min_p) {
    // mirrors the device kernel (f32 arithmetic, same order)
    QDist q;
    const std::uint32_t K = static_cast<std::uint32_t>(std::min<std::size_t>({ids.size(), logits.size(), q_max}));
    float lg[q_max];
    for (std::uint32_t i = 0; i < K; ++i) {
        q.id[i] = ids[i];
        lg[i] = logits[i];
    }
    for (std::uint32_t i = 1; i < K; ++i) {  // insertion sort: logit desc, id asc
        std::uint32_t j = i;
        while (j > 0 && (lg[j - 1] < lg[j] || (lg[j - 1] == lg[j] && q.id[j - 1] > q.id[j]))) {
            std::swap(lg[j - 1], lg[j]);
            std::swap(q.id[j - 1], q.id[j]);
            --j;
        }
    }
    for (std::uint32_t i = 0; i < K; ++i) q.p[i] = std::exp((lg[i] - m) * inv_t) / sum;
    std::uint32_t n = K;
    const bool closed = top_k > 0 && top_k <= K;
    if (closed) n = top_k;
    if (top_p < 1.f) {
        float z = 1.f;
        if (closed) {
            z = 0.f;
            for (std::uint32_t i = 0; i < n; ++i) z += q.p[i];
        }
        float cum = 0.f;
        for (std::uint32_t j = 0; j < n; ++j) {
            cum += q.p[j] / z;
            if (cum >= top_p) {
                n = j + 1;
                break;
            }
        }
    }
    if (min_p > 0.f && n > 0) {
        const float thr = min_p * q.p[0];
        std::uint32_t j = 0;
        while (j < n && q.p[j] >= thr) ++j;
        n = std::max<std::uint32_t>(j, 1);
    }
    float z = 0.f;
    for (std::uint32_t i = 0; i < n; ++i) z += q.p[i];
    for (std::uint32_t i = 0; i < n; ++i) q.p[i] /= z;
    q.n = n;
    return q;
}

std::uint32_t sampleQ(const QDist& q, double u) {
    if (q.n == 0) return 0;
    const float target = static_cast<float>(u);
    float cum = 0.f;
    for (std::uint32_t i = 0; i < q.n; ++i) {
        cum += q.p[i];
        if (cum > target) return q.id[i];
    }
    return q.id[q.n - 1];
}

std::uint32_t sampleResidual(std::span<const std::uint32_t> ids, std::span<const double> p, const QDist& q, double u) {
    const std::size_t n = std::min(ids.size(), p.size());
    double z = 0;
    std::size_t best = 0;
    for (std::size_t i = 0; i < n; ++i) {
        z += std::max(0.0, p[i] - static_cast<double>(q.of(ids[i])));
        if (p[i] > p[best]) best = i;
    }
    if (n == 0) return 0;
    if (!(z > 0)) return ids[best];
    const double target = u * z;
    double cum = 0;
    std::size_t last = best;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = std::max(0.0, p[i] - static_cast<double>(q.of(ids[i])));
        if (w <= 0) continue;
        cum += w;
        last = i;
        if (cum > target) return ids[i];
    }
    return ids[last];
}

}  // namespace whirl::spec
