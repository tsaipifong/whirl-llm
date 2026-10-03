// Image spans of a token sequence for multi-section RoPE (Qwen3-VL "imrope")
// and the image embedding rows. Shared by the model (attention prep, row
// injection), the vision front end (placeholder expansion) and the server.
// SPDX-License-Identifier: Apache-2.0
//
// M-RoPE image positions follow llama.cpp tools/mtmd (MTMD_POS_TYPE_MROPE, MIT)
// / HF Qwen3-VL get_rope_index; see THIRD_PARTY_NOTICES.md.
// Ported from the research prototype (model/qwen35.zig, item VIS).

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace whirl::qwen35 {

// Token ids >= this are image placeholders (content-hash ids, one per image token).
constexpr std::uint32_t image_id_base = 0x80000000u;

// One image's tokens in a sequence: cache positions start .. start + n - 1, an
// nx x ny grid in row-major order.
struct VisSpan {
    std::uint32_t start = 0;
    std::uint32_t n = 0;
    std::uint32_t nx = 0;
    std::uint32_t ny = 0;
    // RoPE position delta before this image (sum over earlier images of n - max(nx, ny))
    std::int32_t delta0 = 0;
    // host projected embeddings [n][n_embd] f32; nullptr = not encoded (the span
    // is fully inside the reused prefix cache)
    const float* emb = nullptr;
};

// The image spans of one sequence, sorted by start. Token k outside every span
// has RoPE position k - (delta of the spans before it) in all three sections;
// image token i of a span at t = start - delta0 has {t, t + i / nx, t + i % nx}.
struct VisMap {
    std::span<const VisSpan> spans;

    bool active() const { return !spans.empty(); }

    std::array<std::int32_t, 3> rope(std::uint32_t k) const {
        std::int32_t d = 0;
        for (const VisSpan& sp : spans) {
            if (k < sp.start) break;
            const std::int32_t t = static_cast<std::int32_t>(sp.start) - sp.delta0;
            if (k < sp.start + sp.n) {
                const std::uint32_t idx = k - sp.start;
                return {t, t + static_cast<std::int32_t>(idx / sp.nx), t + static_cast<std::int32_t>(idx % sp.nx)};
            }
            d = sp.delta0 + static_cast<std::int32_t>(sp.n) - static_cast<std::int32_t>(std::max(sp.nx, sp.ny));
        }
        const std::int32_t p = static_cast<std::int32_t>(k) - d;
        return {p, p, p};
    }

    // RoPE delta after the last image (text positions = cache position - this)
    std::int32_t deltaEnd() const {
        if (spans.empty()) return 0;
        const VisSpan& sp = spans.back();
        return sp.delta0 + static_cast<std::int32_t>(sp.n) - static_cast<std::int32_t>(std::max(sp.nx, sp.ny));
    }

    // span containing cache position k (nullptr: none)
    const VisSpan* spanAt(std::uint32_t k) const {
        for (const VisSpan& sp : spans) {
            if (k < sp.start) return nullptr;
            if (k < sp.start + sp.n) return &sp;
        }
        return nullptr;
    }
};

}  // namespace whirl::qwen35
