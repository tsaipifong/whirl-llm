// Vision prompt side (host only): image placeholders -> per-image content token ids and multi-section RoPE spans.
// SPDX-License-Identifier: Apache-2.0
//
// tokenId / expand moved verbatim from src/vision/vision.cpp (arch025-lane2 leg 5 S6): the HTTP front end
// (src/server/http.cpp buildVis, module protocol) calls expand on the connection threads, and with these two in their own
// file it links them without the HIP encoder (vision.cpp: code object, pinned weights, device buffers). Declared in
// whirl/vision.h ("prompt side").

#include "whirl/vision.h"

#include <algorithm>

namespace whirl::vision {

namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

}  // namespace

std::uint32_t tokenId(const std::array<std::uint8_t, 32>& h, std::uint32_t i) {
    return qwen35::image_id_base | static_cast<std::uint32_t>(wyhash(0x69760000ull + i, h) & 0x7fffffffull);
}

std::vector<std::uint32_t> expand(std::span<const std::uint32_t> tokens, std::uint32_t pad_id, std::span<const Prepared* const> imgs,
                                  std::span<qwen35::VisSpan> spans) {
    std::vector<std::uint32_t> out;
    out.reserve(tokens.size());
    std::size_t k = 0;
    i32 delta = 0;
    for (std::uint32_t t : tokens) {
        if (t != pad_id) {
            out.push_back(t);
            continue;
        }
        if (k >= imgs.size() || k >= spans.size()) throw VisionError("ImagePlaceholderMismatch");
        const Prepared* p = imgs[k];
        const u32 n = p->nTokens();
        spans[k] = qwen35::VisSpan{static_cast<u32>(out.size()), n, p->nx, p->ny, delta, nullptr};
        for (u32 i = 0; i < n; ++i) out.push_back(tokenId(p->hash, i));
        delta += static_cast<i32>(n) - static_cast<i32>(std::max(p->nx, p->ny));
        k += 1;
    }
    if (k != imgs.size()) throw VisionError("ImagePlaceholderMismatch");
    return out;
}

}  // namespace whirl::vision
