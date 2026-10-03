// Server side of image input: the encoder behind an interface (production:
// whirl::vision::Vision on the model stream, backend_qwen35.cpp; tests: a host
// mock), the per-request image state and the embedding cache entries.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the research prototype (server.zig, item VIS).

#pragma once

#include "whirl/vision.h"
#include "whirl/vismap.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace whirl::server {

class ServerVision {
public:
    virtual ~ServerVision() = default;
    // Decode + preprocess one image (connection threads; must be thread-safe).
    // Throws vision::VisionError ("ImageDecode", "ImageTooLarge").
    virtual vision::Prepared prepare(std::span<const std::uint8_t> bytes) const = 0;
    virtual std::uint32_t projDim() const = 0;
    // Encode on the model stream, borrowing `lend` (engine thread). Synchronous.
    virtual vision::EncodeStats encode(const vision::Prepared& p, std::span<const vision::Region> lend, float* out,
                                       std::size_t n) = 0;
    // Encoder kernels / resident weights loaded (VRAM in use).
    virtual bool loaded() const = 0;
    // Release after the idle timeout; true when it released now.
    virtual bool idleTick() = 0;
    virtual std::uint32_t idleS() const = 0;
    // VRAM free now (for the release log).
    virtual std::uint64_t vramFree() const = 0;
};

// Projected image embeddings by content hash (host memory, LRU, engine thread only).
struct EmbEntry {
    std::array<std::uint8_t, 32> hash{};
    std::vector<float> data;
    std::uint64_t last = 0;
    std::uint32_t pins = 0;
};

// A request's images: preprocessed pixels (content hash), their token spans in
// the prompt (multi-section RoPE) and, once the request starts, the embedding
// cache entries it holds.
struct JobVis {
    std::vector<vision::Prepared> preps;
    std::vector<qwen35::VisSpan> spans;
    qwen35::VisMap vmap;
    std::vector<EmbEntry*> ents;
};

// Server-wide vision settings (set at startup, read by connection threads).
struct VisionConfig {
    ServerVision* vis = nullptr;  // nullptr: no --mmproj
    bool allow_files = false;     // --allow-local-images
    std::uint32_t image_pad_id = 0;
    std::uint32_t vision_end_id = 0;
    // image ends at >= this position are chunk splits of the prompt's schedule and
    // the last one keeps a shared checkpoint (WHIRL_VIS_CKPT_MIN, 0 = off)
    std::uint32_t ck_min = 1024;
    std::uint64_t cache_bytes = 1ull << 30;  // --vis-cache-mb
    // diagnostics (WHIRL_VIS_DUMP=prefix): each encoded image's embeddings to prefix.<req>.f32
    std::string dump_prefix;
};

// Base64 (standard alphabet, with or without padding) -> bytes; false on bad input.
bool base64Decode(std::string_view in, std::vector<std::uint8_t>& out);

}  // namespace whirl::server
