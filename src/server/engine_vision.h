// Server engine, image input (arch025-N4b, from class Engine): the embedding cache by content hash (LRU over unpinned
// entries, capped at VisionConfig::cache_bytes), the per-slot image maps for the model and the encoder's idle release.
// Main thread only.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the research prototype (server.zig, item VIS).

#pragma once

#include "backend.h"
#include "device/backend.h"
#include "vision_iface.h"
#include "whirl/device_mem.h"
#include "whirl/vision.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace whirl::server {

// The members in their former Engine order (vis_cfg_ .. vis_rel_pending_, the last members of Engine): ~Engine
// destroys them at the same point and in the same order.
class ImageEmbeddings {
public:
    ImageEmbeddings(ServerModel& m, device::Backend& ops) : m_(m), ops_(ops) {}
    ImageEmbeddings(const ImageEmbeddings&) = delete;
    ImageEmbeddings& operator=(const ImageEmbeddings&) = delete;

    // the settings; with an encoder: the regions the model lends it and the device memory counters before any image
    void attach(const VisionConfig& vc);
    const VisionConfig& config() const { return cfg_; }
    // The embeddings of image p (cache hit, or encode now on the model stream, borrowing the prefill scratch);
    // pinned until unpin.
    EmbEntry* get(const vision::Prepared& p, std::uint64_t req, std::uint32_t slot);
    void unpin(EmbEntry* ent);
    // Embeddings for the spans that the prefill will run (from `from` on); jv nullptr: the slot has no images.
    void prepare(std::uint32_t slot, JobVis* jv, std::uint64_t req, std::uint32_t from);
    // release the job's embedding pins (the job ends)
    void done(std::uint32_t slot, JobVis* jv);
    bool wakeNeeded() const;
    // Vision idle: unload the encoder (kernels, resident weights, copy stream) after its idle timeout, and the RoPE
    // row buffer once no slot needs it.
    void idle();
    // statistics (Engine::Stats vis_enc / vis_hit) and the cache size
    std::uint64_t encoded() const { return n_enc_; }
    std::uint64_t hits() const { return n_hit_; }
    std::uint64_t bytes() const { return bytes_; }

private:
    // drop unpinned entries, least recently used first, down to the cap
    void evict();

    ServerModel& m_;
    device::Backend& ops_;
    VisionConfig cfg_;
    std::vector<std::unique_ptr<EmbEntry>> cache_;
    std::uint64_t bytes_ = 0;
    std::uint64_t seq_ = 0;
    std::uint64_t n_enc_ = 0, n_hit_ = 0;
    std::array<std::array<std::uint64_t, 2>, 16> lend_raw_{};
    std::vector<vision::Region> lend_;
    hip::LocalMemSnap snap0_;  // device memory counters right after the mmproj was loaded (no image yet)
    bool rel_pending_ = false;
};

}  // namespace whirl::server
