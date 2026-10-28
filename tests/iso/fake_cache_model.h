// A host-only server::CacheModel for the cache-side isolation programs (ARCH-1 8.5, arch025-lane2 leg 3: ckpt_store,
// page_pool). The recurrent-state buffers come from the device contract's NullBackend (host memory), so the cache code
// runs its real copies; the model calls it makes (commitSeq / dropPending / mapPages) are recorded.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cache/cache_model.h"
#include "device/null_backend.h"
#include "whirl/model_types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace iso {

class FakeCacheModel final : public whirl::server::CacheModel {
public:
    // n_layer layers; the DeltaNet ones (gdn[i]) get conv / ssm state buffers of the given sizes from ops
    FakeCacheModel(whirl::device::NullBackend& ops, std::vector<bool> gdn, std::uint64_t conv_b, std::uint64_t ssm_b, std::uint32_t n_embd,
                   std::uint32_t n_vocab)
        : ops_(ops), conv_b_(conv_b), ssm_b_(ssm_b) {
        cfg_.n_layer = static_cast<std::uint32_t>(gdn.size());
        cfg_.n_embd = n_embd;
        cfg_.n_vocab = n_vocab;
        for (bool g : gdn) {
            conv_.push_back(g ? ops_.malloc(conv_b) : 0);
            ssm_.push_back(g ? ops_.malloc(ssm_b) : 0);
        }
    }
    ~FakeCacheModel() {
        for (auto p : conv_)
            if (p) ops_.free(p);
        for (auto p : ssm_)
            if (p) ops_.free(p);
    }
    const whirl::qwen35::Config& cfg() const override { return cfg_; }
    whirl::device::Stream stream() const override { return nullptr; }
    std::span<const whirl::device::DevPtr> convState() const override { return conv_; }
    std::span<const whirl::device::DevPtr> ssmState() const override { return ssm_; }
    std::uint64_t convBytes() const override { return conv_b_; }
    std::uint64_t ssmBytes() const override { return ssm_b_; }
    std::uint32_t curSeq() const override { return cur_; }
    std::uint32_t seqPages() const override { return seq_pages_; }
    void commitSeq(std::uint32_t s) override { calls.push_back("commitSeq(" + std::to_string(s) + ")"); }
    void dropPending(std::uint32_t s) override { calls.push_back("dropPending(" + std::to_string(s) + ")"); }
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) override {
        std::string c = "mapPages(" + std::to_string(s) + "," + std::to_string(first) + ",";
        for (std::int32_t p : phys) c += std::to_string(p) + ";";
        calls.push_back(c + ")");
    }

    std::uint32_t cur_ = 0, seq_pages_ = 0;
    std::vector<std::string> calls;

private:
    whirl::device::NullBackend& ops_;
    whirl::qwen35::Config cfg_;
    std::uint64_t conv_b_, ssm_b_;
    std::vector<whirl::device::DevPtr> conv_, ssm_;
};

}  // namespace iso
