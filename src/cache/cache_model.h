// The narrow model interface of the server's cache side (arch025-lane2 D2.2): exactly the model calls CkptStore
// (ckpt_store), PagePool (page_pool) and TierBridge (tier_bridge) make, so those modules - and PrefixStore through
// them - depend on this header instead of the whole ServerModel (src/server/backend.h). ServerModel derives from it
// (single inheritance: the declarations moved verbatim out of ServerModel, the semantics of every call stay those of
// the Model member of the same name); the engine still hands over its ServerModel&.
// Prior art: llama.cpp's llama_memory_i (the KV / recurrent memory sees a narrow interface, not llama_context) and
// Lakos levelization (dependency inversion: the lower layer declares what it needs), progress\arch025-lane2.PRIOR_ART.md.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/backend.h"     // device::Stream, device::DevPtr
#include "model/qwen35/kv.h"    // qwen35::KvArray (server::KvArr)
#include "whirl/model_types.h"  // qwen35::Config

#include <cstdint>
#include <span>

namespace whirl::server {

using device::DevPtr;
// One KV pool array (every attention layer, then the MTP layer): base address and bytes per token row (the model's
// qwen35::KvArray; also declared by server/backend.h).
using KvArr = qwen35::KvArray;

class CacheModel {
public:
    virtual const qwen35::Config& cfg() const = 0;
    virtual device::Stream stream() const = 0;
    // per-layer recurrent state of the current sequence (0 = attention layer)
    virtual std::span<const DevPtr> convState() const = 0;
    virtual std::span<const DevPtr> ssmState() const = 0;
    virtual std::uint64_t convBytes() const = 0;
    virtual std::uint64_t ssmBytes() const = 0;
    virtual std::uint32_t curSeq() const = 0;
    virtual std::uint32_t seqPages() const = 0;
    virtual void commitSeq(std::uint32_t s) = 0;
    virtual void dropPending(std::uint32_t s) = 0;
    virtual void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys) = 0;

protected:
    ~CacheModel() = default;  // not deleted through this interface (ServerModel has the virtual destructor)
};

}  // namespace whirl::server
