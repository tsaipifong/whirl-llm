// Prefix cache: lookup / insertion / eviction decisions over the per-slot
// checkpoints, the shared prefix checkpoints (item S) and the host tier index.
// Host-only and GPU-free: the engine keeps the device copies and asks this
// module which checkpoint to resume from, which one to overwrite and which
// entry to evict. Dependency direction: engine -> PrefixCache -> cache::Tier.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace whirl::cache {

using Tokens = std::span<const std::uint32_t>;

std::size_t lcpLen(Tokens a, Tokens b);
bool prefixEq(Tokens a, Tokens b);

// Index metadata of a state checkpoint (the engine's Ckpt derives from it and
// adds the device buffers).
struct CkptKey {
    bool valid = false;
    std::uint32_t pos = 0;
    std::uint64_t seq = 0;
    const char* kind = "";
    bool has_logits = true;
};

// Index metadata of a shared prefix checkpoint (the engine's Spe derives from it).
struct SpeKey {
    bool valid = false;
    std::uint32_t n = 0;
    const char* kind = "";
    std::vector<std::uint32_t> tokens;
    std::uint64_t last_used = 0;
    bool pin = false;
    std::uint64_t uses = 0;
};

// Cache part of a slot (the engine's Slot derives from it).
struct SlotCache {
    std::vector<std::uint32_t> cache_tokens;
    std::uint64_t last_used = 0;
};

// Allocation-free view of an array of a type derived from T (std::span of the
// derived type, seen as its T base).
template <class T>
class View {
public:
    View() = default;
    template <class D>
        requires std::is_convertible_v<D*, T*>
    View(std::span<D> s) : base_(const_cast<void*>(static_cast<const void*>(s.data()))), n_(s.size()) {
        at_ = [](void* b, std::size_t i) -> T* { return static_cast<D*>(b) + i; };
    }
    template <class D>
        requires std::is_convertible_v<D*, T*>
    View(std::vector<D>& v) : View(std::span<D>(v)) {}
    template <class D>
        requires std::is_convertible_v<const D*, T*>
    View(const std::vector<D>& v) : View(std::span<const D>(v)) {}
    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    T& operator[](std::size_t i) const { return *at_(base_, i); }

private:
    void* base_ = nullptr;
    std::size_t n_ = 0;
    T* (*at_)(void*, std::size_t) = nullptr;
};

// Checkpoint metadata of a tier entry.
struct TierCk {
    std::uint32_t pos = 0;
    std::uint8_t valid = 0, has_logits = 0, kind = 0;
};

// What the prefix cache needs from a host tier (RAM / SSD index). The server
// implements it over tier::Tier (tier_adapter.h); unit tests use a mock.
class Tier {
public:
    using Entry = const void*;
    virtual ~Tier() = default;
    virtual std::uint32_t minTokens() const = 0;
    virtual std::size_t entryCount() const = 0;
    virtual Entry entryAt(std::size_t i) const = 0;
    virtual Entry find(std::uint64_t id) const = 0;  // nullptr: none
    virtual std::uint64_t id(Entry e) const = 0;
    virtual Tokens tokens(Entry e) const = 0;
    virtual std::uint32_t nTok(Entry e) const = 0;
    virtual std::uint32_t nck(Entry e) const = 0;
    virtual TierCk ck(Entry e, std::uint32_t k) const = 0;
};

struct Config {
    bool enabled = true;         // !--no-prefix-cache
    bool lcp_on = false;         // LCP checkpoints
    std::size_t sys_min = 0;     // system-message split minimum (0 = off)
    std::uint32_t kv_page = 64;  // tokens per KV page
};

struct SlotMatch {
    std::uint32_t reuse = 0;
    int ck = -1;  // index of the checkpoint to resume from (-1: none)
    std::size_t lcp = 0;
};

struct TierResume {
    std::uint32_t reuse = 0;
    std::uint8_t kind = 0;
};

// Eviction victim when the KV pool is full.
struct Victim {
    enum Kind { none, slot, spe } kind = none;
    std::size_t index = 0;
};

using PosPred = std::function<bool(std::size_t)>;          // position is on the chunk schedule
using FloorFn = std::function<std::size_t(std::size_t)>;   // schedule floor at or below a position
using SlotPred = std::function<bool(std::size_t)>;         // slot index may be evicted

class PrefixCache {
public:
    virtual ~PrefixCache() = default;
    virtual const Config& config() const = 0;
    bool enabled() const { return config().enabled; }
    virtual void attachTier(const Tier* t) = 0;
    virtual const Tier* tier() const = 0;
    // LRU clock shared by slots and shared entries: the next use stamp
    virtual std::uint64_t touch() = 0;

    // lookup
    virtual SlotMatch matchSlot(Tokens cached, View<const CkptKey> cks, Tokens toks) const = 0;
    virtual int matchShared(View<const SpeKey> spes, Tokens toks, const PosPred& on_schedule) const = 0;
    virtual int findShared(View<const SpeKey> spes, Tokens prefix) const = 0;
    virtual TierResume matchTierEntry(Tier::Entry e, Tokens toks) const = 0;
    virtual std::uint64_t findTierShared(Tokens prefix) const = 0;  // tier id of a 1-checkpoint copy, 0 none
    virtual std::uint32_t lcpTarget(View<const SlotCache> slots, View<const SpeKey> spes, Tokens toks,
                                    std::uint32_t reuse, std::uint32_t sys, const FloorFn& floor) const = 0;

    // insertion / eviction
    virtual std::size_t ckptSlotFor(View<const CkptKey> cks, std::uint32_t pos) const = 0;
    virtual int sharedVictim(View<const SpeKey> spes) const = 0;
    virtual Victim evictVictim(View<const SlotCache> slots, const SlotPred& evictable, View<const SpeKey> spes) const = 0;
};

std::unique_ptr<PrefixCache> makePrefixCache(const Config& cfg);

}  // namespace whirl::cache
