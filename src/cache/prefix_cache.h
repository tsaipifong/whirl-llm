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
// TAIL-CK: with at least this many per-slot checkpoints, a full set never overwrites the newest
// 'prefix' checkpoint (the tail checkpoint before a question: the next question about the same
// document resumes there); fewer keep plain oldest-first replacement.
inline constexpr std::size_t ckpt_keep_prefix_min = 3;

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
// a checkpoint (position, kind code: 1 prompt-end, 2 think-open, 3 gen-end, 4 system, 5 prefix) may be resumed
// from (COLD-RESUME: --precise accepts only states bitwise equal to a cold prefill); empty = any
using ResumePred = std::function<bool(std::uint32_t, std::uint8_t)>;

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
    virtual TierResume matchTierEntry(Tier::Entry e, Tokens toks, const ResumePred& accept = {}) const = 0;
    virtual std::uint64_t findTierShared(Tokens prefix) const = 0;  // tier id of a 1-checkpoint copy, 0 none
    // self_slot / self_tier: the requesting slot and the tier entry of its lineage (bug-s5a-cachemiss F2): when the
    // request continues one of them whole it is this session's own earlier state, not another session sharing a
    // prefix, and does not count (precise resumes at a tail checkpoint 4+ pages before it: a copy of the own
    // session would be made). npos / 0: none.
    virtual std::uint32_t lcpTarget(View<const SlotCache> slots, View<const SpeKey> spes, Tokens toks,
                                    std::uint32_t reuse, std::uint32_t sys, const FloorFn& floor,
                                    std::size_t self_slot = static_cast<std::size_t>(-1), std::uint64_t self_tier = 0) const = 0;

    // insertion / eviction
    // slot to write a checkpoint at pos: the one already at pos, else a free one, else the oldest
    // (lowest seq) -- skipping the newest 'prefix' checkpoint when cks.size() >= ckpt_keep_prefix_min
    virtual std::size_t ckptSlotFor(View<const CkptKey> cks, std::uint32_t pos) const = 0;
    virtual int sharedVictim(View<const SpeKey> spes) const = 0;
    virtual Victim evictVictim(View<const SlotCache> slots, const SlotPred& evictable, View<const SpeKey> spes) const = 0;
};

// Superseded session (CACHE-1): an agent framework's compact (history
// replaced by a summary) or compress (middle messages dropped, tool output
// cut) starts a conversation that shares only the system prompt or an early
// prefix with the old one, and is clearly shorter; the old session's tail is
// never asked for again. Retry / regenerate / edit-last diverge near the end
// (short abandoned tail) and a different system prompt shares less than the
// system message: neither supersedes.
struct SupersedeRule {
    double shorter = 0.75;           // new length <= shorter * old length
    std::uint32_t min_tail = 4096;   // abandoned old tail >= max(min_tail, tail_frac * old)
    double tail_frac = 0.25;
};
struct Supersede {
    bool yes = false;
    std::size_t keep = 0;  // shared prefix (the old entry keeps tokens [0, keep))
    // the same task lineage: cut the old entry to keep. false: it shares only the system prompt (and maybe
    // part of the first message) -- a compact, or just another conversation with the same system prompt (a
    // subagent while the main agent waits): never cut, only demoted (evicted first under pressure)
    bool cut = true;
};
// A conversation that keeps tokens [0, keep) of an old_n-token entry and continues differently diverges
// near the end (retry / regenerate / edit-last: the abandoned tail is short) -- not another branch worth
// keeping. The same threshold as supersedes().
bool nearEndDivergence(std::size_t old_n, std::size_t keep, const SupersedeRule& r = {});
// item S / bug-s5a-cachemiss: a host-tier restore at checkpoint (ck_kind, ck_pos) of an n_tok-token entry, for a
// request sharing lcp tokens with it, is a restore of a shared prefix (the slot never updates the entry in place,
// another conversation may still need it) when the checkpoint is a shared kind (system / prefix: code >= 4) --
// except when the request continues the whole entry past that checkpoint (lcp >= n_tok > ck_pos): then it is the
// same session, as a gen-end restore is (precise / COLD-RESUME never resumes at gen-end, so its next turn always
// comes back at the session's own tail checkpoint). A shared prefix entry spilled from VRAM ends at its checkpoint
// (ck_pos == n_tok) and stays shared; another question about the same document diverges inside the entry
// (lcp < n_tok) and stays a branch. (llama.cpp server_prompt_cache keeps one state per continued prompt too.)
bool restoreIsShared(std::uint8_t ck_kind, std::uint32_t ck_pos, std::uint32_t n_tok, std::size_t lcp);
// old_t / new_t: token ids of the old entry and of the new conversation;
// sys: end of the new conversation's system message (0: none / unknown -> no).
// task_end: end of the new conversation's first message after the system message (the start of the turn
// after it); the old entry is cut only when the shared prefix reaches it (an agent's compress keeps the
// system prompt and the first task message). 0: not given (cut whenever it supersedes, the CACHE-1 rule).
Supersede supersedes(Tokens old_t, Tokens new_t, std::size_t sys, const SupersedeRule& r = {}, std::size_t task_end = 0);

// Deferred supersede: the new conversation's first turn marks the old
// entries it would supersede; its second turn cuts those that were not used
// since (stamp unchanged). A parallel session of another project with the
// same system prompt keeps working on its old entry and is left alone; a
// compacted agent never returns to it.
class SupersedeTracker {
public:
    struct Pending {
        std::uint64_t old_id = 0;
        std::size_t keep = 0;
        std::uint64_t stamp = 0;
        bool cut = true;  // Supersede::cut
    };
    void mark(std::uint64_t new_id, std::uint64_t old_id, std::size_t keep, std::uint64_t stamp, bool cut = true);
    // pending marks of new_id (removed from the tracker)
    std::vector<Pending> take(std::uint64_t new_id);
    void forget(std::uint64_t id);  // an entry removed: drop its marks either way
    std::size_t size() const { return items_.size(); }

private:
    struct Item {
        std::uint64_t new_id;
        Pending p;
    };
    std::vector<Item> items_;
};

// Too short to keep in the host tiers (re-prefill is cheaper than a restore).
inline bool tooShortForTier(std::uint32_t tokens, std::uint32_t min_tokens) { return tokens < min_tokens; }

std::unique_ptr<PrefixCache> makePrefixCache(const Config& cfg);

}  // namespace whirl::cache
