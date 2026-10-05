// Simulated VRAM cap (WHIRL_VRAM_LIMIT_MB) and the pure VRAM-fit arithmetic.
// SPDX-License-Identifier: Apache-2.0
//
// WHIRL_VRAM_LIMIT_MB=L makes a larger card behave like one with L MiB of
// VRAM (e.g. an R9700 32 GB as an RX 9070 XT 16 GB, same gfx1201 chip):
//  - hip::memInfo() reports total' = min(total, L) and free' = free - (total - total');
//    other programs' use of the GPU stays as it is on the real card;
//  - hip::wddmMemInfo() lowers the WDDM local budget by the same amount;
//  - hip::describeDevice() reports total_mem = total';
//  - hip::malloc counts this process's live device bytes and refuses an
//    allocation that would go over the cap (hip::Error, op "VramLimit",
//    hipErrorOutOfMemory), so a load that does not fit L fails as on the card.
// Unset (or 0 / not a number): no clamping, no counting.
//
// Header free of HIP: the arithmetic is unit-tested on the CPU.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace whirl::vram {

// MiB from a WHIRL_VRAM_LIMIT_MB value; 0 for empty, 0, negative or not a number.
std::uint64_t parseLimitMb(const char* s);
// Active cap in bytes (0 = none). Read once from WHIRL_VRAM_LIMIT_MB.
std::uint64_t limitBytes();
// Overrides the cap (0 turns it off); for a command-line flag / tests.
void setLimitMb(std::uint64_t mb);
// "VRAM limit 16384 MiB (WHIRL_VRAM_LIMIT_MB)", or "" when no cap is active.
std::string limitNote();

struct Clamped {
    std::uint64_t free = 0, total = 0;
};
// The simulated card: total' = min(total, limit), free' = free - (total - total') (>= 0).
// limit 0 returns the inputs unchanged.
Clamped clampMemInfo(std::uint64_t free, std::uint64_t total, std::uint64_t limit);
// WDDM local budget on the simulated card: budget - (total - min(total, limit)) (>= 0).
std::uint64_t clampBudget(std::uint64_t budget, std::uint64_t total, std::uint64_t limit);

// Live-byte counter with a cap; lock-free (relaxed atomics are enough: only the
// sum matters, never ordering against other memory).
class Counter {
public:
    // cap 0 = unlimited.
    void setCap(std::uint64_t cap) { cap_ = cap; }
    std::uint64_t cap() const { return cap_; }
    std::uint64_t live() const;
    // Adds `bytes` if live + bytes <= cap (or no cap); false (unchanged) otherwise.
    bool tryReserve(std::uint64_t bytes);
    void release(std::uint64_t bytes);

private:
    std::uint64_t cap_ = 0;
    std::atomic<std::uint64_t> live_{0};
};

// Allocation cap of this process: the simulated card's free VRAM when the
// first allocation is made (limit minus what others already used), i.e.
// clampMemInfo(free, total, limit).free at that time.
std::uint64_t processCap(std::uint64_t free_at_first, std::uint64_t total, std::uint64_t limit);

// ---- pre-load fit check and auto-shrink (server startup)
//
// The server picks the prefill batch and the number of prefix checkpoints
// before the weights are loaded; when the estimate says the KV pool left
// afterwards would be below `pool_min_tokens`, it lowers them step by step
// (batch 4096 -> 2048, checkpoints, batch -> 1024, ...) until the pool fits.
// Nothing changes when the pool already fits with the requested values.
struct FitInput {
    std::uint64_t avail = 0;          // VRAM the model may use from now on (free' minus reserve, WDDM room)
    std::uint64_t weights = 0;        // weight bytes uploaded to VRAM
    std::uint64_t fixed = 0;          // batch-independent extras (slot state, draft head, slack)
    std::uint64_t buf_per_row = 0;    // prefill buffers: bytes per batch row ...
    std::uint64_t buf_fixed = 0;      // ... plus a batch-independent part
    std::uint64_t ckpt_bytes = 0;     // one prefix checkpoint
    std::uint32_t parallel = 1;       // slots (each has n_ck checkpoints)
    std::uint64_t kv_per_token = 0;   // smallest KV format the server may pick
    std::uint32_t pool_min_tokens = 0;  // the pool wanted (tight below this)
    std::uint32_t pool_hard_min = 4096;  // below this the server refuses to start
    std::uint32_t batch = 4096;       // requested prefill batch
    std::uint32_t n_ck = 0;           // requested checkpoints per slot
    std::uint32_t n_spe = 0;          // requested shared checkpoints
    bool batch_fixed = false;         // user set the batch (WHIRL_PREFILL_BATCH): do not lower
    bool ck_fixed = false;            // user set checkpoint counts: do not lower
};
struct FitPlan {
    std::uint32_t batch = 0, n_ck = 0, n_spe = 0;
    std::uint64_t need_fixed = 0;   // weights + buffers + checkpoints + fixed with the plan
    std::uint64_t pool_tokens = 0;  // estimated pool left with the plan (smallest KV format)
    bool reduced = false;           // batch or checkpoints were lowered
    bool fits = false;              // pool_tokens >= pool_hard_min
    std::string note;               // what was reduced ("prefill batch 4096 -> 1024, ...")
};
std::uint64_t fixedNeed(const FitInput& in, std::uint32_t batch, std::uint32_t n_ck, std::uint32_t n_spe);
FitPlan planFit(const FitInput& in);

// ---- server defaults by card size
//
// A discrete card with less than 20 GiB (the 16 GB RX 9070 / 9070 XT / 9060 XT,
// or WHIRL_VRAM_LIMIT_MB below that) usually also drives the desktop and other
// programs, so the server defaults to what such a card really holds:
//  - one request slot (--parallel 1) unless --parallel was given;
//  - KV auto picks q8v (K f16, V int8) first; WHIRL_KV=f16 still forces f16;
//  - 1536 MiB of VRAM is kept free for the desktop / other apps (headroom),
//    on top of the usual reserve and WDDM margin.
// WHIRL_VRAM_HEADROOM_MB sets the headroom on any card (0 = none); an explicit
// WHIRL_POOL_RESERVE_MB (the user's own reserve) turns the default headroom off.
// Cards with 20 GiB or more (R9700 32 GB, RX 7900 XTX 24 GB) and the UMA iGPU
// keep the regular defaults (4 slots, f16 when the floor pool fits, no headroom).
inline constexpr std::uint64_t small_card_below = 20ull << 30;
inline constexpr std::uint32_t small_card_headroom_mb = 1536;

struct CardInput {
    std::uint64_t total = 0;          // describeDevice total_mem (the simulated size under WHIRL_VRAM_LIMIT_MB)
    bool uma = false;                 // integrated GPU (shared memory): never "small"
    std::optional<std::uint32_t> parallel_arg;  // --parallel / -np when given
    std::uint32_t parallel_default = 4;
    bool kv_auto = true;              // WHIRL_KV unset / auto (and a dense model)
    std::optional<std::uint32_t> headroom_mb_env;  // WHIRL_VRAM_HEADROOM_MB when set
    bool reserve_explicit = false;    // WHIRL_POOL_RESERVE_MB set
};
struct CardDefaults {
    bool small = false;
    std::uint32_t parallel = 4;
    bool parallel_auto = false;       // parallel lowered by the small-card rule
    bool prefer_q8v = false;          // KV auto: q8v before f16
    std::uint64_t headroom = 0;       // bytes kept free for the desktop / other apps
    std::string note;                 // startup log line ("" for a regular card with no headroom)
};
CardDefaults cardDefaults(const CardInput& in);

// KV format on a small card (KV auto, dense model): q8v when the floor pool (one
// full request) fits as q8v; else still q8v with a smaller pool when that pool
// holds at least small_card_q8v_min_tokens (the pool, not the slot context, then
// bounds a request: longer prompts get a clean 400); else q8h (K int8 too).
inline constexpr std::uint64_t small_card_q8v_min_tokens = 65536;
enum class SmallKv { q8v, q8v_short, q8h };
SmallKv smallCardKv(std::uint64_t avail, std::uint64_t floor_tokens, std::uint64_t q8v_bytes_per_token);

}  // namespace whirl::vram
