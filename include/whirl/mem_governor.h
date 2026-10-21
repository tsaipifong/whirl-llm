// Runtime memory governor (v0.2.2.2): keeps the server inside the Windows video memory (WDDM) budget when the
// budget drops while it runs (another program takes more VRAM), and gives the memory back when the budget recovers.
// SPDX-License-Identifier: Apache-2.0
//
// Owner, 2026-10-21: R9700 users do not accept slow runs. Going over the WDDM budget lets Windows page WHIRL's GPU
// buffers out of VRAM and decode becomes several times slower (rel0221: ~100-119 -> 10-43 tok/s). At startup the
// KV pool is sized under the budget with a margin (whirl/pool_budget.h); this governor handles a budget that drops
// later. It runs at the existing sampler cadence (each "batch |" status line, each request end, an idle sample;
// never in the decode loop) and acts in this order, one log line per action:
//   1. free reusable cache: idle slots' prefix cache (already spilled to the RAM tier at request end when it had
//      room) and, while a request still runs, the VRAM checkpoint buffers of idle slots and of shared prefix
//      checkpoints (returned to Windows at once; those slots take no new request until they are restored);
//   2. concurrency: while a smaller KV pool waits for the running requests to finish, new requests queue;
//   3. context for new requests: at most the smaller pool;
//   4. the KV pool is re-allocated smaller as soon as every slot is idle (this returns the VRAM to Windows);
//   5. only then, if still over the budget while a request runs, the "generation is slower" warning.
// The goal is WHIRL's own VRAM: the pinned RAM tier is system RAM, not VRAM (17 GiB pinned left the R9700 budget at
// 31.02 GiB, rel0221), so it is never shrunk for the VRAM budget (one info line per episode says so).
// It restores with hysteresis: shrink when less than margin/2 is free under the budget, back to the full margin;
// grow (when idle) only after more than margin + margin/2 has been free for recover_s seconds and at least hold_s
// after the last shrink, to margin + margin/4 free, or straight back to the startup pool once that fits with
// endMargin (3/4 of the margin) free: then the episode ends and new requests get the startup context again. hold_s
// doubles (up to hold_max_s) when a shrink follows a grow within osc_window_s, and goes back to hold_s after
// quiet_reset_s without a shrink. For trust_s after a re-allocation the local usage counts as at most what it should be (the WDDM
// counters show a free late, on the R9700 until the next memory operation), except after a grow whose measured usage is over the
// budget: that is a real overshoot (REVIEW-GOV-2 C-1: the HIP resource cache kept the freed pool), acted on at once and never
// reported as the end of the episode.
// A running request's KV is never touched: the pool changes only while every slot is idle. Output is unchanged
// (only how many tokens the pool holds changes; tokens never depend on it).
//
// Header free of HIP: the decisions are unit-tested on the CPU with injected budgets (tests/unit).

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace whirl::memgov {

struct Config {
    std::uint64_t per_tok = 0;       // KV pool bytes per token
    std::uint32_t page = 256;        // tokens per pool page
    std::uint32_t pool_start = 0;    // pool tokens at startup (it never grows beyond)
    std::uint32_t pool_min = 4096;   // smallest pool
    std::uint32_t ctx = 0;           // context per request at startup
    double recover_s = 30.0;
    double hold_s = 60.0;
    double hold_max_s = 900.0;
    double osc_window_s = 600.0;
    std::uint64_t min_grow = 1ull << 30;  // grow by at least this much (or up to pool_start)
    double settle_s = 3.0;           // no decision this soon after a re-allocation (the WDDM counters follow a free late)
    // for this long after a re-allocation the local usage counts as at most what the re-allocation should give (the
    // usage before it + the bytes it added or freed): on the R9700 the WDDM usage showed the old pool's free over several
    // seconds (a grow of 4.8 GiB first read +8.1 GiB, a shrink of 1.0 GiB read -1.8 GiB), and acting on those reads
    // shrank and grew the pool back and forth
    double trust_s = 120.0;
    // in an episode, a shrink that is not urgent (more than half the margin still free) needs at least this deficit: a
    // re-allocation drops the VRAM prefix cache, so a few pages are not worth it
    std::uint64_t min_shrink = 256ull << 20;
    // this long without an over-budget sample: the doubled hold goes back to hold_s
    double quiet_reset_s = 1800.0;
};

// what must stay free under the budget for the governor to grow straight back to the startup pool (and end the
// episode): 3/4 of the margin (startup sized the pool with the full margin; the trigger is at margin/2)
inline std::uint64_t endMargin(std::uint64_t margin) { return margin - margin / 4; }

// One sample, with the engine's state.
struct Input {
    double now_s = 0.0;               // monotonic seconds
    std::uint64_t local = 0;          // WDDM local usage of this process
    std::uint64_t budget = 0;         // WDDM local budget (0: unknown -> no action)
    std::uint64_t margin = 0;         // poolbudget::defaultMargin for this budget (what must stay free)
    std::uint32_t pool = 0;           // KV pool tokens now
    bool all_idle = false;            // no slot prefills, decodes or restores
    std::uint64_t parked = 0;         // checkpoint VRAM freed by step 1 (restored later)
    std::uint64_t freeable = 0;       // checkpoint VRAM step 1 can free now (idle slots, shared checkpoints)
};

struct Plan {
    bool episode_start = false;       // this sample started an over-budget episode
    bool free_cache = false;          // 1. free idle prefix cache and park checkpoint buffers
    bool ram_tier_note = false;       // log once that the RAM tier is untouched (system RAM, not VRAM)
    bool hold_admission = false;      // 2. new requests queue (a smaller pool waits for idle)
    std::uint32_t ctx_new = 0;        // 3. context per request for new requests
    std::optional<std::uint32_t> pool_target;  // 4. re-allocate to this many tokens when every slot is idle
    bool shrink = false;              // pool_target is smaller than the pool
    bool warn_slow = false;           // 5. over the budget with a request running
    bool restore_cache = false;       // re-allocate the parked checkpoint buffers now (they fit)
    bool episode_end = false;         // back to the startup pool, nothing parked
    std::int64_t deficit = 0;         // local + parked + margin - budget (bytes; > 0: short)
};

class Governor {
public:
    Governor() = default;
    explicit Governor(const Config& c) : cfg_(c), ctx_new_(c.ctx) {}
    Plan tick(const Input& in);
    // the engine re-allocated the pool to `pool` tokens at now_s; expected_local: the local usage it should give (the
    // usage before + the bytes added or freed; 0 = unknown)
    void resized(std::uint32_t pool, bool grew, double now_s, std::uint64_t expected_local = 0);
    // a grow the plan asked for did not fit the VRAM other programs use now (the engine checked): wait recover_s again
    void growDenied(double now_s);
    bool reduced() const { return episode_; }
    double hold() const { return hold_; }
    std::uint32_t ctxNew() const { return ctx_new_; }
    std::uint32_t shrinks() const { return n_shrink_; }
    std::uint32_t grows() const { return n_grow_; }
    const Config& config() const { return cfg_; }

private:
    std::uint32_t pagesUp(std::uint64_t bytes) const;
    std::uint32_t pagesDown(std::uint64_t bytes) const;
    Config cfg_;
    bool episode_ = false;
    double hold_ = -1.0;                  // < 0: cfg_.hold_s
    double t_shrink_ = -1e18, t_grow_ = -1e18, t_resized_ = -1e18;
    std::uint64_t expected_local_ = 0;
    bool last_grew_ = false;              // the last re-allocation was a grow
    std::optional<double> recover_since_;
    std::uint32_t ctx_new_ = 0;
    std::uint32_t n_shrink_ = 0, n_grow_ = 0;
};

// "/health" and "/props" object: the governor's effective values.
struct Status {
    bool on = false;
    bool reduced = false;
    std::uint32_t pool_tokens = 0, pool_tokens_start = 0;
    std::uint32_t ctx_per_request = 0;
    std::uint32_t slots_admitting = 0, slots = 0;
    bool admission_held = false;
    std::uint64_t parked_mib = 0;
    std::uint32_t shrinks = 0, grows = 0;
    std::string last_action;
};
std::string statusJson(const Status& s);

}  // namespace whirl::memgov
