// Runtime memory governor (see include/whirl/mem_governor.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/mem_governor.h"

#include <algorithm>
#include <format>

namespace whirl::memgov {

std::uint32_t Governor::pagesUp(std::uint64_t bytes) const {
    const std::uint64_t pb = cfg_.per_tok * cfg_.page;
    return pb ? static_cast<std::uint32_t>((bytes + pb - 1) / pb) : 0;
}

std::uint32_t Governor::pagesDown(std::uint64_t bytes) const {
    const std::uint64_t pb = cfg_.per_tok * cfg_.page;
    return pb ? static_cast<std::uint32_t>(bytes / pb) : 0;
}

Plan Governor::tick(const Input& in) {
    Plan p;
    p.ctx_new = ctx_new_;
    if (in.budget == 0 || cfg_.per_tok == 0 || cfg_.page == 0) return p;
    // just re-allocated: the WDDM usage may not show the free yet (R9700: part of a 2.7 GiB free showed ~0.3 s later)
    if (in.now_s - t_resized_ < cfg_.settle_s) return p;
    const double hold = hold_ < 0.0 ? cfg_.hold_s : hold_;
    const auto i64 = [](std::uint64_t v) { return static_cast<std::int64_t>(v); };
    // what is short for the full state (startup margin free, parked checkpoint buffers back): > 0 short
    const std::int64_t def = i64(in.local) + i64(in.parked) + i64(in.margin) - i64(in.budget);
    p.deficit = def;
    // act when less than half the margin is free under the budget
    const bool trigger = i64(in.local) + i64(in.margin / 2) > i64(in.budget);
    const std::uint32_t pool_min = (std::max(cfg_.pool_min, cfg_.page) + cfg_.page - 1) / cfg_.page * cfg_.page;
    if (trigger) {
        recover_since_.reset();
        // a grow since the last over-budget sample, and over again soon after it: the budget oscillates
        if (t_grow_ > t_shrink_ && in.now_s - t_grow_ < cfg_.osc_window_s) hold_ = std::min(hold * 2.0, cfg_.hold_max_s);
        t_shrink_ = in.now_s;
        if (!episode_) {
            episode_ = true;
            p.episode_start = true;
            p.ram_tier_note = true;
        }
        p.free_cache = true;
    }
    if (episode_) {
        if (def > 0) {
            // 5. the pool that covers the deficit (whole pages), never below the smallest pool
            const std::uint64_t cut = static_cast<std::uint64_t>(pagesUp(static_cast<std::uint64_t>(def))) * cfg_.page;
            std::uint32_t target = in.pool > cut ? static_cast<std::uint32_t>(in.pool - cut) : 0;
            target = std::max(target, std::min(pool_min, in.pool));
            if (target < in.pool) {
                p.pool_target = target;
                p.shrink = true;
            }
        } else if (!trigger) {
            // the parked checkpoint buffers fit with the full margin: back now
            if (in.parked > 0) p.restore_cache = true;
            // grow back (when idle) once more than margin + margin/2 has been free for recover_s, hold_s after the last
            // over-budget sample
            const std::int64_t band = i64(in.margin / 2);
            if (def < -band) {
                if (!recover_since_) recover_since_ = in.now_s;
            } else {
                recover_since_.reset();
            }
            if (in.all_idle && recover_since_ && in.now_s - *recover_since_ >= cfg_.recover_s && in.now_s - t_shrink_ >= hold &&
                in.pool < cfg_.pool_start) {
                const std::uint64_t add = static_cast<std::uint64_t>(pagesDown(static_cast<std::uint64_t>(-def))) * cfg_.page;
                const std::uint32_t target = static_cast<std::uint32_t>(std::min<std::uint64_t>(cfg_.pool_start, in.pool + add));
                const std::uint64_t min_add =
                    std::min<std::uint64_t>(static_cast<std::uint64_t>(pagesUp(cfg_.min_grow)) * cfg_.page, cfg_.pool_start - in.pool);
                if (target > in.pool && target - in.pool >= min_add) p.pool_target = target;
            }
            if (in.pool >= cfg_.pool_start && in.parked == 0) {
                episode_ = false;
                p.episode_end = true;
                recover_since_.reset();
            }
        }
    }
    // 3. queue new requests only while a needed shrink waits for the running ones
    p.hold_admission = p.shrink && trigger && !in.all_idle;
    // 4. new requests get at most the pool they will run in, while it is smaller than at startup (at the startup pool the
    // context is the startup one, whatever it is: normal runs are unchanged)
    const std::uint32_t pool_eff = p.shrink ? *p.pool_target : in.pool;
    ctx_new_ = pool_eff < cfg_.pool_start ? std::min(cfg_.ctx, pool_eff) : cfg_.ctx;
    p.ctx_new = ctx_new_;
    // 6. over the budget with a request running (its KV cannot move): it is slower until it finishes
    p.warn_slow = in.local > in.budget && !in.all_idle;
    return p;
}

void Governor::resized(std::uint32_t pool, bool grew, double now_s) {
    t_resized_ = now_s;
    if (grew) {
        t_grow_ = now_s;
        ++n_grow_;
    } else {
        ++n_shrink_;
    }
    recover_since_.reset();
    ctx_new_ = pool < cfg_.pool_start ? std::min(cfg_.ctx, pool) : cfg_.ctx;
}

std::string statusJson(const Status& s) {
    std::string j = std::format("{{\"on\":{},\"reduced\":{},\"pool_tokens\":{},\"pool_tokens_start\":{},\"ctx_per_request\":{},"
                                "\"slots_admitting\":{},\"slots\":{},\"admission_held\":{},\"parked_mib\":{},\"shrinks\":{},"
                                "\"grows\":{},\"last_action\":\"",
                                s.on ? "true" : "false", s.reduced ? "true" : "false", s.pool_tokens, s.pool_tokens_start,
                                s.ctx_per_request, s.slots_admitting, s.slots, s.admission_held ? "true" : "false", s.parked_mib,
                                s.shrinks, s.grows);
    for (const char ch : s.last_action) {
        if (ch == '"' || ch == '\\') j += '\\';
        if (static_cast<unsigned char>(ch) < 0x20) continue;
        j += ch;
    }
    return j + "\"}";
}

}  // namespace whirl::memgov
