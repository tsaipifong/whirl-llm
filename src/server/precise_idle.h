// PRECISE-IDLE (feat026-pidle, 0.2.6 line): the pure part of the background re-prefill of a finished reply.
// SPDX-License-Identifier: Apache-2.0
//
// Why: --precise (COLD-RESUME) resumes a conversation's next turn only from a checkpoint that a prefill produced at a
// point of the new prompt's cold chunk schedule; the end of the reply (gen-end) is not one (its history came from the
// decode / verify kernels), so the next turn re-prefills from the previous prompt's tail checkpoint: the previous
// prompt's last chunk + the whole reply + the new message (cachebal: +0.6 s R9700, +2.5 s 8060S per turn).
//
// What: while the server is idle, the engine prefills the slot's cached tokens (prompt + reply) again from a cold-exact
// checkpoint `from`, chunk by chunk on the cold schedule, up to `to` = the start of the last chunk of that schedule,
// and keeps a 'prefix' checkpoint there. The next turn then prefills at most that last chunk + its new message.
//
// The idle schedule: the cold schedule (include/whirl/prefill_schedule.h) of the n cached tokens with the forced
// splits every continuation of them shares (the system split and the image ends: functions of the tokens before them)
// and no think-open split (a next prompt's think-open split is its own last token, after these n tokens).
//
// Why it is bitwise (the checkpoint is only used where the existing COLD-RESUME rule accepts it: kind 'prefix', on the
// actual next prompt's schedule, PrefixStore::resumeOk): if `to` is a point of the idle schedule and of the next
// prompt P (same tokens before `to`, `to` < |P|), both walks cut [0, to] at the same points: below `to` their forced
// splits are the same (prefix functions); a step either is one regular chunk in both or ends at a split <= `to` in
// both, else `to` would not be a point of one of them. So the idle prefill from `from` runs exactly the chunks a cold
// prefill of P runs between `from` and `to` (merging whole chunks into one forward does not change bits, the rule the
// schedule is built on), and `from` is cold-exact by the same COLD-RESUME rule a direct resume of P there relies on.
//
// Written for WHIRL. Prior art: progress/feat026-pidle.PRIOR_ART.md (llama.cpp context checkpoints near the prompt
// end and idle-slot saving, no background compute; SGLang deterministic mode: chunk cuts aligned to a fixed size).

#pragma once

#include "cache/prefix_cache.h"
#include "whirl/prefill_schedule.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace whirl::server::pidle {

// the idle schedule of n cached tokens (sys: the system split of those tokens, 0 = none; imgs: their image-end splits)
struct Sched {
    std::size_t n = 0;
    std::size_t sys = 0;
    std::span<const std::uint32_t> imgs;
};

// the checkpoint point: the start of the idle schedule's last chunk (0 = none)
inline std::size_t target(const Sched& s) {
    return s.n < 2 ? 0 : sched::schedFloor(s.n, s.sys, 0, s.imgs, s.n - 1);
}

inline bool onSched(const Sched& s, std::size_t p) { return sched::onSchedule(s.n, s.sys, 0, s.imgs, p); }

// kinds a re-prefill may start from: produced by a prefill at a split every continuation shares ('prefix': TAIL-CK,
// LCP, image end, an earlier PRECISE-IDLE; 'system': the system split). Not prompt-end / think-open (cut at a point of
// that prompt only), not gen-end (decode history), not "restored" (unknown kind).
inline bool startKind(std::string_view k) { return k == "prefix" || k == "system"; }

struct Plan {
    int ck = -1;             // index of the checkpoint to start from (-1: nothing to do)
    std::uint32_t from = 0;  // its position
    std::uint32_t to = 0;    // the checkpoint point
    bool ok() const { return ck >= 0 && to > from; }
};

// the latest start checkpoint on the idle schedule below the checkpoint point
inline Plan plan(cache::View<const cache::CkptKey> cks, const Sched& s) {
    Plan p;
    const std::size_t t = target(s);
    if (t == 0) return p;
    for (std::size_t i = 0; i < cks.size(); ++i) {
        const cache::CkptKey& c = cks[i];
        if (!c.valid || c.pos == 0 || c.pos >= t || !startKind(c.kind) || !onSched(s, c.pos)) continue;
        if (p.ck < 0 || c.pos > p.from) {
            p.ck = static_cast<int>(i);
            p.from = c.pos;
        }
    }
    if (p.ck >= 0) p.to = static_cast<std::uint32_t>(t);
    return p;
}

// rows of the next idle forward from cur: one schedule chunk (a request that arrives waits for at most one chunk),
// never past `to`
inline std::size_t chunkLen(const Sched& s, std::size_t cur, std::size_t to) {
    const std::array<std::size_t, 1> stops = {to};
    return sched::execLen(s.n, s.sys, 0, s.imgs, cur, sched::prefill_chunk, stops);
}

}  // namespace whirl::server::pidle
