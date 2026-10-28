// Speculative-decoding RULES shared by the two decode loops (arch025-R12cli route B, owner 2026-10-28 「B 共用規則元件」):
// the CLI's specDecode (src/cli/decode.cpp, one sequence, q::Model single-sequence API) and the server's Engine decode
// phases (src/server/engine.cpp, A slots, batched API). The loops (device call order, KV, snapshots / replay) stay per
// front end; every rule that decides WHAT is drafted, accepted or stopped is here, once. Pure: no Model / Slot / KV type;
// inputs are histories, counters and caps the caller reads from its own state, outputs are decisions (the shape of
// llama.cpp common_speculative_n_max / common_speculative_get_output_limits; progress\arch025-r12cliB.PRIOR_ART.md).
// Draft acceptance itself is the shared sampling::Acceptor (src/sampling/acceptor.h, arch025-E1a): both loops call it.
//
// The difference table these rules settle is progress\arch025-r12cli.HANDOFF.md section 1 (D1..D15):
//   D1  relaxacc repetition guard history  -> relaxRepeats: the server rule (prompt + trunk tokens, then next + drafts)
//   D2  n-gram vs MTP score draft count    -> ngramAltDrafts + mtpScore: the server rule (previous auto pick or 3)
//   D3  n-gram draft cap                   -> ngramDraftCap + ctxRoom (each loop passes its own snapshot / row budget)
//   D4  pickDrafts cycle counter           -> caller-owned input of qwen35::pickDrafts (CLI: cycles of this call;
//                                             server: policy_cycles_ with relaxacc, else n_cycles_), unchanged
//   D6  relaxacc synthetic cycle cost      -> synthCycleMs
//   D8  context stop                       -> decodeRoom (each loop passes its own draft room)
//   D14 per-cycle MTP draft cap            -> soloDraftCap = slotDraftCap for one slot (the server's draftCap)
//   D16 WHIRL_MTP_ADAPT draft count        -> adaptDrafts (arch025-lane1; same result for one sequence)
//   D17 acceptance EMA                     -> accEma (arch025-lane1)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "spec/policy.h"       // DraftAccept, DraftTiming
#include "spec/slot_drafts.h"  // slotDraftCap
#include "whirl/model_types.h" // max_ng_drafts, max_drafts, max_small_batch
#include "whirl/relax_accept.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>

namespace whirl::spec::rules {

// D8: a decode cycle with `room` draft rows after the token at `pos` still fits in a context of `ctx` tokens.
inline bool decodeRoom(std::uint32_t pos, std::uint32_t room, std::uint32_t ctx) { return pos + room + 1 < ctx; }

// D3 / verify room: draft rows that fit before the context end after the token at `pos` (the verify writes pos ..
// pos + drafts and one more row must stay free).
inline std::uint64_t ctxRoom(std::uint32_t pos, std::uint64_t ctx) {
    return ctx > pos + 2ull ? ctx - (pos + 2ull) : 0;
}

// D3: n-gram (literal) drafts per slot for A decoding slots: the literal snapshot budget `lit_cap` and the verify rows
// `vrows` shared by the A slots (one row each for the slot's own token), --ngram-max when set, one draft without the
// fused decode (it snapshots row 0 only).
inline std::uint32_t ngramDraftCap(std::uint32_t lit_cap, std::uint32_t vrows, std::uint32_t A, std::uint32_t ngram_max,
                                   bool fused) {
    std::uint32_t lim = std::min({qwen35::max_ng_drafts, lit_cap / A, vrows / A > 0 ? vrows / A - 1 : 0u});
    if (ngram_max > 0) lim = std::min(lim, ngram_max);
    if (!fused) lim = std::min(lim, 1u);
    return lim;
}

// D14: MTP drafts per cycle of one sequence (the CLI) = the server's draft cap for one slot (Engine::draftCap ->
// slotDraftCap with batch_drafts = max_drafts, no replay, max_small_batch verify rows). drafts >= 1 (mtpOpts clamps).
inline std::uint32_t soloDraftCap(std::uint32_t drafts, bool fused) {
    return server::slotDraftCap(qwen35::max_drafts, drafts, fused, false, 1, qwen35::max_small_batch);
}

// D2: the MTP draft count an n-gram proposal is scored against: the last automatic pick (0: none yet -> 3), within the
// cycle's MTP draft cap, at least 1. It is decided BEFORE this cycle's MTP draft count.
inline std::uint32_t ngramAltDrafts(std::uint32_t prev_nd, std::uint32_t cap) {
    return std::max(1u, std::min(prev_nd > 0 ? prev_nd : 3u, cap));
}

// D2: tokens per ms of MTP with nd drafts (0 while the timing model has no estimate).
inline float mtpScore(const qwen35::DraftAccept& acc, const qwen35::DraftTiming& tm, std::uint32_t nd) {
    if (auto t = tm.estimate(nd)) return acc.expected(nd) / *t;
    return 0.0f;
}

// D1: relaxacc repetition guard. hist = the tokens the trunk holds BEFORE `next` (prompt included: a repeat of the
// prompt is a repeat), then next, dr[0..acc]: would keeping draft dr[acc] end in an n-gram already in the recent tokens?
inline bool relaxRepeats(std::span<const std::uint32_t> hist, std::uint32_t next, std::span<const std::uint32_t> dr,
                         std::uint32_t acc) {
    std::array<std::uint32_t, qwen35::max_ng_drafts + 2> tail{};
    tail[0] = next;
    for (std::uint32_t r = 0; r <= acc; ++r) tail[r + 1] = dr[r];
    return relax::extendsRepeat(hist, std::span<const std::uint32_t>(tail.data(), acc + 2));
}

// D16: WHIRL_MTP_ADAPT's draft count: the largest ceil(acc_ema + adapt) over the decoding sequences, at least 1, within
// the cycle's MTP draft cap (server: every MTP slot of the cycle; CLI: its one sequence).
// `ema(s)` reads the EMA of each element s of `seqs` (the caller's own sequence type: Slot*, a float, ...).
template <class Seqs, class Ema>
std::uint32_t adaptDrafts(const Seqs& seqs, Ema&& ema, std::uint32_t adapt, std::uint32_t cap) {
    std::uint32_t want = 1;
    for (const auto& s : seqs) want = std::max(want, static_cast<std::uint32_t>(std::ceil(ema(s) + static_cast<float>(adapt))));
    return std::min(cap, want);
}
inline std::uint32_t adaptDrafts(std::span<const float> acc_emas, std::uint32_t adapt, std::uint32_t cap) {
    return adaptDrafts(acc_emas, [](float e) { return e; }, adapt, cap);
}

// D17: the acceptance EMA a sequence keeps for D16 after a cycle that kept `acc` drafts.
inline float accEma(float ema, std::uint32_t acc) { return 0.7f * ema + 0.3f * static_cast<float>(acc); }

// D6: relaxacc's deterministic cycle cost for the draft-count policy (wall-clock jitter would change the kept tokens):
// a base per slot (a slots), + slope per extra verify row (rows = all verify rows), + 3% per sequential MTP draft step.
inline float synthCycleMs(float a, float rows, float slope, std::uint32_t steps) {
    return 100.0f * (a + slope * (rows - a) + 0.03f * static_cast<float>(steps));
}

}  // namespace whirl::spec::rules
