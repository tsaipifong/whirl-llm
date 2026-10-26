// Per-slot MTP draft allocation run by the server engine (moved from src/server/engine.cpp by arch025-E1b; the
// bodies are the engine's, with Slot::dacc / opt_ / m_ reads turned into parameters).
// SPDX-License-Identifier: Apache-2.0

#include "spec/slot_drafts.h"

#include <algorithm>
#include <optional>

namespace whirl::server {

std::vector<std::uint32_t> planSlotDrafts(std::span<const DraftPlanSlot> slots, const qwen35::CycleCost& cost,
                                          std::uint32_t uniform, std::uint32_t cap, std::uint32_t rows, std::uint32_t snaps) {
    std::vector<qwen35::AllocSlot> in(slots.size());
    std::vector<std::uint32_t> prev(slots.size(), 0);
    bool have_prev = true;
    for (std::size_t k = 0; k < slots.size(); ++k) {
        const DraftPlanSlot& s = slots[k];
        in[k].acc = s.acc;
        in[k].cap = cap;
        in[k].ctx_k = static_cast<float>(s.pos) / 1024.0f;
        in[k].ng_rows = s.ng_rows;
        in[k].ng_e = s.ng_e;
        if (s.ng_rows == 0) {
            prev[k] = std::min(s.d_prev, cap);
            have_prev = have_prev && s.d_prev > 0;
        }
    }
    if (!have_prev) prev.clear();
    const qwen35::AllocResult r =
        qwen35::allocDrafts(in, cost, qwen35::AllocBudget{rows, snaps}, uniform, std::span<const std::uint32_t>(prev));
    std::vector<std::uint32_t> out;
    for (std::size_t k = 0; k < slots.size(); ++k)
        if (slots[k].ng_rows == 0) out.push_back(r.d[k]);
    return out;
}

void splitDraftsByGain(std::span<const qwen35::DraftAccept* const> acc, std::uint32_t nd, std::uint32_t rows,
                       std::uint32_t cap, std::span<std::uint32_t> nd_m) {
    std::uint32_t used = nd * static_cast<std::uint32_t>(acc.size());
    float e_tot = 0;
    for (const qwen35::DraftAccept* a : acc) e_tot += a->expected(nd);
    const float step_frac = 0.03f;
    for (std::uint32_t it = 0; it < 16; ++it) {
        std::uint32_t mx = 0;
        for (auto x : nd_m) mx = std::max(mx, x);
        // best receiver: largest marginal of its next draft
        std::optional<std::size_t> r_best;
        float g_best = 0;
        for (std::size_t k = 0; k < acc.size(); ++k) {
            if (nd_m[k] >= cap) continue;
            const float g = acc[k]->expected(nd_m[k] + 1) - acc[k]->expected(nd_m[k]);
            if (!r_best || g > g_best) {
                r_best = k;
                g_best = g;
            }
        }
        if (!r_best) break;
        const std::size_t r = *r_best;
        const bool needs_step = nd_m[r] + 1 > mx;
        const float bar = needs_step ? step_frac * e_tot : 0;
        if (used < rows) {
            // a spare row: worth it when the draft is likely enough
            if (g_best > 0.35f && g_best > bar) {
                nd_m[r] += 1;
                used += 1;
                e_tot += g_best;
                continue;
            }
        }
        // best donor: smallest marginal of its last draft (not the receiver)
        std::optional<std::size_t> d_best;
        float l_best = 0;
        for (std::size_t k = 0; k < acc.size(); ++k) {
            if (k == r || nd_m[k] == 0) continue;
            const float l = acc[k]->expected(nd_m[k]) - acc[k]->expected(nd_m[k] - 1);
            if (!d_best || l < l_best) {
                d_best = k;
                l_best = l;
            }
        }
        if (!d_best) break;
        const std::size_t d = *d_best;
        if (g_best - l_best <= bar || g_best - l_best < 0.02f) break;
        nd_m[r] += 1;
        nd_m[d] -= 1;
        e_tot += g_best - l_best;
    }
}

std::uint32_t slotDraftCap(std::uint32_t batch_drafts, std::uint32_t n_draft, bool fused, bool replay, std::uint32_t A,
                           std::uint32_t vrows) {
    std::uint32_t nd = std::min(batch_drafts, n_draft);
    if (!fused) nd = std::min(nd, 1u);
    while (nd > 0 && (A * (nd + 1) > vrows || (!replay && A * nd > qwen35::gdn_max_snap))) nd -= 1;
    return nd;
}

}  // namespace whirl::server
