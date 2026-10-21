// KV pool vs the WDDM budget after allocation (see include/whirl/pool_budget.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/pool_budget.h"

#include <algorithm>
#include <format>

namespace whirl::poolbudget {

Trim fitPool(std::uint32_t pool_tokens, std::uint64_t local_usage, std::uint64_t local_budget, std::uint64_t margin,
             std::uint64_t per_tok, std::uint32_t page, std::uint32_t min_tokens) {
    Trim t;
    t.pool_tokens = pool_tokens;
    if (local_budget == 0 || per_tok == 0 || page == 0) return t;
    if (local_usage + margin <= local_budget) return t;
    t.over = local_usage + margin - local_budget;
    const std::uint64_t page_bytes = per_tok * page;
    const std::uint64_t drop_pages = (t.over + page_bytes - 1) / page_bytes;
    const std::uint64_t have_pages = pool_tokens / page;
    // the smallest pool kept, rounded up to whole pages
    const std::uint64_t min_pages = (static_cast<std::uint64_t>(min_tokens) + page - 1) / page;
    std::uint64_t keep = have_pages > drop_pages ? have_pages - drop_pages : 0;
    if (keep < min_pages) {
        keep = std::min(min_pages, have_pages);
        t.at_floor = true;
    }
    t.pool_tokens = static_cast<std::uint32_t>(keep * page);
    t.shrink = t.pool_tokens < pool_tokens;
    return t;
}

Margin defaultMargin(const MarginIn& in) {
    constexpr std::uint64_t MiB = 1ull << 20;
    const auto up = [](double b) { return (static_cast<std::uint64_t>(b) + MiB - 1) / MiB * MiB; };
    Margin m;
    // room for growth
    if (in.room_override) {
        m.room = *in.room_override / MiB * MiB;
        m.why = std::format("{} MiB room (WHIRL_POOL_BUDGET_MARGIN_MB)", m.room / MiB);
    } else {
        const std::uint64_t floor = (in.display ? margin_floor_display_mb : margin_floor_mb) * MiB;
        const std::uint64_t frac = up(static_cast<double>(in.local_budget) * margin_budget_frac);
        m.room = std::max(floor, frac);
        m.why = m.room == floor ? std::format("{} MiB room ({})", floor / MiB, in.display ? "a display is attached to this GPU" : "no display on this GPU")
                                : std::format("{} MiB room (6% of the {} MiB budget)", m.room / MiB, in.local_budget / MiB);
        if (in.moe && moe_extra_mb > 0) {
            m.room += moe_extra_mb * MiB;
            m.why += std::format(" + {} MiB (MoE)", moe_extra_mb);
        }
    }
    // other processes' usage at startup that the budget does not show yet
    if (in.others_known) {
        const std::uint64_t start = in.budget_start ? in.budget_start : in.local_budget;
        const std::uint64_t dropped = start > in.local_budget ? start - in.local_budget : 0;
        m.pending = in.others > dropped ? up(static_cast<double>(in.others - dropped)) : 0;
        m.why += std::format(" + {} MiB of the {} MiB other processes use that the budget does not show yet", m.pending / MiB,
                             up(static_cast<double>(in.others)) / MiB);
        if (dropped > 0) m.why += std::format(" (it dropped {} MiB since startup)", dropped / MiB);
    } else {
        m.why += "; other processes' VRAM unknown";
    }
    m.bytes = m.pending + m.room;
    return m;
}

bool usageFell(std::uint64_t usage_before, std::uint64_t usage_after, std::uint64_t freed) {
    if (usage_after >= usage_before) return false;
    return usage_before - usage_after >= freed / 2;
}

// fbdbe38 behaviour, kept for the failing test (REVIEW-GOV-3 question 4): the encoder is not counted
std::uint64_t visionReserve(const VisionVram& v) {
    (void)v;
    return 0;
}

std::uint64_t kvAvail(std::uint64_t avail, std::uint64_t reserve) { return avail > reserve ? avail - reserve : 0; }

}  // namespace whirl::poolbudget
