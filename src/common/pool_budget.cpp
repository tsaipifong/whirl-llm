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
    const std::uint64_t floor = (in.display ? margin_floor_display_mb : margin_floor_mb) * MiB;
    const std::uint64_t frac = up(static_cast<double>(in.local_budget) * margin_budget_frac);
    const std::uint64_t oth = in.others_known ? up(static_cast<double>(in.others) * margin_others_k) : 0;
    Margin m;
    m.bytes = std::max({floor, frac, oth});
    if (m.bytes == floor)
        m.why = std::format("{} MiB floor ({})", floor / MiB, in.display ? "a display is attached to this GPU" : "no display on this GPU");
    else if (m.bytes == oth)
        m.why = std::format("{:g} x the {} MiB other processes use on this GPU", margin_others_k, in.others / MiB);
    else
        m.why = std::format("{:g}% of the {} MiB budget", margin_budget_frac * 100.0, in.local_budget / MiB);
    if (!in.others_known) m.why += "; other processes' VRAM unknown";
    if (in.moe && moe_extra_mb > 0) {
        m.bytes += moe_extra_mb * MiB;
        m.why += std::format(" + {} MiB (MoE)", moe_extra_mb);
    }
    return m;
}

bool usageFell(std::uint64_t usage_before, std::uint64_t usage_after, std::uint64_t freed) {
    if (usage_after >= usage_before) return false;
    return usage_before - usage_after >= freed / 2;
}

}  // namespace whirl::poolbudget
