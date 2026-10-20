// KV pool vs the WDDM budget after allocation (see include/whirl/pool_budget.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/pool_budget.h"

#include <algorithm>

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

}  // namespace whirl::poolbudget
