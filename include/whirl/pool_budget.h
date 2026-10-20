// KV pool vs the WDDM video memory budget after the pool is allocated (issue #5).
// SPDX-License-Identifier: Apache-2.0
//
// The server sizes the KV pool from the WDDM local budget it reads before allocating the pool. That
// budget can be lower a moment later: in issue #5 (R9700, 12 GiB RAM tier) it read 31.02 GiB when the
// pool was sized and 29.55 GiB right after the pool was allocated, so the server started 0.37 GiB over
// the budget and Windows began moving its GPU memory to system RAM. The server therefore reads the
// budget again once the pool exists and shrinks the pool (whole pages) until local usage + margin fits.
//
// Header free of HIP: the arithmetic is unit-tested on the CPU with injected budget values.

#pragma once

#include <cstdint>

namespace whirl::poolbudget {

struct Trim {
    std::uint32_t pool_tokens = 0;  // the pool to keep (== the input when nothing changes)
    bool shrink = false;            // pool_tokens < the input
    bool at_floor = false;          // the fit needs a pool below min_tokens: kept at min_tokens, still over
    std::uint64_t over = 0;         // local + margin - budget before the shrink (0: fits)
};

// local_usage / local_budget: WDDM numbers read with the pool allocated; margin: what must stay free
// under the budget; per_tok: KV bytes per token; page: tokens per page; min_tokens: smallest pool kept.
// Freed bytes are assumed to lower the local usage by exactly their size (WDDM counts a few percent
// more per allocation, so the real fit is a little better). local_budget 0 (unknown): no change.
Trim fitPool(std::uint32_t pool_tokens, std::uint64_t local_usage, std::uint64_t local_budget, std::uint64_t margin,
             std::uint64_t per_tok, std::uint32_t page, std::uint32_t min_tokens);

}  // namespace whirl::poolbudget
