// feat026-cascade: cascade (shared-prefix) decode / verify attention for the multi-slot server (FlashInfer
// arXiv 2501.01005 cascade, Hydragen arXiv 2402.05099; progress/feat026-cascade.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0
//
// When rows of several sequences map the SAME physical KV pages for their first P keys (item S shared system
// prompt pages, refcounted in the page pool), the decode attention of those rows is split into
//   prefix: keys [0, P) of all the rows in ONE multi-row group launch (the existing attn_wsplit2 / attn_wsplit1
//           on gfx1201, attn_ws4g on gfx1151: K / V of the prefix read once for every row of the group,
//           queries packed as WMMA columns), positions all set to P - 1 (no causal mask inside the prefix);
//   suffix: keys [P, pos] of each row through the normal kernel choice, with the row's page-table offset moved
//           by P / 256 pages and its position moved by -P (RoPE is already in the stored K, so only the mask
//           and the split ranges see the position);
//   merge:  attn_cascade_merge folds each row's prefix partials (LSE) into split 0 of its suffix partials, and
//           the normal attn_combine / attn_combine_q8 finishes.
// Not bitwise equal to the plain path (split ranges and merge order change): precise mode never takes it;
// balance / fast opt in with WHIRL_CASCADE=1 (acceptance = task accuracy, KL / top-1 only as guards).
//
// This header is pure host C++ (no HIP calls): the planner is inline so the mock model of the server tests
// uses the same code as the model.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace whirl::ops {

inline constexpr std::uint32_t kCascadeMaxRows = 32;  // kernels::kMaxVerifyRows
inline constexpr std::uint32_t kCascadePage = 256;    // KV_PAGE

struct CascadeConfig {
    bool on = false;                // WHIRL_CASCADE=1 (and not precise)
    std::uint32_t min_keys = 2048;  // WHIRL_CASCADE_MIN: shortest shared prefix worth a cascade (whole pages)
    std::uint32_t min_seqs = 2;     // sequences that must share it
};

// rows [first, first + count) (whole sequences, consecutive in the batch) share keys [0, keys)
struct CascadeGroup {
    std::uint32_t first = 0, count = 0, keys = 0;
};
struct CascadePlan {
    std::vector<CascadeGroup> groups;
    std::array<std::int32_t, kCascadeMaxRows> keys{};  // per row: shared prefix keys (0 = not in a group)
    bool any() const { return !groups.empty(); }
};

// kernel argument of attn_cascade_rows (kernels/attn_cascade.hip, same layout): per row the suffix position
// and page-table offset, and the prefix position (P - 1; -1 = row not in a cascade group)
struct CascadeRows {
    std::int32_t spos[kCascadeMaxRows] = {};
    std::int32_t sbase[kCascadeMaxRows] = {};
    std::int32_t ppos[kCascadeMaxRows] = {};
    // hip::LaunchArgPtrWords: no word of this kernel argument holds a device address (launch log)
    static bool launchPtrWord(std::uint32_t) { return false; }
};
static_assert(sizeof(CascadeRows) == 3 * kCascadeMaxRows * 4);

// physical pages of the sequence whose page table starts at `base` (-1 = unknown / unmapped)
using CascadePages = std::function<std::span<const std::int32_t>(std::int32_t base)>;

// leading pages two page lists have in common (same physical page, both mapped)
inline std::uint32_t cascadeSharedPages(std::span<const std::int32_t> a, std::span<const std::int32_t> b, std::uint32_t limit) {
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>({a.size(), b.size(), limit}));
    std::uint32_t i = 0;
    while (i < n && a[i] >= 0 && a[i] == b[i]) ++i;
    return i;
}

// The plan for one batch of rows (row_pos / row_base as the batched verify sets them: a sequence's rows are
// consecutive and share row_base). Greedy over the batch's sequence blocks: a run of consecutive blocks is
// extended while the pages all of them share from page 0 stay >= min_keys and the run's rows fit max_rows
// (the prefix kernel's group cap). A row's prefix ends at a whole page at or before its own position, so every
// row keeps at least its own key in the suffix. max_rows == 0: no plan.
inline CascadePlan planCascade(std::span<const std::int32_t> row_pos, std::span<const std::int32_t> row_base, const CascadePages& pages,
                               const CascadeConfig& cfg, std::uint32_t max_rows) {
    CascadePlan pl;
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>({row_pos.size(), row_base.size(), kCascadeMaxRows}));
    if (!cfg.on || max_rows < 2 || n < 2 || cfg.min_seqs < 1) return pl;
    const std::uint32_t min_pages = std::max<std::uint32_t>(1, (cfg.min_keys + kCascadePage - 1) / kCascadePage);
    struct Block {
        std::uint32_t r0, nr, cap;  // first row, rows, prefix page limit (min over its rows of pos / 256)
        std::span<const std::int32_t> pg;
    };
    std::vector<Block> bl;
    for (std::uint32_t r = 0; r < n;) {
        std::uint32_t c = 1;
        std::int32_t mn = row_pos[r];
        while (r + c < n && row_base[r + c] == row_base[r]) {
            mn = std::min(mn, row_pos[r + c]);
            ++c;
        }
        const std::uint32_t cap = mn > 0 ? static_cast<std::uint32_t>(mn) / kCascadePage : 0;
        bl.push_back(Block{r, c, cap, pages ? pages(row_base[r]) : std::span<const std::int32_t>{}});
        r += c;
    }
    auto emit = [&](std::size_t b0, std::size_t b1, std::uint32_t pg) {
        if (b1 - b0 < cfg.min_seqs || b1 - b0 < 2 || pg < min_pages) return;
        CascadeGroup g;
        g.first = bl[b0].r0;
        g.count = bl[b1 - 1].r0 + bl[b1 - 1].nr - g.first;
        g.keys = pg * kCascadePage;
        for (std::uint32_t r = g.first; r < g.first + g.count; ++r) pl.keys[r] = static_cast<std::int32_t>(g.keys);
        pl.groups.push_back(g);
    };
    std::size_t b0 = 0;
    while (b0 < bl.size()) {
        std::uint32_t common = std::min<std::uint32_t>(bl[b0].cap, static_cast<std::uint32_t>(bl[b0].pg.size()));
        std::uint32_t rows = bl[b0].nr;
        std::size_t b1 = b0 + 1;
        while (b1 < bl.size()) {
            const std::uint32_t sh = std::min(cascadeSharedPages(bl[b0].pg, bl[b1].pg, common), bl[b1].cap);
            if (sh < min_pages || rows + bl[b1].nr > max_rows) break;
            common = sh;
            rows += bl[b1].nr;
            ++b1;
        }
        emit(b0, b1, common);
        b0 = b1;
    }
    return pl;
}

// the attn_cascade_rows argument for a plan (rows outside every group: suffix = the plain row, ppos -1)
inline CascadeRows cascadeRows(const CascadePlan& pl, std::span<const std::int32_t> row_pos, std::span<const std::int32_t> row_base) {
    CascadeRows t;
    const std::size_t n = std::min<std::size_t>({row_pos.size(), row_base.size(), kCascadeMaxRows});
    for (std::size_t r = 0; r < n; ++r) {
        const std::int32_t p = pl.keys[r];
        t.spos[r] = row_pos[r] - p;
        t.sbase[r] = row_base[r] + p / static_cast<std::int32_t>(kCascadePage);
        t.ppos[r] = p > 0 ? p - 1 : -1;
    }
    return t;
}

// ---- CPU reference of the merge (attn_cascade_merge, kernels/attn_cascade.hip; keep in step) and of a
// split partial, for the unit tests. Partial = (m, l, acc[d]) with acc unnormalised against m.
struct AttnPartial {
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.f;
    std::vector<float> acc;
};
// prefix partials (its splits) folded into the suffix's split 0
inline void cascadeMergeRef(AttnPartial& s0, std::span<const AttnPartial> prefix) {
    const float ninf = -std::numeric_limits<float>::infinity();
    float mp = ninf;
    for (const AttnPartial& p : prefix) mp = std::max(mp, p.m);
    float lp = 0.f;
    for (const AttnPartial& p : prefix)
        if (p.m > ninf) lp += std::exp(p.m - mp) * p.l;
    const float M = std::max(s0.m, mp);
    if (M == ninf) return;
    const float f0 = s0.m > ninf ? std::exp(s0.m - M) : 0.f, fp = mp > ninf ? std::exp(mp - M) : 0.f;
    for (std::size_t d = 0; d < s0.acc.size(); ++d) {
        float op = 0.f;
        for (const AttnPartial& p : prefix)
            if (p.m > ninf) op += std::exp(p.m - mp) * p.acc[d];
        const float a0 = s0.m > ninf ? s0.acc[d] : 0.f;
        s0.acc[d] = a0 * f0 + op * fp;
    }
    s0.l = (s0.m > ninf ? s0.l * f0 : 0.f) + lp * fp;
    s0.m = M;
}
// attn_combine of partials (no gate)
inline std::vector<float> cascadeCombineRef(std::span<const AttnPartial> parts, std::size_t hd) {
    const float ninf = -std::numeric_limits<float>::infinity();
    float M = ninf;
    for (const AttnPartial& p : parts) M = std::max(M, p.m);
    float L = 0.f;
    for (const AttnPartial& p : parts)
        if (p.m > ninf) L += std::exp(p.m - M) * p.l;
    std::vector<float> o(hd, 0.f);
    for (std::size_t d = 0; d < hd; ++d) {
        float a = 0.f;
        for (const AttnPartial& p : parts)
            if (p.m > ninf) a += std::exp(p.m - M) * p.acc[d];
        o[d] = a / L;
    }
    return o;
}
// one split's partial of query q over keys [b0, b1) (scores q.k * scale, keys > pos masked)
inline AttnPartial cascadePartialRef(std::span<const float> q, const std::function<std::span<const float>(std::int32_t)>& key,
                                     const std::function<std::span<const float>(std::int32_t)>& val, std::int32_t b0, std::int32_t b1,
                                     std::int32_t pos, float scale) {
    AttnPartial r;
    r.acc.assign(q.size(), 0.f);
    for (std::int32_t p = b0; p < b1 && p <= pos; ++p) {
        const auto k = key(p);
        float s = 0.f;
        for (std::size_t d = 0; d < q.size(); ++d) s += q[d] * k[d];
        s *= scale;
        const float mn = std::max(r.m, s);
        const float f = r.m == -std::numeric_limits<float>::infinity() ? 0.f : std::exp(r.m - mn);
        const float e = std::exp(s - mn);
        const auto v = val(p);
        for (std::size_t d = 0; d < q.size(); ++d) r.acc[d] = r.acc[d] * f + e * v[d];
        r.l = r.l * f + e;
        r.m = mn;
    }
    return r;
}

}  // namespace whirl::ops
