// DFlash2 batched drafting (feat029-dflashbatch, DFlash T3): one drafter forward over the draft blocks of
// several server slots instead of one forward per slot (llama.cpp common/speculative.cpp draft-dflash:
// every drafting sequence's block [anchor, MASK x (B - 1)] in one batch, each at its own positions).
// SPDX-License-Identifier: Apache-2.0
//
// Pure planning (no GPU): which requests share one forward, where each block's rows sit in the stacked
// buffers, the stacked token / position arrays, the per-slot walk outputs, the env switch and the scratch
// bytes. Model::dflashDraftBatch (src/model/dflash_batch.cpp) and the server engine use it; the unit tests
// check it against brute force.
//
// Layout of a group of S requests (block B, P = B - 1 drafts):
//   block rows   [S * B]      request i owns rows i*B .. i*B + B - 1 (row i*B = its anchor)
//   draft rows   [S * P]      request i owns rows i*P .. i*P + P - 1 (block rows i*B + 1 .. i*B + B - 1)
//   walk output  [S * (1+P)]  request i: out[i*(1+P)] = draft count, then the drafts
// Row-wise stages (embedding, norms, matmuls, RoPE with per-row positions, SwiGLU, adds, top-k) run once
// over the stacked rows; the per-block stages (block conv, attention against the slot's own ring = the
// per-sequence mask, selector lattice, walk) run per request on its rows. Every block is full size B
// whatever width the slot verifies (padding, as llama.cpp and vLLM's padded drafter batch): the width
// (feat029-dflash2) only truncates the result.
#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace whirl::dflash::batch {

using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;

// One slot's draft request (what Model::dflashDraft takes).
struct Req {
    u32 slot = 0, anchor = 0, pos = 0;
    float p_min = 0;
    u32 n_min = 0;
};

// WHIRL_DFLASH_BATCH: "1" / "on" = one drafter forward over all drafting slots of a cycle; anything else
// (unset, "0", "off") = one forward per slot (the old path, default until the GPU A/B).
inline bool parseEnabled(const char* v) {
    if (!v) return false;
    const std::string_view s(v);
    return s == "1" || s == "on" || s == "true";
}

// Requests per drafter forward: at most max_slots (the scratch dflashBatchSetup allocated) and at most
// max_rows / block (the matmul row capacity); at least 1.
inline u32 groupCap(u32 block, u32 max_slots, u32 max_rows) {
    const u32 by_rows = block > 0 ? max_rows / block : 0;
    return std::max<u32>(1, std::min(max_slots, by_rows));
}

// Consecutive groups [begin, begin + count) covering n requests in order, each <= cap.
struct Group {
    u32 begin = 0, count = 0;
};
inline std::vector<Group> groups(u32 n, u32 cap) {
    std::vector<Group> g;
    if (cap == 0) cap = 1;
    for (u32 b = 0; b < n; b += cap) g.push_back({b, std::min(cap, n - b)});
    return g;
}

// A slot may appear at most once per forward (its ring is read by its block only; two blocks of one slot
// in one cycle would be a scheduling bug). True when the slots are distinct and < n_slots.
inline bool validSlots(std::span<const Req> reqs, u32 n_slots) {
    std::vector<bool> seen(n_slots, false);
    for (const Req& r : reqs) {
        if (r.slot >= n_slots || seen[r.slot]) return false;
        seen[r.slot] = true;
    }
    return true;
}

inline u64 blockRow(u32 i, u32 t, u32 B) { return static_cast<u64>(i) * B + t; }
inline u64 draftRow(u32 i, u32 t, u32 B) { return static_cast<u64>(i) * (B - 1) + t; }
inline u64 outWord(u32 i, u32 B) { return static_cast<u64>(i) * B; }  // 1 + P = B words per request

// Stacked block tokens: [anchor_i, mask x (B - 1)] per request.
inline std::vector<i32> tokens(std::span<const Req> reqs, u32 B, u32 mask_id) {
    std::vector<i32> t(reqs.size() * static_cast<std::size_t>(B), static_cast<i32>(mask_id));
    for (std::size_t i = 0; i < reqs.size(); ++i) t[i * B] = static_cast<i32>(reqs[i].anchor);
    return t;
}

// Stacked positions: pos_i .. pos_i + B - 1 per request (RoPE reads them per row).
inline std::vector<i32> positions(std::span<const Req> reqs, u32 B) {
    std::vector<i32> p(reqs.size() * static_cast<std::size_t>(B));
    for (std::size_t i = 0; i < reqs.size(); ++i)
        for (u32 t = 0; t < B; ++t) p[i * B + t] = static_cast<i32>(reqs[i].pos + t);
    return p;
}

// Per-request drafts from the walk outputs (count clamped to [0, P], like a corrupt count never reading
// past the request's words).
inline std::vector<std::vector<u32>> decodeOut(std::span<const i32> out, u32 n, u32 B) {
    std::vector<std::vector<u32>> d(n);
    const u32 P = B > 0 ? B - 1 : 0;
    for (u32 i = 0; i < n; ++i) {
        const u64 w = outWord(i, B);
        if (w >= out.size()) break;
        const u64 c = static_cast<u64>(std::clamp<i32>(out[w], 0, static_cast<i32>(P)));
        for (u64 k = 0; k < c && w + 1 + k < out.size(); ++k) d[i].push_back(static_cast<u32>(out[w + 1 + k]));
    }
    return d;
}

// Scratch of the batched forward for `slots` blocks (dflashBatchSetup): the block buffers of
// Model::dflashSetup times slots, plus the gathered draft rows. ds = the conv projection rows
// (2 * conv_k * n_embd / conv_group, as dflash::memory).
struct MemInput {
    u32 n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, block = 0, conv_k = 0, conv_group = 1;
    u32 sel_topk = 0, sel_rank = 0, vocab = 0;
};
inline u64 scratchBytes(const MemInput& in, u32 slots) {
    if (slots < 2 || in.block == 0) return 0;  // one slot: the single-block path, no batch scratch
    const u64 S = slots, E = in.n_embd, B = in.block, P = B - 1, kvw = static_cast<u64>(in.n_head_kv) * in.head_dim;
    const u64 qw = static_cast<u64>(in.n_head) * in.head_dim, ds = 2ull * in.conv_k * (E / (in.conv_group ? in.conv_group : 1));
    const u64 K = in.sel_topk, R = in.sel_rank;
    const u64 rows = S * B, drows = S * P;
    return rows * 4 * (2 + 5 * E + ds + 2 * qw + 2 * kvw + 2ull * in.n_ff) + drows * E * 4 + drows * in.vocab * 4 + drows * K * 8 +
           drows * R * 4 + drows * K * K * 4 + S * B * 4;
}

// ---- feat029-dflashingest: batched context ingest (the verified rows' target features of several slots)
// Model::dflashIngestBatch stacks every request's rows: request i owns stacked rows off[i] .. off[i] + n_i - 1
// (off = ingestOffsets). Row-wise stages (fc over the target-layer column blocks, enc_norm, each layer's k / v
// matmul, k_norm, RoPE with the per-row positions) run once over the stacked rows; the ring writes
// (position % ring) and the cpos mirror stay per slot. llama.cpp draft-dflash process() ingests per sequence;
// vLLM's EAGLE proposer feeds the flattened multi-request hidden states in one drafter pass
// (progress\feat029-dflashingest.PRIOR_ART.md).
// WHIRL_DFLASH_BATCH_INGEST (only with WHIRL_DFLASH_BATCH=1): "0" / "off" / "false" keeps the per-slot ingest while the
// drafter forward is batched (the GPU runner's A/B arm for the ingest alone); anything else = batched (default).
inline bool parseIngestEnabled(const char* v) {
    if (!v) return true;
    const std::string_view s(v);
    return !(s == "0" || s == "off" || s == "false");
}

struct IngestReq {
    u32 slot = 0;     // drafter context slot
    u32 cap_row = 0;  // first row in the capture buffer (whirl::dflash::planIngest's cap_row)
    u32 n = 0;        // rows
    u32 pos = 0;      // position of cap_row
};

// Prefix sums of the row counts: size reqs.size() + 1, off[i] = first stacked row of request i, back() = total.
inline std::vector<u64> ingestOffsets(std::span<const IngestReq> reqs) {
    std::vector<u64> off(reqs.size() + 1, 0);
    for (std::size_t i = 0; i < reqs.size(); ++i) off[i + 1] = off[i] + reqs[i].n;
    return off;
}

// Stacked positions: pos_i .. pos_i + n_i - 1 per request.
inline std::vector<i32> ingestPositions(std::span<const IngestReq> reqs) {
    std::vector<i32> p;
    for (const IngestReq& r : reqs)
        for (u32 t = 0; t < r.n; ++t) p.push_back(static_cast<i32>(r.pos + t));
    return p;
}

// Batchable: distinct slots < n_slots and no empty request (two ingests of one slot in one batch would race on
// its ring rows; the engine ingests each slot once per cycle).
inline bool validIngest(std::span<const IngestReq> reqs, u32 n_slots) {
    std::vector<bool> seen(n_slots, false);
    for (const IngestReq& r : reqs) {
        if (r.n == 0 || r.slot >= n_slots || seen[r.slot]) return false;
        seen[r.slot] = true;
    }
    return true;
}

// Consecutive groups of requests whose rows fit cap_rows stacked rows; a request with more rows than cap_rows is a
// group of its own with single = true (it goes through the one-slot dflashIngest, which splits it).
struct IngestGroup {
    u32 begin = 0, count = 0;
    bool single = false;
};
inline std::vector<IngestGroup> ingestGroups(std::span<const IngestReq> reqs, u32 cap_rows) {
    std::vector<IngestGroup> g;
    u64 rows = 0;
    for (u32 i = 0; i < static_cast<u32>(reqs.size()); ++i) {
        const u32 n = reqs[i].n;
        if (n > cap_rows) {
            g.push_back({i, 1, true});
            rows = 0;
            continue;
        }
        if (g.empty() || g.back().single || rows + n > cap_rows) {
            g.push_back({i, 0, false});
            rows = 0;
        }
        g.back().count += 1;
        rows += n;
    }
    for (IngestGroup& x : g)
        if (x.count == 1) x.single = true;  // one request: the one-slot path (bit-identical to the unbatched engine)
    return g;
}

// Stacked ingest rows the batch scratch holds: one verify block per slot, at most the matmul row capacity.
inline u32 ingestCap(u32 block, u32 slots, u32 max_rows) {
    const u64 want = static_cast<u64>(block) * slots;
    return static_cast<u32>(std::min<u64>(want, max_rows));
}

// Scratch of the batched ingest for `rows` stacked rows: gathered features [n_target_layers][rows][n_embd],
// fc output, k / v [rows][n_head_kv * head_dim], positions.
inline u64 ingestScratchBytes(const MemInput& in, u32 n_target_layers, u32 rows) {
    if (rows == 0) return 0;
    const u64 E = in.n_embd, kvw = static_cast<u64>(in.n_head_kv) * in.head_dim;
    return static_cast<u64>(rows) * 4 * (static_cast<u64>(n_target_layers) * E + E + 2 * kvw + 1);
}

}  // namespace whirl::dflash::batch
