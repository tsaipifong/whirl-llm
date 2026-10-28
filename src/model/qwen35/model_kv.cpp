// qwen35::Model's KV pool and sequence functions (kvBytesPerTokenFmt .. restoreSeqSnapshot) over KvPool / SeqTable
// (kv.h, kv.cpp). Moved verbatim from src/model/qwen35/kv.cpp (arch025-lane2 N9 S2c): the model core, so kv.cpp does
// not include the Model class. (Before arch025-M2 they were in src/model/loader.cpp.)
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (allocKvPool, setupSeqs, snapshots).

#include "model/qwen35/kv.h"

#include "whirl/kv_layout.h"
#include "whirl/model.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

// the KV pool's device side: this Model's allocation ledger (allocZero; freed in ledger order by ~Model), freeAlloc
KvPool::Device ledger(Model& m) {
    return KvPool::Device{[&m](u64 n) { return m.allocZero(n); }, [&m](DevPtr p) { m.freeAlloc(p); }};
}

}  // namespace

// ---------------------------------------------------------------------------
// Model: KV pool and sequences (moved from src/model/loader.cpp)

u64 Model::kvBytesPerTokenFmt(KvFormat fmt) const {
    const u64 e = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    return kvLayers() * kv::layerBytes(e, fmt);  // whirl/kv_layout.h (arch024-S11c F-6)
}

const char* Model::kvName() const {
    switch (kv_fmt) {
        case KvFormat::f16: return "f16";
        case KvFormat::q4: return "q4 (4-bit + f16 scale / 32, Hadamard-rotated q/k)";
        case KvFormat::q8v: return "q8v (K f16, V int8 + f16 scale / 32)";
        case KvFormat::q8h: return "q8h (int8 + f16 scale / 32, Hadamard-rotated q/k)";
        case KvFormat::q8: break;
    }
    return "q8 (int8 + f16 scale / 32)";
}

u32 Model::kvLayers() const {
    u32 n = 0;
    for (u32 i = 0; i < cfg.n_layer; ++i)
        if (cfg.isAttn(i)) n += 1;
    if (mtp) n += 1;
    return n;
}

void Model::allocKvPool(u32 tokens) {
    const u32 pages = KvPool::allocate(cfg, mtp.has_value(), tokens, ledger(*this));
    if (seqs.empty()) {
        seq_pages = pages;
        ptab = alloc(static_cast<u64>(pages) * 4);
        std::vector<i32> idv(pages);
        for (u32 j = 0; j < pages; ++j) idv[j] = static_cast<i32>(j);
        hip::upload(ptab, idv.data(), idv.size() * 4);
        max_ctx = static_cast<u32>(static_cast<u64>(pages) * kv_page);
    }
}

void Model::freeKvPool() {
    if (pool_pages == 0) return;
    KvPool::releaseArrays(cfg, ledger(*this));
    if (seqs.empty()) {
        // allocKvPool made the single-sequence page table
        if (ptab != 0) freeAlloc(ptab);
        ptab = 0;
        seq_pages = 0;
        max_ctx = 0;
    }
    pool_pages = 0;
}

void Model::mapPages(u32 s, u32 first, std::span<const i32> phys) {
    SeqTable::mapPages(s, first, phys, [](DevPtr dst, const void* src, std::size_t n) { hip::upload(dst, src, n); });
}

void Model::setupSeqs(u32 n, u32 slot_ctx) {
    if (n == 0 || n > gdn_max_seg || pool_pages != 0 || slot_ctx < kv_page) throw ModelError("BadSlots");
    const u64 E = cfg.n_embd;
    std::vector<Seq> ss(n);
    for (u32 si = 0; si < n; ++si) {
        Seq& sq = ss[si];
        if (si == 0) {
            sq.conv = conv_state;
            sq.ssm = ssm_state;
            sq.hid = mtp_h;
            sq.kv_base = 0;
            continue;
        }
        state_tables.emplace_back(cfg.n_layer, 0);
        sq.conv = std::span<DevPtr>(state_tables.back());
        state_tables.emplace_back(cfg.n_layer, 0);
        sq.ssm = std::span<DevPtr>(state_tables.back());
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            if (ssm_state[i] == 0) continue;
            sq.conv[i] = allocZero(convBytes());
            sq.ssm[i] = allocZero(ssmBytes());
        }
        sq.hid = allocZero(static_cast<u64>(max_small_batch) * E * 4);
        sq.kv_base = si * ((slot_ctx + kv_page - 1) / kv_page);
    }
    seqs = std::move(ss);
    // replay needs the replay-mode fused segment kernels (gfx1201 code object)
    if (gdn_replay && !(fusedDecode() && k.caps.gdn_replay)) gdn_replay = false;
    if (gdn_replay) {
        gdn_ord.assign(cfg.n_layer, 0);
        u32 n_gdn = 0;
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            gdn_ord[i] = n_gdn;
            if (ssm_state[i] != 0) n_gdn += 1;
        }
        for (Seq& sq : seqs) sq.pend = alloc(static_cast<u64>(n_gdn) * pendLayerBytes());
    }
    seq_pages = (slot_ctx + kv_page - 1) / kv_page;
    ptab = allocZero(static_cast<u64>(n) * seq_pages * 4);
    out_tok = allocZero(static_cast<u64>(n) * ctl_words * 4);
    draft_samp.assign(n, spec::DraftSample{});
    draft_qbuf = mtp ? allocZero(static_cast<u64>(n) * max_drafts * spec::q_words * 4) : DevPtr{0};
    kvbase_buf = allocZero(static_cast<u64>(max_verify_rows) * 4);
    mtp_in = alloc(static_cast<u64>(max_verify_rows) * E * 4);
    max_ctx = slot_ctx;
    selectSeq(0);
}

void Model::restoreSeqSnapshot(u32 s, u32 set) {
    const Seq& sq = seqs[s];
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (sq.ssm[i] == 0) continue;
        std::swap(sq.ssm[i], snap_ssm[set][i]);
        std::swap(sq.conv[i], snap_conv[set][i]);
    }
}

}  // namespace whirl::qwen35
