// qwen35 KV pool and sequence table (arch025-M2, R-8b), see kv.h. KvPool::allocate / releaseArrays and
// SeqTable::mapPages are the bodies of the former Model::allocKvPool / freeKvPool / mapPages (src/model/loader.cpp) with
// the device calls going through KvPool::Device / the upload argument; kvArrays is the former
// Qwen35Backend::kvArrays (src/server/backend_qwen35.cpp). The Model functions below (kvBytesPerTokenFmt ..
// restoreSeqSnapshot) moved here from loader.cpp unchanged, apart from allocKvPool / freeKvPool / mapPages calling the
// above.
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

void KvPool::fixFormat(KvFormat f) {
    if (fixed_ || pool_pages != 0) throw ModelError("KvFormatFixed", std::string("the KV format is ") + kvFormatName(fmt_));
    fmt_ = f;
    fixed_ = true;
}

u32 KvPool::allocate(const Config& cfg, bool has_mtp, u32 tokens, const Device& dev) {
    if (pool_pages != 0) throw ModelError("KvPoolAllocated");
    fixed_ = true;
    const u32 pages = (tokens + kv_page - 1) / kv_page;
    const u64 rows = static_cast<u64>(pages) * kv_page;
    const u64 elems = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    const bool q8 = kvQuant(fmt_), q4 = kvIs4bit(fmt_);
    const bool kq8 = q8 && !kvKf16(fmt_);  // K format (q8v: f16)
    // element bytes x2 (q4: one byte per two values)
    const u64 ebk = q4 ? 1 : kq8 ? 2 : 4;
    const u64 ebv = q4 ? 1 : q8 ? 2 : 4;
    // MTP KV first: the draft head reads it every step, so if the process ever ends up over
    // its WDDM budget it should not be the last (most likely demoted) allocation
    if (has_mtp) {
        mtp_kc = dev.alloc_zero(rows * elems * ebk / 2);
        mtp_vc = dev.alloc_zero(rows * elems * ebv / 2);
        if (kq8) mtp_ks = dev.alloc_zero(rows * elems / 32 * 2);
        if (q8) mtp_vs = dev.alloc_zero(rows * elems / 32 * 2);
    }
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (!cfg.isAttn(i)) continue;
        kcache[i] = dev.alloc_zero(rows * elems * ebk / 2);
        vcache[i] = dev.alloc_zero(rows * elems * ebv / 2);
        if (kq8) kscale[i] = dev.alloc_zero(rows * elems / 32 * 2);
        if (q8) vscale[i] = dev.alloc_zero(rows * elems / 32 * 2);
    }
    pool_pages = pages;
    return pages;
}

void KvPool::releaseArrays(const Config& cfg, const Device& dev) {
    auto drop = [&](DevPtr& p) {
        if (p != 0) dev.freeAlloc(p);
        p = 0;
    };
    drop(mtp_kc);
    drop(mtp_vc);
    drop(mtp_ks);
    drop(mtp_vs);
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (!cfg.isAttn(i)) continue;
        drop(kcache[i]);
        drop(vcache[i]);
        drop(kscale[i]);
        drop(vscale[i]);
    }
}

std::vector<KvArray> kvArrays(const KvPool& pool, const Config& cfg, bool has_mtp) {
    std::vector<KvArray> out;
    const std::uint64_t elems = static_cast<std::uint64_t>(cfg.n_head_kv) * cfg.head_dim;
    const kv::LayerRows L = kv::layerRows(elems, pool.kv_fmt);
    const auto layer = [&](DevPtr k, DevPtr v, DevPtr ks, DevPtr vs) {
        for (std::size_t j = 0; j < L.n; ++j) {
            const kv::Part p = L.a[j].part;
            out.push_back({p == kv::Part::k ? k : p == kv::Part::v ? v : p == kv::Part::k_scale ? ks : vs, L.a[j].bytes});
        }
    };
    for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
        if (!cfg.isAttn(i)) continue;
        layer(pool.kcache[i], pool.vcache[i], pool.kscale[i], pool.vscale[i]);
    }
    if (has_mtp) layer(pool.mtp_kc, pool.mtp_vc, pool.mtp_ks, pool.mtp_vs);
    return out;
}

void SeqTable::mapPages(u32 s, u32 first, std::span<const i32> phys, const std::function<void(DevPtr, const void*, std::size_t)>& upload) const {
    if (phys.empty()) return;
    if (first + phys.size() > seq_pages) throw ModelError("ContextTooLong");
    const u64 off = (static_cast<u64>(seqs[s].kv_base) + first) * 4;
    upload(ptab + off, phys.data(), phys.size() * 4);
}

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

KvArgs Model::kvArgs(const KvLayer& kv, DevPtr kvbase, u32 tab0) const {
    KvArgs a;
    a.k = kv.k;
    a.v = kv.v;
    a.ks = kv.ks;
    a.vs = kv.vs;
    a.ptab = ptab;
    a.kvbase = kvbase;
    a.tab0 = static_cast<i32>(tab0);
    return a;
}

void Model::reset() {
    dropPending(cur_seq);
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (conv_state[i] != 0) hip::memset(conv_state[i], 0, static_cast<std::size_t>(convBytes()));
        if (ssm_state[i] != 0) hip::memset(ssm_state[i], 0, static_cast<std::size_t>(ssmBytes()));
    }
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

u64 Model::pendLayerBytes() const {
    return pendSsmOff() + static_cast<u64>(max_small_batch) * cfg.n_v_heads * (cfg.d_state + cfg.headV() + 2) * 4;
}
u64 Model::pendSsmOff() const { return static_cast<u64>(max_small_batch) * cfg.convCh() * 4; }
u64 Model::pendPtr(const Seq& sq, std::size_t i, bool conv) const {
    const u64 base = sq.pend + static_cast<u64>(gdn_ord[i]) * pendLayerBytes();
    return conv ? base : base + pendSsmOff();
}

void Model::selectSeq(u32 s) {
    const Seq& sq = seqs[s];
    cur_seq = s;
    conv_state = sq.conv;
    ssm_state = sq.ssm;
    mtp_h = sq.hid;
    kv_off = sq.kv_base;
    ctl_off = s * ctl_words;
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
