// qwen35 KV pool and sequence table (arch025-M2, R-8b), see kv.h. KvPool::allocate / releaseArrays and
// SeqTable::mapPages are the bodies of the former Model::allocKvPool / freeKvPool / mapPages (src/model/loader.cpp) with
// the device calls going through KvPool::Device / the upload argument; kvArrays is the former
// Qwen35Backend::kvArrays (src/model/qwen35/server_model.cpp). The Model functions that used to follow
// (kvBytesPerTokenFmt .. restoreSeqSnapshot) are in src/model/qwen35/model_kv.cpp (arch025-lane2 N9 S2c).
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (allocKvPool, setupSeqs, snapshots).

#include "model/qwen35/kv.h"

#include "model/qwen35/model_error.h"
#include "whirl/kv_layout.h"
#include "whirl/model_types.h"  // Config, kv_page

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

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

}  // namespace whirl::qwen35
