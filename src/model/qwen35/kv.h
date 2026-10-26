// qwen35 KV pool and sequence table (arch025-M2, R-8b): qwen35::KvPool (the paged K / V arrays of every attention layer
// and the MTP layer, the KV format, the page count) and qwen35::SeqTable (the server slots: per-sequence state views,
// page-table offsets, the page table itself); kvArrays, the pool arrays in a tier page's order.
// SPDX-License-Identifier: Apache-2.0
//
// Model derives from both (include/whirl/model.h), so every reader keeps its spelling (kcache[i], pool_pages, seqs[s],
// ptab, ...). The KV format is fixed once (KvPool::fixFormat, called by Model::load right after numerics::chooseKv;
// allocate fixes it too) and read-only afterwards: kv_fmt is a const reference. The device memory behind the pointers
// stays in the Model's allocation ledger (Model::allocZero / freeAlloc through KvPool::Device): the ledger's order is
// the process's free order at teardown (alloc log, invariant I2), see progress\arch025-M2.PRIOR_ART.md.
// Same split as llama.cpp's llama_kv_cache (type_k / type_v given at construction, getters only; the cells / seq ->
// stream map apart from the tensors) and vLLM's frozen KV cache spec.

#pragma once

#include "whirl/hip.h"
#include "whirl/kv_format.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <vector>

namespace whirl::qwen35 {

using hip::DevPtr;

struct Config;  // include/whirl/model.h

// One KV pool array (every attention layer in layer order, then the MTP layer): base address and bytes per token row.
// Page p of the array is at base + p * kv_page * row. (server::KvArr is this type.)
struct KvArray {
    DevPtr base = 0;
    std::uint64_t row = 0;
};

class KvPool {
public:
    // The device side of allocate / releaseArrays: Model::allocZero and Model::freeAlloc (the Model's ledger).
    struct Device {
        std::function<DevPtr(std::uint64_t)> alloc_zero;
        std::function<void(DevPtr)> freeAlloc;  // named like Model::freeAlloc (raw_free_check: no raw .free( call sites)
    };

    KvPool() = default;
    KvPool(const KvPool&) = delete;  // kv_fmt refers to this object's fmt_
    KvPool& operator=(const KvPool&) = delete;

    // The KV format (read-only; q8 until fixFormat, the former initializer). Properties: kvQuant / kvRotated /
    // kvKf16 / kvIs4bit (whirl/kv_format.h).
    const KvFormat& kv_fmt = fmt_;
    // The one writer of the format: once, before the pool is allocated (ModelError "KvFormatFixed" otherwise).
    void fixFormat(KvFormat f);

    // per-layer arrays (0 for DeltaNet layers; sized n_layer by Model::load), the MTP layer's, and the pool size
    std::vector<DevPtr> kcache, vcache, kscale, vscale;
    DevPtr mtp_kc = 0, mtp_vc = 0, mtp_ks = 0, mtp_vs = 0;
    std::uint32_t pool_pages = 0;

    // Allocates the arrays for `tokens` tokens (whole pages; ModelError "KvPoolAllocated" if pool_pages != 0): the MTP
    // layer first (has_mtp), then every attention layer; per layer K, V, K scales (int8 K), V scales (int8 / 4-bit V).
    // Sets pool_pages last (an allocation that throws leaves it 0) and returns the page count.
    std::uint32_t allocate(const Config& cfg, bool has_mtp, std::uint32_t tokens, const Device& dev);
    // Frees every non-null array in the same order and zeroes the pointers; pool_pages is the caller's (Model::freeKvPool).
    void releaseArrays(const Config& cfg, const Device& dev);

private:
    KvFormat fmt_ = KvFormat::q8;
    bool fixed_ = false;
};

// The pool arrays of every attention layer then the MTP layer (has_mtp), in the order of a tier entry's page layout:
// K, V, K scales (q8 / q8h / q4), V scales (q8*, q4); rows from whirl/kv_layout.h (the same definition as
// Model::kvBytesPerTokenFmt, arch024-S11c F-6). Pure: reads the pointers, the format and cfg only.
std::vector<KvArray> kvArrays(const KvPool& pool, const Config& cfg, bool has_mtp);

// One sequence (a server slot): recurrent state, KV page-table offset, MTP
// hidden rows and replay buffer.
struct Seq {
    // per-layer state pointers (0 for attention layers); views into tables
    // owned by the Model, shared with Model::conv_state / ssm_state while selected
    std::span<DevPtr> conv;
    std::span<DevPtr> ssm;
    DevPtr hid = 0;               // normed trunk hidden rows feeding the next MTP step
    std::uint32_t kv_base = 0;    // offset of this sequence's page table
    DevPtr pend = 0;              // DeltaNet replay inputs (gdn_replay)
    std::uint32_t npend = 0;
};

struct SeqTable {
    std::uint32_t seq_pages = 0;  // page-table entries per sequence
    DevPtr ptab = 0;              // [sequence][seq_pages] i32 physical page ids
    std::deque<std::vector<DevPtr>> state_tables;  // backing storage of the per-sequence state views
    std::vector<Seq> seqs;
    std::uint32_t cur_seq = 0;
    std::uint32_t kv_off = 0;  // selected sequence's kv_base

    // Writes the physical page ids of pages [first, first + phys.size()) of sequence s's page table
    // (ModelError "ContextTooLong" past seq_pages).
    void mapPages(std::uint32_t s, std::uint32_t first, std::span<const std::int32_t> phys,
                  const std::function<void(DevPtr, const void*, std::size_t)>& upload) const;
};

}  // namespace whirl::qwen35
