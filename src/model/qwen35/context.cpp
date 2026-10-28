// qwen35::Context members defined outside the forward sources (arch025-lane2 N9 S4): the device-memory ledger
// (alloc / allocZero / freeAlloc), the recurrent-state sizes and snapshot sets, layerMats (moved verbatim from
// src/model/loader.cpp), and the sequence selection / KV arguments / replay offsets / reset the forward calls
// (moved verbatim from src/model/qwen35/model_kv.cpp, before arch025-lane2 kv.cpp). Only Model:: became Context::.
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen35/context.h"

#include "device/backend.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

}  // namespace

// ---------------------------------------------------------------------------
// from src/model/loader.cpp

DevPtr Context::alloc(u64 bytes) {
    device::DevBuf b = device::DevBuf::alloc(device::hipBackend(), bytes);
    const DevPtr p = b.get();
    allocations.push_back(std::move(b));
    alloc_total += bytes;
    return p;
}

DevPtr Context::allocZero(u64 bytes) {
    const DevPtr p = alloc(bytes);
    hip::memset(p, 0, static_cast<std::size_t>(bytes));
    return p;
}

void Context::freeAlloc(DevPtr p) {
    for (std::size_t i = 0; i < allocations.size(); ++i)
        if (allocations[i].get() == p) {
            device::DevBuf gone = std::move(allocations[i]);
            if (i + 1 != allocations.size()) allocations[i] = std::move(allocations.back());
            allocations.pop_back();
            alloc_total -= gone.bytes();  // WDDM-LOG review P3-8: alloc_total stays the live sum
            gone.reset();
            return;
        }
}

u64 Context::convBytes() const { return static_cast<u64>(cfg.d_conv - 1) * cfg.convCh() * 4; }

u64 Context::ssmBytes() const { return static_cast<u64>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4; }

void Context::ensureSnapshots(u32 n) {
    const u32 want = std::min(n, gdn_max_snap);
    while (snap_sets < want) {
        const u32 kk = snap_sets;
        std::vector<DevPtr> conv(cfg.n_layer, 0), ssm(cfg.n_layer, 0);
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            if (ssm_state[i] == 0) continue;
            conv[i] = allocZero(convBytes());
            ssm[i] = allocZero(ssmBytes());
        }
        snap_conv[kk] = std::move(conv);
        snap_ssm[kk] = std::move(ssm);
        snap_sets += 1;
    }
}

MatList Context::layerMats(const Layer& L) const {
    MatList out;
    if (L.kind == LayerKind::attn) {
        out.add(L.attn.q);
        out.add(L.attn.k);
        out.add(L.attn.v);
        out.add(L.attn.o);
    } else {
        out.add(L.gdn.qkv);
        out.add(L.gdn.gate);
        out.add(L.gdn.beta);
        out.add(L.gdn.alpha);
        out.add(L.gdn.out);
    }
    out.add(L.ffn_gate);
    out.add(L.ffn_up);
    out.add(L.ffn_down);
    return out;
}

// ---------------------------------------------------------------------------
// from src/model/qwen35/model_kv.cpp

KvArgs Context::kvArgs(const KvLayer& kv, DevPtr kvbase, u32 tab0) const {
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

void Context::reset() {
    dropPending(cur_seq);
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (conv_state[i] != 0) hip::memset(conv_state[i], 0, static_cast<std::size_t>(convBytes()));
        if (ssm_state[i] != 0) hip::memset(ssm_state[i], 0, static_cast<std::size_t>(ssmBytes()));
    }
}

u64 Context::pendLayerBytes() const {
    return pendSsmOff() + static_cast<u64>(max_small_batch) * cfg.n_v_heads * (cfg.d_state + cfg.headV() + 2) * 4;
}

u64 Context::pendSsmOff() const { return static_cast<u64>(max_small_batch) * cfg.convCh() * 4; }

u64 Context::pendPtr(const Seq& sq, std::size_t i, bool conv) const {
    const u64 base = sq.pend + static_cast<u64>(gdn_ord[i]) * pendLayerBytes();
    return conv ? base : base + pendSsmOff();
}

void Context::selectSeq(u32 s) {
    const Seq& sq = seqs[s];
    cur_seq = s;
    conv_state = sq.conv;
    ssm_state = sq.ssm;
    mtp_h = sq.hid;
    kv_off = sq.kv_base;
    ctl_off = s * ctl_words;
}

// ---------------------------------------------------------------------------
// from src/model/config.cpp (arch025-lane2 leg 3: the forward links them; declared in context.h)

std::size_t tuneBucket(std::uint32_t n) {
    if (n <= 32) return 3;
    if (n <= 48) return 4;
    if (n <= 64) return 5;
    if (n <= 96) return 6;
    if (n <= 128) return 2;
    if (n <= 192) return 7;
    if (n <= 256) return 8;
    if (n <= 384) return 9;
    if (n <= 768) return 0;
    if (n <= tune_small_max) return 10;
    return 1;
}

const char* opClassName(OpClass c) {
    static const char* names[] = {"embed", "norm", "quant", "matmul", "attn", "gdn_prep", "gdn_rec", "gdn_norm", "misc", "head"};
    return names[static_cast<int>(c)];
}

// ---------------------------------------------------------------------------
// profiling (from src/model/loader.cpp, arch025-lane2 leg 3)

Profile::Profile() : events(cap), classes(cap, OpClass::misc) {}
Profile::~Profile() {
    for (device::EventH& e : events) e.reset();  // in index order, as before arch024 S11
}
void Profile::record(hip::Stream s, OpClass cls) {
    if (count == cap) flush();
    if (!events[count]) events[count] = device::EventH::adopt(device::hipBackend(), hip::eventCreate());  // timing event
    hip::eventRecord(events[count].get(), s);
    classes[count] = cls;
    count += 1;
}
void Profile::flush() {
    if (count == 0) return;
    hip::sync();
    for (std::size_t i = 1; i < count; ++i) ms[static_cast<std::size_t>(classes[i])] += hip::eventElapsedMs(events[i - 1].get(), events[i].get());
    std::swap(events[count - 1], events[0]);
    classes[0] = classes[count - 1];
    count = 1;
}
void Profile::reset() {
    ms.fill(0);
    matmul_bytes = 0;
    count = 0;
}

}  // namespace whirl::qwen35
