// qwen35::Context members defined outside the forward sources (arch025-lane2 N9 S4): the device-memory ledger
// (alloc / allocZero / freeAlloc), the recurrent-state sizes and snapshot sets, layerMats (moved verbatim from
// src/model/loader.cpp), and the sequence selection / KV arguments / replay offsets / reset the forward calls
// (moved verbatim from src/model/qwen35/model_kv.cpp, before arch025-lane2 kv.cpp). Only Model:: became Context::.
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen35/context.h"

#include "device/backend.h"

#include <algorithm>
#include <array>
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

// v0252-f1arg3: the multi-block verify argmax switch and its per-Context workspace (see the header)
int Context::argmaxMbMode(const device::Profile& prof, bool moe, bool fast, const std::optional<std::string>& env) {
    if (!env) return fast && (moe ? prof.argmax_mb_moe : prof.argmax_mb_dense) ? 1 : 0;
    if (*env == "0") return 0;
    return *env == "2" ? 2 : 1;
}

void Context::setArgmaxMb(int mode) { argmax_mb = k.argmax_rows_to_mb != nullptr && k.argmax_rank_rows_to_mb != nullptr ? mode : 0; }

DevPtr Context::allocCtl(u64 bytes) {
    if (argmax_mb == 0) return allocZero(bytes);
    // one allocation: [workspace am_ws_bytes][control words]; both zeroed here, once, before any launch on `stream`
    am_ws = allocZero(am_ws_bytes + bytes);
    return am_ws + am_ws_bytes;
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

// v0252-e1s8 (E1 in snapshot mode): the spare set of the select verifies, like one more snapshot set
void Context::ensureSelectSpare() {
    if (e1_conv.size() == cfg.n_layer) return;
    std::vector<DevPtr> conv(cfg.n_layer, 0), ssm(cfg.n_layer, 0);
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (ssm_state[i] == 0) continue;
        conv[i] = allocZero(convBytes());
        ssm[i] = allocZero(ssmBytes());
    }
    e1_conv = std::move(conv);
    e1_ssm = std::move(ssm);
}

// v0252-e1s8: the host settle of a select verify. Its device rule is kernels/gdn_select.h (gdn_sel_pick / gdn_sel_out, what
// the gfx1151 kernels and the server tests' mock run; that header's gdn_sel_settle is this renaming - written out here because
// the forward.* modules do not include kernel headers; tests/iso/forward.gdn settle_select pins it to the rule). The select
// verify of sequence s read candidate c = acc of P = [snapshot sets 0 .. m - 1, live] (m = Seq::sel_cm), wrote its snapshot r
// into P[r < c ? r : r + 1] and its final state into the spare set: renamed, set r <- P[r < c ? r : r + 1], live <- spare,
// spare <- P[c] - its outputs then sit where an in-place verify leaves them
void Context::settleSelect(u32 s, u32 acc) {
    if (s >= seqs.size()) throw ModelError("BadBatch");
    Seq& sq = seqs[s];
    if (sq.sel_cm < 0 || e1_conv.size() != sq.conv.size() || static_cast<u32>(sq.sel_cm) > snap_sets) throw ModelError("NothingToSettle");
    const u32 m = static_cast<u32>(sq.sel_cm);
    const u32 c = std::min(acc, m);
    for (std::size_t i = 0; i < sq.conv.size(); ++i) {
        if (sq.ssm[i] == 0) continue;
        const auto cand = [&](u32 j, bool conv) -> DevPtr {
            return j < m ? (conv ? snap_conv[j][i] : snap_ssm[j][i]) : (conv ? sq.conv[i] : sq.ssm[i]);
        };
        std::array<DevPtr, gdn_max_snap + 1> pc{}, ps{};
        for (u32 j = 0; j <= m; ++j) {
            pc[j] = cand(j, true);
            ps[j] = cand(j, false);
        }
        for (u32 r = 0; r < m; ++r) {
            snap_conv[r][i] = pc[r < c ? r : r + 1];
            snap_ssm[r][i] = ps[r < c ? r : r + 1];
        }
        sq.conv[i] = e1_conv[i];
        sq.ssm[i] = e1_ssm[i];
        e1_conv[i] = pc[c];
        e1_ssm[i] = ps[c];
    }
    sq.sel_cm = -1;
    sq.held = true;
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
