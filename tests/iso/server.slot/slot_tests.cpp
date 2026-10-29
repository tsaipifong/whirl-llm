// whirl-iso-server-slot-tests: the server's request slot (src/server/engine_slot.*, arch025-engine S1: Phase, SlotData,
// Slot and parkSlotCkpts, moved verbatim out of engine.h / engine_gov.cpp) built alone (ARCH-1 8.5) with its declared
// dependencies: the checkpoint store (src/cache/ckpt_store.cpp) on the host-memory NullBackend, the narrow model
// interface (cache/cache_model.h, test double tests/iso/fake_cache_model.h) and the headers of the slot's member types
// (prefix store, tier bridge, sampler, stream, spec policy, WDDM decode watch). No engine, no ServerModel, no HIP.
// Pins: parkSlotCkpts == the c1ca7ce engine_gov.cpp:145-152 body (bytes = ckpts.size() * bytes(true), the spill fence
// waited before the buffers are freed, ckpts cleared, parked set); Slot's F1 wait (destructor, move-assign) and its
// noexcept moves == the c1ca7ce engine.h:153-175 contract.
// SPDX-License-Identifier: Apache-2.0
#include "server/engine_slot.h"

#include "cache/ckpt_store.h"
#include "device/null_backend.h"
#include "iso/fake_cache_model.h"
#include "iso/iso_test.h"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace dv = whirl::device;

constexpr std::uint64_t kConv = 100, kSsm = 300;

// a store over a 4-layer model (DeltaNet in layers 1 and 3), a tier stream and a RAM block the "spill" copies into
struct Rig {
    dv::NullBackend nb;
    iso::FakeCacheModel m{nb, {false, true, false, true}, kConv, kSsm, 8, 10};
    sv::CkptStore cs{nb, m, {kConv, kSsm, false}};
    dv::Stream ts = nullptr;
    void* dst = nullptr;
    std::size_t live0 = 0;
    Rig() {
        ts = nb.streamCreateNonBlocking();
        dst = nb.hostMalloc(1 << 16);
        live0 = nb.liveAllocations();
        nb.trackPending(true);
    }
    sv::Slot slot(std::size_t n = 2) {
        sv::Slot sl;
        sl.ckpts.resize(n);
        cs.unpark(sl.ckpts, true);
        return sl;
    }
    // a spill of the slot's checkpoints on the tier stream, fenced by spill_ev (as TierBridge does)
    void spill(sv::Slot& sl) {
        for (const sv::Ckpt& c : sl.ckpts) nb.copyAsync(reinterpret_cast<dv::DevPtr>(dst), c.dev.get(), 64, ts);
        if (!sl.spill_ev) sl.spill_ev = dv::EventH::create(nb);
        nb.eventRecord(sl.spill_ev.get(), ts);
    }
    bool waited(std::uint64_t syncs0) const { return nb.pendingFreeViolations() == 0 && nb.syncCalls() > syncs0; }
    void finish() {
        nb.completeAll();
        nb.hostFree(dst);
        nb.streamDestroy(ts);
    }
};

void testParkSlotCkpts() {
    Rig r;
    sv::Slot sl = r.slot(3);
    ISO_CHECK(!sl.parked && sl.ckpts.size() == 3);
    ISO_CHECK(r.nb.liveAllocations() == r.live0 + 3);
    r.spill(sl);
    const std::uint64_t s0 = r.nb.syncCalls();
    const std::uint64_t freed = sv::parkSlotCkpts(r.cs, sl);
    // the c1ca7ce formula: three slot checkpoints of bytes(true) each
    ISO_CHECK(freed == 3 * r.cs.bytes(true));
    ISO_CHECK(r.waited(s0));  // the spill fence was waited before the buffers it reads were freed
    ISO_CHECK(sl.ckpts.empty() && sl.parked);
    ISO_CHECK(r.nb.liveAllocations() == r.live0);
    sl.spill_ev.reset();
    r.finish();
}

void testParkEmptySlot() {
    Rig r;
    sv::Slot sl;
    const std::uint64_t freed = sv::parkSlotCkpts(r.cs, sl);
    ISO_CHECK(freed == 0);  // no checkpoints: nothing freed, still marked parked
    ISO_CHECK(sl.parked && sl.ckpts.empty());
    r.finish();
}

void testSlotF1() {
    static_assert(std::is_nothrow_move_constructible_v<sv::Slot> && std::is_nothrow_move_assignable_v<sv::Slot>);
    static_assert(!std::is_copy_constructible_v<sv::Slot> && !std::is_copy_assignable_v<sv::Slot>);
    {  // destructor waits the spill fence before the checkpoint buffers are freed
        Rig r;
        auto sl = std::make_unique<sv::Slot>(r.slot());
        r.spill(*sl);
        const std::uint64_t s0 = r.nb.syncCalls();
        sl.reset();
        ISO_CHECK(r.waited(s0));
        ISO_CHECK(r.nb.liveAllocations() == r.live0);
        r.finish();
    }
    {  // move-assign over a slot with a pending spill waits, then takes the other's buffers
        Rig r;
        sv::Slot a = r.slot();
        sv::Slot b = r.slot();
        const dv::DevPtr b_dev = b.ckpts[0].dev.get();
        r.spill(a);
        const std::uint64_t s0 = r.nb.syncCalls();
        a = std::move(b);
        ISO_CHECK(r.waited(s0));
        ISO_CHECK(a.ckpts.size() == 2 && a.ckpts[0].dev.get() == b_dev);
        ISO_CHECK(r.nb.liveAllocations() == r.live0 + 2);
        a = sv::Slot{};
        ISO_CHECK(r.nb.liveAllocations() == r.live0);
        r.finish();
    }
    {  // move construction frees nothing and waits for nothing
        Rig r;
        std::vector<sv::Slot> v;
        v.push_back(r.slot());
        r.spill(v[0]);
        const std::uint64_t s0 = r.nb.syncCalls();
        v.reserve(8);  // relocation by the noexcept move constructor
        ISO_CHECK(r.nb.syncCalls() == s0 && r.nb.liveAllocations() == r.live0 + 2);
        v.clear();
        ISO_CHECK(r.waited(s0));
        r.finish();
    }
}

void testSlotDefaults() {
    // the c1ca7ce SlotData initial values the scheduler relies on
    const sv::Slot sl;
    ISO_CHECK(sl.id == 0 && sl.phase == sv::Phase::idle && sl.job == nullptr && sl.reqId() == 0);
    ISO_CHECK(sl.greedy && !sl.ignore_eos && sl.finish == "length" && !sl.done && !sl.parked);
    ISO_CHECK(sl.pend.size() == sv::max_small_batch && sl.batch.size() == sv::max_small_batch + 1);
    ISO_CHECK(sv::max_small_batch == whirl::qwen35::max_small_batch);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.slot",
                    {{"park_slot_ckpts", testParkSlotCkpts},
                     {"park_empty_slot", testParkEmptySlot},
                     {"slot_f1", testSlotF1},
                     {"slot_defaults", testSlotDefaults}},
                    argc, argv);
}
