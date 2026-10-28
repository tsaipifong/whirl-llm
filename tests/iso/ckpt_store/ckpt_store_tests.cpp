// whirl-iso-ckpt-store-tests: src/cache/ckpt_store.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 3) with its declared
// dependencies: the narrow model interface (cache/cache_model.h, contract.cache), the device contract with the
// NullBackend (host memory), and the owner check (threads: engine_thread.cpp + the server log it reports to). No
// ServerModel, no engine, no HIP layer: the checkpoint layout and the save / load copies run on host memory.
// SPDX-License-Identifier: Apache-2.0
#include "cache/ckpt_store.h"
#include "device/null_backend.h"
#include "iso/fake_cache_model.h"
#include "iso/iso_test.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace dv = whirl::device;

std::uint8_t* at(dv::DevPtr p) { return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(p)); }

void fill(dv::DevPtr p, std::uint64_t n, std::uint8_t seed) {
    for (std::uint64_t i = 0; i < n; ++i) at(p)[i] = static_cast<std::uint8_t>(seed + i);
}

bool same(dv::DevPtr a, dv::DevPtr b, std::uint64_t n) { return std::memcmp(at(a), at(b), n) == 0; }

// 4 layers, DeltaNet state in layers 1 and 3; conv 100 B, ssm 300 B, n_embd 8, n_vocab 10
constexpr std::uint64_t kConv = 100, kSsm = 300;

void testBytesAndLayout() {
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {false, true, false, true}, kConv, kSsm, 8, 10);
    sv::CkptStore st(ops, m, {kConv, kSsm, false});
    // per DeltaNet layer conv + ssm (256-byte aligned) + the hidden row (aligned) + the logits row
    ISO_CHECK(st.bytes(false) == 2 * (256 + 512) + 256 + 40);
    ISO_CHECK(st.bytes(true) == st.bytes(false));  // no WHIRL_CKPT_HOST: slot checkpoints are in VRAM too
    sv::Ckpt c;
    st.init(c, true);
    const dv::DevPtr b = c.dev.get();
    ISO_CHECK(b != 0 && c.conv.size() == 4 && c.ssm.size() == 4);
    ISO_CHECK(c.conv[0] == 0 && c.ssm[0] == 0 && c.conv[2] == 0 && c.ssm[2] == 0);
    ISO_CHECK(c.conv[1] == b && c.ssm[1] == b + 256 && c.conv[3] == b + 768 && c.ssm[3] == b + 1024);
    ISO_CHECK(c.hid == b + 1536 && c.logits == b + 1792);
    sv::CkptStore host(ops, m, {kConv, kSsm, true});
    ISO_CHECK(host.bytes(true) == 256 && host.bytes(false) == st.bytes(false));  // pinned: only the hidden row in VRAM
}

void testSaveLoadRoundTrip() {
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {false, true, false, true}, kConv, kSsm, 8, 10);
    m.cur_ = 2;
    sv::CkptStore st(ops, m, {kConv, kSsm, false});
    sv::Ckpt c;
    st.init(c, false);
    fill(m.convState()[1], kConv, 1);
    fill(m.ssmState()[1], kSsm, 2);
    fill(m.convState()[3], kConv, 3);
    fill(m.ssmState()[3], kSsm, 4);
    const dv::DevPtr hid = ops.malloc(32), logits = ops.malloc(40);
    fill(hid, 32, 5);
    fill(logits, 40, 6);
    st.save(c, hid, logits);
    ISO_CHECK(m.calls.size() == 1 && m.calls[0] == "commitSeq(2)");  // the pending replay is committed first
    ISO_CHECK(same(c.conv[1], m.convState()[1], kConv) && same(c.ssm[1], m.ssmState()[1], kSsm));
    ISO_CHECK(same(c.conv[3], m.convState()[3], kConv) && same(c.ssm[3], m.ssmState()[3], kSsm));
    ISO_CHECK(same(c.hid, hid, 32) && same(c.logits, logits, 40) && c.has_logits);
    // the model state moves on; load puts the checkpoint back (after dropping the pending replay)
    fill(m.convState()[1], kConv, 99);
    fill(m.ssmState()[3], kSsm, 98);
    st.load(c);
    ISO_CHECK(m.calls.size() == 2 && m.calls[1] == "dropPending(2)");
    std::vector<std::uint8_t> want(kConv);
    for (std::uint64_t i = 0; i < kConv; ++i) want[i] = static_cast<std::uint8_t>(1 + i);
    ISO_CHECK(std::memcmp(at(m.convState()[1]), want.data(), kConv) == 0);
    ISO_CHECK(same(m.ssmState()[3], c.ssm[3], kSsm));
    st.save(c, std::nullopt, std::nullopt);
    ISO_CHECK(!c.has_logits);
    st.release(c, {});
    ISO_CHECK(c.conv.empty() && c.ssm.empty() && c.hid == 0 && c.logits == 0 && !c.dev);
    ops.free(hid);
    ops.free(logits);
}

void testPinnedCheckpoint() {
    // WHIRL_CKPT_HOST: conv / ssm / logits in one pinned host buffer, the hidden row stays in VRAM
    dv::NullBackend ops;
    iso::FakeCacheModel m(ops, {true, false}, kConv, kSsm, 8, 10);
    sv::CkptStore st(ops, m, {kConv, kSsm, true});
    sv::Ckpt c;
    st.init(c, true);
    ISO_CHECK(c.host && !c.dev && c.hid_buf);
    ISO_CHECK(c.conv[0] == c.host.addr() && c.ssm[0] == c.host.addr() + 256 && c.logits == c.host.addr() + 768);
    ISO_CHECK(c.hid == c.hid_buf.get());
    sv::Ckpt d;
    st.init(d, false);  // not host_ok: the VRAM layout
    ISO_CHECK(d.dev && !d.host);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("ckpt_store",
                    {{"bytes_and_layout", testBytesAndLayout}, {"save_load_round_trip", testSaveLoadRoundTrip}, {"pinned_checkpoint", testPinnedCheckpoint}},
                    argc, argv);
}
