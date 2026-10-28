// whirl-iso-tier-kv-tests: src/tier/kv_tier.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 3; module tier.kv, split from
// tier_bridge in leg 2) with its declared dependencies: the device contract with the NullBackend (host memory) and the
// platform layer (fs / proc / env). It includes no server header: no prefix store, no tier bridge, no engine, no
// model. The tests are the entry file layout arithmetic, the logical block ranges and the content hash.
// SPDX-License-Identifier: Apache-2.0
#include "device/null_backend.h"
#include "iso/iso_test.h"
#include "tier/kv_tier.h"

#include <cstdint>
#include <vector>

namespace {

namespace t = whirl::tier;

void testLayoutRegions() {
    t::Layout l;
    l.ck_bytes = 3 * 1024 * 1024;  // a checkpoint: 3 MiB -> 2 blocks of 2 MiB
    l.page_bytes = 1 << 20;
    l.page_tokens = 64;
    l.tok_cap = 1000;
    ISO_CHECK(l.ckStride() == 2 * t::block_bytes);
    ISO_CHECK(l.tokRegion() == 4096);                                  // 1000 tokens * 4 B, sector aligned
    ISO_CHECK(l.maxKvBlocks() == (16 * (1ull << 20) + t::block_bytes - 1) / t::block_bytes + 1);  // ceil(1000/64) pages
    ISO_CHECK(l.tblRegion() == 4096);                                  // 9 BlkRef of 32 B, sector aligned
    ISO_CHECK(l.dataOff() == t::header_bytes + 4096 + 4096);
    t::Entry e;
    e.nck = 2;
    e.pages = 3;
    ISO_CHECK(e.kvOff(l) == 4 * t::block_bytes && e.extent(l) == 4 * t::block_bytes + 3 * (1ull << 20));
}

void testLogicalBlockRange() {
    const auto r = t::Tier::lbRange(0, 1);
    ISO_CHECK(r[0] == 0 && r[1] == 1);
    const auto s = t::Tier::lbRange(t::block_bytes - 1, 2);  // straddles the first boundary
    ISO_CHECK(s[0] == 0 && s[1] == 2);
    const auto u = t::Tier::lbRange(3 * t::block_bytes, t::block_bytes);
    ISO_CHECK(u[0] == 3 && u[1] == 4);
    const auto z = t::Tier::lbRange(5 * t::block_bytes, 0);
    ISO_CHECK(z[0] == 5 && z[1] == 5);
}

void testBlockHash() {
    std::vector<std::uint8_t> a(4096), b(4096);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = b[i] = static_cast<std::uint8_t>(i * 31 + 7);
    ISO_CHECK(t::blockHash(a.data(), a.size()) == t::blockHash(b.data(), b.size()));
    b[4095] ^= 1;
    ISO_CHECK(t::blockHash(a.data(), a.size()) != t::blockHash(b.data(), b.size()));
    const auto h = t::blockHash(a.data(), a.size());
    ISO_CHECK(h[0] != 0 || h[1] != 0);  // (0, 0) means "unknown"
}

void testCreateOnNullBackend() {
    // the RAM tier on host memory: nothing stored, nothing in use, no SSD entries
    whirl::device::NullBackend ops;
    auto tier = t::Tier::create(ops, 8 * t::block_bytes, 42, 0);
    ISO_CHECK(tier != nullptr);
    ISO_CHECK(tier->countRam() == 0 && tier->ramUsedBytes() == 0 && tier->ssdEntries() == 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("tier.kv",
                    {{"layout_regions", testLayoutRegions},
                     {"logical_block_range", testLogicalBlockRange},
                     {"block_hash", testBlockHash},
                     {"create_on_null_backend", testCreateOnNullBackend}},
                    argc, argv);
}
