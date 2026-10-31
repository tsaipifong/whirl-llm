// whirl-iso-server-slot-cache-tests: module server.slot_cache, part S6 (src/server/engine_pages.h: SlotPagesCounters,
// the shared prefix checkpoint / KV page counters; arch025-engine S6, the former Engine members stat_spe_new_ ..
// stat_evictions_ of engine.h:339-340 at 421e22b, their six writes in engine_pages.cpp:112/135/170/242/278/283 and the
// six reads in engine.cpp:184-189 (Engine::stats)) compiled with only its own header (ARCH-1 8.5 isolation).
// The checks pin what the move must keep, with expected values written out: every counter starts at 0; each write
// method changes exactly its own counter by exactly the former statement's amount (+= 1, or += the uint32 token count,
// summed in 64 bits); the six reads return them. Layout pins (DESIGN v3 2.5: cnt_ sits where stat_spe_new_ was, so
// every Engine member keeps its offset): six uint64, 8-aligned, no padding, standard layout, trivially destroyed.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_pages.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

namespace s = whirl::server;
using C = s::SlotPagesCounters;

static_assert(sizeof(C) == 6 * sizeof(std::uint64_t));
static_assert(alignof(C) == alignof(std::uint64_t));
static_assert(std::is_standard_layout_v<C>);
static_assert(std::is_trivially_destructible_v<C>);
static_assert(std::is_trivially_copyable_v<C>);
// the token count of a shared prefix checkpoint (cache::SpeKey::n) is a uint32; the former statement added it to a
// uint64 counter
static_assert(std::is_same_v<decltype(&C::noteSpeTok), void (C::*)(std::uint32_t)>);

struct Want {
    std::uint64_t spe_new = 0, spe_hits = 0, spe_tok = 0, spe_evict = 0, unshare = 0, evictions = 0;
};

void same(const C& c, const Want& w) {
    ISO_CHECK(c.speNew() == w.spe_new);
    ISO_CHECK(c.speHits() == w.spe_hits);
    ISO_CHECK(c.speTok() == w.spe_tok);
    ISO_CHECK(c.speEvict() == w.spe_evict);
    ISO_CHECK(c.unshares() == w.unshare);
    ISO_CHECK(c.evictions() == w.evictions);
}

void testStart() {
    const C c;
    same(c, {});
    const C d{};
    same(d, {});
}

void testEachWrite() {
    C c;
    c.noteSpeNew();
    same(c, {1, 0, 0, 0, 0, 0});
    c.noteSpeHit();
    same(c, {1, 1, 0, 0, 0, 0});
    c.noteSpeTok(37);
    same(c, {1, 1, 37, 0, 0, 0});
    c.noteSpeEvict();
    same(c, {1, 1, 37, 1, 0, 0});
    c.noteUnshare();
    same(c, {1, 1, 37, 1, 1, 0});
    c.noteEviction();
    same(c, {1, 1, 37, 1, 1, 1});
}

void testCounts() {
    C c;
    for (int i = 0; i < 5; ++i) c.noteSpeNew();
    for (int i = 0; i < 3; ++i) c.noteSpeHit();
    for (int i = 0; i < 4; ++i) c.noteSpeEvict();
    for (int i = 0; i < 2; ++i) c.noteUnshare();
    for (int i = 0; i < 7; ++i) c.noteEviction();
    c.noteSpeTok(0);
    c.noteSpeTok(1000);
    c.noteSpeTok(24);
    same(c, {5, 3, 1024, 4, 2, 7});
}

void testTokensIn64Bits() {
    C c;
    const std::uint32_t big = std::numeric_limits<std::uint32_t>::max();
    c.noteSpeTok(big);
    c.noteSpeTok(big);
    c.noteSpeTok(2);
    same(c, {0, 0, 8589934592ull, 0, 0, 0});   // 2 * 4294967295 + 2 = 2^33: no 32-bit wrap
}

void testCopy() {
    C c;
    c.noteSpeNew();
    c.noteSpeTok(9);
    c.noteEviction();
    const C d = c;                            // a plain value (trivially copyable): Engine::stats reads, never copies
    same(d, {1, 0, 9, 0, 0, 1});
    c.noteEviction();
    same(d, {1, 0, 9, 0, 0, 1});
    same(c, {1, 0, 9, 0, 0, 2});
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.slot_cache",
                    {{"start", testStart},
                     {"each_write", testEachWrite},
                     {"counts", testCounts},
                     {"tokens_64", testTokensIn64Bits},
                     {"copy", testCopy}},
                    argc, argv);
}
