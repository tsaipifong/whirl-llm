// whirl-iso-server-kv-fit-tests: the KV capacity planning of whirl-server's startup (src/common/pool_budget.cpp,
// src/common/startup_fit.cpp) built with only its declared dependencies (mem.fit's vram_limit.cpp, chat's json.cpp,
// platform env) (ARCH-1 8.5 isolation, arch025-lane1 leg 8). Pins the sizing rules leg 8 moved out of
// src/server/server_main.cpp against verbatim copies of the 5728296 code over value grids, plus smoke checks of the
// module's older functions (the full tables stay in whirl-tests mem_fit_tests.cpp).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/pool_budget.h"
#include "whirl/startup_fit.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace sf = whirl::startfit;

namespace {

// ---- verbatim pre-change code (5728296, src/server/server_main.cpp) ------------------------------------------------
constexpr std::uint64_t old_alignUp(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }  // whirl/small_helpers.h
constexpr std::uint32_t old_ctx_per_slot_default = 131072;  // :162
constexpr std::size_t old_n_ckpt_default = 4;               // :165
constexpr std::size_t old_n_ckpt_default_par = 3;           // :170
// :348-352 (pool_req, slot_ctx)
void old_contextPlan(std::optional<std::uint32_t> ctx, std::optional<std::uint32_t> ctx_per_slot, bool is_uma, std::uint32_t kv_page,
                     std::optional<std::uint32_t>& pool_req, std::uint32_t& slot_ctx) {
    pool_req = ctx ? ctx : (is_uma ? std::optional<std::uint32_t>(262144) : std::nullopt);
    slot_ctx = static_cast<std::uint32_t>(old_alignUp(
        ctx_per_slot ? *ctx_per_slot : (pool_req ? std::min(*pool_req, old_ctx_per_slot_default) : old_ctx_per_slot_default),
        kv_page));
}
// :457
std::size_t old_ckDef(std::uint32_t parallel) { return parallel == 1 ? old_n_ckpt_default : (parallel <= 4 ? old_n_ckpt_default_par : 1); }
// :459-462 (the text of the logW and the clamp)
std::optional<std::string> old_cap(std::size_t& n_ck, std::size_t max_ck) {
    std::optional<std::string> w;
    if (n_ck > max_ck) {
        w = std::format("WHIRL_SERVE_CKPTS={} is above the maximum ({}); using {}", n_ck, max_ck, max_ck);
        n_ck = max_ck;
    }
    return w;
}
// :898 (sfi.ck without WHIRL_CKPT_HOST) and :1049-1050 (ck_stride before the alignment)
std::uint64_t old_ck_slot(std::uint64_t st, std::uint32_t n_vocab, std::uint32_t n_embd) {
    return st + static_cast<std::uint64_t>(n_vocab) * 4 + static_cast<std::uint64_t>(n_embd) * 4;
}
std::uint64_t old_ck_ram(std::uint64_t n_gdn, std::uint64_t conv, std::uint64_t ssm, std::uint32_t n_embd, std::uint32_t V) {
    return n_gdn * (conv + ssm) + static_cast<std::uint64_t>(n_embd) * 4 + static_cast<std::uint64_t>(V) * 4;
}
// :1048-1055 (ram_def / entry_mb)
std::uint64_t old_ramDef(std::uint32_t slot_ctx, std::uint64_t kv_f16, std::size_t n_ck, std::uint64_t ck_stride, std::uint64_t auto_min,
                         std::uint64_t& entry_mb) {
    const std::uint64_t full = static_cast<std::uint64_t>(slot_ctx) * kv_f16 + n_ck * ck_stride;
    entry_mb = old_alignUp(full, 1ull << 30) >> 20;
    return std::max(auto_min, entry_mb);
}

const std::vector<std::optional<std::uint32_t>> k_ctx = {std::nullopt, 0u,      1u,      63u,     64u,     1000u,   1024u,  4096u,
                                                         65536u,       131071u, 131072u, 131073u, 200000u, 262144u, 1048576u,
                                                         4294967232u};

void testContextPlan() {
    std::size_t n = 0;
    for (const auto& ctx : k_ctx)
        for (const auto& cps : k_ctx)
            for (const bool uma : {false, true})
                for (const std::uint32_t page : {1u, 64u, 256u}) {
                    std::optional<std::uint32_t> pr;
                    std::uint32_t sc = 0;
                    old_contextPlan(ctx, cps, uma, page, pr, sc);
                    const sf::ContextPlan p = sf::contextPlan(ctx, cps, uma, page);
                    ISO_CHECK(p.pool_req == pr);
                    ISO_CHECK(p.slot_ctx == sc);
                    ++n;
                }
    ISO_CHECK(n == 16u * 16u * 2u * 3u);
    ISO_CHECK(sf::ctx_per_slot_default == 131072u && sf::uma_pool_default == 262144u);
    std::printf("context plan: %zu cases == 5728296\n", n);
}

void testCheckpoints() {
    for (std::uint32_t p = 0; p <= 40; ++p) ISO_CHECK(sf::ckptDefault(p) == old_ckDef(p));
    ISO_CHECK(sf::ckptDefault(1) == 4u && sf::ckptDefault(2) == 3u && sf::ckptDefault(4) == 3u && sf::ckptDefault(5) == 1u);
    std::size_t n = 0;
    for (const std::size_t max : {std::size_t{0}, std::size_t{1}, std::size_t{8}, std::size_t{16}})
        for (std::size_t v = 0; v <= 40; ++v) {
            std::size_t a = v;
            const std::optional<std::string> w = old_cap(a, max);
            const sf::CkptCap c = sf::capCkpts(v, max);
            ISO_CHECK(c.n == a);
            ISO_CHECK((c.warn.empty() ? std::nullopt : std::optional<std::string>(c.warn)) == w);
            ++n;
        }
    // the checkpoint bytes: the slot re-check (planSlots) and the RAM tier stride (reserveRamTier) wrote the same sum
    for (const std::uint64_t st : {0ull, 1ull, 150ull << 20, 0x123456789ull})
        for (const std::uint32_t nv : {0u, 1u, 151936u, 248320u, 0xffffffffu})
            for (const std::uint32_t ne : {0u, 5120u, 0xffffffffu}) {
                ISO_CHECK(sf::ckptBytes(st, nv, ne) == old_ck_slot(st, nv, ne));
                ++n;
            }
    for (const std::uint64_t ng : {0ull, 1ull, 48ull})
        for (const std::uint64_t cv : {0ull, 12345ull})
            for (const std::uint64_t ss : {0ull, 3ull << 20}) {
                ISO_CHECK(sf::ckptBytes(ng * (cv + ss), 248320, 5120) == old_ck_ram(ng, cv, ss, 5120, 248320));
                ++n;
            }
    std::printf("checkpoints: ckptDefault 0..40, %zu cap / bytes cases == 5728296\n", n);
}

void testRamFloor() {
    std::size_t n = 0;
    for (const std::uint32_t sc : {0u, 1024u, 131072u, 262144u})
        for (const std::uint64_t kv : {0ull, 16384ull, 65536ull})
            for (const std::size_t nck : {std::size_t{0}, std::size_t{1}, std::size_t{4}})
                for (const std::uint64_t stride : {0ull, 2ull << 20, 152ull << 20})
                    for (const std::uint64_t amin : {0ull, 8192ull}) {
                        std::uint64_t e = 0;
                        const std::uint64_t d = old_ramDef(sc, kv, nck, stride, amin, e);
                        const sf::RamFloor r = sf::ramFloor(sc, kv, nck, stride, amin);
                        ISO_CHECK(r.entry_mb == e && r.floor_mb == d);
                        ++n;
                    }
    std::printf("ram floor: %zu cases == 5728296\n", n);
}

// the module's older functions, linked from the isolated objects (smoke: the tables are in whirl-tests)
void testSmoke() {
    ISO_CHECK(sf::slotsThatFit(1, 0, 0, 0, 1, 1) == 1u);
    ISO_CHECK(sf::poolTokensWith(1, 1000, 0, 0, 10) == 100u);
    whirl::poolbudget::RoomIn ri;
    ri.free = 8ull << 30;
    ri.margin = 0;
    ISO_CHECK(whirl::poolbudget::kvRoom(ri).bytes <= ri.free);
    ISO_CHECK(sf::mmprojPinnedBytes(1ull << 30) >= (1ull << 30));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.kv_fit", {{"context_plan", testContextPlan}, {"checkpoints", testCheckpoints}, {"ram_floor", testRamFloor}, {"smoke", testSmoke}},
                    argc, argv);
}
