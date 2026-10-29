// whirl-iso-server-kv-fit-tests: the KV capacity planning of whirl-server's startup (src/common/pool_budget.cpp,
// src/common/startup_fit.cpp) built with only its declared dependencies (mem.fit's vram_limit.cpp, chat's json.cpp,
// platform env) (ARCH-1 8.5 isolation, arch025-lane1 leg 8). Pins the sizing rules leg 8 moved out of
// src/server/server_main.cpp against verbatim copies of the 5728296 code over value grids, plus smoke checks of the
// module's older functions (the full tables stay in tests/unit/server_kv_fit_tests.cpp, in whirl-tests and whirl-unit-server-kv-fit-tests).
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

// ---- leg 9 (arch025-lane1): the rules left in 729c080 src/server/server_main.cpp, verbatim copies ------------------
// a config with qwen35::Config's isAttn (include/whirl/model_types.h:101, verbatim) and n_layer
struct FakeCfg {
    std::uint32_t n_layer = 0;
    std::uint32_t interval = 1;
    bool isAttn(std::uint32_t il) const { return (il + 1) % interval == 0; }
};
// :886-888 (planSlots ng); makeEngine :941-943 counted model.ssm_state[li] != 0, which Model::load sets exactly for
// !cfg.isAttn(li) (src/model/loader.cpp:318-322; HANDOFF section 48)
std::uint64_t old_ng(const FakeCfg& cfg) {
    std::uint64_t ng = 0;
    for (std::uint32_t li = 0; li < cfg.n_layer; ++li)
        if (!cfg.isAttn(li)) ++ng;
    return ng;
}
void testDeltaNetLayers() {
    std::size_t n = 0;
    for (std::uint32_t nl = 0; nl <= 96; ++nl)
        for (std::uint32_t iv = 1; iv <= 9; ++iv) {
            const FakeCfg c{nl, iv};
            ISO_CHECK(sf::deltaNetLayers(c) == old_ng(c));
            ++n;
        }
    ISO_CHECK(sf::deltaNetLayers(FakeCfg{64, 4}) == 48u);  // Qwen3.8-27B: 64 layers, attention every 4th
    std::printf("deltanet layers: %zu cases == 729c080\n", n);
}
// :1086-1094 (reserveRamTier: the per-GPU text of the "kv tier: pinned memory now:" line)
std::string old_per(const std::vector<sf::PinnedGpu>& gs) {
    std::string per;
    for (const sf::PinnedGpu& g0 : gs) {
        struct {
            std::uint64_t nonlocal_budget, shared_system, adapter_shared;
        } g{g0.nonlocal_budget, g0.shared_system, g0.adapter_shared};
        const std::uint64_t own = g0.own_nonlocal;
        per += std::format("{}{} MiB {} (this process {} MiB, all programs {})", per.empty() ? "" : ", ",
                           (g.nonlocal_budget > 0 ? g.nonlocal_budget : g.shared_system) >> 20,
                           g.nonlocal_budget > 0 ? "non-local budget" : "shared GPU memory", own >> 20,
                           g.adapter_shared ? std::format("{} MiB", g.adapter_shared >> 20) : std::string("unknown"));
    }
    return per;
}
void testPinnedGpusText() {
    std::size_t n = 0;
    const std::uint64_t vals[] = {0, 1, (1ull << 20) - 1, 1ull << 20, 3ull << 30, (16ull << 30) + 12345};
    std::vector<sf::PinnedGpu> all;
    for (const std::uint64_t nb : vals)
        for (const std::uint64_t own : {0ull, 5ull << 20})
            for (const std::uint64_t ss : vals)
                for (const std::uint64_t as : vals) {
                    const sf::PinnedGpu g{nb, own, ss, as};
                    ISO_CHECK(sf::pinnedGpusText({g}) == old_per({g}));
                    all.push_back(g);
                    ++n;
                }
    ISO_CHECK(sf::pinnedGpusText({}) == old_per({}));
    for (std::size_t k = 2; k <= 4; ++k) {
        const std::vector<sf::PinnedGpu> v(all.begin() + static_cast<std::ptrdiff_t>(k * 7), all.begin() + static_cast<std::ptrdiff_t>(k * 7 + k));
        ISO_CHECK(sf::pinnedGpusText(v) == old_per(v));
        ++n;
    }
    std::printf("pinned gpus text: %zu cases == 729c080\n", n);
}
// :1276-1285 (planPool: the pool without -c / a fit, and the 4096-token refusal)
std::uint32_t old_poolTokens(std::uint64_t avail, std::uint64_t per_tok, std::uint32_t cap_max, std::uint32_t kv_page) {
    const std::uint64_t t = std::min(avail / per_tok, static_cast<std::uint64_t>(8) * cap_max);
    return static_cast<std::uint32_t>(t / kv_page * kv_page);
}
std::string old_refusal(std::uint32_t pool_tokens) {
    if (pool_tokens < 4096)
        return std::format("not enough GPU memory left for the KV cache: a pool of {} tokens (at least 4096 needed).\n"
                           "  Use a smaller model or quantization, give --ctx explicitly,\n"
                           "  or close other programs that use the GPU.",
                           pool_tokens);
    return {};
}
// :1395-1403 (the "kv pool N tokens < floor" warning; floor_second_ctx :167)
std::string old_floorWarn(std::uint32_t pool_now, std::uint32_t slot_ctx_eff, std::uint32_t parallel, bool hint) {
    constexpr std::uint32_t floor_second_ctx = 65536;
    const std::uint64_t floor = static_cast<std::uint64_t>(slot_ctx_eff) + (parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0);
    if (pool_now < floor)
        return std::format("kv pool {} tokens < {} (one {}-token request + a second one reaching {}): concurrent long requests wait in "
                           "the queue, or end early while generating (finish_reason length, finish_cause kv_pool_exhausted){}",
                           pool_now, floor, slot_ctx_eff, parallel > 1 ? std::min(floor_second_ctx, slot_ctx_eff) : 0,
                           hint ? "; --balance allows int8 KV (a longer pool)" : "");
    return {};
}
void testPoolTokens() {
    std::size_t n = 0;
    for (const std::uint64_t av : {0ull, 1ull, 1ull << 20, 3ull << 30, 31ull << 30, 200ull << 30})
        for (const std::uint64_t pt : {1ull, 1024ull, 16384ull, 65536ull, 147456ull})
            for (const std::uint32_t cap : {0u, 4096u, 262144u})
                for (const std::uint32_t pg : {1u, 256u}) {
                    ISO_CHECK(sf::defaultPoolTokens(av, pt, static_cast<std::uint64_t>(8) * cap, pg) == old_poolTokens(av, pt, cap, pg));
                    ++n;
                }
    for (const std::uint32_t p : {0u, 1u, 4095u, 4096u, 4097u, 262144u}) {
        ISO_CHECK(sf::poolRefusal(p) == old_refusal(p));
        ++n;
    }
    for (const std::uint32_t pn : {0u, 4096u, 65535u, 65536u, 131071u, 131072u, 196608u, 262144u})
        for (const std::uint32_t sc : {1024u, 65536u, 65792u, 131072u, 262144u})
            for (const std::uint32_t par : {1u, 2u, 4u})
                for (const bool h : {false, true}) {
                    ISO_CHECK(sf::poolFloorWarn(pn, sc, par, h) == old_floorWarn(pn, sc, par, h));
                    ++n;
                }
    std::printf("pool tokens / refusal / floor: %zu cases == 729c080\n", n);
}
// :1408-1410 (reportPool ck_mb: one checkpoint in MiB, the sum ckptBytes holds)
void testCkptMib() {
    std::size_t n = 0;
    for (const std::uint64_t ng : {0ull, 1ull, 48ull, 30ull})
        for (const std::uint64_t cb : {0ull, 12288ull, 98304ull})
            for (const std::uint64_t sb : {0ull, 1048576ull, 2097152ull})
                for (const std::uint32_t v : {0u, 151936u, 248320u})
                    for (const std::uint32_t e : {0u, 2048u, 5120u}) {
                        const double old = static_cast<double>(ng * (cb + sb) + static_cast<std::uint64_t>(v) * 4 + static_cast<std::uint64_t>(e) * 4) /
                                           (1024.0 * 1024.0);
                        ISO_CHECK(sf::ckptMib(ng * (cb + sb), v, e) == old);
                        ++n;
                    }
    std::printf("ckpt MiB: %zu cases == 729c080\n", n);
}
// :1722-1726 and :1747-1750 (runServer: the "target not reached" entry and the /health note)
std::optional<std::string> old_target(bool parallel_given, std::uint32_t ctx_eff, std::uint64_t target_ctx, std::uint32_t parallel) {
    if (parallel_given && ctx_eff < target_ctx)
        return std::format("target not reached: context {} tokens per request with {} slot(s), less than {}", ctx_eff, parallel, target_ctx);
    return std::nullopt;
}
std::string old_note(const std::vector<std::string>& adjusted) {
    std::string note;
    for (const std::string& a : adjusted)
        if (a.find("--ctx-per-slot") != std::string::npos || a.rfind("-c ", 0) == 0) note += (note.empty() ? "" : "; ") + a;
    return note.empty() ? std::string() : "adjusted at startup to fit the GPU memory: " + note;
}
void testStartupNotes() {
    std::size_t n = 0;
    for (const bool pg : {false, true})
        for (const std::uint32_t ce : {0u, 4096u, 16383u, 16384u, 131072u})
            for (const std::uint32_t sc : {1024u, 16384u, 131072u})
                for (const std::uint32_t par : {1u, 3u}) {
                    const std::uint64_t target = std::min<std::uint32_t>(sc, 16384);
                    ISO_CHECK(sf::targetNotReached(pg, ce, target, par) == old_target(pg, ce, target, par));
                    ++n;
                }
    const std::vector<std::vector<std::string>> lists = {
        {},
        {"--parallel 4 -> 3"},
        {"-c 262144 -> 131072"},
        {"--ctx-per-slot 131072 -> 65536", "--parallel 2 -> 1", "-c 9 -> 8"},
        {"x -c 5", "-cx", "WHIRL_KV_RAM_MB 1 -> 0 MiB", "a --ctx-per-slot b"},
    };
    for (const auto& l : lists) {
        ISO_CHECK(sf::startupNote(l) == old_note(l));
        ++n;
    }
    std::printf("startup notes: %zu cases == 729c080\n", n);
}
// :446-449 (buildModel: the WHIRL_PREFILL_BATCH below-minimum warning; prefill_chunk / prefill_merge = 1024 / 256,
// include/whirl/prefill_schedule.h:28-29)
std::string old_pbWarn(std::uint32_t x, const std::string& v) {
    if (x < whirl::vram::min_prefill_batch)
        return std::format("WHIRL_PREFILL_BATCH={} is below the smallest prefill batch the scheduler needs ({} rows: one {}-row chunk + "
                           "a {}-row tail); using {}",
                           v, whirl::vram::min_prefill_batch, std::size_t{1024}, std::size_t{256}, whirl::vram::min_prefill_batch);
    return {};
}
void testPrefillBatchWarn() {
    std::size_t n = 0;
    for (const std::uint32_t x : {0u, 1u, 1024u, 1279u, 1280u, 1281u, 4096u, 0xffffffffu})
        for (const char* t : {"0", "1279", "01279", "1280", " 12", "4096"}) {
            ISO_CHECK(whirl::vram::prefillBatchFloorWarn(x, t, 1024, 256) == old_pbWarn(x, t));
            ++n;
        }
    std::printf("prefill batch warning: %zu cases == 729c080\n", n);
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
    return iso::run("server.kv_fit", {{"context_plan", testContextPlan}, {"checkpoints", testCheckpoints}, {"ram_floor", testRamFloor},
                                        {"deltanet_layers", testDeltaNetLayers}, {"pinned_gpus_text", testPinnedGpusText},
                                        {"pool_tokens", testPoolTokens}, {"ckpt_mib", testCkptMib}, {"startup_notes", testStartupNotes},
                                        {"prefill_batch_warn", testPrefillBatchWarn}, {"smoke", testSmoke}},
                    argc, argv);
}
