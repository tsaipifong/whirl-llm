// Host micro-benchmark of the KV page path at 128k context (arch024 S2, for S4's P-speed check; PLAN v3 N4).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. CPU only, no GPU: the real Engine (scheduling, page pool, prefix cache, quarantine, tier
// spills / restores) over tests/server/mock_backend.h's MockModel and device::NullBackend, driven over HTTP like
// whirl-server-tests. It uses only interfaces that exist at d5fc637 (v0.2.3): Engine, HttpServer, MockModel,
// NullBackend, tier::Tier, Engine::stats(), NullBackend::copyCalls() / copiedBytes(). The same source file is
// therefore built at d5fc637 (A) and at the step under test (B); S4 runs them interleaved.
//
// One round (per --rounds):
//   1. prompt A (ctx - 64 tokens)                 a cold 128k prefill: pool pages ensured page by page
//   2. prompt A + 32 tokens                       prefix reuse of A in its slot (copy-on-write of the shared tail)
//   3. prompt B (ctx - 64 tokens, other tokens)   the pool holds less than A + B: A's pages are evicted / spilled
//   4. prompt A again                             served from the RAM tier (restore) or the pool
// Printed per round: wall ms per request and the engine / backend counters the S4 condition reads:
//   pages per 128k prompt (ctx / kv_page), Stats cow_pages / q_waits / evictions / spill_waits,
//   NullBackend copyCalls / copiedBytes (equal in A and B = the same device-side work).
// Usage: whirl-page-bench --gguf MODEL.gguf [--rounds N] [--ctx TOKENS]   (the GGUF is read for its tokenizer only)

#include "http_client.h"
#include "mock_backend.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/log.h"
#include "tier/kv_tier.h"
#include "whirl/gguf.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace whirl;
using namespace whirl::server;

namespace {

std::vector<std::uint32_t> makePrompt(std::uint32_t n, std::uint64_t seed, const std::vector<std::uint32_t>& prefix = {}) {
    std::vector<std::uint32_t> p = prefix;
    std::uint64_t z = seed * 0x9E3779B97F4A7C15ull + 1;
    while (p.size() < n) {
        z ^= z >> 33;
        z *= 0xff51afd7ed558ccdull;
        z ^= z >> 33;
        p.push_back(static_cast<std::uint32_t>(1000 + z % 20000));
    }
    return p;
}

std::string body(const std::vector<std::uint32_t>& prompt, std::uint32_t max_tokens) {
    std::string s = "{\"prompt\":[";
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(prompt[i]);
    }
    s += "],\"max_tokens\":" + std::to_string(max_tokens) + ",\"temperature\":0,\"ignore_eos\":true,\"stream\":false}";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path;
    int rounds = 1;
    std::uint32_t ctx = 131072;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else {
            std::fprintf(stderr, "usage: whirl-page-bench --gguf MODEL.gguf [--rounds N] [--ctx TOKENS]\n");
            return 2;
        }
    }
    if (gguf_path.empty()) {
        if (const char* e = std::getenv("WHIRL_TEST_GGUF")) gguf_path = e;
    }
    if (gguf_path.empty() || rounds < 1 || ctx < 4096) {
        std::fprintf(stderr, "usage: whirl-page-bench --gguf MODEL.gguf [--rounds N] [--ctx TOKENS]\n");
        return 2;
    }
    Log::setConsole(false);
    std::unique_ptr<gguf::File> file;
    std::unique_ptr<Tokenizer> tok;
    try {
        file = std::make_unique<gguf::File>(gguf::File::open(gguf_path));
        tok = std::make_unique<Tokenizer>(Tokenizer::fromGguf(*file));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cannot load the tokenizer from %s: %s\n", gguf_path.c_str(), ex.what());
        return 2;
    }

    device::NullBackend ops;
    test::MockConfig mc;
    mc.parallel = 2;
    mc.slot_ctx = ctx;
    mc.pool_tokens = ctx + ctx / 2;  // A and B (128k each) do not both fit: evictions
    mc.max_batch = 4096;
    mc.mtp = false;
    auto model = std::make_unique<test::MockModel>(ops, *tok, mc);
    EngineOptions eo;
    eo.parallel = mc.parallel;
    eo.ctx = mc.slot_ctx;
    eo.model_name = "mock";
    eo.use_mtp = false;
    auto engine = std::make_unique<Engine>(*model, ops, *tok, eo);
    engine->allocState();
    engine->initPool();
    // RAM tier (host memory over NullBackend), so request ends spill and evicted prefixes come back by restore
    const std::uint64_t tier_mb = 1024;
    auto tier = tier::Tier::create(ops, tier_mb << 20, 42, 0);
    {
        tier::Config tc;
        tc.ram_bytes = tier_mb << 20;
        tc.min_tokens = 512;
        tc.ssd_delay_ms = 0;
        tier::Layout lay;
        const auto& cfg = model->cfg();
        std::uint64_t n_gdn = 0;
        for (auto p : model->ssmState())
            if (p) ++n_gdn;
        lay.ck_bytes = n_gdn * (model->convBytes() + model->ssmBytes()) + static_cast<std::uint64_t>(cfg.n_embd) * 4 +
                       static_cast<std::uint64_t>(cfg.n_vocab) * 4;
        std::uint64_t per_tok = 0;
        for (const KvArr& a : model->kvArrays()) per_tok += a.row;
        lay.page_bytes = per_tok * qwen35::kv_page;
        lay.page_tokens = qwen35::kv_page;
        lay.tok_cap = mc.slot_ctx;
        engine->attachTier(tier.get());
        tier->start(tc, lay, 0x1234);
    }
    auto http = std::make_unique<HttpServer>(*engine);
    http->start("127.0.0.1", 0);
    std::thread loop([&] { engine->runLoop(); });

    const std::uint32_t n = ctx - 64;
    const auto A = makePrompt(n, 1);
    const auto A2 = makePrompt(n + 32, 2, A);
    const auto B = makePrompt(n, 3);
    const std::string bodies[4] = {body(A, 8), body(A2, 8), body(B, 8), body(A, 8)};
    const char* names[4] = {"A", "A+32", "B", "A again"};
    std::printf("page_bench: ctx %u, pages per prompt %u (kv_page %u), rounds %d\n", ctx, ctx / qwen35::kv_page, qwen35::kv_page,
                rounds);
    int bad = 0;
    for (int r = 0; r < rounds; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        double ms[4] = {};
        for (int k = 0; k < 4; ++k) {
            const auto res = test::httpRequest("127.0.0.1", http->port(), "POST", "/v1/completions", bodies[k]);
            ms[k] = res.ms;
            if (res.status != 200 || res.body.find("\"choices\"") == std::string::npos) {
                std::printf("page_bench: round %d request %s: HTTP %d %s\n", r, names[k], res.status, res.err.c_str());
                ++bad;
            }
        }
        const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const Engine::Stats st = engine->stats();
        std::printf("page_bench: round %d total_ms %.1f | A %.1f | A+32 %.1f | B %.1f | A_again %.1f | cow_pages %llu q_waits %llu "
                    "evictions %llu spill_waits %llu | copy_calls %llu copied_bytes %llu\n",
                    r, total, ms[0], ms[1], ms[2], ms[3], static_cast<unsigned long long>(st.cow_pages),
                    static_cast<unsigned long long>(st.q_waits), static_cast<unsigned long long>(st.evictions),
                    static_cast<unsigned long long>(st.spill_waits), static_cast<unsigned long long>(ops.copyCalls()),
                    static_cast<unsigned long long>(ops.copiedBytes()));
    }

    engine->stop();
    loop.join();
    http->stop();
    http.reset();
    engine->attachTier(nullptr);
    tier.reset();  // the tier first (its IO thread calls into the engine), then the engine: the product's order
    engine.reset();
    model.reset();
    const bool clean = ops.clean();
    std::printf("page_bench: %s, backend %s\n", bad == 0 ? "ok" : "FAILED", clean ? "clean" : "LEAK");
    return bad == 0 && clean ? 0 : 1;
}
