// REVIEW-GOV-2 independent test for REVIEW-GOV-1 P2-2 (reviewer, written from the bug description only; the fixer's
// tests and mock changes in 04462e8 were not read). CPU only: the real Engine + HTTP front end on the 3a4febd host
// mock model (copied unchanged into tests/review_gov2/mock), with a DeviceOps wrapper that injects allocation
// failures.
//   Bug: when govRestoreCache fails part-way (an allocation throws), the partly re-allocated checkpoint buffers stay
//   allocated while the slot / the shared checkpoints stay parked, and the next restore overwrites them without
//   freeing (leak).
//   Expected: device bytes after the failed restore == before it (parked state), and after the retry succeeds the
//   device bytes and live allocations are back at the startup values.
// Cases: (slot) a parked slot's checkpoints, --parallel 2, no shared checkpoints; (shared) the shared prefix checkpoints,
// --parallel 1 (the only slot runs the request, so only the shared ones are parked).
//   gov_restore_tests --gguf TOKENIZER.gguf [--verbose]
// SPDX-License-Identifier: Apache-2.0

#include "http_client.h"
#include "mock_backend.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/log.h"
#include "whirl/gguf.h"
#include "whirl/mem_governor.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"
#include "whirl/wddm_watch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace whirl;
using namespace whirl::server;

namespace {

int g_pass = 0, g_fail = 0;
bool g_verbose = false;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (cond) {                                                        \
            ++g_pass;                                                      \
        } else {                                                           \
            ++g_fail;                                                      \
            std::printf("  FAIL line %d: %s | ", __LINE__, #cond);         \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

std::unique_ptr<gguf::File> g_file;
std::unique_ptr<Tokenizer> g_tok;

struct LogCap {
    std::mutex mu;
    std::vector<std::string> lines;
    static void hook(void* c, std::string_view l) {
        auto* s = static_cast<LogCap*>(c);
        std::lock_guard<std::mutex> lk(s->mu);
        s->lines.emplace_back(l);
        if (g_verbose) std::printf("    log: %.*s\n", static_cast<int>(std::min<std::size_t>(l.size(), 220)), l.data());
    }
    std::size_t count(std::string_view n) {
        std::lock_guard<std::mutex> lk(mu);
        std::size_t k = 0;
        for (const auto& l : lines) k += l.find(n) != std::string::npos ? 1 : 0;
        return k;
    }
};
LogCap g_log;

// DeviceOps wrapper: forwards to the mock; while armed, lets `allow` device mallocs through and then throws on every
// further device malloc until disarmed (a restore under memory pressure: the retries fail too).
class FailOps final : public tier::DeviceOps {
public:
    explicit FailOps(test::MockDeviceOps& m) : m_(m) {}
    void copyAsync(tier::DevPtr d, tier::DevPtr s, std::uint64_t b, tier::Stream st) override { m_.copyAsync(d, s, b, st); }
    void download(void* d, tier::DevPtr s, std::uint64_t b) override { m_.download(d, s, b); }
    tier::Stream streamCreateNonBlocking() override { return m_.streamCreateNonBlocking(); }
    void streamDestroy(tier::Stream s) override { m_.streamDestroy(s); }
    void streamSync(tier::Stream s) override { m_.streamSync(s); }
    void streamWaitEvent(tier::Stream s, tier::Event e) override { m_.streamWaitEvent(s, e); }
    tier::Event eventCreate() override { return m_.eventCreate(); }
    void eventDestroy(tier::Event e) override { m_.eventDestroy(e); }
    void eventRecord(tier::Event e, tier::Stream s) override { m_.eventRecord(e, s); }
    bool eventDone(tier::Event e) override { return m_.eventDone(e); }
    void eventSync(tier::Event e) override { m_.eventSync(e); }
    void* hostMalloc(std::uint64_t b) override { return m_.hostMalloc(b); }
    void hostFree(void* p) override { m_.hostFree(p); }
    tier::DevPtr malloc(std::uint64_t b) override {
        if (armed_.load()) {
            std::lock_guard<std::mutex> lk(mu_);
            sizes_.push_back(b);
            if (allowed_ >= allow_) {
                ++failures_;
                throw std::runtime_error(std::format("injected allocation failure ({} bytes)", b));
            }
            ++allowed_;
        }
        return m_.malloc(b);
    }
    void free(tier::DevPtr p) override { m_.free(p); }
    void setDevice(int d) override { m_.setDevice(d); }
    const char* lastErrorString() const override { return "injected"; }

    void arm(int allow) {
        std::lock_guard<std::mutex> lk(mu_);
        allow_ = allow;
        allowed_ = 0;
        failures_ = 0;
        sizes_.clear();
        armed_ = true;
    }
    void disarm() { armed_ = false; }
    int failures() {
        std::lock_guard<std::mutex> lk(mu_);
        return failures_;
    }
    std::string sizes() {
        std::lock_guard<std::mutex> lk(mu_);
        std::string s;
        for (auto b : sizes_) s += std::format(" {}", b);
        return s;
    }

private:
    test::MockDeviceOps& m_;
    std::atomic<bool> armed_{false};
    std::mutex mu_;
    int allow_ = 0, allowed_ = 0, failures_ = 0;
    std::vector<std::uint64_t> sizes_;
};

std::string promptJson(std::uint32_t n, std::uint64_t seed) {
    std::string s = "[";
    std::uint64_t z = seed * 0x9E3779B97F4A7C15ull + 1;
    for (std::uint32_t i = 0; i < n; ++i) {
        z ^= z >> 33;
        z *= 0xff51afd7ed558ccdull;
        z ^= z >> 33;
        s += (i ? "," : "") + std::to_string(1000 + z % 20000);
    }
    return s + "]";
}
std::string body(std::uint32_t n, std::uint64_t seed, std::uint32_t gen) {
    return std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"ignore_eos\":true}}", promptJson(n, seed), gen);
}

bool waitFor(const std::function<bool()>& f, int ms) {
    for (int i = 0; i < ms / 20; ++i) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return f();
}

std::uint64_t jsonU(const std::string& s, const std::string& key) {
    const auto p = s.find("\"" + key + "\":");
    if (p == std::string::npos) return ~0ull;
    return std::strtoull(s.c_str() + p + key.size() + 3, nullptr, 10);
}

void runCase(bool shared, bool host = false) {
    std::printf("P2-2 [%s%s]\n", shared ? "shared prefix checkpoints" : "slot checkpoints",
                host ? ", WHIRL_CKPT_HOST: pinned slot checkpoints" : "");
    constexpr std::uint64_t MiB = 1ull << 20;
    test::MockDeviceOps mock;
    FailOps ops(mock);
    test::MockConfig mc;
    mc.parallel = shared ? 1 : 2;
    mc.slot_ctx = 16384;
    mc.pool_tokens = 65536;
    mc.mtp = false;
    mc.cycle_us = 5000;  // a long-running request: ~5 ms per token
    auto model = std::make_unique<test::MockModel>(ops, *g_tok, mc);
    EngineOptions eo;
    eo.parallel = mc.parallel;
    eo.ctx = mc.slot_ctx;
    eo.model_name = "mock";
    eo.use_mtp = false;
    eo.n_ck = 2;
    eo.n_spe = shared ? 2 : 0;
    eo.ckpt_host = host;
    eo.sys_min = 2048;
    eo.gather_ms = 0;
    eo.seed = 1;
    auto eng = std::make_unique<Engine>(*model, ops, *g_tok, eo);
    eng->allocState();
    eng->initPool();
    const std::uint64_t fixed = 28000 * MiB;
    std::uint64_t pool0 = 0;
    for (const KvArr& a : model->kvArrays()) pool0 += a.row * mc.pool_tokens;
    const std::uint64_t M = pool0 / 2;
    std::atomic<std::uint64_t> budget{fixed + mock.deviceBytes() + M + M / 4};
    wddmwatch::Config vc;
    vc.idle_sample_s = 0.2;
    vc.governor = true;
    eng->setVramProbe(
        [&] {
            wddmwatch::Sample x;
            x.ok = true;
            x.local_usage = fixed + mock.deviceBytes();
            x.local_budget = budget.load();
            x.nonlocal_usage = 400 * MiB;
            return x;
        },
        true, std::nullopt, vc);
    memgov::Config gc;
    gc.page = qwen35::kv_page;
    gc.pool_start = mc.pool_tokens;
    gc.pool_min = 4096;
    gc.ctx = mc.slot_ctx;
    gc.recover_s = 0.4;
    gc.hold_s = 0.6;
    gc.settle_s = 0.5;
    for (const KvArr& a : model->kvArrays()) gc.per_tok += a.row;
    eng->setGovernor(gc, [&](std::uint64_t) { return M; }, 0);
    HttpServer http(*eng, HttpOptions{});
    http.start("127.0.0.1", 0);
    std::thread loop([&] { eng->runLoop(); });
    auto get = [&](const char* p) { return test::httpRequest("127.0.0.1", http.port(), "GET", p).body; };
    auto post = [&](const std::string& b) { return test::httpRequest("127.0.0.1", http.port(), "POST", "/v1/completions", b); };

    // warm-up request (allocations a first request makes are part of the steady state)
    (void)post(body(300, 5, 4));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const std::uint64_t local0 = fixed + mock.deviceBytes();
    budget = local0 + M + M / 4;
    const std::uint64_t dev0 = mock.deviceBytes();
    const std::size_t live0 = mock.liveAllocations();

    // a long request runs
    std::thread ta([&] { (void)post(body(2000, 71, 8000)); });
    CHECK(waitFor([&] { return get("/health").find("\"slots_busy\":1") != std::string::npos; }, 5000), "long request started");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::size_t n_park_log0 = g_log.count("checkpoint buffers freed now");  // before the drop: slot 1's request end may park
    // another program takes VRAM: a quarter of the margin free
    budget = local0 + M / 4;
    if (!shared) (void)post(body(500, 72, 8));  // ends on slot 1: its request-end sample parks slot 1's checkpoints
    // parked: parked_mib > 0, or (WHIRL_CKPT_HOST: only the MTP hidden row is in VRAM, parked_mib rounds to 0) the governor's
    // "checkpoint buffers freed now" log line
    const bool parked = waitFor([&] {
        const auto v = jsonU(get("/health"), "parked_mib");
        return (v != ~0ull && v > 0) || g_log.count("checkpoint buffers freed now") > n_park_log0;
    }, 12000);
    const std::uint64_t dev_parked = mock.deviceBytes();
    CHECK(parked, "checkpoint buffers parked");
    CHECK(dev_parked < dev0, "parking freed device bytes (%llu -> %llu)", (unsigned long long)dev0, (unsigned long long)dev_parked);
    // the budget comes back while the request still runs: the governor restores; the 2nd allocation and every one after
    // it fail
    ops.arm(1);
    budget = local0 + 3 * M;
    const bool failed = waitFor([&] { return ops.failures() >= 1; }, 12000);
    CHECK(failed, "the injected failure was hit by a restore");
    // let at least one more sample retry under the same failure (both versions: no further allocation succeeds)
    std::this_thread::sleep_for(std::chrono::milliseconds(6000));
    const std::uint64_t dev_failed = mock.deviceBytes();
    std::printf("  device bytes: startup %llu, parked %llu, after the failed restore %llu (allocations tried while armed:%s)\n",
                (unsigned long long)dev0, (unsigned long long)dev_parked, (unsigned long long)dev_failed, ops.sizes().c_str());
    CHECK(dev_failed == dev_parked, "device bytes after the failed restore equal those before it (%llu vs %llu, +%lld)",
          (unsigned long long)dev_failed, (unsigned long long)dev_parked, (long long)(dev_failed - dev_parked));
    // the memory is back: the retry succeeds
    ops.disarm();
    const std::size_t n_rest_log0 = g_log.count("checkpoint buffers restored");
    const bool restored = waitFor([&] {
        return jsonU(get("/health"), "parked_mib") == 0 && g_log.count("checkpoint buffers restored") > n_rest_log0;
    }, 15000);
    CHECK(restored, "a later restore succeeded (parked_mib 0)");
    ta.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const std::uint64_t dev_end = mock.deviceBytes();
    const std::size_t live_end = mock.liveAllocations();
    std::printf("  after the successful retry: device bytes %llu (startup %llu), live allocations %zu (startup %zu), resizes %u\n",
                (unsigned long long)dev_end, (unsigned long long)dev0, live_end, live0, model->resizes());
    CHECK(model->resizes() == 0, "no pool re-allocation in this scenario (resizes %u)", model->resizes());
    CHECK(dev_end == dev0, "no leak after the retry: device bytes back at startup (%llu vs %llu, +%lld)", (unsigned long long)dev_end,
          (unsigned long long)dev0, (long long)(dev_end - dev0));
    CHECK(live_end == live0, "live allocations back at startup (%zu vs %zu)", live_end, live0);
    eng->stop();
    loop.join();
    http.stop();
    eng.reset();
    model.reset();
}

}  // namespace

int main(int argc, char** argv) {
    std::string gg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gg = argv[++i];
        else if (a == "--verbose") g_verbose = true;
    }
    if (gg.empty()) {
        std::fprintf(stderr, "usage: gov_restore_tests --gguf TOKENIZER.gguf\n");
        return 2;
    }
    Log::setConsole(false);
    Log::setHook(&LogCap::hook, &g_log);
    g_file = std::make_unique<gguf::File>(gguf::File::open(gg));
    g_tok = std::make_unique<Tokenizer>(Tokenizer::fromGguf(*g_file));
    runCase(false);
    runCase(true);
    runCase(false, true);
    Log::setHook(nullptr, nullptr);
    std::printf("review_gov2 restore tests: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail || g_pass == 0) ? 1 : 0;
}
