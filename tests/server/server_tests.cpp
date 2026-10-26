// Server tests: protocol / sampling unit tests and end-to-end HTTP tests of
// the engine against the host-memory mock model (no GPU).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-server-tests [--gguf TOKENIZER.gguf] [--only NAME[,NAME...]] [--verbose]
// The GGUF only provides the tokenizer (vocabulary + merges); default: the
// Qwen3.8-27B UD-Q4_K_M file of this machine, or WHIRL_TEST_GGUF.

#include "../test_env.h"
#include "http_client.h"
#include "mock_backend.h"
#include "server/draft_caps.h"
#include "server/engine.h"
#include "server/engine_thread.h"
#include "server/http.h"
#include "server/log.h"
#include "server/prefix_store.h"
#include "server/protocol.h"
#include "server/tier_bridge.h"
#include "server/tokens.h"
#include "tier/kv_tier.h"
#include "tier/ram_size.h"
#include "whirl/vram_limit.h"
#include "whirl/gpu_select.h"
#include "whirl/startup_fit.h"
#include "whirl/mem_governor.h"
#include "whirl/pool_budget.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"
#include "whirl/vismap.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace whirl;
using namespace whirl::server;
using whirl::test::HttpResult;
using whirl::test::MockConfig;
using whirl::device::NullBackend;
using whirl::test::MockModel;

namespace {

int g_pass = 0, g_fail = 0;
bool g_verbose = false;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (cond) {                                                                       \
            ++g_pass;                                                                     \
        } else {                                                                          \
            ++g_fail;                                                                     \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const auto& va_ = (a);                                                                      \
        const auto& vb_ = (b);                                                                      \
        if (va_ == vb_) {                                                                           \
            ++g_pass;                                                                               \
        } else {                                                                                    \
            ++g_fail;                                                                               \
            std::fprintf(stderr, "  FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b);           \
        }                                                                                           \
    } while (0)

std::string g_gguf;
std::unique_ptr<gguf::File> g_file;
std::unique_ptr<Tokenizer> g_tok;

const Tokenizer& tok() { return *g_tok; }

std::string decode(std::span<const std::uint32_t> ids) {
    std::string s;
    for (std::uint32_t t : ids) s += tok().piece(static_cast<TokenId>(t), true);
    return s;
}

// ---------------------------------------------------------------------------
// test server (mock model + engine + HTTP on a free port)

struct ServerSetup {
    MockConfig mc;
    EngineOptions eo;
    bool tier = false;
    std::uint64_t tier_ram_mb = 256;
    std::optional<std::string> ssd_dir;
    std::uint32_t tier_min = 512;
    bool tier_shutdown = false;  // --kv-ssd-mode shutdown (default: always, delay 0)
    std::uint64_t fingerprint = 0x1234;
    // vision: attach a host mock encoder (image_url parts accepted)
    bool vision = false;
    bool allow_files = false;
    std::uint32_t vis_ck_min = 1024;
    HttpOptions http;
    // WDDM-LOG: injected video memory samples (Engine::setVramProbe), warn as on a discrete GPU
    std::function<whirl::wddmwatch::Sample()> vram_probe;
    std::optional<whirl::wddmwatch::Sample> vram_base;  // the sizing-time baseline (nullopt: the first sample)
    whirl::wddmwatch::Config vram_cfg;
    bool vram_startup_warned = false;
    std::function<std::optional<whirl::wddmwatch::Others>()> vram_others;
    // v0.2.2.1 startup fit: the "startup" JSON for /props and /health, and the context note
    std::string startup_json, ctx_note;
    // v0.2.2.2 memory governor (per_tok is filled in from the mock's pool arrays)
    std::optional<whirl::memgov::Config> gov;
    std::function<std::uint64_t(std::uint64_t)> gov_margin;
};

// vision: host mock encoder (real stb decode + preprocessing size rule, deterministic
// fake embeddings), counts encodes
class MockVision final : public ServerVision {
public:
    vision::Prepared prepare(std::span<const std::uint8_t> bytes) const override {
        const vision::Rgb src = vision::decodeImage(bytes);
        const auto ts = vision::targetSize(src.w, src.h, vision::SizeOpt{32, 8 * 1024, 4096 * 1024});
        vision::Prepared p;
        p.nx = ts[0] / 32;
        p.ny = ts[1] / 32;
        const std::span<const std::uint8_t> parts[1] = {src.px};
        p.hash = vision::sha256(parts);
        p.rgb = vision::resizePadCeil(src, ts[0], ts[1]);
        return p;
    }
    std::uint32_t projDim() const override { return 8; }
    vision::EncodeStats encode(const vision::Prepared& p, std::span<const vision::Region>, float* out, std::size_t n) override {
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p.hash[i % 32]) + static_cast<float>(i);
        encodes.fetch_add(1);
        loaded_ = true;
        vision::EncodeStats st;
        st.n_patches = p.nTokens() * 4;
        return st;
    }
    bool loaded() const override { return loaded_; }
    bool idleTick() override { return false; }
    std::uint32_t idleS() const override { return 60; }
    std::atomic<int> encodes{0};

private:
    bool loaded_ = false;
};

class TestServer {
public:
    explicit TestServer(const ServerSetup& su) : su_(su) {
        model_ = std::make_unique<MockModel>(ops_, tok(), su.mc);
        EngineOptions eo = su.eo;
        eo.parallel = su.mc.parallel;
        eo.ctx = su.mc.slot_ctx;
        if (eo.model_name.empty()) eo.model_name = "mock";
        eo.use_mtp = su.mc.mtp && eo.use_mtp;
        eo.use_dflash = su.mc.dflash && eo.use_dflash;
        engine_ = std::make_unique<Engine>(*model_, ops_, tok(), eo);
        engine_->allocState();
        engine_->initPool();
        if (su.vision) {
            vis_ = std::make_unique<MockVision>();
            VisionConfig vc;
            vc.vis = vis_.get();
            vc.allow_files = su.allow_files;
            vc.ck_min = su.vis_ck_min;
            const TokenId pad = tok().find("<|image_pad|>"), ve = tok().find("<|vision_end|>");
            vc.image_pad_id = static_cast<std::uint32_t>(pad);
            vc.vision_end_id = static_cast<std::uint32_t>(ve);
            engine_->attachVision(vc);
        }
        if (su.tier) {
            tier_ = tier::Tier::create(ops_, su.tier_ram_mb << 20, 42, 0);
            tier::Config tc;
            tc.ram_bytes = su.tier_ram_mb << 20;
            tc.ssd_dir = su.ssd_dir;
            tc.ssd_cap = 4ull << 30;
            tc.min_tokens = su.tier_min;
            tc.ssd_delay_ms = 0;
            tc.ssd_shutdown_only = su.tier_shutdown;
            tier::Layout lay;
            const auto& cfg = model_->cfg();
            std::uint64_t n_gdn = 0;
            for (auto p : model_->ssmState())
                if (p) ++n_gdn;
            lay.ck_bytes = n_gdn * (model_->convBytes() + model_->ssmBytes()) + static_cast<std::uint64_t>(cfg.n_embd) * 4 +
                           static_cast<std::uint64_t>(cfg.n_vocab) * 4;
            std::uint64_t per_tok = 0;
            for (const KvArr& a : model_->kvArrays()) per_tok += a.row;
            lay.page_bytes = per_tok * qwen35::kv_page;
            lay.page_tokens = qwen35::kv_page;
            lay.tok_cap = su.mc.slot_ctx;
            engine_->attachTier(tier_.get());
            tier_->start(tc, lay, su.fingerprint);
        }
        if (!su.startup_json.empty() || !su.ctx_note.empty()) engine_->setStartup(su.startup_json, su.ctx_note);
        if (su.vram_probe) engine_->setVramProbe(su.vram_probe, true, su.vram_base, su.vram_cfg, su.vram_startup_warned, su.vram_others);
        if (su.gov) {
            whirl::memgov::Config gc = *su.gov;
            gc.per_tok = 0;
            for (const KvArr& a : model_->kvArrays()) gc.per_tok += a.row;
            engine_->setGovernor(gc, su.gov_margin, su.tier ? su.tier_ram_mb : 0);
        }
        http_ = std::make_unique<HttpServer>(*engine_, su.http);
        http_->start("127.0.0.1", 0);
        loop_ = std::thread([this] { engine_->runLoop(); });
    }
    ~TestServer() { shutdown(); }
    // the engine loop stops (as after Ctrl+C) while the HTTP front end still accepts
    void stopEngine() {
        engine_->stop();
        if (loop_.joinable()) loop_.join();
    }
    void shutdown() {
        if (!engine_) return;
        engine_->stop();
        if (loop_.joinable()) loop_.join();
        http_->stop();
        http_.reset();
        engine_->attachTier(nullptr);
        // FR-CACHE P3-5: the tier first (its destructor joins the IO thread, whose wake callback calls into
        // the engine), then the engine -- the product's order (server_main declares the tier after the engine)
        tier_.reset();
        engine_.reset();
        model_.reset();
    }
    // arch024 S3: shutdown() with a callback before each step ("tier": the tier is destroyed next, "engine", "model",
    // "end"), so a test can watch what the destructors do
    void shutdownSteps(const std::function<void(const char*)>& at) {
        engine_->stop();
        if (loop_.joinable()) loop_.join();
        http_->stop();
        http_.reset();
        engine_->attachTier(nullptr);
        at("tier");
        tier_.reset();
        at("engine");
        engine_.reset();
        at("model");
        model_.reset();
        at("end");
    }
    HttpResult post(const std::string& path, const std::string& body) {
        return test::httpRequest("127.0.0.1", http_->port(), "POST", path, body);
    }
    HttpResult get(const std::string& path) { return test::httpRequest("127.0.0.1", http_->port(), "GET", path); }
    MockModel& model() { return *model_; }
    MockVision* vision() { return vis_.get(); }
    Engine& engine() { return *engine_; }
    tier::Tier* tierp() { return tier_.get(); }
    // REVIEW-ARCH3 P2-1: the tier (entries, statistics) belongs to the engine thread, and the engine spills a slot
    // only after it sent the response. Read tier state through these: they run on the engine thread at the start of
    // its next loop iteration, i.e. after the last answered request's spill. Only while the engine loop runs.
    template <class F>
    void onEngine(F&& f) {
        engine_->runOnLoop(std::function<void()>(std::forward<F>(f)));
    }
    tier::Stats tierStats() {
        tier::Stats st;
        onEngine([&] { st = tier_->stats; });
        return st;
    }
    std::size_t ssdEntries() {
        std::size_t n = 0;
        onEngine([&] { n = tier_->ssdEntries(); });
        return n;
    }
    NullBackend& ops() { return ops_; }
    std::uint16_t port() const { return http_->port(); }

private:
    ServerSetup su_;
    NullBackend ops_;
    std::unique_ptr<MockVision> vis_;
    std::unique_ptr<MockModel> model_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<tier::Tier> tier_;
    std::unique_ptr<HttpServer> http_;
    std::thread loop_;
};

// Log capture (counts lines matching substrings).
struct LogCapture {
    std::mutex mu;
    std::vector<std::string> lines;
    static void hook(void* ctx, std::string_view l) {
        auto* c = static_cast<LogCapture*>(ctx);
        std::lock_guard<std::mutex> lk(c->mu);
        c->lines.emplace_back(l);
    }
    std::size_t count(std::string_view needle) {
        std::lock_guard<std::mutex> lk(mu);
        std::size_t n = 0;
        for (const auto& l : lines)
            if (l.find(needle) != std::string::npos) ++n;
        return n;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        lines.clear();
    }
};
LogCapture g_log;

std::string idsJson(std::span<const std::uint32_t> ids) {
    std::string s = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(ids[i]);
    }
    return s + "]";
}

// deterministic token-id prompt
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

std::string completionBody(std::span<const std::uint32_t> prompt, std::uint32_t max_tokens, bool stream = false,
                           const std::string& extra = "") {
    return std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0,\"ignore_eos\":true,\"stream\":{}{}}}",
                       idsJson(prompt), max_tokens, stream ? "true" : "false", extra);
}

json::Value parseBody(const HttpResult& r) {
    try {
        return json::parse(r.body);
    } catch (const std::exception&) {
        return json::Value();
    }
}

std::string completionText(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* ch = v.get("choices");
    if (!ch || !ch->isArray() || ch->asArray().empty()) return "<no choices>";
    const json::Value* t = ch->asArray()[0].get("text");
    return t && t->isString() ? t->asString() : "<no text>";
}

std::uint64_t cachedTokens(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* u = v.get("usage");
    if (!u) return 0;
    const json::Value* d = u->get("prompt_tokens_details");
    if (!d) return 0;
    const json::Value* c = d->get("cached_tokens");
    return c ? static_cast<std::uint64_t>(c->asInt()) : 0;
}

// stream: concatenation of the text deltas
std::string streamText(const HttpResult& r, bool chat, std::string* finish = nullptr) {
    std::string out;
    for (const std::string& ev : test::sseEvents(r.body)) {
        if (ev == "[DONE]") continue;
        json::Value v;
        try {
            v = json::parse(ev);
        } catch (const std::exception&) {
            continue;
        }
        const json::Value* ch = v.get("choices");
        if (!ch || !ch->isArray() || ch->asArray().empty()) continue;
        const json::Value& c0 = ch->asArray()[0];
        if (finish) {
            const json::Value* fr = c0.get("finish_reason");
            if (fr && fr->isString()) *finish = fr->asString();
        }
        if (chat) {
            const json::Value* d = c0.get("delta");
            if (d) {
                const json::Value* c = d->get("content");
                if (c && c->isString()) out += c->asString();
            }
        } else {
            const json::Value* t = c0.get("text");
            if (t && t->isString()) out += t->asString();
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// unit tests

// CONC-8S: MTP draft caps under load with DeltaNet snapshot sets (gfx1151, no replay): with relaxacc
// the replay path's {8, 7, 4, 3} (literal drafts keep the old 8-set budget); exact acceptance: 8 / k
void testDraftCaps() {
    const std::array<std::uint32_t, qwen35::gdn_max_seg + 1> def = {0, 8, 7, 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    const SnapDraftCaps a = snapshotDraftCaps(def, 8, 4, false, std::nullopt, true);
    CHECK_EQ(a.caps[1], 8u);
    CHECK_EQ(a.caps[2], 7u);
    CHECK_EQ(a.caps[3], 4u);
    CHECK_EQ(a.caps[4], 3u);  // was 8 / 4 = 2: concurrent fast output != solo
    CHECK_EQ(a.caps[5], 2u);
    CHECK_EQ(a.sets, 14u);    // 2 slots x 7 drafts
    CHECK_EQ(a.lit_sets, 8u);  // what the old caps allocated (n-gram budget unchanged)
    for (std::uint32_t k = 1; k <= 4; ++k) CHECK(k * a.caps[k] <= qwen35::gdn_max_snap);
    // fixed 3 drafts: 3 per cycle at every load (old: 2 with 3-4 slots)
    const SnapDraftCaps b = snapshotDraftCaps(def, 3, 4, false, std::nullopt, true);
    CHECK_EQ(std::min(b.caps[4], 3u), 3u);
    CHECK_EQ(std::min(b.caps[3], 3u), 3u);
    CHECK_EQ(b.sets, 12u);
    CHECK_EQ(b.lit_sets, 8u);
    // one slot: nothing changes
    const SnapDraftCaps c = snapshotDraftCaps(def, 8, 1, false, std::nullopt, true);
    CHECK_EQ(c.sets, 8u);
    CHECK_EQ(c.lit_sets, 8u);
    // WHIRL_MTP_BATCH_DRAFTS: used as given, literal drafts may use all sets; WHIRL_SNAP_SETS raises both
    const SnapDraftCaps d = snapshotDraftCaps(def, 8, 4, true, std::nullopt, true);
    CHECK_EQ(d.caps[4], 3u);
    CHECK_EQ(d.lit_sets, 0u);
    const SnapDraftCaps e = snapshotDraftCaps(def, 8, 1, false, 12u, true);
    CHECK_EQ(e.sets, 12u);
    CHECK_EQ(e.lit_sets, 12u);
    // exact acceptance (precise / balance, fast with WHIRL_RELAX=0): the old 8 / k caps, all sets
    const SnapDraftCaps f = snapshotDraftCaps(def, 8, 4, false, std::nullopt, false);
    CHECK_EQ(f.caps[2], 4u);
    CHECK_EQ(f.caps[3], 2u);
    CHECK_EQ(f.caps[4], 2u);
    CHECK_EQ(f.sets, 8u);
    CHECK_EQ(f.lit_sets, 0u);
}

void testProtocol() {
    CHECK_EQ(partialHold("abc<to", "<tool_call>"), std::size_t(3));
    CHECK_EQ(partialHold("abc", "<tool_call>"), std::size_t(0));
    CHECK_EQ(partialHold("x</think", "</think>"), std::size_t(7));
    CHECK_EQ(validUtf8Prefix("ab\xE4\xB8"), std::size_t(2));
    CHECK_EQ(validUtf8Prefix("ab\xE4\xB8\xAD"), std::size_t(5));
    CHECK_EQ(validUtf8Prefix("a\x80" "b"), std::size_t(3));
    CHECK_EQ(sanitizeUtf8("a\xFF" "b"), std::string("a\xEF\xBF\xBD" "b"));
    CHECK_EQ(sanitizeUtf8("ok"), std::string("ok"));
    {
        std::string s;
        appendJsonStr(s, "a\"b\n\x01");
        CHECK_EQ(s, std::string("\"a\\\"b\\n\\u0001\""));
    }
    {
        std::string s;
        appendFloat(s, 12.5);
        appendFloat(s, std::nan(""));
        CHECK_EQ(s, std::string("12.5null"));
    }
    // Out.view: think-open chat
    {
        Out o;
        o.chat = true;
        o.think_open = true;
        o.text = "reasoning here</think>\n\nanswer";
        o.valid = o.text.size();
        auto v = o.view(false);
        CHECK_EQ(std::string(v.reason), std::string("reasoning here"));
        CHECK_EQ(std::string(v.content), std::string("answer"));
        o.text = "partial reason</thi";
        o.valid = o.text.size();
        v = o.view(false);
        CHECK_EQ(std::string(v.reason), std::string("partial reason"));
        CHECK(v.content.empty());
    }
    {
        Out o;
        o.chat = true;
        o.text = "  <thi";
        o.valid = o.text.size();
        auto v = o.view(false);
        CHECK(v.content.empty() && v.reason.empty());
        o.text = "  <think>r</think>c";
        o.valid = o.text.size();
        v = o.view(true);
        CHECK_EQ(std::string(v.reason), std::string("r"));
        CHECK_EQ(std::string(v.content), std::string("c"));
    }
    {
        Out o;
        o.chat = false;
        o.stops = {"STOP"};
        o.text = "hello ST";
        o.valid = o.text.size();
        auto v = o.view(false);
        CHECK_EQ(std::string(v.content), std::string("hello "));
        CHECK(!v.stopped);
        o.text = "hello STOP more";
        o.valid = o.text.size();
        v = o.view(false);
        CHECK_EQ(std::string(v.content), std::string("hello "));
        CHECK(v.stopped);
    }
    {
        Out o;
        o.chat = true;
        o.tools_on = true;
        o.text = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTaipei\n</parameter>\n</function>\n</tool_call>";
        o.valid = o.text.size();
        auto v = o.view(true);
        CHECK_EQ(std::string(v.content), std::string("Let me check."));
        CHECK(v.tool.has_value());
        json::Array tools;
        tools.push_back(json::parse(
            R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}})"));
        auto calls = parseToolCalls(*v.tool, tools);
        CHECK(calls.has_value() && calls->size() == 1);
        if (calls && !calls->empty()) {
            CHECK_EQ((*calls)[0].name, std::string("get_weather"));
            CHECK_EQ((*calls)[0].args, std::string("{\"city\":\"Taipei\"}"));
        }
        auto c2 = parseToolCalls("<tool_call>\n<function=get_weather>\n<parameter=city>\n123\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", tools);
        CHECK(c2 && (*c2)[0].args == "{\"city\":\"123\",\"days\":3}");
        auto c3 = parseToolCalls("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": [1, 2]}}</tool_call>", tools);
        CHECK(c3 && (*c3)[0].name == "f" && (*c3)[0].args == "{\"a\":[1,2]}");
        CHECK(!parseToolCalls("<tool_call>garbage</tool_call>", tools).has_value());
        CHECK(!parseToolCalls("<tool_call><function=></function></tool_call>", tools).has_value());
    }
    // parseParams
    {
        const json::Value v = json::parse(R"({"temperature":0,"top_k":5,"stop":["a",""],"max_tokens":7,"seed":-1,"stream":true,"stream_options":{"include_usage":true}})");
        const Params p = parseParams(v.asObject(), true, true);
        CHECK(p.temperature == 0.0f && p.top_k == 5 && p.stop.size() == 1 && p.max_tokens == 7u && p.stream && p.include_usage);
        CHECK(p.seed_given && p.seed == ~std::uint64_t(0));
        const Params q = parseParams(json::parse("{}").asObject(), false, true);
        CHECK(q.temperature == 0.7f && q.top_p == 0.8f);
        bool threw = false;
        try {
            parseParams(json::parse(R"({"top_p":0})").asObject(), true, true);
        } catch (const RequestError& e) {
            threw = std::string(e.what()) == "top_p must be in (0, 1]";
        }
        CHECK(threw);
        threw = false;
        try {
            parseParams(json::parse(R"({"n":2})").asObject(), true, true);
        } catch (const RequestError&) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        Timings t;
        t.cache_n = 1;
        t.prompt_n = 2;
        t.prompt_ms = 4;
        t.predicted_n = 3;
        t.predicted_ms = 6;
        std::string s;
        t.write(s);
        CHECK_EQ(s, std::string("{\"cache_n\":1,\"prompt_n\":2,\"prompt_ms\":4,\"prompt_per_token_ms\":2,\"prompt_per_second\":500,"
                                "\"predicted_n\":3,\"predicted_ms\":6,\"predicted_per_token_ms\":2,\"predicted_per_second\":500,"
                                "\"draft_n\":0,\"draft_n_accepted\":0}"));
        const std::string e = errorResponse(404, "not found");
        CHECK(e.find("HTTP/1.1 404 Not Found\r\n") == 0);
        CHECK(e.find("{\"error\":{\"message\":\"not found\",\"type\":\"not_found_error\",\"code\":404}}") != std::string::npos);
    }
}

void testSampler() {
    std::vector<Cand> cands(8);
    Sampler s;
    s.cands = &cands;
    s.inv_t = 1;
    s.top_k = 3;
    s.top_p = 1;
    s.min_p = 0;
    const float logits[8] = {1, 5, 3, 3, 0, -1, 2, 4};
    for (std::uint32_t i = 0; i < 8; ++i) cands[i] = {i, logits[i], 0};
    s.m = 5;
    double z = 0;
    for (float l : logits) z += std::exp(static_cast<double>(l - 5));
    s.sum = static_cast<float>(z);
    CHECK(filterCands(s, 8, false));
    CHECK(s.n == 3 && s.closed);
    CHECK(cands[0].id == 1 && cands[1].id == 7 && (cands[2].id == 2));
    double tot = 0;
    for (std::size_t i = 0; i < s.n; ++i) tot += cands[i].p;
    CHECK(std::fabs(tot - 1.0) < 1e-9);
    // top_k larger than the candidates and an open set: escalation
    s.top_k = 20;
    for (std::uint32_t i = 0; i < 8; ++i) cands[i] = {i, logits[i], 0};
    CHECK(!filterCands(s, 8, false));
    // uniforms keyed by (seed, index)
    Sampler a, b;
    a.seed = b.seed = 7;
    a.pos_idx = b.pos_idx = 3;
    CHECK(a.uniform() == b.uniform());
    b.pos_idx = 4;
    CHECK(a.uniform() != b.uniform());
    CHECK(a.uniform() >= 0 && a.uniform() < 1);
}

void testChatTokens() {
    ChatTokens ct(tok());
    CHECK(ct.im_start != ChatTokens::none && ct.im_end != ChatTokens::none && ct.think != ChatTokens::none &&
          ct.nl != ChatTokens::none && ct.system != ChatTokens::none);
    auto enc = [](const std::string& s) {
        const auto ids = tok().encode(s, true);
        return std::vector<std::uint32_t>(ids.begin(), ids.end());
    };
    const auto p1 = enc("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    CHECK_EQ(ct.thinkOpenSplit(p1), p1.size() - 1);
    const std::size_t b = ct.sysBoundary(p1);
    CHECK(b > 0 && b < p1.size() && p1[b - 2] == ct.im_end && p1[b - 1] == ct.nl && p1[b] == ct.im_start);
    // a system text that mentions <|im_end|> does not end the system message there
    const auto p1b = enc("<|im_start|>system\nA turn ends with <|im_end|> (not here).<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n");
    const std::size_t b1 = ct.sysBoundary(p1b);
    CHECK(b1 > 0 && p1b[b1] == ct.im_start && p1b[b1 + 1] != ct.system);
    const auto p2 = enc("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    CHECK_EQ(ct.thinkOpenSplit(p2), std::size_t(0));
    CHECK_EQ(ct.sysBoundary(p2), std::size_t(0));
    // the next turn re-renders "<think>\n\n</think>": "\n\n" is one token
    const auto p3 = enc("<think>\n\n</think>");
    CHECK(p3.size() == 3 && p3[0] == ct.think);
    const auto norm = ChatTokens::crlfToLf(tok());
    const auto crlf = enc("\r\n");
    const auto lf = enc("\n");
    CHECK(crlf.size() == 1 && lf.size() == 1 && norm[crlf[0]] == lf[0]);
}

// ---------------------------------------------------------------------------
// end-to-end tests (mock model)

ServerSetup baseSetup(bool mtp = true) {
    ServerSetup su;
    su.mc.parallel = 4;
    su.mc.slot_ctx = 16384;
    su.mc.pool_tokens = 65536;
    su.mc.mtp = mtp;
    su.eo.use_mtp = mtp;
    su.eo.n_draft = 4;
    su.eo.mtp_auto = true;
    su.eo.batch_drafts = {0, 8, 7, 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    su.eo.n_ck = 2;
    su.eo.n_spe = 2;
    su.eo.sys_min = 2048;
    su.eo.gather_ms = 0;
    su.eo.seed = 1;
    return su;
}

// Read-only compatibility endpoints (GET /props, /version) and the quiet 404 of
// other server types' probe paths.
void testCompatEndpoints() {
    ServerSetup su = baseSetup();
    su.eo.model_name = "mock-alias";
    su.eo.model_file = "models/sub\\mock-Q4_K_M.gguf";  // directories must never reach the response
    TestServer srv(su);
    for (const char* p : {"/props", "/v1/props"}) {
        const auto r = srv.get(p);
        CHECK_EQ(r.status, 200);
        const json::Value v = parseBody(r);
        CHECK(v.isObject());
        if (!v.isObject()) continue;
        const json::Value* dg = v.get("default_generation_settings");
        CHECK(dg && dg->isObject());
        if (dg && dg->isObject()) {
            const json::Value* n_ctx = dg->get("n_ctx");
            CHECK(n_ctx && n_ctx->asInt() == static_cast<std::int64_t>(su.mc.slot_ctx));
            const json::Value* m = dg->get("model");
            CHECK(m && m->isString() && m->asString() == "mock-alias");
        }
        const json::Value* ts = v.get("total_slots");
        CHECK(ts && ts->asInt() == static_cast<std::int64_t>(su.mc.parallel));
        const json::Value* mp = v.get("model_path");
        CHECK(mp && mp->isString() && mp->asString() == "mock-Q4_K_M.gguf");
        const json::Value* al = v.get("model_alias");
        CHECK(al && al->isString() && al->asString() == "mock-alias");
        const json::Value* mod = v.get("modalities");
        CHECK(mod && mod->isObject() && mod->get("vision") && mod->get("vision")->isBool());
        const json::Value* bi = v.get("build_info");
        CHECK(bi && bi->isString() && bi->asString().starts_with("whirl "));
        CHECK(r.body.find("models/") == std::string::npos && r.body.find('\\') == std::string::npos);
    }
    CHECK_EQ(srv.post("/props", "{}").status, 405);
    const auto ver = srv.get("/version");
    CHECK_EQ(ver.status, 200);
    const json::Value vv = parseBody(ver);
    const json::Value* vs = vv.isObject() ? vv.get("version") : nullptr;
    const json::Value* vn = vv.isObject() ? vv.get("name") : nullptr;
    CHECK(vs && vs->isString() && !vs->asString().empty());
    CHECK(vn && vn->isString() && vn->asString() == "whirl");
    if (vs && vs->isString()) {
        const json::Value pv = parseBody(srv.get("/props"));
        const json::Value* bi = pv.isObject() ? pv.get("build_info") : nullptr;
        CHECK(bi && bi->isString() && bi->asString() == "whirl " + vs->asString());
    }
    // probes for LM Studio / Ollama: 404 without a warning line
    g_log.clear();
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(srv.get("/api/v1/models").status, 404);
        CHECK_EQ(srv.get("/api/tags").status, 404);
    }
    CHECK_EQ(g_log.count(" W GET /api/"), 0u);
    CHECK(g_log.count("GET /api/tags -> 404") <= 1);
    // an unknown path still warns
    CHECK_EQ(srv.get("/nope-compat").status, 404);
    CHECK_EQ(g_log.count(" W GET /nope-compat -> 404"), 1u);
    // reasoning effort: an unknown value is a warning, not a 400; aliases and the
    // "reasoning" object are accepted and logged as the canonical effort
    g_log.clear();
    const std::string q = R"({"messages":[{"role":"user","content":"q"}],"max_tokens":4,"temperature":0,)";
    CHECK_EQ(srv.post("/v1/chat/completions", q + R"("reasoning_effort":"turbo"})").status, 200);
    CHECK_EQ(g_log.count("unknown reasoning_effort \"turbo\" ignored"), 1u);
    CHECK_EQ(srv.post("/v1/chat/completions", q + R"("reasoning":{"effort":"ultra"}})").status, 200);
    CHECK_EQ(g_log.count("thinking on, reasoning_effort xhigh"), 1u);
    CHECK_EQ(srv.post("/v1/chat/completions", q + R"("reasoning":{"enabled":false}})").status, 200);
    CHECK_EQ(g_log.count("thinking off"), 1u);
}

// WDDM-LOG (issues #5 / #6): injected WDDM samples -> request-end vram line, spill warning, recovery
// line, /health "vram" object; model output is not affected (same text with and without a probe)
void testVramWatch() {
    auto waitFor = [](const std::function<bool()>& f, int ms) {
        for (int i = 0; i < ms / 10; ++i) {
            if (f()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return f();
    };
    std::mutex mu;
    whirl::wddmwatch::Sample cur;
    constexpr std::uint64_t MiB = 1ull << 20;
    cur.ok = true;
    cur.local_usage = 29000 * MiB;
    cur.local_budget = 31232 * MiB;
    cur.nonlocal_usage = 400 * MiB;
    auto set = [&](std::uint64_t used, std::uint64_t budget, std::uint64_t shared) {
        std::lock_guard<std::mutex> lk(mu);
        cur.local_usage = used * MiB;
        cur.local_budget = budget * MiB;
        cur.nonlocal_usage = shared * MiB;
    };
    const std::string q = R"({"messages":[{"role":"user","content":"q"}],"max_tokens":8,"temperature":0})";
    std::string plain_text;
    {
        TestServer plain(baseSetup());
        const auto r = plain.post("/v1/chat/completions", q);
        CHECK_EQ(r.status, 200);
        plain_text = r.body.substr(r.body.find("\"choices\""), r.body.find("\"usage\"") - r.body.find("\"choices\""));
        CHECK(plain.get("/health").body.find("\"vram\"") == std::string::npos);  // no probe: no object
    }
    g_log.clear();
    ServerSetup su = baseSetup();
    su.vram_probe = [&] {
        std::lock_guard<std::mutex> lk(mu);
        return cur;
    };
    TestServer srv(su);
    CHECK_EQ(g_log.count("vram watch: vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start)"), 1u);
    CHECK(srv.get("/health").body.find("\"vram\":{\"local_mib\":29000,\"budget_mib\":31232,\"shared_mib\":400,\"spill\":false}") !=
          std::string::npos);
    auto r1 = srv.post("/v1/chat/completions", q);
    CHECK_EQ(r1.status, 200);
    CHECK(r1.body.substr(r1.body.find("\"choices\""), r1.body.find("\"usage\"") - r1.body.find("\"choices\"")) == plain_text);
    // the request-end lines are logged after the response is sent
    CHECK(waitFor([&] { return g_log.count("| vram local 28.32 / budget 30.50 GiB, shared 400 MiB (+0 MiB since start)") >= 1; }, 5000));
    CHECK_EQ(g_log.count(" W GPU memory spill"), 0u);
    // another program takes VRAM: the budget drops under WHIRL's usage, Windows moves 600 MiB out
    set(29000, 28672, 1000);
    CHECK_EQ(srv.post("/v1/chat/completions", q).status, 200);
    CHECK(waitFor([&] {
        return g_log.count(" W GPU memory spill: WHIRL's GPU memory is 0.32 GiB over the Windows video memory (WDDM) budget, and "
                           "Windows has likely moved part of WHIRL's GPU buffers to system RAM (shared GPU memory +600 MiB since startup).") == 1;
    }, 5000));
    CHECK(waitFor([&] {
        return srv.get("/health").body.find("\"vram\":{\"local_mib\":29000,\"budget_mib\":28672,\"shared_mib\":1000,\"spill\":true}") !=
               std::string::npos;
    }, 5000));
    const std::size_t n_vram = g_log.count("| vram local 28.32 / budget 28.00 GiB");
    CHECK_EQ(srv.post("/v1/chat/completions", q).status, 200);  // still over, within the minute: no repeat
    CHECK(waitFor([&] { return g_log.count("| vram local 28.32 / budget 28.00 GiB") > n_vram; }, 5000));
    CHECK_EQ(g_log.count("GPU memory spill continues"), 0u);
    // the other program exits
    set(29000, 31232, 450);
    CHECK_EQ(srv.post("/v1/chat/completions", q).status, 200);
    CHECK(waitFor([&] { return g_log.count(" I GPU memory spill is over (it lasted ") == 1; }, 5000));
    CHECK(waitFor([&] { return srv.get("/health").body.find("\"spill\":false") != std::string::npos; }, 5000));

    // review P3-7: while idle the probe is still sampled (here every 0.2 s; the server: once a minute), so /health and
    // the spill lines follow without a request; P3-3: the pool size and other programs' VRAM are in the warning
    {
        g_log.clear();
        set(29000, 31232, 400);
        ServerSetup si = baseSetup();
        si.vram_probe = su.vram_probe;
        si.vram_cfg.idle_sample_s = 0.2;
        si.vram_cfg.pool_tokens = 334080;
        si.vram_others = [] { return std::optional<whirl::wddmwatch::Others>(whirl::wddmwatch::Others{3328 * MiB, 205 * MiB}); };
        TestServer idle(si);
        CHECK(idle.get("/health").body.find("\"spill\":false") != std::string::npos);
        set(29000, 28672, 1000);  // no request
        CHECK(waitFor([&] { return idle.get("/health").body.find("\"budget_mib\":28672,\"shared_mib\":1000,\"spill\":true") != std::string::npos; },
                      5000));
        CHECK(waitFor([&] { return g_log.count(" W GPU memory spill: ") == 1; }, 5000));
        CHECK_EQ(g_log.count("(the KV pool is now 334080 tokens)"), 1u);
        CHECK_EQ(g_log.count("Other programs use 3.25 GiB of VRAM on this GPU now (0.20 GiB when the server started)."), 1u);
        set(29000, 31232, 400);
        CHECK(waitFor([&] { return g_log.count(" I GPU memory spill is over (it lasted ") == 1; }, 5000));
        CHECK(waitFor([&] { return idle.get("/health").body.find("\"spill\":false") != std::string::npos; }, 5000));
    }
    // review P3-4: a spill already there at startup (against the sizing-time baseline) shows in /health at once; the
    // startup line already warned, so the watch does not repeat it
    {
        g_log.clear();
        set(29000, 31232, 1000);
        ServerSetup ss = baseSetup();
        ss.vram_probe = su.vram_probe;
        whirl::wddmwatch::Sample base0 = cur;
        base0.nonlocal_usage = 400 * MiB;  // the read the pool was sized from
        ss.vram_base = base0;
        ss.vram_startup_warned = true;
        TestServer st(ss);
        CHECK(st.get("/health").body.find("\"spill\":true") != std::string::npos);
        CHECK_EQ(g_log.count(" W GPU memory spill"), 0u);
        CHECK_EQ(g_log.count("(+600 MiB since start)"), 1u);  // the vram watch line, against the sizing baseline
        set(29000, 31232, 400);
    }
}

// v0.2.2.2 memory governor on the mock engine with injected budgets: the WDDM local usage is the mock's live device
// bytes, so freeing checkpoint buffers or re-allocating the pool lowers it as on the GPU. (1) nothing happens while the
// budget has room; (2) the budget drops while a long request runs: idle cache is freed, new requests queue, the running
// request's output is untouched and the pool is re-allocated smaller only once every slot is idle; (3) the budget
// recovers: after the hold the pool grows back to its startup size (hysteresis); every output equals the reference.
void testGovernor() {
    auto waitFor = [](const std::function<bool()>& f, int ms) {
        for (int i = 0; i < ms / 10; ++i) {
            if (f()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return f();
    };
    constexpr std::uint64_t MiB = 1ull << 20;
    std::atomic<std::uint64_t> budget{28000 * MiB + (1ull << 40)};  // room until the test sets it
    std::atomic<std::uint64_t> extra{0};  // usage outside the pool that grew after startup (R9700 S2: 0.19 GiB)
    std::atomic<NullBackend*> opsp{nullptr};
    const std::uint64_t fixed = 28000 * MiB;  // "the weights" (not allocated by the mock)
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.pool_tokens = 131072;
    su.mc.slot_ctx = 16384;
    su.mc.cycle_us = 2000;
    su.tier = true;
    su.tier_ram_mb = 256;
    su.vram_probe = [&] {
        whirl::wddmwatch::Sample x;
        NullBackend* o = opsp.load();
        x.ok = true;
        x.local_usage = fixed + extra.load() + (o ? o->deviceBytes() : 0);
        x.local_budget = budget.load();
        x.nonlocal_usage = 400 * MiB;
        return x;
    };
    su.vram_cfg.idle_sample_s = 0.2;
    su.vram_cfg.governor = true;
    whirl::memgov::Config gc;
    gc.page = qwen35::kv_page;
    gc.pool_start = 131072;
    gc.pool_min = 4096;
    gc.ctx = 16384;
    gc.recover_s = 0.4;
    gc.hold_s = 0.6;
    gc.settle_s = 0.5;
    su.gov = gc;
    std::uint64_t pool0 = 0;  // pool bytes at startup, set below
    su.gov_margin = [&](std::uint64_t) { return pool0 / 2; };
    g_log.clear();
    TestServer srv(su);
    for (const KvArr& a : srv.model().kvArrays()) pool0 += a.row * 131072;
    const std::uint64_t M = pool0 / 2;
    const std::uint64_t local0 = fixed + srv.ops().deviceBytes();
    budget = local0 + M;  // exactly the margin free, as the startup sizing leaves it: no action
    opsp = &srv.ops();
    const auto pA = makePrompt(3000, 71), pB = makePrompt(700, 72), pC = makePrompt(2600, 73);
    const std::uint32_t gA = 1500, gB = 8, gC = 16;
    const std::string refA = decode(srv.model().greedyReference(pA, gA, true));
    const std::string refB = decode(srv.model().greedyReference(pB, gB, true));
    const std::string refC = decode(srv.model().greedyReference(pC, gC, true));
    // (1) room: requests run, the governor stays quiet
    CHECK(completionText(srv.post("/v1/completions", completionBody(pB, gB))) == refB);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK_EQ(g_log.count("memory governor:"), 0u);
    CHECK(srv.get("/health").body.find("\"governor\":{\"on\":true,\"reduced\":false,\"pool_tokens\":131072,") != std::string::npos);
    // (2) a long request runs; the budget drops to a quarter of the margin free
    std::string gotA, gotC;
    std::thread ta([&] { gotA = completionText(srv.post("/v1/completions", completionBody(pA, gA))); });
    CHECK(waitFor([&] { return srv.get("/health").body.find("\"slots_busy\":1") != std::string::npos; }, 5000));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    budget = local0 + M / 4;
    // B ends while A runs: the request-end sample sees the drop (A busy): B's slot is parked, new requests queue
    CHECK(completionText(srv.post("/v1/completions", completionBody(pB, gB))) == refB);
    CHECK(waitFor([&] { return g_log.count(" W memory governor: new requests queue until the running request(s) finish") == 1; }, 3000));
    CHECK_EQ(g_log.count("memory governor: RAM tier kept at 256 MiB"), 1u);
    CHECK(waitFor([&] { return g_log.count("memory governor: freed reusable cache:") >= 1; }, 3000));
    CHECK(waitFor([&] { return srv.get("/health").body.find("\"admission_held\":true") != std::string::npos; }, 3000));
    std::thread tc([&] { gotC = completionText(srv.post("/v1/completions", completionBody(pC, gC))); });
    CHECK(waitFor([&] { return srv.get("/health").body.find("\"queued\":1") != std::string::npos; }, 3000));
    // the running request's KV is untouched: no re-allocation while it runs
    CHECK_EQ(srv.model().resizes(), 0u);
    CHECK(gotA.empty());
    ta.join();
    tc.join();
    CHECK(gotA == refA);
    CHECK(gotC == refC);
    CHECK_EQ(srv.model().resizes(), 1u);
    CHECK(srv.model().poolPages() * qwen35::kv_page < 131072u);
    CHECK_EQ(g_log.count(" W memory governor: KV pool re-allocated 131072 -> "), 1u);
    CHECK(g_log.count("memory governor: checkpoint buffers restored") >= 1);
    // after the shrink the full margin is free again (within a page and the parked buffers' rounding)
    const std::uint64_t local1 = fixed + srv.ops().deviceBytes();
    CHECK(budget.load() > local1 && budget.load() - local1 + qwen35::kv_page * 96 >= M);
    // the same prompt again: served from the RAM tier or recomputed, the output is the same
    CHECK(completionText(srv.post("/v1/completions", completionBody(pC, gC))) == refC);
    CHECK_EQ(srv.model().resizes(), 1u);  // stable: no second shrink
    // a prompt longer than the smaller context for new requests is rejected as too long (and logged why)
    CHECK(srv.get("/props").body.find("\"n_ctx\":16384") != std::string::npos);  // the pool still holds 16384
    // (3) the other program exits: the budget is back at its startup value (not more: REVIEW-GOV-1 P2-1, the real GPU never
    // gives more back), and usage outside the pool grew by M/8 meanwhile; after recover_s + hold the pool grows straight back
    // to its startup size when idle (one grow) and the episode ends
    budget = local0 + M;
    extra = M / 8;
    CHECK(waitFor([&] { return srv.model().resizes() == 2u; }, 6000));
    CHECK(waitFor([&] { return g_log.count("memory governor: back to the startup settings (KV pool 131072 tokens") == 1; }, 3000));
    CHECK_EQ(srv.model().poolPages() * qwen35::kv_page, 131072u);
    CHECK(waitFor([&] { return srv.get("/health").body.find("\"reduced\":false,\"pool_tokens\":131072,") != std::string::npos; }, 3000));
    CHECK(srv.get("/health").body.find("\"ctx_per_request\":16384,") != std::string::npos);
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    CHECK_EQ(srv.model().resizes(), 2u);  // stable with 7/8 of the margin free: no shrink back
    CHECK(completionText(srv.post("/v1/completions", completionBody(pA, 64))) == decode(srv.model().greedyReference(pA, 64, true)));
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
    // the spill warning names the governor (the watch's advice), logged after the governor's own lines
    if (const std::size_t w = g_log.count(" W GPU memory spill"); w > 0) CHECK(g_log.count("WHIRL's memory governor frees cache") >= 1);

    // (4) a re-allocation that fails keeps the old pool (logged), and the server keeps serving
    {
        g_log.clear();
        srv.model().fail_resize = true;
        budget = local0 + M / 4;
        CHECK(waitFor([&] { return g_log.count("memory governor: the KV pool could not be re-allocated to ") >= 1; }, 3000));
        CHECK(completionText(srv.post("/v1/completions", completionBody(pB, gB))) == refB);
        budget = local0 + 3 * M;
    }
}

// REVIEW-GOV-1 P2-2: a checkpoint-buffer restore that fails part way (allocation error) frees what it got, the slot stays
// parked, the next sample tries again, and once it succeeds the device memory is exactly what it was before the episode
// (no leaked buffers). nth = 2: the parked slot's second checkpoint fails; nth = 4: the second shared prefix checkpoint.
void testGovernorRestore() {
    auto waitFor = [](const std::function<bool()>& f, int ms) {
        for (int i = 0; i < ms / 10; ++i) {
            if (f()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return f();
    };
    constexpr std::uint64_t MiB = 1ull << 20;
    for (const std::uint64_t nth : {2ull, 4ull}) {
        std::atomic<std::uint64_t> budget{28000 * MiB + (1ull << 40)};
        std::atomic<NullBackend*> opsp{nullptr};
        const std::uint64_t fixed = 28000 * MiB;
        ServerSetup su = baseSetup();
        su.mc.parallel = 2;
        su.mc.pool_tokens = 131072;
        su.mc.slot_ctx = 16384;
        su.mc.cycle_us = 2000;
        su.tier = true;
        su.tier_ram_mb = 256;
        su.vram_probe = [&] {
            whirl::wddmwatch::Sample x;
            NullBackend* o = opsp.load();
            x.ok = true;
            x.local_usage = fixed + (o ? o->deviceBytes() : 0);
            x.local_budget = budget.load();
            x.nonlocal_usage = 400 * MiB;
            return x;
        };
        su.vram_cfg.idle_sample_s = 0.2;
        su.vram_cfg.governor = true;
        whirl::memgov::Config gc;
        gc.page = qwen35::kv_page;
        gc.pool_start = 131072;
        gc.pool_min = 4096;
        gc.ctx = 16384;
        gc.recover_s = 0.4;
        gc.hold_s = 0.6;
        gc.settle_s = 0.5;
        su.gov = gc;
        std::uint64_t pool0 = 0;
        su.gov_margin = [&](std::uint64_t) { return pool0 / 2; };
        g_log.clear();
        TestServer srv(su);
        for (const KvArr& a : srv.model().kvArrays()) pool0 += a.row * 131072;
        const std::uint64_t M = pool0 / 2;
        const std::uint64_t dev0 = srv.ops().deviceBytes();
        const std::uint64_t local0 = fixed + dev0;
        budget = local0 + M;
        opsp = &srv.ops();
        const auto pA = makePrompt(3000, 81), pB = makePrompt(700, 82);
        const std::uint32_t gA = 1500, gB = 8;
        const std::string refA = decode(srv.model().greedyReference(pA, gA, true));
        const std::string refB = decode(srv.model().greedyReference(pB, gB, true));
        std::string gotA;
        std::thread ta([&] { gotA = completionText(srv.post("/v1/completions", completionBody(pA, gA))); });
        CHECK(waitFor([&] { return srv.get("/health").body.find("\"slots_busy\":1") != std::string::npos; }, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        budget = local0 + M / 4;
        // B ends while A runs: B's slot (and the shared checkpoints) are parked: their buffers are freed
        CHECK(completionText(srv.post("/v1/completions", completionBody(pB, gB))) == refB);
        CHECK(waitFor([&] { return g_log.count("MiB of checkpoint buffers freed now") >= 1; }, 3000));
        CHECK(srv.ops().deviceBytes() < dev0);
        // the budget has room again before A ends: the restore runs; its nth allocation fails once
        srv.ops().failMalloc(nth);
        budget = local0 + 3 * M;
        CHECK(waitFor([&] { return g_log.count("memory governor: restoring the checkpoint buffers failed") >= 1; }, 6000));
        CHECK(waitFor([&] { return g_log.count("memory governor: checkpoint buffers restored") >= 1; }, 6000));
        ta.join();
        CHECK(gotA == refA);
        CHECK(waitFor([&] { return srv.get("/health").body.find("\"reduced\":false,") != std::string::npos; }, 3000));
        CHECK_EQ(srv.model().resizes(), 0u);
        // exactly the buffers of before: nothing leaked by the failed try
        CHECK_EQ(srv.ops().deviceBytes(), dev0);
        CHECK(completionText(srv.post("/v1/completions", completionBody(pB, gB))) == refB);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK(srv.get("/health").body.find("\"parked_mib\":0,") != std::string::npos);
    }
}

// v0.2.2.1: effective values after the startup fit in /props and /health; a prompt over a context per request that
// startup lowered is rejected as usual, and the log says why
void testStartupAdjust() {
    ServerSetup su = baseSetup();
    su.startup_json = R"({"ctx_per_request":16384,"pool_tokens":65536,"parallel":4,"kv_ram_mb":0,"wddm_margin_mib":2048,)"
                      R"("other_processes_mib":3307,"display_attached":false,"adjusted":["--ctx-per-slot 131072 -> 16384"]})";
    su.ctx_note = "adjusted at startup to fit the GPU memory: --ctx-per-slot 131072 -> 16384";
    TestServer srv(su);
    const auto h = srv.get("/health");
    CHECK_EQ(h.status, 200);
    CHECK(h.body.find(",\"startup\":{\"ctx_per_request\":16384,") != std::string::npos);
    CHECK(h.body.find("\"adjusted\":[\"--ctx-per-slot 131072 -> 16384\"]") != std::string::npos);
    const auto pr = srv.get("/props");
    CHECK_EQ(pr.status, 200);
    CHECK(pr.body.find(",\"startup\":{\"ctx_per_request\":16384,") != std::string::npos);
    g_log.clear();
    const auto p = makePrompt(16400, 3);
    auto r = srv.post("/v1/completions", completionBody(p, 4));
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("prompt is too long for the context size") != std::string::npos);
    CHECK(g_log.count("does not fit the context per request of 16384 tokens (adjusted at startup to fit the GPU memory: "
                      "--ctx-per-slot 131072 -> 16384)") == 1);
    // without an adjustment: no startup object, no note
    TestServer plain(baseSetup());
    CHECK(plain.get("/health").body.find("\"startup\"") == std::string::npos);
    g_log.clear();
    CHECK_EQ(plain.post("/v1/completions", completionBody(p, 4)).status, 400);
    CHECK_EQ(g_log.count("does not fit the context per request"), 0u);
}

void testEndpoints() {
    TestServer srv(baseSetup());
    auto h = srv.get("/health");
    CHECK_EQ(h.status, 200);
    CHECK(h.body.find("\"status\":\"ok\"") != std::string::npos && h.body.find("\"slots\":4") != std::string::npos);
    auto m = srv.get("/v1/models");
    CHECK_EQ(m.status, 200);
    CHECK(m.body.find("\"id\":\"mock\"") != std::string::npos && m.body.find("\"mtp\":true") != std::string::npos);
    CHECK_EQ(srv.get("/nope").status, 404);
    CHECK_EQ(srv.get("/v1/chat/completions").status, 405);
    CHECK_EQ(srv.post("/health", "").status, 405);
    auto opt = test::httpRequest("127.0.0.1", srv.port(), "OPTIONS", "/v1/chat/completions", {}, 600,
                                 "Origin: http://localhost:3000\r\n");
    CHECK_EQ(opt.status, 204);
    CHECK(opt.headers.find("Access-Control-Allow-Methods: GET, POST, OPTIONS") != std::string::npos);
    auto bad = srv.post("/v1/chat/completions", "{not json");
    CHECK_EQ(bad.status, 400);
    CHECK(bad.body.find("request body is not valid JSON") != std::string::npos);
    CHECK(srv.post("/v1/chat/completions", "{}").body.find("'messages' is required") != std::string::npos);
    CHECK(srv.post("/v1/completions", R"({"prompt":[]})").body.find("'prompt' must not be empty") != std::string::npos);
    CHECK(srv.post("/v1/completions", R"({"prompt":[999999999]})").body.find("prompt token ids must be integers") !=
          std::string::npos);
    auto img = srv.post("/v1/chat/completions",
                        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]})");
    CHECK_EQ(img.status, 400);
    // too long for the context
    {
        const auto p = makePrompt(16400, 3);
        auto r = srv.post("/v1/completions", completionBody(p, 4));
        CHECK_EQ(r.status, 400);
        CHECK(r.body.find("prompt is too long for the context size") != std::string::npos);
    }
    CHECK_EQ(srv.model().poison(), 0u);
}

// ---- connection handling (timeouts, connection cap, header / body limits)

bool hasHeader(const HttpResult& r, std::string_view line) { return r.headers.find(line) != std::string::npos; }

int rawStatus(const std::string& resp) {
    return resp.size() > 12 && resp.compare(0, 5, "HTTP/") == 0 ? std::atoi(resp.c_str() + 9) : 0;
}

// one raw request; the response status (0: closed without a response)
int rawRequestStatus(std::uint16_t port, const std::string& req) {
    const auto s = test::rawConnect("127.0.0.1", port);
    if (s == test::kNoSock) return -1;
    test::rawSend(s, req);
    const std::string resp = test::rawRecvAll(s, 5);
    test::rawClose(s);
    return rawStatus(resp);
}

void testHttpLimits() {
    ServerSetup su = baseSetup();
    su.http.recv_timeout_ms = 300;
    su.http.header_timeout_ms = 800;
    TestServer srv(su);
    const std::uint16_t port = srv.port();
    // an idle client (connects, sends nothing) is closed after the read timeout
    {
        const auto s = test::rawConnect("127.0.0.1", port);
        CHECK(s != test::kNoSock);
        const double t0 = nowSeconds();
        bool closed = false;
        const std::string resp = test::rawRecvAll(s, 5, &closed);
        const double dt = nowSeconds() - t0;
        test::rawClose(s);
        CHECK(closed);
        CHECK(resp.empty());
        CHECK(dt >= 0.2 && dt < 2.0);
        if (g_verbose) std::printf("  idle client closed after %.2f s\n", dt);
    }
    // a slow client (one header byte per 100 ms: every read in time, the whole
    // header not) gets 408 at the header deadline
    {
        const auto s = test::rawConnect("127.0.0.1", port);
        CHECK(test::rawSend(s, "GET /health HTTP/1.1\r\n"));
        std::atomic<bool> quit{false};
        std::thread drip([&] {
            for (int i = 0; i < 40 && !quit.load(); ++i) {
                if (!test::rawSend(s, "X")) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        const double t0 = nowSeconds();
        bool closed = false;
        const std::string resp = test::rawRecvAll(s, 5, &closed);
        const double dt = nowSeconds() - t0;
        quit = true;
        drip.join();
        test::rawClose(s);
        CHECK_EQ(rawStatus(resp), 408);
        if (rawStatus(resp) != 408) std::printf("  slow client got %zu bytes: [%s]\n", resp.size(), resp.substr(0, 200).c_str());
        CHECK(dt >= 0.6 && dt < 2.5);
        if (g_verbose) std::printf("  slow client: 408 after %.2f s\n", dt);
    }
    // header limits: too many lines, too many bytes, one line too long
    {
        std::string many = "GET /health HTTP/1.1\r\n";
        for (int i = 0; i < 150; ++i) many += std::format("X-H{}: v\r\n", i);
        CHECK_EQ(rawRequestStatus(port, many + "\r\n"), 431);
        std::string big = "GET /health HTTP/1.1\r\n";
        for (int i = 0; i < 10; ++i) big += std::format("X-B{}: {}\r\n", i, std::string(8000, 'b'));
        CHECK_EQ(rawRequestStatus(port, big + "\r\n"), 431);
        CHECK_EQ(rawRequestStatus(port, "GET /health HTTP/1.1\r\nX-L: " + std::string(70000, 'l') + "\r\n\r\n"), 431);
        std::string ok = "GET /health HTTP/1.1\r\n";
        for (int i = 0; i < 20; ++i) ok += std::format("X-H{}: v\r\n", i);
        CHECK_EQ(rawRequestStatus(port, ok + "\r\n"), 200);
    }
    // Content-Length: overflow-safe, digits only, one value
    {
        auto withLen = [&](const std::string& h) {
            return rawRequestStatus(port, "POST /v1/completions HTTP/1.1\r\n" + h + "\r\n{}");
        };
        CHECK_EQ(withLen("Content-Length: 18446744073709551617\r\n"), 413);  // 2^64 + 1 (wrapped to 1 before)
        CHECK_EQ(withLen("Content-Length: 99999999999999999999999999999\r\n"), 413);
        CHECK_EQ(withLen("Content-Length: 67108865\r\n"), 413);  // 64 MiB + 1
        CHECK_EQ(withLen("Content-Length: 12a\r\n"), 400);
        CHECK_EQ(withLen("Content-Length: -1\r\n"), 400);
        CHECK_EQ(withLen("Content-Length: 2\r\nContent-Length: 3\r\n"), 400);
        CHECK_EQ(withLen("Content-Length: 2\r\n"), 400);  // "{}" -> 'prompt' is required
    }
    // a claimed 64 MiB body that never comes: closed after the read timeout
    {
        const auto s = test::rawConnect("127.0.0.1", port);
        test::rawSend(s, "POST /v1/completions HTTP/1.1\r\nContent-Length: 67108864\r\n\r\n{\"prompt\"");
        bool closed = false;
        const double t0 = nowSeconds();
        test::rawRecvAll(s, 5, &closed);
        test::rawClose(s);
        CHECK(closed && nowSeconds() - t0 < 2.0);
    }
    // REVIEW-ARCH3: this request once took 21 s and failed (1 of 10 full runs). Print what the client saw (a connect
    // timeout carries the local port and its TCP rows: the open CB-FLAKE lost-connect item), not only the status.
    const auto h = srv.get("/health");
    CHECK_EQ(h.status, 200);
    if (h.status != 200) std::fprintf(stderr, "  /health after the limit checks: status %d after %.0f ms: %s\n", h.status, h.ms, h.err.c_str());
}

void testHttpConnCap() {
    ServerSetup su = baseSetup();
    su.http.recv_timeout_ms = 10000;
    su.http.max_conn = 4;
    TestServer srv(su);
    std::vector<std::uintptr_t> idle;
    for (int i = 0; i < 4; ++i) idle.push_back(test::rawConnect("127.0.0.1", srv.port()));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto over = srv.get("/health");
    CHECK_EQ(over.status, 503);
    CHECK(over.body.find("too many open connections") != std::string::npos);
    for (auto s : idle) test::rawClose(s);
    int st = 0;
    for (int i = 0; i < 40 && st != 200; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        st = srv.get("/health").status;
    }
    CHECK_EQ(st, 200);
    // stop() does not wait for idle connections (their read timeout is 10 s here)
    idle.clear();
    for (int i = 0; i < 3; ++i) idle.push_back(test::rawConnect("127.0.0.1", srv.port()));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const double t0 = nowSeconds();
    srv.shutdown();
    const double dt = nowSeconds() - t0;
    CHECK(dt < 1.5);
    if (g_verbose) std::printf("  stop() with 3 idle connections: %.2f s\n", dt);
    for (auto s : idle) test::rawClose(s);
}

// A streaming client that stops reading (connection open) must not stall the
// engine: another request completes, and the stuck one is dropped after the send
// timeout with its slot freed.
void testHttpStuckReader() {
    ServerSetup su = baseSetup();
    su.mc.cycle_us = 2000;  // the stuck stream generates longer than the send timeout
    // the engine must not wait for the stuck client at all: the other request
    // finishes well inside the send timeout (a blocking send with only
    // SO_SNDTIMEO would hold every slot for up to 2 s)
    su.http.send_timeout_ms = 2000;
    su.http.sndbuf = 4096;
    TestServer srv(su);
    g_log.clear();
    const auto a = test::rawConnect("127.0.0.1", srv.port(), 1024);
    CHECK(a != test::kNoSock);
    const std::string body = completionBody(makePrompt(64, 71), 12000, true);
    CHECK(test::rawSend(a, std::format("POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: "
                                       "{}\r\n\r\n{}",
                                       body.size(), body)));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const auto b = test::httpRequest("127.0.0.1", srv.port(), "POST", "/v1/completions",
                                     completionBody(makePrompt(64, 72), 16), 15);
    CHECK_EQ(b.status, 200);
    CHECK(b.ms < 1000);
    if (g_verbose) std::printf("  request beside a stuck reader: %.0f ms\n", b.ms);
    bool lost = false;
    for (int i = 0; i < 200 && !lost; ++i) {
        lost = g_log.count("client connection lost") > 0;
        if (!lost) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(lost);
    bool idle = false;
    for (int i = 0; i < 100 && !idle; ++i) {
        idle = srv.get("/health").body.find("\"slots_busy\":0") != std::string::npos;
        if (!idle) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(idle);
    test::rawClose(a);
    CHECK_EQ(srv.post("/v1/completions", completionBody(makePrompt(32, 73), 8)).status, 200);
}

// REVIEW-022 F2: an HTTP/1.0 request must not get Transfer-Encoding (RFC 9112 section 6.1). Since HTTP-DROP an
// SSE stream is chunked for HTTP/1.1; for HTTP/1.0 it goes out unframed and ends at the server's close, as
// before. Both streams carry the events and end with [DONE].
void testHttpStream10() {
    ServerSetup su = baseSetup();
    TestServer srv(su);
    const std::string body = completionBody(makePrompt(64, 91), 8, true);
    for (const char* ver : {"HTTP/1.0", "HTTP/1.1"}) {
        const auto s = test::rawConnect("127.0.0.1", srv.port());
        CHECK(s != test::kNoSock);
        if (s == test::kNoSock) continue;
        test::rawSend(s, std::string("POST /v1/completions ") + ver + "\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\n\r\n" + body);
        bool closed = false;
        const std::string resp = test::rawRecvAll(s, 20, &closed);
        test::rawClose(s);
        const std::size_t he = resp.find("\r\n\r\n");
        std::string h = resp.substr(0, he == std::string::npos ? 0 : he + 2);
        for (auto& ch : h) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        const bool te = h.find("\r\ntransfer-encoding: chunked\r\n") != std::string::npos;
        const std::string b = he == std::string::npos ? std::string() : resp.substr(he + 4);
        CHECK_EQ(rawStatus(resp), 200);
        CHECK(closed);
        CHECK(b.find("data: [DONE]") != std::string::npos);
        if (std::string_view(ver) == "HTTP/1.0") {
            CHECK(!te);
            CHECK(b.starts_with("data: "));  // no chunk-size line in front of the events
        } else {
            CHECK(te);
            CHECK(b.ends_with("0\r\n\r\n"));  // the last chunk
        }
        if (g_verbose) std::printf("  %s stream: %zu bytes, chunked %d\n", ver, resp.size(), te ? 1 : 0);
    }
}

// HTTP-DROP: a connect to the server that is never answered (WSAETIMEDOUT after 21 s, seen in the
// server gate about once per 200 requests). The server closed every connection first, so the server
// side of each finished connection sat in TIME_WAIT for 120 s; a new connection from the same client
// port (Windows hands ephemeral ports out again) whose SYN that TIME_WAIT does not accept -- after a
// request of tens of KB, as agent prompts are -- is silently dropped, and the client retries for 21 s.
// Here every request reuses the port of the previous one. A connect must be answered (or refused
// because the client's own TIME_WAIT holds the 4-tuple: the OS then picks another port) at once, never
// left waiting; and a finished request must leave no server-side TIME_WAIT behind.
void testHttpPortReuse() {
    ServerSetup su = baseSetup();
    TestServer srv(su);
    int n = 0, slow = 0, failed = 0, blocked = 0, srv_tw = 0;
    double worst = 0;
    const auto big = completionBody(makePrompt(12000, 81), 4);  // ~90 KB body, as the gate's chat requests
    const auto small = completionBody(makePrompt(64, 82), 4);
    const auto bigs = completionBody(makePrompt(12000, 83), 4, true);
    const int reps = std::getenv("WHIRL_TEST_REUSE_REPS") ? std::max(1, std::atoi(std::getenv("WHIRL_TEST_REUSE_REPS"))) : 12;
    std::uint16_t lp = test::freeLocalPort();
    CHECK(lp != 0);
    for (int i = 0; i < reps && failed == 0; ++i) {
        const std::string& body = i % 3 == 0 ? big : i % 3 == 1 ? small : bigs;
        auto r = test::httpRequestFrom(lp, "127.0.0.1", srv.port(), "POST", "/v1/completions", body, 60);
        ++n;
        if (r.status != 200 && r.err.find("WSA error 10048") != std::string::npos && r.ms < 1000) {
            ++blocked;  // the 4-tuple is in the client's own TIME_WAIT: refused at once, so take another port
            lp = test::freeLocalPort();
            r = test::httpRequestFrom(lp, "127.0.0.1", srv.port(), "POST", "/v1/completions", body, 60);
        }
        worst = std::max(worst, r.ms);
        if (r.status != 200) {
            ++failed;
            std::printf("  port reuse: request %d from local port %u failed after %.0f ms: %s\n", i, lp, r.ms, r.err.c_str());
        } else if (r.ms > 2000) {
            ++slow;
        }
        // the server side of the finished connection: no TIME_WAIT (the client closed first)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        srv_tw += test::tcpRowCount(srv.port(), lp, "TIME_WAIT");
    }
    std::printf("  port reuse: %d requests, %d failed, %d slower than 2 s, %d refused at once (client TIME_WAIT), "
                "%d server-side TIME_WAIT, worst %.0f ms\n",
                n, failed, slow, blocked, srv_tw, worst);
    CHECK_EQ(failed, 0);
    CHECK_EQ(slow, 0);
    CHECK_EQ(srv_tw, 0);
}

void testCors() {
    CHECK(isLoopbackOrigin("http://localhost"));
    CHECK(isLoopbackOrigin("http://localhost:5173"));
    CHECK(isLoopbackOrigin("https://127.0.0.1:8443"));
    CHECK(isLoopbackOrigin("http://[::1]:3000"));
    CHECK(isLoopbackOrigin("HTTP://LocalHost:80"));
    CHECK(!isLoopbackOrigin("null"));
    CHECK(!isLoopbackOrigin("http://localhost.evil.com"));
    CHECK(!isLoopbackOrigin("http://localhost:80/x"));
    CHECK(!isLoopbackOrigin("http://localhost:"));
    CHECK(!isLoopbackOrigin("http://127.0.0.1.nip.io"));
    CHECK(!isLoopbackOrigin("ftp://localhost"));
    CHECK(!isLoopbackOrigin("http://localhost:1234567"));
    CHECK(corsHeaders("http://localhost\r\nX: y", {}).empty());
    const std::string chat = R"({"messages":[{"role":"user","content":"q"}],"max_tokens":4,"temperature":0)";
    auto req = [](std::uint16_t port, const char* method, const char* path, const std::string& body, const std::string& origin) {
        return test::httpRequest("127.0.0.1", port, method, path, body, 30,
                                 origin.empty() ? std::string() : "Origin: " + origin + "\r\n");
    };
    {
        TestServer srv(baseSetup());
        const std::uint16_t p = srv.port();
        for (const char* o : {"http://localhost:5173", "http://127.0.0.1", "https://[::1]:8443"}) {
            const auto r = req(p, "GET", "/health", "", o);
            CHECK_EQ(r.status, 200);
            CHECK(hasHeader(r, std::string("Access-Control-Allow-Origin: ") + o + "\r\n"));
            CHECK(hasHeader(r, "Vary: Origin"));
        }
        for (const char* o : {"https://evil.example", "null", "http://localhost.evil.com", ""}) {
            const auto r = req(p, "GET", "/health", "", o);
            CHECK_EQ(r.status, 200);
            CHECK(!hasHeader(r, "Access-Control-"));
        }
        // engine-written responses (non-stream, stream, error) carry it too
        const std::string lo = "http://localhost:8000";
        const auto ns = req(p, "POST", "/v1/chat/completions", chat + "}", lo);
        CHECK_EQ(ns.status, 200);
        CHECK(hasHeader(ns, "Access-Control-Allow-Origin: http://localhost:8000"));
        const auto st = req(p, "POST", "/v1/chat/completions", chat + R"(,"stream":true})", lo);
        CHECK_EQ(st.status, 200);
        CHECK(hasHeader(st, "Content-Type: text/event-stream"));
        CHECK(hasHeader(st, "Access-Control-Allow-Origin: http://localhost:8000"));
        CHECK(test::sseEvents(st.body).size() >= 2);
        // FR-SRV P2-4: a POST from a disallowed Origin is refused (403), not run
        const auto st_evil = req(p, "POST", "/v1/chat/completions", chat + R"(,"stream":true})", "https://evil.example");
        CHECK_EQ(st_evil.status, 403);
        CHECK(!hasHeader(st_evil, "Access-Control-"));
        const auto bad = req(p, "POST", "/v1/chat/completions", "{not json", lo);
        CHECK_EQ(bad.status, 400);
        CHECK(hasHeader(bad, "Access-Control-Allow-Origin: http://localhost:8000"));
        // preflight
        const auto pf = req(p, "OPTIONS", "/v1/chat/completions", "", lo);
        CHECK_EQ(pf.status, 204);
        CHECK(hasHeader(pf, "Access-Control-Allow-Origin: http://localhost:8000"));
        CHECK(hasHeader(pf, "Access-Control-Allow-Methods: GET, POST, OPTIONS"));
        CHECK(hasHeader(pf, "Access-Control-Allow-Headers: *"));
        const auto pf_evil = req(p, "OPTIONS", "/v1/chat/completions", "", "https://evil.example");
        CHECK_EQ(pf_evil.status, 204);
        CHECK(!hasHeader(pf_evil, "Access-Control-"));
    }
    {
        ServerSetup su = baseSetup();
        su.http.cors_origins = {"https://App.Example.com/"};
        TestServer srv(su);
        const auto ok = req(srv.port(), "GET", "/health", "", "https://app.example.com");
        CHECK(hasHeader(ok, "Access-Control-Allow-Origin: https://app.example.com\r\n"));
        CHECK(hasHeader(req(srv.port(), "GET", "/health", "", "http://localhost:1"), "Access-Control-Allow-Origin: http://localhost:1"));
        CHECK(!hasHeader(req(srv.port(), "GET", "/health", "", "https://evil.example"), "Access-Control-"));
        const auto pf = req(srv.port(), "OPTIONS", "/v1/completions", "", "https://app.example.com");
        CHECK(hasHeader(pf, "Access-Control-Allow-Methods: GET, POST, OPTIONS"));
    }
    {
        ServerSetup su = baseSetup();
        su.http.cors_origins = {"*"};
        TestServer srv(su);
        for (const char* o : {"https://evil.example", "null", ""}) {
            const auto r = req(srv.port(), "GET", "/health", "", o);
            CHECK(hasHeader(r, "Access-Control-Allow-Origin: *\r\n"));
            CHECK(!hasHeader(r, "Vary: Origin"));
        }
        const auto st = req(srv.port(), "POST", "/v1/chat/completions", chat + R"(,"stream":true})", "https://evil.example");
        CHECK(hasHeader(st, "Access-Control-Allow-Origin: *"));
        const auto pf = req(srv.port(), "OPTIONS", "/v1/chat/completions", "", "");
        CHECK_EQ(pf.status, 204);
        CHECK(hasHeader(pf, "Access-Control-Allow-Origin: *"));
        CHECK(hasHeader(pf, "Access-Control-Allow-Methods: GET, POST, OPTIONS"));
    }
}

// greedy completions == the mock's own reference; MTP on and off
void testGreedyMatchesReference() {
    for (bool mtp : {true, false}) {
        TestServer srv(baseSetup(mtp));
        for (std::uint32_t n : {5u, 300u, 1500u}) {
            const auto p = makePrompt(n, n);
            const auto ref = srv.model().greedyReference(p, 80, true);
            auto r = srv.post("/v1/completions", completionBody(p, 80));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(completionText(r), decode(ref));
            const json::Value v = parseBody(r);
            CHECK(v.get("usage") && v.get("usage")->get("completion_tokens")->asInt() == 80);
        }
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
}

void testStreamEqualsNonStream() {
    TestServer srv(baseSetup());
    const auto p = makePrompt(700, 11);
    auto a = srv.post("/v1/completions", completionBody(p, 60, false));
    auto b = srv.post("/v1/completions", completionBody(p, 60, true, ",\"stream_options\":{\"include_usage\":true}"));
    CHECK_EQ(b.status, 200);
    CHECK(b.headers.find("text/event-stream") != std::string::npos);
    std::string fin;
    CHECK_EQ(streamText(b, false, &fin), completionText(a));
    CHECK_EQ(fin, std::string("length"));
    const auto evs = test::sseEvents(b.body);
    CHECK(!evs.empty() && evs.back() == "[DONE]");
    CHECK(evs.size() >= 3 && evs[evs.size() - 2].find("\"choices\":[]") != std::string::npos);
    // chat endpoint: stream vs non-stream content
    const std::string msgs = R"("messages":[{"role":"user","content":"Explain speculative decoding in one paragraph."}])";
    auto c1 = srv.post("/v1/chat/completions",
                       "{" + msgs + R"(,"max_tokens":40,"temperature":0,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}})");
    auto c2 = srv.post("/v1/chat/completions",
                       "{" + msgs + R"(,"max_tokens":40,"temperature":0,"ignore_eos":true,"stream":true,"chat_template_kwargs":{"enable_thinking":false}})");
    const json::Value v1 = parseBody(c1);
    std::string content1;
    if (const json::Value* ch = v1.get("choices"))
        if (const json::Value* msg = ch->asArray()[0].get("message"))
            if (const json::Value* c = msg->get("content"); c && c->isString()) content1 = c->asString();
    CHECK(!content1.empty());
    CHECK_EQ(streamText(c2, true), content1);
    CHECK(test::sseEvents(c2.body).front().find("{\"role\":\"assistant\",\"content\":null}") != std::string::npos);
    CHECK_EQ(srv.model().poison(), 0u);
}

// multi-turn: the second turn reuses the cache and equals a cold run
void testPrefixCacheMultiTurn() {
    ServerSetup su = baseSetup();
    ServerSetup su_nc = su;
    su_nc.eo.no_prefix_cache = true;
    TestServer srv(su);
    TestServer cold(su_nc);
    const auto p1 = makePrompt(3000, 21);
    auto r1 = srv.post("/v1/completions", completionBody(p1, 50));
    const auto out1 = srv.model().greedyReference(p1, 50, true);
    CHECK_EQ(completionText(r1), decode(out1));
    // turn 2 = turn 1 + its output + a new message
    std::vector<std::uint32_t> p2 = p1;
    p2.insert(p2.end(), out1.begin(), out1.end());
    const auto p2b = makePrompt(static_cast<std::uint32_t>(p2.size() + 400), 22, p2);
    auto r2 = srv.post("/v1/completions", completionBody(p2b, 50));
    auto r2c = cold.post("/v1/completions", completionBody(p2b, 50));
    CHECK_EQ(completionText(r2), completionText(r2c));
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(p2b, 50, true)));
    const std::uint64_t cached = cachedTokens(r2);
    CHECK(cached >= p1.size());  // gen-end checkpoint: the prompt and the generated tokens
    CHECK_EQ(cachedTokens(r2c), 0u);
    // identical prompt again: prompt-end checkpoint, nothing to prefill
    auto r3 = srv.post("/v1/completions", completionBody(p2b, 50));
    CHECK_EQ(completionText(r3), completionText(r2));
    CHECK_EQ(cachedTokens(r3), p2b.size());
    // cache_prompt=false: no reuse, same text
    auto r4 = srv.post("/v1/completions", completionBody(p2b, 50, false, ",\"cache_prompt\":false"));
    CHECK_EQ(completionText(r4), completionText(r2));
    CHECK_EQ(cachedTokens(r4), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(cold.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
}

// TAIL-CK: the start of the last cold-schedule chunk of an n-token text prompt (no system / think /
// image splits), computed here independently of the engine: 1024-token chunks, the rest (<= 1024 +
// 256) merged into the last one
std::uint64_t coldTail(std::size_t n) {
    std::size_t pos = 0;
    while (n - pos > 1024 + 256) pos += 1024;
    return pos;
}

ServerSetup tailSetup(std::uint32_t np, std::size_t n_ck, bool tail = true) {
    ServerSetup su = baseSetup();
    su.mc.parallel = np;
    su.mc.slot_ctx = 32768;
    su.mc.pool_tokens = 98304;
    su.eo.n_ck = n_ck;
    su.eo.tail_ck = tail;
    return su;
}

// arch024-S5a: COLD-RESUME through the engine (the PrefixStore settings come from EngineOptions). Turn 2 = turn 1 +
// its output + a new message. balance resumes from turn 1's gen-end checkpoint; precise (cold_resume) may not (gen-end
// is not cold-exact, and turn 1's prompt-end at 3000 is off turn 2's schedule), so it resumes from the TAIL-CK 'prefix'
// checkpoint at the start of turn 1's last schedule chunk (2048; precise keeps one for every prompt). Both answers equal
// a cold run.
void testColdResumeEngine() {
    for (const bool precise : {false, true}) {
        ServerSetup su = baseSetup();
        su.eo.n_ck = 3;  // enough for the tail checkpoint (cache::ckpt_keep_prefix_min)
        su.eo.cold_resume = precise;
        ServerSetup su_nc = su;
        su_nc.eo.no_prefix_cache = true;
        TestServer srv(su);
        TestServer cold(su_nc);
        const auto p1 = makePrompt(3000, 31);
        auto r1 = srv.post("/v1/completions", completionBody(p1, 40));
        const auto out1 = srv.model().greedyReference(p1, 40, true);
        CHECK_EQ(completionText(r1), decode(out1));
        std::vector<std::uint32_t> p2 = p1;
        p2.insert(p2.end(), out1.begin(), out1.end());
        const auto p2b = makePrompt(static_cast<std::uint32_t>(p2.size() + 400), 32, p2);
        auto r2 = srv.post("/v1/completions", completionBody(p2b, 40));
        auto r2c = cold.post("/v1/completions", completionBody(p2b, 40));
        CHECK_EQ(completionText(r2), completionText(r2c));
        const std::uint64_t cached = cachedTokens(r2);
        if (precise) {
            CHECK_EQ(cached, coldTail(p1.size()));
            CHECK(g_log.count("keeping a tail checkpoint @2048") > 0);
        } else {
            CHECK(cached >= p1.size());
        }
        if (g_verbose) std::fprintf(stderr, "  cold_resume %d: turn 2 cached %llu\n", precise ? 1 : 0, static_cast<unsigned long long>(cached));
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
}

// TAIL-CK (PREFIX-QA, REL3): the same long document with a different question, one message each.
// Before, the 2nd question re-prefilled the whole document (no checkpoint at or before the
// divergence on first sight; the LCP 'prefix' checkpoint only helped the 3rd). Now the tail
// checkpoint of the 1st request (start of its last schedule chunk) is reused. Server defaults:
// -np 1 -> 4 checkpoints per slot, -np 2..4 -> 3. With 2 (or WHIRL_TAIL_CK=0) no tail is taken
// and the behaviour is the old one. Every answer equals the greedy reference (= a cold run).
void testDocQa() {
    struct Case {
        std::uint32_t np;
        std::size_t n_ck;
        bool tail;
    };
    for (const Case c : {Case{1, 4, true}, Case{2, 3, true}, Case{4, 3, true}, Case{2, 2, true}, Case{1, 4, false}}) {
        TestServer srv(tailSetup(c.np, c.n_ck, c.tail));
        const auto doc = makePrompt(12000, 91);
        std::vector<std::uint64_t> cached;
        std::vector<std::size_t> lens;
        for (std::uint64_t q = 0; q < 3; ++q) {
            const auto p = makePrompt(static_cast<std::uint32_t>(doc.size() + 90 + 10 * q), 500 + q, doc);
            auto r = srv.post("/v1/completions", completionBody(p, 24));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 24, true)));
            cached.push_back(cachedTokens(r));
            lens.push_back(p.size());
        }
        if (g_verbose)
            std::printf("  doc_qa np %u n_ck %zu tail %d: doc %zu tok, cached q1 %llu, q2 %llu, q3 %llu\n", c.np, c.n_ck, c.tail ? 1 : 0,
                        doc.size(), (unsigned long long)cached[0], (unsigned long long)cached[1], (unsigned long long)cached[2]);
        CHECK_EQ(cached[0], 0u);
        CHECK(cached[2] + 1280 >= doc.size());  // 3rd question: LCP checkpoint or tail, as before
        const bool on = c.tail && c.n_ck >= 3;
        if (on) CHECK_EQ(cached[1], coldTail(lens[0]));  // 2nd question: the 1st one's tail checkpoint
        else CHECK_EQ(cached[1], 0u);                    // old behaviour (tail off): full prefill
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
    // a follow-up turn first (its prompt-end / gen-end fill the set), then a new conversation with
    // another question about the same document: the slot policy kept the first request's tail
    for (const Case c : {Case{1, 4, true}, Case{2, 3, true}}) {
        TestServer srv(tailSetup(c.np, c.n_ck, c.tail));
        const auto doc = makePrompt(12000, 95);
        const auto q1 = makePrompt(12100, 530, doc);
        auto r1 = srv.post("/v1/completions", completionBody(q1, 24));
        auto out1 = srv.model().greedyReference(q1, 24, true);
        CHECK_EQ(completionText(r1), decode(out1));
        std::vector<std::uint32_t> t2 = q1;
        t2.insert(t2.end(), out1.begin(), out1.end());
        t2 = makePrompt(static_cast<std::uint32_t>(t2.size() + 150), 531, t2);
        auto r2 = srv.post("/v1/completions", completionBody(t2, 24));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(t2, 24, true)));
        CHECK(cachedTokens(r2) >= q1.size());  // gen-end of turn 1
        const auto q2 = makePrompt(12090, 532, doc);
        auto r3 = srv.post("/v1/completions", completionBody(q2, 24));
        CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(q2, 24, true)));
        CHECK_EQ(cachedTokens(r3), coldTail(q1.size()));  // turn 1's tail, kept over its prompt-end
        if (g_verbose)
            std::printf("  doc_qa follow-up np %u: turn 2 cached %llu, new question cached %llu\n", c.np,
                        (unsigned long long)cachedTokens(r2), (unsigned long long)cachedTokens(r3));
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// TAIL-CK bitwise guard: the tail checkpoint must sit on the cold chunk schedule. The mock's
// sched_grid makes a prefill forward that starts off the 1024 grid change the state (as a GPU
// forward split elsewhere rounds differently), so a tail placed anywhere else (splitting the 1st
// request's forward there, and resuming the 2nd there) changes the text vs. the greedy reference.
void testTailCkOnSchedule() {
    for (std::uint32_t np : {1u, 2u}) {
        ServerSetup su = tailSetup(np, np == 1 ? 4 : 3);
        su.mc.sched_grid = 1024;
        TestServer srv(su);
        const auto doc = makePrompt(9000, 93);
        for (std::uint64_t q = 0; q < 2; ++q) {
            const auto p = makePrompt(static_cast<std::uint32_t>(doc.size() + 100 + 37 * q), 520 + q, doc);
            auto r = srv.post("/v1/completions", completionBody(p, 32));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 32, true)));
            if (q == 1) CHECK_EQ(cachedTokens(r), coldTail(doc.size() + 100));
        }
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// FIX-TEST-B: the shared cold prefill schedule (include/whirl/prefill_schedule.h, used by the engine
// and by whirl seqtest's GPU gate). Every forward the engine can run ends at a schedule point, a
// resume point is a schedule point, the tail checkpoint is 257..1280 rows before the end, and no
// forward crosses a kept checkpoint.
void testPrefillSchedule() {
    using namespace whirl::sched;
    const std::span<const std::uint32_t> noimg;
    std::uint64_t forwards = 0;
    for (std::size_t n : {1u, 64u, 1023u, 1024u, 1280u, 1281u, 2048u, 2304u, 2305u, 4000u, 4096u, 8232u, 9500u, 30001u, 131072u}) {
        for (std::size_t sys : {0u, 2500u}) {
            for (std::size_t th : {std::size_t{0}, n > 1 ? n - 1 : 0}) {
                if (sys >= n) continue;
                std::vector<std::size_t> pts;  // schedule points (chunk starts) + n
                for (std::size_t pos = 0; pos < n;) {
                    pts.push_back(pos);
                    const std::size_t len = prefillChunkLen(schedNext(n, sys, th, noimg, pos) - pos);
                    CHECK(len >= 1 && len <= prefill_chunk + prefill_merge);
                    pos += len;
                }
                pts.push_back(n);
                for (std::size_t a = 0; a <= n; a += (n > 5000 ? 509 : 1)) {
                    const bool on = std::find(pts.begin(), pts.end(), a) != pts.end();
                    if (on != onSchedule(n, sys, th, noimg, a) || (a < n && schedFloor(n, sys, th, noimg, a) > a)) {
                        CHECK(false);
                        break;
                    }
                }
                for (std::size_t p : pts) CHECK(onSchedule(n, sys, th, noimg, p));
                const std::size_t tail = tailPoint(n, sys, th, noimg);
                CHECK(onSchedule(n, sys, th, noimg, tail) && tail < n);
                if (n >= 4096 && th == 0) CHECK(n - tail >= prefill_merge + 1 && n - tail <= prefill_chunk + prefill_merge);
                for (std::size_t exec : {std::size_t{1024}, std::size_t{2048}, std::size_t{3072}}) {
                    for (std::size_t from : {std::size_t{0}, pts[pts.size() / 2], tail}) {
                        if (from >= n) continue;
                        const std::array<std::size_t, 2> stops = {tail, pts[pts.size() / 3]};
                        const std::vector<std::size_t> e = forwardEnds(n, sys, th, noimg, from, exec, stops);
                        CHECK(!e.empty() && e.back() == n);
                        std::size_t prev = from;
                        for (std::size_t x : e) {
                            forwards += 1;
                            CHECK(x > prev && x - prev <= std::max(exec, prefill_chunk) + prefill_merge);
                            CHECK(onSchedule(n, sys, th, noimg, x));
                            for (std::size_t st : stops) CHECK(!(st > prev && st < x));  // never across a kept checkpoint
                            if (sys > prev && sys < x) CHECK(false);                      // nor across a forced split
                            prev = x;
                        }
                    }
                }
            }
        }
    }
    CHECK(forwards > 1000);
    // no forced splits: chunk starts are the multiples of 1024 below the last chunk
    CHECK(onSchedule(9500, 0, 0, noimg, 8192) && !onSchedule(9500, 0, 0, noimg, 8192 + 64) && tailPoint(9500, 0, 0, noimg) == 9216);
    CHECK(tailPoint(8232, 0, 0, noimg) == 7168);  // 8232 = 7168 + 1064: the tail merges, 8192 is not a chunk start
    CHECK(!onSchedule(8232, 0, 0, noimg, 8192));
    CHECK_EQ(prefillExecFor(2048, 4096), std::size_t{2048});
    // SRV-PP (gfx1151, fill): exec may fill the batch; the forward is capped at the batch (max_rows), so a merged tail
    // that does not fit after 4 chunks runs as its own forward
    CHECK_EQ(prefillExecFor(16384, 4096), std::size_t{3072});  // gfx1201: unchanged
    CHECK_EQ(prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1201, false), 4096), std::size_t{2048});
    CHECK_EQ(prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1201, false), 2048), std::size_t{1024});
    CHECK_EQ(prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1201, true), 4096), std::size_t{2048});  // gfx1201 MoE: unchanged
    // SRV-PP-DENSE: gfx1151 dense keeps 2-chunk forwards (4096 lost 13% at 128k), MoE fills the batch
    CHECK_EQ(prefillExecDefault(whirl::device::k_profile_gfx1151, false), std::size_t{2048});
    CHECK_EQ(prefillExecDefault(whirl::device::k_profile_gfx1151, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1151, false), 4096, true), std::size_t{2048});
    CHECK_EQ(prefillExecFor(16384, 4096, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1151, true), 4096, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(4096, 1280, true), std::size_t{1024});
    CHECK_EQ(prefillExecFor(4096, 4352, true), std::size_t{4096});
    for (std::size_t n : {std::size_t{4100}, std::size_t{8232}, std::size_t{9500}, std::size_t{32777}, std::size_t{131087}}) {
        for (std::uint32_t mb : {1280u, 2048u, 3000u, 4096u, 4352u, 8192u}) {
            for (bool moe : {false, true}) {
                const std::size_t ex = prefillExecFor(prefillExecDefault(whirl::device::k_profile_gfx1151, moe), mb, true);
                const std::array<std::size_t, 1> st = {tailPoint(n, 0, 0, noimg)};
                const std::vector<std::size_t> e = forwardEnds(n, 0, 0, noimg, 0, ex, st, mb);
                std::size_t prev = 0;
                for (std::size_t x : e) {
                    CHECK(x > prev && x - prev <= mb && onSchedule(n, 0, 0, noimg, x));
                    prev = x;
                }
                CHECK(!e.empty() && e.back() == n);
            }
        }
    }
    // 4096-row forwards, the tail (7168 .. 8232, 1064 rows) after them; without the cap 3 chunks + tail
    CHECK((forwardEnds(8232, 0, 0, noimg, 0, 4096, {}, 4096) == std::vector<std::size_t>{4096, 7168, 8232}));
    CHECK((forwardEnds(8232, 0, 0, noimg, 0, 4096, {}, 4352) == std::vector<std::size_t>{4096, 8232}));
    CHECK((forwardEnds(8232, 0, 0, noimg, 0, 4096, {}) == std::vector<std::size_t>{4096, 8232}));
}

// CLI-SCHED: the forced-split helpers the CLI shares with the engine (sysBoundary / sysSplit /
// imageSplits) and the solo forward ends the CLI runs (soloForwardEnds = forwardEnds + the TAIL-CK stop).
void testCliSchedHelpers() {
    using namespace whirl::sched;
    const std::uint32_t IS = 1, SY = 2, IE = 3, NL = 4, U = 5, VE = 6, IMG = 0x80000000u;
    // system boundary: after the first "<|im_end|>" "\n" followed by "<|im_start|>"
    std::vector<std::uint32_t> t = {IS, SY, NL, 7, 8, IE, NL, 9, IE, NL, IS, U, NL, 10};
    CHECK_EQ(sysBoundary(t, IS, SY, IE, NL), std::size_t{10});  // the "<|im_end|>" "\n" "9" inside is not the end
    t[1] = U;
    CHECK_EQ(sysBoundary(t, IS, SY, IE, NL), std::size_t{0});  // not a system message
    CHECK_EQ(sysBoundary(t, 0xffffffffu, SY, IE, NL), std::size_t{0});
    CHECK_EQ(sysSplit(2500, 9000, 2048), std::size_t{2500});
    CHECK_EQ(sysSplit(2000, 9000, 2048), std::size_t{0});  // shorter than sys_min
    CHECK_EQ(sysSplit(2500, 9000, 0), std::size_t{0});     // off
    CHECK_EQ(sysSplit(9000, 9000, 2048), std::size_t{0});  // not inside the prompt
    // image ends: after the <|vision_end|> that follows a run of image ids, >= ck_min, inside the prompt
    std::vector<std::uint32_t> v(3000, U);
    for (std::size_t i = 100; i < 1200; ++i) v[i] = IMG + static_cast<std::uint32_t>(i);
    v[1200] = VE;
    for (std::size_t i = 1500; i < 1600; ++i) v[i] = IMG + 7;
    const std::vector<std::uint32_t> im = imageSplits(v, IMG, VE, 1024);
    CHECK(im.size() == 2 && im[0] == 1201 && im[1] == 1600);
    CHECK(imageSplits(v, IMG, VE, 1300).size() == 1);
    CHECK(imageSplits(v, IMG, VE, 0).empty());
    // solo forward ends: the cold server forwards with the tail stop (n >= 4096 only)
    const std::span<const std::uint32_t> noimg;
    for (std::size_t n : {1u, 1000u, 2100u, 4000u, 4100u, 8232u, 9500u, 32800u}) {
        for (std::size_t th : {std::size_t{0}, n - 1}) {
            const std::size_t tail = tailPoint(n, 0, th, noimg);
            std::vector<std::size_t> st;
            if (n >= tail_ck_min_default && tail > 0 && tail != th) st.push_back(tail);
            CHECK(soloForwardEnds(n, 0, th, noimg, 0, 2048, tail_ck_min_default) == forwardEnds(n, 0, th, noimg, 0, 2048, st));
            CHECK(soloForwardEnds(n, 0, th, noimg, 0, 2048, 0) == forwardEnds(n, 0, th, noimg, 0, 2048, {}));
        }
    }
    // the lengths whose old CLI forwards (max_batch 4096) were off the schedule now end on it
    CHECK((soloForwardEnds(4100, 0, 0, noimg, 0, 2048, tail_ck_min_default) == std::vector<std::size_t>{2048, 3072, 4100}));
    CHECK((soloForwardEnds(8232, 0, 0, noimg, 0, 2048, tail_ck_min_default) == std::vector<std::size_t>{2048, 4096, 6144, 7168, 8232}));
    CHECK(!onSchedule(4100, 0, 0, noimg, 4096) && !onSchedule(8232, 0, 0, noimg, 8192));
    CHECK_EQ(sys_min_default, 2048u);
    CHECK_EQ(img_split_min_default, 1024u);
    CHECK_EQ(tail_ck_min_default, 4096u);
}

// TAIL-CK regression: run a request sequence with the tail checkpoint on and off (same server
// settings otherwise); every answer equals the greedy reference.
struct TailRun {
    std::vector<std::string> text;
    std::vector<std::uint64_t> cached;
};

// chain: each entry is new content appended to the history (last prompt + its answer), an empty
// entry regenerates the last prompt (chat / agent sessions); else each entry is a whole prompt
TailRun tailSeq(const ServerSetup& su, const std::vector<std::vector<std::uint32_t>>& reqs, bool chain, std::uint32_t gen) {
    TestServer srv(su);
    TailRun out;
    std::vector<std::uint32_t> hist, last;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        std::vector<std::uint32_t> p = reqs[i];
        if (chain) {
            if (p.empty()) {
                p = last;
            } else {
                std::vector<std::uint32_t> h = hist;
                h.insert(h.end(), p.begin(), p.end());
                p = std::move(h);
            }
        }
        auto r = srv.post("/v1/completions", completionBody(p, gen));
        CHECK_EQ(r.status, 200);
        const auto ref = srv.model().greedyReference(p, gen, true);
        CHECK_EQ(completionText(r), decode(ref));
        out.text.push_back(completionText(r));
        out.cached.push_back(cachedTokens(r));
        last = p;
        hist = p;
        hist.insert(hist.end(), ref.begin(), ref.end());
    }
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
    return out;
}

void checkNoRegression(const char* name, const TailRun& on, const TailRun& off) {
    CHECK_EQ(on.text.size(), off.text.size());
    std::string a, b;
    for (std::size_t i = 0; i < on.text.size() && i < off.text.size(); ++i) {
        CHECK_EQ(on.text[i], off.text[i]);
        CHECK(on.cached[i] >= off.cached[i]);
        a += std::format(" {}", on.cached[i]);
        b += std::format(" {}", off.cached[i]);
    }
    if (g_verbose) std::printf("  %s cached: tail on%s | off%s\n", name, a.c_str(), b.c_str());
}

void testTailCkRegression() {
    for (std::uint32_t np : {1u, 2u}) {
        const std::size_t nck = np == 1 ? 4 : 3;
        // multi-turn chat with a long first message: history + answer + a new message each turn,
        // regenerates of the last turn in between (prompt-end), next turns resume from gen-end
        {
            const std::vector<std::vector<std::uint32_t>> turns = {makePrompt(6000, 601), makePrompt(300, 602), makePrompt(1800, 603),
                                                                   {},                    makePrompt(250, 604), {}};
            const TailRun on = tailSeq(tailSetup(np, nck, true), turns, true, 40);
            const TailRun off = tailSeq(tailSetup(np, nck, false), turns, true, 40);
            checkNoRegression("chat", on, off);
            for (std::size_t i = 1; i < turns.size(); ++i) CHECK_EQ(on.cached[i], off.cached[i]);
            CHECK(on.cached[3] > 0 && on.cached[5] > 0);  // regenerate: prompt-end, nothing to prefill
        }
        // agent session: a long system/tool preamble, then tool outputs of 1.5k..2.5k tokens each turn
        // (every turn takes a tail checkpoint), with a regenerate in the middle
        {
            std::vector<std::vector<std::uint32_t>> turns = {makePrompt(5000, 611)};
            for (std::uint64_t k = 0; k < 6; ++k) turns.push_back(makePrompt(static_cast<std::uint32_t>(1500 + 200 * k), 612 + k));
            turns.insert(turns.begin() + 4, std::vector<std::uint32_t>{});
            const TailRun on = tailSeq(tailSetup(np, nck, true), turns, true, 48);
            const TailRun off = tailSeq(tailSetup(np, nck, false), turns, true, 48);
            checkNoRegression("agent", on, off);
            for (std::size_t i = 1; i < turns.size(); ++i) CHECK_EQ(on.cached[i], off.cached[i]);
        }
        // regenerate of one long prompt (prompt-end each time), then another question about the same
        // document, then the first again: prompt-end reuse unchanged, the new question resumes at the tail
        {
            const auto doc = makePrompt(8000, 621);
            const auto q1 = makePrompt(8120, 622, doc), q2 = makePrompt(8200, 623, doc);
            const std::vector<std::vector<std::uint32_t>> reqs = {q1, q1, q1, q2, q1};
            const TailRun on = tailSeq(tailSetup(np, nck, true), reqs, false, 40);
            const TailRun off = tailSeq(tailSetup(np, nck, false), reqs, false, 40);
            // (q1 again after q2: with -np >= 2 and the tail off, q2 went to a free slot and q1's slot
            // kept its prompt-end; with the tail on, q2 resumes in q1's slot and q1 again resumes at
            // the tail: up to 1280 tokens to prefill there, against q2's whole document saved)
            const TailRun on_c{{on.text.begin(), on.text.begin() + 4}, {on.cached.begin(), on.cached.begin() + 4}};
            const TailRun off_c{{off.text.begin(), off.text.begin() + 4}, {off.cached.begin(), off.cached.begin() + 4}};
            checkNoRegression("regenerate", on_c, off_c);
            CHECK_EQ(on.text[4], off.text[4]);
            CHECK_EQ(on.cached[1], q1.size());
            CHECK_EQ(on.cached[2], q1.size());
            CHECK_EQ(on.cached[3], coldTail(q1.size()));  // tail off: 0
            CHECK(on.cached[4] >= coldTail(q1.size()));
            if (np == 1) CHECK_EQ(on.cached[4], off.cached[4]);
            CHECK(on.cached[3] + on.cached[4] >= off.cached[3] + off.cached[4]);
            if (g_verbose) std::printf("  regenerate q1 after q2: cached tail on %llu | off %llu\n", (unsigned long long)on.cached[4],
                                       (unsigned long long)off.cached[4]);
        }
    }
}

// --draft dflash: no MTP block, DFlash2 drafts (mock drafter: the true continuation from the
// context row before the anchor, wrong with probability dfl_miss)
ServerSetup dflashSetup(bool ngram) {
    ServerSetup su = baseSetup(false);
    su.mc.dflash = true;
    su.eo.use_dflash = true;
    su.eo.ngram = ngram;
    return su;
}

// DFlash2 drafts: outputs == the greedy reference, the drafter context only ever holds the
// sequence's own rows, drafts are used and mostly accepted
void testDFlashGreedy() {
    for (double miss : {0.0, 0.3}) {
        ServerSetup su = dflashSetup(false);
        su.mc.dfl_miss = miss;
        TestServer srv(su);
        for (std::uint32_t n : {5u, 300u, 1500u, 5000u}) {
            const auto p = makePrompt(n, 40 + n);
            const auto ref = srv.model().greedyReference(p, 120, true);
            auto r = srv.post("/v1/completions", completionBody(p, 120, n == 300));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(n == 300 ? streamText(r, false) : completionText(r), decode(ref));
        }
        const Engine::Stats st = srv.engine().stats();
        CHECK(st.dfl_cycles > 0);
        CHECK(st.dfl_drafted > 0);
        // without wrong drafts every draft of a block is accepted (up to the max_tokens budget)
        if (miss == 0.0) CHECK(st.dfl_accepted * 10 >= st.dfl_drafted * 9);
        // the context follows every cycle (no cold slot): DFlash2 drafts carry most of the
        // 4 x 120 generated tokens
        if (miss == 0.0) CHECK(st.dfl_accepted >= 300);
        CHECK_EQ(st.dfl_cold, 0u);
        if (miss > 0) CHECK(st.dfl_accepted > 0 && st.dfl_accepted < st.dfl_drafted);
        CHECK_EQ(st.dfl_gap, 0u);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().dflBad(), 0u);
        CHECK_EQ(srv.model().dflStale(), 0u);
    }
    // sampling (temperature > 0): a deterministic draft is accepted iff it is the sampled token,
    // so a seeded request gives the same text with and without DFlash2 drafts
    {
        TestServer a(dflashSetup(false));
        ServerSetup su_plain = baseSetup(false);
        su_plain.eo.ngram = false;
        TestServer b(su_plain);
        const auto p = makePrompt(400, 77);
        const std::string body = std::format("{{\"prompt\":{},\"max_tokens\":60,\"temperature\":0.8,\"seed\":5,\"ignore_eos\":true}}", idsJson(p));
        auto ra = a.post("/v1/completions", body);
        auto rb = b.post("/v1/completions", body);
        CHECK_EQ(ra.status, 200);
        CHECK_EQ(completionText(ra), completionText(rb));
        CHECK(a.engine().stats().dfl_cycles > 0);
    }
}

// fast mode's relaxacc relaxes MTP drafts only: DFlash2 drafts keep exact acceptance even with the
// item on in the engine (whirl-server also turns it off for --draft dflash), so the output is the
// greedy reference whatever the drafter proposes
void testDFlashRelaxExact() {
    // FR-DFL P2-3: the wrong drafts must be ones relaxacc WOULD keep (random tokens never fell inside
    // its top-k / alpha window, so applying relaxacc to DFlash2 drafts went unnoticed): runner_up makes
    // every wrong draft the row's second-best token at p / p_max = 0.26 >= alpha 0.1
    for (bool runner_up : {false, true}) {
        ServerSetup su = dflashSetup(false);
        su.mc.dfl_miss = 0.3;
        su.mc.dfl_miss_runner_up = runner_up;
        su.eo.relax.on = true;
        TestServer srv(su);
        g_log.clear();
        for (std::uint32_t n : {7u, 600u}) {
            const auto p = makePrompt(n, 90 + n);
            const auto ref = srv.model().greedyReference(p, 100, true);
            auto r = srv.post("/v1/completions", completionBody(p, 100, false));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(completionText(r), decode(ref));
        }
        const Engine::Stats st = srv.engine().stats();
        CHECK(st.dfl_cycles > 0 && st.dfl_accepted > 0 && st.dfl_accepted < st.dfl_drafted);
        CHECK_EQ(srv.model().dflBad(), 0u);
        // the per-request relaxacc log (written when each request finishes)
        for (int i = 0; i < 500 && g_log.count("relaxacc: ") < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK_EQ(g_log.count("relaxacc: 0 of"), 2u);  // the per-request log: no draft kept by the relaxed rule
    }
    // sampling: relaxacc's typical acceptance must not touch DFlash2 drafts either (seeded text ==
    // plain decoding of the same seed)
    {
        ServerSetup su = dflashSetup(false);
        su.mc.dfl_miss = 0.3;
        su.mc.dfl_miss_runner_up = true;
        su.eo.relax.on = true;
        TestServer a(su);
        ServerSetup su_plain = baseSetup(false);
        su_plain.eo.ngram = false;
        TestServer b(su_plain);
        const auto p = makePrompt(400, 78);
        const std::string body = std::format("{{\"prompt\":{},\"max_tokens\":60,\"temperature\":0.8,\"seed\":9,\"ignore_eos\":true}}", idsJson(p));
        auto ra = a.post("/v1/completions", body);
        auto rb = b.post("/v1/completions", body);
        CHECK_EQ(ra.status, 200);
        CHECK_EQ(completionText(ra), completionText(rb));
        CHECK(a.engine().stats().dfl_cycles > 0);
    }
}

// n-gram and DFlash2 drafts in the same requests (a repetitive toy model: the output cycles, so
// n-gram matches appear and the cost model switches between the two); several slots at once
void testDFlashNgram() {
    ServerSetup su = dflashSetup(true);
    su.mc.markov = 24;
    su.mc.dfl_miss = 0.5;
    su.eo.trace_nd = true;
    TestServer srv(su);
    g_log.clear();
    std::vector<std::vector<std::uint32_t>> prompts;
    for (std::uint32_t i = 0; i < 6; ++i) prompts.push_back(makePrompt(300 + 211 * i, 500 + i));
    std::vector<std::string> got(prompts.size());
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < prompts.size(); ++i)
        th.emplace_back([&, i] { got[i] = completionText(srv.post("/v1/completions", completionBody(prompts[i], 300))); });
    for (auto& t : th) t.join();
    for (std::size_t i = 0; i < prompts.size(); ++i) CHECK_EQ(got[i], decode(srv.model().greedyReference(prompts[i], 300, true)));
    const Engine::Stats st = srv.engine().stats();
    CHECK(st.dfl_cycles > 0);
    // cycles with n-gram drafts and cycles with DFlash2 drafts both occurred
    const std::size_t cyc = g_log.count("trace | cycle");
    CHECK(cyc > 0 && g_log.count("| n-gram 0 |") < cyc);
    CHECK(g_log.count("| dflash 0 |") < cyc);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().dflBad(), 0u);
    CHECK_EQ(srv.model().dflStale(), 0u);
}

// rollback and restore: prefix reuse in the same slot keeps the drafter rows below the reused
// length (rows past it go), a shared prefix checkpoint / host tier restore brings no drafter rows
// (the slot warms up again); every output still equals the cold reference
void testDFlashRollback() {
    ServerSetup su = dflashSetup(false);
    su.mc.parallel = 2;
    su.eo.dfl_warm = 32;
    TestServer srv(su);
    // multi-turn in one slot: turn 2 = turn 1 + output + new text; then a branch that rewinds into turn 1
    const auto p1 = makePrompt(2500, 61);
    auto r1 = srv.post("/v1/completions", completionBody(p1, 80));
    const auto o1 = srv.model().greedyReference(p1, 80, true);
    CHECK_EQ(completionText(r1), decode(o1));
    std::vector<std::uint32_t> p2 = p1;
    p2.insert(p2.end(), o1.begin(), o1.end());
    const auto p2b = makePrompt(static_cast<std::uint32_t>(p2.size() + 300), 62, p2);
    auto r2 = srv.post("/v1/completions", completionBody(p2b, 80));
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(p2b, 80, true)));
    CHECK(cachedTokens(r2) >= p1.size());
    const std::vector<std::uint32_t> p3(p1.begin(), p1.begin() + 1800);
    const auto p3b = makePrompt(2100, 63, p3);
    auto r3 = srv.post("/v1/completions", completionBody(p3b, 80));
    CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(p3b, 80, true)));
    // the same long system prompt from several requests: shared prefix checkpoints
    const auto sys = makePrompt(3000, 64);
    std::vector<std::string> got(3);
    std::vector<std::vector<std::uint32_t>> ps;
    for (std::uint32_t i = 0; i < 3; ++i) ps.push_back(makePrompt(3000 + 100 + 50 * i, 70 + i, sys));
    for (std::uint32_t i = 0; i < 3; ++i) got[i] = completionText(srv.post("/v1/completions", completionBody(ps[i], 60)));
    for (std::uint32_t i = 0; i < 3; ++i) CHECK_EQ(got[i], decode(srv.model().greedyReference(ps[i], 60, true)));
    const Engine::Stats st = srv.engine().stats();
    CHECK(st.spe_hits >= 1);
    CHECK(st.dfl_cycles > 0);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().dflBad(), 0u);
    CHECK_EQ(srv.model().dflStale(), 0u);
    {
        // a shared checkpoint brings no drafter rows: with a warm-up longer than the new text the
        // slot verifies without DFlash2 drafts (cold cycles) and still matches the reference
        ServerSetup sw = dflashSetup(false);
        sw.eo.dfl_warm = 2000;  // the shared checkpoint is @2048, ~1150 rows follow it
        TestServer w(sw);
        std::vector<std::string> g(3);
        for (std::uint32_t i = 0; i < 3; ++i) g[i] = completionText(w.post("/v1/completions", completionBody(ps[i], 60)));
        for (std::uint32_t i = 0; i < 3; ++i) CHECK_EQ(g[i], decode(w.model().greedyReference(ps[i], 60, true)));
        CHECK(w.engine().stats().spe_hits >= 1);
        CHECK(w.engine().stats().dfl_cold > 0);
        CHECK_EQ(w.model().dflStale(), 0u);
    }
}

// FR-DFL P3-3: prefix reuse into a slot whose context ring was overwritten long ago (ring 512, the
// slot ran to ~3060, the next request reuses 2048 positions): no context row below the reuse point
// survives, so the slot restarts cold at the reuse point (warm-up counted in dfl_cold) instead of
// drafting at once against rows the ring no longer holds (dflHoles: a draft missing one of the
// warm rows before it). Outputs equal the reference either way.
void testDFlashTrimRing() {
    ServerSetup su = dflashSetup(false);
    su.mc.parallel = 1;
    su.eo.tail_ck_min = 2048;
    su.eo.n_ck = 4;
    su.mc.dfl_ring = 512;
    su.mc.dfl_need_ctx = 400;
    su.eo.dfl_warm = 400;
    TestServer srv(su);
    const auto p1 = makePrompt(3000, 81);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p1, 60))), decode(srv.model().greedyReference(p1, 60, true)));
    const Engine::Stats s1 = srv.engine().stats();
    CHECK_EQ(srv.model().dflHoles(), 0u);
    // (TAIL-CK: p1's slot checkpoint at the start of its last schedule chunk, 2048)
    const std::vector<std::uint32_t> pre(p1.begin(), p1.begin() + 2100);
    const auto p2 = makePrompt(2130, 82, pre);
    auto r2 = srv.post("/v1/completions", completionBody(p2, 60));
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(p2, 60, true)));
    const std::uint64_t reused = cachedTokens(r2);
    CHECK(reused >= 1024 && reused <= 2100);
    const Engine::Stats s2 = srv.engine().stats();
    CHECK_EQ(srv.model().dflHoles(), 0u);       // never drafted against rows the ring lost
    CHECK(s2.dfl_cold > s1.dfl_cold);           // the warm-up after the reset is counted
    CHECK_EQ(srv.model().dflBad(), 0u);
    CHECK_EQ(srv.model().dflStale(), 0u);
    CHECK_EQ(srv.model().poison(), 0u);
}

// host tier restore with DFlash2: the restored session has KV and state but no drafter rows (reset
// at the restore point); outputs equal the reference and no stale row reaches a draft
void testDFlashTier() {
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_dflash_ssd").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = dflashSetup(false);
    su.mc.parallel = 1;  // B runs in A's slot: its drafter rows must not survive A's restore
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    const auto pa = makePrompt(3000, 601);
    const auto pb = makePrompt(3200, 602);
    {
        TestServer srv(su);
        g_log.clear();
        const auto outa = srv.model().greedyReference(pa, 40, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        std::vector<std::uint32_t> pa2 = pa;
        pa2.insert(pa2.end(), outa.begin(), outa.end());
        pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 603, pa2);
        auto r3 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierStats().restores_ram >= 1);
        CHECK(srv.engine().stats().dfl_cycles > 0);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().dflBad(), 0u);
        CHECK_EQ(srv.model().dflStale(), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}

// concurrent requests (more than slots): every output equals its reference
void testConcurrency(std::uint32_t parallel, std::uint32_t n_req, bool mtp, double floor_tps = 0, std::uint32_t slot_drafts = 0) {
    ServerSetup su = baseSetup(mtp);
    su.eo.decode_min_tps = floor_tps;
    su.eo.slot_drafts = slot_drafts;
    su.mc.parallel = parallel;
    su.mc.pool_tokens = 131072;
    TestServer srv(su);
    std::vector<std::vector<std::uint32_t>> prompts;
    for (std::uint32_t i = 0; i < n_req; ++i) prompts.push_back(makePrompt(200 + 377 * i % 2500, 100 + i));
    std::vector<std::string> got(n_req);
    std::vector<std::thread> th;
    for (std::uint32_t i = 0; i < n_req; ++i)
        th.emplace_back([&, i] {
            const bool stream = i % 3 == 0;
            auto r = srv.post("/v1/completions", completionBody(prompts[i], 64, stream));
            got[i] = stream ? streamText(r, false) : completionText(r);
        });
    for (auto& t : th) t.join();
    std::uint32_t ok = 0;
    for (std::uint32_t i = 0; i < n_req; ++i)
        if (got[i] == decode(srv.model().greedyReference(prompts[i], 64, true))) ++ok;
    CHECK_EQ(ok, n_req);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
    const auto h = srv.get("/health");
    CHECK(h.body.find("\"busy\":false") != std::string::npos);
}

// WHIRL_SLOT_DRAFTS=2 planner (planSlotDrafts / qwen35::allocDrafts): pure, synthetic cycle times
// T = 20 + 0.8 R + 2.5 S + 3 [R > 16] + 0.002 sum(rows_i ctx_i)
float planTrueMs(const qwen35::CycleFeat& x) {
    return 20.0f + 0.8f * x.rows + 2.5f * x.steps + (x.rows > 16 ? 3.0f : 0.0f) + 0.002f * x.row_ctx;
}

qwen35::CycleFeat planFeat(std::span<const DraftPlanSlot> ps, std::span<const std::uint32_t> d) {
    qwen35::CycleFeat x;
    std::size_t i = 0;
    for (const DraftPlanSlot& s : ps) {
        const std::uint32_t r = s.ng_rows > 0 ? s.ng_rows : d[i] + 1;
        if (s.ng_rows == 0) x.steps = std::max(x.steps, static_cast<float>(d[i++]));
        x.rows += static_cast<float>(r);
        x.row_ctx += static_cast<float>(r) * static_cast<float>(s.pos) / 1024.0f;
    }
    return x;
}

void testSlotDraftPlan() {
    qwen35::CycleCost cc;
    std::uint64_t st = 0x9E3779B97F4A7C15ull;
    const auto rnd = [&st](std::uint32_t lo, std::uint32_t hi) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        return lo + static_cast<std::uint32_t>(st >> 33) % (hi - lo + 1);
    };
    for (int i = 0; i < 64; ++i) {
        qwen35::CycleFeat x;
        const std::uint32_t n = rnd(1, 4);
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint32_t d = rnd(0, 6);
            x.rows += static_cast<float>(d + 1);
            x.steps = std::max(x.steps, static_cast<float>(d));
            x.row_ctx += static_cast<float>(d + 1) * static_cast<float>(rnd(1, 64));
        }
        cc.observe(x, planTrueMs(x));
    }
    const qwen35::DraftAccept hi(0.92f), lo(0.3f);
    std::vector<DraftPlanSlot> ps(2);
    ps[0].acc = &hi;
    ps[1].acc = &lo;
    ps[0].pos = ps[1].pos = 8192;
    const std::uint32_t U = 3, cap = 8;
    const auto d = planSlotDrafts(ps, cc, U, cap, 16, 15);
    CHECK(d.size() == 2 && d[0] > d[1]);                                     // uneven: more to the high-acceptance slot
    CHECK(d[0] >= 1 && d[1] >= 1 && d[0] <= cap && d[1] <= cap);             // 1 <= d_i <= cap
    CHECK(d[0] + d[1] + 2 <= 16);                                            // verify rows
    const std::vector<std::uint32_t> u = {U, U};
    const float tu = cc.predict(planFeat(ps, u)), td = cc.predict(planFeat(ps, d));
    CHECK(hi.expected(d[0]) / td >= 0.97f * hi.expected(U) / tu - 1e-6f);    // throughput floor, both slots
    CHECK(lo.expected(d[1]) / td >= 0.97f * lo.expected(U) / tu - 1e-6f);
    const auto ds = planSlotDrafts(ps, cc, U, cap, 16, 5);                    // 5 snapshot sets
    CHECK(ds[0] + ds[1] <= 5 && ds[1] >= 1);
    // one slot: the uniform count, exactly as without the planner
    const auto d1 = planSlotDrafts(std::span<const DraftPlanSlot>(ps.data(), 1), cc, U, cap, 16, 15);
    CHECK(d1.size() == 1 && d1[0] == U);
    // an n-gram slot keeps its rows and is skipped in the result
    std::vector<DraftPlanSlot> mix = {ps[0], ps[1], DraftPlanSlot{}};
    mix[2].ng_rows = 6;
    mix[2].ng_e = 3.0f;
    mix[2].pos = 4096;
    const auto dm = planSlotDrafts(mix, cc, 2, cap, 16, 15);
    CHECK(dm.size() == 2 && dm[0] + dm[1] + 2 + 6 <= 16 && dm[0] >= 1 && dm[1] >= 1);
}

// WHIRL_SLOT_DRAFTS=2 with one decoding slot: the same draft count every cycle as without it
// (fixed draft count, no n-gram: the automatic choices depend on measured wall-clock times)
void testSlotDraftsOneSlot() {
    std::vector<std::string> nds[2];
    std::string text[2];
    const auto p = makePrompt(700, 42);
    for (int m = 0; m < 2; ++m) {
        ServerSetup su = baseSetup(true);
        su.mc.parallel = 1;
        su.eo.slot_drafts = m == 0 ? 0 : 2;
        su.eo.mtp_auto = false;
        su.eo.ngram = false;
        su.eo.trace_nd = true;
        TestServer srv(su);
        g_log.clear();
        auto r = srv.post("/v1/completions", completionBody(p, 96));
        text[m] = completionText(r);
        std::lock_guard<std::mutex> lk(g_log.mu);
        for (const auto& l : g_log.lines) {
            const std::size_t a = l.find("| nd ");
            if (l.find("trace | cycle") != std::string::npos && a != std::string::npos)
                nds[m].push_back(l.substr(a, l.find(" |", a + 5) - a));
        }
        CHECK_EQ(text[m], decode(srv.model().greedyReference(p, 96, true)));
        CHECK_EQ(srv.model().poison(), 0u);
    }
    CHECK(!nds[0].empty() && nds[0] == nds[1]);
}

// decode floor bookkeeping (DecodeFloor): pure math on given times
void testDecodeFloorMath() {
    using ST = DecodeFloor::SlotTok;
    {
        DecodeFloor f(0);
        CHECK(!f.on());
        const ST d[1] = {{0, 7, 0}};
        f.startPeriod(0, d);
        CHECK(f.prefillAllowed(10, d));
        CHECK_EQ(f.plan(1024, 4096).rows, std::size_t{4096});  // off: the whole prefill batch
    }
    {
        DecodeFloor f(20);
        CHECK(f.on());
        // no period yet / no decoding slot: prefill always allowed
        const ST d0[1] = {{0, 7, 0}};
        CHECK(f.prefillAllowed(0, d0));
        CHECK(f.prefillAllowed(0, {}));
        f.startPeriod(0, d0);
        // 400 ms later 2 tokens: 20 tok/s needs 8 -> decode first
        const ST d1[1] = {{0, 7, 2}};
        CHECK(!f.prefillAllowed(400, d1));
        // 500 ms: 10 tokens = 20 tok/s -> prefill may run
        const ST d2[1] = {{0, 7, 10}};
        CHECK(f.prefillAllowed(500, d2));
        // a second slot that starts decoding mid-period is accounted from then
        f.startPeriod(500, d2);
        const ST d3[2] = {{0, 7, 30}, {2, 9, 1}};
        CHECK(f.prefillAllowed(900, d3));   // slot 2 first seen now: owes nothing yet
        const ST d4[2] = {{0, 7, 30}, {2, 9, 1}};
        CHECK(!f.prefillAllowed(1000, d4));  // 100 ms, 0 new tokens -> owes 2
        const ST d5[2] = {{0, 7, 30}, {2, 9, 4}};
        CHECK(f.prefillAllowed(1000, d5));
        // a new request in slot 0 restarts its accounting
        const ST d6[2] = {{0, 8, 0}, {2, 9, 12}};
        CHECK(f.prefillAllowed(1100, d6));
        const ST d7[2] = {{0, 8, 0}, {2, 9, 12}};
        CHECK(!f.prefillAllowed(1300, d7));
    }
    {
        // starvation guard: decode-only time per period <= 4 x its prefill time
        DecodeFloor f(20);
        const ST d0[1] = {{1, 3, 0}};
        f.startPeriod(0, d0);
        f.notePrefill(1024, 400);
        const ST d1[1] = {{1, 3, 1}};
        CHECK(!f.prefillAllowed(1500, d1));  // 1100 ms decode-only < 1600
        CHECK(f.prefillAllowed(2001, d1));   // >= 1600: prefill runs although below the floor
    }
    {
        // plan: row budget from the prefill rate (stall ~ gap_tokens at the floor rate), and cycles
        DecodeFloor f(20);
        const ST d0[1] = {{0, 1, 0}};
        f.startPeriod(0, d0);
        CHECK_EQ(f.plan(1024, 4096).rows, std::size_t{1024});  // no measurement yet: the oldest chunk only
        f.notePrefill(2700, 1000);                              // 2.7 rows/ms
        for (int i = 0; i < 20; ++i) {
            const ST d[1] = {{0, 1, static_cast<std::uint64_t>(5 * (i + 1))}};
            f.noteCycle(40, false, d);                          // 40 ms cycles, 5 tok/cycle
        }
        const DecodeFloor::Plan p = f.plan(1024, 4096);
        CHECK_EQ(p.rows, std::size_t{1350});                    // 2.7 rows/ms * 500 ms
        CHECK(p.reachable);
        // k * 5 >= 20 * (0.5 + k * 0.04) -> k >= 10 / 4.2 -> 3
        CHECK_EQ(p.cycles, 3u);
        CHECK_EQ(f.plan(2048, 4096).rows, std::size_t{2048});   // the oldest chunk always fits
        CHECK_EQ(f.plan(1024, 1200).rows, std::size_t{1200});   // never more than the prefill batch
        DecodeFloor g(10);
        g.startPeriod(0, d0);
        g.notePrefill(2700, 1000);
        CHECK_EQ(g.plan(1024, 4096).rows, std::size_t{2700});  // lower floor: longer forwards allowed
    }
    {
        // floor out of reach (1 tok/cycle at 100 ms < 20 tok/s): flagged, cycles capped
        DecodeFloor f(20);
        const ST d0[1] = {{0, 1, 0}};
        f.startPeriod(0, d0);
        f.notePrefill(1024, 400);
        for (int i = 0; i < 10; ++i) {
            const ST d[1] = {{0, 1, static_cast<std::uint64_t>(i + 1)}};
            f.noteCycle(100, false, d);
        }
        const DecodeFloor::Plan p = f.plan(1024, 4096);
        CHECK(!p.reachable);
        CHECK(p.cycles > 0);
    }
}

// decode floor end to end (mock with simulated device time): a streaming-length
// request keeps decoding while three long prompts prefill; outputs stay exact
void testDecodeFloor() {
    struct Run {
        double main_end_ms = 0, last_other_end_ms = 0;
        bool exact = false;
    };
    // reverse: the 3 prompts arrive first and the main request 60 ms later (a conversation's
    // next turn while subagent prompts prefill): once decoding it is protected as well
    auto run = [](double floor_tps, bool reverse) {
        ServerSetup su = baseSetup();
        su.mc.parallel = 4;
        su.mc.slot_ctx = 16384;
        su.mc.pool_tokens = 131072;
        su.mc.prefill_us_per_row = 100;  // 1024-row chunk ~ 100 ms
        su.mc.cycle_us = 5000;  // ~600 tok/s alone
        su.eo.decode_min_tps = floor_tps;
        TestServer srv(su);
        const auto pm = makePrompt(300, 500);
        std::vector<std::vector<std::uint32_t>> po;
        // reverse: longer prompts, so that they still prefill for a while once the main request decodes
        for (std::uint32_t i = 0; i < 3; ++i) po.push_back(makePrompt(reverse ? 12288 : 6144, 600 + i));
        const std::uint32_t g_main = 250;
        Run r;
        std::string got_main;
        std::vector<std::string> got(3);
        const double t0 = nowSeconds();
        auto main_req = [&] {
            auto x = srv.post("/v1/completions", completionBody(pm, g_main, true));
            got_main = streamText(x, false);
            r.main_end_ms = (nowSeconds() - t0) * 1000.0;
        };
        std::thread tm;
        if (!reverse) {
            tm = std::thread(main_req);
            std::this_thread::sleep_for(std::chrono::milliseconds(60));  // main is decoding
        }
        std::vector<std::thread> th;
        std::vector<double> ends(3);
        for (std::uint32_t i = 0; i < 3; ++i)
            th.emplace_back([&, i] {
                auto x = srv.post("/v1/completions", completionBody(po[i], 4));
                got[i] = completionText(x);
                ends[i] = (nowSeconds() - t0) * 1000.0;
            });
        if (reverse) {
            std::this_thread::sleep_for(std::chrono::milliseconds(60));  // the prompts are prefilling
            tm = std::thread(main_req);
        }
        for (auto& t : th) t.join();
        tm.join();
        r.last_other_end_ms = *std::max_element(ends.begin(), ends.end());
        bool ok = got_main == decode(srv.model().greedyReference(pm, g_main, true));
        for (std::uint32_t i = 0; i < 3; ++i) ok = ok && got[i] == decode(srv.model().greedyReference(po[i], 4, true));
        r.exact = ok && srv.model().poison() == 0 && srv.model().hidMismatch() == 0;
        return r;
    };
    g_log.clear();
    const Run off = run(0, false);
    const Run on = run(150, false);
    CHECK(g_log.count("decode floor: ") >= 1);
    const Run rev_off = run(0, true);
    g_log.clear();
    const Run rev = run(150, true);
    CHECK(off.exact);
    CHECK(on.exact);
    CHECK(rev_off.exact);
    CHECK(rev.exact);
    // off: the main request advances one cycle per (up to 4096-row) forward and ends after
    // the prefills; on: >= 150 tok/s over the ~2 s of prefill finishes its 250 tokens first
    if (g_verbose)
        std::printf("    floor off: main %.0f ms, others %.0f ms; floor 150: main %.0f ms, others %.0f ms\n", off.main_end_ms,
                    off.last_other_end_ms, on.main_end_ms, on.last_other_end_ms);
    CHECK(off.main_end_ms > off.last_other_end_ms);
    CHECK(on.main_end_ms < on.last_other_end_ms);
    // reverse: the main request's prefill waits behind the older prompts' chunks; once it decodes,
    // the floor holds their prefill back (they finish later than without a floor)
    if (g_verbose)
        std::printf("    reverse order: floor off: main %.0f ms, others %.0f ms; floor 150: main %.0f ms, others %.0f ms\n",
                    rev_off.main_end_ms, rev_off.last_other_end_ms, rev.main_end_ms, rev.last_other_end_ms);
    CHECK(g_log.count("decode floor | 150 tok/s per decoding slot (1 decoding)") >= 1);
    CHECK(rev.last_other_end_ms > rev_off.last_other_end_ms + 100);
}

// small KV pool: idle slots are evicted (LRU) and requests still match
void testPoolPressure() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 4;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 8192;  // 32 pages: two 3k sessions (12 pages each) + headroom
    TestServer srv(su);
    g_log.clear();
    std::vector<std::vector<std::uint32_t>> prompts;
    for (std::uint32_t i = 0; i < 6; ++i) prompts.push_back(makePrompt(3000, 300 + i));
    for (int round = 0; round < 2; ++round) {
        for (std::uint32_t i = 0; i < 6; ++i) {
            auto r = srv.post("/v1/completions", completionBody(prompts[i], 32));
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(prompts[i], 32, true)));
        }
    }
    // concurrently: two 3k sessions fit next to each other (idle slots are evicted)
    std::vector<std::thread> th;
    std::vector<int> st(2);
    for (int i = 0; i < 2; ++i)
        th.emplace_back([&, i] { st[i] = srv.post("/v1/completions", completionBody(prompts[i], 16)).status; });
    for (auto& t : th) t.join();
    for (int s : st) CHECK_EQ(s, 200);
    CHECK(srv.engine().stats().evictions > 0);
    CHECK(g_log.count("kv pool full: evicting idle slot") > 0);
    CHECK_EQ(srv.model().poison(), 0u);
}

// VRAM-01 / REVIEW-RC4-C P2-3, P3-2: a long prompt on the smallest prefill batch. The engine is sized as
// whirl-server sizes it for WHIRL_PREFILL_BATCH (max_batch, then prefill_exec = prefillExecFor), and
// the mock throws BatchTooLarge for a forward of more rows than max_batch (as Model::forward). 2300
// tokens = one 1024-row chunk + a 1276-row merged tail: the tail forward needs a batch of >= 1280.
void testPrefillBatchFloor() {
    for (const bool mtp : {true, false})
        for (const std::uint32_t env_batch : {512u, 1024u, 1280u}) {
            ServerSetup su = baseSetup(mtp);
            su.mc.max_batch = vram::prefillBatchFloor(env_batch);  // what whirl-server now does with WHIRL_PREFILL_BATCH
            su.eo.prefill_exec = server::prefillExecFor(2048, su.mc.max_batch);
            TestServer srv(su);
            const auto p = makePrompt(2300, 91);
            auto r = srv.post("/v1/completions", completionBody(p, 16));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
            CHECK(srv.model().maxPrefillRows() > server::prefill_chunk && srv.model().maxPrefillRows() <= su.mc.max_batch);
            CHECK_EQ(srv.model().poison(), 0u);
            if (r.status != 200) std::fprintf(stderr, "  (prefill_batch_floor: WHIRL_PREFILL_BATCH=%u, mtp %d, max_batch %u, status %d)\n", env_batch,
                                         mtp ? 1 : 0, su.mc.max_batch, r.status);
        }
}

// shared system-prompt checkpoint (item S): sessions with the same long
// system message reuse its checkpoint; outputs equal the cold result
void testSharedSystemPrompt() {
    ServerSetup su = baseSetup();
    su.eo.sys_min = 512;
    su.mc.pool_tokens = 131072;
    TestServer srv(su);
    g_log.clear();
    std::string sys = "You are a careful assistant. ";
    for (int i = 0; i < 160; ++i) sys += std::format("Rule {}: answer precisely and cite the step number. ", i);
    auto body = [&](const std::string& user) {
        std::string b = R"({"messages":[{"role":"system","content":)";
        appendJsonStr(b, sys);
        b += R"(},{"role":"user","content":)";
        appendJsonStr(b, user);
        b += R"(}],"max_tokens":24,"temperature":0,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}})";
        return b;
    };
    // reference texts from an engine without prefix cache
    ServerSetup su_nc = su;
    su_nc.eo.no_prefix_cache = true;
    std::vector<std::string> users = {"What is 2+2?", "Name a prime number.", "Say hello.", "Count to three.",
                                      "Describe the sky.", "What is a slot?"};
    std::vector<std::string> ref;
    {
        TestServer cold(su_nc);
        for (const auto& u : users) ref.push_back(cold.post("/v1/chat/completions", body(u)).body);
    }
    auto content = [](const std::string& b) {
        try {
            const json::Value v = json::parse(b);
            return v.get("choices")->asArray()[0].get("message")->get("content")->asString();
        } catch (const std::exception&) {
            return std::string("<bad>");
        }
    };
    // first one alone (creates the system checkpoint), then the rest concurrently
    CHECK_EQ(content(srv.post("/v1/chat/completions", body(users[0])).body), content(ref[0]));
    std::vector<std::string> got(users.size());
    std::vector<std::thread> th;
    for (std::size_t i = 1; i < users.size(); ++i)
        th.emplace_back([&, i] { got[i] = srv.post("/v1/chat/completions", body(users[i])).body; });
    for (auto& t : th) t.join();
    for (std::size_t i = 1; i < users.size(); ++i) CHECK_EQ(content(got[i]), content(ref[i]));
    const auto s = srv.engine().stats();
    CHECK(s.spe_new >= 1);
    CHECK(s.spe_hits >= users.size() - 1);
    CHECK(g_log.count("shared prefix: kept system checkpoint") >= 1);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
}

// feat026-cascade: sessions sharing one long system prompt decode together; with the mock's cascade on, verify rows
// read their shared prefix through the group's first sequence's page table (ops::planCascade on the engine's real
// page tables). Off: the plain path (no cascade rows) and the cold reference texts; on: the same texts, cascade rows
// taken, every prefix page really shared (cascadeBad 0), no poisoned KV / state. With and without MTP rows.
void testCascadeSharedPrompt() {
    std::string sys = "You are a careful assistant. ";
    for (int i = 0; i < 160; ++i) sys += std::format("Rule {}: answer precisely and cite the step number. ", i);
    auto body = [&](const std::string& user) {
        std::string b = R"({"messages":[{"role":"system","content":)";
        appendJsonStr(b, sys);
        b += R"(},{"role":"user","content":)";
        appendJsonStr(b, user);
        b += R"(}],"max_tokens":64,"temperature":0,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}})";
        return b;
    };
    auto content = [](const std::string& b) {
        try {
            const json::Value v = json::parse(b);
            return v.get("choices")->asArray()[0].get("message")->get("content")->asString();
        } catch (const std::exception&) {
            return std::string("<bad>");
        }
    };
    const std::vector<std::string> users = {"What is 2+2?", "Name a prime number.", "Say hello.", "Count to three.", "Describe the sky."};
    for (const bool mtp : {true, false}) {
        ServerSetup base = baseSetup(mtp);
        base.eo.sys_min = 512;
        base.mc.pool_tokens = 131072;
        base.mc.cycle_us = 300;  // decode cycles long enough for the sessions to overlap
        std::vector<std::string> ref;
        {
            ServerSetup su_nc = base;
            su_nc.eo.no_prefix_cache = true;
            TestServer cold(su_nc);
            for (const auto& u : users) ref.push_back(content(cold.post("/v1/chat/completions", body(u)).body));
        }
        // feat026-cascade-r1 C2: cap 32 (gfx1151 attn_ws4g) and the real gfx1201 caps 5 (27B, GQA 6) / 4 (Ornith, GQA 8)
        struct V {
            std::uint32_t ck, cap;
        };
        for (const V v : {V{0, 32}, V{256, 32}, V{256, 5}, V{256, 4}}) {
            const std::uint32_t ck = v.ck;
            ServerSetup su = base;
            su.mc.cascade_min_keys = ck;
            su.mc.cascade_max_rows = v.cap;
            TestServer srv(su);
            CHECK_EQ(content(srv.post("/v1/chat/completions", body(users[0])).body), ref[0]);  // makes the system checkpoint
            std::vector<std::string> got(users.size());
            std::vector<std::thread> th;
            for (std::size_t i = 1; i < users.size(); ++i)
                th.emplace_back([&, i] { got[i] = content(srv.post("/v1/chat/completions", body(users[i])).body); });
            for (auto& t : th) t.join();
            for (std::size_t i = 1; i < users.size(); ++i) CHECK_EQ(got[i], ref[i]);
            if (ck == 0) CHECK_EQ(srv.model().cascadeRows(), 0u);
            // no MTP: one row per slot, so several slots fit any cap >= 2. With MTP and a gfx1201 cap the rows of two
            // slots usually exceed it (documented limit, ops test testCascadeCapMtp): printed, not asserted (the
            // mock's draft count adapts)
            if (ck > 0 && (v.cap == 32 || !mtp)) CHECK(srv.model().cascadeRows() > 0);
            CHECK_EQ(srv.model().cascadeBad(), 0u);
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
            std::printf("  cascade mtp=%d min=%u cap=%u: cascade rows %llu\n", mtp ? 1 : 0, ck, v.cap,
                        static_cast<unsigned long long>(srv.model().cascadeRows()));
        }
    }
}

// host tiers: an evicted session is restored from RAM, then (new engine on
// the same SSD directory) from SSD; outputs equal the cold result
void testTiers() {
    // this test uses <base>\server_test_ssd
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;  // one 3k session (+ growth) at a time in VRAM
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.eo.n_ck = 2;
    const auto pa = makePrompt(3000, 501);
    const auto pb = makePrompt(3200, 502);
    const auto outa = [&] {
        TestServer probe(baseSetup());
        return probe.model().greedyReference(pa, 40, true);
    }();
    std::vector<std::uint32_t> pa2 = pa;
    pa2.insert(pa2.end(), outa.begin(), outa.end());
    pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 503, pa2);
    {
        TestServer srv(su);
        g_log.clear();
        auto r1 = srv.post("/v1/completions", completionBody(pa, 40));
        CHECK_EQ(completionText(r1), decode(outa));
        // B evicts A's pages (pool of 16 pages)
        auto r2 = srv.post("/v1/completions", completionBody(pb, 40));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pb, 40, true)));
        // A's next turn: restored from the RAM tier
        auto r3 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(cachedTokens(r3) >= pa.size());
        CHECK(srv.tierStats().restores_ram >= 1);
        CHECK(g_log.count("kv tier: restored") >= 1);
        // let the SSD writes finish (delay 0). A was reused (restored) and is written; B is a one-shot
        // prompt and, with the seen-once filter, is not
        for (int i = 0; i < 400 && srv.ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(srv.ssdEntries() >= 1);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
    {
        // restart: the SSD index brings the entries back; A's turn restores from SSD
        TestServer srv(su);
        g_log.clear();
        CHECK(srv.ssdEntries() >= 1);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierStats().restores_ssd >= 1);
        CHECK(cachedTokens(r) >= pa.size());
        CHECK_EQ(srv.model().poison(), 0u);
    }
    {
        // a different fingerprint: nothing restored (entries are foreign)
        ServerSetup su2 = su;
        su2.fingerprint = 0x9999;
        TestServer srv(su2);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK_EQ(srv.tierStats().restores_ssd + srv.tierStats().restores_ram, 0u);
        CHECK_EQ(cachedTokens(r), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}

// arch025-k2warn (bug:k2-8060s-p1): the restart server's RAM tier (sized once at startup from the RAM available
// then; 512 MiB on the 8060S vs an 801 MiB entry) is smaller than the SSD entry: the restore cannot start
// (Tier::ensureBlocks "missing > n_blocks_" -> beginRestore nullptr -> startRestore false -> cold prefill). The
// fallback is right; it must be logged (one W line naming both sizes), not silent. Fails before the fix: no line.
void testTierRestoreTooBig() {
    const std::filesystem::path base = whirl::test::testTmpBase();
    const std::string dir = narrow((base / L"server_test_ssd_toobig").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.eo.n_ck = 2;
    const auto pa = makePrompt(3000, 511);
    const auto pb = makePrompt(3200, 512);
    const auto outa = [&] {
        TestServer probe(baseSetup());
        return probe.model().greedyReference(pa, 40, true);
    }();
    std::vector<std::uint32_t> pa2 = pa;
    pa2.insert(pa2.end(), outa.begin(), outa.end());
    pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 513, pa2);
    {
        // as testTiers: A restored from RAM once (reused), then written to SSD
        TestServer srv(su);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        srv.post("/v1/completions", completionBody(pb, 40));
        auto r3 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierStats().restores_ram >= 1);
        for (int i = 0; i < 400 && srv.ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(srv.ssdEntries() >= 1);
    }
    // precondition: the largest SSD file is larger than the 2 MiB RAM tier of the restart
    std::uintmax_t max_file = 0;
    for (const auto& f : std::filesystem::directory_iterator(dir, ec))
        if (f.is_regular_file()) max_file = std::max(max_file, f.file_size());
    CHECK(max_file > (2ull << 20));
    {
        ServerSetup su2 = su;
        su2.tier_ram_mb = 2;  // one 2 MiB arena block
        TestServer srv(su2);
        g_log.clear();
        CHECK(srv.ssdEntries() >= 1);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        // the cold fallback: same output, nothing restored, nothing reused
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK_EQ(srv.tierStats().restores_ssd + srv.tierStats().restores_ram, 0u);
        CHECK_EQ(cachedTokens(r), 0u);
        // ... and it is said, once, at warning level, with both sizes (MiB) and the consequence
        CHECK_EQ(g_log.count("| kv tier: SSD entry of "), 1u);
        CHECK_EQ(g_log.count("MiB) is larger than the RAM tier (2.0 MiB) and cannot be read back; prefilling from scratch"), 1u);
        CHECK([] {  // that line is a warning ("... W req N | slot N | kv tier: ...")
            std::lock_guard<std::mutex> lk(g_log.mu);
            for (const auto& l : g_log.lines)
                if (l.find("| kv tier: SSD entry of ") != std::string::npos) return l.find(" W req ") != std::string::npos;
            return false;
        }());
        CHECK_EQ(g_log.count("kv tier: restoring "), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}


// FR-CACHE revert-check (engine.cpp spillSlot's tier_pages clamp): in-place spill after a same-slot
// regenerate (reuse < the entry's length), then eviction and a restore of the regenerated branch: the
// entry's pages past the divergence must have been re-copied (without the clamp the restore returns the
// old branch's KV and the output differs; the rest of this suite passed with the clamp removed).
void testTierInPlaceRegen() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.tier_min = 512;
    su.eo.n_ck = 4;
    const auto pa = makePrompt(3000, 801);
    const auto pb = makePrompt(3200, 802);
    TestServer srv(su);
    g_log.clear();
    const auto outa = srv.model().greedyReference(pa, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
    std::vector<std::uint32_t> base = pa;
    base.insert(base.end(), outa.begin(), outa.end());
    const auto p2 = makePrompt(static_cast<std::uint32_t>(base.size() + 300), 803, base);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p2, 40))), decode(srv.model().greedyReference(p2, 40, true)));
    // regenerate turn 2 with another user message: resumes at the gen-end checkpoint of turn 1
    const auto p3 = makePrompt(static_cast<std::uint32_t>(base.size() + 300), 804, base);
    const auto out3 = srv.model().greedyReference(p3, 40, true);
    auto r3 = srv.post("/v1/completions", completionBody(p3, 40));
    CHECK_EQ(completionText(r3), decode(out3));
    srv.onEngine([] {});  // r3's spill (logged after its response) is done
    const std::size_t inplace = g_log.count("in place");
    CHECK(inplace >= 1);
    // B evicts the slot
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
    std::vector<std::uint32_t> p4 = p3;
    p4.insert(p4.end(), out3.begin(), out3.end());
    p4 = makePrompt(static_cast<std::uint32_t>(p4.size() + 200), 805, p4);
    auto r4 = srv.post("/v1/completions", completionBody(p4, 40));
    CHECK_EQ(completionText(r4), decode(srv.model().greedyReference(p4, 40, true)));
    CHECK(srv.tierStats().restores_ram >= 1);
    CHECK(cachedTokens(r4) > base.size());
    CHECK_EQ(srv.model().poison(), 0u);
    std::fprintf(stderr, "  (tier_inplace: in-place spills %zu, restores_ram %llu, cached %u)\n", inplace,
                 static_cast<unsigned long long>(srv.tierStats().restores_ram), static_cast<unsigned>(cachedTokens(r4)));
}

// bug-s5a-cachemiss (R9700 srv_cache_precise tier p4, 2026-10-24): in precise (COLD-RESUME) a session's next turn
// can never resume at its gen-end checkpoint, so the RAM restore picks the session's own tail checkpoint (kind
// 'prefix'). The restore then treated the entry as a shared prefix entry (item S: checkpoint kind >= 4): the slot
// lost the lineage, the turn's spill made a NEW entry next to the old one (two entries per session, about twice the
// RAM tier), the new entry was not marked reused (not written to SSD at shutdown), and under a smaller RAM tier the
// copy pushed another session's entry out (gate: E turn 2 cached 0, 4 SSD files for 5 sessions). Balance resumes at
// gen-end and updates the entry in place. Expected in both modes: one entry per session lineage, extended in place,
// marked reused. Guard: another question about the same document (diverging inside the entry) is still a branch
// that leaves the entry as it was.
void testTierPreciseContinuation() {
    for (const bool precise : {false, true}) {
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;  // B evicts A
        su.tier = true;
        su.tier_min = 512;
        su.eo.n_ck = 3;  // room for the tail checkpoint
        su.eo.cold_resume = precise;
        TestServer srv(su);
        g_log.clear();
        const auto pa = makePrompt(3000, 821);
        const auto pb = makePrompt(3200, 822);
        const auto outa = srv.model().greedyReference(pa, 40, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        std::vector<std::uint32_t> pa2 = pa;
        pa2.insert(pa2.end(), outa.begin(), outa.end());
        pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 823, pa2);
        auto r2 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierStats().restores_ram >= 1);
        if (precise) CHECK_EQ(cachedTokens(r2), coldTail(pa.size()));
        else CHECK(cachedTokens(r2) >= pa.size());
        // the entries of A's lineage (they start with A's first 2048 tokens)
        struct Lin {
            std::size_t n = 0;
            std::uint32_t max_tok = 0;
            std::uint64_t id = 0;
            bool reused = false;
        };
        const auto lineage = [&] {
            Lin l;
            srv.onEngine([&] {
                for (const tier::Entry* e : srv.tierp()->entries) {
                    if (e->n_tok < 2048 || e->tokens.size() < 2048) continue;
                    if (!std::equal(pa.begin(), pa.begin() + 2048, e->tokens.begin())) continue;
                    l.n += 1;
                    if (e->n_tok > l.max_tok) {
                        l.max_tok = e->n_tok;
                        l.id = e->id;
                        l.reused = e->reused;
                    }
                }
            });
            return l;
        };
        const Lin a = lineage();
        std::fprintf(stderr, "  (tier_precise_cont precise %d: A entries %zu, longest %u tok, reused %d, cached %u)\n", precise ? 1 : 0,
                     a.n, a.max_tok, a.reused ? 1 : 0, static_cast<unsigned>(cachedTokens(r2)));
        CHECK_EQ(a.n, std::size_t{1});              // turn 2 extended turn 1's entry (no second copy)
        CHECK(a.max_tok >= pa2.size());             // ... up to turn 2's end
        CHECK(a.reused);                            // a multi-turn session: written to SSD at shutdown
        CHECK(g_log.count("(in place") >= 1);
        // guard: another question about the same document (diverges at 2900, inside the entry) is a branch
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        const std::vector<std::uint32_t> doc(pa.begin(), pa.begin() + 2900);
        const auto q = makePrompt(3100, 824, doc);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(q, 40))), decode(srv.model().greedyReference(q, 40, true)));
        bool kept = false;
        srv.onEngine([&] {
            if (const tier::Entry* e = srv.tierp()->find(a.id)) kept = e->n_tok == a.max_tok;
        });
        CHECK(kept);  // the branch left A's entry as it was
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
}

// bug-s5a-cachemiss review F4: a slot that lost the lineage (here: the RAM restore of A's turn 1 fails, so turn 2
// is prefilled cold) spills a fresh entry that holds A's turn-1 entry whole. Like llama.cpp's server_prompt_cache
// (drop cached prompts fully contained in the new one), the old copy is replaced -- fix-r21: only when the new entry
// keeps a restore point as deep as the old one's deepest; otherwise the old one stays, demoted. Both modes.
void testTierDropContained() {
    for (const bool precise : {false, true}) {
        // fix-r21 R3-1: an SSD tier in shutdown mode (written only at close), to count the session's SSD files
        const std::filesystem::path base = whirl::test::testTmpBase();
        const std::string dir = narrow((base / (precise ? L"server_test_drop_contained_p" : L"server_test_drop_contained_b")).wstring());
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;
        su.tier = true;
        su.tier_min = 512;
        su.ssd_dir = dir;
        su.tier_shutdown = true;
        su.eo.kv_idle_flush_ms = 0;
        su.eo.n_ck = 3;
        su.eo.cold_resume = precise;
        TestServer srv(su);
        g_log.clear();
        const auto pa = makePrompt(3000, 841);
        const auto pb = makePrompt(3200, 842);
        const auto outa = srv.model().greedyReference(pa, 40, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        std::vector<std::uint32_t> pa2 = pa;
        pa2.insert(pa2.end(), outa.begin(), outa.end());
        pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 843, pa2);
        srv.ops().failHostToDevCopy(1);
        auto r2 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK(!srv.ops().hostCopyFailArmed());  // the restore failed
        srv.ops().failHostToDevCopy(0);
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pa2, 40, true)));
        srv.onEngine([] {});
        std::size_t na = 0, nd = 0;
        std::uint32_t longest = 0;
        for (int i = 0; i < 300; ++i) {
            bool busy = false;
            srv.onEngine([&] {
                na = 0;
                nd = 0;
                longest = 0;
                for (const tier::Entry* e : srv.tierp()->entries) {
                    busy = busy || e->io_busy || e->loading || e->pin > 0;
                    if (e->n_tok < 2048 || e->tokens.size() < 2048) continue;
                    if (!std::equal(pa.begin(), pa.begin() + 2048, e->tokens.begin())) continue;
                    // a shared prefix entry (one checkpoint at its end; the cold prefill keeps an LCP checkpoint
                    // with the old entry, which is right: the lineage was lost) is not a session entry
                    if (e->nck == 1 && e->ck[0].valid != 0 && e->ck[0].kind >= 4 && e->ck[0].pos == e->n_tok) continue;
                    na += 1;
                    nd += e->demoted ? 1 : 0;
                    longest = std::max(longest, e->n_tok);
                }
            });
            if (!busy) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::fprintf(stderr, "  (tier_drop_contained precise %d: A entries %zu, longest %u tok)\n", precise ? 1 : 0, na,
                     static_cast<unsigned>(longest));
        // fix-r21 (review R2-1): balance -- the cold-prefill entry holds A's turn 1 whole but not its gen-end 3039
        // (its deepest restore point), so turn 1's entry stays, demoted (first eviction victim). R3-1: precise
        // resumes only at system / prefix checkpoints (turn 1's prompt-end 3000 is never on the cold schedule of a
        // longer request), and the new entry keeps turn 1's 2048 one, so turn 1's entry is replaced
        CHECK_EQ(na, (precise ? std::size_t{1} : std::size_t{2}));
        CHECK_EQ(nd, (precise ? std::size_t{0} : std::size_t{1}));
        CHECK(longest >= pa2.size());
        if (precise) CHECK(g_log.count("tok): replaced") >= 1);
        else CHECK_EQ(g_log.count("tok): replaced"), 0u);
        CHECK_EQ(g_log.count("holds a deeper restore point"), (precise ? std::size_t{0} : std::size_t{1}));
        CHECK_EQ(srv.model().poison(), 0u);
        // R3-1 / R4-1 (lead decision: restore points after a restart over one file per session): precise writes
        // one session file (the replaced entry's 2048 point lives in the shared prefix entry, which inherits
        // 'reused' and is written too); balance also writes the kept entry -- it alone holds the gen-end 3039
        srv.stopEngine();
        std::size_t na_ssd = 0;
        for (const tier::Entry* e : srv.tierp()->entries) {
            if (e->n_tok < 2048 || e->tokens.size() < 2048) continue;
            if (!std::equal(pa.begin(), pa.begin() + 2048, e->tokens.begin())) continue;
            if (e->nck == 1 && e->ck[0].valid != 0 && e->ck[0].kind >= 4 && e->ck[0].pos == e->n_tok) continue;
            na_ssd += e->ssd.has_value() ? 1 : 0;
        }
        std::fprintf(stderr, "  (tier_drop_contained precise %d: A session entries on SSD after close %zu)\n", precise ? 1 : 0, na_ssd);
        CHECK_EQ(na_ssd, (precise ? std::size_t{1} : std::size_t{2}));
        std::filesystem::remove_all(dir, ec);
    }
    // fix-r21: the drop path -- a retry of exactly turn 1's history whose restore fails: the cold prefill keeps a
    // checkpoint at its prompt end, as deep as the old entry's deepest, so the old entry is replaced (balance)
    {
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;
        su.tier = true;
        su.tier_min = 512;
        su.eo.n_ck = 3;
        TestServer srv(su);
        g_log.clear();
        const auto pa = makePrompt(3000, 851);
        const auto pb = makePrompt(3200, 852);
        const auto outa = srv.model().greedyReference(pa, 40, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        std::vector<std::uint32_t> pr = pa;
        pr.insert(pr.end(), outa.begin(), outa.end() - 1);  // turn 1's entry holds pa + outa minus the last token
        srv.ops().failHostToDevCopy(1);
        auto r2 = srv.post("/v1/completions", completionBody(pr, 40));
        srv.ops().failHostToDevCopy(0);
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pr, 40, true)));
        srv.onEngine([] {});
        std::fprintf(stderr, "  (tier_drop_contained retry: replaced %zu, kept %zu)\n", g_log.count("tok): replaced"),
                     g_log.count("holds a deeper restore point"));
        CHECK(g_log.count("tok): replaced") >= 1);
        CHECK_EQ(g_log.count("holds a deeper restore point"), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// FIX-CACHE P2-3 (FR-CACHE): more prefix checkpoints per slot than a tier entry holds (WHIRL_SERVE_CKPTS=12).
// Before: tier::Entry::ck, Slot::rs_map / rs_tail and beginRestore's `chosen` (max_ck = 8) were read and
// written out of bounds. Now the count is clamped to max_ck (warning); spill / restore stay exact.
void testServeCkptsClamp() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.tier_min = 512;
    su.eo.n_ck = 12;
    const auto pa = makePrompt(3000, 811);
    const auto pb = makePrompt(3200, 812);
    g_log.clear();
    TestServer srv(su);
    CHECK(g_log.count("above the maximum") >= 1);
    const auto outa = srv.model().greedyReference(pa, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
    std::vector<std::uint32_t> pa2 = pa;
    pa2.insert(pa2.end(), outa.begin(), outa.end());
    pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 813, pa2);
    auto r = srv.post("/v1/completions", completionBody(pa2, 40));
    CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
    CHECK(srv.tierStats().restores_ram >= 1);
    CHECK(cachedTokens(r) >= pa.size());
    std::size_t over = 0, n_entries = 0;
    srv.onEngine([&] {
        for (const tier::Entry* e : srv.tierp()->entries) over += e->nck > tier::max_ck ? 1 : 0;
        n_entries = srv.tierp()->entries.size();
    });
    CHECK_EQ(over, 0u);
    CHECK(n_entries > 0);
    CHECK_EQ(srv.model().poison(), 0u);
}

// Chat-shaped token prompts (ChatML special tokens + random body ids) for the supersede / branch tests.
struct ChatBuild {
    ChatTokens ct;
    std::uint32_t user = 0, assistant = 0;
    ChatBuild() : ct(tok()) {
        user = static_cast<std::uint32_t>(tok().encode("user", false).at(0));
        assistant = static_cast<std::uint32_t>(tok().encode("assistant", false).at(0));
    }
    // "<|im_start|>" role "\n" body "<|im_end|>" "\n"
    void turn(std::vector<std::uint32_t>& v, std::uint32_t role, std::uint32_t n, std::uint64_t seed) const {
        v.push_back(ct.im_start);
        v.push_back(role);
        v.push_back(ct.nl);
        const auto b = makePrompt(n, seed);
        v.insert(v.end(), b.begin(), b.end());
        v.push_back(ct.im_end);
        v.push_back(ct.nl);
    }
    void genPrompt(std::vector<std::uint32_t>& v) const {
        v.push_back(ct.im_start);
        v.push_back(assistant);
        v.push_back(ct.nl);
    }
    // system + user turns (bodies n_i) + generation prompt
    std::vector<std::uint32_t> conv(std::uint32_t sys_n, std::uint64_t sys_seed, std::initializer_list<std::pair<std::uint32_t, std::uint64_t>> users) const {
        std::vector<std::uint32_t> v;
        turn(v, ct.system, sys_n, sys_seed);
        for (const auto& u : users) turn(v, user, u.first, u.second);
        genPrompt(v);
        return v;
    }
    // previous prompt + its completion + "<|im_end|>" + newline + a new user turn + generation prompt
    std::vector<std::uint32_t> next(const std::vector<std::uint32_t>& prev, const std::vector<std::uint32_t>& out, std::uint32_t n,
                                    std::uint64_t seed) const {
        std::vector<std::uint32_t> v = prev;
        v.insert(v.end(), out.begin(), out.end());
        v.push_back(ct.im_end);
        v.push_back(ct.nl);
        turn(v, user, n, seed);
        genPrompt(v);
        return v;
    }
};

// TIER-EVICT (INT-RC4F, 8060S srv_cache_precise sys 53/54): `--kv-ssd-mode shutdown` with a RAM tier too small
// for the system checkpoint plus the sessions. Requests sharing a long system message attach its checkpoint from
// VRAM (no tier restore), so before the fix its RAM tier entry was never marked reused and, in shutdown mode,
// an entry pushed out of RAM was dropped: the restart found no system checkpoint. Now the attach marks the
// entry reused and RAM pressure writes it to SSD (async) instead of dropping it; one-shot sessions are still
// never written while running, and with enough RAM nothing is written while running.
void testShutdownEvictSys() {
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd_evict").wstring());
    const ChatBuild cb;
    const std::uint64_t ram_small = []() -> std::uint64_t {
        if (const char* e = std::getenv("WHIRL_TEST_EVICT_RAM_MB"); e && *e) return std::strtoull(e, nullptr, 10);
        return 12;
    }();
    for (int mode = 0; mode < 2; ++mode) {  // 0: RAM pressure, 1: RAM tier large enough
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 8192;
        su.tier = true;
        su.ssd_dir = dir;
        su.tier_min = 512;
        su.tier_shutdown = true;
        su.tier_ram_mb = mode == 0 ? ram_small : 256;
        su.eo.sys_min = 512;
        su.eo.n_ck = 2;
        std::uint32_t sys_n = 0;
        std::uint64_t sys_id = 0;
        {
            TestServer srv(su);
            g_log.clear();
            for (int i = 0; i < 6; ++i) {
                const auto p = cb.conv(2000, 951, {{300, 960 + static_cast<std::uint64_t>(i)}});
                auto r = srv.post("/v1/completions", completionBody(p, 16));
                CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
            }
            tier::Tier* t = srv.tierp();
            for (int i = 0; i < 400; ++i) {
                const tier::Stats st = srv.tierStats();
                if (st.ssd_writes >= st.evict_writes) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            CHECK(srv.engine().stats().spe_hits >= 5);
            srv.onEngine([&] {
            // writes while running: only the eviction writes of reused entries
            CHECK_EQ(t->stats.ssd_writes, t->stats.evict_writes);
            if (mode == 0) {
                CHECK(t->stats.ram_evictions >= 1);
                CHECK(t->stats.evict_writes >= 1);
                CHECK(g_log.count("evicted from RAM: written to SSD") >= 1);
                for (const tier::Entry* e : t->entries) {
                    if (e->nck >= 1 && e->ck[0].kind == 4 && e->ck[0].pos == e->n_tok) {  // the system checkpoint
                        sys_n = e->n_tok;
                        sys_id = e->id;
                        CHECK(e->reused);
                        CHECK(e->ssd.has_value());
                    }
                }
                CHECK(sys_id != 0);
            } else {
                CHECK_EQ(t->stats.ram_evictions, 0u);
                CHECK_EQ(t->stats.ssd_writes, 0u);
            }
            std::fprintf(stderr, "  (shutdown_evict_sys mode %d: RAM %llu MiB, RAM evictions %llu, eviction writes %llu, dropped %llu, entries %zu)\n",
                         mode, static_cast<unsigned long long>(su.tier_ram_mb), static_cast<unsigned long long>(t->stats.ram_evictions),
                         static_cast<unsigned long long>(t->stats.evict_writes), static_cast<unsigned long long>(t->stats.dropped),
                         t->entries.size());
            });
            CHECK_EQ(srv.model().poison(), 0u);
        }
        if (mode == 0 && sys_id != 0) {
            // restart: the system checkpoint is on the SSD; a new conversation with the same system message reuses it
            TestServer srv(su);
            g_log.clear();
            bool found = false;
            srv.onEngine([&] { found = srv.tierp()->find(sys_id) != nullptr; });
            CHECK(found);
            const auto p = cb.conv(2000, 951, {{300, 990}});
            auto r = srv.post("/v1/completions", completionBody(p, 16));
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
            CHECK(cachedTokens(r) >= sys_n);
            CHECK(srv.tierStats().restores_ssd >= 1);
            CHECK(g_log.count("checkpoint 'system'") >= 1);
            CHECK_EQ(srv.model().poison(), 0u);
        }
        std::filesystem::remove_all(dir, ec);
    }
}

// TIER-IDLE end to end (mock engine, --kv-ssd-mode shutdown, idle delay overridden to 300 ms): after the server
// sits idle, the reused session (resumed once) is written to SSD in the background; the one-shot prompt is not;
// staying idle writes nothing more; a request arriving while an idle write runs (writes slowed by a WriteSim)
// is answered bit-exactly and stops the pass; a restart restores the idle-written session from SSD.
class SlowWriteSim final : public tier::WriteSim {
public:
    std::atomic<int> delay_ms{0};
    std::atomic<int> writes{0};
    bool write(void* fh, const std::uint8_t* buf, std::uint64_t len, std::uint64_t off) override {
        writes.fetch_add(1);
        if (const int d = delay_ms.load(); d > 0) std::this_thread::sleep_for(std::chrono::milliseconds(d));
        std::uint64_t done = 0;
        while (done < len) {
            const DWORD n = static_cast<DWORD>(std::min<std::uint64_t>(len - done, 64ull << 20));
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(off + done);
            ov.OffsetHigh = static_cast<DWORD>((off + done) >> 32);
            DWORD w = 0;
            if (WriteFile(static_cast<HANDLE>(fh), buf + done, n, &w, &ov) == 0 || w == 0) return false;
            done += w;
        }
        return true;
    }
    bool flush(void* fh) override { return FlushFileBuffers(static_cast<HANDLE>(fh)) != 0; }
    bool dead() const override { return false; }
};

void testIdleWrite() {
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd_idle").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 8192;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.tier_shutdown = true;
    su.tier_ram_mb = 256;
    su.eo.kv_idle_flush_ms = 300;
    auto waitFor = [](const std::function<bool()>& f, int ms) {
        for (int i = 0; i < ms / 10; ++i) {
            if (f()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return f();
    };
    auto ask = [](TestServer& srv, const std::vector<std::uint32_t>& p, double* ms = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        auto r = srv.post("/v1/completions", completionBody(p, 16));
        if (ms) *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
        return r;
    };
    const auto p0 = makePrompt(3000, 601);  // one-shot
    const auto p1 = makePrompt(3000, 602);  // a session: turn 1 ...
    const auto p2 = makePrompt(3300, 603, p1);  // ... turn 2 resumes it (reused)
    const auto p3 = makePrompt(3600, 604, p2);  // turn 3 (changes the entry after its idle write)
    std::uint64_t sess_id = 0;
    {
        TestServer srv(su);
        tier::Tier* t = srv.tierp();
        g_log.clear();
        ask(srv, p0);
        ask(srv, p1);
        ask(srv, p2);
        // the idle write: the session only
        CHECK(waitFor([&] {
            bool done = false;
            srv.onEngine([&] { done = t->stats.idle_writes >= 1 && !t->idleWriting(); });
            return done;
        }, 10000));
        CHECK(g_log.count("idle write done") >= 1);
        srv.onEngine([&] {
        const tier::Entry* sess = nullptr;
        const tier::Entry* one = nullptr;
        for (const tier::Entry* e : t->entries) {
            if (e->tokens.size() >= p2.size() && std::equal(p2.begin(), p2.end(), e->tokens.begin())) sess = e;
            if (e->tokens.size() >= p0.size() && std::equal(p0.begin(), p0.end(), e->tokens.begin())) one = e;
        }
        CHECK(sess != nullptr && one != nullptr);
        if (sess) {
            CHECK(sess->reused && sess->ssd.has_value() && !sess->hasDirty());
            sess_id = sess->id;
        }
        if (one) CHECK(!one->reused && !one->ssd.has_value());  // one-shot: never written while running
        CHECK_EQ(t->stats.idle_writes, 1u);
        CHECK_EQ(t->stats.ssd_writes, t->stats.idle_writes);  // nothing else written while running
        });
        // staying idle (5 idle delays): no second pass, no write
        const tier::Stats st0 = srv.tierStats();
        const std::uint64_t passes = st0.idle_passes, writes = st0.ssd_writes;
        const std::size_t fired = g_log.count("idle for ");
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        const tier::Stats st1 = srv.tierStats();
        CHECK_EQ(st1.idle_passes, passes);
        CHECK_EQ(st1.ssd_writes, writes);
        CHECK_EQ(g_log.count("idle for "), fired);
        // baseline latency of a fresh request with no idle write running
        double ms_base = 0, ms_mid = 0;
        ask(srv, makePrompt(2500, 605), &ms_base);
        // turn 3 changes the session; then, while its idle write runs (slowed), a new request arrives
        SlowWriteSim sim;
        sim.delay_ms = 250;
        tier::g_write_sim.store(&sim);
        ask(srv, p3);
        const std::size_t fired3 = g_log.count("idle for ");
        CHECK(waitFor([&] { return g_log.count("idle for ") > fired3; }, 10000));
        const auto q = makePrompt(2500, 606);
        ask(srv, q, &ms_mid);
        CHECK(sim.writes.load() >= 1);
        CHECK(g_log.count("idle write stopped by a request") >= 1);
        // the write that was running completes; nothing else was started for the one-shot prompts
        CHECK(waitFor([&] {
            bool idle = true;
            srv.onEngine([&] {
                for (const tier::Entry* e : t->entries)
                    if (e->io_busy) idle = false;
            });
            return idle;
        }, 20000));
        tier::g_write_sim.store(nullptr);
        srv.onEngine([&] {
        CHECK_EQ(t->stats.ssd_writes, t->stats.idle_writes);
        if (const tier::Entry* s = t->find(sess_id)) CHECK(s->ssd.has_value() && s->ssd->n_tok >= p3.size() - 64);
        for (const tier::Entry* e : t->entries)
            if (!e->reused) CHECK(!e->ssd.has_value());
        std::fprintf(stderr, "  (idle_write: request latency %.1f ms with no idle write, %.1f ms while one ran; idle writes %llu, passes %llu)\n",
                     ms_base, ms_mid, static_cast<unsigned long long>(t->stats.idle_writes),
                     static_cast<unsigned long long>(t->stats.idle_passes));
        });
        CHECK(ms_mid < ms_base + 2000.0);  // not held behind the slowed write
        CHECK_EQ(srv.model().poison(), 0u);
        // the close flush below does not rewrite it (clean)
    }
    {
        TestServer srv(su);
        g_log.clear();
        bool found = false;
        srv.onEngine([&] { found = srv.tierp()->find(sess_id) != nullptr; });
        CHECK(found);
        const auto p4 = makePrompt(3900, 607, p3);
        auto r = ask(srv, p4);
        CHECK(cachedTokens(r) >= p2.size());
        CHECK(srv.tierStats().restores_ssd >= 1);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}

// CLOSE-REUSED (owner decision): in --kv-ssd-mode shutdown the flush when WHIRL is closed (graceful stop) writes
// only reused entries, like the RAM-eviction and idle writes: a one-shot prompt never reaches the SSD. Mock engine
// end to end: one one-shot prompt + a 2-turn session, idle write off; on the graceful stop only the session is
// written; the restart restores it from SSD and the one-shot prompt is a cold prefill (exact output both times).
void testCloseReused() {
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd_close").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 8192;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.tier_shutdown = true;
    su.tier_ram_mb = 256;
    su.eo.kv_idle_flush_ms = 0;  // no idle write: only the close flush writes
    auto ask = [](TestServer& srv, const std::vector<std::uint32_t>& p) {
        auto r = srv.post("/v1/completions", completionBody(p, 16));
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
        return r;
    };
    const auto p0 = makePrompt(3000, 701);      // one-shot
    const auto p1 = makePrompt(3000, 702);      // a session: turn 1 ...
    const auto p2 = makePrompt(3300, 703, p1);  // ... turn 2 resumes it (reused)
    std::uint64_t sess_id = 0, one_id = 0;
    {
        TestServer srv(su);
        tier::Tier* t = srv.tierp();
        g_log.clear();
        ask(srv, p0);
        ask(srv, p1);
        ask(srv, p2);
        std::uint64_t writes = 0;
        srv.onEngine([&] {  // after p2's spill (REVIEW-ARCH3 P2-1: the response comes first)
            for (const tier::Entry* e : t->entries) {
                if (e->tokens.size() >= p2.size() && std::equal(p2.begin(), p2.end(), e->tokens.begin())) sess_id = e->id;
                if (e->tokens.size() >= p0.size() && std::equal(p0.begin(), p0.end(), e->tokens.begin())) one_id = e->id;
            }
            writes = t->stats.ssd_writes;
        });
        CHECK(sess_id != 0 && one_id != 0);
        CHECK_EQ(writes, 0u);  // nothing while running
        srv.stopEngine();                   // graceful stop: Engine::flushTierOnStop
        const tier::Entry* sess = t->find(sess_id);
        const tier::Entry* one = t->find(one_id);
        CHECK(sess != nullptr && one != nullptr);
        if (sess) CHECK(sess->reused && sess->ssd.has_value() && !sess->hasDirty());
        if (one) CHECK(!one->reused && !one->ssd.has_value());  // the one-shot prompt is not written at close
        for (const tier::Entry* e : t->entries)
            if (!e->reused) CHECK(!e->ssd.has_value());
        CHECK_EQ(t->stats.ssd_writes, 1u);
        CHECK_EQ(t->ssdEntries(), 1u);
        CHECK(g_log.count("shutdown: tier writes done") >= 1);
    }
    {
        TestServer srv(su);
        tier::Tier* t = srv.tierp();
        g_log.clear();
        srv.onEngine([&] {
            CHECK_EQ(t->ssdEntries(), 1u);
            CHECK(t->find(sess_id) != nullptr);
            CHECK(t->find(one_id) == nullptr);
        });
        const auto p3 = makePrompt(3600, 704, p2);
        auto r = ask(srv, p3);
        CHECK(cachedTokens(r) >= p2.size());  // the reused session comes back from SSD
        CHECK(srv.tierStats().restores_ssd >= 1);
        auto r0 = ask(srv, p0);  // the one-shot prompt: cold, same output
        CHECK_EQ(cachedTokens(r0), 0u);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}

// FIX-CACHE P2-4 (FR-CACHE), the agent pattern on one slot: the main agent's long session goes idle while a
// subagent with the same system prompt (another task) runs two turns; then the main agent resumes.
// supersede 1: before, the subagent's second turn cut the idle main entry to the system prompt (verified:
// resumed with 2505 cached of 8513); now it is only demoted and restored whole (exact output).
// supersede 0 (WHIRL_KV_SUPERSEDE=0): the baseline the demotion must match.
void testAgentIdleMain(bool supersede) {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 16384;
    su.mc.pool_tokens = 16384;
    su.tier = true;
    su.tier_min = 512;
    su.eo.kv_supersede = supersede;
    const ChatBuild cb;
    const auto main1 = cb.conv(2500, 901, {{6000, 902}});
    const auto sub1 = cb.conv(2500, 901, {{500, 903}});
    TestServer srv(su);
    g_log.clear();
    const auto out_m1 = srv.model().greedyReference(main1, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(main1, 40))), decode(out_m1));
    const auto out_s1 = srv.model().greedyReference(sub1, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(sub1, 40))), decode(out_s1));
    const auto sub2 = cb.next(sub1, out_s1, 200, 904);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(sub2, 40))), decode(srv.model().greedyReference(sub2, 40, true)));
    if (supersede) CHECK(g_log.count("looks superseded") >= 1);  // the rule still marks it (system prompt shared)
    const auto main2 = cb.next(main1, out_m1, 200, 905);
    auto r = srv.post("/v1/completions", completionBody(main2, 40));
    CHECK_EQ(completionText(r), decode(srv.model().greedyReference(main2, 40, true)));
    std::fprintf(stderr, "  (agent supersede=%d: main resumed with %llu cached of %zu, restores_ram %llu, superseded %llu, demoted %llu)" "\n",
                 supersede ? 1 : 0, static_cast<unsigned long long>(cachedTokens(r)), main1.size(),
                 static_cast<unsigned long long>(srv.tierStats().restores_ram), static_cast<unsigned long long>(srv.tierStats().superseded),
                 static_cast<unsigned long long>(srv.tierStats().demoted));
    CHECK(cachedTokens(r) >= main1.size());
    CHECK(srv.tierStats().restores_ram >= 1);
    CHECK_EQ(srv.tierStats().superseded, 0u);
    if (supersede) CHECK(srv.tierStats().demoted >= 1);
    CHECK_EQ(srv.model().poison(), 0u);
}
void testAgentIdleMainNoSupersede() { testAgentIdleMain(false); }
void testAgentIdleMainSupersede() { testAgentIdleMain(true); }

// FIX-CACHE P3-3 (FR-CACHE): a branch from an earlier turn (edit an earlier message) restores a long
// session's inner checkpoint and takes the entry over (owner). Before: its spill updated the entry in place
// from the branch point, so the original later turns were gone from the tiers. Now the branch gets a new
// entry and the original session is restored whole when it resumes.
void testBranchFromEarlierTurn() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 16384;
    su.mc.pool_tokens = 16384;
    su.tier = true;
    su.tier_min = 512;
    su.eo.n_ck = 4;
    const ChatBuild cb;
    const auto t1 = cb.conv(2500, 921, {{3000, 922}});
    TestServer srv(su);
    g_log.clear();
    const auto o1 = srv.model().greedyReference(t1, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(t1, 40))), decode(o1));
    const auto t2 = cb.next(t1, o1, 6000, 923);
    const auto o2 = srv.model().greedyReference(t2, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(t2, 40))), decode(o2));
    // another conversation takes the slot (the session goes to the RAM tier)
    const auto u = makePrompt(3000, 924);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(u, 40))), decode(srv.model().greedyReference(u, 40, true)));
    // the branch: turn 2 edited (shares turn 1 only)
    const auto b2 = cb.next(t1, o1, 300, 925);
    auto rb = srv.post("/v1/completions", completionBody(b2, 40));
    CHECK_EQ(completionText(rb), decode(srv.model().greedyReference(b2, 40, true)));
    const std::uint64_t restores_b = srv.tierStats().restores_ram;
    // the original session resumes
    const auto t3 = cb.next(t2, o2, 200, 926);
    auto r = srv.post("/v1/completions", completionBody(t3, 40));
    CHECK_EQ(completionText(r), decode(srv.model().greedyReference(t3, 40, true)));
    std::fprintf(stderr, "  (branch: branch cached %llu, restores after branch %llu; original resumed with %llu cached of %zu)\n",
                 static_cast<unsigned long long>(cachedTokens(rb)), static_cast<unsigned long long>(restores_b),
                 static_cast<unsigned long long>(cachedTokens(r)), t2.size());
    CHECK(restores_b >= 1);                 // the branch did restore turn 1 from the tier
    CHECK(cachedTokens(rb) >= t1.size());
    CHECK(cachedTokens(r) >= t2.size());    // the original later turn survived
    CHECK_EQ(srv.model().poison(), 0u);
}

// FIX-CACHE P2-4: an agent's compress (same system prompt and first task message, the middle history
// dropped, a short summary) still cuts the old session to the shared part (same task lineage).
void testSupersedeCompressCuts() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 1;
    su.mc.slot_ctx = 16384;
    su.mc.pool_tokens = 16384;
    su.tier = true;
    su.tier_min = 512;
    const ChatBuild cb;
    const auto old1 = cb.conv(2500, 911, {{500, 912}, {8000, 913}});
    auto c1 = cb.conv(2500, 911, {{500, 912}, {300, 914}});
    TestServer srv(su);
    g_log.clear();
    const auto out_o = srv.model().greedyReference(old1, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(old1, 40))), decode(out_o));
    const auto out_c1 = srv.model().greedyReference(c1, 40, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(c1, 40))), decode(out_c1));
    const auto c2 = cb.next(c1, out_c1, 200, 915);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(c2, 40))), decode(srv.model().greedyReference(c2, 40, true)));
    srv.onEngine([] {});  // c2's spill (which applies the cut) runs after its response (REVIEW-ARCH3 P2-1)
    CHECK(g_log.count("superseded (compact / compress): cut") >= 1);
    CHECK(srv.tierStats().superseded >= 1);
    CHECK_EQ(srv.model().poison(), 0u);
}

// FIX-RC4-A P2-1 (RESTORE-EXC): a device copy throwing while a restore is under way -- in the first pump
// inside Tier::beginRestore (RAM restore: every piece is copied there) or in a later pump (SSD restore) --
// fails the restore cleanly through the engine: the request is prefilled cold (same output), no entry
// stays pinned, nothing stays loading / busy, and the entry restores again afterwards.
void testRestoreThrows() {
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd_throw").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.eo.n_ck = 2;
    const auto pa = makePrompt(3000, 701);
    const auto pb = makePrompt(3200, 702);
    const auto outa = [&] {
        TestServer probe(baseSetup());
        return probe.model().greedyReference(pa, 40, true);
    }();
    std::vector<std::uint32_t> pa2 = pa;
    pa2.insert(pa2.end(), outa.begin(), outa.end());
    pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 703, pa2);
    const auto settled = [](TestServer& srv) {
        // the tier is touched by the engine thread only; wait until it is quiet, then look
        for (int i = 0; i < 300; ++i) {
            bool busy = false;
            srv.onEngine([&] {
                for (const tier::Entry* e : srv.tierp()->entries) busy = busy || e->io_busy || e->loading || e->pin > 0;
            });
            if (!busy) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    {
        TestServer srv(su);
        auto r1 = srv.post("/v1/completions", completionBody(pa, 40));
        CHECK_EQ(completionText(r1), decode(outa));
        auto r2 = srv.post("/v1/completions", completionBody(pb, 40));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pb, 40, true)));
        // A's next turn would restore from RAM: its first restore copy throws (inside Tier::beginRestore)
        g_log.clear();
        srv.ops().failHostToDevCopy(1);
        auto r3 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK(!srv.ops().hostCopyFailArmed());  // the injected failure happened
        srv.ops().failHostToDevCopy(0);
        CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(settled(srv));  // the restore target is not left pinned (it could never be evicted again)
        CHECK_EQ(srv.tierStats().restores_ram, 0u);
        for (int i = 0; i < 400 && srv.ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK_EQ(srv.model().poison(), 0u);
    }
    {
        // restart: A's turn restores from SSD; a later restore copy throws (first pump or pollRestores)
        TestServer srv(su);
        CHECK(srv.ssdEntries() >= 1);
        g_log.clear();
        srv.ops().failHostToDevCopy(3);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK(!srv.ops().hostCopyFailArmed());
        srv.ops().failHostToDevCopy(0);
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(settled(srv));
        CHECK_EQ(srv.model().poison(), 0u);
        // nothing is stuck: the same prompt restores now (from RAM or SSD) and gives the same output
        auto r2 = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(settled(srv));
    }
    std::filesystem::remove_all(dir, ec);
}

// concurrent requests on the same evicted prefix: one restore (delay hit)
void testRestoreConcurrent() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 4;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 16384;
    su.tier = true;
    su.tier_min = 512;
    TestServer srv(su);
    const auto base = makePrompt(3000, 601);
    srv.post("/v1/completions", completionBody(base, 8));
    // fill the pool with other sessions so the base is evicted from VRAM
    for (std::uint32_t i = 0; i < 4; ++i) srv.post("/v1/completions", completionBody(makePrompt(3500, 610 + i), 8));
    g_log.clear();
    std::vector<std::vector<std::uint32_t>> ps;
    for (std::uint32_t i = 0; i < 3; ++i) ps.push_back(makePrompt(3000 + 8 + 200, 620 + i, [&] {
        auto b = base;
        auto o = srv.model().greedyReference(base, 8, true);
        b.insert(b.end(), o.begin(), o.end());
        return b;
    }()));
    std::vector<std::string> got(3);
    std::vector<std::thread> th;
    for (std::uint32_t i = 0; i < 3; ++i)
        th.emplace_back([&, i] { got[i] = completionText(srv.post("/v1/completions", completionBody(ps[i], 16))); });
    for (auto& t : th) t.join();
    for (std::uint32_t i = 0; i < 3; ++i) CHECK_EQ(got[i], decode(srv.model().greedyReference(ps[i], 16, true)));
    const auto st = srv.tierStats();
    if (g_verbose) std::fprintf(stderr, "  restores ram %llu ssd %llu, waits %zu\n", static_cast<unsigned long long>(st.restores_ram),
                                static_cast<unsigned long long>(st.restores_ssd), g_log.count("waiting for the restore"));
    CHECK(st.restores_ram >= 1);
    CHECK_EQ(srv.model().poison(), 0u);
    CHECK_EQ(srv.model().hidMismatch(), 0u);
}

// sampling: seeded requests reproduce (alone and under load), seeds differ
void testSampling() {
    TestServer srv(baseSetup());
    const auto p = makePrompt(400, 701);
    auto body = [&](int seed) {
        return std::format("{{\"prompt\":{},\"max_tokens\":40,\"temperature\":1.0,\"top_k\":8,\"top_p\":0.95,\"seed\":{},\"ignore_eos\":true}}",
                           idsJson(p), seed);
    };
    const std::string a = completionText(srv.post("/v1/completions", body(5)));
    const std::string b = completionText(srv.post("/v1/completions", body(5)));
    const std::string c = completionText(srv.post("/v1/completions", body(6)));
    CHECK_EQ(a, b);
    CHECK(a != c);
    // under load: 4 concurrent seeded requests
    std::vector<std::string> got(4);
    std::vector<std::thread> th;
    for (int i = 0; i < 4; ++i) th.emplace_back([&, i] { got[i] = completionText(srv.post("/v1/completions", body(5))); });
    for (auto& t : th) t.join();
    for (const auto& g : got) CHECK_EQ(g, a);
    CHECK_EQ(srv.model().poison(), 0u);
}

// stop strings and end-of-turn
void testStopAndEos() {
    ServerSetup su = baseSetup();
    su.mc.eos_rate = 0.03;
    TestServer srv(su);
    const auto p = makePrompt(500, 801);
    const auto ref = srv.model().greedyReference(p, 400, false);
    auto r = srv.post("/v1/completions",
                      std::format("{{\"prompt\":{},\"max_tokens\":400,\"temperature\":0}}", idsJson(p)));
    const json::Value v = parseBody(r);
    CHECK_EQ(completionText(r), decode(ref));
    const std::string fr = v.get("choices")->asArray()[0].get("finish_reason")->asString();
    CHECK_EQ(fr, std::string(ref.size() < 400 ? "stop" : "length"));
    // a stop string inside the reference text
    const std::string full = decode(ref);
    if (full.size() > 30) {
        const std::string stop = full.substr(full.size() / 2, 6);
        auto r2 = srv.post("/v1/completions", std::format("{{\"prompt\":{},\"max_tokens\":400,\"temperature\":0,\"stop\":[{}]}}",
                                                          idsJson(p), [&] {
                                                              std::string s;
                                                              appendJsonStr(s, stop);
                                                              return s;
                                                          }()));
        const std::string t2 = completionText(r2);
        CHECK_EQ(t2, full.substr(0, full.find(stop)));
        CHECK(parseBody(r2).get("choices")->asArray()[0].get("finish_reason")->asString() == "stop");
    }
    CHECK_EQ(srv.model().poison(), 0u);
}

// ---------------------------------------------------------------------------
// vision (image_url parts) against the mock encoder

// 24-bit BMP (bottom-up rows, 4-byte padded) of a w x h image with a pattern from seed
std::string makeBmp(int w, int h, int seed) {
    const int row = (w * 3 + 3) / 4 * 4;
    const int size = 54 + row * h;
    std::string b(static_cast<std::size_t>(size), '\0');
    auto put32 = [&](int at, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b[static_cast<std::size_t>(at + i)] = static_cast<char>((v >> (8 * i)) & 0xff);
    };
    b[0] = 'B';
    b[1] = 'M';
    put32(2, static_cast<std::uint32_t>(size));
    put32(10, 54);
    put32(14, 40);
    put32(18, static_cast<std::uint32_t>(w));
    put32(22, static_cast<std::uint32_t>(h));
    b[26] = 1;
    b[28] = 24;
    put32(34, static_cast<std::uint32_t>(row * h));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                b[static_cast<std::size_t>(54 + y * row + x * 3 + c)] = static_cast<char>((x * 7 + y * 13 + c * 50 + seed * 31) & 0xff);
    return b;
}

std::string b64enc(const std::string& s) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    std::size_t i = 0;
    for (; i + 2 < s.size(); i += 3) {
        const std::uint32_t v = (static_cast<std::uint8_t>(s[i]) << 16) | (static_cast<std::uint8_t>(s[i + 1]) << 8) | static_cast<std::uint8_t>(s[i + 2]);
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += T[(v >> 6) & 63];
        o += T[v & 63];
    }
    if (i + 1 == s.size()) {
        const std::uint32_t v = static_cast<std::uint8_t>(s[i]) << 16;
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += "==";
    } else if (i + 2 == s.size()) {
        const std::uint32_t v = (static_cast<std::uint8_t>(s[i]) << 16) | (static_cast<std::uint8_t>(s[i + 1]) << 8);
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += T[(v >> 6) & 63];
        o += '=';
    }
    return o;
}

std::string imagePart(const std::string& url) { return R"({"type":"image_url","image_url":{"url":")" + url + R"("}})"; }

std::string visBody(const std::vector<std::string>& parts, const std::string& q, int max_tokens = 16, const std::string& extra = "") {
    std::string c = "[";
    for (const auto& p : parts) c += p + ",";
    std::string qs;
    appendJsonStr(qs, q);
    c += R"({"type":"text","text":)" + qs + "}]";
    return R"({"messages":[{"role":"user","content":)" + c + "}]" +
           std::format(R"(,"max_tokens":{},"temperature":0,"ignore_eos":true,"chat_template_kwargs":{{"enable_thinking":false}}{}}})", max_tokens, extra);
}

std::string chatContent(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* ch = v.get("choices");
    if (!ch || !ch->isArray() || ch->asArray().empty()) return "<no choices>";
    const json::Value* m = ch->asArray()[0].get("message");
    const json::Value* t = m ? m->get("content") : nullptr;
    return t && t->isString() ? t->asString() : "<no content>";
}

std::int64_t promptTokens(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* u = v.get("usage");
    const json::Value* p = u ? u->get("prompt_tokens") : nullptr;
    return p ? p->asInt() : -1;
}

void testVision() {
    // base64 decoder (prototype semantics: padded when len % 4 == 0, else unpadded; zero leftover bits)
    {
        std::vector<std::uint8_t> o;
        CHECK(base64Decode("aGVsbG8=", o) && std::string(o.begin(), o.end()) == "hello");
        CHECK(base64Decode("aGVsbG8", o) && std::string(o.begin(), o.end()) == "hello");
        CHECK(base64Decode("", o) && o.empty());
        CHECK(!base64Decode("aGVsbG9", o));  // non-zero leftover bits
        CHECK(!base64Decode("aGV=bG8=", o));
        CHECK(!base64Decode("aGVsbG8*", o));
        CHECK(!base64Decode("aGVsbA=", o));  // padding without a multiple of 4
    }
    // no mmproj: the prototype's message
    {
        ServerSetup su;
        TestServer srv(su);
        auto r = srv.post("/v1/chat/completions", visBody({imagePart("data:image/bmp;base64," + b64enc(makeBmp(64, 48, 1)))}, "What is this?"));
        CHECK_EQ(r.status, 400);
        CHECK(r.body.find("image content needs an mmproj") != std::string::npos);
    }
    ServerSetup su;
    su.vision = true;
    su.mc.slot_ctx = 16384;
    TestServer srv(su);
    MockVision& mv = *srv.vision();
    const std::string img1 = "data:image/bmp;base64," + b64enc(makeBmp(300, 200, 1));
    const std::string img2 = "data:image/bmp;base64," + b64enc(makeBmp(300, 200, 2));
    // 300 x 200 -> 288 x 192 (smart resize, multiple of 32) -> 9 x 6 = 54 image tokens
    auto text_only = srv.post("/v1/chat/completions", visBody({}, "What is this?"));
    CHECK_EQ(text_only.status, 200);
    const std::int64_t n_text = promptTokens(text_only);
    auto a = srv.post("/v1/chat/completions", visBody({imagePart(img1)}, "What is this?"));
    CHECK_EQ(a.status, 200);
    // the placeholder (vision_start, image_pad, vision_end = 3 tokens) replaced by 54 image tokens + start / end
    CHECK_EQ(promptTokens(a), n_text + 3 - 1 + 54);
    CHECK_EQ(mv.encodes.load(), 1);
    CHECK_EQ(srv.engine().stats().vis_enc, 1u);
    // same prompt again: fully reused (no embeddings needed)
    auto a2 = srv.post("/v1/chat/completions", visBody({imagePart(img1)}, "What is this?"));
    CHECK_EQ(chatContent(a2), chatContent(a));
    CHECK(static_cast<std::int64_t>(cachedTokens(a2)) >= promptTokens(a2) - 1);
    CHECK_EQ(mv.encodes.load(), 1);
    // same image, different question: the embeddings come from the cache
    auto b = srv.post("/v1/chat/completions", visBody({imagePart(img1)}, "Describe the colors."));
    CHECK_EQ(b.status, 200);
    CHECK_EQ(mv.encodes.load(), 1);
    CHECK_EQ(srv.engine().stats().vis_hit, 1u);
    // a different image: new ids (no reuse of the first image), one more encode
    auto c = srv.post("/v1/chat/completions", visBody({imagePart(img2)}, "What is this?"));
    CHECK_EQ(c.status, 200);
    CHECK_EQ(mv.encodes.load(), 2);
    CHECK(cachedTokens(c) < 10);
    CHECK(chatContent(c) != chatContent(a));
    // two images in one message
    auto d = srv.post("/v1/chat/completions", visBody({imagePart(img1), imagePart(img2)}, "Compare them."));
    CHECK_EQ(d.status, 200);
    const std::int64_t n_cmp = promptTokens(srv.post("/v1/chat/completions", visBody({}, "Compare them.")));
    CHECK_EQ(promptTokens(d), n_cmp + 2 * (3 - 1 + 54));
    CHECK_EQ(mv.encodes.load(), 2);
    // streaming works with images
    auto s = srv.post("/v1/chat/completions", visBody({imagePart(img2)}, "What is this?", 16, R"(,"stream":true)"));
    CHECK_EQ(streamText(s, true), chatContent(c));
    // errors
    auto bad = [&](const std::string& body, const char* needle) {
        auto r = srv.post("/v1/chat/completions", body);
        const bool ok = r.status == 400 && r.body.find(needle) != std::string::npos;
        if (!ok) std::fprintf(stderr, "    (expected 400 with '%s', got %d %s)\n", needle, r.status, r.body.substr(0, 160).c_str());
        CHECK(ok);
    };
    bad(visBody({imagePart("http://example.com/a.png")}, "x"), "remote image URLs are not fetched");
    bad(visBody({imagePart("C:\\\\nope.png")}, "x"), "--allow-local-images");
    bad(visBody({imagePart("data:image/png;base64,***")}, "x"), "invalid base64 in data: URL");
    bad(visBody({imagePart("data:image/png,abc")}, "x"), "data: URLs must be base64");
    bad(visBody({imagePart("data:image/png;base64" )}, "x"), "malformed data: URL");
    bad(visBody({imagePart("data:image/png;base64," + b64enc("not an image at all"))}, "x"), "cannot decode the image");
    bad(visBody({R"({"type":"image_url","image_url":{}})"}, "x"), "image_url.url is required");
    bad(visBody({R"({"type":"image_url","image_url":5})"}, "x"), "image_url must be an object {url} or a string");
    bad(visBody({R"({"type":"video","video":"x.mp4"})"}, "x"), "video content is not supported by this server");
    bad(visBody({imagePart(img1)}, "<|vision_start|><|image_pad|><|vision_end|> two pads"), "image placeholders do not match");
    bad(R"({"messages":[{"role":"system","content":[)" + imagePart(img1) + R"(]},{"role":"user","content":"hi"}],"max_tokens":4})",
        "System message cannot contain images.");
    // text-only on the vision server: no encode
    auto t2 = srv.post("/v1/chat/completions", visBody({}, "What is this?"));
    CHECK_EQ(chatContent(t2), chatContent(text_only));
    CHECK_EQ(mv.encodes.load(), 2);
    CHECK_EQ(srv.model().poison(), 0u);
    srv.shutdown();
    // --allow-local-images: a file path works
    {
        ServerSetup su2;
        su2.vision = true;
        su2.allow_files = true;
        TestServer srv2(su2);
        const std::string p = (std::filesystem::temp_directory_path() / "whirl_vis_test.bmp").string();
        writeFile(p, makeBmp(300, 200, 1));
        std::string esc;
        appendJsonStr(esc, p);
        auto r = srv2.post("/v1/chat/completions", visBody({R"({"type":"image_url","image_url":{"url":)" + esc + "}}"}, "What is this?"));
        CHECK_EQ(r.status, 200);
        CHECK_EQ(chatContent(r), chatContent(a));
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
}


// ---------------------------------------------------------------------------
// arch024 S3: CkptStore (M2) and invariant F1 (never free a checkpoint buffer a spill may still read), on NullBackend's
// tracking mode (stream order). A spill is modelled the way Tier::copyOut + Tier::fence do it: copies that read the
// checkpoint buffers on a tier stream, then the slot's spill_ev recorded on that stream.

std::uint64_t al256(std::uint64_t x) { return (x + 255) / 256 * 256; }

// ---------------------------------------------------------------------------
// arch024-S4 PagePool (M3). Modeled on vLLM tests/v1/core/test_prefix_caching.py (shared-prefix ref counts, free
// queue order, eviction order) and llama.cpp's "a failed find_slot changes nothing" (progress/arch024-S4.PRIOR_ART.md).

struct PpRig {
    NullBackend nb;
    std::unique_ptr<MockModel> model;
    std::unique_ptr<PagePool> pool;
    device::Stream ts = nullptr;  // the tier stream (quarantine events)
    std::vector<KvArr> arrs;
    explicit PpRig(std::uint32_t pages = 8, std::vector<std::uint32_t> extra = {}) {
        MockConfig mc;
        mc.parallel = 2;
        mc.pool_tokens = pages * kv_page;
        mc.kv_extra_rows = std::move(extra);
        model = std::make_unique<MockModel>(nb, tok(), mc);
        pool = std::make_unique<PagePool>(nb, *model);
        arrs = model->kvArrays();
        pool->init(model->poolPages(), arrs);
        ts = nb.streamCreateNonBlocking();
    }
    ~PpRig() { nb.streamDestroy(ts); }
    // holders of every page counted once per holder: free + quarantined + held == pages (no page lost or doubled)
    std::size_t refSum() const {
        std::size_t n = 0;
        for (std::uint32_t p = 0; p < pool->pages(); ++p) n += pool->refCount(static_cast<std::int32_t>(p));
        return n;
    }
    // work on the tier stream that the next event there covers (a spill still reading)
    void spillWork() {
        static std::uint8_t buf[64];
        nb.copyAsync(reinterpret_cast<DevPtr>(buf), reinterpret_cast<DevPtr>(buf + 32), 16, ts);
    }
};

// scripted evictor: each call frees the next holder's pages (unref), false when none is left
struct ListEvictor final : PagePool::Evictor {
    PagePool& pool;
    std::vector<std::vector<std::int32_t>*> victims;
    std::size_t calls = 0;
    explicit ListEvictor(PagePool& p) : pool(p) {}
    bool evictOne() override {
        ++calls;
        for (auto* v : victims) {
            if (v->empty()) continue;
            pool.unref(*v);
            v->clear();
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// arch024-S5a: PrefixStore (M4) driven directly: the chunk schedule, the COLD-RESUME table, cacheMatch / pickSlot,
// the shared prefix table (match / eviction) and the TAIL-CK point. Synthetic chat ids, no model.

namespace ps_test {
constexpr std::uint32_t kImStart = 1, kImEnd = 2, kNl = 3, kSystem = 4, kThink = 5, kVisEnd = 6;

ChatTokens chatIds() {
    ChatTokens ct;
    ct.im_start = kImStart;
    ct.im_end = kImEnd;
    ct.nl = kNl;
    ct.system = kSystem;
    ct.think = kThink;
    return ct;
}

// n tokens: filler 1000 + i (seed shifts it, so two seeds share nothing); sys > 0: a leading system message whose
// boundary (the next <|im_start|>) is at sys; think: ends with "<think>" "\n"
std::vector<std::uint32_t> prompt(std::size_t n, std::size_t sys = 0, bool think = false, std::uint32_t seed = 0) {
    std::vector<std::uint32_t> t(n);
    for (std::size_t i = 0; i < n; ++i) t[i] = 1000 + seed * 1000000 + static_cast<std::uint32_t>(i);
    if (sys > 0) {
        t[0] = kImStart;
        t[1] = kSystem;
        t[sys - 2] = kImEnd;
        t[sys - 1] = kNl;
        t[sys] = kImStart;
    }
    if (think) {
        t[n - 2] = kThink;
        t[n - 1] = kNl;
    }
    return t;
}

PrefixStore::Config cfgFor(bool precise) {
    PrefixStore::Config c;
    c.cold_resume = precise;  // --precise turns COLD-RESUME on; balance / fast leave it off
    c.sys_min = sched::sys_min_default;
    c.kv_page = kv_page;
    return c;
}

void setCk(Ckpt& c, std::uint32_t pos, const char* kind, std::uint64_t seq) {
    c.valid = true;
    c.pos = pos;
    c.kind = kind;
    c.seq = seq;
    c.has_logits = true;
}

struct TSlot : SlotPrefix {
    bool idle = true;
};
}  // namespace ps_test

void testPrefixStore() {
    using namespace ps_test;
    const ChatTokens ct = chatIds();
    // ---- the chunk schedule: the store's functions are the prefill_schedule.h ones over the prompt's splits
    {
        PrefixStore st(ct, cfgFor(false));
        const auto a = prompt(9500, 3000);
        CHECK_EQ(ct.sysBoundary(a), std::size_t{3000});
        CHECK_EQ(st.sysSplit(a), 3000u);
        CHECK_EQ(st.sysSplit(prompt(9500, 1000)), 0u);  // shorter than sys_min (2048): no split
        CHECK_EQ(st.sysSplit(prompt(9500)), 0u);        // no system message
        PrefixStore::Config off = cfgFor(false);
        off.sys_min = 0;  // WHIRL_SYS_MIN=0
        CHECK_EQ(PrefixStore(ct, off).sysSplit(a), 0u);
        CHECK(st.imgSplits(a).empty());  // no mmproj
        const std::vector<std::uint32_t> noimg;
        bool same = true;
        const std::vector<std::vector<std::uint32_t>> ps = {prompt(9500), prompt(9500, 3000), prompt(9500, 0, true),
                                                            prompt(20000, 2500, true)};
        for (const auto& p : ps) {
            const std::size_t sys = sched::sysSplit(ct.sysBoundary(p), p.size(), sched::sys_min_default);
            const std::size_t th = ct.thinkOpenSplit(p);
            for (std::size_t x = 0; x <= p.size(); x += 37) {
                same = same && st.onSchedule(p, x) == sched::onSchedule(p.size(), sys, th, noimg, x);
                same = same && st.schedFloor(p, x) == sched::schedFloor(p.size(), sys, th, noimg, x);
            }
        }
        CHECK(same);
        // known points (no system message, 9500 tokens): 1024-token chunks, the 1308-token tail split at 9216
        const auto b = prompt(9500);
        CHECK(st.onSchedule(b, 0) && st.onSchedule(b, 4096) && st.onSchedule(b, 8192) && st.onSchedule(b, 9216));
        CHECK(!st.onSchedule(b, 4100) && !st.onSchedule(b, 8256));
        CHECK(st.onSchedule(a, 3000) && st.onSchedule(a, 4024) && !st.onSchedule(a, 3072));  // chunks restart at sys
        // vision: image ends at >= ck_min are splits once an mmproj is attached
        std::vector<std::uint32_t> v = prompt(6000);
        for (std::size_t i = 1500; i < 2100; ++i) v[i] = qwen35::image_id_base + 7;
        v[2100] = kVisEnd;
        st.setVision({true, kVisEnd, sched::img_split_min_default});
        CHECK(st.imgSplits(v) == (std::vector<std::uint32_t>{2101}));
        CHECK(st.onSchedule(v, 2101) && !st.onSchedule(v, 2048));
        st.setVision({true, kVisEnd, 4000});  // the image ends below ck_min: no split
        CHECK(st.imgSplits(v).empty());
        st.setVision({});
        CHECK(st.imgSplits(v).empty());
    }
    // ---- COLD-RESUME table: precise / balance x checkpoint kind x on / off the schedule of the new prompt
    {
        struct Row {
            bool precise;
            const char* kind;
            bool on;
            bool ok;
        };
        const Row rows[] = {
            {false, "prompt-end", true, true}, {false, "prompt-end", false, true},
            {false, "think-open", true, true}, {false, "think-open", false, true},
            {false, "gen-end", true, true},    {false, "gen-end", false, true},
            {false, "system", true, true},     {false, "system", false, true},
            {false, "prefix", true, true},     {false, "prefix", false, true},
            {false, "restored", true, true},   {false, "restored", false, true},
            {true, "prompt-end", true, true},  {true, "prompt-end", false, false},
            {true, "think-open", true, true},  {true, "think-open", false, false},
            {true, "gen-end", true, false},    {true, "gen-end", false, false},
            {true, "system", true, true},      {true, "system", false, false},
            {true, "prefix", true, true},      {true, "prefix", false, false},
            {true, "restored", true, true},    {true, "restored", false, false},
        };
        const auto toks = prompt(10000);  // schedule 0, 1024, ..., 8192, 9216, 10000
        const std::uint32_t pos_on = 4096, pos_off = 4100;
        int bad = 0;
        for (const Row& r : rows) {
            PrefixStore st(ct, cfgFor(r.precise));
            const std::uint32_t pos = r.on ? pos_on : pos_off;
            if (st.resumeOk(toks, pos, ckKindCode(r.kind)) != r.ok) {
                ++bad;
                std::fprintf(stderr, "  resumeOk %s %s %s: expected %d\n", r.precise ? "precise" : "balance", r.kind,
                             r.on ? "on" : "off", r.ok ? 1 : 0);
            }
            // the same rule through cacheMatch: one checkpoint of that kind at pos in a slot that cached the prompt
            TSlot sl;
            sl.cache_tokens = toks;
            sl.ckpts.resize(3);
            setCk(sl.ckpts[1], pos, r.kind, 1);
            const PrefixStore::Match m = st.cacheMatch(sl, toks);
            const bool used = m.ck == &sl.ckpts[1] && m.reuse == pos && m.lcp == toks.size();
            const bool none = m.ck == nullptr && m.reuse == 0 && m.lcp == toks.size();
            if (r.ok ? !used : !none) {
                ++bad;
                std::fprintf(stderr, "  cacheMatch %s %s %s: reuse %u\n", r.precise ? "precise" : "balance", r.kind,
                             r.on ? "on" : "off", m.reuse);
            }
            // the slot's checkpoints are untouched (the filter works on a copy)
            if (!sl.ckpts[1].valid) ++bad;
        }
        CHECK_EQ(bad, 0);
        CHECK_EQ(int{ckKindCode("gen-end")}, 3);
        CHECK_EQ(std::string(ckKindStr(ckKindCode("prefix"))), std::string("prefix"));
        CHECK_EQ(std::string(ckKindStr(0)), std::string("restored"));
        // an older turn's gen-end (9000) and a cold 'prefix' (8192): balance resumes at the longer, precise at the cold
        TSlot sl;
        sl.cache_tokens = toks;
        sl.ckpts.resize(3);
        setCk(sl.ckpts[0], 9000, "gen-end", 2);
        setCk(sl.ckpts[2], 8192, "prefix", 1);
        const auto more = prompt(10500);  // the same prefix, 500 tokens more
        CHECK_EQ(PrefixStore(ct, cfgFor(false)).cacheMatch(sl, more).reuse, 9000u);
        const PrefixStore::Match mp = PrefixStore(ct, cfgFor(true)).cacheMatch(sl, more);
        CHECK(mp.ck == &sl.ckpts[2] && mp.reuse == 8192 && mp.lcp == 10000);
        // a checkpoint past the common prefix is never used
        CHECK_EQ(PrefixStore(ct, cfgFor(false)).cacheMatch(sl, prompt(10000, 0, false, 1)).reuse, 0u);
        // --no-prefix-cache: nothing matches
        PrefixStore::Config dis = cfgFor(false);
        dis.enabled = false;
        CHECK_EQ(PrefixStore(ct, dis).cacheMatch(sl, more).reuse, 0u);
    }
    // ---- shared prefix checkpoints: match (prefix, on the schedule, longest) and the replacement victim
    {
        PrefixStore st(ct, cfgFor(false));
        auto& sp = st.spes();
        sp.resize(3);
        const auto a = prompt(9500, 3000);
        auto keep = [&](Spe& s, std::size_t n, const char* kind, std::uint64_t used) {
            s.valid = true;
            s.n = static_cast<std::uint32_t>(n);
            s.kind = kind;
            s.tokens.assign(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            s.last_used = used;
        };
        CHECK(st.speMatch(a) == nullptr);
        keep(sp[0], 3000, "system", 5);
        CHECK(st.speMatch(a) == &sp[0]);
        CHECK(st.speMatch(prompt(9500, 3000, false, 1)) == nullptr);  // another system message
        CHECK(st.speMatch(std::span<const std::uint32_t>(a).subspan(0, 3000)) == nullptr);  // n must be < the prompt
        keep(sp[1], 5048, "prefix", 6);  // longer and on the schedule (3000 + 2 x 1024)
        CHECK(st.speMatch(a) == &sp[1]);
        keep(sp[2], 7000, "prefix", 7);  // longer but off the new prompt's schedule
        CHECK(st.onSchedule(a, 5048) && !st.onSchedule(a, 7000));
        CHECK(st.speMatch(a) == &sp[1]);
        // victim: a free entry first, else the least recently used one that is not pinned
        sp[2].valid = false;
        CHECK_EQ(st.cache().sharedVictim(sp), 2);
        keep(sp[2], 7000, "prefix", 7);
        CHECK_EQ(st.cache().sharedVictim(sp), 0);
        sp[0].pin = true;
        CHECK_EQ(st.cache().sharedVictim(sp), 1);
        sp[1].pin = sp[2].pin = true;
        CHECK_EQ(st.cache().sharedVictim(sp), -1);
        sp[0].pin = sp[1].pin = sp[2].pin = false;
        CHECK_EQ(st.cache().findShared(sp, std::span<const std::uint32_t>(a).subspan(0, 5048)), 1);
        CHECK_EQ(st.cache().findShared(sp, std::span<const std::uint32_t>(a).subspan(0, 5000)), -1);
        // pickSlot: the shared entry counts for every free slot, so the tie goes to the least recently used slot;
        // a slot whose own checkpoint resumes further wins; a busy slot is never picked
        std::vector<TSlot> slots(3);
        for (std::size_t i = 0; i < slots.size(); ++i) {
            slots[i].ckpts.resize(3);
            slots[i].last_used = 10 + i;
        }
        auto avail = [](const TSlot& s) { return s.idle; };
        CHECK(st.pickSlot(std::span<TSlot>(slots), a, avail) == &slots[0]);
        slots[2].cache_tokens = a;
        setCk(slots[2].ckpts[0], 8192, "prompt-end", 1);
        CHECK(st.pickSlot(std::span<TSlot>(slots), a, avail) == &slots[2]);
        slots[2].idle = false;
        CHECK(st.pickSlot(std::span<TSlot>(slots), a, avail) == &slots[0]);
        slots[0].last_used = 20;
        CHECK(st.pickSlot(std::span<TSlot>(slots), a, avail) == &slots[1]);
        for (TSlot& s : slots) s.idle = false;
        CHECK(st.pickSlot(std::span<TSlot>(slots), a, avail) == nullptr);
        // no shared entry and no own checkpoint: plain LRU over the free slots (llama.cpp's fallback too)
        for (Spe& s : sp) s.valid = false;
        slots[0].idle = slots[1].idle = true;
        CHECK(st.pickSlot(std::span<TSlot>(slots), prompt(4000, 0, false, 2), avail) == &slots[1]);
    }
    // ---- TAIL-CK: the slot checkpoint at the start of the prompt's last schedule chunk
    {
        const PrefixStore bal(ct, cfgFor(false)), pre(ct, cfgFor(true));
        auto slotFor = [](std::size_t n_ck) {
            TSlot s;
            s.ckpts.resize(n_ck);
            s.on_sched = true;
            return s;
        };
        TSlot s = slotFor(3);
        const auto big = prompt(9500);
        CHECK_EQ(bal.tailCk(s, big), 9216u);
        CHECK_EQ(pre.tailCk(s, big), 9216u);
        const auto small = prompt(4000);  // below tail_ck_min (4096): balance keeps none, precise always keeps one
        CHECK_EQ(bal.tailCk(s, small), 0u);
        CHECK_EQ(pre.tailCk(s, small), 3072u);
        CHECK_EQ(bal.tailCk(slotFor(2), big), 0u);  // fewer than ckpt_keep_prefix_min checkpoints
        TSlot off = slotFor(3);
        off.on_sched = false;  // resumed off the cold schedule
        CHECK_EQ(bal.tailCk(off, big), 0u);
        PrefixStore::Config nt = cfgFor(false);
        nt.tail_ck = false;  // WHIRL_TAIL_CK=0
        CHECK_EQ(PrefixStore(ct, nt).tailCk(s, big), 0u);
        s.reuse = 9216;  // resumes there already
        CHECK_EQ(bal.tailCk(s, big), 0u);
        s.reuse = 0;
        std::uint32_t SlotPrefix::* const others[] = {&SlotPrefix::sys_split, &SlotPrefix::lcp_ck, &SlotPrefix::img_ck};
        for (std::uint32_t SlotPrefix::* f : others) {
            TSlot t = slotFor(3);
            t.*f = 9216;  // another checkpoint is kept there
            CHECK_EQ(bal.tailCk(t, big), 0u);
        }
        // think-open prompt (9500, the split at 9499): balance drops the tail (it is the think-open split itself),
        // precise moves it one schedule point earlier (the think-open split moves with the conversation)
        const auto th = prompt(9500, 0, true);
        TSlot t = slotFor(3);
        t.pf_split = static_cast<std::uint32_t>(ct.thinkOpenSplit(th));
        CHECK_EQ(t.pf_split, 9499u);
        CHECK_EQ(bal.tailCk(t, th), 0u);
        CHECK_EQ(pre.tailCk(t, th), 9216u);
        // with a system message the chunks restart at it: 3000 + 6 x 1024 = 9144, tail 356 tokens
        CHECK_EQ(bal.tailCk(s, prompt(9500, 3000)), 9144u);
    }
}

// arch024-S5a: the image-end split through the engine (attachVision passes the vision splits to the PrefixStore).
// With WHIRL_VIS_CKPT_MIN low enough for the mock's 54-token image, the first request keeps a shared 'prefix'
// checkpoint at the end of the image; the default (1024) keeps none for it.
void testVisionSplitEngine() {
    const std::string img = "data:image/bmp;base64," + b64enc(makeBmp(300, 200, 3));
    for (const std::uint32_t ck_min : {16u, 1024u}) {
        ServerSetup su;
        su.vision = true;
        su.mc.slot_ctx = 16384;
        su.vis_ck_min = ck_min;
        TestServer srv(su);
        const std::size_t before = g_log.count("at the end of the image @");
        auto a = srv.post("/v1/chat/completions", visBody({imagePart(img)}, "What is this?"));
        CHECK_EQ(a.status, 200);
        const std::size_t kept = g_log.count("at the end of the image @") - before;
        if (ck_min == 16) {
            CHECK_EQ(kept, 1u);
            CHECK(srv.engine().stats().spe_new >= 1);
        } else {
            CHECK_EQ(kept, 0u);
        }
    }
}

void testPagePool() {
    // ---- init and the free order: increasing page ids, then LIFO reuse (free stack like vLLM's free queue tail)
    {
        PpRig r(8);
        CHECK_EQ(r.pool->pages(), 8u);
        CHECK_EQ(r.pool->freeCount(), 8u);
        CHECK_EQ(r.refSum(), 0u);
        std::vector<std::int32_t> got;
        for (int i = 0; i < 3; ++i) got.push_back(*r.pool->takeFree());
        CHECK(got == (std::vector<std::int32_t>{0, 1, 2}));
        CHECK_EQ(r.pool->refCount(1), 1u);
        const std::int32_t one[] = {1};
        r.pool->unref(one);
        CHECK_EQ(r.pool->refCount(1), 0u);
        CHECK_EQ(r.pool->takeFree(), std::optional<std::int32_t>(1));  // the page just freed comes back first
        CHECK_EQ(r.pool->takeFree(), std::optional<std::int32_t>(3));
        CHECK_EQ(r.pool->freeCount(), 4u);
        CHECK_EQ(r.pool->reclaimable(), 4u);
        CHECK_EQ(PagePool::pagesFor(0), 0u);
        CHECK_EQ(PagePool::pagesFor(1), 1u);
        CHECK_EQ(PagePool::pagesFor(kv_page), 1u);
        CHECK_EQ(PagePool::pagesFor(kv_page + 1), 2u);
        // init again (govResize): every page free, no holder
        r.pool->init(5, r.arrs);
        CHECK_EQ(r.pool->freeCount(), 5u);
        CHECK_EQ(r.refSum(), 0u);
        CHECK_EQ(r.pool->takeFree(), std::optional<std::int32_t>(0));
    }
    // ---- shared prefix: ref counts (vLLM test_prefill: two requests on one prefix -> ref_cnt 2)
    {
        PpRig r(8);
        ListEvictor ev(*r.pool);
        std::vector<std::int32_t> a, b;
        CHECK(r.pool->ensure(a, 0, 3 * kv_page, ev));
        CHECK(a == (std::vector<std::int32_t>{0, 1, 2}));
        // b shares a's first two pages (attachSpe / createSpe: ref + push)
        for (std::size_t i = 0; i < 2; ++i) {
            r.pool->ref(a[i]);
            b.push_back(a[i]);
        }
        CHECK_EQ(r.pool->refCount(0), 2u);
        CHECK_EQ(r.pool->refCount(1), 2u);
        CHECK_EQ(r.pool->refCount(2), 1u);
        CHECK(r.pool->sharedFrom(a, 0));
        CHECK(r.pool->sharedFrom(a, kv_page + 1));       // keep 1: page 1 still shared
        CHECK(!r.pool->sharedFrom(a, 2 * kv_page + 1));  // keep 2: only page 2 from there, own
        CHECK(!r.pool->sharedFrom(a, 3 * kv_page + 1));  // keep past the list
        CHECK_EQ(r.pool->soleOwned(a), 1u);
        CHECK_EQ(r.pool->soleOwned(b), 0u);
        // b goes: nothing freed (a still holds them)
        r.pool->unref(b);
        b.clear();
        CHECK_EQ(r.pool->freeCount(), 5u);
        CHECK_EQ(r.pool->refCount(0), 1u);
        CHECK(!r.pool->sharedFrom(a, 0));
        r.pool->unref(a);
        a.clear();
        CHECK_EQ(r.pool->freeCount(), 8u);
        CHECK_EQ(r.refSum(), 0u);
        CHECK_EQ(ev.calls, 0u);
    }
    // ---- copy: every pool array, page length kv_page * row, at the page offsets (f16 K/V and q8h / q4 shaped K, V,
    // K scales, V scales)
    for (const auto& extra : {std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{8, 8}, std::vector<std::uint32_t>{4, 12}}) {
        PpRig r(4, extra);
        for (const KvArr& x : r.arrs) {
            auto* p = reinterpret_cast<std::uint8_t*>(x.base);
            const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * x.row;
            for (std::uint64_t i = 0; i < len; ++i) p[1 * len + i] = static_cast<std::uint8_t>(i * 7 + x.row);
        }
        const std::uint64_t c0 = r.nb.copyCalls(), b0 = r.nb.copiedBytes();
        r.pool->copy(3, 1);
        CHECK_EQ(r.nb.copyCalls() - c0, r.arrs.size());
        std::uint64_t bytes = 0;
        bool same = true, others = true;
        for (const KvArr& x : r.arrs) {
            auto* p = reinterpret_cast<const std::uint8_t*>(x.base);
            const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * x.row;
            bytes += len;
            for (std::uint64_t i = 0; i < len; ++i) {
                same = same && p[3 * len + i] == p[1 * len + i];
                others = others && p[0 * len + i] == 0 && p[2 * len + i] == 0;
            }
        }
        CHECK(same);
        CHECK(others);
        CHECK_EQ(r.nb.copiedBytes() - b0, bytes);
    }
    // ---- COW (cowPages): reuse > 0 copies the boundary page, releases the rest; spill quarantines them
    {
        PpRig r(8);
        ListEvictor ev(*r.pool);
        std::vector<std::int32_t> a, b;
        CHECK(r.pool->ensure(a, 0, 3 * kv_page, ev));
        for (std::int32_t p : a) {
            r.pool->ref(p);
            b.push_back(p);
        }
        const std::uint64_t c0 = r.nb.copyCalls();
        CHECK(r.pool->cow(a, 0, kv_page + 10, false, ev) == PagePool::Cow::done);  // keep 1
        CHECK_EQ(a.size(), 2u);
        CHECK_EQ(a[0], 0);
        CHECK_EQ(a[1], 3);  // the next free page, a copy of page 1
        CHECK_EQ(r.nb.copyCalls() - c0, r.arrs.size());
        CHECK_EQ(r.pool->refCount(0), 2u);
        CHECK_EQ(r.pool->refCount(1), 1u);  // only b now
        CHECK_EQ(r.pool->refCount(2), 1u);
        CHECK_EQ(r.pool->refCount(3), 1u);
        CHECK_EQ(r.pool->stats().cow_pages, 0u);  // counted for spills only
        CHECK(!r.pool->sharedFrom(a, kv_page + 10));
        // keep past the list: unchanged, no copy
        const std::uint64_t c1 = r.nb.copyCalls();
        CHECK(r.pool->cow(a, 0, 2 * kv_page + 1, false, ev) == PagePool::Cow::unchanged);
        CHECK_EQ(r.nb.copyCalls(), c1);
        // reuse 0: every page released, no copy
        CHECK(r.pool->cow(b, 1, 0, false, ev) == PagePool::Cow::done);
        CHECK(b.empty());
        CHECK_EQ(r.nb.copyCalls(), c1);
        CHECK_EQ(r.pool->refCount(0), 1u);
        CHECK_EQ(r.pool->freeCount() + r.refSum(), 8u);
        // spill: the released pages are quarantined (held until the spill's event), counted in cow_pages
        r.pool->setQuarantineStream(r.ts);
        CHECK(r.pool->cow(a, 0, 0, true, ev) == PagePool::Cow::done);
        CHECK(a.empty());
        CHECK_EQ(r.pool->batches(), 1u);
        CHECK_EQ(r.pool->stats().cow_pages, 2u);
        r.pool->reclaim(false);  // tracking off: every event done
        CHECK(r.pool->quarantineEmpty());
        CHECK_EQ(r.pool->freeCount(), 8u);
        // no page for the copy: no_page, the list and the counts untouched
        std::vector<std::int32_t> c, d;
        CHECK(r.pool->ensure(c, 0, 8 * kv_page, ev));
        for (std::int32_t p : c) {
            r.pool->ref(p);
            d.push_back(p);
        }
        const std::size_t refs = r.refSum();
        const std::uint64_t c2 = r.nb.copyCalls();
        CHECK(r.pool->cow(c, 0, kv_page + 1, false, ev) == PagePool::Cow::no_page);
        CHECK_EQ(c.size(), 8u);
        CHECK_EQ(r.refSum(), refs);
        CHECK_EQ(r.nb.copyCalls(), c2);
    }
    // ---- quarantine: reclaimed only after its event (tracking mode); reclaim(force) orders the model stream instead
    {
        PpRig r(4);
        ListEvictor ev(*r.pool);
        std::vector<std::int32_t> a;
        CHECK(r.pool->ensure(a, 0, 2 * kv_page, ev));
        // no tier: quarantine throws NoTier and keeps nothing
        bool threw = false;
        try {
            r.pool->quarantine(a);
        } catch (const std::runtime_error& ex) {
            threw = std::string(ex.what()) == "NoTier";
        }
        CHECK(threw);
        CHECK(r.pool->quarantineEmpty());
        r.pool->setQuarantineStream(r.ts);
        r.nb.trackPending(true);
        r.spillWork();
        r.pool->quarantine(a);  // freePages of a slot whose spill still reads them
        a.clear();
        CHECK_EQ(r.pool->batches(), 1u);
        CHECK_EQ(r.pool->freeCount(), 2u);
        CHECK_EQ(r.pool->reclaimable(), 4u);
        r.pool->reclaim(false);
        CHECK_EQ(r.pool->batches(), 1u);  // event pending: still held
        CHECK_EQ(r.pool->refCount(0), 1u);
        r.nb.completeAll();
        r.pool->reclaim(false);
        CHECK(r.pool->quarantineEmpty());
        CHECK_EQ(r.pool->freeCount(), 4u);
        CHECK_EQ(r.pool->stats().q_waits, 0u);
        // two batches, the first done, the second pending: reclaim(false) takes the first only
        std::vector<std::int32_t> x, y;
        CHECK(r.pool->ensure(x, 0, kv_page, ev));
        CHECK(r.pool->ensure(y, 1, kv_page, ev));
        r.spillWork();
        r.pool->quarantine(x);
        r.nb.completeAll();
        r.spillWork();
        r.pool->quarantine(y);
        r.pool->reclaim(false);
        CHECK_EQ(r.pool->batches(), 1u);
        CHECK_EQ(r.pool->refCount(x[0]), 0u);
        CHECK_EQ(r.pool->refCount(y[0]), 1u);
        // force: the pending one back without a host wait (stream order on the GPU), counted in q_waits
        const std::uint64_t s0 = r.nb.syncCalls();
        r.pool->reclaim(true);
        CHECK(r.pool->quarantineEmpty());
        CHECK_EQ(r.pool->refCount(y[0]), 0u);
        CHECK_EQ(r.pool->stats().q_waits, 1u);
        CHECK_EQ(r.nb.syncCalls(), s0);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        r.nb.completeAll();
        r.nb.trackPending(false);
    }
    // ---- take: free first; then the done quarantine; then the evictor (in its order); then the pending quarantine
    {
        PpRig r(4);
        r.pool->setQuarantineStream(r.ts);
        ListEvictor ev(*r.pool);
        std::vector<std::int32_t> v1, v2, q;
        CHECK(r.pool->ensure(v1, 0, kv_page, ev));      // page 0
        CHECK(r.pool->ensure(v2, 1, 2 * kv_page, ev));  // pages 1, 2
        CHECK(r.pool->ensure(q, 0, kv_page, ev));       // page 3
        CHECK_EQ(r.pool->freeCount(), 0u);
        CHECK_EQ(ev.calls, 0u);
        r.nb.trackPending(true);
        r.spillWork();
        r.pool->quarantine(q);  // pending
        ev.victims = {&v2, &v1};
        auto p = r.pool->take(ev);
        CHECK(p.has_value());
        CHECK_EQ(ev.calls, 1u);  // the evictor before the pending quarantine
        CHECK_EQ(*p, 2);         // v2 freed 1 then 2: LIFO
        CHECK(v2.empty() && !v1.empty());
        CHECK_EQ(r.pool->batches(), 1u);
        CHECK_EQ(r.pool->take(ev), std::optional<std::int32_t>(1));
        CHECK_EQ(ev.calls, 1u);  // a free page: no evictor call
        CHECK_EQ(r.pool->take(ev), std::optional<std::int32_t>(0));
        CHECK_EQ(ev.calls, 2u);  // v1 next
        // nothing left to evict: the pending quarantine with the model stream ordered after it
        CHECK_EQ(r.pool->take(ev), std::optional<std::int32_t>(3));
        CHECK_EQ(ev.calls, 3u);
        CHECK_EQ(r.pool->stats().q_waits, 1u);
        // done quarantine: taken back before the evictor is asked
        std::vector<std::int32_t> back = {3};
        r.pool->quarantine(back);
        r.nb.completeAll();
        CHECK_EQ(r.pool->take(ev), std::optional<std::int32_t>(3));
        CHECK_EQ(ev.calls, 3u);
        // all held, nothing to evict, no quarantine: nullopt, counts unchanged
        const std::size_t refs = r.refSum();
        CHECK(!r.pool->take(ev).has_value());
        CHECK_EQ(ev.calls, 4u);
        CHECK_EQ(r.refSum(), refs);
        CHECK_EQ(r.pool->freeCount(), 0u);
        r.nb.trackPending(false);
    }
    // ---- ensure: already enough (no change); more than one sequence may hold (false, nothing taken); failure keeps the
    // pages taken so far (WHIRL's partial grow; the caller throws KvPoolFull / waits) and loses none
    {
        PpRig r(4);
        ListEvictor ev(*r.pool);
        std::vector<std::int32_t> a;
        CHECK(r.pool->ensure(a, 0, 0, ev));
        CHECK(a.empty());
        CHECK(r.pool->ensure(a, 0, kv_page, ev));
        CHECK(r.pool->ensure(a, 0, kv_page - 5, ev));
        CHECK_EQ(a.size(), 1u);
        std::vector<std::int32_t> big;
        CHECK(!r.pool->ensure(big, 1, static_cast<std::uint64_t>(r.model->seqPages() + 1) * kv_page, ev));
        CHECK(big.empty());
        CHECK_EQ(r.pool->freeCount(), 3u);
        CHECK_EQ(ev.calls, 0u);
        std::vector<std::int32_t> b;
        CHECK(!r.pool->ensure(b, 1, 6 * kv_page, ev));
        CHECK_EQ(b.size(), 3u);  // partial
        CHECK_EQ(r.pool->freeCount(), 0u);
        CHECK_EQ(r.refSum(), 4u);
        CHECK_EQ(ev.calls, 1u);
        // the evictor frees a: the retry grows b by those pages
        ev.victims = {&a};
        CHECK(!r.pool->ensure(b, 1, 6 * kv_page, ev));
        CHECK_EQ(b.size(), 4u);
        CHECK_EQ(r.refSum(), 4u);
        CHECK(r.pool->ensure(b, 1, 4 * kv_page, ev));
    }
}

struct CkRig {
    NullBackend nb;
    std::unique_ptr<MockModel> model;
    std::unique_ptr<CkptStore> cs;
    device::Stream ts = nullptr;  // the tier stream
    void* dst = nullptr;          // the RAM tier block the spill writes
    std::size_t live0 = 0;
    std::uint64_t dev0 = 0;
    explicit CkRig(bool ckpt_host = false) {
        MockConfig mc;
        mc.parallel = 2;
        mc.pool_tokens = 4096;
        model = std::make_unique<MockModel>(nb, tok(), mc);
        cs = std::make_unique<CkptStore>(nb, *model, CkptStore::Layout{model->convBytes(), model->ssmBytes(), ckpt_host});
        ts = nb.streamCreateNonBlocking();
        dst = nb.hostMalloc(1 << 20);
        live0 = nb.liveAllocations();
        dev0 = nb.deviceBytes();
        nb.trackPending(true);
    }
    static DevPtr base(const Ckpt& c) { return c.dev ? c.dev.get() : c.host.addr(); }
    // spill of these checkpoints, fenced by ev (made if empty)
    void spill(std::span<const Ckpt> cks, device::EventH& ev) {
        for (const Ckpt& c : cks) nb.copyAsync(reinterpret_cast<DevPtr>(dst), base(c), 64, ts);
        if (!ev) ev = device::EventH::create(nb);
        nb.eventRecord(ev.get(), ts);
    }
    void spill(Slot& sl) { spill(sl.ckpts, sl.spill_ev); }
    Slot slot(std::size_t n = 2) {
        Slot sl;
        sl.ckpts.resize(n);
        cs->unpark(sl.ckpts, true);
        return sl;
    }
    // the rest of the teardown is not under test; true when nothing leaked
    bool finish() {
        nb.completeAll();
        nb.hostFree(dst);
        nb.streamDestroy(ts);
        cs.reset();
        model.reset();
        return nb.clean();
    }
};

void testCkptStore() {
    // ---- host_ok branch and sizes (the formula of Engine::initCkpt / ckptVram before arch024 S3)
    for (const bool ckpt_host : {false, true}) {
        CkRig r(ckpt_host);
        const auto& cfg = r.model->cfg();
        std::uint64_t n_gdn = 0;
        for (auto p : r.model->ssmState())
            if (p) ++n_gdn;
        CHECK(n_gdn > 0);
        const std::uint64_t parts = n_gdn * (al256(32) + al256(64));
        const std::uint64_t hid = al256(static_cast<std::uint64_t>(cfg.n_embd) * 4);
        const std::uint64_t full = parts + hid + static_cast<std::uint64_t>(cfg.n_vocab) * 4;
        CHECK_EQ(r.cs->bytes(false), full);
        CHECK_EQ(r.cs->bytes(true), ckpt_host ? hid : full);
        Ckpt a, b;
        const long host0 = r.nb.liveHost();
        r.cs->init(a, true);   // a per-slot checkpoint
        r.cs->init(b, false);  // a shared prefix checkpoint: always VRAM
        if (ckpt_host) {
            CHECK(static_cast<bool>(a.host) && static_cast<bool>(a.hid_buf) && !a.dev);
            CHECK_EQ(a.host.bytes(), parts + static_cast<std::uint64_t>(cfg.n_vocab) * 4);
            CHECK_EQ(a.hid_buf.bytes(), static_cast<std::uint64_t>(cfg.n_embd) * 4);
            CHECK_EQ(a.hid, a.hid_buf.get());
            CHECK_EQ(a.logits, a.host.addr() + parts);
            CHECK_EQ(r.nb.liveHost(), host0 + 1);
            CHECK_EQ(r.nb.liveAllocations(), r.live0 + 3);
        } else {
            CHECK(static_cast<bool>(a.dev) && !a.host && !a.hid_buf);
            CHECK_EQ(a.dev.bytes(), full);
            CHECK_EQ(a.hid, a.dev.get() + parts);
            CHECK_EQ(a.logits, a.dev.get() + parts + hid);
            CHECK_EQ(r.nb.liveHost(), host0);
            CHECK_EQ(r.nb.liveAllocations(), r.live0 + 2);
        }
        CHECK(static_cast<bool>(b.dev) && !b.host);
        CHECK_EQ(b.dev.bytes(), full);
        CHECK_EQ(r.nb.deviceBytes(), r.dev0 + (ckpt_host ? a.hid_buf.bytes() : full) + full);
        // conv / ssm slices exactly on the DeltaNet layers, in layer order, 256-byte aligned parts
        std::uint64_t at = CkRig::base(a);
        bool slices = a.conv.size() == cfg.n_layer && a.ssm.size() == cfg.n_layer;
        for (std::uint32_t li = 0; slices && li < cfg.n_layer; ++li) {
            if (r.model->ssmState()[li] == 0) {
                slices = a.conv[li] == 0 && a.ssm[li] == 0;
                continue;
            }
            slices = a.conv[li] == at && a.ssm[li] == at + al256(32);
            at += al256(32) + al256(64);
        }
        CHECK(slices);
        // park -> unpark: live counts and device bytes back to the same
        const std::size_t live1 = r.nb.liveAllocations();
        const std::uint64_t dev1 = r.nb.deviceBytes();
        std::vector<Ckpt> v(1);
        v[0] = std::move(a);
        r.cs->park(v, {});
        CHECK(v[0].conv.empty() && v[0].hid == 0 && v[0].logits == 0 && !v[0].dev && !v[0].host && !v[0].hid_buf);
        CHECK(r.nb.liveAllocations() < live1);
        r.cs->unpark(v, true);
        CHECK_EQ(r.nb.liveAllocations(), live1);
        CHECK_EQ(r.nb.deviceBytes(), dev1);
        r.cs->release(v[0], {});
        r.cs->release(b, {});
        CHECK_EQ(r.nb.liveAllocations(), r.live0);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        CHECK(r.finish());
    }
    // ---- save / load: the copies between the current sequence's state and the checkpoint
    {
        CkRig r;
        Ckpt c;
        r.cs->init(c, true);
        r.nb.trackPending(false);
        std::uint64_t n_gdn = 0;
        for (auto p : r.model->ssmState())
            if (p) ++n_gdn;
        const std::uint64_t n0 = r.nb.copyCalls();
        r.cs->save(c, std::nullopt, r.model->logits());
        CHECK_EQ(r.nb.copyCalls() - n0, 2 * n_gdn + 1);
        CHECK(c.has_logits);
        const std::uint64_t n1 = r.nb.copyCalls();
        r.cs->save(c, r.model->mtpH(), std::nullopt);
        CHECK_EQ(r.nb.copyCalls() - n1, 2 * n_gdn + 1);
        CHECK(!c.has_logits);
        const std::uint64_t n2 = r.nb.copyCalls();
        r.cs->load(c);
        CHECK_EQ(r.nb.copyCalls() - n2, 2 * n_gdn);
        r.cs->release(c, {});
        CHECK(r.finish());
    }
}

// F1: every way a slot / shared checkpoint can go away with a spill pending waits first (0 violations, a sync ran);
// freeing past the fence is caught (the harness is not blind)
void testCkptF1() {
    static_assert(std::is_nothrow_move_constructible_v<Slot> && std::is_nothrow_move_assignable_v<Slot>);
    static_assert(std::is_nothrow_move_constructible_v<Spe> && std::is_nothrow_move_assignable_v<Spe>);
    static_assert(!std::is_copy_constructible_v<Slot> && !std::is_copy_assignable_v<Slot>);
    static_assert(!std::is_copy_constructible_v<Spe> && !std::is_copy_assignable_v<Spe>);
    auto pass = [](CkRig& r, std::uint64_t syncs0, const char* what) {
        const bool ok = r.nb.pendingFreeViolations() == 0 && r.nb.syncCalls() > syncs0;
        if (!ok)
            std::fprintf(stderr, "  F1 %s: violations %zu, syncs %llu -> %llu\n", what, r.nb.pendingFreeViolations(),
                         static_cast<unsigned long long>(syncs0), static_cast<unsigned long long>(r.nb.syncCalls()));
        CHECK(ok);
    };
    {  // vector clear
        CkRig r;
        std::vector<Slot> v;
        v.push_back(r.slot());
        v.push_back(r.slot());
        r.spill(v[0]);
        r.spill(v[1]);
        const auto s0 = r.nb.syncCalls();
        v.clear();
        pass(r, s0, "clear");
        CHECK(r.finish());
    }
    {  // vector growth (moves free nothing, need no wait), then resize smaller
        CkRig r;
        std::vector<Slot> v;
        v.push_back(r.slot());
        r.spill(v[0]);
        const auto sg = r.nb.syncCalls();
        for (int i = 0; i < 8; ++i) v.push_back(r.slot());  // reallocations move v[0] with its spill pending
        CHECK_EQ(r.nb.syncCalls(), sg);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        r.spill(v[8]);
        const auto s0 = r.nb.syncCalls();
        v.resize(8);
        pass(r, s0, "resize");
        v.clear();
        CHECK(r.finish());
    }
    {  // exception unwinding
        CkRig r;
        const auto s0 = r.nb.syncCalls();
        try {
            Slot sl = r.slot();
            r.spill(sl);
            throw std::runtime_error("restore failed");
        } catch (const std::runtime_error&) {
        }
        pass(r, s0, "unwind");
        CHECK(r.finish());
    }
    {  // move-assign over a slot whose spill is pending
        CkRig r;
        Slot a = r.slot();
        Slot b = r.slot();
        const DevPtr b_dev = b.ckpts[0].dev.get();
        r.spill(a);
        const auto s0 = r.nb.syncCalls();
        a = std::move(b);
        pass(r, s0, "move-assign");
        CHECK(a.ckpts.size() == 2 && a.ckpts[0].dev.get() == b_dev);
        CHECK(b.ckpts.empty() && !b.spill_ev);
        CHECK_EQ(r.nb.liveAllocations(), r.live0 + 2);
        a = Slot{};
        CHECK(r.finish());
    }
    {  // governor park (govFreeCache's call: parkSlotCkpts passes the slot's spill fence)
        CkRig r;
        Slot sl = r.slot();
        r.spill(sl);
        const auto s0 = r.nb.syncCalls();
        const std::uint64_t freed = parkSlotCkpts(*r.cs, sl);
        pass(r, s0, "park");
        CHECK_EQ(freed, 2 * r.cs->bytes(true));
        CHECK(sl.parked && sl.ckpts.empty());
        CHECK_EQ(r.nb.liveAllocations(), r.live0);
        sl.spill_ev.reset();
        CHECK(r.finish());
    }
    {  // shared prefix checkpoints: spill (reads ck) and ready (a copy into ck on another stream)
        CkRig r;
        device::Stream s2 = r.nb.streamCreateNonBlocking();
        std::vector<Spe> v(2);
        for (Spe& sp : v) r.cs->init(sp.ck, false);
        r.spill(std::span<const Ckpt>(&v[0].ck, 1), v[0].spill_ev);
        r.nb.copyAsync(v[1].ck.dev.get(), reinterpret_cast<DevPtr>(r.dst), 64, s2);
        v[1].ready_ev = device::EventH::create(r.nb);
        r.nb.eventRecord(v[1].ready_ev.get(), s2);
        const auto s0 = r.nb.syncCalls();
        v.clear();
        pass(r, s0, "spe clear");
        CHECK(r.nb.syncCalls() >= s0 + 2);
        // release with the fences passed (govFreeCache's shared part)
        Spe sp;
        r.cs->init(sp.ck, false);
        r.spill(std::span<const Ckpt>(&sp.ck, 1), sp.spill_ev);
        const auto s1 = r.nb.syncCalls();
        r.cs->release(sp.ck, {sp.spill_ev.get(), sp.ready_ev.get()});
        pass(r, s1, "spe release");
        sp = Spe{};
        r.nb.streamDestroy(s2);
        CHECK(r.finish());
    }
    {  // the harness catches a free past the fence: release without the fence, with a spill pending
        CkRig r;
        Slot sl = r.slot(1);
        r.spill(sl);
        r.cs->release(sl.ckpts[0], {});
        CHECK_EQ(r.nb.pendingFreeViolations(), 1u);
        sl = Slot{};
        CHECK(r.finish());
    }
}

// arch024-S9 (M7, F1): the governor parks the shared prefix checkpoints through PrefixStore. A pending spill (reads
// ck) and a pending ready copy (writes ck) must be waited for before the buffers go: waitSpeFences syncs both, and
// parkSpes alone (no wait first) still syncs through the fences it passes to CkptStore::release (0 violations either
// way). unparkSpes gives every entry fresh buffers; a failure part way frees what it got and rethrows (no leak, table
// empty, retry works).
void testGovParkSpes() {
    using namespace ps_test;
    const ChatTokens ct = chatIds();
    auto fill = [](CkRig& r, PrefixStore& ps, device::Stream s2) {
        ps.spes().resize(2);
        for (Spe& sp : ps.spes()) r.cs->init(sp.ck, false);
        r.spill(std::span<const Ckpt>(&ps.spes()[0].ck, 1), ps.spes()[0].spill_ev);
        r.nb.copyAsync(ps.spes()[1].ck.dev.get(), reinterpret_cast<DevPtr>(r.dst), 64, s2);
        ps.spes()[1].ready_ev = device::EventH::create(r.nb);
        r.nb.eventRecord(ps.spes()[1].ready_ev.get(), s2);
    };
    for (const bool wait_first : {true, false}) {
        CkRig r;
        PrefixStore ps(ct, cfgFor(false));
        device::Stream s2 = r.nb.streamCreateNonBlocking();
        fill(r, ps, s2);
        CHECK(r.nb.liveAllocations() > r.live0);
        const auto s0 = r.nb.syncCalls();
        if (wait_first) {
            ps.waitSpeFences();
            CHECK(r.nb.syncCalls() >= s0 + 2);  // the spill and the ready fence
        }
        const std::uint64_t b = ps.parkSpes(*r.cs);
        CHECK(r.nb.syncCalls() >= s0 + 2);      // without the wait first, release syncs the fences itself
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        CHECK_EQ(b, 2 * r.cs->bytes(false));
        CHECK(ps.spes().empty());
        CHECK_EQ(r.nb.liveAllocations(), r.live0);
        r.nb.streamDestroy(s2);
        CHECK(r.finish());
    }
    {  // unpark: fresh buffers for every entry; then a failure part way (the 2nd allocation) frees the first
        CkRig r;
        PrefixStore ps(ct, cfgFor(false));
        const std::size_t live = r.nb.liveAllocations();
        const std::uint64_t dev = r.nb.deviceBytes();
        CHECK_EQ(ps.unparkSpes(*r.cs, 3), 3 * r.cs->bytes(false));
        CHECK_EQ(ps.spes().size(), 3u);
        CHECK(r.nb.liveAllocations() > live);
        CHECK_EQ(r.nb.deviceBytes() - dev, 3 * r.cs->bytes(false));
        CHECK_EQ(ps.parkSpes(*r.cs), 3 * r.cs->bytes(false));
        CHECK_EQ(r.nb.liveAllocations(), live);
        r.nb.failMalloc(2);
        bool threw = false;
        try {
            (void)ps.unparkSpes(*r.cs, 3);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
        CHECK(ps.spes().empty());
        CHECK_EQ(r.nb.liveAllocations(), live);
        CHECK_EQ(r.nb.deviceBytes(), dev);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        CHECK_EQ(ps.unparkSpes(*r.cs, 2), 2 * r.cs->bytes(false));  // the next try starts clean
        CHECK_EQ(ps.parkSpes(*r.cs), 2 * r.cs->bytes(false));
        CHECK(r.finish());
    }
}

// arch024-S9 C-3: server_main.cpp reads other processes' VRAM and its pinned host memory through device::Backend.
// (1) the NullBackend fakes and their thread record; (2) the startup margin input (poolbudget::MarginIn as
// server_main's budgetMargin builds it from others0) follows the backend's value; (3) the spill warning path as
// server_main wires it (the probe's pinned from pinnedHostBytes, the others lambda from gpuOthers): pinned growth is
// not a spill, other growth is, and the warning names the other programs' VRAM now and at startup.
void testBackendOthersPinned() {
    constexpr std::uint64_t MiB = 1ull << 20;
    {  // (1)
        NullBackend nb;
        CHECK(!nb.gpuOthers(0).ok);
        CHECK_EQ(nb.pinnedHostBytes(), 0u);
        nb.mem.others.ok = true;
        nb.mem.others.others = 3 * 1024 * MiB;
        nb.mem.pinned = 17 * 1024 * MiB;
        nb.recordThreads(true);
        CHECK(nb.gpuOthers(1).ok);
        CHECK_EQ(nb.gpuOthers(1).others, 3 * 1024 * MiB);
        CHECK_EQ(nb.pinnedHostBytes(), 17 * 1024 * MiB);
        const auto calls = nb.callThreads();
        CHECK_EQ(calls.size(), 3u);
        if (calls.size() == 3) {
            CHECK(std::string(NullBackend::callName(calls[0].first)) == "gpuOthers");
            CHECK(std::string(NullBackend::callName(calls[2].first)) == "pinnedHostBytes");
        }
        nb.recordThreads(false);
    }
    {  // (2)
        NullBackend nb;
        device::Backend& be = nb;
        auto margin = [&] {
            const hip::GpuOthers o = be.gpuOthers(0);
            poolbudget::MarginIn mi;
            mi.local_budget = 30000 * MiB;
            mi.budget_start = 30000 * MiB;
            mi.others_known = o.ok;
            mi.others = o.others;
            mi.display = o.display_known && o.display;
            return poolbudget::defaultMargin(mi).bytes;
        };
        const std::uint64_t unknown = margin();
        nb.mem.others.ok = true;
        nb.mem.others.others = 0;
        const std::uint64_t none = margin();
        nb.mem.others.others = 4096 * MiB;
        nb.mem.others.display_known = nb.mem.others.display = true;
        const std::uint64_t busy = margin();
        CHECK(unknown > 0 && none > 0);
        CHECK(busy > none);  // the margin grows with what other programs use (and a monitor on the card)
    }
    {  // (3)
        auto waitFor = [](const std::function<bool()>& f, int ms) {
            for (int i = 0; i < ms / 10; ++i) {
                if (f()) return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return f();
        };
        std::mutex mu;
        std::uint64_t shared = 400 * MiB;
        std::atomic<NullBackend*> opsp{nullptr};
        ServerSetup su = baseSetup();
        su.vram_probe = [&] {
            std::lock_guard<std::mutex> lk(mu);
            whirl::wddmwatch::Sample x;
            x.ok = true;
            x.local_usage = 29000 * MiB;
            x.local_budget = 31232 * MiB;
            x.nonlocal_usage = shared;
            NullBackend* o = opsp.load();
            x.pinned = o ? o->pinnedHostBytes() : 0;  // server_main: s.pinned = be->pinnedHostBytes()
            return x;
        };
        su.vram_cfg.idle_sample_s = 0.2;
        const hip::GpuOthers o0{true, 0, 0, 205 * MiB, false, false, 0};  // others0, read at startup
        su.vram_others = [&opsp, o0]() -> std::optional<whirl::wddmwatch::Others> {
            NullBackend* o = opsp.load();
            if (!o) return std::nullopt;
            const hip::GpuOthers g = o->gpuOthers(0);  // server_main: be->gpuOthers(dev)
            if (!g.ok) return std::nullopt;
            return whirl::wddmwatch::Others{g.others, o0.ok ? std::optional<std::uint64_t>(o0.others) : std::nullopt};
        };
        g_log.clear();
        TestServer srv(su);
        {
            std::lock_guard<std::mutex> lk(mu);
            srv.ops().mem.others.ok = true;
            srv.ops().mem.others.others = 3328 * MiB;
            opsp = &srv.ops();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // 600 MiB more non-local, all of it pinned host memory WHIRL allocated: not a spill
        {
            std::lock_guard<std::mutex> lk(mu);
            srv.ops().mem.pinned = 600 * MiB;
            shared = 1000 * MiB;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        CHECK_EQ(g_log.count(" W GPU memory spill"), 0u);
        CHECK(srv.get("/health").body.find("\"spill\":false") != std::string::npos);
        // 600 MiB more that is not pinned: a spill; the warning names the other programs' VRAM (gpuOthers)
        {
            std::lock_guard<std::mutex> lk(mu);
            shared = 1600 * MiB;
        }
        CHECK(waitFor([&] { return g_log.count(" W GPU memory spill: ") == 1; }, 5000));
        CHECK_EQ(g_log.count("Other programs use 3.25 GiB of VRAM on this GPU now (0.20 GiB when the server started)."), 1u);
        CHECK(waitFor([&] { return srv.get("/health").body.find("\"spill\":true") != std::string::npos; }, 5000));
        opsp = nullptr;
    }
}

// govRestoreCache's failure cleanup passes {}: the buffers it frees were just allocated, so no spill has read them;
// another slot's spill still pending on the tier stream does not count against them (0 violations, no sync needed)
void testCkptRestoreFail() {
    for (const std::uint64_t nth : {1ull, 2ull}) {
        CkRig r;
        Slot busy = r.slot();
        r.spill(busy);  // pending all through the restore
        Slot sl = r.slot();
        r.spill(sl);
        parkSlotCkpts(*r.cs, sl);  // the slot was parked after its spill
        const std::size_t live_parked = r.nb.liveAllocations();
        const auto s0 = r.nb.syncCalls();
        r.nb.failMalloc(nth);
        bool threw = false;
        try {
            sl.ckpts.clear();
            sl.ckpts.resize(2);
            r.cs->unpark(sl.ckpts, true);
        } catch (const std::exception&) {
            threw = true;
            r.cs->park(sl.ckpts, {});
            sl.ckpts.clear();
        }
        CHECK(threw);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        CHECK_EQ(r.nb.syncCalls(), s0);
        CHECK_EQ(r.nb.liveAllocations(), live_parked);
        // the shared prefix checkpoints' cleanup the same way
        std::vector<Spe> spes(2);
        r.nb.failMalloc(2);
        threw = false;
        try {
            for (Spe& sp : spes) r.cs->init(sp.ck, false);
        } catch (const std::exception&) {
            threw = true;
            for (Spe& sp : spes) r.cs->release(sp.ck, {});
            spes.clear();
        }
        CHECK(threw);
        CHECK_EQ(r.nb.pendingFreeViolations(), 0u);
        CHECK_EQ(r.nb.liveAllocations(), live_parked);
        r.nb.failMalloc(0);
        busy = Slot{};
        sl = Slot{};
        CHECK(r.finish());
    }
}

// arch024 S6a: the tier IO thread through the server -- every Backend call it makes is setDevice or eventSync,
// including what the engine's wake callback (Engine::attachTier's setWake lambda, run on the IO thread after each
// job) does. Spills and SSD writes on one server, an SSD restore (a read job with a fence) after a restart.
void testTierIoThreadCalls() {
    using Call = NullBackend::Call;
    const std::filesystem::path base = whirl::test::testTmpBase();  // WHIRL_TEST_TMP, else %TEMP%\whirl-tests\<pid>
    const std::string dir = narrow((base / L"server_test_ssd_io").wstring());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 4096;
    su.tier = true;
    su.ssd_dir = dir;
    su.tier_min = 512;
    su.eo.n_ck = 2;
    const auto pa = makePrompt(3000, 511);
    const auto pb = makePrompt(3200, 512);
    std::vector<std::uint32_t> pa2;
    std::size_t io_calls = 0, io_sync = 0, io_other = 0, io_dev = 0;
    auto tally = [&](TestServer& srv, std::thread::id io) {
        for (const auto& [c, th] : srv.ops().callThreads()) {
            if (th != io) continue;
            ++io_calls;
            if (c == Call::event_sync) {
                ++io_sync;
            } else if (c == Call::set_device) {
                ++io_dev;
            } else {
                ++io_other;
                std::printf("  tier_io_thread_calls: the IO thread called %s\n", NullBackend::callName(c));
            }
        }
    };
    {
        TestServer srv(su);
        const std::thread::id io = srv.tierp()->ioThreadId();
        CHECK(io != std::thread::id{});
        srv.ops().recordThreads(true);
        const auto outa = srv.model().greedyReference(pa, 40, true);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
        pa2 = pa;
        pa2.insert(pa2.end(), outa.begin(), outa.end());
        pa2 = makePrompt(static_cast<std::uint32_t>(pa2.size() + 300), 513, pa2);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa2, 40))), decode(srv.model().greedyReference(pa2, 40, true)));
        for (int i = 0; i < 400 && srv.ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(srv.ssdEntries() >= 1);
        srv.shutdown();
        tally(srv, io);
    }
    const std::size_t calls_first = io_calls;
    {
        TestServer srv(su);
        const std::thread::id io = srv.tierp()->ioThreadId();
        srv.ops().recordThreads(true);
        CHECK(srv.ssdEntries() >= 1);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierStats().restores_ssd >= 1);
        srv.shutdown();
        tally(srv, io);
    }
    std::printf("  tier_io_thread_calls: IO-thread calls %zu (first server %zu): eventSync %zu, setDevice %zu, other %zu\n", io_calls,
                calls_first, io_sync, io_dev, io_other);
    CHECK_EQ(io_other, 0u);
    CHECK(io_sync >= 1);  // the SSD restore's read waited for its fence on the IO thread (the test is not blind)
    std::filesystem::remove_all(dir, ec);
}

// ~Engine with the tier attached, tracking on from before the last request (its request-end spill and the stop's
// flush are tracked): walks the server's teardown order once (tier, then engine, then model) and reports whether a
// slot's spill was still pending when ~Engine ran (the Slot destructor's wait); ~Engine frees nothing past a fence
// arch024-S14 (PLAN c-3): the CPU part of the environment matrix, as tables over the pure functions the server and
// --list-devices call (no re-implementation of server_main's glue; the GPU cells are tools/gates/env_matrix.ps1).
// The machine is the owner's: R9700 (gfx1201, 31.9 GiB, USB4 eGPU at pci:71:0) and the Radeon 8060S (gfx1151, BIOS
// carve-out 96 GB = 95.8 GiB dedicated, pci:196:0) in a 128 GB ROG Flow Z13 (Windows sees 33981464576 bytes at 96 GB).
//   E1 one vs two GPUs visible: the default and --device in both enumeration orders; the RAM tier's pinned room with
//      one and with two HIP GPUs (a pinned buffer is charged on each: s8-719); "another program pins memory" (v3 C-b2):
//      the room shrinks by what that program pins (startup inputs' adapter_shared / pinned_max), never grows.
//   E2 (selection part) the default GPU follows the carve-out: at 16 GB the R9700 has more memory and is the default.
//   E3 R9700 hidden (HIP_VISIBLE_DEVICES): the default is the 8060S, --device pci:71:0 / r9700 fail with a text that
//      lists the GPU that is there.
//   E4 a program holding VRAM (vram_hog) raises the startup margin; one holding pinned memory lowers the room.
//   E5 the automatic RAM tier at 32 / 64 / 128 GB of system RAM, then the 20480 MiB cap first (v0.2.2.2 order).
//   E6 (CPU part) the eGPU replugged on another port gets another bus number: pci:71:0 then fails, r9700 still works.
void testEnvMatrix() {
    using namespace whirl::gpusel;
    constexpr std::uint64_t MiB = 1ull << 20;
    const auto r9700 = [](int index, int bus) {
        GpuRecord g;
        g.index = index;
        g.name = "AMD Radeon AI PRO R9700";
        g.arch = "gfx1201";
        g.pci_bus = bus;
        g.total_mib = g.fit_mib = 32624;
        g.compute_units = 64;
        g.supported = true;
        return g;
    };
    const auto s8060 = [](int index, std::uint64_t carve_mib) {
        GpuRecord g;
        g.index = index;
        g.name = "AMD Radeon(TM) 8060S Graphics";
        g.arch = "gfx1151";
        g.pci_bus = 196;
        g.fit_mib = carve_mib;
        g.total_mib = carve_mib + 16204;
        g.compute_units = 40;
        g.integrated = g.supported = true;
        return g;
    };
    const auto nameOf = [](const std::vector<GpuRecord>& v, const std::vector<int>& sel) {
        return sel.size() == 1 ? v[static_cast<std::size_t>(sel[0])].name : std::string("?");
    };
    const auto errorOf = [](const std::vector<GpuRecord>& v, std::string_view spec) {
        try {
            (void)selectGpus(v, spec);
        } catch (const SelectError& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    // ---- E1: both visible, both enumeration orders (the HIP order is the runtime's; the choice must not depend on it)
    for (int order = 0; order < 2; ++order) {
        const std::vector<GpuRecord> both =
            order == 0 ? std::vector<GpuRecord>{r9700(0, 71), s8060(1, 98099)} : std::vector<GpuRecord>{s8060(0, 98099), r9700(1, 71)};
        CHECK(nameOf(both, selectGpus(both, "")).find("8060S") != std::string::npos);  // most GPU memory (owner 2026-10-23)
        CHECK(defaultGpu(both).reason.find("most GPU memory") == 0);
        CHECK(nameOf(both, selectGpus(both, "pci:71:0")).find("R9700") != std::string::npos);
        CHECK(nameOf(both, selectGpus(both, "pci:196:0")).find("8060S") != std::string::npos);
        CHECK(nameOf(both, selectGpus(both, "r9700")).find("R9700") != std::string::npos);
        CHECK(nameOf(both, selectGpus(both, "8060s")).find("8060S") != std::string::npos);
        // the listing marks exactly one [default], the 8060S
        const std::string lt = listingText(both, autoPick(both));
        CHECK(lt.find("[default]") != std::string::npos && lt.find("[default]") == lt.rfind("[default]"));
        const std::size_t dp = lt.find("[default]"), ls = lt.rfind(static_cast<char>(10), dp);
        CHECK(dp != std::string::npos && lt.substr(ls == std::string::npos ? 0 : ls + 1, dp - (ls == std::string::npos ? 0 : ls + 1)).find("8060S") != std::string::npos);
        // E2 selection part: a 16 GB carve-out (15.8 GiB) makes the R9700 the default
        const std::vector<GpuRecord> small =
            order == 0 ? std::vector<GpuRecord>{r9700(0, 71), s8060(1, 16179)} : std::vector<GpuRecord>{s8060(0, 16179), r9700(1, 71)};
        CHECK(nameOf(small, selectGpus(small, "")).find("R9700") != std::string::npos);
        CHECK(selectGpus(both, "") != selectGpus(both, "pci:71:0"));
        // two GPUs, one asked for: the startup line names it and how to choose the other
        const std::vector<int> pick = selectGpus(both, "pci:71:0");
        CHECK(restrictionNote(both, pick, "").find("GPU pci:71:0 (AMD Radeon AI PRO R9700): chosen with --device") == 0);
    }
    // E1 one visible (HIP_VISIBLE_DEVICES=<the R9700's ordinal>: the runtime renumbers it to 0)
    {
        const std::vector<GpuRecord> one = {r9700(0, 71)};
        CHECK(nameOf(one, selectGpus(one, "")).find("R9700") != std::string::npos);
        CHECK(nameOf(one, selectGpus(one, "pci:71:0")).find("R9700") != std::string::npos);
        CHECK(restrictionNote(one, selectGpus(one, ""), "").find(": the only GPU") != std::string::npos);
        // the user's variable is reported, never overridden
        const std::string var = userVisibilityVar([](const char* k) { return std::string_view(k) == "HIP_VISIBLE_DEVICES" ? "1" : nullptr; });
        CHECK(var == "HIP_VISIBLE_DEVICES");
        CHECK(userVarNote(var, "1").find("HIP_VISIBLE_DEVICES=1 is set, so WHIRL leaves the GPU visibility as given") == 5);
    }
    // ---- E3: the eGPU is not there (hidden, or unplugged before the start)
    {
        const std::vector<GpuRecord> igpu = {s8060(0, 98099)};
        CHECK(nameOf(igpu, selectGpus(igpu, "")).find("8060S") != std::string::npos);
        for (const char* spec : {"pci:71:0", "r9700"}) {
            const std::string e = errorOf(igpu, spec);
            if (e.empty()) std::printf("  E3: --device %s did not fail with only the 8060S\n", spec);
            CHECK(!e.empty());
            // clear: it names what was asked, lists the GPU that is there and says how to choose
            CHECK(e.find(spec) != std::string::npos);
            CHECK(e.find("GPUs in this computer:") != std::string::npos && e.find("8060S Graphics (gfx1151), pci:196:0") != std::string::npos);
            CHECK(e.find("Choose one with --device") != std::string::npos);
            CHECK(e.find("hip") == std::string::npos);  // a selection text, not a runtime error
        }
        CHECK(!errorOf(igpu, "1").empty());  // an index that is not there
    }
    // ---- E6 (CPU part): replugged on another port, the R9700 comes back with bus 72
    {
        const std::vector<GpuRecord> moved = {s8060(0, 98099), r9700(1, 72)};
        const std::string e = errorOf(moved, "pci:71:0");
        CHECK(!e.empty() && e.find("R9700 (gfx1201), pci:72:0") != std::string::npos);
        CHECK(nameOf(moved, selectGpus(moved, "r9700")).find("R9700") != std::string::npos);
        CHECK(listingText(moved, autoPick(moved)).find("check again after replugging it") != std::string::npos);
    }
    // ---- E1 / E4: the RAM tier's pinned room (startfit::pinnedRoomMb, the "startup inputs:" pin.* fields)
    // 8060S at 96 GB: SharedSystemMemory 16991223808, this process 1448083456 pinned (embedding), every program 1575485440
    // (tests/pins/startup_table.txt s8060_explicit_17346); the R9700's non-local budget 16185819136.
    const std::uint64_t ram = 33981464576ull;
    const auto igpuState = [&](std::uint64_t hog) {
        startfit::PinnedState s;
        s.total_phys = ram;
        s.pinned_max = 1575485440ull + hog;
        s.gpus.push_back({0, 1448083456ull, 16991223808ull, 1575485440ull + hog});
        return s;
    };
    const auto bothState = [&](std::uint64_t hog) {
        startfit::PinnedState s = igpuState(hog);
        s.gpus.push_back({16185819136ull, 1448083456ull, 16991223808ull, 1575485440ull + hog});  // the R9700 maps it too
        return s;
    };
    const std::uint64_t room1 = startfit::pinnedRoomMb(igpuState(0));
    const std::uint64_t room2 = startfit::pinnedRoomMb(bothState(0));
    CHECK_EQ(room1, 12288u);  // min(16204 - 1502 - 2048, 16203 - 1502 - 2048) = 12653 -> whole 512 MiB
    CHECK_EQ(room2, 4096u);   // the R9700's 15435 MiB budget / 2 GPUs - 1502 - 2048 = 4167 (s8-719: 4096 worked, 8192 failed)
    CHECK(room2 >= 4096 && room2 < 8192);
    // golden's explicit 8192 MiB tier: as given with one HIP GPU (the s8-719 root fix), clamped with two (not off)
    const auto fit = [](std::uint64_t mb, std::uint64_t room, bool expl) {
        return startfit::ramTierFit(mb, 24696061952ull, 0, expl, room, 9216);
    };
    CHECK(!fit(8192, room1, true).has_value() || fit(8192, room1, true)->mb == 8192);
    CHECK(fit(8192, room2, true).has_value() && fit(8192, room2, true)->mb == room2 && fit(8192, room2, true)->pinned);
    // another program pins memory (v3 C-b2): the room follows it down, by about what it pins, and never goes up
    std::uint64_t prev = room1;
    for (const std::uint64_t hog_mb : {0ull, 512ull, 2048ull, 4096ull, 8192ull, 12288ull}) {
        const std::uint64_t r = startfit::pinnedRoomMb(igpuState(hog_mb * MiB));
        CHECK(r <= prev);
        CHECK(r + hog_mb + 512 > room1 && r + hog_mb <= room1 + 512);  // whole 512 MiB steps (or 0 at the end)
        prev = r;
        const std::uint64_t r2 = startfit::pinnedRoomMb(bothState(hog_mb * MiB));
        CHECK(r2 <= r);  // two HIP GPUs never have more room than one
    }
    CHECK_EQ(startfit::pinnedRoomMb(igpuState(12288 * MiB)), 0u);  // 12288 MiB pinned elsewhere: no room left
    // the same on the R9700 alone in a 128 GB machine: half the RAM is far away, so only the adapter's own count (every
    // program's pinned memory on it, PDH adapter_shared; REVIEW-S8-719 C3) can lower the room
    const auto r97State = [&](std::uint64_t hog) {
        startfit::PinnedState s;
        s.total_phys = (128ull << 30) - 378273792ull;
        s.pinned_max = 950009856ull + hog;
        s.gpus.push_back({16185819136ull, 1048576000ull, 16991223808ull, 950009856ull + hog});
        return s;
    };
    CHECK_EQ(startfit::pinnedRoomMb(r97State(0)), 12288u);           // 15436 - 1000 (this process) - 2048 = 12388
    CHECK_EQ(startfit::pinnedRoomMb(r97State(4096 * MiB)), 8192u);   // 15436 - 5002 (906 + 4096 of the other program) - 2048
    // the automatic tier the room turns off is reported as off, with a reason that no log scanner reads as a failure
    const std::optional<startfit::RamFit> off = fit(8192, startfit::pinnedRoomMb(igpuState(8192 * MiB)), false);  // room 4096 < entry
    CHECK(off.has_value() && off->mb == 0 && off->below_entry);
    const std::string why = startfit::pinnedWhy(4096, 2) + startfit::pinnedOffWhy(true, 9216);
    CHECK(why.find("hipError") == std::string::npos && why.find("| failed:") == std::string::npos && why.find("PAL failed") == std::string::npos);
    // ---- E4: a program holding VRAM (vram_hog) on the card: the startup margin grows (through device::Backend)
    {
        NullBackend nb;
        device::Backend& be = nb;
        nb.mem.others.ok = true;
        std::uint64_t last = 0;
        for (const std::uint64_t hog_mb : {0ull, 2048ull, 4096ull, 8192ull}) {
            nb.mem.others.others = hog_mb * MiB;
            const hip::GpuOthers o = be.gpuOthers(0);
            poolbudget::MarginIn mi;
            mi.local_budget = mi.budget_start = 30000 * MiB;
            mi.others_known = o.ok;
            mi.others = o.others;
            const std::uint64_t m = poolbudget::defaultMargin(mi).bytes;
            CHECK(m >= last);
            last = m;
        }
    }
    // ---- E5: the automatic RAM tier by system RAM (discrete GPU), then the startup fit with the 20480 MiB cap first
    struct E5 {
        std::uint64_t gb, auto_mb, fit_mb;
        bool safe;
    };
    for (const E5& r : {E5{32, 8192, 8192, false}, E5{64, 16384, 16384, false}, E5{128, 32768, 20480, true}}) {
        tier::RamTierInput in;
        in.total_phys = (r.gb << 30) - 378273792ull;  // what Windows reports (33981464576 on the 32 GB R9700 box: measured)
        in.avail_phys = in.total_phys * 3 / 4;
        in.entry_mb = 9216;
        const tier::RamTierSize sz = tier::ramTierSize(in);
        if (sz.mb != r.auto_mb) std::printf("  E5 %llu GB: auto %llu MiB (%s)\n", static_cast<unsigned long long>(r.gb),
                                            static_cast<unsigned long long>(sz.mb), sz.reason.c_str());
        CHECK_EQ(sz.mb, r.auto_mb);
        const std::optional<startfit::RamFit> f = startfit::ramTierFit(sz.mb, in.avail_phys, sz.mb, false);
        CHECK_EQ(f ? f->mb : sz.mb, r.fit_mb);
        CHECK((f && f->safe) == r.safe);
    }
    // explicit sizes: the cap first, so asking for more never gives less (0.2.2.1 gave 22528 -> 20480 but 24064 -> 16384)
    std::uint64_t last = 0;
    for (const std::uint64_t ask : {8192ull, 16384ull, 20480ull, 22528ull, 24064ull, 32768ull}) {
        const std::optional<startfit::RamFit> f = startfit::ramTierFit(ask, 64ull << 30, 0, true);
        const std::uint64_t got = f ? f->mb : ask;
        CHECK(got >= last && got <= startfit::ram_tier_safe_mb);
        last = got;
    }
    CHECK_EQ(last, 20480u);
}

void testCkptEngineTeardown() {
    ServerSetup su = baseSetup(false);
    su.mc.parallel = 2;
    su.tier = true;
    su.tier_min = 512;
    TestServer srv(su);
    const auto p1 = makePrompt(1500, 301), p2 = makePrompt(1500, 302);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p1, 8))), decode(srv.model().greedyReference(p1, 8, true)));
    srv.ops().trackPending(true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p2, 8))), decode(srv.model().greedyReference(p2, 8, true)));
    std::uint64_t syncs_engine = 0, syncs_after = 0;
    std::size_t viol_tier = 0, viol_engine = 0, viol_after = 0;
    srv.shutdownSteps([&](const char* step) {
        const std::string s = step;
        if (s == "tier") viol_tier = srv.ops().pendingFreeViolations();
        if (s == "engine") {
            syncs_engine = srv.ops().syncCalls();
            viol_engine = srv.ops().pendingFreeViolations();
        }
        if (s == "model") {
            syncs_after = srv.ops().syncCalls();
            viol_after = srv.ops().pendingFreeViolations();
        }
    });
    CHECK_EQ(viol_after, viol_engine);  // ~Engine freed nothing past a fence
    for (const auto& v : srv.ops().violations())
        std::printf("  ckpt_engine_teardown: violation: stream %llu work %llu release #%llu %s\n", static_cast<unsigned long long>(v.stream),
                    static_cast<unsigned long long>(v.seq), static_cast<unsigned long long>(v.order), v.host ? "hostFree" : "free");
    // the number of syncs in ~Engine varies run to run (1 or 2: how many slots' spills were still unsynced); print only
    // whether one ran
    std::printf("  ckpt_engine_teardown: violations before ~Tier %zu, before ~Engine %zu, after %zu; %s\n", viol_tier, viol_engine,
                viol_after,
                syncs_after > syncs_engine ? "a spill was pending at ~Engine: the Slot / Spe wait ran"
                                           : "no spill pending at ~Engine");
    srv.ops().completeAll();
}

}  // namespace

// ---------------------------------------------------------------------------
// FIX-SRV: regression tests for the FR-SRV review findings (each failed before its fix)

std::uint64_t usageCompletion(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* u = v.get("usage");
    const json::Value* c = u ? u->get("completion_tokens") : nullptr;
    return c ? static_cast<std::uint64_t>(c->asInt()) : ~0ull;
}

std::string finishReason(const HttpResult& r) {
    const json::Value v = parseBody(r);
    const json::Value* ch = v.get("choices");
    if (!ch || !ch->isArray() || ch->asArray().empty()) return "<none>";
    const json::Value* f = ch->asArray()[0].get("finish_reason");
    return f && f->isString() ? f->asString() : "<none>";
}

bool waitIdle(TestServer& srv, double max_s) {
    const double t0 = nowSeconds();
    while (nowSeconds() - t0 < max_s) {
        const auto h = srv.get("/health");
        if (h.body.find("\"slots_busy\":0") != std::string::npos && h.body.find("\"queued\":0") != std::string::npos) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

std::string rawPost(std::uint16_t port, const std::string& body) {
    return std::format("POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                       port, body.size(), body);
}

// P2-1: a tool call cut by max_tokens / a stop string is never returned as a complete call
void testFrToolTruncated() {
    json::Array tools;
    tools.push_back(json::parse(
        R"({"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}}})"));
    // parser: a parameter without </parameter>, a call without </tool_call> / </function>
    CHECK(!parseToolCalls("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhello wor", tools));
    CHECK(!parseToolCalls("<tool_call>\n<function=run>\n<parameter=cmd>\nrm -rf build", tools));
    CHECK(!parseToolCalls("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n", tools));
    CHECK(!parseToolCalls("<tool_call>\n<function=write_file", tools));
    CHECK(!parseToolCalls("<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}", tools));  // JSON without </tool_call>
    // complete calls still parse (closed by </function> without </tool_call>: e.g. a stop string there)
    CHECK(parseToolCalls("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>", tools));
    CHECK(parseToolCalls("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>", tools));
    // the finish decision (StreamState::finish): finish_reason length keeps the text, no tool_calls
    Out o;
    o.chat = true;
    o.tools_on = true;
    o.text = "Writing.\n<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhello wor";
    o.valid = o.text.size();
    const ToolResolution a = resolveTools(o.view(true), "length", tools);
    CHECK(!a.calls);
    CHECK_EQ(a.reason, std::string("length"));
    CHECK(a.content.find("hello wor") != std::string::npos);
    // even a complete first call: the generation was cut (a later call is missing) -> text + length
    Out o2 = o;
    o2.text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<fun";
    o2.valid = o2.text.size();
    const ToolResolution b = resolveTools(o2.view(true), "length", tools);
    CHECK(!b.calls);
    CHECK_EQ(b.reason, std::string("length"));
    // FIX-TEST-A (FR-TEST P3-2): the length rule on its own. max_tokens reached right after a complete,
    // well-formed call (</tool_call> written): the text parses, but the generation was still cut, so
    // finish stays length and there are no tool_calls (mutation M3 -- parse on length too -- passed every
    // case above because each was also syntactically incomplete).
    Out o4 = o;
    o4.text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
    o4.valid = o4.text.size();
    const Out::View v4 = o4.view(true);
    CHECK(v4.tool.has_value() && parseToolCalls(*v4.tool, tools).has_value());  // the region itself is a valid call
    const ToolResolution d = resolveTools(v4, "length", tools);
    CHECK(!d.calls);
    CHECK_EQ(d.reason, std::string("length"));
    CHECK(d.content.find("</tool_call>") != std::string::npos);
    // a generation that ended normally with a complete call: tool_calls as before
    Out o3 = o;
    o3.text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
    o3.valid = o3.text.size();
    const ToolResolution c = resolveTools(o3.view(true), "stop", tools);
    CHECK(c.calls.has_value());
    CHECK_EQ(c.reason, std::string("tool_calls"));
}

// P2-2: a non-streaming request whose client disconnects frees its slot (decoding, prefilling, queued)
void testFrNonStreamDisconnect() {
    {  // decoding
        ServerSetup su = baseSetup();
        su.mc.cycle_us = 1000;
        TestServer srv(su);
        g_log.clear();
        const auto a = test::rawConnect("127.0.0.1", srv.port());
        test::rawSend(a, rawPost(srv.port(), completionBody(makePrompt(64, 81), 6000, false)));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        test::rawClose(a);
        const double t0 = nowSeconds();
        CHECK(waitIdle(srv, 5.0));
        CHECK(nowSeconds() - t0 < 1.0);
        CHECK(g_log.count("client disconnected") >= 1);
    }
    {  // prefilling (2.4 s of prefill)
        ServerSetup su = baseSetup();
        su.mc.prefill_us_per_row = 300;
        TestServer srv(su);
        const auto a = test::rawConnect("127.0.0.1", srv.port());
        test::rawSend(a, rawPost(srv.port(), completionBody(makePrompt(8000, 82), 64, false)));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        test::rawClose(a);
        const double t0 = nowSeconds();
        CHECK(waitIdle(srv, 10.0));
        CHECK(nowSeconds() - t0 < 1.0);
    }
    {  // queued behind a busy slot: never started
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.cycle_us = 1000;
        TestServer srv(su);
        g_log.clear();
        auto send = [&](std::uint64_t seed) {
            const auto s = test::rawConnect("127.0.0.1", srv.port());
            test::rawSend(s, rawPost(srv.port(), completionBody(makePrompt(64, seed), 4000, false)));
            return s;
        };
        const auto a = send(83);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const auto b = send(84);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        test::rawClose(b);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        test::rawClose(a);
        const double t0 = nowSeconds();
        CHECK(waitIdle(srv, 10.0));
        CHECK(nowSeconds() - t0 < 1.0);
        CHECK_EQ(g_log.count("starting after"), std::size_t(0));  // b never took the slot
    }
}

// P2-3: KV pool too small for concurrent long prompts: they queue (all 200), never 500
void testFrPoolPrefill() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 8192;
    TestServer srv(su);
    std::vector<std::thread> th;
    std::vector<HttpResult> rs(3);
    for (int i = 0; i < 3; ++i)
        th.emplace_back([&, i] { rs[i] = srv.post("/v1/completions", completionBody(makePrompt(6000 - 500 * i, 90 + i), 16)); });
    for (auto& t : th) t.join();
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(rs[i].status, 200);
        CHECK_EQ(completionText(rs[i]), decode(srv.model().greedyReference(makePrompt(6000 - 500 * i, 90 + i), 16, true)));
    }
    CHECK_EQ(srv.model().poison(), 0u);
    // the residual path (a prefill that still runs out of pages) answers 503 + Retry-After
    const std::string e503 = errorResponse(503, "full", "Retry-After: 5\r\n");
    CHECK(e503.starts_with("HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Type: application/json"));
}

// P3-8 (with P2-3): decode-time pool exhaustion is not a silent "length"
void testFrPoolDecode() {
    ServerSetup su = baseSetup();
    su.mc.parallel = 2;
    su.mc.slot_ctx = 8192;
    su.mc.pool_tokens = 8192;
    TestServer srv(su);
    std::vector<std::thread> th;
    std::vector<HttpResult> rs(2);
    const std::uint32_t pn[2] = {500, 3000}, mt[2] = {7000, 4500};
    for (int i = 0; i < 2; ++i)
        th.emplace_back([&, i] { rs[i] = srv.post("/v1/completions", completionBody(makePrompt(pn[i], 95 + i), mt[i])); });
    for (auto& t : th) t.join();
    int cut = 0;
    for (int i = 0; i < 2; ++i) {
        CHECK_EQ(rs[i].status, 200);
        if (usageCompletion(rs[i]) < mt[i]) {
            ++cut;
            CHECK_EQ(finishReason(rs[i]), std::string("length"));
            CHECK(rs[i].body.find("\"finish_cause\":\"kv_pool_exhausted\"") != std::string::npos);
        } else {
            CHECK(rs[i].body.find("finish_cause") == std::string::npos);
        }
    }
    CHECK(cut >= 1);
}

// P2-4: disallowed Origin -> 403 for POST (also text/plain simple requests); Host allow-list (DNS rebinding)
void testFrCrossOrigin() {
    const std::string chat = R"({"messages":[{"role":"user","content":"q"}],"max_tokens":4,"temperature":0})";
    CHECK(hostAllowed("192.168.1.5:8080", {}, "0.0.0.0"));
    CHECK(hostAllowed("[fe80::1]:80", {}, "::"));
    CHECK(!hostAllowed("evil.example", {}, "0.0.0.0"));
    CHECK(hostAllowed("192.168.1.5", {}, "192.168.1.5"));
    CHECK(!hostAllowed("192.168.1.6", {}, "192.168.1.5"));
    CHECK(hostAllowed("[fe80::1]:8080", {}, "fe80::1"));
    CHECK(hostAllowed("localhost.:8080", {}, "127.0.0.1"));
    CHECK(!hostAllowed("localhost:80x", {}, "127.0.0.1"));
    CHECK(!hostAllowed("127.0.0.1.nip.io", {}, "127.0.0.1"));
    CHECK(hostAllowed("anything.example", {"*"}, "127.0.0.1"));
    CHECK(hostAllowed("", {}, "127.0.0.1"));  // no Host header (HTTP/1.0 tools)
    {
        ServerSetup su = baseSetup();
        TestServer srv(su);
        const std::uint16_t p = srv.port();
        g_log.clear();
        const auto r = test::httpRequest("127.0.0.1", p, "POST", "/v1/completions", completionBody(makePrompt(32, 82), 8), 30,
                                         "Origin: http://evil.example\r\nContent-Type: text/plain\r\n");
        CHECK_EQ(r.status, 403);
        CHECK_EQ(g_log.count("POST /v1/completions | stream"), std::size_t(0));  // never built / run
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "POST", "/v1/chat/completions", chat, 30, "Origin: null\r\n").status, 403);
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "POST", "/v1/chat/completions", chat, 30, "Origin: http://localhost:3000\r\n").status, 200);
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "POST", "/v1/chat/completions", chat, 30).status, 200);  // no Origin (SDK, curl)
        // GET endpoints stay answerable (without CORS headers) for other origins: no state change
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "GET", "/health", "", 30, "Origin: http://evil.example\r\n").status, 200);
        // DNS rebinding: Host must be loopback (or the bound address / --allow-host)
        auto rawReq = [&](const std::string& req) {
            const auto s = test::rawConnect("127.0.0.1", p);
            test::rawSend(s, req);
            const std::string resp = test::rawRecvAll(s, 10);
            test::rawClose(s);
            return resp;
        };
        const std::string b = completionBody(makePrompt(16, 83), 4);
        auto post = [&](const std::string& host) {
            return rawReq(std::format("POST /v1/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                      host, b.size(), b));
        };
        CHECK(post("evil.example").starts_with("HTTP/1.1 403"));
        CHECK(post(std::format("evil.example:{}", p)).starts_with("HTTP/1.1 403"));
        CHECK(rawReq("GET /props HTTP/1.1\r\nHost: rebind.evil.example\r\n\r\n").starts_with("HTTP/1.1 403"));
        CHECK(post(std::format("localhost:{}", p)).starts_with("HTTP/1.1 200"));
        CHECK(post(std::format("127.0.0.1:{}", p)).starts_with("HTTP/1.1 200"));
        CHECK(post(std::format("[::1]:{}", p)).starts_with("HTTP/1.1 200"));
        CHECK(post("LOCALHOST").starts_with("HTTP/1.1 200"));
    }
    {  // --allow-host extends the Host list; --cors-origin extends the Origin list
        ServerSetup su = baseSetup();
        su.http.allowed_hosts = {"myhost.lan"};
        su.http.cors_origins = {"https://app.example.com"};
        TestServer srv(su);
        const std::uint16_t p = srv.port();
        const auto s = test::rawConnect("127.0.0.1", p);
        test::rawSend(s, std::format("GET /health HTTP/1.1\r\nHost: myhost.lan:{}\r\n\r\n", p));
        CHECK(test::rawRecvAll(s, 10).starts_with("HTTP/1.1 200"));
        test::rawClose(s);
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "POST", "/v1/chat/completions", chat, 30, "Origin: https://app.example.com\r\n").status, 200);
        CHECK_EQ(test::httpRequest("127.0.0.1", p, "POST", "/v1/chat/completions", chat, 30, "Origin: https://evil.example\r\n").status, 403);
    }
    {  // --allow-local-images: never for a web page (an Origin that is not a loopback page), even with --cors-origin *
        ServerSetup su = baseSetup();
        su.vision = true;
        su.allow_files = true;
        su.http.cors_origins = {"*"};
        TestServer srv(su);
        const std::string img = (std::filesystem::temp_directory_path() / "whirl_fr_srv_img.bmp").string();
        {
            FILE* f = std::fopen(img.c_str(), "wb");
            const std::string bmp = makeBmp(64, 64, 7);
            if (f) {
                std::fwrite(bmp.data(), 1, bmp.size(), f);
                std::fclose(f);
            }
        }
        std::string esc;
        appendJsonStr(esc, img);
        const std::string body = R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":)" + esc +
                                 R"(}},{"type":"text","text":"what"}]}],"max_tokens":4,"temperature":0})";
        const auto web = test::httpRequest("127.0.0.1", srv.port(), "POST", "/v1/chat/completions", body, 30, "Origin: https://evil.example\r\n");
        CHECK_EQ(web.status, 400);
        CHECK(web.body.find("local image") != std::string::npos);
        CHECK_EQ(test::httpRequest("127.0.0.1", srv.port(), "POST", "/v1/chat/completions", body, 30).status, 200);  // no Origin: allowed
        std::filesystem::remove(img);
    }
}

// P2-5: --draft dflash without DeltaNet replay (gfx1151 / 8060S): drafts are produced, output unchanged
void testFrDFlashNoReplay() {
    // FR-DFL P3-4: the decision server_main makes before sizing the KV pool (dflashSnapSets)
    CHECK_EQ(dflashSnapSets(false, true, false, 8, false, std::nullopt), 7u);   // block 8: one full block
    CHECK_EQ(dflashSnapSets(false, true, false, 16, false, std::nullopt), 15u);
    CHECK_EQ(dflashSnapSets(false, true, false, 8, true, std::nullopt), std::min<std::uint32_t>(qwen35::max_ng_drafts, qwen35::gdn_max_snap));
    CHECK_EQ(dflashSnapSets(false, true, false, 8, false, 12u), 12u);           // WHIRL_SNAP_SETS raises it
    CHECK_EQ(dflashSnapSets(false, true, true, 8, true, std::nullopt), 0u);     // replay: none
    CHECK_EQ(dflashSnapSets(true, false, false, 8, true, std::nullopt), 0u);    // MTP: the MTP caps decide
    CHECK_EQ(dflashSnapSets(false, false, false, 8, true, std::nullopt), 0u);
#ifdef WHIRL_SOURCE_DIR
    {
        // ... and server_main makes it after the DFlash2 setup, before the KV pool is allocated (the
        // engine's own fallback allocation comes after the pool: the pool would not leave room)
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/src/server/server_main.cpp", std::ios::binary);
        const std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::size_t setup = src.find("model.dflashSetup(opt.parallel");
        const std::size_t sets = src.find("model.ensureSnapshots(dflashSnapSets(");
        const std::size_t pool = src.find("model.allocKvPool(");
        CHECK(setup != std::string::npos && sets != std::string::npos && pool != std::string::npos);
        CHECK(setup < sets && sets < pool);
    }
#endif
    CHECK_EQ(literalSnapSets(15, std::nullopt), 15u);
    CHECK_EQ(literalSnapSets(7, 10u), 10u);
    CHECK_EQ(literalSnapSets(7, 99u), qwen35::gdn_max_snap);
    CHECK_EQ(literalSnapSets(0, std::nullopt), 1u);
    for (bool ngram : {false, true}) {
        ServerSetup su = dflashSetup(ngram);
        su.mc.replay = false;
        su.mc.dfl_miss = 0.3;
        if (ngram) su.mc.markov = 40;
        TestServer srv(su);
        g_log.clear();
        for (std::uint32_t n : {5u, 300u, 1500u}) {
            const auto p = makePrompt(n, 140 + n);
            const auto ref = srv.model().greedyReference(p, 120, true);
            auto r = srv.post("/v1/completions", completionBody(p, 120, n == 300));
            CHECK_EQ(r.status, 200);
            CHECK_EQ(n == 300 ? streamText(r, false) : completionText(r), decode(ref));
        }
        const Engine::Stats st = srv.engine().stats();
        CHECK(st.dfl_drafted > 0);
        CHECK(st.dfl_accepted > 0);
        CHECK(srv.model().snapSets() > 0);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK(waitIdle(srv, 5.0));
        CHECK(g_log.count("literal drafts/cycle (DFlash2 / n-gram)") >= 1);  // the batch log shows them
    }
    {  // MTP without replay still matches the reference (snapshot rollback path of the mock)
        ServerSetup su = baseSetup();
        su.mc.replay = false;
        TestServer srv(su);
        const auto p = makePrompt(700, 150);
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p, 100))), decode(srv.model().greedyReference(p, 100, true)));
        CHECK_EQ(srv.model().poison(), 0u);
    }
}

// FR-DFL P3-1: the VRAM fit check counts the DFlash2 drafter (official 27B drafter: weights ~1.9 GiB,
// capture 200 MiB, 80 MiB of context ring per slot, ~65 MiB scratch, snapshot sets without replay)
void testDFlashFit() {
    constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
    qwen35::DFlashConfig dc;
    dc.n_layer = 5, dc.n_embd = 5120, dc.n_head = 32, dc.n_head_kv = 8, dc.head_dim = 128, dc.n_ff = 17408;
    dc.block = 8, dc.window = 2048, dc.conv_k = 2, dc.conv_group = 16, dc.sel_topk = 16, dc.sel_rank = 256;
    dc.target_layers = {6, 20, 34, 48, 62};
    qwen35::Config t;
    t.n_embd = 5120;
    t.n_vocab = 248320;
    const std::uint64_t w = 1961 * MiB;
    const whirl::dflash::Mem m = dflashFitBytes(dc, w, t, 2048, 4, 0, 0);
    CHECK_EQ(m.weights, w);
    CHECK_EQ(m.capture, 200 * MiB);                    // 5 target layers x 2048 rows x 5120 x f32
    CHECK_EQ(m.rings, 4 * (80 * MiB + 2048 * 4));      // per slot: 5 layers x K,V x 2048 rows x 8 x 128 x f32 + positions
    CHECK_EQ(m.scratch, 67998688u);                    // dflashSetup's draft block + 2048 ingest rows
    CHECK_EQ(m.snapshots, 0u);
    CHECK(m.total() > 2 * GiB + 400 * MiB && m.total() < 2 * GiB + 500 * MiB);
    const whirl::dflash::Mem ms = dflashFitBytes(dc, w, t, 2048, 1, 15, 150 * MiB);  // replay-less: 15 sets
    CHECK_EQ(ms.snapshots, 15 * 150 * MiB);
    CHECK_EQ(ms.rings, 80 * MiB + 2048 * 4);
    // a fit that has room without the drafter is reduced once the drafter is counted
    vram::FitInput fi;
    fi.weights = 15 * GiB;
    fi.fixed = 512 * MiB;
    fi.buf_per_row = 576 * 1024;
    fi.buf_fixed = 64 * MiB;
    fi.ckpt_bytes = 150 * MiB;
    fi.parallel = 1;
    fi.kv_per_token = 64 * 1024;
    fi.pool_min_tokens = 32768;
    fi.batch = 4096;
    fi.n_ck = 4;
    fi.n_spe = 2;
    fi.avail = vram::fixedNeed(fi, 4096, 4, 2) + 40000ull * fi.kv_per_token;
    const vram::FitPlan pa = vram::planFit(fi);
    CHECK(!pa.reduced && pa.fits && pa.pool_tokens >= 39000);
    vram::FitInput fd = fi;
    fd.fixed += m.total();
    const vram::FitPlan pb = vram::planFit(fd);
    CHECK(pb.reduced);
    CHECK(pb.batch < 4096 || pb.n_ck < 4);
#ifdef WHIRL_SOURCE_DIR
    {
        // server_main adds it to the fit input before planFit, and decides --draft dflash before that
        std::ifstream in(std::string(WHIRL_SOURCE_DIR) + "/src/server/server_main.cpp", std::ios::binary);
        const std::string src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::size_t choose = src.find("whirl::dflash::choose(src, dm, why, true)");
        const std::size_t est = src.find("qwen35::estimateLoad(f, cfg, lo.embd_on_host, use_dflash)");
        const std::size_t add = src.find("fi.fixed += dm.total();");
        const std::size_t plan = src.find("vram::planFit(fi)");
        CHECK(choose != std::string::npos && est != std::string::npos && add != std::string::npos && plan != std::string::npos);
        CHECK(choose < est && est < add && add < plan);
    }
#endif
}

// P3-1: a request arriving after the engine loop has stopped gets 503 at once (no hang)
// REVIEW-ARCH4 P3-5: a runOnLoop call that times out is cancelled: when the loop later runs its queued calls, the
// timed-out closure (which may capture the caller's locals by reference) must not run.
void testRunOnLoopTimeoutCancels() {
    ServerSetup su = baseSetup();
    TestServer srv(su);
    srv.stopEngine();  // the loop has returned: nothing runs the queued call
    auto ran = std::make_shared<std::atomic<bool>>(false);
    bool threw = false;
    try {
        srv.engine().runOnLoop([ran] { ran->store(true); }, std::chrono::milliseconds(50));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    srv.engine().runLoop();  // stop_ is set: runs the queued calls once and returns
    CHECK(!ran->load());
    // (the claimed path, a call the running loop executes, is what every onEngine() reader in this file uses)
}

void testFrShutdown503() {
    ServerSetup su = baseSetup();
    TestServer srv(su);
    srv.stopEngine();
    const double t0 = nowSeconds();
    const auto r = test::httpRequest("127.0.0.1", srv.port(), "POST", "/v1/completions", completionBody(makePrompt(16, 160), 4), 5);
    CHECK_EQ(r.status, 503);
    CHECK(nowSeconds() - t0 < 2.0);
}

// P3-3, P3-4, P3-6: protocol details
void testFrProtocol() {
    // P3-3: what is streamed is always a prefix of the final (sanitized) text
    const std::vector<std::string> cases = {std::string("ab\xE0" "bc\xF0\x9F\x98\x80" "d\xC3"),
                                            std::string("\xED\xA0\x80x\xC3\xA9\xF4\x90\x80\x80y"), std::string("\x80\xBFq\xE2\x82"),
                                            std::string("ok \xE4\xBD\xA0\xE5\xA5\xBD"), std::string("\xC0\xAF" "a\xF0\x9F" "b")};
    for (const std::string& bytes : cases) {
        Out o;
        std::vector<std::string> streamed;
        for (char c : bytes) {
            o.text.push_back(c);
            o.valid = sanitizeUtf8Prefix(o.text, o.valid);
            streamed.emplace_back(o.text.substr(0, o.valid));
        }
        const std::string fin = sanitizeUtf8(o.text);
        CHECK(isValidUtf8(fin));
        for (const std::string& s : streamed) CHECK(fin.starts_with(s));
    }
    CHECK_EQ(sanitizeUtf8("ab\xE0" "bc"), std::string("ab\xEF\xBF\xBD" "bc"));
    // P3-4: a tiny positive temperature is greedy; > 100 has a correct message
    {
        const Params p = parseParams(json::parse(R"({"temperature":1e-40})").asObject(), true, true);
        CHECK(p.temperature == 0.0f);
        const Params q = parseParams(json::parse(R"({"temperature":0.0005})").asObject(), true, true);
        CHECK(q.temperature == 0.0f);
        const Params r = parseParams(json::parse(R"({"temperature":0.01})").asObject(), true, true);
        CHECK(r.temperature > 0.0f);
        std::string msg;
        try {
            parseParams(json::parse(R"({"temperature":150})").asObject(), true, true);
        } catch (const RequestError& e) {
            msg = e.what();
        }
        CHECK_EQ(msg, std::string("temperature must be in [0, 100]"));
    }
    // P3-6: stop strings: at most 16, each at most 256 bytes
    {
        auto threw = [](const std::string& js) {
            try {
                parseParams(json::parse(js).asObject(), true, true);
            } catch (const RequestError&) {
                return true;
            }
            return false;
        };
        auto list = [](int n) {
            std::string s = R"({"stop":[)";
            for (int i = 0; i < n; ++i) s += std::format("{}\"s{}\"", i ? "," : "", i);
            return s + "]}";
        };
        CHECK(threw(list(17)));
        CHECK(!threw(list(16)));
        CHECK(threw("{\"stop\":\"" + std::string(257, 'x') + "\"}"));
        CHECK(!threw("{\"stop\":\"" + std::string(256, 'x') + "\"}"));
    }
}

// P3-7: completion_tokens counts the tokens up to the one that completes the stop string
void testFrStopTokens() {
    ServerSetup su = baseSetup();  // MTP: several tokens per cycle
    TestServer srv(su);
    const auto p = makePrompt(500, 170);
    const auto ref = srv.model().greedyReference(p, 300, true);
    const std::string full = decode(ref);
    for (std::size_t at : {std::size_t(40), full.size() / 2}) {
        const std::string stop = full.substr(at, 5);
        const std::size_t end = full.find(stop) + stop.size();
        std::uint64_t want = 0;
        while (want < ref.size() && decode(std::span<const std::uint32_t>(ref.data(), want)).size() < end) ++want;
        std::string js;
        appendJsonStr(js, stop);
        auto r = srv.post("/v1/completions",
                          std::format("{{\"prompt\":{},\"max_tokens\":300,\"temperature\":0,\"ignore_eos\":true,\"stop\":[{}]}}", idsJson(p), js));
        CHECK_EQ(finishReason(r), std::string("stop"));
        CHECK_EQ(usageCompletion(r), want);
    }
}

// P3-12: a follow-up with a few new tokens on a cached prefix is not held behind a long prefill
void testFrShortFollowUp() {
    ServerSetup su = baseSetup();
    su.mc.prefill_us_per_row = 250;  // 12000 rows: 3 s of prefill
    TestServer srv(su);
    const auto p1 = makePrompt(400, 180);
    const auto o1 = srv.model().greedyReference(p1, 16, true);
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(p1, 16))), decode(o1));
    std::vector<std::uint32_t> p2 = p1;
    p2.insert(p2.end(), o1.begin(), o1.end());
    for (std::uint32_t t : makePrompt(6, 181)) p2.push_back(t);
    HttpResult rl;
    std::thread longer([&] { rl = srv.post("/v1/completions", completionBody(makePrompt(12000, 182), 4)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(400));  // the long prefill is running
    const double t0 = nowSeconds();
    const auto r2 = srv.post("/v1/completions", completionBody(p2, 1));  // done when its prefill is
    const double dt = nowSeconds() - t0;
    longer.join();
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(p2, 1, true)));
    CHECK(cachedTokens(r2) >= p1.size());  // the follow-up reused its prefix (a short chunk)
    CHECK(dt < 1.5);                       // one 2048-row chunk of the long prefill (~0.5 s), not all of it (3 s)
    CHECK_EQ(rl.status, 200);
    CHECK_EQ(srv.model().poison(), 0u);
}

// P3-11: control characters in a request path are escaped in the log
void testFrLogInjection() {
    TestServer srv(baseSetup());
    g_log.clear();
    const auto s = test::rawConnect("127.0.0.1", srv.port());
    test::rawSend(s, "GET /x\x1b[2Jy\x07 HTTP/1.1\r\nHost: localhost\r\n\r\n");
    CHECK(test::rawRecvAll(s, 10).starts_with("HTTP/1.1 404"));
    test::rawClose(s);
    CHECK_EQ(g_log.count("/x\\x1b[2Jy\\x07 -> 404"), std::size_t(1));
    CHECK_EQ(g_log.count("\x1b"), std::size_t(0));
}

namespace {

// ---------------------------------------------------------------------------
// arch024-S6b: TierBridge (M5, spill half) driven directly. MockModel + NullBackend (host memory) and a real RAM-only
// tier::Tier as the tier (no SSD directory, so its IO thread copies nothing): the spans are checked against the
// model's pool arrays and checkpoint buffers, and the arena against the sources (Tier::verifySpans).

struct TbRig {
    NullBackend nb;
    std::unique_ptr<MockModel> model;
    std::vector<KvArr> kv;
    std::unique_ptr<ChatTokens> ct;
    std::unique_ptr<PrefixStore> ps;
    std::unique_ptr<CkptStore> cs;
    std::unique_ptr<tier::Tier> tier;
    std::unique_ptr<TierBridge> tb;
    std::uint32_t n_gdn = 0;
    explicit TbRig(std::vector<std::uint32_t> extra_rows = {}, const TierBridge::Config& tc = {}, std::uint32_t min_tok = 64) {
        MockConfig mc;
        mc.parallel = 2;
        mc.slot_ctx = 16384;
        mc.pool_tokens = 131072;  // 512 pages of the mock's KV page
        mc.mtp = false;
        mc.kv_extra_rows = std::move(extra_rows);
        model = std::make_unique<MockModel>(nb, tok(), mc);
        kv = model->kvArrays();
        ct = std::make_unique<ChatTokens>(tok());
        ps = std::make_unique<PrefixStore>(*ct, PrefixStore::Config{});
        cs = std::make_unique<CkptStore>(nb, *model, CkptStore::Layout{model->convBytes(), model->ssmBytes(), false});
        tier = tier::Tier::create(nb, 64ull << 20, 42, 0);
        tier::Config c;
        c.ram_bytes = 64ull << 20;
        c.min_tokens = min_tok;
        c.ssd_delay_ms = 0;
        tier::Layout lay;
        const auto& cfg = model->cfg();
        for (auto p : model->ssmState())
            if (p) ++n_gdn;
        lay.ck_bytes = n_gdn * (model->convBytes() + model->ssmBytes()) + static_cast<std::uint64_t>(cfg.n_embd) * 4 +
                       static_cast<std::uint64_t>(cfg.n_vocab) * 4;
        std::uint64_t per_tok = 0;
        for (const KvArr& a : kv) per_tok += a.row;
        lay.page_bytes = per_tok * qwen35::kv_page;
        lay.page_tokens = qwen35::kv_page;
        lay.tok_cap = mc.slot_ctx;
        tb = std::make_unique<TierBridge>(nb, *model, *ct, *ps, kv, tc);
        tb->attach(tier.get());
        ps->attachTier(tier.get());
        tier->start(c, lay, 7);
    }
    ~TbRig() {
        tb->attach(nullptr);
        ps->attachTier(nullptr);
        tier.reset();
    }
    // a slot holding toks with n checkpoints (seq order given), pages first_page.. (enough for the tokens)
    struct S {
        SlotPrefix sp;
        SlotTier st;
    };
    std::unique_ptr<S> slot(const std::vector<std::uint32_t>& toks, std::vector<std::uint32_t> ck_pos,
                            std::vector<std::uint64_t> ck_seq, std::int32_t first_page) {
        auto s = std::make_unique<S>();
        s->sp.cache_tokens = toks;
        s->sp.ckpts.resize(ck_pos.size());
        for (std::size_t j = 0; j < ck_pos.size(); ++j) {
            Ckpt& c = s->sp.ckpts[j];
            cs->init(c, true);
            // distinct bytes per checkpoint, so a swapped source shows in the arena
            std::vector<std::uint8_t> fill(static_cast<std::size_t>(cs->bytes(true)), static_cast<std::uint8_t>(0x10 + j));
            std::memcpy(reinterpret_cast<void*>(c.dev.get()), fill.data(), fill.size());
            c.valid = ck_pos[j] > 0;
            c.pos = ck_pos[j];
            c.seq = ck_seq[j];
            c.kind = "prompt-end";
        }
        const std::size_t np = (toks.size() + qwen35::kv_page - 1) / qwen35::kv_page;
        for (std::size_t p = 0; p < np; ++p) s->sp.pages.push_back(first_page + static_cast<std::int32_t>(p));
        s->sp.ckpt_seq = 10;
        return s;
    }
    // "kv<a>:p<page>" or "ck<j>.<part>" for a span source, "?" when it is neither
    std::string where(DevPtr p, const S& s) const {
        const std::uint64_t pool_pages = model->poolPages();
        for (std::size_t a = 0; a < kv.size(); ++a) {
            const std::uint64_t len = static_cast<std::uint64_t>(qwen35::kv_page) * kv[a].row;
            if (p >= kv[a].base && p < kv[a].base + pool_pages * len) {
                if ((p - kv[a].base) % len != 0) return "kv" + std::to_string(a) + ":unaligned";
                return "kv" + std::to_string(a) + ":p" + std::to_string((p - kv[a].base) / len);
            }
        }
        for (std::size_t j = 0; j < s.sp.ckpts.size(); ++j) {
            const Ckpt& c = s.sp.ckpts[j];
            if (p == c.hid) return "ck" + std::to_string(j) + ".hid";
            if (p == c.logits) return "ck" + std::to_string(j) + ".logits";
            for (std::size_t l = 0; l < c.conv.size(); ++l) {
                if (c.conv[l] != 0 && p == c.conv[l]) return "ck" + std::to_string(j) + ".conv" + std::to_string(l);
                if (c.ssm[l] != 0 && p == c.ssm[l]) return "ck" + std::to_string(j) + ".ssm" + std::to_string(l);
            }
        }
        return "?";
    }
    std::string spanText(const S& s) const {
        std::string t;
        for (const tier::Span& sp : tb->spans())
            t += where(sp.dev, s) + " @" + std::to_string(sp.off) + " n" + std::to_string(sp.len) + " t" + std::to_string(sp.ck) + "\n";
        return t;
    }
};

std::uint64_t fnvText(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

std::vector<std::uint32_t> tbTokens(std::size_t n, std::uint32_t salt) {
    std::vector<std::uint32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint32_t>(1000 + (i * 7 + salt) % 5000);
    return v;
}

void testTierBridgeSpill() {
    // spill span sequence, three KV formats (pool arrays: f16 = K, V; q8h / q4 = K, V, K scales, V scales; the
    // extra row sizes are the S0a pin's). Snapshot hashes recorded on arch024-s6b (after the move; the move itself is
    // pinned by whirl-pins engine --tier, recorded on 906e367 before it).
    struct Fmt {
        const char* name;
        std::vector<std::uint32_t> extra;
        std::uint64_t fresh_hash, inplace_hash;
    };
    const std::vector<Fmt> fmts = {
        {"f16", {}, 0x83168f9bf3c62e8aull, 0x68a026a3b2084d92ull},
        {"q8h", {4, 6}, 0x7f632464c775016dull, 0x05b94f8ef68bce42ull},
        {"q4", {2, 10}, 0x3572179c3aa3f550ull, 0x9ebc20c1243b0759ull},
    };
    for (const Fmt& f : fmts) {
        TbRig r(f.extra);
        const std::size_t arrs = r.kv.size();
        CHECK(arrs > 0);
        CHECK_EQ(arrs % (f.extra.empty() ? 2 : 4), static_cast<std::size_t>(0));
        // P1 = 3 pages - 20 tokens, 3 checkpoints at KP - 50 / P1 / 2 KP with seq 3 / 1 / 2 (the spill writes older ones
        // first); pages 5, 6, 7
        constexpr std::uint32_t KP = qwen35::kv_page;
        const std::uint32_t P1 = 3 * KP - 20, P2 = 5 * KP - 10;
        const auto toks = tbTokens(P1, 1);
        auto s = r.slot(toks, {KP - 50, P1, 2 * KP}, {3, 1, 2}, 5);
        const std::uint64_t spills0 = r.tier->stats.spills;
        // not idle: nothing
        r.tb->spillSlot(0, false, s->sp, s->st);
        CHECK_EQ(r.tier->stats.spills, spills0);
        CHECK_EQ(s->st.tier_id, static_cast<std::uint64_t>(0));
        // idle: a new entry
        g_log.clear();
        r.tb->spillSlot(0, true, s->sp, s->st);
        CHECK_EQ(r.tier->stats.spills, spills0 + 1);
        CHECK(s->st.tier_id != 0);
        CHECK(s->st.spill_busy);
        CHECK_EQ(s->st.spill_upto, P1);
        CHECK_EQ(s->st.tier_pages, P1 / KP);
        CHECK_EQ(g_log.count("kv tier: spill " + std::to_string(P1) + " tok to RAM (new entry, pages 0..3"), static_cast<std::size_t>(1));
        const std::string t1 = r.spanText(*s);
        // structure: checkpoints in seq order (ck1, ck2, ck0), each 2 * n_gdn + 2 pieces, contiguous from j * stride;
        // then 3 pages, each one piece per pool array in kvArrays() order, contiguous from kv_off + page * page_bytes
        const tier::Layout lay = r.tier->lay;
        tier::Entry* x = r.tier->find(s->st.tier_id);
        CHECK(x != nullptr);
        const auto& sp = r.tb->spans();
        const std::size_t per_ck = 2 * r.n_gdn + 2;
        CHECK_EQ(sp.size(), 3 * per_ck + 3 * arrs);
        bool ck_ok = sp.size() == 3 * per_ck + 3 * arrs;
        const std::size_t ck_order[3] = {1, 2, 0};
        for (std::size_t k = 0; ck_ok && k < 3; ++k) {
            std::uint64_t off = ck_order[k] * lay.ckStride();
            for (std::size_t i = 0; i < per_ck; ++i) {
                const tier::Span& a = sp[k * per_ck + i];
                const std::string w = r.where(a.dev, *s);
                ck_ok = ck_ok && a.off == off && w.rfind("ck" + std::to_string(ck_order[k]) + ".", 0) == 0;
                off += a.len;
            }
            ck_ok = ck_ok && off - ck_order[k] * lay.ckStride() == lay.ck_bytes;
        }
        CHECK(ck_ok);
        bool pg_ok = x != nullptr;
        for (std::size_t pg = 0; pg_ok && pg < 3; ++pg) {
            std::uint64_t off = x->kvOff(lay) + pg * lay.page_bytes;
            for (std::size_t a = 0; a < arrs; ++a) {
                const tier::Span& e = sp[3 * per_ck + pg * arrs + a];
                pg_ok = pg_ok && e.off == off && r.where(e.dev, *s) == "kv" + std::to_string(a) + ":p" + std::to_string(5 + pg) &&
                        e.ck == 0xff;
                off += e.len;
            }
            pg_ok = pg_ok && off - (x->kvOff(lay) + pg * lay.page_bytes) == lay.page_bytes;
        }
        CHECK(pg_ok);
        // the arena holds the sources (NullBackend copies at once)
        r.nb.eventSync(s->st.spill_ev.get());
        if (x) CHECK_EQ(r.tier->verifySpans(x, sp).bad, static_cast<std::uint64_t>(0));
        // in place: the slot went on to P2 (5 pages - 10), one new checkpoint at P2 replaced the oldest (ck1, seq 1 -> 4);
        // only it and pages 2.. are written (vp = min(tier_pages 2, n_tok / KP 2))
        const auto toks2 = tbTokens(P2, 1);
        s->sp.cache_tokens = toks2;
        s->sp.pages.push_back(10);
        s->sp.pages.push_back(11);
        s->sp.ckpts[1].pos = P2;
        s->sp.ckpts[1].seq = 4;
        const std::uint64_t id1 = s->st.tier_id;
        g_log.clear();
        r.tb->spillSlot(0, true, s->sp, s->st);
        CHECK_EQ(s->st.tier_id, id1);
        CHECK_EQ(g_log.count("kv tier: spill " + std::to_string(P2) + " tok to RAM (in place, pages 2..5"), static_cast<std::size_t>(1));
        CHECK_EQ(r.tb->spans().size(), per_ck + 3 * arrs);
        const std::string t2 = r.spanText(*s);
        r.nb.eventSync(s->st.spill_ev.get());
        if (x) CHECK_EQ(r.tier->verifySpans(x, r.tb->spans()).bad, static_cast<std::uint64_t>(0));
        // below the tier's minimum: nothing
        auto small = r.slot(tbTokens(40, 2), {40}, {1}, 100);
        const std::uint64_t spills1 = r.tier->stats.spills;
        r.tb->spillSlot(1, true, small->sp, small->st);
        CHECK_EQ(r.tier->stats.spills, spills1);
        const std::uint64_t h1 = fnvText(t1), h2 = fnvText(t2);
        if (g_verbose || f.fresh_hash == 0)
            std::printf("  tier_bridge %s: fresh spans %zu fnv %016llx, in place fnv %016llx\n", f.name, 3 * per_ck + 3 * arrs,
                        static_cast<unsigned long long>(h1), static_cast<unsigned long long>(h2));
        if (g_verbose) std::printf("%s", t1.c_str());
        if (f.fresh_hash != 0) {
            CHECK_EQ(h1, f.fresh_hash);
            CHECK_EQ(h2, f.inplace_hash);
        }
        s.reset();
        small.reset();
    }
    // a shared prefix checkpoint: its checkpoint (one piece set at 0) then its pages; once
    {
        TbRig r;
        Spe e;
        r.cs->init(e.ck, false);
        e.valid = true;
        e.n = 128;
        e.kind = "system";
        e.tokens = tbTokens(128, 3);
        e.pages = {40, 41};
        r.tb->spillSpe(e);
        CHECK(e.tier_id != 0);
        CHECK(e.spill_busy);
        CHECK(static_cast<bool>(e.ready_ev));
        CHECK_EQ(r.tb->spans().size(), (2 * r.n_gdn + 2) + 2 * r.kv.size());
        CHECK_EQ(r.tb->spans().front().off, static_cast<std::uint64_t>(0));
        const std::uint64_t spills = r.tier->stats.spills;
        Spe e2;
        e2.valid = true;
        e2.n = 128;
        e2.tokens = e.tokens;
        r.tb->spillSpe(e2);  // already in the tier: no copy, the same entry
        CHECK_EQ(r.tier->stats.spills, spills);
        CHECK_EQ(e2.tier_id, e.tier_id);
        r.nb.eventSync(e.spill_ev.get());
    }
}

// a chat prompt: system, first user message, then the tail (assistant turns) as filler tokens
std::vector<std::uint32_t> tbChat(const std::string& user, std::size_t total, std::uint32_t salt) {
    const std::string head = "<|im_start|>system\nYou are a careful coding assistant.<|im_end|>\n<|im_start|>user\n" + user +
                             "<|im_end|>\n<|im_start|>assistant\n";
    const auto ids = tok().encode(head, true);
    std::vector<std::uint32_t> v(ids.begin(), ids.end());
    const auto fill = tbTokens(total > v.size() ? total - v.size() : 0, salt);
    v.insert(v.end(), fill.begin(), fill.end());
    return v;
}

void testTierBridgeSupersede() {
    // CACHE-1 through TierBridge: the old session (slot 0, 6400 tokens) and a compacted one (slot 1, same system prompt
    // and first user message, 1200 tokens). Its first spill (a new entry) marks the old entry; its second (in place)
    // cuts it to the shared prefix -- unless the old entry was used again in between.
    for (int variant = 0; variant < 4; ++variant) {
        // 0: mark -> apply cuts; 1: the old entry updated between mark and apply (not cut); 2: only the system prompt
        // shared (demoted, not cut); 3: supersede off (no mark)
        TierBridge::Config tc;
        tc.supersede = variant != 3;
        TbRig r({}, tc);
        const auto old_t = tbChat("Refactor the tier code.", 6400, 11);
        // a checkpoint inside the shared prefix (a cut keeps the entry) and two after it (their RAM blocks are what a cut
        // frees: the mock's KV pages are too small to fill a tier block)
        auto a = r.slot(old_t, {16, 3000, 6400}, {1, 2, 3}, 0);
        r.tb->spillSlot(0, true, a->sp, a->st);
        tier::Entry* eo = r.tier->find(a->st.tier_id);
        CHECK(eo != nullptr);
        if (!eo) continue;
        const std::uint32_t old_n = eo->n_tok;
        CHECK_EQ(old_n, 6400u);
        const std::string user = variant == 2 ? "Write the release notes." : "Refactor the tier code.";
        auto new_t = tbChat(user, 1200, 99);
        auto b = r.slot(new_t, {1200}, {1}, 200);
        r.nb.eventSync(a->st.spill_ev.get());
        g_log.clear();
        r.tb->spillSlot(1, true, b->sp, b->st);  // first spill: mark (before apply: the old entry is untouched)
        CHECK_EQ(r.tb->supersedePending(), static_cast<std::size_t>(variant == 3 ? 0 : 1));
        CHECK_EQ(g_log.count("looks superseded by this conversation"), static_cast<std::size_t>(variant == 3 ? 0 : 1));
        CHECK_EQ(eo->n_tok, old_n);
        CHECK_EQ(r.tier->stats.superseded, static_cast<std::uint64_t>(0));
        if (variant == 1) {
            // the old session goes on (in place): its stamp changes, the mark no longer applies
            auto more = tbChat("Refactor the tier code.", 6400 + qwen35::kv_page, 11);
            a->sp.cache_tokens = more;
            a->sp.pages.push_back(150);
            a->sp.ckpts[2].pos = 6400 + qwen35::kv_page;
            a->sp.ckpts[2].seq = 4;
            r.tb->spillSlot(0, true, a->sp, a->st);
            CHECK_EQ(a->st.tier_id, eo->id);
            CHECK_EQ(eo->n_tok, 6400u + qwen35::kv_page);
            r.nb.eventSync(a->st.spill_ev.get());
        }
        const std::uint32_t before_apply = eo->n_tok;
        // second spill of the new conversation (it went on, in place): apply
        auto nt2 = new_t;
        const auto tail = tbTokens(100, 5);
        nt2.insert(nt2.end(), tail.begin(), tail.end());
        b->sp.cache_tokens = nt2;
        b->sp.pages.push_back(230);
        b->sp.pages.push_back(231);
        b->sp.ckpts[0].pos = 1300;
        b->sp.ckpts[0].seq = 2;
        r.nb.eventSync(b->st.spill_ev.get());
        g_log.clear();
        r.tb->spillSlot(1, true, b->sp, b->st);
        CHECK_EQ(g_log.count("kv tier: spill 1300 tok to RAM (in place"), static_cast<std::size_t>(1));
        CHECK_EQ(r.tb->supersedePending(), static_cast<std::size_t>(0));
        tier::Entry* e2 = r.tier->find(eo->id);
        if (variant == 0) {
            // cut to the checkpoint inside the shared prefix (the "cut to" line needs freed RAM blocks; the mock's
            // entries fit in the same blocks before and after, so the counter is the check here)
            CHECK_EQ(r.tier->stats.superseded, static_cast<std::uint64_t>(1));
            CHECK(e2 != nullptr && e2->n_tok < old_n);
            CHECK(e2 != nullptr && e2->n_tok <= cache::lcpLen(old_t, new_t));
            CHECK(e2 != nullptr && e2->n_tok == 16u);
        } else if (variant == 2) {
            CHECK_EQ(g_log.count("kept, evicted first under RAM / SSD pressure"), static_cast<std::size_t>(1));
            CHECK(e2 != nullptr && e2->n_tok == old_n);
            CHECK_EQ(r.tier->stats.demoted, static_cast<std::uint64_t>(1));
        } else {
            CHECK_EQ(g_log.count("superseded (compact / compress)"), static_cast<std::size_t>(0));
            CHECK(e2 != nullptr && e2->n_tok == before_apply);
        }
        r.nb.eventSync(b->st.spill_ev.get());
        a.reset();
        b.reset();
    }
}

void testTierBridgeFlush() {
    // the shutdown flush is bounded: a tick that stays busy returns "pending" once the bound passed (not after the
    // tick gives up at ~2000 calls); during it the SSD delay is 0 and the tier is flushing, after it both are back
    for (const bool shutdown_mode : {false, true}) {
        TierBridge::Config tc;
        tc.flush_ms = 150;
        tc.flush_long_ms = 450;
        TbRig r({}, tc);
        r.tier->cfg.ssd_delay_ms = 1234;
        r.tier->cfg.ssd_shutdown_only = shutdown_mode;
        int calls = 0;
        bool seen_flush = true;
        const double t0 = nowSeconds();
        const bool pending = r.tb->flushOnStop([&] {
            ++calls;
            seen_flush = seen_flush && r.tier->flushing && r.tier->cfg.ssd_delay_ms == 0;
            return calls < 2000;
        });
        const double ms = (nowSeconds() - t0) * 1000.0;
        const double bound = shutdown_mode ? 450.0 : 150.0;
        CHECK(pending);
        CHECK(seen_flush);
        CHECK(ms >= bound);
        CHECK(ms < bound + 2000.0);
        CHECK(calls < 2000);
        CHECK_EQ(r.tier->cfg.ssd_delay_ms, static_cast<std::uint64_t>(1234));
        CHECK(!r.tier->flushing);
    }
    {
        // nothing pending: one tick, returns at once
        TbRig r;
        int calls = 0;
        const bool pending = r.tb->flushOnStop([&] {
            ++calls;
            return false;
        });
        CHECK(!pending);
        CHECK_EQ(calls, 1);
    }
    {
        // no tier: no tick
        TbRig r;
        r.tb->attach(nullptr);
        int calls = 0;
        CHECK(!r.tb->flushOnStop([&] { return ++calls > 0; }));
        CHECK_EQ(calls, 0);
        r.tb->attach(r.tier.get());
    }
}

// ---------------------------------------------------------------------------
// fix-k2 (review s15 K2, mt-p3 probe): dropContained's transitive drop. RAM entries of one history, each a prefix of
// x (the new entry), restore points of kind 1 (prompt-end; balance config, no COLD-RESUME filter). Bug: in one
// dropContained(x) pass o vouched for e (o holds a point in [deep_e, e.n_tok]) and was then dropped itself (x covers
// deep_o), so e's restore range was lost -- depending on the t->entries order. Invariant checked here for EVERY
// permutation of the creation order (t->entries order): x stays; every original entry's deepest restore point range
// [deep, n_tok] still holds a point in some remaining entry (no restore depth lost); the kept set is the expected
// minimal one (by role, the same in every order: order-independent, no extra RAM).
namespace {
struct K2Spec {
    char role;
    std::uint32_t n;
    std::vector<std::uint32_t> pts;
};

// runs dropContained on one creation order; returns the roles still present (sorted) and checks the invariant
std::string k2Run(const std::vector<K2Spec>& specs, const std::vector<std::size_t>& order, std::uint32_t x_n,
                  const std::vector<std::uint32_t>& x_pts, std::string& lost) {
    TbRig r;
    const std::vector<std::uint32_t> xt = tbTokens(x_n, 11);
    auto mk = [&](std::uint32_t n, const std::vector<std::uint32_t>& pts) {
        tier::Entry* en = r.tier->newEntry(static_cast<std::uint32_t>(pts.size()));
        en->tokens.assign(xt.begin(), xt.begin() + n);
        en->n_tok = n;
        en->pages = static_cast<std::uint32_t>((n + qwen35::kv_page - 1) / qwen35::kv_page);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            en->ck[j].pos = pts[j];
            en->ck[j].valid = 1;
            en->ck[j].kind = 1;
            en->ck[j].seq = static_cast<std::uint32_t>(j + 1);
        }
        return en;
    };
    std::vector<std::pair<char, std::uint64_t>> ids;
    tier::Entry* x = nullptr;
    for (std::size_t i : order) {
        if (i == specs.size()) {
            x = mk(x_n, x_pts);
            ids.push_back({'x', x->id});
        } else {
            ids.push_back({specs[i].role, mk(specs[i].n, specs[i].pts)->id});
        }
    }
    CHECK(r.tier->entries.size() == specs.size() + 1);
    r.tb->dropContained(0, x);
    std::string kept;
    for (const auto& [role, id] : ids)
        if (r.tier->find(id)) kept.push_back(role);
    std::sort(kept.begin(), kept.end());
    lost.clear();
    for (const K2Spec& s : specs) {
        const std::uint32_t deep = *std::max_element(s.pts.begin(), s.pts.end());
        bool held = false;
        for (const tier::Entry* en : r.tier->entries)
            for (std::uint32_t j = 0; j < tier::max_ck && !held; ++j) {
                const tier::CkMeta& c = en->ck[j];
                held = c.valid != 0 && c.pos >= deep && c.pos <= s.n && c.pos <= en->n_tok;  // all entries share xt's prefix
            }
        if (!held) lost += std::string(1, s.role) + "[" + std::to_string(deep) + "," + std::to_string(s.n) + "] ";
    }
    return kept;
}

void k2AllOrders(const char* name, const std::vector<K2Spec>& specs, std::uint32_t x_n, const std::vector<std::uint32_t>& x_pts,
                 const std::string& want_kept) {
    std::vector<std::size_t> order(specs.size() + 1);
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::size_t n_orders = 0, n_bad = 0;
    do {
        std::string lost;
        const std::string kept = k2Run(specs, order, x_n, x_pts, lost);
        ++n_orders;
        const bool ok = kept == want_kept && lost.empty();
        if (!ok) {
            ++n_bad;
            std::string os;
            for (std::size_t i : order) os.push_back(i == specs.size() ? 'x' : specs[i].role);
            std::fprintf(stderr, "  (tier_drop_k2 %s order %s: kept %s (want %s), restore range lost: %s)\n", name, os.c_str(),
                         kept.c_str(), want_kept.c_str(), lost.empty() ? "none" : lost.c_str());
        }
        CHECK_EQ(kept, want_kept);
        CHECK(lost.empty());
    } while (std::next_permutation(order.begin(), order.end()));
    std::fprintf(stderr, "  (tier_drop_k2 %s: %zu orders, %zu wrong)\n", name, n_orders, n_bad);
}
}  // namespace

void testTierDropK2() {
    // A (the review's case): e (1000, @900) ⊂ o (2000, @950 @2000) ⊂ x (3000, @2000 @3000). x covers o; only e (or o)
    // holds a point in [900,1000]; keeping e (1000 tok) instead of o (2000) is the smaller set.
    k2AllOrders("A", {{'e', 1000, {900}}, {'o', 2000, {950, 2000}}}, 3000, {2000, 3000}, "ex");
    // B (chain of two vouchers): e (1000, @900), o (2000, @950 @1500), q (2500, @1800 @2500), x (3000, @3000).
    // q is not covered (kept, demoted); q covers o (@1800 in [1500,2000]) -> o goes; then nothing remaining holds
    // a point in [900,1000] but e itself -> e stays.
    k2AllOrders("B", {{'e', 1000, {900}}, {'o', 2000, {950, 1500}}, {'q', 2500, {1800, 2500}}}, 3000, {3000}, "eqx");
    // C (shorter voucher, no extra RAM): e (1000, @900), s (950, @950), x (3000, @3000). s is kept (nothing holds 950);
    // s covers e (@950 in [900,1000]) -> e goes.
    k2AllOrders("C", {{'e', 1000, {900}}, {'s', 950, {950}}}, 3000, {3000}, "sx");
    // D (all covered by x directly): e (1000, @900), o (2000, @1500), x (3000, @1000 @2000 @3000) -> both go.
    k2AllOrders("D", {{'e', 1000, {900}}, {'o', 2000, {1500}}}, 3000, {1000, 2000, 3000}, "x");
}

// ---------------------------------------------------------------------------
// arch024-S7: TierBridge restore half driven directly (TbRig: MockModel + NullBackend + a RAM-only tier::Tier).
// The engine-level versions (restore_conc, restore_throws, tier_inplace) stay; these call TierBridge itself:
// restoreMatch / startRestore / pumpRestore / restoreDone / restoreFailed / finishTail / cutTail / queuedReuse.

namespace {

// fill a slot's KV pages with a pattern (per array, page and byte), so a restore that reads the wrong place shows
void tbFillPages(TbRig& r, const std::vector<std::int32_t>& pages, std::uint8_t salt) {
    for (std::size_t a = 0; a < r.kv.size(); ++a) {
        const std::uint64_t len = static_cast<std::uint64_t>(qwen35::kv_page) * r.kv[a].row;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            auto* p = reinterpret_cast<std::uint8_t*>(r.kv[a].base + static_cast<std::uint64_t>(pages[i]) * len);
            for (std::uint64_t b = 0; b < len; ++b) p[b] = static_cast<std::uint8_t>(salt + a * 31 + i * 7 + b);
        }
    }
}

// pages src[i] and dst[i] hold the same bytes in every pool array (the first n of them)
bool tbSamePages(TbRig& r, const std::vector<std::int32_t>& src, const std::vector<std::int32_t>& dst, std::size_t n) {
    for (std::size_t a = 0; a < r.kv.size(); ++a) {
        const std::uint64_t len = static_cast<std::uint64_t>(qwen35::kv_page) * r.kv[a].row;
        for (std::size_t i = 0; i < n; ++i) {
            const auto* p = reinterpret_cast<const void*>(r.kv[a].base + static_cast<std::uint64_t>(src[i]) * len);
            const auto* q = reinterpret_cast<const void*>(r.kv[a].base + static_cast<std::uint64_t>(dst[i]) * len);
            if (std::memcmp(p, q, len) != 0) return false;
        }
    }
    return true;
}

bool tbSameCk(TbRig& r, const Ckpt& a, const Ckpt& b) {
    const std::size_t n = static_cast<std::size_t>(r.cs->bytes(true));
    return std::memcmp(reinterpret_cast<const void*>(a.dev.get()), reinterpret_cast<const void*>(b.dev.get()), n) == 0;
}

// the steps the engine passes, recorded in call order
struct TbSteps {
    std::vector<std::string> calls;
    TierBridge::RestoreSteps steps(SlotPrefix& sp) {
        TierBridge::RestoreSteps s;
        s.drop_cache = [this, &sp] {
            calls.push_back("drop");
            for (Ckpt& c : sp.ckpts) c.valid = false;
            sp.cache_tokens.clear();
        };
        s.drafter_reset = [this](std::uint32_t n) { calls.push_back("dfl" + std::to_string(n)); };
        s.share = [this](Ckpt&, std::uint64_t) { calls.push_back("share"); };
        return s;
    }
};

}  // namespace

void testTierBridgeRestore() {
    constexpr std::uint32_t KP = qwen35::kv_page;
    // a session of N tokens (19 pages - 16), checkpoints at 600 / 1100 / N (seq 1 / 2 / 3), spilled from pages 5..
    const std::uint32_t N = 19 * KP - 16;
    for (const std::vector<std::uint32_t>& extra : {std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{4, 6}}) {
        TierBridge::Config tc;
        tc.verify = true;
        TbRig r(extra, tc);
        const auto toks = tbTokens(N, 5);
        auto src = r.slot(toks, {600, 1100, N}, {1, 2, 3}, 5);
        tbFillPages(r, src->sp.pages, 0x21);
        r.tb->spillSlot(0, true, src->sp, src->st);
        r.nb.eventSync(src->st.spill_ev.get());
        const std::uint64_t eid = src->st.tier_id;
        CHECK(eid != 0);
        tier::Entry* x = r.tier->find(eid);
        CHECK(x != nullptr);
        if (!x) return;
        // the next turn: the session plus 300 tokens
        std::vector<std::uint32_t> next = toks;
        const auto more = tbTokens(300, 9);
        next.insert(next.end(), more.begin(), more.end());

        // (1) restoreMatch: an empty slot gets the entry (reuse N: the last checkpoint); a slot already holding the
        // session does not (no gain); a short prompt (gain under min_gain) does not
        auto dst = r.slot({}, {0, 0, 0}, {0, 0, 0}, 0);
        dst->sp.pages.clear();
        dst->sp.ckpt_seq = 0;
        const auto mt = r.tb->restoreMatch(dst->sp, next);
        CHECK(mt.has_value());
        if (!mt) return;
        CHECK_EQ(mt->e, x);
        CHECK_EQ(mt->reuse, N);
        CHECK(!r.tb->restoreMatch(src->sp, next).has_value());
        CHECK(!r.tb->restoreMatch(dst->sp, tbTokens(400, 5)).has_value());

        // (2) startRestore into pages 100.. (the scheduler gave them), main part = matched checkpoint + pages
        for (std::uint32_t p = 0; p < x->pages; ++p) dst->sp.pages.push_back(100 + static_cast<std::int32_t>(p));
        g_log.clear();
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(r.tb->startRestore(7, 1, next, dst->sp, dst->st, *mt, t0));
        CHECK(dst->st.restore != nullptr);
        CHECK(!dst->st.rs_shared);
        CHECK_EQ(x->owner, 1u);
        CHECK_EQ(g_log.count("req 7 | slot 1 | kv tier: restoring " + std::to_string(N) + " tok from RAM (checkpoint 'prompt-end' @" +
                             std::to_string(N)),
                 static_cast<std::size_t>(1));
        // matched checkpoint (entry k of pos N) -> slot checkpoint 0, the others follow by position as the tail
        std::size_t k_main = mt->ck, n_tail = 0;
        for (std::size_t k = 0; k < x->nck; ++k) n_tail += dst->st.rs_tail[k] ? 1 : 0;
        CHECK_EQ(n_tail, static_cast<std::size_t>(2));
        CHECK(!dst->st.rs_tail[k_main]);
        CHECK(dst->st.rs_map[k_main] >= 0);
        // span order: checkpoint pieces, then one piece per pool array per page, then the tail checkpoints
        const std::size_t per_ck = 2 * r.n_gdn + 2;
        CHECK_EQ(r.tb->spans().size(), 3 * per_ck + x->pages * r.kv.size());
        // queuedReuse: a queued request on the same session sees the restore's prefix; a slot not restoring: 0
        CHECK_EQ(r.tb->queuedReuse(dst->st, next), N);
        CHECK_EQ(r.tb->queuedReuse(src->st, next), 0u);
        CHECK_EQ(r.tb->queuedReuse(dst->st, tbTokens(N, 77)), 0u);

        // (3) pump until restored, then restoreDone: tokens, checkpoints (main valid, tail not yet), tier entry
        TierBridge::Pump p = TierBridge::Pump::pending;
        for (int i = 0; i < 1000 && p == TierBridge::Pump::pending; ++i) p = r.tb->pumpRestore(dst->st);
        CHECK(p == TierBridge::Pump::restored);
        TbSteps ts;
        g_log.clear();
        r.tb->restoreDone(7, 1, dst->sp, dst->st, ts.steps(dst->sp));
        CHECK_EQ(ts.calls.size(), static_cast<std::size_t>(1));
        if (!ts.calls.empty()) CHECK_EQ(ts.calls[0], "dfl" + std::to_string(N));
        CHECK(dst->sp.cache_tokens == toks);
        CHECK_EQ(dst->st.tier_id, eid);
        CHECK_EQ(dst->st.tier_pages, N / KP);
        CHECK_EQ(dst->sp.ckpt_seq, static_cast<std::uint64_t>(3));
        const Ckpt& m0 = dst->sp.ckpts[static_cast<std::size_t>(dst->st.rs_map[k_main])];
        CHECK(m0.valid);
        CHECK_EQ(m0.pos, N);
        CHECK(tbSameCk(r, m0, src->sp.ckpts[2]));
        CHECK(tbSamePages(r, src->sp.pages, dst->sp.pages, x->pages));
        CHECK_EQ(g_log.count("kv tier verify: restore of " + std::to_string(N) + " tok byte-identical"), static_cast<std::size_t>(1));
        CHECK_EQ(g_log.count("kv tier: restored " + std::to_string(N) + " tok from RAM"), static_cast<std::size_t>(1));
        CHECK_EQ(r.tier->stats.restores_ram, static_cast<std::uint64_t>(1));
        std::size_t valid = 0;
        for (const Ckpt& c : dst->sp.ckpts) valid += c.valid ? 1 : 0;
        CHECK_EQ(valid, static_cast<std::size_t>(1));

        // (4) finishTail: the tail checkpoints become valid (pos <= reuse) with their bytes, the restore ends
        dst->sp.reuse = N;
        g_log.clear();
        r.tb->finishTail(1, dst->sp, dst->st, true);
        CHECK(dst->st.restore == nullptr);
        CHECK_EQ(x->pin, 0u);
        valid = 0;
        for (const Ckpt& c : dst->sp.ckpts) valid += c.valid ? 1 : 0;
        CHECK_EQ(valid, static_cast<std::size_t>(3));
        for (std::size_t k = 0; k < x->nck; ++k) {
            const std::int32_t j = dst->st.rs_map[k];
            if (j < 0) continue;
            const Ckpt& c = dst->sp.ckpts[static_cast<std::size_t>(j)];
            // source checkpoint of the same position
            for (const Ckpt& s : src->sp.ckpts)
                if (s.pos == c.pos) CHECK(tbSameCk(r, c, s));
        }
        CHECK_EQ(g_log.count("kv tier verify: restore tail byte-identical (" ), static_cast<std::size_t>(1));
        CHECK(std::none_of(dst->st.rs_tail.begin(), dst->st.rs_tail.end(), [](bool b) { return b; }));

        // (5) in-place regeneration: the restored slot goes on 200 tokens, its next spill updates the same entry in place
        dst->sp.cache_tokens.insert(dst->sp.cache_tokens.end(), more.begin(), more.begin() + 200);
        dst->sp.pages.push_back(200);
        dst->sp.pages.push_back(201);
        dst->sp.ckpts[0].pos = N + 200;
        dst->sp.ckpts[0].seq = 4;
        g_log.clear();
        r.tb->spillSlot(1, true, dst->sp, dst->st);
        CHECK_EQ(dst->st.tier_id, eid);
        CHECK_EQ(g_log.count("kv tier: spill " + std::to_string(N + 200) + " tok to RAM (in place"), static_cast<std::size_t>(1));
        r.nb.eventSync(dst->st.spill_ev.get());
        CHECK_EQ(r.tier->verifySpans(x, r.tb->spans()).bad, static_cast<std::uint64_t>(0));

        // (6) concurrent: two slots restore the same entry; the second starts while the first's tail is still open
        auto a = r.slot({}, {0, 0}, {0, 0}, 0);
        auto b = r.slot({}, {0, 0}, {0, 0}, 0);
        a->sp.pages.clear();
        b->sp.pages.clear();
        const auto mt2 = r.tb->restoreMatch(a->sp, next);
        CHECK(mt2.has_value());
        if (mt2) {
            for (std::uint32_t q = 0; q < mt2->e->pages; ++q) {
                a->sp.pages.push_back(300 + static_cast<std::int32_t>(q));
                b->sp.pages.push_back(360 + static_cast<std::int32_t>(q));
            }
            CHECK(r.tb->startRestore(8, 2, next, a->sp, a->st, *mt2, t0));
            const auto mt3 = r.tb->restoreMatch(b->sp, next);
            CHECK(mt3.has_value());
            if (mt3) CHECK(r.tb->startRestore(9, 3, next, b->sp, b->st, *mt3, t0));
            CHECK_EQ(r.tb->queuedReuse(a->st, next), mt2->reuse);
            TbSteps sa, sb;
            for (auto* s : {a.get(), b.get()}) {
                if (!s->st.restore) continue;
                TierBridge::Pump q = TierBridge::Pump::pending;
                for (int i = 0; i < 1000 && q == TierBridge::Pump::pending; ++i) q = r.tb->pumpRestore(s->st);
                CHECK(q == TierBridge::Pump::restored);
                r.tb->restoreDone(9, 3, s->sp, s->st, (s == a.get() ? sa : sb).steps(s->sp));
                s->sp.reuse = mt2->reuse;
                r.tb->cutTail(3, s->sp, s->st);  // saveCkpt: the tail is cut (what is in counts)
                r.tb->finishTail(3, s->sp, s->st, true);
                CHECK(s->st.restore == nullptr);
            }
            CHECK(tbSamePages(r, a->sp.pages, b->sp.pages, mt2->e->pages));
            CHECK(a->sp.cache_tokens == b->sp.cache_tokens);
            CHECK_EQ(x->pin, 0u);
            CHECK_EQ(r.tier->stats.restores_ram, static_cast<std::uint64_t>(3));
        }

        // (7) a restore copy throws (F1 path): the restore fails cleanly (fence, drop_cache, end), the entry is not
        // left pinned, and the same entry restores afterwards
        auto f = r.slot({}, {0, 0}, {0, 0}, 0);
        f->sp.pages.clear();
        f->sp.cache_tokens = tbTokens(100, 3);
        const auto mt4 = r.tb->restoreMatch(f->sp, next);
        CHECK(mt4.has_value());
        if (mt4) {
            for (std::uint32_t q = 0; q < mt4->e->pages; ++q) f->sp.pages.push_back(420 + static_cast<std::int32_t>(q));
            r.nb.failHostToDevCopy(2);
            bool threw = false, started = false;
            try {
                started = r.tb->startRestore(10, 4, next, f->sp, f->st, *mt4, t0);
            } catch (const std::exception&) {
                threw = true;  // Engine::tryRestore logs "restore failed to start" and prefills cold
            }
            TbSteps sf;
            if (started) {
                TierBridge::Pump q = TierBridge::Pump::pending;
                for (int i = 0; i < 1000 && q == TierBridge::Pump::pending; ++i) q = r.tb->pumpRestore(f->st);
                CHECK(q == TierBridge::Pump::failed);
                g_log.clear();
                r.tb->restoreFailed(10, 4, f->st, sf.steps(f->sp));
                CHECK_EQ(g_log.count("req 10 | slot 4 | kv tier: restore failed; prefilling from scratch"), static_cast<std::size_t>(1));
                CHECK_EQ(sf.calls.size(), static_cast<std::size_t>(1));
                if (!sf.calls.empty()) CHECK_EQ(sf.calls[0], std::string("drop"));
                CHECK(f->sp.cache_tokens.empty());
                CHECK(f->st.restore == nullptr);
                CHECK(std::none_of(f->st.rs_tail.begin(), f->st.rs_tail.end(), [](bool v) { return v; }));
            }
            CHECK(threw || started);
            CHECK(!r.nb.hostCopyFailArmed());
            r.nb.failHostToDevCopy(0);
            CHECK_EQ(x->pin, 0u);
            CHECK_EQ(r.tier->stats.restores_ram, static_cast<std::uint64_t>(3));
            // again, no failure: restores
            auto g = r.slot({}, {0, 0}, {0, 0}, 0);
            g->sp.pages.clear();
            const auto mt5 = r.tb->restoreMatch(g->sp, next);
            CHECK(mt5.has_value());
            if (mt5) {
                for (std::uint32_t q = 0; q < mt5->e->pages; ++q) g->sp.pages.push_back(450 + static_cast<std::int32_t>(q));
                CHECK(r.tb->startRestore(11, 5, next, g->sp, g->st, *mt5, t0));
                TierBridge::Pump q = TierBridge::Pump::pending;
                for (int i = 0; i < 1000 && q == TierBridge::Pump::pending; ++i) q = r.tb->pumpRestore(g->st);
                CHECK(q == TierBridge::Pump::restored);
                TbSteps sg;
                r.tb->restoreDone(11, 5, g->sp, g->st, sg.steps(g->sp));
                r.tb->finishTail(5, g->sp, g->st, true);
                CHECK(tbSamePages(r, a->sp.pages, g->sp.pages, mt5->e->pages));
                CHECK_EQ(x->pin, 0u);
            }
        }
        src.reset();
    }
}

}  // namespace

// arch024 S10 (M8): thread ownership. FakeLoop is a thread that owns an EngineThread the way runLoop does (enter at the
// start, leave at the end) and runs the calls given to it.
int g_owner_expected = 0;  // violations a test causes on purpose (main compares each test's count with it)

struct FakeLoop {
    EngineThread& o;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool quit = false;
    std::exception_ptr err;
    explicit FakeLoop(EngineThread& t) : o(t) {
        std::promise<void> up;
        auto f = up.get_future();
        th = std::thread([this, &up] {
            o.enter();
            up.set_value();
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                cv.wait(lk, [&] { return quit || job; });
                if (job) {
                    try {
                        job();
                    } catch (...) {
                        err = std::current_exception();
                    }
                    job = nullptr;
                    cv.notify_all();
                    continue;
                }
                break;
            }
            o.leave();
        });
        f.wait();
    }
    // fn on the owner thread; rethrows what it threw
    void run(std::function<void()> fn) {
        std::unique_lock<std::mutex> lk(mu);
        job = std::move(fn);
        err = nullptr;
        cv.notify_all();
        cv.wait(lk, [&] { return !job; });
        if (err) std::rethrow_exception(err);
    }
    void stop() {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        th.join();
    }
    ~FakeLoop() { stop(); }
};

// fn from a thread that is not the owner: must throw std::logic_error (throwing mode) and count one violation
bool blocked(const std::function<void()>& fn) {
    const std::uint64_t v0 = EngineThread::violations();
    bool threw = false;
    try {
        fn();
    } catch (const std::logic_error& e) {
        threw = std::string_view(e.what()).starts_with("owner check: ");
    }
    g_owner_expected += 1;
    return threw && EngineThread::violations() == v0 + 1;
}

// fn without a violation (the owner thread, or no owner)
bool allowed(const std::function<void()>& fn) {
    const std::uint64_t v0 = EngineThread::violations();
    try {
        fn();
    } catch (const std::logic_error&) {
        return false;
    }
    return EngineThread::violations() == v0;
}

void testOwnerCheck() {
    CHECK(EngineThread::throwing());  // main: test programs run in throwing mode (PLAN M8 "test build")
    // ---- PagePool (M3): startup (no owner) allowed; a call from another thread while the owner runs is blocked
    //      before anything changes; the same calls on the owner thread pass
    {
        EngineThread own;  // before the rig: the modules keep a pointer to it until the rig is gone
        PpRig r(8);
        r.pool->setOwner(&own);
        CHECK(!own.running());
        CHECK(allowed([&] { r.pool->init(r.model->poolPages(), r.arrs); }));  // startup (initPool before runLoop)
        CHECK(allowed([&] { (void)r.pool->takeFree(); }));
        {
            FakeLoop L(own);
            CHECK(own.running());
            const std::size_t free0 = r.pool->freeCount();
            std::vector<std::int32_t> pages;
            struct NoEvict final : PagePool::Evictor {
                bool evictOne() override { return false; }
            } ev;
            const std::int32_t one[] = {0};
            CHECK(blocked([&] { (void)r.pool->takeFree(); }));
            CHECK(blocked([&] { (void)r.pool->take(ev); }));
            CHECK(blocked([&] { r.pool->ref(0); }));
            CHECK(blocked([&] { r.pool->unref(one); }));
            CHECK(blocked([&] { (void)r.pool->ensure(pages, 0, 100, ev); }));
            CHECK(blocked([&] { (void)r.pool->cow(pages, 0, 1, false, ev); }));
            CHECK(blocked([&] { r.pool->copy(1, 0); }));
            CHECK(blocked([&] { r.pool->quarantine(one); }));
            CHECK(blocked([&] { r.pool->reclaim(true); }));
            CHECK(blocked([&] { r.pool->setQuarantineStream(r.ts); }));
            CHECK(blocked([&] { r.pool->init(4, r.arrs); }));
            CHECK_EQ(r.pool->freeCount(), free0);  // nothing changed
            CHECK_EQ(r.pool->refCount(0), 1u);
            // the owner thread (runOnLoop / a Job): no violation
            CHECK(allowed([&] {
                L.run([&] {
                    auto p = r.pool->takeFree();
                    r.pool->ref(*p);
                    const std::int32_t pp[] = {*p, *p};
                    r.pool->unref(pp);
                    (void)r.pool->ensure(pages, 0, 100, ev);
                    r.pool->unref(pages);
                    r.pool->reclaim(false);
                });
            }));
            CHECK_EQ(r.pool->freeCount(), free0);
        }
        CHECK(!own.running());  // the loop returned: allowed again
        CHECK(allowed([&] { (void)r.pool->takeFree(); }));
    }
    // ---- CkptStore (M2), PrefixStore (M4), TierBridge (M5)
    {
        EngineThread own;
        TbRig r;
        r.cs->setOwner(&own);
        r.ps->setOwner(&own);
        r.tb->setOwner(&own);
        Ckpt c;
        CHECK(allowed([&] { r.cs->init(c, false); }));  // startup (allocState)
        const std::vector<std::uint32_t> toks(200, 7u);
        {
            FakeLoop L(own);
            SlotTier st;
            CHECK(blocked([&] { r.cs->release(c, {}); }));
            CHECK(blocked([&] { r.cs->save(c, std::nullopt, std::nullopt); }));
            CHECK(blocked([&] { r.cs->load(c); }));
            CHECK(blocked([&] { Ckpt d; r.cs->init(d, false); }));
            CHECK(blocked([&] { r.ps->waitSpeFences(); }));
            CHECK(blocked([&] { (void)r.ps->speMatch(toks); }));
            CHECK(blocked([&] { r.ps->setVision(PrefixStore::Vision{}); }));
            CHECK(blocked([&] { r.ps->attachTier(r.tier.get()); }));
            CHECK(blocked([&] { (void)r.tb->tick(); }));
            CHECK(blocked([&] { r.tb->waitSpill(st); }));
            CHECK(blocked([&] { r.tb->idleActivity(); }));
            CHECK(blocked([&] { r.tb->countSpillWait(); }));
            CHECK(blocked([&] { r.tb->attach(r.tier.get()); }));
            CHECK(blocked([&] { r.tb->pageSpans(0, 0); }));
            CHECK(c.dev);  // release was blocked before it freed anything
            CHECK(allowed([&] {
                L.run([&] {
                    (void)r.ps->speMatch(toks);
                    r.ps->waitSpeFences();
                    (void)r.tb->tick();
                    r.tb->waitSpill(st);
                    r.tb->idleActivity();
                    r.cs->release(c, {});
                });
            }));
            CHECK(!c.dev);
        }
    }
    // ---- release mode (the product): no exception, the call goes on, one "E owner check:" line per call site, every
    //      violation counted
    {
        EngineThread own;
        PpRig r(8);
        r.pool->setOwner(&own);
        FakeLoop L(own);
        EngineThread::setThrowing(false);
        EngineThread::resetLoggedForTests();
        g_log.clear();
        const std::uint64_t v0 = EngineThread::violations();
        std::optional<std::int32_t> a, b;
        bool threw = false;
        try {
            a = r.pool->takeFree();
            b = r.pool->takeFree();
        } catch (...) {
            threw = true;
        }
        EngineThread::setThrowing(true);
        g_owner_expected += 2;
        CHECK(!threw);
        CHECK(a && b && *a != *b);  // both calls did their work, as before S10
        CHECK_EQ(EngineThread::violations(), v0 + 2);
        std::ostringstream id;
        id << std::this_thread::get_id();
        CHECK_EQ(g_log.count("E owner check: PagePool::takeFree called on thread " + id.str()), 1u);
        CHECK_EQ(g_log.count("owner check:"), 1u);
    }
    // ---- Engine: runLoop owns, runOnLoop runs on it (no throw), the governor glue (M7) is checked, startup allowed
    {
        ServerSetup su = baseSetup();
        TestServer srv(su);
        Engine& e = srv.engine();
        std::thread::id inside;
        bool ran = false;
        CHECK(allowed([&] {
            e.runOnLoop([&] {
                inside = std::this_thread::get_id();
                e.owner().check("test::runOnLoop");
                e.setVramProbe(nullptr, false);  // M7 entry point on the owner thread
                ran = true;
            });
        }));
        CHECK(ran);
        CHECK(e.owner().running());
        CHECK(inside == e.owner().owner());
        CHECK(inside != std::this_thread::get_id());
        CHECK(blocked([&] { e.setVramProbe(nullptr, false); }));
        CHECK(blocked([&] { e.setGovernor(whirl::memgov::Config{}, nullptr, 0); }));
        // the engine's modules are wired to its owner: an Engine entry point that goes straight to them is blocked there
        CHECK(e.modulesOwned());
        CHECK(blocked([&] { e.attachTier(nullptr); }));  // TierBridge::attach, before anything is detached
        // an HTTP request after the blocked calls still works (nothing was left half done)
        const auto res = test::httpRequest("127.0.0.1", srv.port(), "POST", "/v1/completions", completionBody(makePrompt(8, 40), 4), 30);
        CHECK_EQ(res.status, 200);
        srv.stopEngine();
        CHECK(!e.owner().running());
        CHECK(allowed([&] { e.setVramProbe(nullptr, false); }));  // no loop: allowed
    }
}

// ---------------------------------------------------------------------------
// arch024-S8: the HTTP front end against a fake EngineApi (M6). No Engine, no model: the fake answers each job
// through the real response writer (StreamState over a ResponseCtx), so the endpoints, /props, /health, the error
// paths and stream == non-stream are checked on http.cpp + stream.cpp alone.

class FakeApi final : public EngineApi {
public:
    FakeApi() : ids_(7), rcx_{"fake-model", tok(), ids_} {
        f_.model_name = "fake-model";
        f_.model_file = "C:\\models\\fake-file.gguf";
        f_.tmpl = chat::TemplateKind::a;
        f_.tok = &tok();
        f_.use_mtp = true;
        f_.numerics_json = "{\"mode\":\"fake\"}";
        f_.startup_json = "{\"slots\":3}";
        f_.kv_ssd_mode = "always";
        f_.arch = "qwen35";
        f_.n_vocab = static_cast<std::uint32_t>(tok().size());
        f_.n_embd = 64;
        f_.n_layer = 4;
    }
    std::uint64_t nextJobId() override { return next_id_.fetch_add(1); }
    void submitAndWait(Job& job) override {
        building_.fetch_sub(1);  // as Engine::submitAndWait: the queued job is no longer being built
        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_.push_back({job.kind, job.tokens, job.params.stream, job.think, job.endpoint});
        }
        // the reply: one token per push, as the decode cycle does; a thinking chat starts inside <think>
        const std::string reply = job.think ? "Short reasoning here.</think>\n\nThe answer is 42, said the fake engine."
                                            : "The answer is 42, said the fake engine.";
        const auto ids = tok().encode(reply, true);
        Out out;
        out.chat = job.kind == JobKind::chat;
        out.think_open = job.kind == JobKind::chat && job.think;
        out.stops = job.params.stop;
        StreamState st;
        st.job = &job;
        st.cx = &rcx_;
        st.begin();
        std::uint32_t n = 0;
        bool stopped = false, cut = false;
        for (TokenId t : ids) {
            if (job.params.max_tokens && n >= *job.params.max_tokens) {
                cut = true;
                break;
            }
            const std::uint32_t u = static_cast<std::uint32_t>(t);
            ++n;
            if (st.push(out, std::span<const std::uint32_t>(&u, 1))) {
                stopped = true;
                break;
            }
        }
        const Final f = st.finish(out, cut && !stopped ? "length" : "stop");
        Timings tim;
        tim.prompt_n = static_cast<std::uint32_t>(job.tokens.size());
        tim.predicted_n = n;
        st.end(out, f, tim, static_cast<std::uint32_t>(job.tokens.size()));
    }
    void beginBuilding() override { building_.fetch_add(1); }
    void endBuilding() override { building_.fetch_sub(1); }
    Health health() override {
        Health h;
        h.active = 1;
        h.queued = 2;
        h.slots = 3;
        h.vram_ok = true;
        h.vram_local = 5ull << 20;
        h.vram_budget = 7ull << 20;
        return h;
    }
    Stats statsShared() const override {
        Stats s;
        s.ssd_written = 11;
        s.ssd_entries = 2;
        return s;
    }
    std::string governorJson() override { return "{\"fake_governor\":true}"; }
    std::uint32_t ctxPublished() const override { return ctx_.load(); }
    EngineFacts facts() const override {
        ++facts_calls_;
        return f_;
    }
    struct Seen {
        JobKind kind;
        std::vector<std::uint32_t> tokens;
        bool stream, think;
        std::string endpoint;
    };
    std::vector<Seen> jobs() {
        std::lock_guard<std::mutex> lk(mu_);
        return jobs_;
    }
    EngineFacts f_;
    std::atomic<std::uint32_t> ctx_{4096};
    std::atomic<int> building_{0};
    mutable std::atomic<int> facts_calls_{0};

private:
    IdGen ids_;
    ResponseCtx rcx_;
    std::atomic<std::uint64_t> next_id_{1};
    std::mutex mu_;
    std::vector<Seen> jobs_;
};

void testHttpFakeEngine() {
    FakeApi api;
    auto http = std::make_unique<HttpServer>(api);
    http->start("127.0.0.1", 0);
    const std::uint16_t port = http->port();
    auto get = [&](const char* p) { return test::httpRequest("127.0.0.1", port, "GET", p, {}); };
    auto post = [&](const char* p, const std::string& b) { return test::httpRequest("127.0.0.1", port, "POST", p, b); };
    // facts: taken once, when the HttpServer was built; later changes of the source are not seen
    CHECK_EQ(api.facts_calls_.load(), 1);
    api.f_.model_name = "changed-after-start";
    // /health: health() + the startup / governor JSON
    {
        const auto h = get("/health");
        CHECK_EQ(h.status, 200);
        const std::string want = "{\"status\":\"ok\",\"busy\":true,\"slots_busy\":1,\"slots\":3,\"queued\":2,\"vram\":{\"local_mib\":5,"
                                 "\"budget_mib\":7,\"shared_mib\":0,\"spill\":false},\"startup\":{\"slots\":3},\"governor\":{\"fake_governor\":true}}";
        CHECK(h.body == want);
        if (h.body != want) std::fprintf(stderr, "  health: %s\n", h.body.c_str());
        CHECK_EQ(get("/v1/health").status, 200);
        CHECK_EQ(post("/health", "").status, 405);
    }
    // /v1/models: facts (id, arch, n_vocab, n_embd, n_layer, mtp) + the published context
    {
        const auto m = get("/v1/models");
        CHECK_EQ(m.status, 200);
        CHECK(m.body.find("\"id\":\"fake-model\"") != std::string::npos);
        CHECK(m.body.find(std::format("\"meta\":{{\"arch\":\"qwen35\",\"n_ctx\":4096,\"n_vocab\":{},\"n_embd\":64,\"n_layer\":4,\"mtp\":true}}",
                                      tok().size())) != std::string::npos);
        CHECK_EQ(get("/models").status, 200);
    }
    // /props: facts + the live context (ctxPublished is a call, not a fact) + stats + governor
    {
        api.ctx_ = 2048;
        const auto p = get("/props");
        CHECK_EQ(p.status, 200);
        // (build_info carries the version: compared up to "whirl " and from the closing quote on)
        const std::string pre = "{\"default_generation_settings\":{\"id\":0,\"n_ctx\":2048,\"model\":\"fake-model\",\"speculative\":true,"
                                "\"is_processing\":true},\"total_slots\":3,\"model_path\":\"fake-file.gguf\",\"model_alias\":\"fake-model\","
                                "\"modalities\":{\"vision\":false,\"audio\":false},\"build_info\":\"whirl ";
        const std::string post_ = "\",\"numerics\":{\"mode\":\"fake\"},\"startup\":{\"slots\":3},\"governor\":{\"fake_governor\":true},"
                                  "\"kv_ssd\":{\"mode\":\"always\",\"bytes_written\":11,\"bytes_shared\":0,\"bytes_used\":0,\"entries\":2,\"superseded\":0}}";
        const bool ok = p.body.starts_with(pre) && p.body.ends_with(post_) && p.body.size() > pre.size() + post_.size() &&
                        p.body.find('"', pre.size()) == p.body.size() - post_.size();
        CHECK(ok);
        if (!ok) std::fprintf(stderr, "  props: %s\n", p.body.c_str());
        CHECK_EQ(get("/v1/props").status, 200);
        CHECK(get("/version").body.find("\"name\":\"whirl\"") != std::string::npos);
    }
    CHECK_EQ(api.facts_calls_.load(), 1);
    // error paths: answered by the HTTP layer, never queued (submitAndWait not called)
    {
        CHECK_EQ(get("/nope").status, 404);
        CHECK_EQ(get("/api/tags").status, 404);  // known probe
        CHECK_EQ(get("/v1/chat/completions").status, 405);
        const auto bad = post("/v1/chat/completions", "{not json");
        CHECK_EQ(bad.status, 400);
        CHECK(bad.body.find("request body is not valid JSON") != std::string::npos);
        CHECK(post("/v1/chat/completions", "[1]").body.find("request body must be a JSON object") != std::string::npos);
        CHECK(post("/v1/chat/completions", "{}").body.find("'messages' is required") != std::string::npos);
        CHECK(post("/v1/completions", "{}").body.find("'prompt' is required") != std::string::npos);
        CHECK(post("/v1/completions", R"({"prompt":[]})").body.find("'prompt' must not be empty") != std::string::npos);
        CHECK(post("/v1/completions", R"({"prompt":["a","b"]})").body.find("only one prompt per request") != std::string::npos);
        CHECK(post("/v1/completions", R"({"prompt":true})").body.find("'prompt' must be a string") != std::string::npos);
        // n_vocab comes from the facts: the first id past it is refused
        CHECK(post("/v1/completions", std::format("{{\"prompt\":[{}]}}", tok().size())).body.find("prompt token ids must be integers") !=
              std::string::npos);
        CHECK(post("/v1/completions", R"({"prompt":""})").body.find("the prompt is empty") != std::string::npos);
        const auto img = post("/v1/chat/completions",
                              R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]})");
        CHECK_EQ(img.status, 400);
        CHECK(img.body.find("image content needs an mmproj") != std::string::npos);
        CHECK(post("/v1/chat/completions", R"({"messages":[{"role":"user","content":[{"type":"video","video":"x"}]}]})")
                  .body.find("video content is not supported") != std::string::npos);
        CHECK(api.jobs().empty());
    }
    // completions: stream == non-stream; the job's tokens are the product's completionTokens
    {
        const auto a = post("/v1/completions", R"({"prompt":"def quicksort(xs):\n    ","max_tokens":64})");
        const auto b = post("/v1/completions",
                            R"({"prompt":"def quicksort(xs):\n    ","max_tokens":64,"stream":true,"stream_options":{"include_usage":true}})");
        CHECK_EQ(a.status, 200);
        CHECK_EQ(b.status, 200);
        CHECK(b.headers.find("text/event-stream") != std::string::npos);
        std::string fin;
        CHECK_EQ(streamText(b, false, &fin), completionText(a));
        CHECK_EQ(completionText(a), std::string("The answer is 42, said the fake engine."));
        CHECK_EQ(fin, std::string("stop"));
        const auto evs = test::sseEvents(b.body);
        CHECK(!evs.empty() && evs.back() == "[DONE]");
        CHECK(evs.size() >= 3 && evs[evs.size() - 2].find("\"choices\":[]") != std::string::npos);
        const auto js = api.jobs();
        CHECK_EQ(js.size(), std::size_t(2));
        if (js.size() == 2) {
            const auto want = tok().encode("def quicksort(xs):\n    ", true);
            CHECK(js[0].kind == JobKind::completion && !js[0].stream && js[1].stream);
            CHECK(js[0].tokens == std::vector<std::uint32_t>(want.begin(), want.end()));
            CHECK(js[1].tokens == js[0].tokens);
        }
        // a max_tokens cut: finish_reason length in both
        const auto c = post("/v1/completions", R"({"prompt":"x","max_tokens":3})");
        const auto d = post("/v1/completions", R"({"prompt":"x","max_tokens":3,"stream":true})");
        std::string fin2;
        CHECK_EQ(streamText(d, false, &fin2), completionText(c));
        CHECK_EQ(fin2, std::string("length"));
        CHECK(c.body.find("\"finish_reason\":\"length\"") != std::string::npos);
    }
    // chat: stream == non-stream (content and reasoning), the first chunk is the role delta; tokens = chatPrompt
    {
        const std::string msgs = R"("messages":[{"role":"user","content":"What is 6 x 7?"}])";
        for (bool think : {false, true}) {
            const std::string kw = think ? R"("chat_template_kwargs":{"enable_thinking":true})" : R"("chat_template_kwargs":{"enable_thinking":false})";
            const auto n0 = api.jobs().size();
            const auto c1 = post("/v1/chat/completions", "{" + msgs + ",\"max_tokens\":64," + kw + "}");
            const auto c2 = post("/v1/chat/completions", "{" + msgs + ",\"max_tokens\":64,\"stream\":true," + kw + "}");
            CHECK_EQ(c1.status, 200);
            const json::Value v1 = parseBody(c1);
            std::string content1, reason1;
            if (const json::Value* ch = v1.get("choices"))
                if (const json::Value* m = ch->asArray()[0].get("message")) {
                    if (const json::Value* c = m->get("content"); c && c->isString()) content1 = c->asString();
                    if (const json::Value* r = m->get("reasoning_content"); r && r->isString()) reason1 = r->asString();
                }
            CHECK_EQ(content1, std::string("The answer is 42, said the fake engine."));
            CHECK_EQ(streamText(c2, true), content1);
            CHECK_EQ(reason1, std::string(think ? "Short reasoning here." : ""));
            std::string reason2;
            for (const std::string& ev : test::sseEvents(c2.body)) {
                if (ev == "[DONE]") continue;
                const json::Value v = json::parse(ev);
                const json::Value* ch = v.get("choices");
                if (!ch || ch->asArray().empty()) continue;
                if (const json::Value* dl = ch->asArray()[0].get("delta"))
                    if (const json::Value* r = dl->get("reasoning_content"); r && r->isString()) reason2 += r->asString();
            }
            CHECK_EQ(reason2, reason1);
            CHECK(test::sseEvents(c2.body).front().find("{\"role\":\"assistant\",\"content\":null}") != std::string::npos);
            const auto js = api.jobs();
            CHECK_EQ(js.size(), n0 + 2);
            if (js.size() == n0 + 2) {
                chat::ChatRequest req = chat::parseChatRequest(json::parse("{" + msgs + "," + kw + "}"));
                const ChatPrompt cp = chatPrompt(tok(), chat::TemplateKind::a, req, false);
                CHECK(js[n0].kind == JobKind::chat && js[n0].think == think);
                CHECK(js[n0].tokens == std::vector<std::uint32_t>(cp.ids.begin(), cp.ids.end()));
                CHECK(js[n0 + 1].tokens == js[n0].tokens);
            }
        }
    }
    // ids: the chatcmpl- prefix from the IdGen; the same seed gives the same sequence
    {
        IdGen a(7), b(7);
        CHECK_EQ(a.id("x-", 24), b.id("x-", 24));
        CHECK_EQ(a.next(), b.next());
        const json::Value v = parseBody(post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})"));
        const json::Value* id = v.get("id");
        CHECK(id && id->isString() && id->asString().starts_with("chatcmpl-") && id->asString().size() == 9 + 24);
    }
    http->stop();
    http.reset();
    CHECK_EQ(api.building_.load(), 0);  // every connection's beginBuilding has its endBuilding
}

// arch024-S11c (F-7, S11b mutant C1): the startup's checkpoint VRAM ("VRAM use: ... checkpoints X GiB") is
// memInfo before and after allocState; allocStateMeasured must read memInfo first, allocate, then read it again
// (C1 = allocState before the first read: the checkpoints show 0 GiB and "other" absorbs them).
void testAllocStateMeasured() {
    using Call = NullBackend::Call;
    NullBackend ops;
    ServerSetup su;
    MockModel model(ops, tok(), su.mc);
    EngineOptions eo = su.eo;
    eo.parallel = su.mc.parallel;
    eo.ctx = su.mc.slot_ctx;
    eo.model_name = "mock";
    eo.use_mtp = su.mc.mtp && eo.use_mtp;
    eo.use_dflash = false;
    Engine e(model, ops, tok(), eo);
    ops.recordThreads(true);
    const Engine::StateMem r = e.allocStateMeasured();
    ops.recordThreads(false);
    const auto calls = ops.callThreads();
    std::size_t mallocs = 0, infos = 0;
    for (const auto& c : calls) {
        mallocs += c.first == Call::malloc || c.first == Call::host_malloc;
        infos += c.first == Call::mem_info;
    }
    CHECK(eo.n_ck > 0 && mallocs > 0);  // the checkpoint buffers are allocated through the Backend
    CHECK_EQ(infos, std::size_t{2});
    CHECK(!calls.empty() && calls.front().first == Call::mem_info);
    CHECK(!calls.empty() && calls.back().first == Call::mem_info);
    CHECK(r.before.total == ops.mem.info.total && r.after.total == ops.mem.info.total);
}

// arch024-S11c: test scratch folders are private to the process unless WHIRL_TEST_TMP names one
// (%TEMP%\whirl-tests was shared: concurrent agents' test runs collided)
void testTestTmpBase() {
    const std::filesystem::path b = whirl::test::testTmpBase();
    CHECK(b == whirl::test::testTmpBase());  // one folder per process
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) {
        CHECK(b == std::filesystem::path(widen(t)));
    } else {
        CHECK_EQ(b.filename().wstring(), std::to_wstring(GetCurrentProcessId()));
        CHECK_EQ(b.parent_path().filename().wstring(), std::wstring(L"whirl-tests"));
        CHECK(std::filesystem::is_directory(b));
    }
}


// ---------------------------------------------------------------------------
// review/s5a-cachemiss: independent test, written from the symptom description only (not from tier_precise_cont).
// Symptom: in precise (COLD-RESUME) a session's next turn resumes at its own tail checkpoint (kind 'prefix'); the
// restore was taken as a shared-prefix restore, so the next spill made a NEW RAM entry instead of updating the
// session's entry in place: two entries per session, the new one not reused and not written to SSD at shutdown.
// Expected: one entry per session, in place, reused, one SSD file per session; a true shared system entry and
// another question on the same document unchanged. Every answer equals the cold greedy reference (bitwise).
namespace rev5a {

// entries of the tier whose tokens start with `pre`
std::vector<const tier::Entry*> withPrefix(tier::Tier* t, std::span<const std::uint32_t> pre) {
    std::vector<const tier::Entry*> v;
    for (const tier::Entry* e : t->entries) {
        if (e->n_tok < pre.size() || e->tokens.size() < pre.size()) continue;
        if (std::equal(pre.begin(), pre.end(), e->tokens.begin())) v.push_back(e);
    }
    return v;
}

// a shared prefix entry (spillSpe: one checkpoint of a shared kind at the entry's end)
bool isSpeEntry(const tier::Entry* e) { return e->nck == 1 && e->ck[0].kind >= 4 && e->ck[0].pos == e->n_tok; }

// the session's own (lineage) entries among withPrefix
std::vector<const tier::Entry*> lineage(tier::Tier* t, std::span<const std::uint32_t> pre) {
    std::vector<const tier::Entry*> v;
    for (const tier::Entry* e : withPrefix(t, pre))
        if (!isSpeEntry(e)) v.push_back(e);
    return v;
}

std::vector<std::uint32_t> next(const std::vector<std::uint32_t>& p, const std::vector<std::uint32_t>& out, std::uint32_t add,
                                std::uint64_t seed) {
    std::vector<std::uint32_t> v = p;
    v.insert(v.end(), out.begin(), out.end());
    return makePrompt(static_cast<std::uint32_t>(v.size() + add), seed, v);
}

}  // namespace rev5a

void testRevS5aSessions() {
    using namespace rev5a;
    const std::filesystem::path base = whirl::test::testTmpBase();
    const std::string dir = narrow((base / L"server_test_rev5a").wstring());
    for (const bool precise : {false, true}) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;  // one session in VRAM: the other one is spilled / restored every turn
        su.tier = true;
        su.ssd_dir = dir;
        su.tier_min = 512;
        su.tier_shutdown = true;  // --kv-ssd-mode shutdown: only reused entries reach the SSD (at close)
        su.eo.n_ck = 3;           // room for the tail checkpoint
        su.eo.cold_resume = precise;
        const auto a1 = makePrompt(3000, 7101);
        const auto b1 = makePrompt(3200, 7102);
        const std::span<const std::uint32_t> a_key(a1.data(), 1024), b_key(b1.data(), 1024);
        std::vector<std::uint32_t> a4;
        std::size_t a3_len = 0;
        {
            TestServer srv(su);
            g_log.clear();
            auto turn = [&](const std::vector<std::uint32_t>& p) {
                const auto ref = srv.model().greedyReference(p, 40, true);
                auto r = srv.post("/v1/completions", completionBody(p, 40));
                CHECK_EQ(completionText(r), decode(ref));
                srv.onEngine([] {});  // the spill after the response is done
                return std::make_pair(ref, cachedTokens(r));
            };
            const auto [oa1, ca1] = turn(a1);
            const auto [ob1, cb1] = turn(b1);
            const auto a2 = next(a1, oa1, 300, 7103);
            const auto [oa2, ca2] = turn(a2);  // restored from RAM (precise: at A's own tail checkpoint)
            CHECK(ca2 >= 2048);
            const auto b2 = next(b1, ob1, 300, 7104);
            const auto [ob2, cb2] = turn(b2);  // evicts A after its turn 2 spill
            CHECK(cb2 >= 2048);
            const std::size_t inplace_after_b2 = g_log.count("in place");
            const auto a3 = next(a2, oa2, 300, 7105);
            a3_len = a3.size();
            const auto [oa3, ca3] = turn(a3);
            CHECK(ca3 > a2.size() - 1300);  // turn 2's own KV came back (past turn 1)
            CHECK(ca3 > a1.size() + oa1.size());
            std::size_t na = 0, nb = 0, n_all = 0;
            bool a_reused = false, b_reused = false;
            srv.onEngine([&] {
                const auto ea = lineage(srv.tierp(), a_key), eb = lineage(srv.tierp(), b_key);
                na = ea.size();
                nb = eb.size();
                n_all = srv.tierp()->entries.size();
                a_reused = na == 1 && ea[0]->reused;
                b_reused = nb == 1 && eb[0]->reused;
            });
            std::fprintf(stderr, "  (rev5a precise %d: A entries %zu reused %d, B entries %zu reused %d, all %zu, in place %zu/%zu, cached a2 %llu a3 %llu)\n",
                         precise ? 1 : 0, na, a_reused ? 1 : 0, nb, b_reused ? 1 : 0, n_all, inplace_after_b2, g_log.count("in place"),
                         static_cast<unsigned long long>(ca2), static_cast<unsigned long long>(ca3));
            CHECK_EQ(na, 1u);
            CHECK_EQ(nb, 1u);
            CHECK(a_reused);
            CHECK(b_reused);
            CHECK(inplace_after_b2 >= 2);  // A turn 2 and B turn 2 updated their entries in place
            a4 = next(a3, oa3, 200, 7106);
            CHECK_EQ(srv.model().poison(), 0u);
            CHECK_EQ(srv.model().hidMismatch(), 0u);
        }  // close: shutdown mode writes the reused entries
        {
            TestServer srv(su);
            g_log.clear();
            std::size_t na = 0, nb = 0, nssd = 0;
            srv.onEngine([&] {
                na = lineage(srv.tierp(), a_key).size();
                nb = lineage(srv.tierp(), b_key).size();
                nssd = srv.tierp()->ssdEntries();
            });
            std::fprintf(stderr, "  (rev5a precise %d restart: SSD entries %zu, A %zu, B %zu)\n", precise ? 1 : 0, nssd, na, nb);
            CHECK_EQ(na, 1u);
            CHECK_EQ(nb, 1u);
            const auto ref = srv.model().greedyReference(a4, 40, true);
            auto r = srv.post("/v1/completions", completionBody(a4, 40));
            CHECK_EQ(completionText(r), decode(ref));
            CHECK(srv.tierStats().restores_ssd >= 1);
            CHECK(cachedTokens(r) > a3_len - 1300);  // A's newest turn (3) came back from the SSD
            CHECK_EQ(srv.model().poison(), 0u);
        }
        std::filesystem::remove_all(dir, ec);
    }
}

// guards: (1) two sessions sharing a long system message, then diverging (tail > the near-end threshold): the shared
// system entry is never extended / owned, each session keeps exactly one entry; (2) another question on the same
// document (lcp < n_tok of the first question's entry): the first entry is left as it was, the second is its own.
void testRevS5aGuards() {
    using namespace rev5a;
    const ChatBuild cb;
    for (const bool precise : {false, true}) {
        {
            ServerSetup su = baseSetup();
            su.mc.parallel = 1;
            su.mc.slot_ctx = 8192;
            su.mc.pool_tokens = 8192;
            su.tier = true;
            su.tier_min = 512;
            su.eo.sys_min = 512;
            su.eo.n_ck = 3;
            su.eo.cold_resume = precise;
            TestServer srv(su);
            g_log.clear();
            const auto s1 = cb.conv(2000, 7201, {{4500, 7202}});
            const auto s2 = cb.conv(2000, 7201, {{4500, 7203}});
            std::vector<std::uint32_t> sys_toks;
            std::uint32_t sys_n = 0;
            std::uint64_t sys_id = 0;
            auto turn = [&](const std::vector<std::uint32_t>& p) {
                const auto ref = srv.model().greedyReference(p, 24, true);
                auto r = srv.post("/v1/completions", completionBody(p, 24));
                CHECK_EQ(completionText(r), decode(ref));
                srv.onEngine([] {});
                return std::make_pair(ref, cachedTokens(r));
            };
            const auto [o11, c11] = turn(s1);
            const auto [o21, c21] = turn(s2);
            CHECK(c21 >= 2000);
            const auto s1b = next(s1, o11, 200, 7204);
            const auto [o12, c12] = turn(s1b);
            CHECK(c12 > 2100);
            const auto s2b = next(s2, o21, 200, 7205);
            const auto [o22, c22] = turn(s2b);
            CHECK(c22 > 2100);
            const auto s1c = next(s1b, o12, 200, 7206);
            const auto [o13, c13] = turn(s1c);
            CHECK(c13 >= (precise ? 6101u : s1.size()));  // precise: s1c's own tail (schedule 2005 + k * 1024)
            const std::span<const std::uint32_t> k1(s1.data(), 2600), k2(s2.data(), 2600);
            std::size_t n1 = 0, n2 = 0, nsys = 0, sys_owned = 0, all = 0;
            srv.onEngine([&] {
                n1 = lineage(srv.tierp(), k1).size();
                n2 = lineage(srv.tierp(), k2).size();
                all = srv.tierp()->entries.size();
                for (const tier::Entry* e : srv.tierp()->entries)
                    if (e->nck >= 1 && e->ck[0].kind == 4 && e->ck[0].pos == e->n_tok) {
                        ++nsys;
                        sys_n = e->n_tok;
                        sys_id = e->id;
                        if (e->owner) ++sys_owned;
                    }
            });
            std::fprintf(stderr, "  (rev5a guard sys precise %d: S1 %zu, S2 %zu, sys entries %zu (n %u, owned %zu), all %zu; cached %llu %llu %llu %llu)\n",
                         precise ? 1 : 0, n1, n2, nsys, sys_n, sys_owned, all, (unsigned long long)c21, (unsigned long long)c12,
                         (unsigned long long)c22, (unsigned long long)c13);
            CHECK_EQ(n1, 1u);
            CHECK_EQ(n2, 1u);
            CHECK(nsys <= 1);
            CHECK_EQ(sys_owned, 0u);
            CHECK_EQ(srv.model().poison(), 0u);
            (void)sys_id;
        }
        {
            ServerSetup su = baseSetup();
            su.mc.parallel = 1;
            su.mc.slot_ctx = 8192;
            su.mc.pool_tokens = 8192;
            su.tier = true;
            su.tier_min = 512;
            su.eo.n_ck = 3;
            su.eo.cold_resume = precise;
            TestServer srv(su);
            g_log.clear();
            const auto doc = makePrompt(6000, 7301);
            const auto q1 = makePrompt(6090, 7302, doc);
            const auto q2 = makePrompt(6100, 7303, doc);
            const auto other = makePrompt(7000, 7304);
            auto turn = [&](const std::vector<std::uint32_t>& p) {
                const auto ref = srv.model().greedyReference(p, 24, true);
                auto r = srv.post("/v1/completions", completionBody(p, 24));
                CHECK_EQ(completionText(r), decode(ref));
                srv.onEngine([] {});
                return std::make_pair(ref, cachedTokens(r));
            };
            const auto [oq1, cq1] = turn(q1);
            std::vector<std::uint32_t> q1_toks;
            std::uint32_t q1_n = 0;
            std::uint64_t q1_id = 0;
            srv.onEngine([&] {
                for (const tier::Entry* e : withPrefix(srv.tierp(), q1)) {
                    q1_id = e->id;
                    q1_n = e->n_tok;
                    q1_toks.assign(e->tokens.begin(), e->tokens.begin() + e->n_tok);
                }
            });
            const auto [oo, co] = turn(other);  // evicts the document
            const auto [oq2, cq2] = turn(q2);   // restored from q1's entry, diverges inside it
            CHECK(cq2 >= 4096);
            const auto [oo2, co2] = turn(other);  // evicts again: q2 spilled
            bool q1_same = false;
            std::size_t nq2 = 0, ndoc = 0;
            srv.onEngine([&] {
                if (const tier::Entry* e = srv.tierp()->find(q1_id))
                    q1_same = e->n_tok == q1_n && std::equal(q1_toks.begin(), q1_toks.end(), e->tokens.begin());
                nq2 = withPrefix(srv.tierp(), q2).size();
                ndoc = withPrefix(srv.tierp(), doc).size();
            });
            std::fprintf(stderr, "  (rev5a guard doc precise %d: q1 entry %016llx n %u unchanged %d, q2 entries %zu, doc entries %zu, cached q2 %llu)\n",
                         precise ? 1 : 0, (unsigned long long)q1_id, q1_n, q1_same ? 1 : 0, nq2, ndoc, (unsigned long long)cq2);
            CHECK(q1_id != 0);
            CHECK(q1_same);
            CHECK_EQ(nq2, 1u);
            CHECK_EQ(srv.model().poison(), 0u);
        }
    }
}

// probe (reported, not a pass condition): precise turn 2 after a long answer (tail past the near-end threshold of
// spillSlot): does the session still end with one entry?
void testRevS5aLongAnswerProbe() {
    using namespace rev5a;
    for (const bool precise : {false, true}) {
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 16384;
        su.mc.pool_tokens = 12288;
        su.tier = true;
        su.tier_min = 512;
        su.eo.n_ck = 3;
        su.eo.cold_resume = precise;
        TestServer srv(su);
        g_log.clear();
        const auto a1 = makePrompt(3000, 7401);
        const auto b1 = makePrompt(11000, 7402);
        const auto oa1 = srv.model().greedyReference(a1, 4800, true);
        auto r1 = srv.post("/v1/completions", completionBody(a1, 4800));
        CHECK_EQ(completionText(r1), decode(oa1));
        srv.onEngine([] {});
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(b1, 16))), decode(srv.model().greedyReference(b1, 16, true)));
        srv.onEngine([] {});
        const auto a2 = next(a1, oa1, 300, 7403);
        const auto ref2 = srv.model().greedyReference(a2, 24, true);
        auto r2 = srv.post("/v1/completions", completionBody(a2, 24));
        CHECK_EQ(completionText(r2), decode(ref2));
        srv.onEngine([] {});
        CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(b1, 16))), decode(srv.model().greedyReference(b1, 16, true)));
        srv.onEngine([] {});
        std::size_t na = 0, nre = 0;
        srv.onEngine([&] {
            for (const tier::Entry* e : lineage(srv.tierp(), std::span<const std::uint32_t>(a1.data(), 1024))) {
                ++na;
                nre += e->reused ? 1 : 0;
            }
        });
        std::fprintf(stderr, "  (rev5a long-answer probe precise %d: A turn-1 entry %zu tok answer, cached a2 %llu, A entries %zu (reused %zu), in place %zu)\n",
                     precise ? 1 : 0, oa1.size(), (unsigned long long)cachedTokens(r2), na, nre, g_log.count("in place"));
        // bug-s5a-cachemiss review F1 (hard since the F1 fix): a long answer leaves one reused entry per session
        CHECK_EQ(na, std::size_t{1});
        CHECK_EQ(nre, std::size_t{1});
        CHECK_EQ(srv.model().poison(), 0u);
    }
}


// probe (reported): precise, two sessions alternating 4 turns each on one slot. Does a session's own earlier turn
// become a "shared prefix" (lcpTarget counts the session's own tier entry) and does a later turn then restore from
// that shared copy (restoreIsShared: pos == n_tok -> shared -> lineage lost again)?
void testRevS5aTurnsProbe() {
    using namespace rev5a;
    for (const bool precise : {false, true}) {
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;
        su.tier = true;
        su.tier_min = 512;
        su.eo.n_ck = 3;
        su.eo.cold_resume = precise;
        TestServer srv(su);
        g_log.clear();
        std::vector<std::uint32_t> pa = makePrompt(3000, 7501), pb = makePrompt(3200, 7502);
        const std::span<const std::uint32_t> ka(pa.data(), 1024);
        const std::vector<std::uint32_t> kb_v(pb.begin(), pb.begin() + 1024);
        const std::vector<std::uint32_t> ka_v(pa.begin(), pa.begin() + 1024);
        (void)ka;
        std::string trace;
        for (int t = 0; t < 3; ++t) {  // (a 4th B turn exceeds slot_ctx)
            for (int s = 0; s < 2; ++s) {
                std::vector<std::uint32_t>& p = s == 0 ? pa : pb;
                const auto ref = srv.model().greedyReference(p, 40, true);
                auto r = srv.post("/v1/completions", completionBody(p, 40));
                CHECK_EQ(completionText(r), decode(ref));
                srv.onEngine([] {});
                std::size_t nl = 0, ns = 0;
                srv.onEngine([&] {
                    for (const tier::Entry* e : withPrefix(srv.tierp(), s == 0 ? ka_v : kb_v)) (isSpeEntry(e) ? ns : nl) += 1;
                });
                trace += std::format(" {}{}:c{}/L{}/S{}", s == 0 ? 'A' : 'B', t + 1, cachedTokens(r), nl, ns);
                // bug-s5a-cachemiss review F2 (hard since the F2 fix): every turn keeps one lineage entry and no shared
                // copy of the session's own earlier turn (A and B share no prefix)
                CHECK_EQ(nl, std::size_t{1});
                CHECK_EQ(ns, std::size_t{0});
                if (t > 0) CHECK(cachedTokens(r) >= 2048);  // turns 2, 3 resume from the session's own entry
                p = next(p, ref, 300, 7510 + static_cast<std::uint64_t>(t * 2 + s));
            }
        }
        std::fprintf(stderr, "  (rev5a turns probe precise %d:%s; in place %zu, new entry %zu, shared spills %zu)\n", precise ? 1 : 0,
                     trace.c_str(), g_log.count("(in place"), g_log.count("(new entry"), g_log.count("shared prefix: spill"));
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
}


// review round 2 probes (print only): (a) dropContained after a lost lineage (failed restore), then the same
// conversation's turn 2 is regenerated with another last message: which entry / checkpoint serves it?
// (b) two different sessions on one slot with the same long document: is the shared LCP copy still made?
void testRevS5aR2Probe() {
    using namespace rev5a;
    for (const bool precise : {false, true}) {
        {
            ServerSetup su = baseSetup();
            su.mc.parallel = 1;
            su.mc.slot_ctx = 8192;
            su.mc.pool_tokens = 4096;
            su.tier = true;
            su.tier_min = 512;
            su.eo.n_ck = 3;
            su.eo.cold_resume = precise;
            TestServer srv(su);
            g_log.clear();
            const auto pa = makePrompt(3000, 7601), pb = makePrompt(3200, 7602);
            const auto outa = srv.model().greedyReference(pa, 40, true);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
            const auto pa2 = next(pa, outa, 300, 7603);
            srv.ops().failHostToDevCopy(1);
            auto r2 = srv.post("/v1/completions", completionBody(pa2, 40));
            srv.ops().failHostToDevCopy(0);
            CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pa2, 40, true)));
            srv.onEngine([] {});
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
            srv.onEngine([] {});
            // regenerate turn 2 with another last user message (diverges right after turn 1's answer)
            const auto pa2b = next(pa, outa, 300, 7604);
            auto r3 = srv.post("/v1/completions", completionBody(pa2b, 40));
            CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2b, 40, true)));
            std::fprintf(stderr, "  (rev5a r2 regen-after-lost-lineage precise %d: turn-1 len %zu, regen cached %llu, contained-drops %zu)\n",
                         precise ? 1 : 0, pa.size() + outa.size(), static_cast<unsigned long long>(cachedTokens(r3)),
                         g_log.count("is contained in the new entry"));
            // fix-r21 (review R2-1): the cold-prefill entry of turn 2 holds turn 1 whole but not its gen-end
            // checkpoint, so turn 1's entry must stay (demoted): the regenerated turn 2 resumes at turn 1's gen-end
            // in balance (as before fix v2: 3039 of 3040), and precise keeps its 2048 (cold schedule) restore
            const std::size_t len1 = pa.size() + outa.size();
            if (precise) CHECK(cachedTokens(r3) >= 2048);
            else CHECK(cachedTokens(r3) + 1 >= len1);
            if (!precise) CHECK_EQ(g_log.count("tok): replaced"), 0u);
            CHECK_EQ(srv.model().poison(), 0u);
        }
        {
            ServerSetup su = baseSetup();
            su.mc.parallel = 1;
            su.mc.slot_ctx = 8192;
            su.mc.pool_tokens = 8192;
            su.tier = true;
            su.tier_min = 512;
            su.eo.n_ck = 3;
            su.eo.cold_resume = precise;
            TestServer srv(su);
            g_log.clear();
            const auto doc = makePrompt(6000, 7701);
            std::string cs;
            for (std::uint64_t q = 0; q < 3; ++q) {
                const auto p = makePrompt(static_cast<std::uint32_t>(6090 + 10 * q), 7702 + q, doc);
                auto r = srv.post("/v1/completions", completionBody(p, 24));
                CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 24, true)));
                srv.onEngine([] {});
                cs += std::format(" q{}:{}", q + 1, cachedTokens(r));
            }
            std::fprintf(stderr, "  (rev5a r2 same-doc sessions precise %d:%s; 'common prefix with an earlier session' %zu, shared spills %zu)\n",
                         precise ? 1 : 0, cs.c_str(), g_log.count("common prefix with an earlier session"), g_log.count("shared prefix: spill"));
            CHECK_EQ(srv.model().poison(), 0u);
        }
    }
}


// review round 4 probe (print only): lost lineage (failed restore) in --kv-ssd-mode shutdown, close, restart, then
// regenerate turn 2 (diverges right after turn 1's answer): which restore point survived the restart?
void testRevS5aR4RestartProbe() {
    using namespace rev5a;
    const std::filesystem::path base = whirl::test::testTmpBase();
    const std::string dir = narrow((base / L"server_test_rev5a_r4").wstring());
    for (const bool precise : {false, true}) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        ServerSetup su = baseSetup();
        su.mc.parallel = 1;
        su.mc.slot_ctx = 8192;
        su.mc.pool_tokens = 4096;
        su.tier = true;
        su.ssd_dir = dir;
        su.tier_min = 512;
        su.tier_shutdown = true;
        su.eo.n_ck = 3;
        su.eo.cold_resume = precise;
        const auto pa = makePrompt(3000, 7601), pb = makePrompt(3200, 7602);
        std::vector<std::uint32_t> outa;
        std::uint64_t pre_cached = 0;
        std::size_t n_before = 0;
        {
            TestServer srv(su);
            g_log.clear();
            outa = srv.model().greedyReference(pa, 40, true);
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pa, 40))), decode(outa));
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
            const auto pa2 = next(pa, outa, 300, 7603);
            srv.ops().failHostToDevCopy(1);
            auto r2 = srv.post("/v1/completions", completionBody(pa2, 40));
            srv.ops().failHostToDevCopy(0);
            CHECK_EQ(completionText(r2), decode(srv.model().greedyReference(pa2, 40, true)));
            srv.onEngine([] {});
            CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
            srv.onEngine([] {});
            srv.onEngine([&] { n_before = withPrefix(srv.tierp(), std::span<const std::uint32_t>(pa.data(), 1024)).size(); });
            (void)pre_cached;
        }
        {
            TestServer srv(su);
            g_log.clear();
            std::size_t na = 0, nssd = 0;
            srv.onEngine([&] {
                na = withPrefix(srv.tierp(), std::span<const std::uint32_t>(pa.data(), 1024)).size();
                nssd = srv.tierp()->ssdEntries();
            });
            const auto pa2b = next(pa, outa, 300, 7604);
            auto r3 = srv.post("/v1/completions", completionBody(pa2b, 40));
            CHECK_EQ(completionText(r3), decode(srv.model().greedyReference(pa2b, 40, true)));
            std::fprintf(stderr, "  (rev5a r4 restart precise %d: A entries before close %zu, SSD entries %zu, A after restart %zu, regen cached %llu)\n",
                         precise ? 1 : 0, n_before, nssd, na, static_cast<unsigned long long>(cachedTokens(r3)));
            // fix-r21 R4-1 / R4-2: the restore point of turn 1 survives the restart -- balance its gen-end (kept
            // entry stays reused, so the close flush writes it), precise the 2048 prefix checkpoint (the entry that
            // covers it inherits 'reused'); as on 28f21a3
            CHECK_EQ(cachedTokens(r3), (precise ? std::uint64_t{2048} : std::uint64_t{pa.size() + outa.size() - 1}));
            CHECK_EQ(srv.model().poison(), 0u);
        }
        std::filesystem::remove_all(dir, ec);
    }
}

int main(int argc, char** argv) {
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
    // arch024 M8: the test program is the "test build": an owner check violation throws std::logic_error; every test
    // must end with no violation other than the ones it causes on purpose (g_owner_expected)
    EngineThread::setThrowing(true);
    std::set<std::string> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) g_gguf = argv[++i];
        else if (a == "--only" && i + 1 < argc) {
            std::string s = argv[++i];
            std::size_t p = 0;
            while (p <= s.size()) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                only.insert(s.substr(p, e - p));
                p = e + 1;
            }
        } else if (a == "--verbose") g_verbose = true;
    }
    if (g_gguf.empty()) {
        if (const char* e = std::getenv("WHIRL_TEST_GGUF")) g_gguf = e;
    }
    if (g_gguf.empty()) {
        std::fprintf(stderr,
                     "whirl-server-tests needs a qwen35 GGUF for its tokenizer (the model itself is mocked):\n"
                     "  whirl-server-tests --gguf MODEL.gguf [--only NAME,...] [--verbose]   (or set WHIRL_TEST_GGUF)\n");
        return 2;
    }
    Log::setConsole(g_verbose);
    Log::setHook(&LogCapture::hook, &g_log);
    try {
        g_file = std::make_unique<gguf::File>(gguf::File::open(g_gguf));
        g_tok = std::make_unique<Tokenizer>(Tokenizer::fromGguf(*g_file));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cannot load the tokenizer from %s: %s\n", g_gguf.c_str(), ex.what());
        return 2;
    }
    struct T {
        const char* name;
        std::function<void()> fn;
    };
    const std::vector<T> tests = {
        {"vision", testVision},
        {"draft_caps", testDraftCaps},
        {"protocol", testProtocol},
        {"sampler", testSampler},
        {"chat_tokens", testChatTokens},
        {"endpoints", testEndpoints},
        {"http_fake_engine", testHttpFakeEngine},
        {"vram_watch", testVramWatch},
        {"startup_adjust", testStartupAdjust},
        {"governor", testGovernor},
        {"governor_restore", testGovernorRestore},
        {"ckpt_store", testCkptStore},
        {"ckpt_f1", testCkptF1},
        {"ckpt_restore_fail", testCkptRestoreFail},
        {"gov_park_spes", testGovParkSpes},
        {"backend_others_pinned", testBackendOthersPinned},
        {"env_matrix", testEnvMatrix},
        {"ckpt_engine_teardown", testCkptEngineTeardown},
        {"tier_io_thread_calls", testTierIoThreadCalls},
        {"alloc_state_measured", testAllocStateMeasured},
        {"test_tmp_base", testTestTmpBase},
        {"page_pool", testPagePool},
        {"tier_bridge_spill", testTierBridgeSpill},
        {"tier_bridge_supersede", testTierBridgeSupersede},
        {"tier_bridge_flush", testTierBridgeFlush},
        {"tier_drop_k2", testTierDropK2},
        {"tier_bridge_restore", testTierBridgeRestore},
        {"prefix_store", testPrefixStore},
        {"cold_resume_engine", testColdResumeEngine},
        {"rev5a_sessions", testRevS5aSessions},
        {"rev5a_guards", testRevS5aGuards},
        {"rev5a_long_probe", testRevS5aLongAnswerProbe},
        {"rev5a_turns_probe", testRevS5aTurnsProbe},
        {"rev5a_r2_probe", testRevS5aR2Probe},
        {"rev5a_r4_restart", testRevS5aR4RestartProbe},
        {"vision_split_engine", testVisionSplitEngine},
        {"compat_endpoints", testCompatEndpoints},
        {"http_limits", testHttpLimits},
        {"http_conn_cap", testHttpConnCap},
        {"http_stuck_reader", testHttpStuckReader},
        {"http_port_reuse", testHttpPortReuse},
        {"http_stream10", testHttpStream10},
        {"cors", testCors},
        {"greedy_reference", testGreedyMatchesReference},
        {"stream", testStreamEqualsNonStream},
        {"prefix_cache", testPrefixCacheMultiTurn},
        {"dflash_trim_ring", testDFlashTrimRing},
        {"doc_qa", testDocQa},
        {"tail_ck_sched", testTailCkOnSchedule},
        {"prefill_sched", testPrefillSchedule},
        {"cli_sched_helpers", testCliSchedHelpers},
        {"tail_ck_regress", testTailCkRegression},
        {"concurrency", [] { testConcurrency(4, 12, true); }},
        {"concurrency16", [] { testConcurrency(16, 24, true); }},
        {"concurrency_nomtp", [] { testConcurrency(4, 8, false); }},
        {"dflash_greedy", testDFlashGreedy},
        {"dflash_relax_exact", testDFlashRelaxExact},
        {"dflash_ngram", testDFlashNgram},
        {"dflash_rollback", testDFlashRollback},
        {"dflash_tier", testDFlashTier},
        {"concurrency_floor", [] { testConcurrency(4, 12, true, 20); }},
        {"prefill_batch_floor", testPrefillBatchFloor},
        {"slot_draft_plan", testSlotDraftPlan},
        {"slot_drafts_one_slot", testSlotDraftsOneSlot},
        {"concurrency_slot2", [] { testConcurrency(4, 12, true, 0, 2); }},
        {"concurrency_slot2_c2", [] { testConcurrency(2, 8, true, 0, 2); }},
        {"concurrency16_slot2", [] { testConcurrency(16, 24, true, 0, 2); }},
        {"decode_floor_math", testDecodeFloorMath},
        {"decode_floor", testDecodeFloor},
        {"pool", testPoolPressure},
        {"sys_prompt", testSharedSystemPrompt},
        {"cascade_sys", testCascadeSharedPrompt},
        {"tiers", testTiers},
        {"tier_restore_too_big", testTierRestoreTooBig},
        {"shutdown_evict_sys", testShutdownEvictSys},
        {"idle_write", testIdleWrite},
        {"close_reused", testCloseReused},
        {"restore_conc", testRestoreConcurrent},
        {"restore_throws", testRestoreThrows},
        {"tier_inplace", testTierInPlaceRegen},
        {"tier_precise_cont", testTierPreciseContinuation},
        {"tier_drop_contained", testTierDropContained},
        {"ckpts_clamp", testServeCkptsClamp},
        {"branch_turn", testBranchFromEarlierTurn},
        {"agent_idle_main", testAgentIdleMainNoSupersede},
        {"agent_idle_main_sup", testAgentIdleMainSupersede},
        {"supersede_compress", testSupersedeCompressCuts},
        {"sampling", testSampling},
        {"stop_eos", testStopAndEos},
        {"fr_tool_truncated", testFrToolTruncated},
        {"fr_nonstream_disconnect", testFrNonStreamDisconnect},
        {"fr_pool_prefill", testFrPoolPrefill},
        {"fr_pool_decode", testFrPoolDecode},
        {"fr_cross_origin", testFrCrossOrigin},
        {"fr_dflash_noreplay", testFrDFlashNoReplay},
        {"dflash_fit", testDFlashFit},
        {"fr_shutdown_503", testFrShutdown503},
        {"run_on_loop_timeout_cancels", testRunOnLoopTimeoutCancels},
        {"fr_protocol", testFrProtocol},
        {"fr_stop_tokens", testFrStopTokens},
        {"fr_log_injection", testFrLogInjection},
        {"fr_short_followup", testFrShortFollowUp},
        {"owner_check", testOwnerCheck},
    };
    // FIX-TEST-A (FR-TEST P3-3): an --only name that matches no test, or a run with 0 checks, is a usage
    // error (exit 2), never "0 passed, 0 failed" with exit 0
    for (const std::string& n : only) {
        bool known = false;
        for (const T& t : tests) known = known || n == t.name;
        if (!known) {
            std::fprintf(stderr, "whirl-server-tests: --only: no test named \"%s\"\n", n.c_str());
            Log::setHook(nullptr, nullptr);
            return 2;
        }
    }
    for (const T& t : tests) {
        if (!only.empty() && !only.count(t.name)) continue;
        const int f0 = g_fail, p0 = g_pass;
        const double t0 = nowSeconds();
        const std::uint64_t ov0 = EngineThread::violations();
        const int oe0 = g_owner_expected;
        try {
            t.fn();
        } catch (const std::exception& ex) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL %s: exception %s\n", t.name, ex.what());
        }
        // arch024 M8: a violation on an HTTP thread may be turned into a 500 by the handler: count them, not only throws
        if (EngineThread::violations() - ov0 != static_cast<std::uint64_t>(g_owner_expected - oe0)) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL %s: %llu owner check violation(s), %d expected\n", t.name,
                         static_cast<unsigned long long>(EngineThread::violations() - ov0), g_owner_expected - oe0);
        }
        std::printf("%-18s %s (%d checks, %.1f s)\n", t.name, g_fail == f0 ? "ok" : "FAILED", g_pass - p0 + g_fail - f0,
                    nowSeconds() - t0);
        std::fflush(stdout);
    }
    std::printf("server tests: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("server tests: owner check violations %llu (%d caused on purpose by owner_check)\n",
                static_cast<unsigned long long>(EngineThread::violations()), g_owner_expected);
    Log::setHook(nullptr, nullptr);
    if (g_pass + g_fail == 0) {
        std::printf("server tests: 0 checks ran: refusing to report success\n");
        return 2;
    }
    return g_fail ? 1 : 0;
}
