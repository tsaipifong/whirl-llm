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
#include "server/http.h"
#include "server/log.h"
#include "server/protocol.h"
#include "server/tokens.h"
#include "tier/kv_tier.h"
#include "whirl/vram_limit.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace whirl;
using namespace whirl::server;
using whirl::test::HttpResult;
using whirl::test::MockConfig;
using whirl::test::MockDeviceOps;
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
    hip::LocalMemSnap memSnap() const override { return {}; }
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
    HttpResult post(const std::string& path, const std::string& body) {
        return test::httpRequest("127.0.0.1", http_->port(), "POST", path, body);
    }
    HttpResult get(const std::string& path) { return test::httpRequest("127.0.0.1", http_->port(), "GET", path); }
    MockModel& model() { return *model_; }
    MockVision* vision() { return vis_.get(); }
    Engine& engine() { return *engine_; }
    tier::Tier* tierp() { return tier_.get(); }
    MockDeviceOps& ops() { return ops_; }
    std::uint16_t port() const { return http_->port(); }

private:
    ServerSetup su_;
    MockDeviceOps ops_;
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
    CHECK_EQ(srv.get("/health").status, 200);
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
    CHECK_EQ(prefillExecFor(prefillExecDefault(false, false), 4096), std::size_t{2048});
    CHECK_EQ(prefillExecFor(prefillExecDefault(false, false), 2048), std::size_t{1024});
    CHECK_EQ(prefillExecFor(prefillExecDefault(false, true), 4096), std::size_t{2048});  // gfx1201 MoE: unchanged
    // SRV-PP-DENSE: gfx1151 dense keeps 2-chunk forwards (4096 lost 13% at 128k), MoE fills the batch
    CHECK_EQ(prefillExecDefault(true, false), std::size_t{2048});
    CHECK_EQ(prefillExecDefault(true, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(prefillExecDefault(true, false), 4096, true), std::size_t{2048});
    CHECK_EQ(prefillExecFor(16384, 4096, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(prefillExecDefault(true, true), 4096, true), std::size_t{4096});
    CHECK_EQ(prefillExecFor(4096, 1280, true), std::size_t{1024});
    CHECK_EQ(prefillExecFor(4096, 4352, true), std::size_t{4096});
    for (std::size_t n : {std::size_t{4100}, std::size_t{8232}, std::size_t{9500}, std::size_t{32777}, std::size_t{131087}}) {
        for (std::uint32_t mb : {1280u, 2048u, 3000u, 4096u, 4352u, 8192u}) {
            for (bool moe : {false, true}) {
                const std::size_t ex = prefillExecFor(prefillExecDefault(true, moe), mb, true);
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
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
        CHECK(srv.tierp()->stats.restores_ram >= 1);
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

// host tiers: an evicted session is restored from RAM, then (new engine on
// the same SSD directory) from SSD; outputs equal the cold result
void testTiers() {
    // WHIRL_TEST_TMP, else %TEMP%\whirl-tests; this test uses <base>\server_test_ssd
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
        CHECK(srv.tierp()->stats.restores_ram >= 1);
        CHECK(g_log.count("kv tier: restored") >= 1);
        // let the SSD writes finish (delay 0). A was reused (restored) and is written; B is a one-shot
        // prompt and, with the seen-once filter, is not
        for (int i = 0; i < 400 && srv.tierp()->ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(srv.tierp()->ssdEntries() >= 1);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
    {
        // restart: the SSD index brings the entries back; A's turn restores from SSD
        TestServer srv(su);
        g_log.clear();
        CHECK(srv.tierp()->ssdEntries() >= 1);
        auto r = srv.post("/v1/completions", completionBody(pa2, 40));
        CHECK_EQ(completionText(r), decode(srv.model().greedyReference(pa2, 40, true)));
        CHECK(srv.tierp()->stats.restores_ssd >= 1);
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
        CHECK_EQ(srv.tierp()->stats.restores_ssd + srv.tierp()->stats.restores_ram, 0u);
        CHECK_EQ(cachedTokens(r), 0u);
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
    const std::size_t inplace = g_log.count("in place");
    CHECK(inplace >= 1);
    // B evicts the slot
    CHECK_EQ(completionText(srv.post("/v1/completions", completionBody(pb, 40))), decode(srv.model().greedyReference(pb, 40, true)));
    std::vector<std::uint32_t> p4 = p3;
    p4.insert(p4.end(), out3.begin(), out3.end());
    p4 = makePrompt(static_cast<std::uint32_t>(p4.size() + 200), 805, p4);
    auto r4 = srv.post("/v1/completions", completionBody(p4, 40));
    CHECK_EQ(completionText(r4), decode(srv.model().greedyReference(p4, 40, true)));
    CHECK(srv.tierp()->stats.restores_ram >= 1);
    CHECK(cachedTokens(r4) > base.size());
    CHECK_EQ(srv.model().poison(), 0u);
    std::fprintf(stderr, "  (tier_inplace: in-place spills %zu, restores_ram %llu, cached %u)\n", inplace,
                 static_cast<unsigned long long>(srv.tierp()->stats.restores_ram), static_cast<unsigned>(cachedTokens(r4)));
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
    CHECK(srv.tierp()->stats.restores_ram >= 1);
    CHECK(cachedTokens(r) >= pa.size());
    std::size_t over = 0;
    for (const tier::Entry* e : srv.tierp()->entries) over += e->nck > tier::max_ck ? 1 : 0;
    CHECK_EQ(over, 0u);
    CHECK(!srv.tierp()->entries.empty());
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
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
            for (int i = 0; i < 400 && t->stats.ssd_writes < t->stats.evict_writes; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            CHECK(srv.engine().stats().spe_hits >= 5);
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
            CHECK_EQ(srv.model().poison(), 0u);
        }
        if (mode == 0 && sys_id != 0) {
            // restart: the system checkpoint is on the SSD; a new conversation with the same system message reuses it
            TestServer srv(su);
            g_log.clear();
            const tier::Entry* e = srv.tierp()->find(sys_id);
            CHECK(e != nullptr);
            const auto p = cb.conv(2000, 951, {{300, 990}});
            auto r = srv.post("/v1/completions", completionBody(p, 16));
            CHECK_EQ(completionText(r), decode(srv.model().greedyReference(p, 16, true)));
            CHECK(cachedTokens(r) >= sys_n);
            CHECK(srv.tierp()->stats.restores_ssd >= 1);
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
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
        CHECK(waitFor([&] { return t->stats.idle_writes >= 1 && !t->idleWriting(); }, 10000));
        CHECK(g_log.count("idle write done") >= 1);
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
        // staying idle (5 idle delays): no second pass, no write
        const std::uint64_t passes = t->stats.idle_passes, writes = t->stats.ssd_writes;
        const std::size_t fired = g_log.count("idle for ");
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        CHECK_EQ(t->stats.idle_passes, passes);
        CHECK_EQ(t->stats.ssd_writes, writes);
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
            for (const tier::Entry* e : t->entries)
                if (e->io_busy) return false;
            return true;
        }, 20000));
        tier::g_write_sim.store(nullptr);
        CHECK_EQ(t->stats.ssd_writes, t->stats.idle_writes);
        if (const tier::Entry* s = t->find(sess_id)) CHECK(s->ssd.has_value() && s->ssd->n_tok >= p3.size() - 64);
        for (const tier::Entry* e : t->entries)
            if (!e->reused) CHECK(!e->ssd.has_value());
        std::fprintf(stderr, "  (idle_write: request latency %.1f ms with no idle write, %.1f ms while one ran; idle writes %llu, passes %llu)\n",
                     ms_base, ms_mid, static_cast<unsigned long long>(t->stats.idle_writes),
                     static_cast<unsigned long long>(t->stats.idle_passes));
        CHECK(ms_mid < ms_base + 2000.0);  // not held behind the slowed write
        CHECK_EQ(srv.model().poison(), 0u);
        // the close flush below does not rewrite it (clean)
    }
    {
        TestServer srv(su);
        g_log.clear();
        CHECK(srv.tierp()->find(sess_id) != nullptr);
        const auto p4 = makePrompt(3900, 607, p3);
        auto r = ask(srv, p4);
        CHECK(cachedTokens(r) >= p2.size());
        CHECK(srv.tierp()->stats.restores_ssd >= 1);
        CHECK_EQ(srv.model().poison(), 0u);
    }
    std::filesystem::remove_all(dir, ec);
}

// CLOSE-REUSED (owner decision): in --kv-ssd-mode shutdown the flush when WHIRL is closed (graceful stop) writes
// only reused entries, like the RAM-eviction and idle writes: a one-shot prompt never reaches the SSD. Mock engine
// end to end: one one-shot prompt + a 2-turn session, idle write off; on the graceful stop only the session is
// written; the restart restores it from SSD and the one-shot prompt is a cold prefill (exact output both times).
void testCloseReused() {
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
        for (const tier::Entry* e : t->entries) {
            if (e->tokens.size() >= p2.size() && std::equal(p2.begin(), p2.end(), e->tokens.begin())) sess_id = e->id;
            if (e->tokens.size() >= p0.size() && std::equal(p0.begin(), p0.end(), e->tokens.begin())) one_id = e->id;
        }
        CHECK(sess_id != 0 && one_id != 0);
        CHECK_EQ(t->stats.ssd_writes, 0u);  // nothing while running
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
        CHECK_EQ(t->ssdEntries(), 1u);
        CHECK(t->find(sess_id) != nullptr);
        CHECK(t->find(one_id) == nullptr);
        const auto p3 = makePrompt(3600, 704, p2);
        auto r = ask(srv, p3);
        CHECK(cachedTokens(r) >= p2.size());  // the reused session comes back from SSD
        CHECK(t->stats.restores_ssd >= 1);
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
                 static_cast<unsigned long long>(srv.tierp()->stats.restores_ram), static_cast<unsigned long long>(srv.tierp()->stats.superseded),
                 static_cast<unsigned long long>(srv.tierp()->stats.demoted));
    CHECK(cachedTokens(r) >= main1.size());
    CHECK(srv.tierp()->stats.restores_ram >= 1);
    CHECK_EQ(srv.tierp()->stats.superseded, 0u);
    if (supersede) CHECK(srv.tierp()->stats.demoted >= 1);
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
    const std::uint64_t restores_b = srv.tierp()->stats.restores_ram;
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
    CHECK(g_log.count("superseded (compact / compress): cut") >= 1);
    CHECK(srv.tierp()->stats.superseded >= 1);
    CHECK_EQ(srv.model().poison(), 0u);
}

// FIX-RC4-A P2-1 (RESTORE-EXC): a device copy throwing while a restore is under way -- in the first pump
// inside Tier::beginRestore (RAM restore: every piece is copied there) or in a later pump (SSD restore) --
// fails the restore cleanly through the engine: the request is prefilled cold (same output), no entry
// stays pinned, nothing stays loading / busy, and the entry restores again afterwards.
void testRestoreThrows() {
    std::filesystem::path base;
    if (const char* t = std::getenv("WHIRL_TEST_TMP"); t && *t) base = std::filesystem::path(widen(t));
    else base = std::filesystem::temp_directory_path() / L"whirl-tests";
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
            for (const tier::Entry* e : srv.tierp()->entries) busy = busy || e->io_busy || e->loading || e->pin > 0;
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
        CHECK_EQ(srv.tierp()->stats.restores_ram, 0u);
        for (int i = 0; i < 400 && srv.tierp()->ssdEntries() < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK_EQ(srv.model().poison(), 0u);
    }
    {
        // restart: A's turn restores from SSD; a later restore copy throws (first pump or pollRestores)
        TestServer srv(su);
        CHECK(srv.tierp()->ssdEntries() >= 1);
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
    const auto& st = srv.tierp()->stats;
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

int main(int argc, char** argv) {
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
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
        {"tiers", testTiers},
        {"shutdown_evict_sys", testShutdownEvictSys},
        {"idle_write", testIdleWrite},
        {"close_reused", testCloseReused},
        {"restore_conc", testRestoreConcurrent},
        {"restore_throws", testRestoreThrows},
        {"tier_inplace", testTierInPlaceRegen},
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
        {"fr_protocol", testFrProtocol},
        {"fr_stop_tokens", testFrStopTokens},
        {"fr_log_injection", testFrLogInjection},
        {"fr_short_followup", testFrShortFollowUp},
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
        try {
            t.fn();
        } catch (const std::exception& ex) {
            ++g_fail;
            std::fprintf(stderr, "  FAIL %s: exception %s\n", t.name, ex.what());
        }
        std::printf("%-18s %s (%d checks, %.1f s)\n", t.name, g_fail == f0 ? "ok" : "FAILED", g_pass - p0 + g_fail - f0,
                    nowSeconds() - t0);
        std::fflush(stdout);
    }
    std::printf("server tests: %d passed, %d failed\n", g_pass, g_fail);
    Log::setHook(nullptr, nullptr);
    if (g_pass + g_fail == 0) {
        std::printf("server tests: 0 checks ran: refusing to report success\n");
        return 2;
    }
    return g_fail ? 1 : 0;
}
