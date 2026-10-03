// Server tests: protocol / sampling unit tests and end-to-end HTTP tests of
// the engine against the host-memory mock model (no GPU).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-server-tests [--gguf TOKENIZER.gguf] [--only NAME[,NAME...]] [--verbose]
// The GGUF only provides the tokenizer (vocabulary + merges); default: the
// Qwen3.8-27B UD-Q4_K_M file of this machine, or WHIRL_TEST_GGUF.

#include "http_client.h"
#include "mock_backend.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/log.h"
#include "server/protocol.h"
#include "server/tokens.h"
#include "tier/kv_tier.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"

#include <atomic>
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
    std::uint64_t fingerprint = 0x1234;
    // vision: attach a host mock encoder (image_url parts accepted)
    bool vision = false;
    bool allow_files = false;
    std::uint32_t vis_ck_min = 1024;
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
    std::uint64_t vramFree() const override { return 0; }
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
        http_ = std::make_unique<HttpServer>(*engine_);
        http_->start("127.0.0.1", 0);
        loop_ = std::thread([this] { engine_->runLoop(); });
    }
    ~TestServer() { shutdown(); }
    void shutdown() {
        if (!engine_) return;
        engine_->stop();
        if (loop_.joinable()) loop_.join();
        http_->stop();
        http_.reset();
        engine_->attachTier(nullptr);
        engine_.reset();
        tier_.reset();
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
    auto opt = test::httpRequest("127.0.0.1", srv.port(), "OPTIONS", "/v1/chat/completions");
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

// concurrent requests (more than slots): every output equals its reference
void testConcurrency(std::uint32_t parallel, std::uint32_t n_req, bool mtp) {
    ServerSetup su = baseSetup(mtp);
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
        // let the SSD writes finish (delay 0)
        for (int i = 0; i < 400 && srv.tierp()->ssdEntries() < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(srv.tierp()->ssdEntries() >= 2);
        CHECK_EQ(srv.model().poison(), 0u);
        CHECK_EQ(srv.model().hidMismatch(), 0u);
    }
    {
        // restart: the SSD index brings the entries back; A's turn restores from SSD
        TestServer srv(su);
        g_log.clear();
        CHECK(srv.tierp()->ssdEntries() >= 2);
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

int main(int argc, char** argv) {
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
        {"protocol", testProtocol},
        {"sampler", testSampler},
        {"chat_tokens", testChatTokens},
        {"endpoints", testEndpoints},
        {"compat_endpoints", testCompatEndpoints},
        {"greedy_reference", testGreedyMatchesReference},
        {"stream", testStreamEqualsNonStream},
        {"prefix_cache", testPrefixCacheMultiTurn},
        {"concurrency", [] { testConcurrency(4, 12, true); }},
        {"concurrency16", [] { testConcurrency(16, 24, true); }},
        {"concurrency_nomtp", [] { testConcurrency(4, 8, false); }},
        {"pool", testPoolPressure},
        {"sys_prompt", testSharedSystemPrompt},
        {"tiers", testTiers},
        {"restore_conc", testRestoreConcurrent},
        {"sampling", testSampling},
        {"stop_eos", testStopAndEos},
    };
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
    return g_fail ? 1 : 0;
}
