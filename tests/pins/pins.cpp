// CPU pins for the arch024 refactor (S0a; PLAN v3 (b) P-cpu, S3 / S4 / S8). CPU only, no GPU, no HIP runtime.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Recorded once on the B0 base (06a1540, before S1) and compared byte for byte at every later
// step (tests/pins/*.txt; approval style: a mismatch fails, --record is explicit). Uses only interfaces that exist at
// 06a1540 (Engine, HttpServer, MockModel, NullBackend, chat::*, Tokenizer, device::Backend), so the same file builds
// on the base worktree.
//
//   whirl-pins tokens --gguf MODEL.gguf --requests tests/pins/requests.jsonl
//       prompt token pin (A7): each chat request -> chat::parseChatRequest -> chat::renderChat (the GGUF's template
//       kind) -> Tokenizer::encode(prompt, true), as http.cpp buildJob does; one line per request: token count and
//       FNV-1a 64 of the ids. Reads the GGUF's tokenizer metadata only (no weights).
//   whirl-pins engine --gguf MODEL.gguf --kv f16|q8h|q4
//       the Engine's device-memory sequence over MockModel + NullBackend (wrapped by a recording Backend): every
//       malloc / hostMalloc / free in order (CkptStore allocation sequence, M2 / S3) and every copyAsync with its
//       source and destination as (allocation ordinal, offset) and its length (KvArr page copies, M3 / S4). --kv sets
//       the pool arrays per attention layer: f16 = K, V; q8h / q4 = K, V, K scales, V scales (row bytes below; the
//       mock's K / V rows are its own 24 / 8 bytes). A fixed sequence of chat requests (shared system prompt, second
//       user, multi-turn continuation) is sent one at a time; no tier (no IO thread), so the order is fixed. The system prompt is above EngineOptions::sys_min (2048 tokens), so a
//       shared prefix entry is made and attached (page copies).
//   whirl-pins engine --gguf MODEL.gguf --kv f16|q8h|q4 --tier   (arch024-S6b, recorded on 906e367 before the move)
//       the same with a 64 MiB RAM tier (min 256 tokens, no SSD tier) and three more requests (a long session, its
//       compacted form and the next turn of that): the spill copies into the arena (TierBridge span sequence, M5), the tier's allocations, and a
//       last line with the tier counters (spills, superseded, demoted).
//   whirl-pins engine --gguf MODEL.gguf --kv f16|q8h|q4 --restore   (arch024-S7, recorded on 869e56c before the move)
//       --tier plus five more requests: a document session under a short system prompt (below sys_min: no shared
//       prefix), two unrelated sessions that take both slots (the document's VRAM copy goes), the document again
//       and its next turn: restores from the RAM tier (TierBridge restore half: the copies from the arena into the
//       slot's pages and checkpoints, main part then tail) and the in-place spill after a restore.
//   whirl-pins engine --gguf MODEL.gguf --kv f16|q8h|q4 --gov   (arch024-S9, recorded on 511271d before the move)
//       the base requests with the memory governor on (an injected WDDM probe: usage = fixed + the NullBackend's device
//       bytes, budget set by phase; every time rule of memgov::Config at 0), then: the budget drops to a quarter of the
//       margin free while idle (free the idle prefix cache, re-allocate the KV pool smaller: govTick / govApply /
//       govFreeCache / govResize / govDrain), a request in the smaller pool, the budget back (the pool grows back to
//       its startup size: the episode ends), a last request. The governor status (/health "governor") after each phase
//       is printed after the device sequence.
// Output goes to stdout; tests/pins/check.ps1 compares it with the recorded file.

#include "http_client.h"
#include "mock_backend.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/log.h"
#include "whirl/chat.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace whirl;
using namespace whirl::server;

namespace {

std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const auto* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
    return h;
}

int tokensPin(const gguf::File& f, const Tokenizer& tok, const std::string& req_path) {
    const std::string_view tmpl_text = f.getStringOr("tokenizer.chat_template", "");
    const chat::TemplateKind kind = chat::detectTemplate(tmpl_text);
    std::printf("template %s (%zu B)\n", chat::templateName(kind), tmpl_text.size());
    std::ifstream in(req_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", req_path.c_str());
        return 2;
    }
    std::string line;
    int n = 0, bad = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        // "<name>\t<request JSON>"; the request may carry "x_allow_media": true (an mmproj is loaded)
        const auto tab = line.find('\t');
        const std::string name = tab == std::string::npos ? std::to_string(n) : line.substr(0, tab);
        const std::string body = tab == std::string::npos ? line : line.substr(tab + 1);
        ++n;
        try {
            const json::Value root = json::parse(body);
            // arch024-S8: /v1/completions rows ("prompt"), as http.cpp buildJob's completion branch
            if (const json::Value* pv = root.asObject().find("prompt")) {
                std::string prompt;
                std::vector<std::uint32_t> v;
                const json::Value* p0 = pv->isArray() && !pv->asArray().empty() ? &pv->asArray()[0] : pv;
                if (p0->isString()) {
                    prompt = p0->asString();
                    const auto ids = tok.encode(prompt, true);
                    v.assign(ids.begin(), ids.end());
                } else {
                    for (const json::Value& it : pv->asArray()) v.push_back(static_cast<std::uint32_t>(it.asInt()));
                }
                std::printf("req %s tokens %zu fnv %016llx prompt_bytes %zu prompt_fnv %016llx\n", name.c_str(), v.size(),
                            static_cast<unsigned long long>(fnv(v.data(), v.size() * 4)), prompt.size(),
                            static_cast<unsigned long long>(fnv(prompt.data(), prompt.size())));
                continue;
            }
            chat::ChatRequest req = chat::parseChatRequest(root);
            const json::Value* am = root.asObject().find("x_allow_media");
            req.options.allow_media = am != nullptr && am->isBool() && am->asBool();
            const std::string prompt = chat::renderChat(kind, req.messages, req.tools, req.options);
            const auto ids = tok.encode(prompt, true);
            std::vector<std::uint32_t> v(ids.begin(), ids.end());
            std::printf("req %s tokens %zu fnv %016llx prompt_bytes %zu prompt_fnv %016llx\n", name.c_str(), v.size(),
                        static_cast<unsigned long long>(fnv(v.data(), v.size() * 4)), prompt.size(),
                        static_cast<unsigned long long>(fnv(prompt.data(), prompt.size())));
        } catch (const std::exception& ex) {
            std::printf("req %s error %s\n", name.c_str(), ex.what());
            ++bad;
        }
    }
    std::printf("requests %d errors %d\n", n, bad);
    return n > 0 && bad == 0 ? 0 : 1;
}

// device::Backend that forwards to a NullBackend and records the memory sequence, addresses as allocation ordinals
class RecBackend final : public device::Backend {
public:
    explicit RecBackend(device::NullBackend& n) : n_(n) {}
    void copyAsync(device::DevPtr dst, device::DevPtr src, std::uint64_t bytes, device::Stream s) override {
        n_.copyAsync(dst, src, bytes, s);
        if (bytes == 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (on_) lines_.push_back("C " + where(dst) + " <- " + where(src) + " n " + std::to_string(bytes) + " s" + streamId(s));
    }
    void download(void* dst, device::DevPtr src, std::uint64_t bytes) override { n_.download(dst, src, bytes); }
    device::Stream streamCreateNonBlocking() override { return n_.streamCreateNonBlocking(); }
    void streamDestroy(device::Stream s) override { n_.streamDestroy(s); }
    void streamSync(device::Stream s) override { n_.streamSync(s); }
    void streamWaitEvent(device::Stream s, device::Event e) override { n_.streamWaitEvent(s, e); }
    device::Event eventCreate() override { return n_.eventCreate(); }
    void eventDestroy(device::Event e) override { n_.eventDestroy(e); }
    void eventRecord(device::Event e, device::Stream s) override { n_.eventRecord(e, s); }
    bool eventDone(device::Event e) override { return n_.eventDone(e); }
    bool eventDoneChecked(device::Event e) override { return n_.eventDoneChecked(e); }
    void eventSync(device::Event e) override { n_.eventSync(e); }
    void* hostMalloc(std::uint64_t bytes) override {
        void* p = n_.hostMalloc(bytes);
        add(reinterpret_cast<std::uint64_t>(p), bytes, "H");
        return p;
    }
    void hostFree(void* p) override {
        drop(reinterpret_cast<std::uint64_t>(p), "FH");
        n_.hostFree(p);
    }
    device::DevPtr malloc(std::uint64_t bytes) override {
        const device::DevPtr p = n_.malloc(bytes);
        add(p, bytes, "M");
        return p;
    }
    void free(device::DevPtr p) override {
        drop(p, "F");
        n_.free(p);
    }
    void setDevice(int d) override { n_.setDevice(d); }
    const char* lastErrorString() const override { return n_.lastErrorString(); }
    hip::MemInfo memInfo() override { return n_.memInfo(); }
    hip::WddmMemInfo wddmMemInfo() override { return n_.wddmMemInfo(); }
    hip::LocalMemSnap localMemSnap() override { return n_.localMemSnap(); }
    hip::UmaMem umaMemory(int device) override { return n_.umaMemory(device); }

    // phase names starting with '~' (destructors) are recorded as a sorted multiset: which buffers a destructor frees
    // is pinned, the order members are destroyed in is not (S2 / S3 turn manual frees into member destruction; F1
    // ordering is tested by the NullBackend tracking mode, not here)
    void record(bool on, const char* phase) {
        std::lock_guard<std::mutex> lk(mu_);
        if (sorted_from_ != std::string::npos) std::sort(lines_.begin() + static_cast<std::ptrdiff_t>(sorted_from_), lines_.end());
        on_ = on;
        lines_.push_back(std::string("# ") + phase + (phase[0] == '~' ? " (sorted)" : ""));
        sorted_from_ = phase[0] == '~' ? lines_.size() : std::string::npos;
    }
    std::vector<std::string> lines() const {
        std::lock_guard<std::mutex> lk(mu_);
        return lines_;
    }

private:
    struct A {
        std::uint64_t bytes, ord;
    };
    void add(std::uint64_t p, std::uint64_t bytes, const char* k) {
        std::lock_guard<std::mutex> lk(mu_);
        const std::uint64_t ord = next_++;
        live_[p] = A{bytes, ord};
        if (on_) lines_.push_back(std::string(k) + " #" + std::to_string(ord) + " " + std::to_string(bytes));
    }
    void drop(std::uint64_t p, const char* k) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = live_.find(p);
        if (on_) lines_.push_back(std::string(k) + " " + (it != live_.end() ? "#" + std::to_string(it->second.ord) : std::string("?")));
        if (it != live_.end()) live_.erase(it);
    }
    std::string where(std::uint64_t v) const {
        auto it = live_.upper_bound(v);
        if (it != live_.begin()) {
            --it;
            if (v - it->first < it->second.bytes || (v - it->first == it->second.bytes && v != it->first))
                return "#" + std::to_string(it->second.ord) + "+" + std::to_string(v - it->first);
        }
        return "?";
    }
    std::string streamId(device::Stream s) {
        auto it = streams_.find(s);
        if (it == streams_.end()) it = streams_.emplace(s, streams_.size()).first;
        return std::to_string(it->second);
    }
    device::NullBackend& n_;
    mutable std::mutex mu_;
    bool on_ = false;
    std::uint64_t next_ = 0;
    std::map<std::uint64_t, A> live_;
    std::map<device::Stream, std::size_t> streams_;
    std::vector<std::string> lines_;
    std::size_t sorted_from_ = std::string::npos;
};

std::string chatBody(const std::string& messages) {
    return "{\"messages\":" + messages + ",\"max_tokens\":8,\"temperature\":0,\"stream\":false}";
}

int enginePin(const Tokenizer& tok, const std::string& kv, bool with_tier, bool with_restore, bool with_gov) {
    test::MockConfig mc;
    mc.parallel = 2;
    mc.slot_ctx = 16384;
    mc.pool_tokens = 32768;
    mc.mtp = false;
    // scale arrays: 2 bytes (f16 scale) per 32 elements of a row; mock K row is 24 bytes -> rows of 2 bytes
    // per 32 elements of the real row are represented by these sizes (distinct, so a swapped array shows)
    if (kv == "q8h") mc.kv_extra_rows = {4, 6};
    else if (kv == "q4") mc.kv_extra_rows = {2, 10};
    else if (kv != "f16") {
        std::fprintf(stderr, "--kv f16|q8h|q4\n");
        return 2;
    }
    device::NullBackend null_ops;
    RecBackend ops(null_ops);
    ops.record(true, "model");
    auto model = std::make_unique<test::MockModel>(ops, tok, mc);
    EngineOptions eo;
    eo.parallel = mc.parallel;
    eo.ctx = mc.slot_ctx;
    eo.model_name = "mock";
    eo.use_mtp = false;
    ops.record(true, "engine");
    auto engine = std::make_unique<Engine>(*model, ops, tok, eo);
    ops.record(true, "allocState");
    engine->allocState();
    ops.record(true, "initPool");
    engine->initPool();
    // arch024-S6b: --tier adds a RAM tier (no SSD tier, so the IO thread copies nothing): the spill copies (span
    // sequence of ckSpans / pageSpans into the arena, M5 TierBridge) and the tier's own allocations are pinned too
    std::unique_ptr<tier::Tier> tr;
    if (with_tier) {
        ops.record(true, "tier");
        tr = tier::Tier::create(ops, 64ull << 20, 42, 0);
        tier::Config tc;
        tc.ram_bytes = 64ull << 20;
        tc.min_tokens = 256;
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
        engine->attachTier(tr.get());
        tr->start(tc, lay, 7);
    }
    // arch024-S9: --gov, the memory governor over an injected WDDM probe (deterministic: each budget change waits for
    // the governor's action before the next request; idle samples every 20 ms)
    constexpr std::uint64_t kFixed = 20000ull << 20;  // "the weights" (not allocated by the mock)
    std::atomic<std::uint64_t> budget{0};
    std::uint64_t local0 = 0, gM = 0;
    std::uint32_t pool_start = 0;
    if (with_gov) {
        std::uint64_t per_tok = 0;
        for (const KvArr& a : model->kvArrays()) per_tok += a.row;
        pool_start = model->poolPages() * qwen35::kv_page;
        gM = per_tok * pool_start / 2;
        local0 = kFixed + null_ops.deviceBytes();
        budget = local0 + gM;  // exactly the margin free, as the startup sizing leaves it
        wddmwatch::Config vc;
        vc.idle_sample_s = 0.02;
        vc.governor = true;
        engine->setVramProbe(
            [&] {
                wddmwatch::Sample x;
                x.ok = true;
                x.local_usage = kFixed + null_ops.deviceBytes();
                x.local_budget = budget.load();
                x.nonlocal_usage = 400ull << 20;
                return x;
            },
            true, std::nullopt, vc, false, nullptr);
        memgov::Config gc;
        gc.per_tok = per_tok;
        gc.page = qwen35::kv_page;
        gc.pool_start = pool_start;
        gc.pool_min = 4096;
        gc.ctx = mc.slot_ctx;
        gc.recover_s = 0.0;
        gc.hold_s = 0.0;
        gc.settle_s = 0.0;
        gc.trust_s = 0.0;
        gc.osc_window_s = 0.0;
        engine->setGovernor(gc, [gM](std::uint64_t) { return gM; }, 0);
    }
    std::vector<std::string> gov_lines;
    auto govWait = [&](const char* what, const std::string& needle) {
        for (int i = 0; i < 500; ++i) {
            if (engine->governorJson().find(needle) != std::string::npos) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        engine->runOnLoop([] {}, std::chrono::milliseconds(10000));
        gov_lines.push_back(std::string("gov ") + what + ": " + engine->governorJson());
    };
    auto http = std::make_unique<HttpServer>(*engine);
    http->start("127.0.0.1", 0);
    std::thread loop([&] { engine->runLoop(); });
    const std::string sys = "{\"role\":\"system\",\"content\":\"You are a careful assistant. Answer in one short sentence. "
                            "Keep code blocks in English, prose in the user's language.\"}";
    std::string sysl;
    for (int i = 0; i < 300; ++i) sysl += " Rule " + std::to_string(i) + ": be precise and brief.";
    const std::string sys2 = "{\"role\":\"system\",\"content\":\"You are a careful assistant." + sysl + "\"}";
    const std::vector<std::pair<const char*, std::string>> reqs = {
        {"sys+u1", "[" + sys2 + ",{\"role\":\"user\",\"content\":\"請用一句話說明 Python 的 list comprehension。\"}]"},
        {"sys+u2", "[" + sys2 + ",{\"role\":\"user\",\"content\":\"Write a C++ function that reverses a string.\"}]"},
        {"sys+u1+turn2", "[" + sys2 + ",{\"role\":\"user\",\"content\":\"請用一句話說明 Python 的 list comprehension。\"},"
                         "{\"role\":\"assistant\",\"content\":\"It builds a list from an iterable in one expression.\"},"
                         "{\"role\":\"user\",\"content\":\"再給一個例子。\"}]"},
        {"short", "[" + sys + ",{\"role\":\"user\",\"content\":\"hi\"}]"},
        {"sys+u3", "[" + sys2 + ",{\"role\":\"user\",\"content\":\"What is a mutex?\"}]"},
    };
    // --tier: three more requests after these: a long session (an abandoned tail over SupersedeRule::min_tail), its
    // compacted form (same system prompt and first user message, much shorter) and the next turn of that, so the superseded-session rule marks the old entry on the first
    // spill and applies (cut / demote) on the second (CACHE-1)
    std::vector<std::pair<const char*, std::string>> all = reqs;
    if (with_tier) {
        const std::string u1 = "{\"role\":\"user\",\"content\":\"請用一句話說明 Python 的 list comprehension。\"}";
        std::string longa;
        for (int i = 0; i < 700; ++i) longa += " Step " + std::to_string(i) + ": the list grows by one item here.";
        all.push_back({"long", "[" + sys2 + "," + u1 + ",{\"role\":\"assistant\",\"content\":\"" + longa + "\"},"
                               "{\"role\":\"user\",\"content\":\"繼續。\"}]"});
        all.push_back({"compact", "[" + sys2 + "," + u1 + ",{\"role\":\"assistant\",\"content\":\"(summary) list comprehension.\"},"
                                  "{\"role\":\"user\",\"content\":\"繼續。\"}]"});
        all.push_back({"compact+turn2", "[" + sys2 + "," + u1 + ",{\"role\":\"assistant\",\"content\":\"(summary) list comprehension.\"},"
                                        "{\"role\":\"user\",\"content\":\"繼續。\"},{\"role\":\"assistant\",\"content\":\"OK.\"},"
                                        "{\"role\":\"user\",\"content\":\"再一個。\"}]"});
    }
    // --restore (arch024-S7): a document session, two unrelated sessions that take both slots, then the document
    // again and its next turn (restores from the RAM tier; the spill after a restore is in place)
    if (with_restore) {
        std::string doc;
        for (int i = 0; i < 260; ++i) doc += " Section " + std::to_string(i) + ": the parser keeps a stack of open scopes.";
        const std::string du = "{\"role\":\"user\",\"content\":\"Summarize this design note:" + doc + "\"}";
        const std::string dsys = "{\"role\":\"system\",\"content\":\"You review design notes.\"}";
        auto filler = [](const char* word, int salt) {
            std::string f;
            for (int i = 0; i < 150; ++i) f += std::string(" ") + word + " " + std::to_string(i * salt) + " is listed.";
            return f;
        };
        all.push_back({"doc", "[" + dsys + "," + du + "]"});
        all.push_back({"fillA", "[{\"role\":\"system\",\"content\":\"You count apples.\"},{\"role\":\"user\",\"content\":\"" +
                                    filler("apple", 3) + "\"}]"});
        all.push_back({"fillB", "[{\"role\":\"system\",\"content\":\"You count pears.\"},{\"role\":\"user\",\"content\":\"" +
                                    filler("pear", 7) + "\"}]"});
        all.push_back({"doc-again", "[" + dsys + "," + du + "]"});
        all.push_back({"doc+turn2", "[" + dsys + "," + du + ",{\"role\":\"assistant\",\"content\":\"A scope stack.\"},"
                                    "{\"role\":\"user\",\"content\":\"再詳細一點。\"}]"});
    }
    int bad = 0;
    for (const auto& [name, msgs] : all) {
        // --tier: the previous request's spill (finishJob, after the reply) is done before the next phase mark
        if (with_tier) engine->runOnLoop([] {}, std::chrono::milliseconds(10000));
        ops.record(true, name);
        const auto res = test::httpRequest("127.0.0.1", http->port(), "POST", "/v1/chat/completions", chatBody(msgs));
        if (res.status != 200) {
            std::printf("request %s: HTTP %d %s\n", name, res.status, res.err.c_str());
            ++bad;
        }
    }
    if (with_tier) engine->runOnLoop([] {}, std::chrono::milliseconds(10000));
    if (with_gov) {
        auto post = [&](const char* name, const std::string& msgs) {
            ops.record(true, name);
            const auto res = test::httpRequest("127.0.0.1", http->port(), "POST", "/v1/chat/completions", chatBody(msgs));
            if (res.status != 200) {
                std::printf("request %s: HTTP %d %s\n", name, res.status, res.err.c_str());
                ++bad;
            }
        };
        govWait("start", "\"shrinks\":0");
        // the budget drops to a quarter of the margin free while every slot is idle: the pool is re-allocated smaller
        ops.record(true, "gov-squeeze");
        budget = local0 + gM / 4;
        govWait("squeeze", "\"shrinks\":1");
        post("gov-small", "[" + sys + ",{\"role\":\"user\",\"content\":\"Explain RAII in one sentence.\"}]");
        govWait("small", "\"shrinks\":1");
        // the budget back: the pool grows to its startup size, the episode ends
        ops.record(true, "gov-recover");
        budget = local0 + gM;
        govWait("recover", "\"pool_tokens\":" + std::to_string(pool_start) + ",");
        post("gov-after", "[" + sys + ",{\"role\":\"user\",\"content\":\"What is a mutex?\"}]");
        govWait("after", "\"grows\":1");
    }
    ops.record(true, "stop");
    engine->stop();
    loop.join();
    http->stop();
    http.reset();
    if (tr) {
        const tier::Stats& ts = tr->stats;
        std::printf("tier: spills %llu spill_bytes %llu restores_ram %llu restore_bytes %llu superseded %llu demoted %llu entries %zu\n",
                    static_cast<unsigned long long>(ts.spills), static_cast<unsigned long long>(ts.spill_bytes),
                    static_cast<unsigned long long>(ts.restores_ram), static_cast<unsigned long long>(ts.restore_bytes),
                    static_cast<unsigned long long>(ts.superseded), static_cast<unsigned long long>(ts.demoted),
                    tr->countRam());
        engine->attachTier(nullptr);
        ops.record(true, "~Tier");
        tr.reset();
    }
    ops.record(true, "~Engine");
    engine.reset();
    ops.record(true, "~MockModel");
    model.reset();
    ops.record(false, "end");
    std::size_t copies = 0, allocs = 0;
    for (const std::string& l : ops.lines()) {
        std::printf("%s\n", l.c_str());
        if (l.rfind("C ", 0) == 0) ++copies;
        if (l.rfind("M ", 0) == 0 || l.rfind("H ", 0) == 0) ++allocs;
    }
    for (const std::string& l : gov_lines) std::printf("%s\n", l.c_str());
    const bool clean = null_ops.clean();
    std::printf("kv %s: allocations %zu copies %zu errors %d backend %s\n", kv.c_str(), allocs, copies, bad, clean ? "clean" : "LEAK");
    return bad == 0 && clean && copies > 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: whirl-pins tokens --gguf F --requests F | engine --gguf F --kv f16|q8h|q4\n");
        return 2;
    }
    const std::string cmd = argv[1];
    std::string gguf_path, req_path, kv = "f16";
    bool with_tier = false, with_restore = false, with_gov = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--requests" && i + 1 < argc) req_path = argv[++i];
        else if (a == "--kv" && i + 1 < argc) kv = argv[++i];
        else if (a == "--tier") with_tier = true;
        else if (a == "--restore") with_tier = with_restore = true;
        else if (a == "--gov") with_gov = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf_path.empty()) {
        std::fprintf(stderr, "--gguf is required\n");
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
    if (cmd == "tokens") {
        if (req_path.empty()) return 2;
        return tokensPin(*file, *tok, req_path);
    }
    if (cmd == "engine") return enginePin(*tok, kv, with_tier, with_restore, with_gov);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
