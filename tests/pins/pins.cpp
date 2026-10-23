// CPU pins for the arch024 refactor (S0a; PLAN v3 (b) P-cpu, S3 / S4 / S8). CPU only, no GPU, no HIP runtime.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Recorded once on the B0 base (06a1540, before S1) and compared byte for byte at every later
// step (tests/pins/*.txt; approval style: a mismatch fails, --record is explicit). Uses only interfaces that exist at
// 06a1540 (Engine, HttpServer, MockModel, NullBackend, chat::*, Tokenizer, device::Backend), so the same file builds
// on the base worktree. Exception (arch024-S8): the tokens mode calls server::chatPrompt / completionTokens, which
// exist from S8 on; the base recording ran an inline copy of the same steps (scratch\s8\base_pins.cpp).
//
//   whirl-pins tokens --gguf MODEL.gguf --requests tests/pins/requests.jsonl
//       prompt token pin (A7): each chat request -> chat::parseChatRequest -> server::chatPrompt (http.cpp: media
//       check, chat::renderChat with the GGUF's template kind, Tokenizer::encode(prompt, true)), the code buildJob runs;
//       completion rows -> server::completionTokens; one line per request: token count and
//       FNV-1a 64 of the ids. Reads the GGUF's tokenizer metadata only (no weights).
//   whirl-pins engine --gguf MODEL.gguf --kv f16|q8h|q4
//       the Engine's device-memory sequence over MockModel + NullBackend (wrapped by a recording Backend): every
//       malloc / hostMalloc / free in order (CkptStore allocation sequence, M2 / S3) and every copyAsync with its
//       source and destination as (allocation ordinal, offset) and its length (KvArr page copies, M3 / S4). --kv sets
//       the pool arrays per attention layer: f16 = K, V; q8h / q4 = K, V, K scales, V scales (row bytes below; the
//       mock's K / V rows are its own 24 / 8 bytes). A fixed sequence of chat requests (shared system prompt, second
//       user, multi-turn continuation) is sent one at a time; no tier (no IO thread), so the order is fixed. The system prompt is above EngineOptions::sys_min (2048 tokens), so a
//       shared prefix entry is made and attached (page copies).
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
            // /v1/completions rows ("prompt"): the product's completionTokens (http.cpp buildJob, arch024-S8); the
            // printed prompt bytes are the request's string (none for token ids)
            if (const json::Value* pv = root.asObject().find("prompt")) {
                const json::Value* p0 = pv->isArray() && !pv->asArray().empty() ? &pv->asArray()[0] : pv;
                const std::string prompt = p0->isString() ? p0->asString() : std::string();
                const std::vector<std::uint32_t> v = completionTokens(tok, static_cast<std::uint32_t>(tok.size()), root.asObject());
                std::printf("req %s tokens %zu fnv %016llx prompt_bytes %zu prompt_fnv %016llx\n", name.c_str(), v.size(),
                            static_cast<unsigned long long>(fnv(v.data(), v.size() * 4)), prompt.size(),
                            static_cast<unsigned long long>(fnv(prompt.data(), prompt.size())));
                continue;
            }
            // chat rows: the product's chatPrompt (http.cpp buildJob, arch024-S8): media check, render, encode
            chat::ChatRequest req = chat::parseChatRequest(root);
            const json::Value* am = root.asObject().find("x_allow_media");
            const ChatPrompt cp = chatPrompt(tok, kind, req, am != nullptr && am->isBool() && am->asBool());
            const std::string& prompt = cp.text;
            std::vector<std::uint32_t> v(cp.ids.begin(), cp.ids.end());
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

int enginePin(const Tokenizer& tok, const std::string& kv) {
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
    int bad = 0;
    for (const auto& [name, msgs] : reqs) {
        ops.record(true, name);
        const auto res = test::httpRequest("127.0.0.1", http->port(), "POST", "/v1/chat/completions", chatBody(msgs));
        if (res.status != 200) {
            std::printf("request %s: HTTP %d %s\n", name, res.status, res.err.c_str());
            ++bad;
        }
    }
    ops.record(true, "stop");
    engine->stop();
    loop.join();
    http->stop();
    http.reset();
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
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--requests" && i + 1 < argc) req_path = argv[++i];
        else if (a == "--kv" && i + 1 < argc) kv = argv[++i];
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
    if (cmd == "engine") return enginePin(*tok, kv);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
