// CPU tests of the Windows launcher's decision logic (src/launcher/launcher.h): no GPU, no console, no files.
// SPDX-License-Identifier: Apache-2.0
//
// Covers: when the launcher runs (and the rule that a scripted start never prompts), the settings file round trip
// and its errors, the answer parsing, the GPU / model / mode / context / port recommendations, the server command
// line and its quoting, the tray menu, the chat request and stream parsing.

#include "launcher/launcher.h"
#include "whirl/json.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace whirl;
using namespace whirl::launcher;

namespace {

int g_fail = 0, g_pass = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (cond) {                                                     \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                           \
    do {                                                                         \
        const auto va_ = (a);                                                    \
        const auto vb_ = (b);                                                    \
        if (va_ == vb_) {                                                        \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::printf("FAIL %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b);   \
        }                                                                        \
    } while (0)

constexpr std::uint64_t GiB = 1024ull * 1024 * 1024;

// ---------------------------------------------------------------------------
// 1. start detection and "never prompt when scripted"

void testDecideStart() {
    StartFacts dbl{0, true, true, 1, false};  // Explorer double-click: our own console
    CHECK(decideStart(dbl) == Start::launcher);

    // every single deviation from the double-click facts is classic (no prompt, no tray)
    StartFacts f = dbl;
    f.n_args = 1;
    CHECK(decideStart(f) == Start::classic);  // any argument
    f = dbl;
    f.console_processes = 2;
    CHECK(decideStart(f) == Start::classic);  // typed in cmd / pwsh, or a .bat / script
    f.console_processes = 5;
    CHECK(decideStart(f) == Start::classic);
    f = dbl;
    f.console_processes = 0;
    CHECK(decideStart(f) == Start::classic);  // no console (service, Task Scheduler, DETACHED_PROCESS)
    f = dbl;
    f.stdin_console = false;
    CHECK(decideStart(f) == Start::classic);  // whirl < file, piped input
    f = dbl;
    f.stdout_console = false;
    CHECK(decideStart(f) == Start::classic);  // whirl > log.txt, | findstr
    f = dbl;
    f.no_launcher_env = true;
    CHECK(decideStart(f) == Start::classic);  // WHIRL_NO_LAUNCHER=1

    // exhaustive: launcher iff exactly the double-click facts
    int launcher_count = 0;
    for (std::size_t na : {0u, 1u, 3u})
        for (int si = 0; si < 2; ++si)
            for (int so = 0; so < 2; ++so)
                for (unsigned cp : {0u, 1u, 2u, 3u})
                    for (int env = 0; env < 2; ++env) {
                        const StartFacts g{na, si != 0, so != 0, cp, env != 0};
                        const bool want = na == 0 && si && so && cp == 1 && !env;
                        CHECK_EQ(decideStart(g) == Start::launcher, want);
                        launcher_count += decideStart(g) == Start::launcher;
                    }
    CHECK_EQ(launcher_count, 1);

    // --setup: interactive console required (a command prompt is fine), never with redirected IO
    CHECK(setupAllowed(StartFacts{1, true, true, 2, false}));
    CHECK(setupAllowed(StartFacts{1, true, true, 1, false}));
    CHECK(!setupAllowed(StartFacts{1, false, true, 2, false}));
    CHECK(!setupAllowed(StartFacts{1, true, false, 2, false}));
    CHECK(!setupAllowed(StartFacts{1, true, true, 0, false}));

    CHECK(!noLauncherEnv(nullptr));
    CHECK(!noLauncherEnv(""));
    CHECK(!noLauncherEnv("0"));
    CHECK(noLauncherEnv("1"));
    CHECK(noLauncherEnv("yes"));
}

// ---------------------------------------------------------------------------
// 2. settings

void testSettings() {
    Settings s;
    s.gpu = "pci:71:0";
    s.model = "C:\\models\\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf";
    s.mode = "fast";
    s.ctx = 65536;
    s.port = 8081;
    std::string why;
    const auto r = parseSettings(serializeSettings(s), &why);
    CHECK(r.has_value());
    CHECK(r && *r == s);
    CHECK(settingsComplete(s));

    // non-ASCII path (UTF-8), a BOM, CRLF line ends, comments, unknown keys, spaces around '='
    Settings u = s;
    u.model = "D:\\\xE6\xA8\xA1\xE5\x9E\x8B\\swift.gguf";  // D:\模型\swift.gguf
    std::string t = "\xEF\xBB\xBF" + serializeSettings(u);
    std::string crlf;
    for (char c : t) {
        if (c == '\n') crlf += '\r';
        crlf += c;
    }
    crlf += "future_key = 42\r\n; another comment\r\n";
    const auto r2 = parseSettings(crlf, &why);
    CHECK(r2 && *r2 == u);
    const auto r3 = parseSettings("version=1\ngpu = pci:3:0 \nmodel= C:\\m.gguf\n", &why);
    CHECK(r3 && r3->gpu == "pci:3:0" && r3->model == "C:\\m.gguf" && r3->mode == "balance" && r3->port == 8080 && r3->ctx == 0);

    // errors
    CHECK(!parseSettings("gpu=pci:3:0\n", &why));  // no version
    CHECK(!parseSettings("version=2\n", &why));
    CHECK(!parseSettings("version=1\nmode=turbo\n", &why));
    CHECK(why.find("mode") != std::string::npos);
    CHECK(!parseSettings("version=1\nport=0\n", &why));
    CHECK(!parseSettings("version=1\nport=70000\n", &why));
    CHECK(!parseSettings("version=1\nctx=100\n", &why));
    CHECK(!parseSettings("version=1\nctx=999999\n", &why));
    CHECK(!parseSettings("version=1\nctx=abc\n", &why));
    CHECK(!parseSettings("version=1\njust words\n", &why));
    CHECK(parseSettings("version=1\nctx=0\n", &why).has_value());
    // incomplete: parsed but not usable
    const auto inc = parseSettings("version=1\nmodel=C:\\m.gguf\n", &why);
    CHECK(inc && !settingsComplete(*inc));

    CHECK(validMode("precise") && validMode("balance") && validMode("fast") && !validMode("Balance") && !validMode(""));
}

// ---------------------------------------------------------------------------
// 3. answers and recommendations

void testAnswers() {
    CHECK(parseAnswer(std::nullopt, 3, false).kind == Answer::Kind::eof);
    CHECK(parseAnswer(std::string(""), 3, false).kind == Answer::Kind::recommended);
    CHECK(parseAnswer(std::string("   \t"), 3, true).kind == Answer::Kind::recommended);
    const Answer a2 = parseAnswer(std::string(" 2 "), 3, false);
    CHECK(a2.kind == Answer::Kind::index && a2.index == 1);
    CHECK(parseAnswer(std::string("4"), 3, false).kind == Answer::Kind::invalid);
    CHECK(parseAnswer(std::string("0"), 3, false).kind == Answer::Kind::invalid);
    CHECK(parseAnswer(std::string("x"), 3, false).kind == Answer::Kind::invalid);
    const Answer t = parseAnswer(std::string("\"C:\\my models\\a.gguf\""), 3, true);  // a dragged-in path
    CHECK(t.kind == Answer::Kind::text && t.text == "C:\\my models\\a.gguf");
    const Answer n = parseAnswer(std::string("65536"), 0, true);  // free-form number question
    CHECK(n.kind == Answer::Kind::text && n.text == "65536");
}

gpusel::GpuRecord gpu(int index, const char* name, const char* arch, int bus, std::uint64_t mib, bool integrated, bool supported) {
    gpusel::GpuRecord r;
    r.index = index;
    r.name = name;
    r.arch = arch;
    r.pci_bus = bus;
    r.total_mib = mib;
    r.fit_mib = mib;
    r.integrated = integrated;
    r.supported = supported;
    return r;
}

void testGpus() {
    // the owner's machine: HIP order 8060S first, R9700 second; the default = most GPU memory
    std::vector<gpusel::GpuRecord> g = {gpu(0, "AMD Radeon(TM) 8060S Graphics", "gfx1151", 196, 96 * 1024, true, true),
                                        gpu(1, "AMD Radeon AI PRO R9700", "gfx1201", 71, 32624, false, true)};
    auto o = gpuOptions(g);
    CHECK_EQ(o.size(), std::size_t{2});
    CHECK_EQ(o[0].pci, std::string("pci:196:0"));
    CHECK_EQ(o[1].pci, std::string("pci:71:0"));
    CHECK_EQ(recommendedIndex(o), 0);  // 96 GB carve-out > 32 GB
    CHECK(o[0].label.find("pci:196:0") != std::string::npos);
    CHECK(o[0].label.find("dedicated") != std::string::npos);
    CHECK_EQ(o[0].fit_bytes, 96ull * GiB);
    // carve-out 16 GB: the R9700 becomes the recommendation (same rule as gpusel::defaultGpu)
    g[0].fit_mib = g[0].total_mib = 16 * 1024;
    o = gpuOptions(g);
    CHECK_EQ(recommendedIndex(o), 1);
    // an unsupported GPU is never recommended even with more memory
    g.push_back(gpu(2, "Some GPU", "gfx1030", 5, 48 * 1024, false, false));
    o = gpuOptions(g);
    CHECK_EQ(recommendedIndex(o), 1);
    CHECK(o[2].label.find("not supported") != std::string::npos);
    CHECK_EQ(recommendedIndex({}), -1);
    // a PCI domain shows in front, as gpusel::pciId
    std::vector<gpusel::GpuRecord> d = {gpu(0, "X", "gfx1201", 3, 16000, false, true)};
    d[0].pci_domain = 1;
    CHECK_EQ(gpuOptions(d)[0].pci, std::string("pci:1:3:0"));
}

void testModels() {
    CHECK(isModelCandidate("C:\\m\\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf"));
    CHECK(isModelCandidate("x.GGUF"));
    CHECK(!isModelCandidate("mmproj-Swift-1.5-F16.gguf"));
    CHECK(!isModelCandidate("C:\\m\\Qwen3.8-27B-DFlash2.gguf"));
    CHECK(!isModelCandidate("notes.txt"));
    CHECK(!isModelCandidate("C:\\gguf\\readme"));

    CHECK_EQ(productRank("Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf"), 0);
    CHECK_EQ(productRank("C:\\models\\swift-1.5-qwen3.8-27b-mxfp4-a-outq6_k.gguf"), 0);
    CHECK_EQ(productRank("Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf"), 1);
    CHECK_EQ(productRank("Ornith-1.5-35B-A3B-MXFP4.gguf"), 1);
    CHECK_EQ(productRank("gpt-oss-20b-MXFP4.gguf"), 2);
    CHECK_EQ(productRank("Qwen3.8-27B-Q4_K_M.gguf"), 3);

    const std::vector<ModelFile> files = {
        {"C:\\models\\Qwen3.8-27B-Q4_K_M.gguf", 16 * GiB},
        {"C:\\models\\Ornith-1.5-35B-A3B-MXFP4.gguf", 20 * GiB},
        {"C:\\models\\mmproj-F16.gguf", 1 * GiB},
        {"C:\\models\\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf", 15 * GiB},
        {"C:\\MODELS\\swift-1.5-qwen3.8-27b-mxfp4-a-outq6_k.gguf", 15 * GiB},  // same file, other case
        {"C:\\models\\Big-122B-MXFP4.gguf", 70 * GiB},
    };
    auto r = rankModels(files, 32 * GiB);
    CHECK_EQ(r.size(), std::size_t{4});  // mmproj and the duplicate removed
    CHECK_EQ(r[0].path, files[3].path);  // the product file first
    CHECK_EQ(r[1].path, files[1].path);  // then Ornith MXFP4
    CHECK_EQ(r[2].path, files[0].path);  // then the rest that fits
    CHECK_EQ(r[3].path, files[5].path);  // does not fit in 32 GiB: last although MXFP4
    // a 12 GiB GPU: nothing above fits; the product file is still listed before the other MXFP4 file
    r = rankModels(files, 12 * GiB);
    CHECK_EQ(r[0].path, files[3].path);
    // unknown GPU memory: everything "fits"
    CHECK(modelFits(70 * GiB, 0));
    CHECK(!modelFits(33 * GiB, 32 * GiB));
}

void testModeContextPort() {
    CHECK_EQ(modeList().size(), std::size_t{3});
    CHECK_EQ(modeList()[0], std::string(recommended_mode));
    CHECK_EQ(std::string(recommended_mode), std::string("balance"));

    // R9700 32 GB, Swift 27B in balance: ~17.5 GiB fixed, q8h KV ~ 36 KiB per token (illustrative numbers)
    CtxInput in{32 * GiB, static_cast<std::uint64_t>(17.5 * GiB), 36 * 1024};
    CtxChoice c = recommendContext(in);
    CHECK(c.known && c.fits);
    // usable = 32 - 3.2 (10% margin) - 17.5 = 11.3 GiB -> 329k tokens -> capped at 262144
    CHECK_EQ(c.max, ctx_hard_max);
    CHECK_EQ(c.recommended, ctx_default_cap);  // 80% of max is above the server's default per request
    CHECK_EQ(c.max % 4096, 0u);
    CHECK_EQ(c.recommended % 4096, 0u);
    // a 16 GB card with a 12 GiB fixed part: tight
    c = recommendContext(CtxInput{16 * GiB, 12 * GiB, 64 * 1024});
    // usable = 16 - 1.6 - 12 = 2.4 GiB -> 39321 tokens -> 36864; recommended 80% -> 29491 -> 28672
    CHECK(c.fits);
    CHECK_EQ(c.max, 36864u);
    CHECK_EQ(c.recommended, 28672u);
    CHECK(c.recommended <= c.max);
    // the 1.5 GiB minimum margin on a small card
    c = recommendContext(CtxInput{8 * GiB, 6 * GiB, 64 * 1024});
    // usable = 8 - 1.5 - 6 = 0.5 GiB -> 8192 tokens
    CHECK_EQ(c.max, 8192u);
    CHECK(c.fits);
    // does not fit
    c = recommendContext(CtxInput{16 * GiB, 15 * GiB, 64 * 1024});
    CHECK(c.known && !c.fits && c.recommended == 0);
    // unknown inputs: the server's default
    c = recommendContext(CtxInput{0, 1, 1});
    CHECK(!c.known && c.recommended == 0);
    c = recommendContext(CtxInput{GiB, 1, 0});
    CHECK(!c.known);
    // monotonic: more memory never recommends less
    std::uint32_t prev = 0;
    for (std::uint64_t m = 16; m <= 128; m += 4) {
        const CtxChoice x = recommendContext(CtxInput{m * GiB, 14 * GiB, 48 * 1024});
        CHECK(x.recommended >= prev);
        CHECK(x.recommended <= x.max || !x.fits);
        prev = x.recommended;
    }

    CHECK_EQ(parseCtx("32768").value_or(0), 32768u);
    CHECK_EQ(parseCtx("32k").value_or(0), 32768u);
    CHECK_EQ(parseCtx(" 128K ").value_or(0), 131072u);
    CHECK(!parseCtx("0"));
    CHECK(!parseCtx("abc"));
    CHECK(!parseCtx("12m"));
    CHECK(!parseCtx("99999999999"));
    CHECK_EQ(clampCtx(100), ctx_min);
    CHECK_EQ(clampCtx(1000000), ctx_hard_max);
    CHECK_EQ(clampCtx(50000), 49152u);

    CHECK_EQ(recommendPort([](std::uint16_t) { return true; }), std::uint16_t{8080});
    CHECK_EQ(recommendPort([](std::uint16_t p) { return p != 8080 && p != 8081; }), std::uint16_t{8082});
    CHECK_EQ(recommendPort([](std::uint16_t) { return false; }), std::uint16_t{8080});
    CHECK_EQ(parsePort("8080").value_or(0), std::uint16_t{8080});
    CHECK(!parsePort("0") && !parsePort("65536") && !parsePort("80a") && !parsePort(""));
}

// ---------------------------------------------------------------------------
// 4. server command line

void testServerArgs() {
    Settings s;
    s.gpu = "pci:71:0";
    s.model = "C:\\models\\Swift.gguf";
    s.mode = "balance";
    s.ctx = 65536;
    s.port = 8080;
    const std::vector<std::string> want = {"serve", "C:\\models\\Swift.gguf", "--device", "pci:71:0", "--balance", "--ctx-per-slot", "65536", "--port", "8080"};
    CHECK(serverArgs(s) == want);
    s.ctx = 0;
    s.mode = "precise";
    s.host = "0.0.0.0";
    const std::vector<std::string> want2 = {"serve", "C:\\models\\Swift.gguf", "--device", "pci:71:0", "--precise", "--host", "0.0.0.0", "--port", "8080"};
    CHECK(serverArgs(s) == want2);
    // never an argument that would make the server (or anything) ask a question
    for (const std::string& a : serverArgs(s)) CHECK(a != "--setup");

    CHECK_EQ(quoteArg("abc"), std::string("abc"));
    CHECK_EQ(quoteArg(""), std::string("\"\""));
    CHECK_EQ(quoteArg("C:\\my models\\a.gguf"), std::string("\"C:\\my models\\a.gguf\""));
    CHECK_EQ(quoteArg("C:\\dir with space\\"), std::string("\"C:\\dir with space\\\\\""));  // trailing backslash doubled
    CHECK_EQ(quoteArg("say \"hi\""), std::string("\"say \\\"hi\\\"\""));
    CHECK_EQ(quoteArg("a\\\"b"), std::string("\"a\\\\\\\"b\""));  // a\"b -> "a\\\"b"
    CHECK_EQ(quoteArg("C:\\nospace\\x.gguf"), std::string("C:\\nospace\\x.gguf"));  // backslashes alone stay
    CHECK_EQ(commandLine("C:\\Program Files\\whirl\\whirl.exe", {"serve", "C:\\m.gguf", "--port", "8080"}),
             std::string("\"C:\\Program Files\\whirl\\whirl.exe\" serve C:\\m.gguf --port 8080"));
}

// ---------------------------------------------------------------------------
// 5. tray menu, chat

void testTrayAndChat() {
    const auto m = trayMenu();
    CHECK_EQ(m.size(), std::size_t{4});
    CHECK(m[0].first == TrayItem::open_chat && m[1].first == TrayItem::show_log && m[2].first == TrayItem::settings &&
          m[3].first == TrayItem::quit);
    for (const auto& [id, label] : m) CHECK(static_cast<int>(id) != 0 && !label.empty());  // 0 = "no command" in TrackPopupMenu

    const std::string body = chatRequestBody({{"user", "\xE4\xBD\xA0\xE5\xA5\xBD \"x\""}}, 0);
    const json::Value v = json::parse(body);
    CHECK(v.isObject());
    const json::Value* msgs = v.asObject().find("messages");
    CHECK(msgs && msgs->isArray() && msgs->asArray().size() == 1);
    CHECK(msgs && msgs->asArray()[0].asObject().find("content")->asString() == "\xE4\xBD\xA0\xE5\xA5\xBD \"x\"");
    CHECK(v.asObject().find("stream") && v.asObject().find("stream")->asBool());
    CHECK(!v.asObject().find("max_tokens"));
    CHECK(json::parse(chatRequestBody({}, 100)).asObject().find("max_tokens")->asInt() == 100);

    // stream split at arbitrary points, CRLF, keep-alive comments, a role-only delta, [DONE]
    const std::string sse =
        ": ping\r\n\r\n"
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\r\n\r\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo \\u4f60\"}}]}\n\n"
        "data: not json\n\n"
        "data: [DONE]\n\n";
    for (std::size_t step : {std::size_t{1}, std::size_t{3}, std::size_t{7}, sse.size()}) {
        std::string pending, text;
        bool done = false;
        for (std::size_t i = 0; i < sse.size(); i += step)
            for (const std::string& d : sseDeltas(std::string_view(sse).substr(i, step), &pending, &done)) text += d;
        CHECK_EQ(text, std::string("Hello \xE4\xBD\xA0"));
        CHECK(done);
    }
    std::string pending;
    bool done = false;
    CHECK(sseDeltas("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}", &pending, &done).empty());  // no line end yet
    CHECK_EQ(sseDeltas("\n", &pending, &done).size(), std::size_t{1});
    CHECK(!done);
}

}  // namespace

int main() {
    testDecideStart();
    testSettings();
    testAnswers();
    testGpus();
    testModels();
    testModeContextPort();
    testServerArgs();
    testTrayAndChat();
    std::printf("launcher tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
