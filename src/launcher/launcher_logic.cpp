// Windows launcher: decision logic (no OS call; tests/launcher/launcher_tests.cpp). See launcher.h.
// SPDX-License-Identifier: Apache-2.0

#include "launcher/launcher.h"

#include "whirl/json.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <map>

namespace whirl::launcher {

namespace {

std::string trim(std::string_view s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return std::string(s.substr(a, b - a));
}

std::string lower(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

bool endsWith(std::string_view s, std::string_view suf) { return s.size() >= suf.size() && s.substr(s.size() - suf.size()) == suf; }

std::string baseName(std::string_view path) {
    const std::size_t p = path.find_last_of("\\/");
    return std::string(p == std::string_view::npos ? path : path.substr(p + 1));
}

template <class T>
std::optional<T> parseUnsigned(std::string_view s) {
    T v{};
    if (s.empty()) return std::nullopt;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

constexpr std::uint64_t gib = 1024ull * 1024 * 1024;

}  // namespace

// ---------------------------------------------------------------------------
// 1. start

Start decideStart(const StartFacts& f) {
    if (f.n_args != 0 || f.no_launcher_env) return Start::classic;
    if (!f.stdin_console || !f.stdout_console) return Start::classic;
    return f.console_processes == 1 ? Start::launcher : Start::classic;
}

bool setupAllowed(const StartFacts& f) { return f.stdin_console && f.stdout_console && f.console_processes >= 1; }

bool noLauncherEnv(const char* value) { return value != nullptr && *value != '\0' && std::string_view(value) != "0"; }

// ---------------------------------------------------------------------------
// 2. settings

bool validMode(std::string_view m) { return m == "precise" || m == "balance" || m == "fast"; }

std::string serializeSettings(const Settings& s) {
    std::string t;
    t += "# WHIRL launcher settings (double-click whirl.exe). `whirl --setup` asks again; you may also edit this file.\n";
    t += std::format("version={}\n", settings_version);
    t += "gpu=" + s.gpu + "\n";
    t += "model=" + s.model + "\n";
    t += "mode=" + s.mode + "\n";
    t += std::format("ctx={}\n", s.ctx);
    t += std::format("port={}\n", s.port);
    t += "host=" + s.host + "\n";
    return t;
}

std::optional<Settings> parseSettings(std::string_view text, std::string* why) {
    auto fail = [&](std::string w) -> std::optional<Settings> {
        if (why) *why = std::move(w);
        return std::nullopt;
    };
    // a UTF-8 byte order mark (Notepad may add one)
    if (text.size() >= 3 && text.substr(0, 3) == "\xEF\xBB\xBF") text.remove_prefix(3);
    std::map<std::string, std::string> kv;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        const std::string line = trim(text.substr(pos, nl - pos));
        pos = nl + 1;
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) return fail("a line without '=': " + line);
        kv[lower(trim(std::string_view(line).substr(0, eq)))] = trim(std::string_view(line).substr(eq + 1));
    }
    if (kv["version"] != std::to_string(settings_version)) return fail("not a version " + std::to_string(settings_version) + " settings file");
    Settings s;
    s.gpu = kv["gpu"];
    s.model = kv["model"];
    if (kv.count("mode")) s.mode = kv["mode"];
    if (!validMode(s.mode)) return fail("mode must be precise, balance or fast, not '" + s.mode + "'");
    if (const std::string& c = kv["ctx"]; !c.empty()) {
        const auto v = parseUnsigned<std::uint32_t>(c);
        if (!v || (*v != 0 && (*v < ctx_min || *v > ctx_hard_max))) return fail("ctx must be 0 or " + std::to_string(ctx_min) + ".." + std::to_string(ctx_hard_max));
        s.ctx = *v;
    }
    if (const std::string& p = kv["port"]; !p.empty()) {
        const auto v = parsePort(p);
        if (!v) return fail("port must be 1..65535");
        s.port = *v;
    }
    if (const std::string& h = kv["host"]; !h.empty()) s.host = h;
    return s;
}

bool settingsComplete(const Settings& s) { return !s.gpu.empty() && !s.model.empty() && validMode(s.mode) && s.port != 0; }

// ---------------------------------------------------------------------------
// 3. questions

Answer parseAnswer(const std::optional<std::string>& line, std::size_t n_options, bool allow_text) {
    Answer a;
    if (!line) {
        a.kind = Answer::Kind::eof;
        return a;
    }
    std::string t = trim(*line);
    if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = trim(std::string_view(t).substr(1, t.size() - 2));
    if (t.empty()) {
        a.kind = Answer::Kind::recommended;
        return a;
    }
    if (const auto v = parseUnsigned<std::size_t>(t); v && *v >= 1 && *v <= n_options) {
        a.kind = Answer::Kind::index;
        a.index = *v - 1;
        return a;
    }
    if (allow_text) {
        a.kind = Answer::Kind::text;
        a.text = t;
        return a;
    }
    a.kind = Answer::Kind::invalid;
    return a;
}

std::vector<GpuOption> gpuOptions(const std::vector<gpusel::GpuRecord>& gpus) {
    const int def = gpusel::defaultGpu(gpus).index;
    std::vector<GpuOption> out;
    for (const gpusel::GpuRecord& g : gpus) {
        GpuOption o;
        o.pci = "pci:" + gpusel::pciId(gpus, g);
        const std::uint64_t mib = g.fit_mib ? g.fit_mib : g.total_mib;
        o.fit_bytes = mib << 20;
        o.supported = g.supported;
        o.recommended = g.index == def && def >= 0;
        o.label = std::format("{} ({}), {:.1f} GiB{}, {}{}", g.name, g.arch, static_cast<double>(mib) / 1024.0,
                              g.integrated ? " dedicated" : "", o.pci, g.supported ? "" : "  [not supported by this build]");
        out.push_back(std::move(o));
    }
    return out;
}

int recommendedIndex(const std::vector<GpuOption>& opts) {
    if (opts.empty()) return -1;
    for (std::size_t i = 0; i < opts.size(); ++i)
        if (opts[i].recommended) return static_cast<int>(i);
    return 0;
}

bool isModelCandidate(std::string_view filename) {
    const std::string n = lower(baseName(filename));
    if (!endsWith(n, ".gguf")) return false;
    if (n.find("mmproj") != std::string::npos) return false;  // vision encoder
    if (n.find("dflash") != std::string::npos) return false;  // DFlash2 drafter
    return true;
}

int productRank(std::string_view filename) {
    const std::string n = lower(baseName(filename));
    const bool mx = n.find("mxfp4") != std::string::npos;
    if (!mx) return 3;
    if (n.find("swift-1.5") != std::string::npos && n.find("27b") != std::string::npos && n.find("mxfp4-a") != std::string::npos) return 0;
    if (n.find("swift-1.5") != std::string::npos || n.find("ornith-1.5") != std::string::npos) return 1;
    return 2;
}

bool modelFits(std::uint64_t model_bytes, std::uint64_t gpu_fit_bytes) { return gpu_fit_bytes == 0 || model_bytes < gpu_fit_bytes; }

std::vector<ModelFile> rankModels(std::vector<ModelFile> files, std::uint64_t gpu_fit_bytes) {
    std::vector<ModelFile> out;
    for (ModelFile& f : files) {
        const std::string k = lower(f.path);
        const bool dup = std::any_of(out.begin(), out.end(), [&](const ModelFile& o) { return lower(o.path) == k; });
        if (!dup && isModelCandidate(f.path)) out.push_back(std::move(f));
    }
    std::stable_sort(out.begin(), out.end(), [&](const ModelFile& a, const ModelFile& b) {
        const bool fa = modelFits(a.bytes, gpu_fit_bytes), fb = modelFits(b.bytes, gpu_fit_bytes);
        if (fa != fb) return fa;
        const int ra = productRank(a.path), rb = productRank(b.path);
        if (ra != rb) return ra < rb;
        return lower(baseName(a.path)) < lower(baseName(b.path));
    });
    return out;
}

const std::vector<std::string>& modeList() {
    static const std::vector<std::string> m = {"balance", "precise", "fast"};
    return m;
}

CtxChoice recommendContext(const CtxInput& in) {
    CtxChoice c;
    if (in.mem_bytes == 0 || in.kv_per_token == 0) return c;
    c.known = true;
    const std::uint64_t margin = std::max<std::uint64_t>(in.mem_bytes / 10, gib * 3 / 2);
    const std::uint64_t used = margin + in.need_fixed;
    const std::uint64_t usable = in.mem_bytes > used ? in.mem_bytes - used : 0;
    std::uint64_t tokens = usable / in.kv_per_token;
    tokens = std::min<std::uint64_t>(tokens, ctx_hard_max);
    tokens -= tokens % 4096;
    c.max = static_cast<std::uint32_t>(tokens);
    c.fits = c.max >= ctx_min;
    if (!c.fits) return c;
    std::uint64_t rec = std::min<std::uint64_t>(c.max * 8ull / 10, ctx_default_cap);
    rec -= rec % 4096;
    c.recommended = static_cast<std::uint32_t>(std::max<std::uint64_t>(rec, ctx_min));
    return c;
}

std::optional<std::uint32_t> parseCtx(std::string_view s) {
    std::string t = lower(trim(s));
    std::uint64_t mul = 1;
    if (!t.empty() && t.back() == 'k') {
        mul = 1024;
        t.pop_back();
    }
    const auto v = parseUnsigned<std::uint64_t>(t);
    if (!v || *v == 0 || *v * mul > 0xffffffffull) return std::nullopt;
    return static_cast<std::uint32_t>(*v * mul);
}

std::uint32_t clampCtx(std::uint32_t v) {
    v = std::clamp(v, ctx_min, ctx_hard_max);
    return v - v % 1024;
}

std::uint16_t recommendPort(const std::function<bool(std::uint16_t)>& is_free, std::uint16_t first) {
    for (std::uint32_t p = first; p < static_cast<std::uint32_t>(first) + 20 && p <= 65535; ++p)
        if (is_free(static_cast<std::uint16_t>(p))) return static_cast<std::uint16_t>(p);
    return first;
}

std::optional<std::uint16_t> parsePort(std::string_view s) {
    const auto v = parseUnsigned<std::uint32_t>(trim(s));
    if (!v || *v == 0 || *v > 65535) return std::nullopt;
    return static_cast<std::uint16_t>(*v);
}

// ---------------------------------------------------------------------------
// 4. server command

std::vector<std::string> serverArgs(const Settings& s) {
    std::vector<std::string> a = {"serve", s.model, "--device", s.gpu, "--" + s.mode};
    if (s.ctx > 0) {
        a.push_back("--ctx-per-slot");
        a.push_back(std::to_string(s.ctx));
    }
    if (!s.host.empty() && s.host != "127.0.0.1") {
        a.push_back("--host");
        a.push_back(s.host);
    }
    a.push_back("--port");
    a.push_back(std::to_string(s.port));
    return a;
}

std::string quoteArg(std::string_view a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string_view::npos) return std::string(a);
    std::string r = "\"";
    std::size_t bs = 0;
    for (char ch : a) {
        if (ch == '\\') {
            ++bs;
            continue;
        }
        if (ch == '"') {
            r.append(bs * 2 + 1, '\\');
        } else {
            r.append(bs, '\\');
        }
        bs = 0;
        r += ch;
    }
    r.append(bs * 2, '\\');  // backslashes before the closing quote
    r += '"';
    return r;
}

std::string commandLine(std::string_view program, const std::vector<std::string>& args) {
    std::string r = quoteArg(program);
    for (const std::string& a : args) r += " " + quoteArg(a);
    return r;
}

// ---------------------------------------------------------------------------
// 5. tray / chat

std::vector<std::pair<TrayItem, std::string>> trayMenu() {
    return {{TrayItem::open_chat, "Open chat"}, {TrayItem::show_log, "Show log"}, {TrayItem::settings, "Settings..."}, {TrayItem::quit, "Quit WHIRL"}};
}

std::string chatRequestBody(const std::vector<ChatMessage>& msgs, int max_tokens) {
    json::Array arr;
    for (const ChatMessage& m : msgs) {
        json::Object o;
        o.set("role", json::Value(m.role));
        o.set("content", json::Value(m.content));
        arr.push_back(json::Value(std::move(o)));
    }
    json::Object req;
    req.set("messages", json::Value(std::move(arr)));
    req.set("stream", json::Value(true));
    if (max_tokens > 0) req.set("max_tokens", json::Value(max_tokens));
    return json::dumpPython(json::Value(std::move(req)), true);
}

std::vector<std::string> sseDeltas(std::string_view bytes, std::string* pending, bool* done) {
    std::vector<std::string> out;
    pending->append(bytes);
    for (;;) {
        const std::size_t nl = pending->find('\n');
        if (nl == std::string::npos) break;
        std::string line = pending->substr(0, nl);
        pending->erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data:", 0) != 0) continue;
        const std::string payload = trim(std::string_view(line).substr(5));
        if (payload == "[DONE]") {
            *done = true;
            continue;
        }
        try {
            const json::Value v = json::parse(payload);
            const json::Value* ch = v.isObject() ? v.asObject().find("choices") : nullptr;
            if (!ch || !ch->isArray() || ch->asArray().empty()) continue;
            const json::Value& c0 = ch->asArray()[0];
            if (!c0.isObject()) continue;
            const json::Value* d = c0.asObject().find("delta");
            if (!d || !d->isObject()) continue;
            const json::Value* content = d->asObject().find("content");
            if (content && content->isString() && !content->asString().empty()) out.push_back(content->asString());
        } catch (const json::ParseError&) {
            // not JSON: ignored (the server never sends that; a proxy might)
        }
    }
    return out;
}

}  // namespace whirl::launcher
