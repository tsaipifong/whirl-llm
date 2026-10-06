// whirl-server-gate: end-to-end gates for an OpenAI-compatible server
// executable on a real model (GPU). Runs against the WHIRL server
// (whirl-server.exe MODEL ...) or, as a black box, against the research
// prototype (whirl.exe serve MODEL ...), records every generated text and
// can compare two runs (e.g. WHIRL vs the prototype: token-identical greedy
// output is expected in precision mode).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-server-gate --exe PATH --kind whirl|proto --model GGUF --suite NAME[,NAME...]
//                     [--work DIR] [--port 8093] [--results OUT.json] [--compare REF.json]
//                     [--cli EXE] [--no-lock] [--server-env NAME=VALUE ...] [--kv FORMAT]
// Suites: basic, mt_cache, pool, tier, sys, restore_conc, vis (see each function;
// vis / vis_speed need --mmproj MMPROJ.gguf and --images DIR).
// A GPU lock file (WHIRL_GPU_LOCK, default %TEMP%\whirl-gpu.lock, opened with
// share mode 0) is held while a server runs, unless WHIRL_R9700_LOCK_HELD is set
// (a parent launcher already holds the lock); each server is stopped (TerminateProcess on
// the PID this program started) before the next one starts.
// Test texts: llama.cpp sources / docs from an external checkout
// (WHIRL_GATE_TEXT_ROOT, required: the root of a llama.cpp source checkout),
// read as data only.
// Defaults: --work = WHIRL_GATE_WORK, else %TEMP%\whirl-tests\gate;
// --server-env adds environment variables to every server (the servers otherwise get no
// WHIRL_* variable except WHIRL_DEVICE; a suite's own settings win), e.g.
// WHIRL_KV_RAM_MB=8192 to turn the host tiers on for an integrated GPU (Radeon 8060S),
// whose servers default to no tiers. --kv: the KV format the sys / restore_conc / vis
// suites pin (default q8v; gfx1151 has no q8v kernels: use f16 or q8 there; auto = no pin).
// --images = WHIRL_GATE_IMAGES (no default; only suite vis needs it).

#include "http_client.h"
#include "whirl/common.h"
#include "whirl/json.h"

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cwctype>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace whirl;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// options / results

struct Opts {
    std::string exe, kind = "whirl", model, work;  // empty: WHIRL_GATE_WORK, else %TEMP%\whirl-tests\gate
    std::uint16_t port = 8093;
    std::string results, compare, cli;
    std::vector<std::string> suites;
    bool lock = true;
    std::string text_root;  // WHIRL_GATE_TEXT_ROOT
    // suite vis: mmproj and the test images
    std::string mmproj;
    std::string images;  // empty: WHIRL_GATE_IMAGES
    std::map<std::string, std::string> server_env;  // --server-env
    std::string kv_pin = "q8v";                     // --kv
} g;

// KV-format pin of the suites that compare servers with different free memory
std::map<std::string, std::string> kvPin() {
    if (g.kv_pin == "auto") return {};
    return {{"WHIRL_KV", g.kv_pin}};
}

json::Object g_results;  // name -> text
int g_pass = 0, g_fail = 0;
std::vector<std::string> g_failed;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  -- ", detail.c_str());
    std::fflush(stdout);
    if (ok) ++g_pass;
    else {
        ++g_fail;
        g_failed.push_back(name);
    }
}

void record(const std::string& key, const std::string& text) { g_results.set(key, json::Value(text)); }

std::string nowStr() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    return std::format("{:02}:{:02}:{:02}", st.wHour, st.wMinute, st.wSecond);
}

void logLine(const std::string& s) {
    std::printf("[%s] %s\n", nowStr().c_str(), s.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// GPU lock (same file and rules as the project's launchers)

class GpuLock {
public:
    // reentrant: nested acquire / release pairs keep the file open until the outermost release
    void acquire() {
        ++depth_;
        if (!g.lock || h_ != INVALID_HANDLE_VALUE || std::getenv("WHIRL_R9700_LOCK_HELD")) return;
        int waited = 0;
        for (;;) {
            h_ = CreateFileW(lockPath().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                             OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h_ != INVALID_HANDLE_VALUE) break;
            if (waited % 30 == 0) logLine(std::format("r9700.lock: another launcher holds the GPU; waiting ({} s)", waited));
            std::this_thread::sleep_for(std::chrono::seconds(5));
            waited += 5;
        }
        for (int w = 0; aliveEngine(); w += 5) {
            if (w >= 600) {
                depth_ = 1;
                release();
                throw std::runtime_error("r9700.lock: a whirl.exe is still alive after 10 min");
            }
            if (w % 30 == 0) logLine("r9700.lock: waiting for a leftover whirl.exe to exit");
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
    void release() {
        if (depth_ > 0 && --depth_ > 0) return;
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
    }
    ~GpuLock() {
        depth_ = 0;
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
    }

private:
    // WHIRL_GPU_LOCK, else %TEMP%\whirl-gpu.lock
    static std::wstring lockPath() {
        if (const char* p = std::getenv("WHIRL_GPU_LOCK"); p && *p) return widen(p);
        return (fs::temp_directory_path() / L"whirl-gpu.lock").wstring();
    }
    static bool aliveEngine() {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap == INVALID_HANDLE_VALUE) return false;
        PROCESSENTRY32W pe{};
        pe.dwSize = sizeof pe;
        bool found = false;
        for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
            std::wstring n = pe.szExeFile;
            for (auto& c : n) c = static_cast<wchar_t>(towlower(c));
            if (n == L"whirl.exe" || n == L"whirl-server.exe") found = true;
        }
        CloseHandle(snap);
        return found;
    }
    HANDLE h_ = INVALID_HANDLE_VALUE;
    int depth_ = 0;
} g_lock;

// ---------------------------------------------------------------------------
// server process

std::wstring quoteArg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    for (wchar_t c : a) {
        if (c == L'"') q += L"\\\"";
        else q += c;
    }
    return q + L"\"";
}

class Server {
public:
    Server(const std::string& tag, const std::vector<std::string>& args, const std::map<std::string, std::string>& env_extra,
           bool fresh_ssd = true)
        : tag_(tag) {
        fs::create_directories(widen(g.work));
        log_ = g.work + "\\server_" + tag + ".log";
        std::error_code ec;
        fs::remove(widen(log_), ec);
        std::map<std::string, std::string> env_add = env_extra;
        for (const auto& [k, v] : g.server_env) env_add.emplace(k, v);  // suite settings win
        if (fresh_ssd && !env_add.count("WHIRL_KV_SSD_DIR")) {
            ssd_ = g.work + "\\ssd_" + tag;
            fs::remove_all(widen(ssd_), ec);
            env_add["WHIRL_KV_SSD_DIR"] = ssd_;
        }
        std::vector<std::string> argv{g.exe};
        if (g.kind == "proto") argv.push_back("serve");
        argv.push_back(g.model);
        for (const char* a : {"--port"}) argv.push_back(a);
        argv.push_back(std::to_string(g.port));
        argv.push_back("--log-file");
        argv.push_back(log_);
        argv.insert(argv.end(), args.begin(), args.end());
        std::wstring cmd;
        for (const auto& a : argv) cmd += (cmd.empty() ? L"" : L" ") + quoteArg(widen(a));
        // environment: ours minus WHIRL_* (WHIRL_DEVICE and WHIRL_VRAM_LIMIT_MB kept: they describe
        // the GPU, so a gate under a simulated 16 GB card runs its servers on that card) plus the extras
        std::vector<std::pair<std::wstring, std::wstring>> envv;
        if (wchar_t* blk = GetEnvironmentStringsW()) {
            for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
                std::wstring s = p;
                const std::size_t eq = s.find(L'=', 1);
                if (eq == std::wstring::npos) continue;
                std::wstring k = s.substr(0, eq), up = k;
                for (auto& c : up) c = static_cast<wchar_t>(towupper(c));
                if (up.rfind(L"WHIRL_", 0) == 0 && up != L"WHIRL_DEVICE" && up != L"WHIRL_VRAM_LIMIT_MB") continue;
                envv.emplace_back(k, s.substr(eq + 1));
            }
            FreeEnvironmentStringsW(blk);
        }
        envv.emplace_back(L"WHIRL_R9700_LOCK_HELD", std::to_wstring(GetCurrentProcessId()));
        for (const auto& [k, v] : env_add) envv.emplace_back(widen(k), widen(v));
        std::wstring block;
        for (const auto& [k, v] : envv) block += k + L"=" + v + L'\0';
        block += L'\0';
        const std::string out_path = g.work + "\\server_" + tag + ".stdout.txt";
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE out = CreateFileW(widen(out_path).c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
        STARTUPINFOW si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = out;
        si.hStdError = out;
        PROCESS_INFORMATION pi{};
        std::vector<wchar_t> cbuf(cmd.begin(), cmd.end());
        cbuf.push_back(0);
        if (!CreateProcessW(nullptr, cbuf.data(), nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                            block.data(), nullptr, &si, &pi)) {
            CloseHandle(out);
            throw std::runtime_error("cannot start " + g.exe);
        }
        CloseHandle(out);
        proc_ = pi.hProcess;
        CloseHandle(pi.hThread);
        logLine(std::format("server {} started (pid {}): {}", tag, pi.dwProcessId, narrow(cmd)));
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            auto r = test::httpRequest("127.0.0.1", g.port, "GET", "/health", {}, 5);
            if (r.status == 200) break;
            DWORD code = 0;
            if (GetExitCodeProcess(proc_, &code) && code != STILL_ACTIVE)
                throw std::runtime_error(std::format("server {} exited during startup (code {}); see {}", tag, code, out_path));
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(400))
                throw std::runtime_error("server did not become healthy in 400 s");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        logLine(std::format("server {} ready in {:.1f} s", tag,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()));
    }
    ~Server() { stop(); }
    // Graceful stop first: a Ctrl+C on the server's (hidden) console, which the WHIRL
    // server handles by flushing its prefix-cache tiers before it exits (the
    // prototype exits at once on Ctrl+C); TerminateProcess if it has not exited after
    // 30 s.
    bool ctrlC(DWORD pid) {
        bool ok = false;
        const bool con_out = GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_CHAR;
        std::fflush(stdout);
        std::fflush(stderr);
        FreeConsole();
        if (AttachConsole(pid)) {
            SetConsoleCtrlHandler(nullptr, TRUE);  // this process ignores the event it sends
            ok = GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0) != 0;
            Sleep(100);  // let the event be delivered before leaving the console
            FreeConsole();
        }
        if (AttachConsole(ATTACH_PARENT_PROCESS) && con_out) {
            FILE* f = nullptr;
            freopen_s(&f, "CONOUT$", "w", stdout);
            freopen_s(&f, "CONOUT$", "w", stderr);
        }
        // keep ignoring stray Ctrl+C deliveries for a moment, then restore the default
        Sleep(200);
        SetConsoleCtrlHandler(nullptr, FALSE);
        return ok;
    }
    void stop() {
        if (!proc_) return;
        const auto t0 = std::chrono::steady_clock::now();
        const bool sent = ctrlC(GetProcessId(proc_));
        if (!sent || WaitForSingleObject(proc_, 30000) != WAIT_OBJECT_0) {
            logLine(std::format("server {}: {}; terminating", tag_, sent ? "no exit 30 s after Ctrl+C" : "Ctrl+C could not be sent"));
            TerminateProcess(proc_, 1);
        } else {
            logLine(std::format("server {} stopped (Ctrl+C) in {:.1f} s", tag_,
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()));
        }
        WaitForSingleObject(proc_, 30000);
        CloseHandle(proc_);
        proc_ = nullptr;
        if (!ssd_.empty()) {
            std::error_code ec;
            fs::remove_all(widen(ssd_), ec);
        }
    }
    std::vector<std::string> logLines() const {
        std::vector<std::string> out;
        try {
            const std::string t = readFile(log_);
            std::size_t p = 0;
            while (p < t.size()) {
                std::size_t e = t.find('\n', p);
                if (e == std::string::npos) e = t.size();
                std::string l = t.substr(p, e - p);
                if (!l.empty() && l.back() == '\r') l.pop_back();
                out.push_back(std::move(l));
                p = e + 1;
            }
        } catch (const std::exception&) {
        }
        return out;
    }
    std::size_t logCount(const std::string& re) const {
        const std::regex r(re);
        std::size_t n = 0;
        for (const auto& l : logLines())
            if (std::regex_search(l, r)) ++n;
        return n;
    }
    bool waitLog(const std::string& re, std::size_t n, int timeout_s) const {
        for (int i = 0; i < timeout_s * 4; ++i) {
            if (logCount(re) >= n) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        return false;
    }
    // Waits until the server's pending RAM -> SSD tier writes are done before the
    // gate stops it (TerminateProcess: an SSD write still waiting for its delay or
    // in flight is lost). The engine logs the tier status line ("kv tier | RAM ...")
    // when its tier work goes idle, so a status line after the last spill line means
    // nothing is pending. (The server's startup status line alone does not count.)
    bool waitTierIdle(int timeout_s) const {
        const std::regex spill("(kv tier|shared prefix): spill "), status(R"(kv tier \| RAM)");
        for (int i = 0; i < timeout_s * 4; ++i) {
            std::ptrdiff_t last_spill = -1, last_status = -1;
            const auto lines = logLines();
            for (std::size_t k = 0; k < lines.size(); ++k) {
                if (std::regex_search(lines[k], spill)) last_spill = static_cast<std::ptrdiff_t>(k);
                if (std::regex_search(lines[k], status)) last_status = static_cast<std::ptrdiff_t>(k);
            }
            if (last_status > last_spill) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        return false;
    }
    const std::string& ssd() const { return ssd_; }

private:
    std::string tag_, log_, ssd_;
    HANDLE proc_ = nullptr;
};

// ---------------------------------------------------------------------------
// requests

struct ChatResult {
    int status = 0;
    std::string content, reasoning, finish;
    std::vector<std::array<std::string, 3>> tool_calls;  // id, name, arguments
    std::int64_t prompt = 0, cached = 0, completion = 0;
    std::string raw;
    double ms = 0;
    std::string text() const {
        std::string t = reasoning + std::string(1, '\0') + content + std::string(1, '\0');
        for (const auto& tc : tool_calls) t += tc[1] + "(" + tc[2] + ");";
        return t;
    }
};

const json::Value* path(const json::Value& v, std::initializer_list<const char*> keys) {
    const json::Value* cur = &v;
    for (const char* k : keys) {
        if (!cur->isObject()) return nullptr;
        cur = cur->get(k);
        if (!cur) return nullptr;
    }
    return cur;
}

std::string str(const json::Value* v) { return v && v->isString() ? v->asString() : std::string(); }

void readUsage(const json::Value& u, ChatResult& r) {
    if (const json::Value* p = u.get("prompt_tokens")) r.prompt = p->asInt();
    if (const json::Value* c = u.get("completion_tokens")) r.completion = c->asInt();
    if (const json::Value* d = path(u, {"prompt_tokens_details", "cached_tokens"})) r.cached = d->asInt();
}

// messages: JSON array text; extra: additional top-level fields (",\"k\":v...")
ChatResult chat(const std::string& messages, const std::string& extra, bool stream = false) {
    std::string body = "{\"model\":\"m\",\"messages\":" + messages + extra;
    if (stream) body += ",\"stream\":true,\"stream_options\":{\"include_usage\":true}";
    body += "}";
    const auto t0 = std::chrono::steady_clock::now();
    const auto h = test::httpRequest("127.0.0.1", g.port, "POST", "/v1/chat/completions", body, 1200);
    ChatResult r;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r.status = h.status;
    r.raw = h.body;
    if (h.status != 200) return r;
    try {
        if (!stream) {
            const json::Value v = json::parse(h.body);
            const json::Value& c0 = v.get("choices")->asArray()[0];
            r.content = str(path(c0, {"message", "content"}));
            r.reasoning = str(path(c0, {"message", "reasoning_content"}));
            r.finish = str(c0.get("finish_reason"));
            if (const json::Value* tcs = path(c0, {"message", "tool_calls"}); tcs && tcs->isArray())
                for (const json::Value& tc : tcs->asArray())
                    r.tool_calls.push_back({str(tc.get("id")), str(path(tc, {"function", "name"})),
                                            str(path(tc, {"function", "arguments"}))});
            if (const json::Value* u = v.get("usage")) readUsage(*u, r);
        } else {
            for (const std::string& ev : test::sseEvents(h.body)) {
                if (ev == "[DONE]") continue;
                const json::Value v = json::parse(ev);
                if (const json::Value* u = v.get("usage"); u && u->isObject()) readUsage(*u, r);
                const json::Value* ch = v.get("choices");
                if (!ch || !ch->isArray() || ch->asArray().empty()) continue;
                const json::Value& c0 = ch->asArray()[0];
                if (const json::Value* d = c0.get("delta")) {
                    r.reasoning += str(d->get("reasoning_content"));
                    r.content += str(d->get("content"));
                    if (const json::Value* tcs = d->get("tool_calls"); tcs && tcs->isArray())
                        for (const json::Value& tc : tcs->asArray())
                            r.tool_calls.push_back({str(tc.get("id")), str(path(tc, {"function", "name"})),
                                                    str(path(tc, {"function", "arguments"}))});
                }
                if (const std::string f = str(c0.get("finish_reason")); !f.empty()) r.finish = f;
            }
        }
    } catch (const std::exception& ex) {
        r.status = -1;
        r.raw = std::string("parse error: ") + ex.what();
    }
    return r;
}

std::string js(const std::string& s) {
    std::string o = "\"";
    json::appendEscaped(o, s);
    return o + "\"";
}

std::string msg(const std::string& role, const std::string& content) {
    return "{\"role\":" + js(role) + ",\"content\":" + js(content) + "}";
}

std::string arr(const std::vector<std::string>& items) {
    std::string s = "[";
    for (std::size_t i = 0; i < items.size(); ++i) s += (i ? "," : "") + items[i];
    return s + "]";
}

// template a (Qwen3.8): reasoning effort "medium" renders no instruction
std::string thinkKw(bool think) {
    return think ? ",\"chat_template_kwargs\":{\"enable_thinking\":true,\"reasoning_effort\":\"medium\"}"
                 : ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
}

std::string greedy(int max_tokens) { return std::format(",\"temperature\":0,\"max_tokens\":{}", max_tokens); }

// ---------------------------------------------------------------------------
// test texts (external llama.cpp checkout, data only)

std::vector<std::string> sortedFiles(const std::string& dir, std::initializer_list<const char*> exts, bool recursive) {
    std::vector<std::string> out;
    auto take = [&](const fs::path& p) {
        const std::string e = narrow(p.extension().wstring());
        for (const char* x : exts)
            if (e == x) out.push_back(narrow(p.wstring()));
    };
    if (recursive) {
        for (const auto& e : fs::recursive_directory_iterator(widen(dir)))
            if (e.is_regular_file()) take(e.path());
    } else {
        for (const auto& e : fs::directory_iterator(widen(dir)))
            if (e.is_regular_file()) take(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// UTF-8-safe prefix of at most n bytes
std::string utf8Cut(const std::string& s, std::size_t n) {
    if (n >= s.size()) return s;
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
    return s.substr(0, n);
}

std::string cleanText(std::string t) {
    // keep valid UTF-8 only (a source file with stray bytes would make invalid JSON)
    std::string o;
    o.reserve(t.size());
    std::size_t i = 0;
    while (i < t.size()) {
        const std::size_t n = utf8LenFromLead(static_cast<unsigned char>(t[i]));
        if (i + n <= t.size() && isValidUtf8(std::string_view(t).substr(i, n))) {
            o.append(t, i, n);
            i += n;
        } else {
            ++i;
        }
    }
    return o;
}

std::string codeText(std::size_t chars, std::size_t skip) {
    static const auto files = sortedFiles(g.text_root + "\\src", {".cpp", ".h"}, false);
    std::string out;
    for (std::size_t i = skip; i < files.size() && out.size() < chars; ++i) {
        const std::string t = cleanText(readFile(files[i]));
        out += "// FILE " + narrow(fs::path(widen(files[i])).filename().wstring()) + "\n";
        out += utf8Cut(t, chars > out.size() ? chars - out.size() : 0);
    }
    return utf8Cut(out, chars);
}

std::string docsText(std::size_t chars, std::size_t skip = 0) {
    static const auto files = sortedFiles(g.text_root + "\\docs", {".md"}, true);
    std::vector<std::string> fl(files.begin() + std::min(skip, files.size()), files.end());
    fl.insert(fl.end(), files.begin(), files.begin() + std::min(skip, files.size()));
    std::string out;
    for (const auto& f : fl) {
        out += "\n## skill: " + narrow(fs::path(widen(f)).stem().wstring()) + "\n" + cleanText(readFile(f));
        if (out.size() >= chars) break;
    }
    return utf8Cut(out, chars);
}

struct ServerLease {
    // the GPU lock is held while the server runs
    std::unique_ptr<Server> s;
    ServerLease(const std::string& tag, const std::vector<std::string>& args, const std::map<std::string, std::string>& env,
                bool fresh_ssd = true) {
        g_lock.acquire();
        try {
            s = std::make_unique<Server>(tag, args, env, fresh_ssd);
        } catch (...) {
            g_lock.release();
            throw;
        }
    }
    ~ServerLease() {
        s.reset();
        g_lock.release();
    }
    Server* operator->() { return s.get(); }
};

template <class F>
std::vector<ChatResult> parallel(const std::vector<F>& jobs) {
    std::vector<ChatResult> res(jobs.size());
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < jobs.size(); ++i) th.emplace_back([&, i] { res[i] = jobs[i](); });
    for (auto& t : th) t.join();
    return res;
}

std::string brief(const ChatResult& r) {
    std::string c = r.content.substr(0, 60);
    for (auto& ch : c)
        if (ch == '\n') ch = ' ';
    return std::format("status {}, prompt {}, cached {}, completion {}, finish {}, \"{}\"", r.status, r.prompt, r.cached,
                       r.completion, r.finish, c);
}

const std::string P_ZH =
    "\xE8\xAB\x8B\xE7\x94\xA8\xE7\xB9\x81\xE9\xAB\x94\xE4\xB8\xAD\xE6\x96\x87\xE8\xA7\xA3\xE9\x87\x8B\xE4\xBB\x80\xE9\xBA\xBC"
    "\xE6\x98\xAF\xE6\x8E\xA8\xE6\xB8\xAC\xE8\xA7\xA3\xE7\xA2\xBC\xEF\xBC\x88speculative decoding\xEF\xBC\x89\xEF\xBC\x8C"
    "\xE4\xBB\xA5\xE5\x8F\x8A\xE5\xAE\x83\xE7\x82\xBA\xE4\xBB\x80\xE9\xBA\xBC\xE8\x83\xBD\xE5\x8A\xA0\xE9\x80\x9F\xE5\xA4\xA7"
    "\xE5\x9E\x8B\xE8\xAA\x9E\xE8\xA8\x80\xE6\xA8\xA1\xE5\x9E\x8B\xE7\x9A\x84\xE7\x94\x9F\xE6\x88\x90\xE3\x80\x82";

const std::string WEATHER_TOOL =
    R"([{"type":"function","function":{"name":"get_weather","description":"Get the current weather of a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"city name"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}}])";

// CLI reference (`EXE chat MODEL @prompt --max-tokens N --no-stream`): body between
// the "prompt tokens:" line and the "\n\n  prefill:" stats
std::optional<std::string> cliRun(const std::string& prompt, bool think, int max_tokens) {
    if (g.cli.empty()) return std::nullopt;
    const std::string pf = g.work + "\\cli_prompt.txt";
    writeFile(pf, prompt);
    const std::string out = g.work + "\\cli_out.txt";
    std::wstring cmd = quoteArg(widen(g.cli)) + L" chat " + quoteArg(widen(g.model)) + L" " + quoteArg(widen("@" + pf)) +
                       L" --max-tokens " + std::to_wstring(max_tokens) + L" --no-stream";
    SetEnvironmentVariableW(L"WHIRL_THINK", think ? L"1" : L"0");
    SetEnvironmentVariableW(L"WHIRL_R9700_LOCK_HELD", std::to_wstring(GetCurrentProcessId()).c_str());
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE oh = CreateFileW(widen(out).c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = oh;
    si.hStdError = oh;
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cb(cmd.begin(), cmd.end());
    cb.push_back(0);
    g_lock.acquire();
    const BOOL ok = CreateProcessW(nullptr, cb.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(oh);
    if (!ok) {
        g_lock.release();
        return std::nullopt;
    }
    WaitForSingleObject(pi.hProcess, 900000);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    g_lock.release();
    SetEnvironmentVariableW(L"WHIRL_THINK", nullptr);
    std::string s = readFile(out);
    std::string t;
    for (char c : s)
        if (c != '\r') t.push_back(c);
    const std::size_t k = t.find("prompt tokens:");
    if (k == std::string::npos) return std::nullopt;
    std::size_t b = t.find('\n', k);
    if (b == std::string::npos) return std::nullopt;
    const std::string rest = t.substr(b + 1);
    const std::size_t e = rest.find("\n\n  prefill:");
    std::string body = rest.substr(0, e);
    while (!body.empty() && (body.back() == ' ' || body.back() == '\n')) body.pop_back();
    while (!body.empty() && (body.front() == ' ' || body.front() == '\n')) body.erase(body.begin());
    return body;
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    return s.substr(a);
}

// ---------------------------------------------------------------------------
// suite basic: endpoints, greedy texts, stream == non-stream, CLI A/B,
// seed reproducibility, thinking, tool round trip, multi-turn, concurrency

void suiteBasic() {
    logLine("== suite basic");
    const std::string P_EN = codeText(4000, 0) + "\n\nSummarize the above in three sentences.";
    struct Case {
        const char* key;
        std::string prompt;
        bool think;
        int max;
    };
    const std::vector<Case> cases = {{"zh_think", P_ZH, true, 200}, {"zh_nothink", P_ZH, false, 200}, {"en_nothink", P_EN, false, 150}};
    // CLI references first: the CLI needs the GPU to itself (the server holds it later)
    std::map<std::string, std::optional<std::string>> cli_ref;
    // (as the original gate: the CLI's non-thinking prompt is not the template's, so
    // only the thinking Chinese prompt and the English summary are compared)
    for (const Case& c : cases)
        if (std::string(c.key) != "zh_nothink") cli_ref[c.key] = cliRun(c.prompt, c.think, c.max);
    ServerLease s("basic", {}, {});
    {
        auto h = test::httpRequest("127.0.0.1", g.port, "GET", "/health");
        check("basic: GET /health", h.status == 200 && h.body.find("\"status\":\"ok\"") != std::string::npos, h.body);
        auto m = test::httpRequest("127.0.0.1", g.port, "GET", "/v1/models");
        check("basic: GET /v1/models", m.status == 200 && m.body.find("\"object\":\"list\"") != std::string::npos);
        // CORS (v0.1.4): loopback Origins are echoed, other Origins get no CORS headers
        auto o = test::httpRequest("127.0.0.1", g.port, "OPTIONS", "/v1/chat/completions", {}, 600,
                                   "Origin: http://localhost:3000\r\n");
        check("basic: OPTIONS preflight (CORS)",
              o.status == 204 && o.headers.find("Access-Control-Allow-Origin: http://localhost:3000") != std::string::npos);
        auto oe = test::httpRequest("127.0.0.1", g.port, "OPTIONS", "/v1/chat/completions", {}, 600,
                                    "Origin: https://evil.example\r\n");
        check("basic: OPTIONS preflight, foreign Origin -> no CORS headers",
              oe.status == 204 && oe.headers.find("Access-Control-") == std::string::npos);
        auto b = test::httpRequest("127.0.0.1", g.port, "POST", "/v1/chat/completions", "{not json");
        check("basic: bad JSON -> 400", b.status == 400 && b.body.find("\"error\"") != std::string::npos, b.body.substr(0, 100));
        auto mm = test::httpRequest("127.0.0.1", g.port, "POST", "/v1/chat/completions", R"({"model":"x"})");
        check("basic: missing messages -> 400", mm.status == 400 && mm.body.find("messages") != std::string::npos);
        auto t = test::httpRequest("127.0.0.1", g.port, "POST", "/v1/chat/completions",
                                   R"({"messages":[{"role":"user","content":"hi"}],"temperature":-1})");
        check("basic: temperature -1 -> 400", t.status == 400);
        auto n = test::httpRequest("127.0.0.1", g.port, "GET", "/nope");
        check("basic: unknown path -> 404", n.status == 404);
    }
    std::map<std::string, ChatResult> solo;
    for (const Case& c : cases) {
        ChatResult r = chat(arr({msg("user", c.prompt)}), greedy(c.max) + thinkKw(c.think));
        check(std::string("basic: greedy ") + c.key, r.status == 200 && !r.content.empty() || !r.reasoning.empty(), brief(r));
        record(std::string("basic/") + c.key, r.text());
        solo[c.key] = r;
        ChatResult st = chat(arr({msg("user", c.prompt)}), greedy(c.max) + thinkKw(c.think), true);
        check(std::string("basic: stream == non-stream ") + c.key,
              st.content == r.content && st.reasoning == r.reasoning && st.finish == r.finish, brief(st));
        if (const auto& cli = cli_ref[c.key]) {
            std::string reason, content = *cli;
            if (c.think) {
                const std::size_t k = cli->find("</think>");
                if (k != std::string::npos) {
                    reason = trim(cli->substr(0, k));
                    content = trim(cli->substr(k + 8));
                } else {
                    reason = trim(*cli);
                    content.clear();
                }
                if (reason.rfind("<think>", 0) == 0) reason = trim(reason.substr(7));
            }
            check(std::string("basic: server greedy == CLI ") + c.key, trim(r.content) == trim(content) && trim(r.reasoning) == reason,
                  std::format("server {} / {} B, cli {} / {} B", r.reasoning.size(), r.content.size(), reason.size(), content.size()));
        }
    }
    check("basic: thinking on gives reasoning_content", !solo["zh_think"].reasoning.empty());
    check("basic: thinking off gives no reasoning", solo["zh_nothink"].reasoning.empty());
    // seed reproducibility (alone and under load)
    {
        const std::string extra = ",\"temperature\":0.7,\"top_p\":0.9,\"seed\":1234,\"max_tokens\":64" + thinkKw(false);
        ChatResult a = chat(arr({msg("user", "Write a haiku about GPUs.")}), extra);
        ChatResult b = chat(arr({msg("user", "Write a haiku about GPUs.")}), extra);
        check("basic: same seed -> same text", a.status == 200 && a.content == b.content, brief(a));
        record("basic/seed1234", a.text());
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 4; ++i)
            jobs.push_back([&] { return chat(arr({msg("user", "Write a haiku about GPUs.")}), extra); });
        auto res = parallel(jobs);
        bool same = true;
        for (const auto& r : res) same = same && r.content == a.content;
        check("basic: same seed under load (4 concurrent) -> same text", same);
    }
    // tool round trip
    {
        const std::string tk = greedy(200) + thinkKw(false) + ",\"tools\":" + WEATHER_TOOL;
        const std::string m1 = msg("user", "What is the weather in Taipei right now? Use the tool.");
        ChatResult r = chat(arr({m1}), tk);
        const bool called = !r.tool_calls.empty() && r.tool_calls[0][1] == "get_weather" &&
                            r.tool_calls[0][2].find("Taipei") != std::string::npos && r.finish == "tool_calls";
        check("basic: tool call parsed (finish tool_calls)", called, brief(r) + (r.tool_calls.empty() ? "" : " args " + r.tool_calls[0][2]));
        record("basic/tool_call", r.text());
        if (called) {
            const std::string am = "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":" + js(r.tool_calls[0][0]) +
                                   ",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":" + js(r.tool_calls[0][2]) + "}}]}";
            const std::string tm = "{\"role\":\"tool\",\"tool_call_id\":" + js(r.tool_calls[0][0]) +
                                   ",\"content\":\"{\\\"city\\\": \\\"Taipei\\\", \\\"temp_c\\\": 24, \\\"sky\\\": \\\"light rain\\\"}\"}";
            ChatResult r2 = chat(arr({m1, am, tm}), tk);
            check("basic: tool result -> final answer", r2.status == 200 && !r2.content.empty() && r2.tool_calls.empty(), brief(r2));
            check("basic: tool round trip reuses the prefix", r2.cached > 0, std::format("cached {}", r2.cached));
            record("basic/tool_answer", r2.text());
        }
    }
    // multi-turn (greedy, thinking off): each turn reuses the previous conversation
    {
        std::vector<std::string> msgs = {msg("system", "You are a concise assistant.")};
        const std::vector<std::string> users = {"Name three planets.", "Which is the largest? One sentence.", "How many moons, roughly?"};
        ChatResult prev;
        bool ok = true;
        std::string detail;
        for (std::size_t i = 0; i < users.size(); ++i) {
            msgs.push_back(msg("user", users[i]));
            ChatResult r = chat(arr(msgs), greedy(120) + thinkKw(false));
            if (i > 0 && prev.finish == "stop" && r.cached < prev.prompt + prev.completion) ok = false;
            detail += std::format("[{} {}/{}] ", i, r.cached, r.prompt);
            record(std::format("basic/multiturn{}", i), r.text());
            msgs.push_back(msg("assistant", r.content));
            prev = r;
        }
        check("basic: multi-turn reuses the whole previous conversation", ok, detail);
    }
    // concurrency: 4 different prompts at once == solo
    {
        const std::vector<std::string> ps = {"Explain a hash map in two sentences.", "What is RAII? Two sentences.",
                                             "Write a Python one-liner that reverses a string.", P_ZH};
        std::vector<ChatResult> alone;
        for (const auto& p : ps) alone.push_back(chat(arr({msg("user", p)}), greedy(96) + thinkKw(false)));
        std::vector<std::function<ChatResult()>> jobs;
        for (const auto& p : ps) jobs.push_back([&, p] { return chat(arr({msg("user", p)}), greedy(96) + thinkKw(false)); });
        auto res = parallel(jobs);
        std::string d;
        bool same = true;
        for (std::size_t i = 0; i < ps.size(); ++i) {
            same = same && res[i].content == alone[i].content;
            d += res[i].content == alone[i].content ? "=" : "x";
            record(std::format("basic/conc{}", i), alone[i].text());
        }
        check("basic: 4 concurrent requests == each alone", same, d);
    }
}

// ---------------------------------------------------------------------------
// suite mt_cache: agent-style multi-turn sessions keep their history cached

const std::string MT_TOOLS =
    R"([{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file from the workspace.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"workspace-relative path"}},"required":["path"]}}},{"type":"function","function":{"name":"run_tests","description":"Run the project's unit tests and return the summary.","parameters":{"type":"object","properties":{"filter":{"type":"string","description":"test name filter"}},"required":[]}}}])";
const std::string MT_FILE = "def parse_range(s):\n    \"\"\"'3-7' -> [3, 4, 5, 6, 7]\"\"\"\n    a, b = s.split(\"-\")\n    return list(range(int(a), int(b)))\n";

struct Turn {
    std::int64_t prompt, cached, completion;
    std::string finish;
};

std::vector<Turn> runSession(const std::string& name, const std::string& sysmsg, const std::vector<std::string>& users, bool think,
                             bool send_reasoning, bool tools, int max_tokens) {
    std::vector<std::string> msgs;
    if (!sysmsg.empty()) msgs.push_back(msg("system", sysmsg));
    std::vector<Turn> turns;
    int k = 0;
    for (const auto& q : users) {
        msgs.push_back(msg("user", q));
        for (int rt = 0; rt < 3; ++rt) {
            const std::string extra = greedy(max_tokens) + thinkKw(think) + (tools ? ",\"tools\":" + MT_TOOLS : "");
            ChatResult r = chat(arr(msgs), extra);
            turns.push_back({r.prompt, r.cached, r.completion, r.finish});
            record(std::format("mt_cache/{}/{}", name, k++), r.text());
            std::string am = "{\"role\":\"assistant\",\"content\":" + (r.content.empty() ? std::string("null") : js(r.content));
            if (send_reasoning && !r.reasoning.empty()) am += ",\"reasoning_content\":" + js(r.reasoning);
            if (!r.tool_calls.empty()) {
                am += ",\"tool_calls\":[";
                for (std::size_t i = 0; i < r.tool_calls.size(); ++i)
                    am += (i ? "," : "") + std::string("{\"id\":") + js(r.tool_calls[i][0]) +
                          ",\"type\":\"function\",\"function\":{\"name\":" + js(r.tool_calls[i][1]) + ",\"arguments\":" +
                          js(r.tool_calls[i][2]) + "}}";
                am += "]";
            }
            am += "}";
            msgs.push_back(am);
            if (r.tool_calls.empty()) break;
            for (const auto& tc : r.tool_calls) {
                const std::string out = tc[1] == "read_file" ? MT_FILE
                                                             : "2 passed, 1 failed: test_parse_range_inclusive (expected [3,4,5,6,7], got [3,4,5,6])";
                msgs.push_back("{\"role\":\"tool\",\"tool_call_id\":" + js(tc[0]) + ",\"content\":" + js(out) + "}");
            }
        }
    }
    return turns;
}

void evaluate(const std::string& name, const std::vector<Turn>& turns, const std::string& mode) {
    bool ok = true;
    std::int64_t lp = 0, lc = 0;
    int misses = 0;
    for (std::size_t i = 0; i < turns.size(); ++i) {
        const Turn& t = turns[i];
        std::optional<std::int64_t> need;
        if (i > 0 && (turns[i - 1].finish == "stop" || turns[i - 1].finish == "tool_calls"))
            need = turns[i - 1].prompt + turns[i - 1].completion;
        if (i > 0) {
            lp += t.prompt;
            lc += t.cached;
            if (t.cached < turns[i - 1].prompt - 1) ok = false;
        }
        const bool hit = !need || t.cached >= *need;
        if (!hit) ++misses;
        std::printf("    turn %zu: prompt %5lld  cached %5lld  completion %4lld  finish %-10s%s\n", i,
                    static_cast<long long>(t.prompt), static_cast<long long>(t.cached), static_cast<long long>(t.completion),
                    t.finish.c_str(), need ? std::format("  history {:5} {}", *need, hit ? "HIT" : "MISS").c_str() : "");
    }
    if (mode == "full" && misses > 0) ok = false;
    if (mode == "agent" && misses > 1) ok = false;
    check(std::format("mt_cache: {} ({}): prefix reuse across turns", name, mode), ok,
          std::format("hit ratio {:.3f}, full-history misses {}", lp ? static_cast<double>(lc) / lp : 0.0, misses));
}

void suiteMtCache() {
    logLine("== suite mt_cache");
    ServerLease s("mt_cache", {}, {});
    const std::string big = "You are a coding agent working in a Python repository. Project notes follow.\n\n" + docsText(14000);
    evaluate("agent", runSession("agent", big, {"The test for parse_range fails. Read src/ranges.py and tell me why.",
                                                "Fix it and show only the corrected function.", "Now run the tests again."},
                                 false, true, true, 400),
             "agent");
    evaluate("think", runSession("think", "", {"What is 17 * 23?", "Add 100 to that.", "Is the result a prime number?"}, true, true,
                                 false, 1200),
             "full");
    evaluate("nothink", runSession("nothink", "You are a concise assistant.",
                                   {"Name three planets of the solar system.", "Which of them is the largest? One sentence.",
                                    "How many moons does it have, roughly?", "Thanks. One more fact about it, one sentence."},
                                   false, true, false, 150),
             "full");
    evaluate("noreason", runSession("noreason", "", {"What is 12 * 12?", "Double it.", "Now subtract 50."}, true, false, false, 1200),
             "prev");
}

// ---------------------------------------------------------------------------
// suite pool: shared paged KV pool, eviction, restore, long request

ChatResult ask(const std::string& text, const std::string& q, int max_tokens = 120) {
    return chat(arr({msg("user", text + "\n\n" + q)}), greedy(max_tokens) + thinkKw(false));
}

void suitePool() {
    logLine("== suite pool");
    const std::string q = "In two sentences, what does the first file above do?";
    const std::vector<std::string> P = {codeText(16000, 0), codeText(16000, 7), codeText(26000, 14)};
    {
        ServerLease s("pool", {"--parallel", "4", "--ctx", "16384"}, {});
        auto a1 = ask(P[0], q), a2 = ask(P[1], q), a3 = ask(P[2], q);
        logLine(std::format("  P1 {} tok, P2 {} tok, P3 {} tok", a1.prompt, a2.prompt, a3.prompt));
        auto b2 = ask(P[1], q), b1 = ask(P[0], q);
        for (auto [k, r] : {std::pair{"P1", &a1}, std::pair{"P2", &a2}, std::pair{"P3", &a3}}) record(std::string("pool/") + k, r->text());
        check("pool: pool full -> an idle slot was evicted", s->logCount("kv pool full: evicting") >= 1);
        check("pool: P2 again: prompt fully reused", b2.cached >= b2.prompt - 1, std::format("cached {} of {}", b2.cached, b2.prompt));
        check("pool: P2 again: same text", b2.content == a2.content);
        check("pool: P1 again (evicted): restored from the RAM tier, same text",
              b1.cached >= b1.prompt - 1 && b1.content == a1.content && s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= 1,
              std::format("cached {} of {}", b1.cached, b1.prompt));
        std::vector<std::function<ChatResult()>> jobs = {[&] { return ask(P[1], q); }, [&] { return ask(P[2], q); }};
        auto res = parallel(jobs);
        check("pool: P2, P3 concurrently == solo", res[0].content == a2.content && res[1].content == a3.content);
    }
    {
        ServerLease s("pool_long", {"--parallel", "4", "--ctx", "65536", "--ctx-per-slot", "65536"}, {});
        std::string big = codeText(135000, 0);
        std::size_t k = big.rfind('\n', big.size() / 2) + 1;
        big = big.substr(0, k) + "// NOTE: the deployment passphrase for the staging cluster is \"amber-falcon-7291\".\n" + big.substr(k);
        auto r = ask(big, "What is the deployment passphrase for the staging cluster stated in a comment above? Reply with the passphrase only.", 30);
        record("pool/long", r.text());
        check(std::format("pool: one {}-token request on a 65536-token pool shared by 4 slots", r.prompt),
              r.content.find("amber-falcon-7291") != std::string::npos, brief(r));
    }
}

// ---------------------------------------------------------------------------
// suite tier: host tiers restore exactly (RAM after eviction, SSD after restart)

void tierMode(const std::string& mode) {
    const bool p1 = mode == "p1";
    const std::vector<std::string> args = p1 ? std::vector<std::string>{"--parallel", "1"} : std::vector<std::string>{"--parallel", "4"};
    struct Sess {
        std::string name, m1;
        bool think;
    };
    std::vector<Sess> S;
    const std::string q1 = "Briefly, what does the first file above do? Answer in two sentences.";
    const std::string q2 = "Now name one function defined in that file and say what it returns, in one sentence.";
    if (p1) {
        S = {{"A", codeText(20000, 0), false}, {"T", codeText(16000, 9), true}};
    } else {
        const char* names = "ABCDE";
        for (int i = 0; i < 5; ++i) S.push_back({std::string(1, names[i]), codeText(14000 + 2000 * i, 3 * i), false});
    }
    auto t1msg = [&](const Sess& ss) { return arr({msg("user", ss.m1 + "\n\n" + q1)}); };
    auto t2msg = [&](const Sess& ss, const ChatResult& a) {
        return arr({msg("user", ss.m1 + "\n\n" + q1), msg("assistant", a.content), msg("user", q2)});
    };
    std::map<std::string, std::pair<ChatResult, ChatResult>> ref;
    {
        ServerLease s("tier_" + mode + "_ref", [&] {
            auto a = args;
            a.insert(a.end(), {"--kv-ram-mb", "0"});
            return a;
        }(), {});
        for (const auto& ss : S) {
            auto a = chat(t1msg(ss), greedy(64) + thinkKw(ss.think));
            auto b = chat(t2msg(ss, a), greedy(96) + thinkKw(ss.think));
            ref[ss.name] = {a, b};
            record("tier/" + mode + "/" + ss.name + "/1", a.text());
            record("tier/" + mode + "/" + ss.name + "/2", b.text());
        }
    }
    const std::string ssd = g.work + "\\tier_gate_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    std::map<std::string, ChatResult> t1;
    {
        ServerLease s("tier_" + mode, args, {{"WHIRL_KV_SSD_DIR", ssd}}, false);
        for (const auto& ss : S) {
            t1[ss.name] = chat(t1msg(ss), greedy(64) + thinkKw(ss.think));
            check(std::format("tier {}: {} turn 1 == reference", mode, ss.name), t1[ss.name].text() == ref[ss.name].first.text());
        }
        // (the spill is logged after the response is sent: wait for the line)
        check(std::format("tier {}: every turn 1 spilled to the RAM tier", mode), s->waitLog("kv tier: spill ", S.size(), 30),
              std::format("{} spills logged", s->logCount("kv tier: spill ")));
        const std::size_t n_ev = p1 ? S.size() - 1 : 1;
        for (std::size_t i = 0; i < n_ev; ++i) {
            auto b = chat(t2msg(S[i], t1[S[i].name]), greedy(96) + thinkKw(S[i].think));
            const auto& rb = ref[S[i].name].second;
            check(std::format("tier {}: {} turn 2 after eviction (RAM restore) == never-evicted", mode, S[i].name),
                  b.text() == rb.text() && b.cached == rb.cached, std::format("cached {} vs {}", b.cached, rb.cached));
        }
        check(std::format("tier {}: RAM restores logged", mode), s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= n_ev);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    std::size_t files = 0;
    if (fs::exists(widen(ssd)))
        for (const auto& e : fs::directory_iterator(widen(ssd))) files += e.is_regular_file() ? 1 : 0;
    check(std::format("tier {}: entries on SSD after the restart", mode), files >= 1, std::format("{} files", files));
    {
        ServerLease s("tier_" + mode + "_restart", args, {{"WHIRL_KV_SSD_DIR", ssd}}, false);
        for (const auto& ss : S) {
            auto b = chat(t2msg(ss, t1[ss.name]), greedy(96) + thinkKw(ss.think));
            const auto& rb = ref[ss.name].second;
            check(std::format("tier {}: {} turn 2 after restart (SSD restore) == never-evicted", mode, ss.name),
                  b.content == rb.content && b.reasoning == rb.reasoning,
                  std::format("cached {} (ref {}), prompt {}", b.cached, rb.cached, b.prompt));
        }
        check(std::format("tier {}: SSD restores logged", mode), s->logCount(R"(kv tier: restored \d+ tok from SSD)") >= 1);
    }
    fs::remove_all(widen(ssd), ec);
}

void suiteTier() {
    logLine("== suite tier");
    tierMode("p1");
    tierMode("p4");
}

// ---------------------------------------------------------------------------
// suite sys: shared system-prompt checkpoints give exactly the cold result

std::string sysTools() {
    const char* names[] = {"read_file", "write_file", "list_dir", "grep", "run_shell", "git_status", "git_diff", "web_fetch",
                           "search_docs", "create_issue", "run_tests", "format_code"};
    std::string s = "[";
    for (int i = 0; i < 12; ++i) {
        std::string d = std::format("MCP tool {}: ", names[i]);
        for (int k = 0; k < 3; ++k) d += "performs the named workspace operation and returns its result as text. ";
        s += (i ? "," : "") + std::string("{\"type\":\"function\",\"function\":{\"name\":") + js(names[i]) + ",\"description\":" + js(d) +
             R"(,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"workspace-relative path"},"pattern":{"type":"string","description":"optional pattern or argument"},"limit":{"type":"integer","description":"maximum number of lines or results"}},"required":["path"]}}})";
    }
    return s + "]";
}

struct SysCase {
    std::string name, sys, user;
    bool think;
};

ChatResult sysAsk(const SysCase& c) {
    return chat(arr({msg("system", c.sys), msg("user", c.user)}), greedy(c.think ? 96 : 48) + thinkKw(c.think) + ",\"tools\":" + sysTools());
}

bool same(const ChatResult& a, const ChatResult& b) { return a.text() == b.text() && a.prompt == b.prompt; }

std::optional<std::int64_t> boundary(Server& s) {
    std::optional<std::int64_t> b;
    const std::regex r(R"(kept system checkpoint @(\d+))");
    for (const auto& l : s.logLines()) {
        std::smatch m;
        if (std::regex_search(l, m, r)) b = std::stoll(m[1].str());
    }
    return b;
}

void suiteSys() {
    logLine("== suite sys");
    const std::string SYS1 =
        "You are a coding agent working in the user's repository. Use the tools to look at files before answering. Answer briefly and "
        "precisely. In the chat format a turn ends with the token <|im_end|> (a system text that mentions it must not end the system "
        "message early). Below are the skills manuals available in this workspace.\n" +
        docsText(40000);
    const std::string SYS2 = "You are a code reviewer. Below are the review guidelines of this team.\n" + docsText(30000, 3);
    const std::vector<std::string> U = {"In one sentence, what is the main topic of the first skill manual above?",
                                        "Name two tools you can call, in one sentence.",
                                        "Which skill above talks about building? Answer in one short sentence.",
                                        "Answer in one sentence: what is your role?",
                                        "What would you do first to find where the KV cache is implemented? One sentence.",
                                        "List three tool names above, comma separated.",
                                        "Is there a skill about hardware? Answer yes or no and name it.",
                                        "Summarize the build skill in one sentence.",
                                        "Which tool would you use to run the test suite? One sentence.",
                                        "Which tool shows uncommitted changes? One sentence."};
    std::vector<SysCase> cs;
    for (int i = 0; i < 4; ++i) cs.push_back({"S" + std::to_string(i), SYS1, U[i], false});
    for (int i = 0; i < 4; ++i) cs.push_back({"C" + std::to_string(i), SYS1, U[4 + i], false});
    for (int i = 0; i < 2; ++i) cs.push_back({"X" + std::to_string(i), SYS1, U[8 + i], false});
    cs.push_back({"T", SYS1, U[0], true});
    for (int i = 0; i < 3; ++i) cs.push_back({"R" + std::to_string(i), SYS2, U[1 + i], false});
    auto byName = [&](const std::string& n) -> const SysCase& {
        for (const auto& c : cs)
            if (c.name == n) return c;
        throw std::runtime_error("case " + n);
    };
    const std::map<std::string, std::string> kv = kvPin();
    std::map<std::string, ChatResult> ref;
    {
        auto env = kv;
        env["WHIRL_NO_PREFIX_CACHE"] = "1";
        ServerLease s("sys_ref", {"--parallel", "4", "--kv-ram-mb", "0"}, env);
        for (const auto& c : cs) {
            ref[c.name] = sysAsk(c);
            record("sys/" + c.name, ref[c.name].text());
        }
    }
    const std::string ssd = g.work + "\\sys_gate_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    std::optional<std::int64_t> B;
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd;
        ServerLease s("sys_tier", {"--parallel", "4"}, env, false);
        auto first = sysAsk(byName("S0"));
        B = boundary(*s.s);
        check("sys: session 0 cold == reference, keeps a system checkpoint", same(first, ref["S0"]) && first.cached == 0 && B.has_value(),
              std::format("boundary {}, cached {}", B ? *B : -1, first.cached));
        for (int i = 1; i < 4; ++i) {
            const auto& c = byName("S" + std::to_string(i));
            auto r = sysAsk(c);
            check("sys: " + c.name + " reuses the system checkpoint, == cold", same(r, ref[c.name]) && B && r.cached == *B,
                  std::format("cached {} (boundary {})", r.cached, B ? *B : -1));
        }
        auto t = sysAsk(byName("T"));
        check("sys: thinking session reuses it, == cold", same(t, ref["T"]) && B && t.cached == *B, std::format("cached {}", t.cached));
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 4; ++i) jobs.push_back([&, i] { return sysAsk(byName("C" + std::to_string(i))); });
        auto res = parallel(jobs);
        bool ok = true;
        std::string d;
        for (int i = 0; i < 4; ++i) {
            ok = ok && same(res[i], ref["C" + std::to_string(i)]) && B && res[i].cached == *B;
            d += std::format("C{} {} ", i, res[i].cached);
        }
        check("sys: 4 concurrent sessions == cold, all reuse", ok, d);
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd + "_ram";
        env["WHIRL_SYS_CKPTS"] = "1";
        ServerLease s("sys_ram", {"--parallel", "4"}, env, false);
        auto a = sysAsk(byName("S0"));
        auto b = sysAsk(byName("R0"));
        auto c = sysAsk(byName("S1"));
        const std::size_t n = s->logCount(R"(kv tier: restored \d+ tok from RAM)");
        check("sys: ram: replaced system checkpoint comes back from the RAM tier, == cold",
              same(a, ref["S0"]) && same(b, ref["R0"]) && same(c, ref["S1"]) && B && c.cached == *B && n >= 1,
              std::format("cached {}, RAM restores {}", c.cached, n));
    }
    fs::remove_all(widen(ssd + "_ram"), ec);
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd;
        ServerLease s("sys_restart", {"--parallel", "4"}, env, false);
        auto a = sysAsk(byName("X0"));
        auto b = sysAsk(byName("X1"));
        const std::size_t n = s->logCount(R"(kv tier: restored \d+ tok from SSD)");
        check("sys: ssd: after a restart the system checkpoint is restored from SSD, == cold",
              same(a, ref["X0"]) && same(b, ref["X1"]) && B && a.cached == *B && b.cached == *B && n >= 1,
              std::format("cached {} / {}, SSD restores {}", a.cached, b.cached, n));
    }
    fs::remove_all(widen(ssd), ec);
    {
        ServerLease s("sys_off", {"--parallel", "4", "--kv-ram-mb", "0"}, kv);
        std::vector<ChatResult> got;
        for (int i = 0; i < 3; ++i) got.push_back(sysAsk(byName("S" + std::to_string(i))));
        check("sys: tiers off: VRAM sharing == cold",
              same(got[0], ref["S0"]) && same(got[1], ref["S1"]) && same(got[2], ref["S2"]) && B && got[1].cached == *B && got[2].cached == *B,
              std::format("{} {} {}", got[0].cached, got[1].cached, got[2].cached));
        std::vector<ChatResult> res(3);
        std::vector<std::thread> th;
        for (int i = 0; i < 3; ++i) {
            th.emplace_back([&, i] { res[i] = sysAsk(byName("R" + std::to_string(i))); });
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        for (auto& t : th) t.join();
        std::vector<std::int64_t> cached;
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            ok = ok && same(res[i], ref["R" + std::to_string(i)]);
            cached.push_back(res[i].cached);
        }
        std::sort(cached.begin(), cached.end());
        const std::size_t n_wait = s->logCount("waiting for the system-message checkpoint");
        check("sys: burst of 3 new sessions: one prefills the system prompt, the others wait and reuse it, == cold",
              ok && cached[0] == 0 && cached[1] > 0 && cached[1] == cached[2] && n_wait >= 1,
              std::format("cached {} {} {}, waits logged {}", cached[0], cached[1], cached[2], n_wait));
    }
}

// ---------------------------------------------------------------------------
// suite restore_conc: tier restores next to running requests, delay hit, prefetch

void suiteRestoreConc() {
    logLine("== suite restore_conc");
    const std::vector<std::string> DEC = {
        "Write a Python LRU cache class with comments and two pytest tests.",
        "Write a thread-safe counter in Rust with comments and a usage example.",
        "Write a simple event bus (on / off / emit) in TypeScript with comments.",
        "Write a Go program that reads a CSV file and prints the average of each column, with comments."};
    const std::string ctx = codeText(30000, 0);
    const std::string m1u = ctx + "\n\nBriefly, what does the first file above do? Two sentences.";
    const std::vector<std::string> qs = {"Name one function defined in that file, one sentence.", "Which file above is the longest? One sentence."};
    const std::string sysA = "You are a careful coding assistant. Reference material follows.\n\n" + codeText(26000, 20);
    const std::string sysB = "You are a terse reviewer. Reference material follows.\n\n" + codeText(9000, 60);
    const std::vector<std::string> bq = {"What is the first file about? One sentence.", "Name one class from the material. One sentence.",
                                         "Is there any template code? One sentence."};
    const std::vector<std::string> bq2 = {"Name one function from the material. One sentence.",
                                          "Which file in the material is the shortest? One sentence.", "Is there any error handling code? One sentence."};
    auto turn = [&](const std::string& msgs, int n) { return chat(msgs, greedy(n) + thinkKw(false)); };
    auto burst = [&](const std::vector<std::string>& q, std::size_t i) { return arr({msg("system", sysA), msg("user", q[i])}); };
    const std::map<std::string, std::string> kv = kvPin();
    ChatResult r1, r2, r3;
    std::vector<ChatResult> rdec, rb, rb2;
    std::string m2, m3;
    {
        ServerLease s("rconc_ref", {"--parallel", "4", "--kv-ram-mb", "0"}, kv);
        r1 = turn(arr({msg("user", m1u)}), 48);
        m2 = arr({msg("user", m1u), msg("assistant", r1.content), msg("user", qs[0])});
        r2 = turn(m2, 48);
        m3 = arr({msg("user", m1u), msg("assistant", r1.content), msg("user", qs[0]), msg("assistant", r2.content), msg("user", qs[1])});
        r3 = turn(m3, 48);
        for (const auto& d : DEC) rdec.push_back(turn(arr({msg("user", d)}), 160));
        for (std::size_t i = 0; i < 3; ++i) rb.push_back(turn(burst(bq, i), 48));
        for (std::size_t i = 0; i < 3; ++i) rb2.push_back(turn(burst(bq2, i), 48));
        record("restore_conc/S1", r1.text());
        record("restore_conc/S2", r2.text());
        record("restore_conc/S3", r3.text());
        for (std::size_t i = 0; i < DEC.size(); ++i) record(std::format("restore_conc/dec{}", i), rdec[i].text());
    }
    const std::string ssd = g.work + "\\restore_conc_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    auto env = kv;
    env["WHIRL_KV_SSD_DIR"] = ssd;
    env["WHIRL_SYS_CKPTS"] = "1";
    {
        ServerLease s("rconc", {"--parallel", "4"}, env, false);
        auto a1 = turn(arr({msg("user", m1u)}), 48);
        check("restore_conc: S turn 1 == reference", a1.text() == r1.text());
        for (int i = 0; i < 4; ++i) turn(arr({msg("user", codeText(1800, 40 + 3 * i) + "\n\nOne sentence: what is this?")}), 8);
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 3; ++i) jobs.push_back([&, i] { return turn(arr({msg("user", DEC[i])}), 160); });
        jobs.push_back([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            return turn(m2, 48);
        });
        auto res = parallel(jobs);
        check("restore_conc: S turn 2 restored next to 3 decoding requests == never-evicted",
              res[3].text() == r2.text() && res[3].cached == r2.cached, std::format("cached {} vs {}", res[3].cached, r2.cached));
        check("restore_conc: decode texts next to the restore == alone",
              res[0].content == rdec[0].content && res[1].content == rdec[1].content && res[2].content == rdec[2].content);
        check("restore_conc: RAM restore logged", s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= 1);
        auto b0 = turn(burst(bq, 0), 48);
        check("restore_conc: burst session 0 == reference", b0.content == rb[0].content);
        turn(arr({msg("system", sysB), msg("user", "One word: ok?")}), 8);
        const std::size_t w0 = s->logCount("waiting for the restore of its prefix");
        const std::size_t rr0 = s->logCount(R"(kv tier: restored \d+ tok from RAM)");
        std::vector<std::function<ChatResult()>> j2 = {[&] { return turn(burst(bq, 1), 48); }, [&] { return turn(burst(bq, 2), 48); },
                                                       [&] { return turn(burst(bq, 0), 48); }};
        auto res2 = parallel(j2);
        check("restore_conc: burst on a RAM shared entry: texts == reference",
              res2[0].content == rb[1].content && res2[1].content == rb[2].content && res2[2].content == rb[0].content);
        const std::size_t nw = s->logCount("waiting for the restore of its prefix") - w0;
        const std::size_t nr = s->logCount(R"(kv tier: restored \d+ tok from RAM)") - rr0;
        check("restore_conc: burst on a RAM shared entry: one restore, the others wait", nr == 1 && nw >= 1,
              std::format("restores {}, waiting {}", nr, nw));
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    {
        ServerLease s("rconc_restart", {"--parallel", "4"}, env, false);
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 4; ++i) jobs.push_back([&, i] { return turn(arr({msg("user", DEC[i])}), 160); });
        jobs.push_back([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            return turn(m3, 48);
        });
        auto res = parallel(jobs);
        check("restore_conc: S turn 3 after restart (queued, prefetched, restored) == never-evicted", res[4].content == r3.content,
              std::format("cached {} (ref {})", res[4].cached, r3.cached));
        const std::size_t npf = s->logCount(R"(prefetching \d+ tok from SSD)");
        const std::size_t nrs = s->logCount(R"(kv tier: restored \d+ tok from (SSD|RAM))");
        check("restore_conc: SSD prefetch while queued + restore logged", npf >= 1 && nrs >= 1, std::format("prefetch {}, restores {}", npf, nrs));
        const std::size_t w0 = s->logCount("waiting for the restore of its prefix");
        std::vector<std::function<ChatResult()>> j2;
        for (std::size_t i = 0; i < 3; ++i) j2.push_back([&, i] { return turn(burst(bq2, i), 48); });
        auto res2 = parallel(j2);
        check("restore_conc: burst of new sessions on an SSD shared entry: texts == reference",
              res2[0].content == rb2[0].content && res2[1].content == rb2[1].content && res2[2].content == rb2[2].content);
        check("restore_conc: burst on an SSD shared entry: the others wait for the one restore",
              s->logCount("waiting for the restore of its prefix") - w0 >= 1);
    }
    fs::remove_all(widen(ssd), ec);
}

// ---------------------------------------------------------------------------
// suite vis: image input (--mmproj, image_url parts). Servers: base (no mmproj),
// ref (mmproj, no prefix cache, no tiers), plain (no MTP), main (defaults, idle
// release after 8 s, fresh SSD dir), restart (same SSD dir). KV format pinned (--kv,
// default q8v: a ref server with more free VRAM would otherwise pick f16).
// Needs --mmproj and the test images in --images (shapes.png, dialog.png, s1080.png).

std::string b64(const std::string& bytes) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    std::size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const std::uint32_t v = (static_cast<std::uint8_t>(bytes[i]) << 16) | (static_cast<std::uint8_t>(bytes[i + 1]) << 8) |
                                static_cast<std::uint8_t>(bytes[i + 2]);
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += T[(v >> 6) & 63];
        o += T[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const std::uint32_t v = static_cast<std::uint8_t>(bytes[i]) << 16;
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += "==";
    } else if (i + 2 == bytes.size()) {
        const std::uint32_t v = (static_cast<std::uint8_t>(bytes[i]) << 16) | (static_cast<std::uint8_t>(bytes[i + 1]) << 8);
        o += T[v >> 18];
        o += T[(v >> 12) & 63];
        o += T[(v >> 6) & 63];
        o += '=';
    }
    return o;
}

std::string imgPart(const std::string& file) {
    static std::map<std::string, std::string> cache;
    auto it = cache.find(file);
    if (it == cache.end()) {
        const std::string mime = file.ends_with(".jpg") ? "image/jpeg" : "image/png";
        it = cache.emplace(file, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:" + mime + ";base64," +
                                     b64(readFile(g.images + "\\" + file)) + "\"}}")
                 .first;
    }
    return it->second;
}

// user message: images then the question
std::string imgMsg(const std::vector<std::string>& files, const std::string& q) {
    std::string c = "[";
    for (const auto& f : files) c += imgPart(f) + ",";
    c += "{\"type\":\"text\",\"text\":" + js(q) + "}]";
    return "{\"role\":\"user\",\"content\":" + c + "}";
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool hasAll(const ChatResult& r, std::initializer_list<const char*> words) {
    const std::string t = lower(r.content);
    for (const char* w : words)
        if (t.find(lower(w)) == std::string::npos) return false;
    return true;
}

std::optional<std::int64_t> logNum(Server& s, const std::string& re) {
    std::optional<std::int64_t> v;
    const std::regex r(re);
    for (const auto& l : s.logLines()) {
        std::smatch m;
        if (std::regex_search(l, m, r)) v = std::stoll(m[1].str());
    }
    return v;
}

std::optional<double> logReal(Server& s, const std::string& re) {
    std::optional<double> v;
    const std::regex r(re);
    for (const auto& l : s.logLines()) {
        std::smatch m;
        if (std::regex_search(l, m, r)) v = std::stod(m[1].str());
    }
    return v;
}

void suiteVis() {
    logLine("== suite vis");
    if (g.mmproj.empty()) throw std::runtime_error("suite vis needs --mmproj");
    if (g.images.empty()) throw std::runtime_error("suite vis needs --images DIR (or WHIRL_GATE_IMAGES)");
    const std::string Q_SHAPES = "Describe this image: the shapes, their colors, and any text.";
    const std::string Q_DIALOG = "What error does this dialog show? Quote the file name and the error code.";
    const std::string Q_TWO = "Two images: say in one sentence what each one shows.";
    const std::string Q_1080 = "Describe this image in two sentences.";
    const std::string Q_1080B = "What is the dominant color in this image? One sentence.";
    const std::string Q_TEXT = "Explain what a hash map is in two sentences.";
    auto ask = [&](const std::string& msgs, int n) { return chat(msgs, greedy(n) + thinkKw(false)); };
    const std::string m_shapes = arr({imgMsg({"shapes.png"}, Q_SHAPES)});
    const std::string m_dialog = arr({imgMsg({"dialog.png"}, Q_DIALOG)});
    const std::string m_two = arr({imgMsg({"shapes.png", "dialog.png"}, Q_TWO)});
    const std::string m_1080 = arr({imgMsg({"s1080.png"}, Q_1080)});
    const std::string m_1080b = arr({imgMsg({"s1080.png"}, Q_1080B)});
    const std::string m_text = arr({msg("user", Q_TEXT)});
    const std::map<std::string, std::string> kv = kvPin();
    const std::vector<std::string> mm = {"--mmproj", g.mmproj};
    std::optional<std::int64_t> pool_base;
    ChatResult base_text;
    {
        ServerLease s("vis_base", {}, kv);
        pool_base = logNum(*s.s, R"(kv pool: (\d+) tokens)");
        base_text = ask(m_text, 96);
        auto r = ask(m_shapes, 32);
        check("vis: base (no mmproj): image part -> 400", r.status == 400 && r.raw.find("mmproj") != std::string::npos, r.raw.substr(0, 120));
    }
    std::map<std::string, ChatResult> ref;
    {
        auto env = kv;
        env["WHIRL_NO_PREFIX_CACHE"] = "1";
        auto args = mm;
        args.insert(args.end(), {"--kv-ram-mb", "0"});
        ServerLease s("vis_ref", args, env);
        ref["shapes"] = ask(m_shapes, 160);
        ref["dialog"] = ask(m_dialog, 160);
        ref["two"] = ask(m_two, 160);
        ref["s1080"] = ask(m_1080, 96);
        ref["s1080b"] = ask(m_1080b, 64);
        ref["text"] = ask(m_text, 96);
        for (const auto& [k, r] : ref) record("vis/" + k, r.text());
        check("vis: ref: shapes content (red, blue, 42)", hasAll(ref["shapes"], {"red", "blue", "42"}), brief(ref["shapes"]));
        check("vis: ref: dialog content (config.yaml, 0x80070002)", hasAll(ref["dialog"], {"config.yaml", "0x80070002"}), brief(ref["dialog"]));
        check("vis: ref: two images", ref["two"].status == 200 && !ref["two"].content.empty(), brief(ref["two"]));
    }
    {
        auto env = kv;
        env["WHIRL_MTP"] = "0";
        ServerLease s("vis_plain", mm, env);
        auto a = ask(m_shapes, 160);
        auto b = ask(m_dialog, 160);
        record("vis/plain_shapes", a.text());
        check("vis: MTP greedy == plain greedy (shapes, dialog)", a.text() == ref["shapes"].text() && b.text() == ref["dialog"].text(),
              brief(a));
    }
    const std::string ssd = g.work + "\\vis_gate_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd;
        auto args = mm;
        args.insert(args.end(), {"--vis-idle-s", "8"});
        ServerLease s("vis_main", args, env, false);
        const auto pool = logNum(*s.s, R"(kv pool: (\d+) tokens)");
        // The pool is sized from the VRAM free at startup, which moves by a few pages between
        // two server starts (other processes; the prototype shows the same 201216 / 199936
        // spread without --mmproj). The encoder is loaded after the pool is allocated, so
        // equal, or <= 5 pages apart with the vision line logged after the pool line, passes.
        {
            std::size_t i_pool = 0, i_vis = 0, i = 0;
            for (const auto& l : s->logLines()) {
                ++i;
                if (l.find("kv pool: ") != std::string::npos) i_pool = i;
                if (l.find("vision: ") != std::string::npos && i_vis == 0) i_vis = i;
            }
            const bool eq = pool && pool_base && *pool == *pool_base;
            const bool jitter = pool && pool_base && std::abs(*pool - *pool_base) <= 5 * 256 && i_pool > 0 && i_vis > i_pool;
            check("vis: KV pool with --mmproj == without (encoder loaded after the pool)", eq || jitter,
                  std::format("{} vs {}{}", pool ? *pool : -1, pool_base ? *pool_base : -1,
                              eq ? "" : " (startup VRAM jitter; vision line after the pool line)"));
        }
        const auto dv = logReal(*s.s, R"(VRAM change (-?[\d.]+) MiB)");
        check("vis: VRAM change at mmproj load <= 2 MiB", dv && std::abs(*dv) <= 2.0, std::format("{} MiB", dv ? *dv : -1.0));
        auto t = ask(m_text, 96);
        check("vis: text answer == base, encoder not loaded", t.text() == base_text.text() && s->logCount("encoded in") == 0, brief(t));
        auto a = ask(m_shapes, 160);
        check("vis: shapes == ref", a.text() == ref["shapes"].text(), brief(a));
        auto a2 = ask(m_shapes, 160);
        check("vis: same image again: prompt fully cached, == ref", a2.text() == ref["shapes"].text() && a2.cached >= a2.prompt - 1,
              std::format("cached {} of {}", a2.cached, a2.prompt));
        auto b = ask(m_dialog, 160);
        check("vis: other image: no reuse of the first image, == ref, answer differs",
              b.text() == ref["dialog"].text() && b.cached == 0 && b.content != a.content, std::format("cached {}", b.cached));
        auto c = ask(m_two, 160);
        check("vis: two images == ref, embeddings from the cache", c.text() == ref["two"].text() && s->logCount("embeddings cached") >= 2,
              brief(c));
        auto d = ask(m_1080, 96);
        check("vis: 1080p == ref", d.text() == ref["s1080"].text(), brief(d));
        auto e = ask(m_1080b, 64);
        check("vis: 1080p, new conversation: starts at the image-end checkpoint, == cold",
              e.text() == ref["s1080b"].text() && e.cached >= 2040, std::format("cached {} of {}", e.cached, e.prompt));
        // one image request next to three text requests
        std::vector<std::function<ChatResult()>> jobs = {[&] { return ask(m_dialog, 160); },
                                                         [&] { return ask(arr({msg("user", "What is RAII? Two sentences.")}), 96); },
                                                         [&] { return ask(m_text, 96); },
                                                         [&] { return ask(arr({msg("user", "Name three sorting algorithms.")}), 64); }};
        std::vector<ChatResult> solo;
        for (const auto& j : jobs) solo.push_back(j());
        auto res = parallel(jobs);
        bool same_all = true;
        for (std::size_t i = 0; i < res.size(); ++i) same_all = same_all && res[i].text() == solo[i].text();
        check("vis: 1 image + 3 text requests concurrently == each alone", same_all && res[0].text() == ref["dialog"].text());
        // idle release
        const bool rel = s->waitLog("encoder released", 1, 40);
        const auto delta = logReal(*s.s, R"(encoder released; VRAM free .* delta (-?[\d.]+) MiB)");
        check("vis: idle release: VRAM back to the level before the first image (delta <= 8 MiB)", rel && delta && std::abs(*delta) <= 8.0,
              std::format("delta {} MiB", delta ? *delta : -1.0));
        auto f = ask(m_shapes, 160);
        check("vis: after the release: shapes == ref (encoder reloaded or cache)", f.text() == ref["shapes"].text(), brief(f));
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd;
        ServerLease s("vis_restart", mm, env, false);
        auto e = ask(m_1080b, 64);
        check("vis: restart: 1080p question restored from SSD, == cold",
              e.text() == ref["s1080b"].text() && s->logCount(R"(kv tier: restored \d+ tok from SSD)") >= 1,
              std::format("cached {} of {}", e.cached, e.prompt));
    }
    fs::remove_all(widen(ssd), ec);
}

// suite vis_speed: image TTFT on a default server with --mmproj (one server per run;
// alternate exes between runs). Per image (s512, s1024, s1080): the first request
// (new image: decode, encode, prefill; max_tokens 1, so the request time is the
// time to the first token) and three new conversations about the same image with
// other questions (median). Appends "exe,image,first_ms,same_ms" lines to
// WORK\vis_speed.csv.
void suiteVisSpeed() {
    logLine("== suite vis_speed");
    if (g.mmproj.empty()) throw std::runtime_error("suite vis_speed needs --mmproj");
    ServerLease s("vis_speed", {"--mmproj", g.mmproj}, {});
    // warm-up text request (kernels, graphs)
    chat(arr({msg("user", "Say hi.")}), greedy(4) + thinkKw(false));
    const std::vector<std::string> qs = {"Describe this image in one sentence.", "What is the dominant color? One word.",
                                         "Is there any text in the image? Yes or no.", "How many objects do you see? One number."};
    std::string csv;
    for (const char* img : {"s512.png", "s1024.png", "s1080.png"}) {
        const ChatResult first = chat(arr({imgMsg({img}, qs[0])}), greedy(1) + thinkKw(false));
        std::vector<double> same;
        for (std::size_t i = 1; i < qs.size(); ++i) same.push_back(chat(arr({imgMsg({img}, qs[i])}), greedy(1) + thinkKw(false)).ms);
        std::sort(same.begin(), same.end());
        logLine(std::format("  {}: first {:.1f} ms (prompt {}), same image new conversation {:.1f} ms (median of {}; cached {})", img, first.ms,
                            first.prompt, same[same.size() / 2], same.size(), first.cached));
        check(std::format("vis_speed: {} answered", img), first.status == 200, brief(first));
        csv += std::format("{},{},{:.2f},{:.2f}\n", g.kind, img, first.ms, same[same.size() / 2]);
    }
    const std::string path = g.work + "\\vis_speed.csv";
    std::string old;
    try {
        old = readFile(path);
    } catch (const std::exception&) {
    }
    writeFile(path, old + csv);
}

}  // namespace

int wmain(int argc, wchar_t** wargv) {
    // the servers we start inherit the "ignore Ctrl+C" attribute: clear one inherited
    // from our own parent so that the Ctrl+C stop reaches them
    SetConsoleCtrlHandler(nullptr, FALSE);
    std::vector<std::string> a;
    for (int i = 1; i < argc; ++i) a.push_back(narrow(wargv[i]));
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto val = [&]() -> std::string {
            if (i + 1 >= a.size()) throw std::runtime_error(a[i] + " needs a value");
            return a[++i];
        };
        if (a[i] == "--exe") g.exe = val();
        else if (a[i] == "--kind") g.kind = val();
        else if (a[i] == "--model") g.model = val();
        else if (a[i] == "--work") g.work = val();
        else if (a[i] == "--port") g.port = static_cast<std::uint16_t>(std::stoi(val()));
        else if (a[i] == "--results") g.results = val();
        else if (a[i] == "--compare") g.compare = val();
        else if (a[i] == "--cli") g.cli = val();
        else if (a[i] == "--no-lock") g.lock = false;
        else if (a[i] == "--mmproj") g.mmproj = val();
        else if (a[i] == "--images") g.images = val();
        else if (a[i] == "--kv") g.kv_pin = val();
        else if (a[i] == "--server-env") {
            const std::string kvp = val();
            const std::size_t eq = kvp.find('=');
            if (eq == std::string::npos || eq == 0) throw std::runtime_error("--server-env needs NAME=VALUE");
            g.server_env[kvp.substr(0, eq)] = kvp.substr(eq + 1);
        }
        else if (a[i] == "--suite") {
            const std::string s = val();
            std::size_t p = 0;
            while (p <= s.size()) {
                std::size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                g.suites.push_back(s.substr(p, e - p));
                p = e + 1;
            }
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a[i].c_str());
            return 2;
        }
    }
    if (const char* tr = std::getenv("WHIRL_GATE_TEXT_ROOT")) g.text_root = tr;
    if (g.work.empty()) {
        if (const char* w = std::getenv("WHIRL_GATE_WORK"); w && *w) g.work = w;
        else g.work = narrow((fs::temp_directory_path() / L"whirl-tests" / L"gate").wstring());
    }
    while (g.work.size() > 3 && (g.work.back() == '\\' || g.work.back() == '/')) g.work.pop_back();
    if (!g.exe.empty()) {
        std::error_code ec;
        fs::create_directories(widen(g.work), ec);  // suites write prompt files here before any server starts
    }
    if (g.images.empty())
        if (const char* im = std::getenv("WHIRL_GATE_IMAGES")) g.images = im;
    if (g.text_root.empty() && !g.exe.empty() && !g.model.empty()) {
        std::fprintf(stderr, "set WHIRL_GATE_TEXT_ROOT to the root of a llama.cpp source checkout (test texts)\n");
        return 2;
    }
    if ((g.exe.empty() || g.model.empty()) && g.compare.empty()) {
        std::fprintf(stderr,
                     "usage: whirl-server-gate --exe PATH --kind whirl|proto --model GGUF --suite basic,mt_cache,pool,tier,sys,restore_conc,vis\n"
                     "                         [--work DIR] [--port N] [--results OUT.json] [--compare REF.json] [--cli EXE] [--no-lock]\n"
                     "                         [--mmproj MMPROJ.gguf] [--images DIR]   (suite vis)\n"
                     "                         [--server-env NAME=VALUE ...] [--kv q8v|q8|f16|auto]   (default q8v)\n"
                     "  env: WHIRL_GATE_TEXT_ROOT (required), WHIRL_GATE_WORK, WHIRL_GATE_IMAGES, WHIRL_GPU_LOCK\n"
                     "       whirl-server-gate --results A.json --compare B.json   (compare two result files only)\n");
        return 2;
    }
    SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    // the servers need the HIP runtime DLLs on PATH
    if (const char* hp = std::getenv("HIP_PATH")) {
        std::string pth = std::string(hp);
        if (!pth.empty() && pth.back() != '\\') pth += '\\';
        pth += "bin;";
        if (const char* old_path = std::getenv("PATH")) pth += old_path;
        SetEnvironmentVariableW(L"PATH", widen(pth).c_str());
    }
    const std::map<std::string, std::function<void()>> suites = {{"basic", suiteBasic}, {"mt_cache", suiteMtCache}, {"pool", suitePool},
                                                                 {"tier", suiteTier},   {"sys", suiteSys},           {"restore_conc", suiteRestoreConc},
                                                                 {"vis", suiteVis},     {"vis_speed", suiteVisSpeed}};
    if (!g.exe.empty()) {
        for (const auto& name : g.suites) {
            auto it = suites.find(name);
            if (it == suites.end()) {
                std::fprintf(stderr, "unknown suite %s\n", name.c_str());
                return 2;
            }
            try {
                it->second();
            } catch (const std::exception& ex) {
                check("suite " + name + " ran", false, ex.what());
            }
        }
        if (!g.results.empty()) writeFile(g.results, json::dumpPython(json::Value(g_results)));
    } else {
        g_results = json::parse(readFile(g.results)).asObject();
    }
    if (!g.compare.empty()) {
        const json::Value ref = json::parse(readFile(g.compare));
        std::size_t n = 0, same = 0;
        for (const auto& [k, v] : g_results.members()) {
            const json::Value* r = ref.asObject().find(k);
            if (!r) continue;
            ++n;
            if (r->asString() == v.asString()) ++same;
            else {
                const std::string& x = v.asString();
                const std::string& y = r->asString();
                std::size_t d = 0;
                while (d < x.size() && d < y.size() && x[d] == y[d]) ++d;
                std::printf("  differs: %s at byte %zu\n", k.c_str(), d);
            }
        }
        check(std::format("compare with {}: {} / {} outputs identical", g.compare, same, n), n > 0 && same == n);
    }
    std::printf("== whirl-server-gate: %d passed, %d failed%s\n", g_pass, g_fail, g_fail ? "" : " (ALL PASS)");
    for (const auto& f : g_failed) std::printf("   FAILED: %s\n", f.c_str());
    return g_fail ? 1 : 0;
}
