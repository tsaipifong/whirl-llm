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
//   whirl-server-gate --selftest   (CPU only: the reply-comparison helpers, no server)
// Suites: basic, mt_cache, mt_cache_precise, pool, tier, sys, restore_conc, vis, fast_stream, golden (see each function;
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
// WHIRL_* variable except WHIRL_DEVICE, WHIRL_VRAM_LIMIT_MB and WHIRL_MODE; a suite's own settings win), e.g.
// WHIRL_KV_RAM_MB=8192 to turn the host tiers on for an integrated GPU (Radeon 8060S),
// whose servers default to no tiers. --kv: the KV format the sys / restore_conc / vis
// suites pin (default auto = no pin: the format each mode ships, fixed at load by mode and model type;
// FR-TEST P2-5: the old default q8v is a format no mode ships; gfx1151 has no q8v kernels).
// --images = WHIRL_GATE_IMAGES (no default; only suite vis needs it).

#include "http_client.h"
#include "whirl/common.h"
#include "whirl/device_mem.h"
#include "whirl/json.h"
#include "whirl/numerics.h"

#include <windows.h>
#include <io.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
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
    std::string kv_pin = "auto";                    // --kv (FIX-TEST-A: shipped formats by default)
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

// Graceful stop of a console child (Server::stop; the --selftest console check drives it with ping.exe):
// a Ctrl+C on the child's (hidden) console, which the WHIRL server handles by flushing its prefix-cache
// tiers before it exits (the prototype exits at once on Ctrl+C).
//
// bug-gatectrlc (2026-10-26): leaving and re-joining consoles must not move this process's own output. Microsoft
// GetStdHandle "Attach/detach behavior": AttachConsole replaces the std handles by console handles unless the process
// was created with STARTF_USESTDHANDLES. pwsh `& gate.exe` under Start-Process -RedirectStandardOutput hands the gate
// the file WITHOUT that flag, so after the 1st stop STD_OUTPUT_HANDLE was the console; the old code judged "stdout is
// a console" from GetStdHandle at every stop, so the 2nd stop reopened stdout on CONOUT$ (a hidden window) and every
// later line (stop line, summary) was lost while the gate still exited 0. Now: the std handles are probed ONCE at
// startup (wmain -> captureStdHandles, from the CRT handles; GetConsoleMode, since GetFileType also says "char" for
// NUL), only a stream that really was a console is reopened on CONOUT$, and the other std handles are put back with
// SetStdHandle after every re-attach (CRT fd and Win32 std handle kept in sync, as llama.cpp server-common.cpp does).
struct StartStd {
    HANDLE h[3] = {nullptr, nullptr, nullptr};
    bool console[3] = {false, false, false};
    bool captured = false;
} g_std;
constexpr DWORD kStdIds[3] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};

bool isConsoleHandle(HANDLE h) {
    DWORD mode = 0;
    return h != nullptr && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
}

void captureStdHandles() {
    FILE* const fs3[3] = {stdin, stdout, stderr};
    for (int i = 0; i < 3; ++i) {
        const int fd = _fileno(fs3[i]);
        HANDLE h = fd >= 0 ? reinterpret_cast<HANDLE>(_get_osfhandle(fd)) : INVALID_HANDLE_VALUE;
        if (h == INVALID_HANDLE_VALUE || h == reinterpret_cast<HANDLE>(-2)) h = GetStdHandle(kStdIds[i]);  // -2: no handle
        g_std.h[i] = h;
        g_std.console[i] = isConsoleHandle(h);
    }
    g_std.captured = true;
}

void restoreStdHandles() {
    for (int i = 0; i < 3; ++i)
        if (!g_std.console[i] && g_std.h[i] != nullptr && g_std.h[i] != INVALID_HANDLE_VALUE) SetStdHandle(kStdIds[i], g_std.h[i]);
}

bool sendCtrlC(DWORD pid) {
    bool ok = false;
    if (!g_std.captured) captureStdHandles();  // wmain probes at startup; this only covers a caller that did not
    std::fflush(stdout);
    std::fflush(stderr);
    FreeConsole();
    if (AttachConsole(pid)) {
        SetConsoleCtrlHandler(nullptr, TRUE);  // this process ignores the event it sends
        ok = GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0) != 0;
        Sleep(100);  // let the event be delivered before leaving the console
        FreeConsole();
    }
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        // the old console handles died with FreeConsole: reopen only the streams that were a console at startup
        FILE* f = nullptr;
        if (g_std.console[1]) freopen_s(&f, "CONOUT$", "w", stdout);
        if (g_std.console[2]) freopen_s(&f, "CONOUT$", "w", stderr);
    }
    restoreStdHandles();  // files / pipes stay where they were (Windows swapped them for console handles on attach)
    // keep ignoring stray Ctrl+C deliveries for a moment, then restore the default
    Sleep(200);
    SetConsoleCtrlHandler(nullptr, FALSE);
    return ok;
}

// bug-gatectrlc selftest helpers (CPU only, ping.exe as the child). The layout of a runner that wraps the gate in
// `Start-Process pwsh -Command "& gate.exe ..." -RedirectStandardOutput F`:
//   selftest  --(new hidden console, STARTF_USESTDHANDLES = file F)-->  --selftest-console-wrap (plays pwsh)
//   wrap      --(same console, NO STARTF_USESTDHANDLES: inherits F as its std handles)-->  --selftest-console-child
//   child     starts N ping.exe children the way Server starts a server and stops each with sendCtrlC, then prints
//             the summary line, which must reach F.
std::wstring selfExe() {
    std::wstring p(32768, L'\0');
    p.resize(GetModuleFileNameW(nullptr, p.data(), static_cast<DWORD>(p.size())));
    return p;
}

int consoleChild(int n, const std::string& dir) {
    std::wstring sys(MAX_PATH, L'\0');
    sys.resize(GetSystemDirectoryW(sys.data(), MAX_PATH));
    int stopped = 0;
    for (int i = 0; i < n; ++i) {
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE out = CreateFileW(widen(dir + "\\ping_" + std::to_string(i) + ".txt").c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                                 &sa, CREATE_ALWAYS, 0, nullptr);
        STARTUPINFOW si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = out;
        si.hStdError = out;
        PROCESS_INFORMATION pi{};
        std::wstring cmd = quoteArg(sys + L"\\ping.exe") + L" -n 60 127.0.0.1";
        std::vector<wchar_t> cbuf(cmd.begin(), cmd.end());
        cbuf.push_back(0);
        const BOOL started = CreateProcessW(nullptr, cbuf.data(), nullptr, nullptr, TRUE,
                                            CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        if (!started) {
            std::printf("console child: cannot start ping.exe (error %lu)\n", GetLastError());
            return 3;
        }
        CloseHandle(pi.hThread);
        Sleep(1500);  // let ping.exe reach its main loop
        const bool sent = sendCtrlC(pi.dwProcessId);
        const bool exited = WaitForSingleObject(pi.hProcess, 10000) == WAIT_OBJECT_0;
        if (!exited) TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hProcess);
        if (sent && exited) ++stopped;
        std::printf("console child: stop %d: sent %d, exited %d\n", i, sent ? 1 : 0, exited ? 1 : 0);
        std::fflush(stdout);
    }
    std::printf("== console child: %d of %d stops ok (summary)\n", stopped, n);
    std::fflush(stdout);
    return stopped == n ? 0 : 1;
}

int consoleWrap(int n, const std::string& dir) {
    // like pwsh `& native.exe`: same console, std handles inherited from this process without STARTF_USESTDHANDLES
    std::wstring cmd = quoteArg(selfExe()) + L" --selftest-console-child " + std::to_wstring(n) + L" " + quoteArg(widen(dir));
    std::vector<wchar_t> cbuf(cmd.begin(), cmd.end());
    cbuf.push_back(0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cbuf.data(), nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT, nullptr, nullptr, &si, &pi))
        return 4;
    CloseHandle(pi.hThread);
    if (WaitForSingleObject(pi.hProcess, 60000) != WAIT_OBJECT_0) TerminateProcess(pi.hProcess, 5);
    WaitForSingleObject(pi.hProcess, 5000);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
}

// Runs the wrapper chain above with n stops; returns the text that reached F ("" if nothing) and the wrap exit code.
std::pair<std::string, int> consoleStopRun(int n) {
    const std::string dir = narrow((fs::temp_directory_path() / L"whirl-tests" /
                                    (L"ctrlc_selftest_" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(n)))
                                       .wstring());
    std::error_code ec;
    fs::create_directories(widen(dir), ec);
    const std::string fpath = dir + "\\wrap_out.txt";
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE f = CreateFileW(widen(fpath).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return {"", -1};
    std::wstring cmd = quoteArg(selfExe()) + L" --selftest-console-wrap " + std::to_wstring(n) + L" " + quoteArg(widen(dir));
    std::vector<wchar_t> cbuf(cmd.begin(), cmd.end());
    cbuf.push_back(0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;  // Start-Process -RedirectStandardOutput / -RedirectStandardError
    si.hStdInput = nullptr;
    si.hStdOutput = f;
    si.hStdError = f;
    PROCESS_INFORMATION pi{};
    int code = -2;
    if (CreateProcessW(nullptr, cbuf.data(), nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, nullptr,
                       nullptr, &si, &pi)) {
        CloseHandle(pi.hThread);
        if (WaitForSingleObject(pi.hProcess, 90000) != WAIT_OBJECT_0) TerminateProcess(pi.hProcess, 6);
        WaitForSingleObject(pi.hProcess, 5000);
        DWORD c = 0;
        GetExitCodeProcess(pi.hProcess, &c);
        code = static_cast<int>(c);
        CloseHandle(pi.hProcess);
    }
    CloseHandle(f);
    std::string text;
    try {
        text = readFile(fpath);
    } catch (const std::exception&) {
    }
    fs::remove_all(widen(dir), ec);
    return {text, code};
}

// bug-server-exit-av: what the exit code of a server stopped with Ctrl+C means. Empty: a clean exit (0). The server
// returns 0 after "server stopped"; 0xC0000005 was the access violation in amdhip64_7.dll during teardown, which the
// gate used to miss (WER holds the process ~26 s, then it exits with that code; the stop still looked graceful).
std::string exitProblem(DWORD code) {
    if (code == 0) return {};
    if (code == 0xC0000005u) return std::format("exit code 0x{:08X} (0xC0000005 = access violation)", code);
    return std::format("exit code 0x{:08X} ({})", code, code);
}

// The line appended to the server's log for a non-zero exit, in the server's own log format (src/server/log.cpp
// "yyyy-mm-dd HH:MM:SS.mmm L message"), so tools/gates/server_log_scan.ps1 (set "exit") finds it.
std::string exitLogLine(const SYSTEMTIME& t, DWORD code) {
    return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03} E server exit: code 0x{:08X} after Ctrl+C ({})", t.wYear, t.wMonth,
                       t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, code, exitProblem(code));
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
        // the GPU, so a gate under a simulated 16 GB card runs its servers on that card; WHIRL_MODE
        // kept: the numerics mode the gate runs in) plus the extras
        std::vector<std::pair<std::wstring, std::wstring>> envv;
        if (wchar_t* blk = GetEnvironmentStringsW()) {
            for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
                std::wstring s = p;
                const std::size_t eq = s.find(L'=', 1);
                if (eq == std::wstring::npos) continue;
                std::wstring k = s.substr(0, eq), up = k;
                for (auto& c : up) c = static_cast<wchar_t>(towupper(c));
                if (up.rfind(L"WHIRL_", 0) == 0 && up != L"WHIRL_DEVICE" && up != L"WHIRL_VRAM_LIMIT_MB" && up != L"WHIRL_MODE") continue;
                bool overridden = false;  // the extras replace an inherited variable (no duplicate entry)
                for (const auto& [ek, ev] : env_add) {
                    std::wstring eu = widen(ek);
                    for (auto& c : eu) c = static_cast<wchar_t>(towupper(c));
                    overridden = overridden || eu == up;
                }
                if (overridden) continue;
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
    // Graceful stop first (sendCtrlC); TerminateProcess if it has not exited after 30 s.
    void stop() {
        if (!proc_) return;
        const auto t0 = std::chrono::steady_clock::now();
        const bool sent = sendCtrlC(GetProcessId(proc_));
        bool terminated = false;
        if (!sent || WaitForSingleObject(proc_, 30000) != WAIT_OBJECT_0) {
            logLine(std::format("server {}: {}; terminating", tag_, sent ? "no exit 30 s after Ctrl+C" : "Ctrl+C could not be sent"));
            TerminateProcess(proc_, 1);
            terminated = true;
        } else {
            DWORD code = 0;
            const bool got = GetExitCodeProcess(proc_, &code) != 0;
            logLine(std::format("server {} stopped (Ctrl+C) in {:.1f} s, exit code {}", tag_,
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                                got ? std::format("0x{:08X}", code) : std::string("unknown")));
            if (got && code != 0) {
                check(std::format("server {} exits with code 0 after Ctrl+C", tag_), false, exitProblem(code));
                SYSTEMTIME st;
                GetLocalTime(&st);
                if (std::FILE* f = _wfopen(widen(log_).c_str(), L"ab")) {
                    const std::string l = exitLogLine(st, code) + "\n";
                    std::fwrite(l.data(), 1, l.size(), f);
                    std::fclose(f);
                }
            }
        }
        WaitForSingleObject(proc_, 30000);
        DWORD code_end = 0;
        if (terminated && GetExitCodeProcess(proc_, &code_end))  // 1 = our TerminateProcess; anything else: it exited on its own first
            logLine(std::format("server {}: exit code 0x{:08X} after the terminate", tag_, code_end));
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
    // The text a --results file records: reasoning, NUL, content, NUL, then the tool calls. The NULs are
    // deliberate separators (text moving between reasoning and content must not compare equal) and show up as
    // \u0000 in results.json (valid JSON; json::parse gives the same bytes back). Not server output (bug-gatenul).
    std::string text() const {
        std::string t = reasoning + std::string(1, '\0') + content + std::string(1, '\0');
        for (const auto& tc : tool_calls) t += tc[1] + "(" + tc[2] + ");";
        return t;
    }
};

// The "A == B" checks compare two replies. Two failed requests (4xx / 5xx, a crashed server) both have
// empty texts, so a plain comparison of them is green: every comparison goes through these helpers.
bool answered(const ChatResult& r) { return r.status == 200 && (!r.content.empty() || !r.reasoning.empty() || !r.tool_calls.empty()); }
bool sameContent(const ChatResult& a, const ChatResult& b) { return answered(a) && answered(b) && a.content == b.content; }
bool sameText(const ChatResult& a, const ChatResult& b) { return answered(a) && answered(b) && a.text() == b.text(); }
// concurrency check (thinking off): each concurrent reply == the same request alone, both HTTP 200 with
// non-empty content ("=" / "x" per request in d)
bool concSame(const std::vector<ChatResult>& res, const std::vector<ChatResult>& alone, std::string& d) {
    bool same = res.size() == alone.size();
    for (std::size_t i = 0; i < res.size() && i < alone.size(); ++i) {
        const bool eq = res[i].status == 200 && alone[i].status == 200 && !res[i].content.empty() && !alone[i].content.empty() &&
                        sameContent(res[i], alone[i]);
        same = same && eq;
        d += eq ? "=" : "x";
    }
    return same;
}

// FIX-TEST-A (FR-TEST P2-2): a concurrency check must compare cold prefills: every reply answered and
// cached == 0 ("cached" lists the cached counts in d)
bool allCold(const std::vector<ChatResult>& res, std::string& d) {
    bool cold = !res.empty();
    for (const auto& r : res) {
        cold = cold && r.status == 200 && r.cached == 0;
        d += std::to_string(r.cached) + " ";
    }
    return cold;
}

// Do the servers run with relaxed acceptance (fast mode's relaxacc)? From WHIRL_MODE / WHIRL_RELAX as
// the servers see them (--server-env, else this process's environment).
bool relaxAccOn(const std::optional<std::string>& mode, const std::optional<std::string>& relax) {
    if (!mode) return false;
    try {
        const numerics::Request r = numerics::parseMode(*mode, "WHIRL_MODE");  // as the server parses it
        return r.mode == numerics::Mode::fast && r.has(numerics::Item::relaxacc) && !(relax && *relax == "0");
    } catch (const std::exception&) {
        return false;  // the server refuses such a mode: no relaxed acceptance to account for
    }
}

// Do the servers run in precise mode? (WHIRL_MODE as the servers see it; unset = the default, balance)
bool preciseMode(const std::optional<std::string>& mode);
// the servers' mode is precise (--server-env WHIRL_MODE, else this process's environment)
bool serversPrecise();
bool preciseMode(const std::optional<std::string>& mode) {
    if (!mode) return false;
    try {
        return numerics::parseMode(*mode, "WHIRL_MODE").mode == numerics::Mode::precise;
    } catch (const std::exception&) {
        return false;
    }
}

// CACHEBAL: one WHIRL_PREFIX_CACHE_VERIFY=1 log line ("prefix-cache verify: cached vs from-scratch prefill
// logits: max abs diff D, KL K, top1 same|DIFFERENT (...)"): the logits of the reused (here SSD-restored) state
// against a from-scratch prefill of the same prompt.
struct CacheVerify {
    double maxd = 0, kl = 0;
    bool top1_same = false;
    long long ndiff = -1;  // "bitwise differing logits N" (REVIEW-022 F3); -1: not in the line (older server)
};
std::optional<CacheVerify> parseCacheVerify(const std::string& line) {
    static const std::regex r(R"(prefix-cache verify: .*max abs diff ([0-9.]+), KL ([0-9.]+(?:e[-+]?[0-9]+)?), top1 (same|DIFFERENT))");
    std::smatch m;
    if (!std::regex_search(line, m, r)) return std::nullopt;
    CacheVerify v{std::stod(m[1].str()), std::stod(m[2].str()), m[3].str() == "same"};
    static const std::regex rd(R"(bitwise differing logits (\d+))");
    if (std::smatch md; std::regex_search(line, md, rd)) v.ndiff = std::stoll(md[1].str());
    return v;
}
// precise: bit-identical logits (max abs diff prints 4 decimals and KL can underflow, so the bit count decides)
bool cacheVerifyBitwise(const CacheVerify& v) { return v.ndiff == 0 && v.maxd == 0 && v.kl == 0 && v.top1_same; }
// The bound: the server's documented diagnostic threshold for a cache hit that is numerically equivalent, not
// bit-identical (docs/guide/en/kv-and-caching.md section 5: KL <= 1e-2 and the same top-1). Measured on the
// R9700, Swift-1.5 27B MXFP4-A (requant 2026-10-19), tier p1 gen-end state vs cold: balance KL 9e-5..2e-4,
// precise KL 1e-8..7e-8; a wrong or mismatched restore is far above it.
constexpr double kCacheVerifyMaxKl = 1e-2;
bool serversPrecise() {
    auto it = g.server_env.find("WHIRL_MODE");
    const char* em = std::getenv("WHIRL_MODE");
    return preciseMode(it != g.server_env.end() ? std::optional<std::string>(it->second) : (em ? std::optional<std::string>(em) : std::nullopt));
}
bool cacheVerifyOk(const CacheVerify& v) { return v.top1_same && v.kl <= kCacheVerifyMaxKl; }

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
    if (h.status != 200) {
        // CB-FLAKE: a failed request says why (connect / send / recv, WSA error, time)
        logLine(std::format("chat request failed: status {}, {:.0f} ms, {}{}", h.status, r.ms, h.err.empty() ? "no client error" : h.err,
                            h.body.empty() ? std::string() : ", body " + h.body.substr(0, 200)));
        return r;
    }
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
// the "prompt tokens:" line and the "\n\n  prefill:" stats, read from the CLI's stdout only.
// SRV-CLI: the CLI writes diagnostics to stderr; with stderr in the same file the MTP draft
// head's "draft head: vocabulary subset ..." line (108 bytes, printed when decode starts, i.e.
// after "prompt tokens:") was parsed as the start of the generated text.
std::optional<std::string> cliRun(const std::string& prompt, bool think, int max_tokens) {
    if (g.cli.empty()) return std::nullopt;
    const std::string pf = g.work + "\\cli_prompt.txt";
    writeFile(pf, prompt);
    const std::string out = g.work + "\\cli_out.txt";
    std::wstring cmd = quoteArg(widen(g.cli)) + L" chat " + quoteArg(widen(g.model)) + L" " + quoteArg(widen("@" + pf)) +
                       L" --max-tokens " + std::to_wstring(max_tokens) + L" --no-stream";
    SetEnvironmentVariableW(L"WHIRL_THINK", think ? L"1" : L"0");
    SetEnvironmentVariableW(L"WHIRL_R9700_LOCK_HELD", std::to_wstring(GetCurrentProcessId()).c_str());
    // FIX-TEST-A: the CLI reference runs in the servers' configuration (--server-env, e.g. WHIRL_MODE=precise);
    // without it a precise server was compared with a default-mode (balance) CLI
    std::map<std::string, std::optional<std::string>> env_before;
    for (const auto& [k, v] : g.server_env) {
        const char* old = std::getenv(k.c_str());
        env_before[k] = old ? std::optional<std::string>(old) : std::nullopt;
        SetEnvironmentVariableW(widen(k).c_str(), widen(v).c_str());
    }
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE oh = CreateFileW(widen(out).c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
    HANDLE eh = CreateFileW(widen(g.work + "\\cli_err.txt").c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, 0, nullptr);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = oh;
    si.hStdError = eh;
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cb(cmd.begin(), cmd.end());
    cb.push_back(0);
    g_lock.acquire();
    const BOOL ok = CreateProcessW(nullptr, cb.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(oh);
    CloseHandle(eh);
    if (!ok) {
        g_lock.release();
        SetEnvironmentVariableW(L"WHIRL_THINK", nullptr);
        for (const auto& [k, v] : env_before) SetEnvironmentVariableW(widen(k).c_str(), v ? widen(*v).c_str() : nullptr);
        return std::nullopt;
    }
    DWORD code = 1;
    if (WaitForSingleObject(pi.hProcess, 900000) == WAIT_OBJECT_0) GetExitCodeProcess(pi.hProcess, &code);
    else TerminateProcess(pi.hProcess, 1);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    g_lock.release();
    SetEnvironmentVariableW(L"WHIRL_THINK", nullptr);
    for (const auto& [k, v] : env_before) SetEnvironmentVariableW(widen(k).c_str(), v ? widen(*v).c_str() : nullptr);
    if (code != 0) return std::nullopt;
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
    std::optional<ServerLease> s_opt;
    s_opt.emplace("basic", std::vector<std::string>{}, std::map<std::string, std::string>{});
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
        check(std::string("basic: greedy ") + c.key, r.status == 200 && (!r.content.empty() || !r.reasoning.empty()), brief(r));
        record(std::string("basic/") + c.key, r.text());
        solo[c.key] = r;
        ChatResult st = chat(arr({msg("user", c.prompt)}), greedy(c.max) + thinkKw(c.think), true);
        check(std::string("basic: stream == non-stream ") + c.key,
              sameContent(st, r) && st.reasoning == r.reasoning && st.finish == r.finish, brief(st));
        // SRV-CLI: with --cli, a missing reference (CLI failed, timed out or printed no "prompt tokens:")
        // fails instead of silently skipping the comparison
        if (cli_ref.count(c.key) && !g.cli.empty() && !cli_ref[c.key])
            check(std::string("basic: server greedy == CLI ") + c.key, false, "no CLI reference (see cli_out.txt / cli_err.txt in --work)");
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
        check("basic: same seed -> same text", a.status == 200 && sameContent(a, b), brief(a));
        record("basic/seed1234", a.text());
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 4; ++i)
            jobs.push_back([&] { return chat(arr({msg("user", "Write a haiku about GPUs.")}), extra); });
        auto res = parallel(jobs);
        bool same = true;
        for (const auto& r : res) same = same && sameContent(r, a);
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
            // COLD-RESUME: precise resumes only at cold-exact checkpoints; a prompt this short has none past 0
            if (serversPrecise())
                logLine(std::format("INFO basic: tool round trip reuses the prefix: cached {} (precise: cold-exact resumes only; not a gate)", r2.cached));
            else check("basic: tool round trip reuses the prefix", r2.cached > 0, std::format("cached {}", r2.cached));
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
        if (serversPrecise()) {
            // COLD-RESUME: precise never continues from the end of a reply (its history came from the decode kernels)
            logLine(std::format("INFO basic: multi-turn reuse {}(precise: cold-exact resumes only; not a gate)", detail));
            const std::size_t ge = (*s_opt)->logCount(R"(reused \d+ tok \(checkpoint 'gen-end')");
            check("basic: precise: no request resumed from a 'gen-end' checkpoint (tool round trip, multi-turn)", ge == 0,
                  std::format("{} gen-end resumes logged", ge));
        } else {
            check("basic: multi-turn reuses the whole previous conversation", ok, detail);
        }
    }
    // concurrency: different prompts at once == solo. Fast mode with relaxed acceptance only
    // guarantees quality within its gate, not concurrent == solo text (the draft cadence depends on
    // the load), so there the relax-on comparison is informational and the gated check runs a
    // server with WHIRL_RELAX=0 (it then tests the kernels' batch invariance).
    // FIX-TEST-A (FR-TEST P2-2): the gated check runs on its own server without the prefix cache and
    // the host tiers, and asserts cached == 0 for every request (alone and concurrent): with the cache on,
    // the concurrent round re-sent the same prompts, hit the prompt-end checkpoints and so only compared
    // concurrent decode, never concurrent (grouped / segmented) prefill. The prompts differ in length and one
    // is longer than a server prefill chunk (1024), so the batched prefill crosses a chunk boundary.
    const std::vector<std::string> conc_ps = {"Explain a hash map in two sentences.", "What is RAII? Two sentences.",
                                              "Write a Python one-liner that reverses a string.", P_ZH,
                                              codeText(6000, 5) + "\n\nWhat does the code above do? One sentence."};
    auto concRound = [&](std::vector<ChatResult>& alone, std::vector<ChatResult>& res) {
        alone.clear();
        for (const auto& p : conc_ps) alone.push_back(chat(arr({msg("user", p)}), greedy(64) + thinkKw(false)));
        std::vector<std::function<ChatResult()>> jobs;
        for (const auto& p : conc_ps) jobs.push_back([&, p] { return chat(arr({msg("user", p)}), greedy(64) + thinkKw(false)); });
        res = parallel(jobs);
    };
    auto concCheck = [&](const std::string& name, bool gated) {
        std::vector<ChatResult> alone, res;
        concRound(alone, res);
        std::string d;
        const bool same = concSame(res, alone, d);
        if (!gated) {
            logLine(std::format("INFO {}: {} ({}; not a gate)", name, same ? "equal" : "differs", d));
            return;
        }
        for (std::size_t i = 0; i < conc_ps.size(); ++i) record(std::format("basic/conc{}", i), alone[i].text());
        std::string cd;
        const bool cold = allCold(alone, cd) && allCold(res, cd);
        check(name + " (cold: cached == 0)", same && cold,
              std::format("{}; cached {}; prompts {}", d, cd, [&] {
                  std::string p;
                  for (const auto& r : alone) p += std::to_string(r.prompt) + " ";
                  return p;
              }()));
    };
    auto envOf = [](const char* name) -> std::optional<std::string> {
        auto it = g.server_env.find(name);
        if (it != g.server_env.end()) return it->second;
        if (const char* m = std::getenv(name)) return std::string(m);
        return std::nullopt;
    };
    const bool relax = relaxAccOn(envOf("WHIRL_MODE"), envOf("WHIRL_RELAX"));
    if (relax) concCheck("basic: concurrent requests == each alone (fast, relaxacc on)", false);
    s_opt.reset();
    std::map<std::string, std::string> cold_env = {{"WHIRL_NO_PREFIX_CACHE", "1"}};
    if (relax) cold_env["WHIRL_RELAX"] = "0";
    ServerLease s2("basic_cold", {"--kv-ram-mb", "0"}, cold_env);
    concCheck(relax ? "basic: 5 concurrent requests == each alone (WHIRL_RELAX=0)" : "basic: 5 concurrent requests == each alone", true);
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

// one request of a session as it was sent (messages array + extra fields) and its reply
struct MtReq {
    std::string messages, extra;
    ChatResult r;
};

std::vector<MtReq> runSessionRec(const std::string& tag, const std::string& name, const std::string& sysmsg,
                                 const std::vector<std::string>& users, bool think, bool send_reasoning, bool tools, int max_tokens) {
    std::vector<std::string> msgs;
    if (!sysmsg.empty()) msgs.push_back(msg("system", sysmsg));
    std::vector<MtReq> reqs;
    int k = 0;
    for (const auto& q : users) {
        msgs.push_back(msg("user", q));
        for (int rt = 0; rt < 3; ++rt) {
            const std::string extra = greedy(max_tokens) + thinkKw(think) + (tools ? ",\"tools\":" + MT_TOOLS : "");
            const std::string m = arr(msgs);
            ChatResult r = chat(m, extra);
            reqs.push_back({m, extra, r});
            record(std::format("{}/{}/{}", tag, name, k++), r.text());
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
    return reqs;
}

std::vector<Turn> runSession(const std::string& name, const std::string& sysmsg, const std::vector<std::string>& users, bool think,
                             bool send_reasoning, bool tools, int max_tokens) {
    std::vector<Turn> turns;
    for (const MtReq& q : runSessionRec("mt_cache", name, sysmsg, users, think, send_reasoning, tools, max_tokens))
        turns.push_back({q.r.prompt, q.r.cached, q.r.completion, q.r.finish});
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
    if (mode == "info") {  // table only (mt_cache_precise: precise resumes only at cold-exact checkpoints)
        logLine(std::format("INFO mt_cache: {}: hit ratio {:.3f}, full-history misses {} (not a gate here)", name,
                            lp ? static_cast<double>(lc) / lp : 0.0, misses));
        return;
    }
    if (mode == "full" && misses > 0) ok = false;
    if (mode == "agent" && misses > 1) ok = false;
    check(std::format("mt_cache: {} ({}): prefix reuse across turns", name, mode), ok,
          std::format("hit ratio {:.3f}, full-history misses {}", lp ? static_cast<double>(lc) / lp : 0.0, misses));
}

// the mt_cache sessions (shared by mt_cache and mt_cache_precise)
struct MtSpec {
    std::string name, sys;
    std::vector<std::string> users;
    bool think, send_reasoning, tools;
    int max_tokens;
    std::string eval;  // evaluate() mode
};
std::vector<MtSpec> mtSpecs() {
    const std::string big = "You are a coding agent working in a Python repository. Project notes follow.\n\n" + docsText(14000);
    return {{"agent", big,
             {"The test for parse_range fails. Read src/ranges.py and tell me why.", "Fix it and show only the corrected function.",
              "Now run the tests again."},
             false, true, true, 400, "agent"},
            {"think", "", {"What is 17 * 23?", "Add 100 to that.", "Is the result a prime number?"}, true, true, false, 1200, "full"},
            {"nothink", "You are a concise assistant.",
             {"Name three planets of the solar system.", "Which of them is the largest? One sentence.", "How many moons does it have, roughly?",
              "Thanks. One more fact about it, one sentence."},
             false, true, false, 150, "full"},
            {"noreason", "", {"What is 12 * 12?", "Double it.", "Now subtract 50."}, true, false, false, 1200, "prev"}};
}

void suiteMtCache() {
    logLine("== suite mt_cache");
    ServerLease s("mt_cache", {}, {});
    for (const MtSpec& m : mtSpecs())
        evaluate(m.name, runSession(m.name, m.sys, m.users, m.think, m.send_reasoning, m.tools, m.max_tokens), m.eval);
}

// ---------------------------------------------------------------------------
// suite mt_cache_precise: in precise mode a multi-turn session with the prefix cache gives exactly what a
// cold prefill gives (COLD-RESUME, 0.2.2). Three servers, all WHIRL_MODE=precise (the suite sets it; it
// overrides --server-env):
//   1. cache on: runs the mt_cache sessions, every request recorded; no 'gen-end' resume may be logged;
//      the agent session (14k-token system prompt) must reuse a prefix on every turn after its first;
//   2. cache on + WHIRL_PREFIX_CACHE_VERIFY=1: replays the recorded requests in the same order (same cache
//      hits); for each hit the server compares the resumed state's logits with a from-scratch prefill:
//      max abs diff 0 and KL 0 (bitwise), one verify line per reply with cached > 0;
//   3. no prefix cache (every request cold): replays the recorded requests; each reply must equal run 1's
//      reply (reasoning, content, tool calls) with the same prompt token count.

// Precise: run 1 (cache) vs run 3 (cold), request by request. "" = all equal, else the differing requests.
std::string mtColdDiff(const std::vector<ChatResult>& cached, const std::vector<ChatResult>& cold) {
    if (cached.size() != cold.size()) return std::format("{} cached replies vs {} cold", cached.size(), cold.size());
    if (cached.empty()) return "no replies";
    std::string d;
    for (std::size_t i = 0; i < cached.size(); ++i) {
        const ChatResult &a = cached[i], &b = cold[i];
        if (!sameText(a, b)) d += std::format("#{} text differs ({} / {}); ", i, brief(a), brief(b));
        else if (a.prompt != b.prompt) d += std::format("#{} prompt {} vs {}; ", i, a.prompt, b.prompt);
        else if (b.cached != 0) d += std::format("#{} cold reply has cached {}; ", i, b.cached);
    }
    return d;
}

// Precise: the verify lines of run 2. Bitwise means max abs diff 0, KL 0, same top-1, and one line per
// reply with cached > 0 (at least one hit, or the check tests nothing).
bool mtPreciseVerifyOk(const std::vector<CacheVerify>& v, std::size_t hits, std::string& d) {
    d = std::format("{} verify lines, {} replies with cached > 0", v.size(), hits);
    bool ok = hits > 0 && v.size() == hits;
    for (std::size_t i = 0; i < v.size(); ++i)
        if (!cacheVerifyBitwise(v[i])) {
            ok = false;
            d += std::format("; #{}: max abs diff {:.4f}, KL {:.2e}, top1 {}, differing logits {}", i, v[i].maxd, v[i].kl,
                             v[i].top1_same ? "same" : "DIFFERENT", v[i].ndiff);
        }
    return ok;
}

void suiteMtCachePrecise() {
    logLine("== suite mt_cache_precise");
    const std::map<std::string, std::string> P = {{"WHIRL_MODE", "precise"}};
    const std::vector<MtSpec> specs = mtSpecs();
    std::vector<std::pair<std::string, std::vector<MtReq>>> rec;  // session name, its requests
    {
        ServerLease s("mtp_cache", {}, P);
        for (const MtSpec& m : specs) {
            rec.emplace_back(m.name, runSessionRec("mt_cache_precise", m.name, m.sys, m.users, m.think, m.send_reasoning, m.tools, m.max_tokens));
            std::vector<Turn> turns;
            for (const MtReq& q : rec.back().second) turns.push_back({q.r.prompt, q.r.cached, q.r.completion, q.r.finish});
            // print the per-turn table (prefix-reuse figures); in precise only the agent session's reuse is a gate
            evaluate(m.name, turns, "info");
        }
        const std::size_t ge = s->logCount(R"(reused \d+ tok \(checkpoint 'gen-end')");
        check("mt_cache_precise: no request resumed from a 'gen-end' checkpoint", ge == 0, std::format("{} gen-end resumes logged", ge));
        const auto& agent = rec.front().second;
        bool reuse = agent.size() >= 2;
        std::string d;
        for (std::size_t i = 1; i < agent.size(); ++i) {
            reuse = reuse && agent[i].r.cached > 0;
            d += std::to_string(agent[i].r.cached) + " ";
        }
        check("mt_cache_precise: agent session reuses a cold-exact prefix on every turn after the first", reuse, "cached " + d);
    }
    {
        auto env = P;
        env["WHIRL_PREFIX_CACHE_VERIFY"] = "1";
        ServerLease s("mtp_verify", {}, env);
        std::size_t hits = 0;
        for (const auto& [name, reqs] : rec)
            for (const MtReq& q : reqs) hits += chat(q.messages, q.extra).cached > 0 ? 1 : 0;
        std::vector<CacheVerify> v;
        for (const auto& l : s->logLines())
            if (auto c = parseCacheVerify(l)) v.push_back(*c);
        std::string d;
        check("mt_cache_precise: every resumed state == from-scratch prefill, bitwise (max abs diff 0, KL 0)", mtPreciseVerifyOk(v, hits, d), d);
    }
    {
        auto env = P;
        env["WHIRL_NO_PREFIX_CACHE"] = "1";
        ServerLease s("mtp_cold", {"--kv-ram-mb", "0"}, env);
        for (const auto& [name, reqs] : rec) {
            std::vector<ChatResult> a, b;
            for (const MtReq& q : reqs) {
                a.push_back(q.r);
                b.push_back(chat(q.messages, q.extra));
            }
            const std::string d = mtColdDiff(a, b);
            check(std::format("mt_cache_precise: {}: every reply with the prefix cache == cold prefill ({} requests)", name, reqs.size()),
                  d.empty(), d);
        }
    }
}

// ---------------------------------------------------------------------------
// suite fast_stream: fast mode (relaxacc forced on, also for MoE) gives the same text
// for a request streamed or not, cold or resumed from its own prefix-cache checkpoints
// (think-open, prompt end, chunk ends). FIX-FS: a resume ran the MTP boundary row as one
// row, a cold prefill inside a larger batch; relaxacc kept drafts that depended on it.
// Reference: a server without the prefix cache (every request cold). A case that shares a
// prefix with the previous one (zh_nothink after zh_think; code_shared after code_long) is
// prefilled in other chunks than cold (resume at, or split for, the shared prefix): fast
// output may differ from cold there (as with concurrent requests; exact acceptance hides
// these last-bit differences), so its repeats are compared with its first reply only.

void suiteFastStream() {
    logLine("== suite fast_stream");
    const std::string code = codeText(12000, 3);
    struct Case {
        const char* key;
        std::string prompt;
        bool think;
        int max;
        bool shared;  // shares a prefix with the previous case
    };
    const std::vector<Case> cases = {{"zh_think", P_ZH, true, 200, false},
                                     {"zh_nothink", P_ZH, false, 200, true},
                                     {"code_long", code + "\n\nSummarize the above in three sentences.", false, 150, false},
                                     {"code_shared", code + "\n\nList the main functions above, one line each.", true, 150, true}};
    const std::map<std::string, std::string> env = {{"WHIRL_MODE", "fast"}, {"WHIRL_RELAX", "1"}};
    std::map<std::string, ChatResult> ref;
    {
        std::map<std::string, std::string> e = env;
        e["WHIRL_NO_PREFIX_CACHE"] = "1";
        ServerLease s("fast_stream_cold", {}, e);
        for (const Case& c : cases) {
            if (c.shared) continue;
            ref[c.key] = chat(arr({msg("user", c.prompt)}), greedy(c.max) + thinkKw(c.think));
            check(std::string("fast_stream: cold reference ") + c.key,
                  ref[c.key].status == 200 && (!ref[c.key].content.empty() || !ref[c.key].reasoning.empty()), brief(ref[c.key]));
        }
    }
    ServerLease s("fast_stream", {}, env);
    // FS-001: the suite is only evidence for the boundary fix when MTP drafts really run and the
    // repeats really resume from the prefix cache; otherwise equal texts prove nothing
    check("fast_stream: MTP speculative decoding is on (model has a nextn layer, WHIRL_MTP != 0)",
          s->logCount("MTP speculative decoding: on") >= 1);
    for (const Case& c : cases) {
        // non-stream / stream twice each: #0 is cold (or a shared-prefix resume), the
        // others resume from this prompt's own checkpoints (full hit or think-open)
        ChatResult first;
        for (int i = 0; i < 4; ++i) {
            const bool st = (i % 2) == 1;
            const ChatResult r = chat(arr({msg("user", c.prompt)}), greedy(c.max) + thinkKw(c.think), st);
            if (i == 0) {
                first = r;
                record(std::string("fast_stream/") + c.key, r.text());
                if (c.shared) {
                    check(std::format("fast_stream: {} #0 (non-stream)", c.key), r.status == 200 && (!r.content.empty() || !r.reasoning.empty()),
                          brief(r));
                    continue;
                }
            }
            if (i > 0)
                check(std::format("fast_stream: {} #{} resumed from the prefix cache (cached tokens > 0)", c.key, i), r.cached > 0,
                      std::format("cached {} of {}", r.cached, r.prompt));
            const ChatResult& want = c.shared ? first : ref[c.key];
            check(std::format("fast_stream: {} #{} ({}) == {}", c.key, i, st ? "stream" : "non-stream", c.shared ? "#0" : "cold"),
                  r.status == 200 && sameContent(r, want) && r.reasoning == want.reasoning && r.finish == want.finish, brief(r));
        }
    }
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
        check("pool: P2 again: same text", sameContent(b2, a2));
        check("pool: P1 again (evicted): restored from the RAM tier, same text",
              b1.cached >= b1.prompt - 1 && sameContent(b1, a1) && s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= 1,
              std::format("cached {} of {}", b1.cached, b1.prompt));
        std::vector<std::function<ChatResult()>> jobs = {[&] { return ask(P[1], q); }, [&] { return ask(P[2], q); }};
        auto res = parallel(jobs);
        check("pool: P2, P3 concurrently == solo", sameContent(res[0], a2) && sameContent(res[1], a3));
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

// arch025-k2warn (bug:k2-8060s-p1): a restart server sizes its RAM tier once at startup from the RAM available then
// ("adjusted: WHIRL_KV_RAM_MB 8192 -> X MiB (...)"). When X is smaller than an SSD entry, that entry cannot be read
// back (the server logs "kv tier: SSD entry of N tok (E MiB) is larger than the RAM tier (R MiB) ...") and the
// request is a cold prefill: the restart checks then fail for a reason in the host's free RAM, not in WHIRL.
// Returns "" (conclusive) or the reason. max_entry_mib: the largest SSD cache file before the restart (MiB, rounded
// up; a file that borrows blocks of another one is smaller than its entry, which the refused-restore line covers).
std::string tierRamInconclusive(const std::vector<std::string>& lines, std::uint64_t max_entry_mib) {
    static const std::regex adj(R"(adjusted: (?:WHIRL_KV_RAM_MB|--kv-ram-mb) \d+ -> (\d+) MiB)");
    static const std::regex refused(R"(kv tier: SSD entry of \d+ tok \(([0-9.]+) MiB\) is larger than the RAM tier \(([0-9.]+) MiB\))");
    std::size_t n_ref = 0;
    std::string first_ref;
    std::optional<std::uint64_t> tier_mib;
    for (const auto& l : lines) {
        std::smatch m;
        if (std::regex_search(l, m, refused)) {
            if (n_ref == 0) first_ref = std::format("entry {} MiB > RAM tier {} MiB", m[1].str(), m[2].str());
            ++n_ref;
        } else if (!tier_mib && std::regex_search(l, m, adj)) {
            tier_mib = std::stoull(m[1].str());
        }
    }
    if (n_ref > 0) return std::format("{} SSD restore(s) refused by the restart server ({})", n_ref, first_ref);
    if (tier_mib && *tier_mib < max_entry_mib)
        return std::format("the restart server's RAM tier was lowered at startup to {} MiB (adjusted: line), smaller than "
                           "the largest SSD entry ({} MiB)",
                           *tier_mib, max_entry_mib);
    return "";
}

// One check of a restart run, held until the run is judged.
struct PendCheck {
    std::string name;
    bool ok;
    std::string detail;
};

// arch025-k2warn: runs `attempt` (it starts the restart server, fills `checks`, copies the server's log into `lines`);
// all checks pass -> they are counted as they are; a check failed and tierRamInconclusive gives a reason ->
// INCONCLUSIVE and one re-run; a second INCONCLUSIVE counts the failed checks as FAIL with the reason.
void restartChecksRamRetry(const std::string& what, std::uint64_t max_entry_mib,
                           const std::function<void(int, std::vector<PendCheck>&, std::vector<std::string>&)>& attempt) {
    for (int a = 0; a < 2; ++a) {
        std::vector<PendCheck> checks;
        std::vector<std::string> lines;
        attempt(a, checks, lines);
        const bool all_ok = std::all_of(checks.begin(), checks.end(), [](const PendCheck& c) { return c.ok; });
        const std::string why = all_ok ? std::string() : tierRamInconclusive(lines, max_entry_mib);
        if (!why.empty() && a == 0) {
            std::printf("  [INCONCLUSIVE] %s: %s; re-running once\n", what.c_str(), why.c_str());
            std::fflush(stdout);
            continue;
        }
        for (const auto& c : checks)
            check(c.name, c.ok,
                  why.empty() || c.ok ? c.detail : std::format("{}; INCONCLUSIVE twice, counted as FAIL: {}", c.detail, why));
        return;
    }
}

std::uint64_t maxFileMiB(const std::string& dir) {
    std::uint64_t mx = 0;
    std::error_code ec;
    if (fs::exists(widen(dir), ec))
        for (const auto& e : fs::directory_iterator(widen(dir), ec))
            if (e.is_regular_file(ec)) mx = std::max<std::uint64_t>(mx, e.file_size(ec));
    return (mx + (1ull << 20) - 1) >> 20;
}

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
        // COLD-RESUME: a precise server never continues a conversation from the end of a generated reply
        if (serversPrecise())
            check(std::format("tier {}: precise: no turn resumed from a 'gen-end' checkpoint", mode),
                  s->logCount(R"(reused \d+ tok \(checkpoint 'gen-end')") == 0,
                  std::format("{} gen-end resumes logged", s->logCount(R"(reused \d+ tok \(checkpoint 'gen-end')")));
    }
    const std::string ssd = g.work + "\\tier_gate_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    std::map<std::string, ChatResult> t1;
    {
        ServerLease s("tier_" + mode, args, {{"WHIRL_KV_SSD_DIR", ssd}}, false);
        for (const auto& ss : S) {
            t1[ss.name] = chat(t1msg(ss), greedy(64) + thinkKw(ss.think));
            check(std::format("tier {}: {} turn 1 == reference", mode, ss.name), sameText(t1[ss.name], ref[ss.name].first));
        }
        // (the spill is logged after the response is sent: wait for the line)
        check(std::format("tier {}: every turn 1 spilled to the RAM tier", mode), s->waitLog("kv tier: spill ", S.size(), 30),
              std::format("{} spills logged", s->logCount("kv tier: spill ")));
        const std::size_t n_ev = p1 ? S.size() - 1 : 1;
        for (std::size_t i = 0; i < n_ev; ++i) {
            auto b = chat(t2msg(S[i], t1[S[i].name]), greedy(96) + thinkKw(S[i].think));
            const auto& rb = ref[S[i].name].second;
            check(std::format("tier {}: {} turn 2 after eviction (RAM restore) == never-evicted", mode, S[i].name),
                  sameText(b, rb) && b.cached == rb.cached, std::format("cached {} vs {}", b.cached, rb.cached));
        }
        check(std::format("tier {}: RAM restores logged", mode), s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= n_ev);
        // CLOSE-COV: shutdown mode writes only reused entries at the stop, so a session that ran one turn here
        // (p1's thinking session T, p4's B-E) would not reach the SSD and would cold-prefill after the restart,
        // where the content-only checks still pass. Each of them runs its turn 2 here too: the entry becomes
        // reused (restored from RAM, or resumed in place), the stop writes it, and the restart checks below
        // require every session to come back from the SSD.
        for (std::size_t i = n_ev; i < S.size(); ++i) {
            auto b = chat(t2msg(S[i], t1[S[i].name]), greedy(96) + thinkKw(S[i].think));
            const auto& rb = ref[S[i].name].second;
            check(std::format("tier {}: {} turn 2 before the restart == never-evicted", mode, S[i].name),
                  sameText(b, rb) && b.cached == rb.cached, std::format("cached {} vs {}", b.cached, rb.cached));
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    std::size_t files = 0;
    if (fs::exists(widen(ssd)))
        for (const auto& e : fs::directory_iterator(widen(ssd))) files += e.is_regular_file() ? 1 : 0;
    check(std::format("tier {}: entries on SSD after the restart (one per session)", mode), files >= S.size(),
          std::format("{} files, {} sessions", files, S.size()));
    std::map<std::string, ChatResult> after;
    // arch025-k2warn: a restart server whose RAM tier came out smaller than the largest SSD entry (low free RAM at
    // its start) is INCONCLUSIVE and runs once more (restartChecksRamRetry), not a FAIL
    const std::uint64_t max_entry_mib = maxFileMiB(ssd);
    restartChecksRamRetry(std::format("tier {}: restart", mode), max_entry_mib,
                          [&](int a, std::vector<PendCheck>& ck, std::vector<std::string>& lines) {
        ServerLease s("tier_" + mode + "_restart" + (a > 0 ? "_rerun" : ""), args, {{"WHIRL_KV_SSD_DIR", ssd}}, false);
        for (const auto& ss : S) {
            auto b = chat(t2msg(ss, t1[ss.name]), greedy(96) + thinkKw(ss.think));
            after[ss.name] = b;
            const auto& rb = ref[ss.name].second;
            ck.push_back({std::format("tier {}: {} turn 2 after restart (SSD restore) == never-evicted", mode, ss.name),
                          sameContent(b, rb) && b.reasoning == rb.reasoning,
                          std::format("cached {} (ref {}), prompt {}", b.cached, rb.cached, b.prompt)});
            // CLOSE-COV: the content check alone passes on a cold prefill too; the restored prefix must cover at
            // least what the never-evicted run reused (its turn 1), which only the session's SSD entry can give
            ck.push_back({std::format("tier {}: {} turn 2 after restart restored from SSD (cached >= never-evicted)", mode, ss.name),
                          b.cached > 0 && b.cached >= rb.cached,
                          std::format("cached {} (ref {}), prompt {}", b.cached, rb.cached, b.prompt)});
        }
        const std::size_t n_ssd = s->logCount(R"(kv tier: restored \d+ tok from SSD)");
        ck.push_back({std::format("tier {}: SSD restores logged (one per session)", mode), n_ssd >= S.size(),
                      std::format("{} SSD restores, {} sessions", n_ssd, S.size())});
        lines = s->logLines();
    });
    std::map<std::string, ChatResult> fresh;
    if (p1) {
        // CACHEBAL: the SSD-restored state against a from-scratch prefill of the same prompt, compared by the
        // server itself (WHIRL_PREFIX_CACHE_VERIFY=1: one log line per cache hit with max abs diff / KL / top-1,
        // then it continues from the from-scratch state). This is the CACHE-1 check "continuing after a restart
        // == no prefix cache" in the form that holds by design: the restored history is a gen-end reuse, its KV
        // and DeltaNet state came from the decode / verify kernels, so it is numerically equivalent to a cold
        // prefill, not bit-identical (kv-and-caching.md section 5).
        restartChecksRamRetry("tier p1: verify run", max_entry_mib, [&](int a, std::vector<PendCheck>& ck, std::vector<std::string>& lines) {
        ServerLease s(std::string("tier_p1_verify") + (a > 0 ? "_rerun" : ""), args,
                      {{"WHIRL_KV_SSD_DIR", ssd}, {"WHIRL_PREFIX_CACHE_VERIFY", "1"}}, false);
        for (const auto& ss : S) fresh[ss.name] = chat(t2msg(ss, t1[ss.name]), greedy(96) + thinkKw(ss.think));
        std::vector<CacheVerify> v;
        for (const auto& l : s->logLines())
            if (auto c = parseCacheVerify(l)) v.push_back(*c);
        const std::size_t n_ssd = s->logCount(R"(kv tier: restored \d+ tok from SSD)");
        ck.push_back({"tier p1: verify run: every session restored from SSD, one verify line each", n_ssd >= S.size() && v.size() == S.size(),
                      std::format("{} SSD restores, {} verify lines, {} sessions", n_ssd, v.size(), S.size())});
        // precise (COLD-RESUME): bitwise, max abs diff 0; balance / fast: numerically equivalent (KL bound, same top-1)
        const bool precise_v = serversPrecise();
        for (std::size_t i = 0; i < S.size() && i < v.size(); ++i)
            ck.push_back({precise_v ? std::format("tier p1: {} turn 2 SSD-restored state == from-scratch prefill, bitwise (precise)", S[i].name)
                                    : std::format("tier p1: {} turn 2 SSD-restored state == from-scratch prefill (KL <= {:.0e}, same top-1)",
                                                  S[i].name, kCacheVerifyMaxKl),
                          precise_v ? cacheVerifyBitwise(v[i]) : cacheVerifyOk(v[i]),
                          std::format("max abs diff {:.4f}, KL {:.2e}, top1 {}, differing logits {}", v[i].maxd, v[i].kl,
                                      v[i].top1_same ? "same" : "DIFFERENT", v[i].ndiff)});
        lines = s->logLines();
        });
    }
    fs::remove_all(widen(ssd), ec);
    if (p1) {
        const bool precise = serversPrecise();
        ServerLease s("tier_p1_nocache", {"--parallel", "1", "--kv-ram-mb", "0"}, {{"WHIRL_NO_PREFIX_CACHE", "1"}});
        for (const auto& ss : S) {
            auto b = chat(t2msg(ss, t1[ss.name]), greedy(96) + thinkKw(ss.think));
            // the verify run continued from its from-scratch prefill, which runs the cold schedule: same bits as
            // a server with no prefix cache
            check(std::format("tier p1: {} turn 2 verify run (from-scratch prefill) == no prefix cache", ss.name),
                  sameContent(b, fresh[ss.name]) && b.reasoning == fresh[ss.name].reasoning,
                  std::format("cached {} vs {}", fresh[ss.name].cached, b.cached));
            // CACHE-1: continuing after a restart (SSD restore) == a run with no prefix cache at all. A gate in
            // precise (COLD-RESUME: only cold-exact checkpoints are resumed, so this is bitwise by construction); in
            // balance / fast a next turn resumes from the end of the generated reply, whose history the decode kernels
            // produced: numerically equivalent only (KL ~2e-4 on the R9700, Swift requant), so the greedy text can
            // differ a few tokens in: the KL check above is the gate, this line is information.
            const std::string name = std::format("tier p1: {} turn 2 after restart (SSD restore) == no prefix cache", ss.name);
            const bool same = sameContent(b, after[ss.name]) && b.reasoning == after[ss.name].reasoning;
            const std::string d = std::format("cached {} vs {}", after[ss.name].cached, b.cached);
            if (precise) check(name, same, d);
            else logLine(std::format("INFO {}: {} ({}; not a gate outside precise)", name, same ? "equal" : "differs", d));
        }
    }
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

bool same(const ChatResult& a, const ChatResult& b) { return sameText(a, b) && a.prompt == b.prompt; }

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
        {
            // CLOSE-REUSED: shutdown mode writes only reused entries, also when WHIRL is closed, so the sessions
            // above (one-shot) are no longer written at the stop. Session D (a file in its user message, long
            // enough for a tail checkpoint past the system prompt) resumes for a second turn: its entry becomes
            // reused, so the stop still writes a session next to the system checkpoint and the dedup check below
            // has two entries sharing the system prompt's blocks.
            const std::string du = codeText(12000, 21) + "\n\nIn one sentence, what does this file do?";
            const std::string tools = ",\"tools\":" + sysTools();
            auto d1 = chat(arr({msg("system", SYS1), msg("user", du)}), greedy(48) + thinkKw(false) + tools);
            auto d2 = chat(arr({msg("system", SYS1), msg("user", du), msg("assistant", d1.content),
                                msg("user", "Name one function in that file, in one sentence.")}),
                           greedy(48) + thinkKw(false) + tools);
            check("sys: session D resumed (turn 2 continues its turn 1 past the system prompt)",
                  B && d2.cached > *B + 1024 && !d2.text().empty(),
                  std::format("cached {} (boundary {}, turn 1 prompt {}, turn 2 status {}, {:.0f} ms)", d2.cached, B ? *B : -1, d1.prompt,
                              d2.status, d2.ms));
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    {
        // CACHE-1 block dedup: sessions sharing the system prompt store its KV once. Counted in the sys_tier log
        // after its graceful stop (the default SSD mode writes at shutdown only; the always mode skips these
        // one-shot sessions). DEDUP-INV: every session above shares the system prompt, so every SSD write but
        // the first must reference its blocks (the flush used to write it in full each time: 0 shared).
        const std::regex shared(R"(MiB, [1-9]\d* MiB shared with other entries)"), written(R"(written to SSD: )");
        std::size_t n = 0, w = 0;
        std::string line;
        std::istringstream in(readFile(g.work + "\\server_sys_tier.log"));
        while (std::getline(in, line)) {
            if (std::regex_search(line, written)) ++w;
            if (std::regex_search(line, shared)) ++n;
        }
        check("sys: SSD entries share the system prompt's KV blocks (dedup, written at shutdown)", w >= 2 && n + 1 >= w,
              std::format("{} of {} SSD writes with shared blocks", n, w));
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
            const auto& rr = ref["R" + std::to_string(i)];
            const bool eq = same(res[i], rr);
            ok = ok && eq;
            cached.push_back(res[i].cached);
            if (!eq) {
                // CB-FLAKE: which reply differs, and where (the texts of a rare failure are otherwise lost)
                const std::string a = res[i].text(), b = rr.text();
                std::size_t k = 0;
                while (k < a.size() && k < b.size() && a[k] == b[k]) ++k;
                auto show = [](std::string t) {
                    for (auto& ch : t)
                        if (ch == '\0') ch = '|';
                        else if (ch == '\n') ch = ' ';
                    return t;
                };
                logLine(std::format("sys burst R{} differs: status {}, prompt {} vs {}, cached {}, completion {} vs {}, first diff byte {}\n"
                                    "    got: {}\n    ref: {}",
                                    i, res[i].status, res[i].prompt, rr.prompt, res[i].cached, res[i].completion, rr.completion, k,
                                    show(a), show(b)));
            }
        }
        std::sort(cached.begin(), cached.end());
        const std::size_t n_wait = s->logCount("waiting for the system-message checkpoint");
        check("sys: burst of 3 new sessions: one prefills the system prompt, the others wait and reuse it, == cold",
              ok && cached[0] == 0 && cached[1] > 0 && cached[1] == cached[2] && n_wait >= 1,
              std::format("cached {} {} {}, waits logged {}", cached[0], cached[1], cached[2], n_wait));
    }
}

// ---------------------------------------------------------------------------
// suite sys_burst (CB-FLAKE repro): the "burst of 3 new sessions" step of suite sys, repeated on fresh
// servers (WHIRL_GATE_BURST_REPS, default 10). Each server first runs S0..S2 serially (as sys_off does),
// then R0..R2 arrive 200 ms apart. Every reply is compared with the no-prefix-cache reference; a
// differing reply logs its first differing byte and both texts. Server logs are kept per repetition.

void suiteSysBurst() {
    logLine("== suite sys_burst");
    const std::string SYS1 =
        "You are a coding agent working in the user's repository. Use the tools to look at files before answering. Answer briefly and "
        "precisely. In the chat format a turn ends with the token <|im_end|> (a system text that mentions it must not end the system "
        "message early). Below are the skills manuals available in this workspace.\n" +
        docsText(40000);
    const std::string SYS2 = "You are a code reviewer. Below are the review guidelines of this team.\n" + docsText(30000, 3);
    const std::vector<SysCase> S = {{"S0", SYS1, "In one sentence, what is the main topic of the first skill manual above?", false},
                                    {"S1", SYS1, "Name two tools you can call, in one sentence.", false},
                                    {"S2", SYS1, "Which skill above talks about building? Answer in one short sentence.", false}};
    const std::vector<SysCase> R = {{"R0", SYS2, "Name two tools you can call, in one sentence.", false},
                                    {"R1", SYS2, "Which skill above talks about building? Answer in one short sentence.", false},
                                    {"R2", SYS2, "Answer in one sentence: what is your role?", false}};
    int reps = 10;
    if (const char* e = std::getenv("WHIRL_GATE_BURST_REPS")) reps = std::max(1, std::atoi(e));
    const std::map<std::string, std::string> kv = kvPin();
    std::map<std::string, ChatResult> ref;
    {
        auto env = kv;
        env["WHIRL_NO_PREFIX_CACHE"] = "1";
        ServerLease s("burst_ref", {"--parallel", "4", "--kv-ram-mb", "0"}, env);
        for (const auto* v : {&S, &R})
            for (const auto& c : *v) {
                ref[c.name] = sysAsk(c);
                record("sys_burst/" + c.name, ref[c.name].text());
            }
    }
    auto firstDiff = [](const std::string& a, const std::string& b) {
        std::size_t i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
        return i;
    };
    auto show = [](std::string t) {
        for (auto& ch : t)
            if (ch == '\0') ch = '|';
            else if (ch == '\n') ch = ' ';
        return t;
    };
    // arrival gaps (ms) between R0, R1 and R2, cycled per repetition (WHIRL_GATE_BURST_STAGGER, comma separated;
    // default 200 as in suite sys): every arrival pattern must give the cold texts
    std::vector<int> stag;
    {
        std::string sv = std::getenv("WHIRL_GATE_BURST_STAGGER") ? std::getenv("WHIRL_GATE_BURST_STAGGER") : "200";
        std::stringstream ss(sv);
        std::string it;
        while (std::getline(ss, it, ',')) stag.push_back(std::max(0, std::atoi(it.c_str())));
        if (stag.empty()) stag.push_back(200);
    }
    int bad = 0;
    for (int rep = 0; rep < reps; ++rep) {
        const int gap = stag[static_cast<std::size_t>(rep) % stag.size()];
        ServerLease s(std::format("burst_{:02}", rep), {"--parallel", "4", "--kv-ram-mb", "0"}, kv);
        bool okS = true;
        for (const auto& c : S) {
            auto r = sysAsk(c);
            if (!same(r, ref[c.name])) {
                okS = false;
                logLine(std::format("rep {} {} differs (serial): cached {}, first diff byte {}", rep, c.name, r.cached,
                                    firstDiff(r.text(), ref[c.name].text())));
            }
        }
        std::vector<ChatResult> res(3);
        std::vector<std::thread> th;
        for (int i = 0; i < 3; ++i) {
            th.emplace_back([&, i] { res[i] = sysAsk(R[i]); });
            std::this_thread::sleep_for(std::chrono::milliseconds(gap));
        }
        for (auto& t : th) t.join();
        bool okR = true;
        std::string d;
        for (int i = 0; i < 3; ++i) {
            const auto& rr = ref[R[i].name];
            const bool eq = same(res[i], rr);
            okR = okR && eq;
            d += std::format("{} cached {} prompt {}/{} compl {} {}; ", R[i].name, res[i].cached, res[i].prompt, rr.prompt,
                             res[i].completion, eq ? "=" : "DIFF");
            if (!eq) {
                const std::size_t k = firstDiff(res[i].text(), rr.text());
                logLine(std::format("rep {} {} first diff byte {}\n    got: {}\n    ref: {}", rep, R[i].name, k, show(res[i].text()),
                                    show(rr.text())));
            }
        }
        const std::size_t n_wait = s->logCount("waiting for the system-message checkpoint");
        if (!(okS && okR)) ++bad;
        check(std::format("sys_burst rep {:02} (gap {} ms): S serial == cold, burst == cold", rep, gap), okS && okR,
              d + std::format("waits {}", n_wait));
    }
    logLine(std::format("sys_burst: {} of {} repetitions differ", bad, reps));
}

// ---------------------------------------------------------------------------
// suite http_stress (CB-FLAKE): many short chat requests (a ~60 KB body on a cached system prompt,
// max_tokens 1..8) on one server, sequential and 3 at a time; every request must get HTTP 200 with an
// answer. WHIRL_GATE_HTTP_N requests per phase (default 300). A lost request logs the client-side step
// and WSA error (chat()).

void suiteHttpStress() {
    logLine("== suite http_stress");
    int N = 300;
    if (const char* e = std::getenv("WHIRL_GATE_HTTP_N")) N = std::max(1, std::atoi(e));
    const std::string SYS = "You are a coding agent. Below are the skills manuals of this workspace.\n" + docsText(40000);
    const std::string tools = ",\"tools\":" + sysTools();
    auto ask = [&](int i) {
        return chat(arr({msg("system", SYS), msg("user", std::format("Request {}: name one tool, one word.", i))}),
                    greedy(1 + i % 8) + thinkKw(false) + tools);
    };
    ServerLease s("http_stress", {"--parallel", "4"}, kvPin());
    std::atomic<int> bad{0}, ok{0};
    std::vector<double> slow;
    std::mutex mu;
    auto one = [&](int i) {
        const ChatResult r = ask(i);
        if (answered(r)) ok.fetch_add(1);
        else {
            bad.fetch_add(1);
            logLine(std::format("http_stress: request {} lost: status {}, {:.0f} ms", i, r.status, r.ms));
        }
        if (r.ms > 5000) {
            std::lock_guard<std::mutex> lk(mu);
            slow.push_back(r.ms);
        }
    };
    for (int i = 0; i < N; ++i) one(i);
    const int bad_seq = bad.load();
    {
        std::vector<std::thread> th;
        for (int t = 0; t < 3; ++t)
            th.emplace_back([&, t] {
                for (int i = t; i < N; i += 3) one(100000 + i);
            });
        for (auto& t : th) t.join();
    }
    check("http_stress: every request answered (sequential, then 3 at a time)", bad.load() == 0,
          std::format("{} ok, {} lost ({} sequential), {} slower than 5 s", ok.load(), bad.load(), bad_seq, slow.size()));
}

// suite sys_d (CB-FLAKE): the "session D resumed" step of suite sys repeated (WHIRL_GATE_D_REPS, default 30)
// on one server with the RAM / SSD tiers on: turn 1 (a file in the user message, past the cached system
// prompt) and right after it turn 2 of the same session. Every turn must be answered; turn 2 must resume.

void suiteSysD() {
    logLine("== suite sys_d");
    int reps = 30;
    if (const char* e = std::getenv("WHIRL_GATE_D_REPS")) reps = std::max(1, std::atoi(e));
    const std::string SYS1 =
        "You are a coding agent working in the user's repository. Use the tools to look at files before answering. Answer briefly and "
        "precisely. In the chat format a turn ends with the token <|im_end|> (a system text that mentions it must not end the system "
        "message early). Below are the skills manuals available in this workspace.\n" +
        docsText(40000);
    const std::string tools = ",\"tools\":" + sysTools();
    const std::string ssd = g.work + "\\sys_d_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    int lost1 = 0, lost2 = 0, noresume = 0;
    {
        auto env = kvPin();
        env["WHIRL_KV_SSD_DIR"] = ssd;
        ServerLease s("sys_d", {"--parallel", "4"}, env, false);
        (void)chat(arr({msg("system", SYS1), msg("user", "Answer in one sentence: what is your role?")}), greedy(16) + thinkKw(false) + tools);
        for (int i = 0; i < reps; ++i) {
            const std::string du = codeText(12000, 21 + i % 40) + "\n\nIn one sentence, what does this file do?";
            auto d1 = chat(arr({msg("system", SYS1), msg("user", du)}), greedy(48) + thinkKw(false) + tools);
            if (!answered(d1)) {
                ++lost1;
                logLine(std::format("sys_d rep {}: turn 1 lost (status {}, {:.0f} ms)", i, d1.status, d1.ms));
                continue;
            }
            auto d2 = chat(arr({msg("system", SYS1), msg("user", du), msg("assistant", d1.content),
                                msg("user", "Name one function in that file, in one sentence.")}),
                           greedy(48) + thinkKw(false) + tools);
            if (!answered(d2)) {
                ++lost2;
                logLine(std::format("sys_d rep {}: turn 2 lost (status {}, {:.0f} ms)", i, d2.status, d2.ms));
            } else if (d2.cached <= d1.prompt - 1024) {
                ++noresume;
                logLine(std::format("sys_d rep {}: turn 2 did not resume (cached {}, turn 1 prompt {})", i, d2.cached, d1.prompt));
            }
        }
        s->waitTierIdle(60);
    }
    fs::remove_all(widen(ssd), ec);
    check("sys_d: every turn answered, every turn 2 resumes turn 1", lost1 + lost2 + noresume == 0,
          std::format("{} reps: turn 1 lost {}, turn 2 lost {}, no resume {}", reps, lost1, lost2, noresume));
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
        check("restore_conc: S turn 1 == reference", sameText(a1, r1));
        for (int i = 0; i < 4; ++i) turn(arr({msg("user", codeText(1800, 40 + 3 * i) + "\n\nOne sentence: what is this?")}), 8);
        std::vector<std::function<ChatResult()>> jobs;
        for (int i = 0; i < 3; ++i) jobs.push_back([&, i] { return turn(arr({msg("user", DEC[i])}), 160); });
        jobs.push_back([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            return turn(m2, 48);
        });
        auto res = parallel(jobs);
        check("restore_conc: S turn 2 restored next to 3 decoding requests == never-evicted",
              sameText(res[3], r2) && res[3].cached == r2.cached, std::format("cached {} vs {}", res[3].cached, r2.cached));
        check("restore_conc: decode texts next to the restore == alone",
              sameContent(res[0], rdec[0]) && sameContent(res[1], rdec[1]) && sameContent(res[2], rdec[2]));
        check("restore_conc: RAM restore logged", s->logCount(R"(kv tier: restored \d+ tok from RAM)") >= 1);
        auto b0 = turn(burst(bq, 0), 48);
        check("restore_conc: burst session 0 == reference", sameContent(b0, rb[0]));
        turn(arr({msg("system", sysB), msg("user", "One word: ok?")}), 8);
        const std::size_t w0 = s->logCount("waiting for the restore of its prefix");
        const std::size_t rr0 = s->logCount(R"(kv tier: restored \d+ tok from RAM)");
        std::vector<std::function<ChatResult()>> j2 = {[&] { return turn(burst(bq, 1), 48); }, [&] { return turn(burst(bq, 2), 48); },
                                                       [&] { return turn(burst(bq, 0), 48); }};
        auto res2 = parallel(j2);
        check("restore_conc: burst on a RAM shared entry: texts == reference",
              sameContent(res2[0], rb[1]) && sameContent(res2[1], rb[2]) && sameContent(res2[2], rb[0]));
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
        check("restore_conc: S turn 3 after restart (queued, prefetched, restored) == never-evicted", sameContent(res[4], r3),
              std::format("cached {} (ref {})", res[4].cached, r3.cached));
        const std::size_t npf = s->logCount(R"(prefetching \d+ tok from SSD)");
        const std::size_t nrs = s->logCount(R"(kv tier: restored \d+ tok from (SSD|RAM))");
        check("restore_conc: SSD prefetch while queued + restore logged", npf >= 1 && nrs >= 1, std::format("prefetch {}, restores {}", npf, nrs));
        const std::size_t w0 = s->logCount("waiting for the restore of its prefix");
        std::vector<std::function<ChatResult()>> j2;
        for (std::size_t i = 0; i < 3; ++i) j2.push_back([&, i] { return turn(burst(bq2, i), 48); });
        auto res2 = parallel(j2);
        check("restore_conc: burst of new sessions on an SSD shared entry: texts == reference",
              sameContent(res2[0], rb2[0]) && sameContent(res2[1], rb2[1]) && sameContent(res2[2], rb2[2]));
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

// arch025-visfix (N18): the three suite-vis rules as functions of the server logs and replies, so --selftest can
// plant the R9700 lines and the faults each check was written for.
//
// (1) KV pool with --mmproj against the pool without. Before 0.2.2.2 the pool was equal (the encoder was loaded after
// the pool); since c80b27a (0.2.2.2, CHANGELOG "the startup KV pool leaves room for the vision encoder") a discrete
// GPU subtracts the encoder reserve from the pool budget on purpose and logs it ("kv pool sizing: ... minus N MiB for
// the vision encoder"). Rule: no reserve line (integrated GPU, WHIRL_POOL_RESERVE_MB) -> equal, or <= 5 pages apart with
// the vision line after the pool line (startup VRAM jitter, as before); a reserve line -> the pool is smaller by the
// reserve, within 5 pages of jitter + 1 page of rounding. The fault the check was written for (encoder VRAM taken
// before the pool is sized, on top of the reserve) makes the drop larger than the reserve and fails; a reserve that
// is logged but not applied (drop 0) fails; a discrete GPU with --mmproj and no reserve line fails.
struct VisPoolFacts {
    std::optional<std::int64_t> base, mm;  // "kv pool: N tokens" without / with --mmproj
    std::optional<double> kib_per_tok;     // "..., X KiB/token" of the --mmproj server
    std::optional<std::int64_t> reserve_mib;  // "minus N MiB for the vision encoder"; nullopt: no reserve line
    bool uma = false;                      // the vision line says "[integrated GPU"
    std::size_t i_pool = 0, i_vis = 0;     // 1-based line numbers of the pool line and the first vision line
};
VisPoolFacts visPoolFacts(std::optional<std::int64_t> base, const std::vector<std::string>& mm_lines) {
    VisPoolFacts f;
    f.base = base;
    static const std::regex rp(R"(kv pool: (\d+) tokens.*?, ([\d.]+) KiB/token)"), rr(R"(minus (\d+) MiB for the vision encoder)");
    std::size_t i = 0;
    for (const auto& l : mm_lines) {
        ++i;
        std::smatch m;
        if (std::regex_search(l, m, rp)) {
            f.mm = std::stoll(m[1].str());
            f.kib_per_tok = std::stod(m[2].str());
            f.i_pool = i;
        } else if (l.find("kv pool: ") != std::string::npos) {
            f.i_pool = i;
        }
        if (std::regex_search(l, m, rr)) f.reserve_mib = std::stoll(m[1].str());
        if (l.find("vision: ") != std::string::npos && f.i_vis == 0) {
            f.i_vis = i;
            f.uma = l.find("[integrated GPU") != std::string::npos;
        }
    }
    return f;
}
bool visPoolOk(const VisPoolFacts& f, std::string& d) {
    constexpr std::int64_t page = 256, jitter = 5 * page;
    if (!f.base || !f.mm) {
        d = "kv pool line missing";
        return false;
    }
    const std::int64_t drop = *f.base - *f.mm;
    if (!f.reserve_mib) {
        d = std::format("{} vs {}, no vision reserve line ({} GPU)", *f.mm, *f.base, f.uma ? "integrated" : "discrete");
        // a discrete GPU with --mmproj always reserves (server_main.cpp planPool), unless WHIRL_POOL_RESERVE_MB
        if (!f.uma) return false;
        return drop == 0 || (std::abs(drop) <= jitter && f.i_pool > 0 && f.i_vis > f.i_pool);
    }
    if (!f.kib_per_tok || *f.kib_per_tok <= 0) {
        d = "KiB/token not in the kv pool line";
        return false;
    }
    const double want = static_cast<double>(*f.reserve_mib) * 1024.0 / *f.kib_per_tok;  // tokens the reserve takes
    d = std::format("{} vs {}: smaller by {} tok, the {} MiB vision reserve is {:.0f} tok (allowed +-{} tok)", *f.mm, *f.base, drop,
                    *f.reserve_mib, want, jitter + page);
    return std::abs(static_cast<double>(drop) - want) <= static_cast<double>(jitter + page);
}

// (2) second turn of the 1080p conversation on the main server: it resumes at the end of the first turn's reply
// ('gen-end' checkpoint, cached >= the first turn's prompt). In precise that resume is cold-exact (COLD-RESUME), so the
// answer == the cold answer is the gate. In balance / fast the history from the gen-end checkpoint was written by the
// decode kernels: numerically equivalent to a cold prefill, not bitwise (CACHEBAL, kv-and-caching.md section 5), so the
// text is information here and the resumed state is gated by the restart verify line below (KL bound + same top-1),
// which measures the same state (the restart restores this turn's prompt-end entry).
bool visTurn2Ok(bool precise, const ChatResult& e2, const ChatResult& cold, std::int64_t first_prompt) {
    return answered(e2) && e2.cached >= first_prompt && (!precise || sameText(e2, cold));
}

// (3) restart: the second turn restored from the SSD. precise: the answer == cold (bitwise by COLD-RESUME). balance /
// fast: the restart server runs with WHIRL_PREFIX_CACHE_VERIFY=1: one verify line for the restore (restored state vs a
// from-scratch prefill, KL <= 1e-2 and the same top-1); the server then continues from the from-scratch state, so the
// answer == cold holds in every mode and stays a gate.
bool visRestartOk(bool precise, std::size_t n_restored, std::int64_t cached, const std::vector<CacheVerify>& v, bool same_text,
                  std::string& d) {
    d = std::format("cached {}, {} SSD restore(s)", cached, n_restored);
    const bool base = same_text && n_restored >= 1 && cached >= 2040;
    if (precise) return base;
    d += std::format(", {} verify line(s)", v.size());
    for (const auto& x : v)
        d += std::format("; max abs diff {:.4f}, KL {:.2e}, top1 {}", x.maxd, x.kl, x.top1_same ? "same" : "DIFFERENT");
    return base && v.size() == 1 && cacheVerifyOk(v[0]);
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
    const std::string Q_1080B2 = "And the second most common color? One sentence.";
    std::string m_1080b2;  // built from the reference answer to m_1080b
    const std::map<std::string, std::string> kv = kvPin();
    const bool precise = serversPrecise();  // the suite's mode (release_gate srv_vis: balance)
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
        // second turn of the 1080p conversation: resumed in place on the main server, so its entry is
        // a reused one and the shutdown flush writes it to the SSD (CLOSE-REUSED: one-shot prompts
        // are never written); the restart check restores it
        m_1080b2 = arr({imgMsg({"s1080.png"}, Q_1080B), msg("assistant", ref["s1080b"].content), msg("user", Q_1080B2)});
        ref["s1080b2"] = ask(m_1080b2, 64);
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
        check("vis: MTP greedy == plain greedy (shapes, dialog)", sameText(a, ref["shapes"]) && sameText(b, ref["dialog"]),
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
        // The pool is sized from the VRAM free at startup, which moves by a few pages between two server starts
        // (other processes; the prototype shows the same 201216 / 199936 spread without --mmproj). Discrete GPU:
        // smaller by the logged vision-encoder reserve (0.2.2.2); integrated GPU: equal (visPoolOk above).
        {
            std::string pd;
            const VisPoolFacts pf = visPoolFacts(pool_base, s->logLines());
            check("vis: KV pool with --mmproj == without, minus the logged vision-encoder reserve (discrete GPU)", visPoolOk(pf, pd), pd);
        }
        // device class and pinned mmproj size from the server's vision line; the limit per class is
        // in whirl/device_mem.h (integrated GPU: + the GPU page tables of the pinned copy)
        const auto dv = logReal(*s.s, R"(VRAM change (-?[\d.]+) MiB)");
        const auto pinned_mib = logNum(*s.s, R"((\d+) pinned MiB in host RAM)");
        const bool uma = s->logCount(R"(VRAM change -?[\d.]+ MiB \[integrated GPU)") > 0;
        const hip::MemClass mcls = uma ? hip::MemClass::integrated : hip::MemClass::discrete;
        const double load_lim = hip::loadLimitMiB(mcls, static_cast<std::uint64_t>(pinned_mib ? *pinned_mib : 0) << 20);
        check("vis: VRAM change at mmproj load within the device-class limit (pinned mmproj copy not in VRAM)",
              dv && pinned_mib && std::abs(*dv) <= load_lim,
              std::format("{} MiB ({} GPU, limit {:.2f} MiB)", dv ? *dv : -1.0, uma ? "integrated" : "discrete", load_lim));
        auto t = ask(m_text, 96);
        check("vis: text answer == base, encoder not loaded", sameText(t, base_text) && s->logCount("encoded in") == 0, brief(t));
        auto a = ask(m_shapes, 160);
        check("vis: shapes == ref", sameText(a, ref["shapes"]), brief(a));
        auto a2 = ask(m_shapes, 160);
        check("vis: same image again: prompt fully cached, == ref", sameText(a2, ref["shapes"]) && a2.cached >= a2.prompt - 1,
              std::format("cached {} of {}", a2.cached, a2.prompt));
        auto b = ask(m_dialog, 160);
        check("vis: other image: no reuse of the first image, == ref, answer differs",
              sameText(b, ref["dialog"]) && b.cached == 0 && b.content != a.content, std::format("cached {}", b.cached));
        auto c = ask(m_two, 160);
        check("vis: two images == ref, embeddings from the cache", sameText(c, ref["two"]) && s->logCount("embeddings cached") >= 2,
              brief(c));
        auto d = ask(m_1080, 96);
        check("vis: 1080p == ref", sameText(d, ref["s1080"]), brief(d));
        auto e = ask(m_1080b, 64);
        check("vis: 1080p, new conversation: starts at the image-end checkpoint, == cold",
              sameText(e, ref["s1080b"]) && e.cached >= 2040, std::format("cached {} of {}", e.cached, e.prompt));
        auto e2 = ask(m_1080b2, 64);
        {
            const bool same = sameText(e2, ref["s1080b2"]);
            check(precise ? "vis: 1080p conversation, second turn: resumed (cached >= first turn), == cold"
                          : "vis: 1080p conversation, second turn: resumed (cached >= first turn; state gated by the restart verify)",
                  visTurn2Ok(precise, e2, ref["s1080b2"], e.prompt), std::format("cached {} of {}", e2.cached, e2.prompt));
            if (!precise)
                logLine(std::format("INFO vis: 1080p second turn (gen-end resume) == cold: {} (not a gate outside precise: numerically "
                                    "equivalent, not bitwise)", same ? "equal" : "differs"));
        }
        // one image request next to three text requests
        std::vector<std::function<ChatResult()>> jobs = {[&] { return ask(m_dialog, 160); },
                                                         [&] { return ask(arr({msg("user", "What is RAII? Two sentences.")}), 96); },
                                                         [&] { return ask(m_text, 96); },
                                                         [&] { return ask(arr({msg("user", "Name three sorting algorithms.")}), 64); }};
        std::vector<ChatResult> solo;
        for (const auto& j : jobs) solo.push_back(j());
        auto res = parallel(jobs);
        bool same_all = true;
        for (std::size_t i = 0; i < res.size(); ++i) same_all = same_all && sameText(res[i], solo[i]);
        check("vis: 1 image + 3 text requests concurrently == each alone", same_all && sameText(res[0], ref["dialog"]));
        // idle release
        const bool rel = s->waitLog("encoder released", 1, 40);
        const auto delta = logReal(*s.s, R"(encoder released; VRAM free .* delta (-?[\d.]+) MiB)");
        const double rel_lim = hip::releaseLimitMiB(mcls);
        check("vis: idle release: VRAM back to the level before the first image (device-class limit)", rel && delta && std::abs(*delta) <= rel_lim,
              std::format("delta {} MiB (limit {:.0f} MiB)", delta ? *delta : -1.0, rel_lim));
        auto f = ask(m_shapes, 160);
        check("vis: after the release: shapes == ref (encoder reloaded or cache)", sameText(f, ref["shapes"]), brief(f));
        // two more encoder load / unload cycles (images not in the embedding cache): the release
        // delta must not grow (a leak per cycle), on both device classes
        {
            bool cyc_ok = rel && delta.has_value();
            for (const char* img : {"s512.png", "s1024.png"}) {
                const std::size_t n_rel = s->logCount("encoder released");
                const std::size_t n_enc = s->logCount("encoded in");
                auto r = ask(arr({imgMsg({img}, Q_1080B)}), 8);
                cyc_ok = cyc_ok && r.status == 200 && s->logCount("encoded in") > n_enc && s->waitLog("encoder released", n_rel + 1, 40);
            }
            const auto last = logReal(*s.s, R"(encoder released; VRAM free .* delta (-?[\d.]+) MiB)");
            const double growth = (last && delta) ? *last - *delta : -1.0;
            check("vis: 2 more encoder load/unload cycles: release delta does not grow (<= 2 MiB)",
                  cyc_ok && last && std::abs(growth) <= hip::releaseGrowthLimitMiB,
                  std::format("delta {} -> {} MiB", delta ? *delta : -1.0, last ? *last : -1.0));
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
        s->waitTierIdle(60);  // pending SSD writes done before the stop
    }
    {
        auto env = kv;
        env["WHIRL_KV_SSD_DIR"] = ssd;
        if (!precise) env["WHIRL_PREFIX_CACHE_VERIFY"] = "1";  // the restored state against a from-scratch prefill (CACHEBAL)
        ServerLease s("vis_restart", mm, env, false);
        auto e = ask(m_1080b2, 64);
        std::vector<CacheVerify> v;
        for (const auto& l : s->logLines())
            if (auto c = parseCacheVerify(l)) v.push_back(*c);
        std::string rd;
        const bool ok = visRestartOk(precise, s->logCount(R"(kv tier: restored \d+ tok from SSD)"), e.cached, v, sameText(e, ref["s1080b2"]), rd);
        check(precise ? "vis: restart: 1080p conversation (second turn) restored from SSD, == cold"
                      : std::format("vis: restart: 1080p conversation (second turn) restored from SSD, state == from-scratch prefill "
                                    "(KL <= {:.0e}, same top-1), answer == cold", kCacheVerifyMaxKl),
              ok, std::format("{} of {}; {}", e.cached, e.prompt, rd));
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

// --selftest: CPU-only checks of the comparison helpers (no server, no GPU)
// ---------------------------------------------------------------------------
// suite golden (ARCH-1 golden-hash gate): short deterministic server runs whose outputs print as hash lines
// ("G srv.<case> ..."), recorded and diffed by tools/golden/golden.ps1 (suite server) like the CLI suites.
// One model x one mode per run (WHIRL_MODE from --server-env). Self-contained prompts (no text root), greedy.
// Paths: server prefill on the engine schedule (short, thinking, ~5k-token document), streaming, the
// in-VRAM prefix cache (turn 2 of a session), RAM-tier restore (warm), SSD-tier restore after a restart (cold),
// and, in precise only, a server with no prefix cache (cold prefill of the same turn) and two concurrent
// requests (batched decode). Vision when --mmproj and --images are given (shapes.png).
// In-run relations (checked here, so a run where the paths already disagree can never be recorded):
//   every mode: the stream == the non-stream reply; turn 2 after the SSD restore == turn 2 after the RAM restore
//               (the tier copies bytes); each restore really happened (cached > 0 and the restore log line)
//   precise:    turn 2 restored (RAM, SSD) == turn 2 on a server with no prefix cache (COLD-RESUME: bitwise);
//               each concurrent reply == the same request alone
// A G line carries only deterministic fields: completion / prompt / cached token counts, finish reason, and the
// FNV-1a 64 hash of reasoning + content (+ tool calls).

std::uint64_t fnv64(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

void gout(const std::string& name, const ChatResult& r) {
    std::printf("G srv.%s status=%d n=%lld prompt=%lld cached=%lld finish=%s h=%016llx\n", name.c_str(), r.status,
                static_cast<long long>(r.completion), static_cast<long long>(r.prompt), static_cast<long long>(r.cached),
                r.finish.empty() ? "-" : r.finish.c_str(), static_cast<unsigned long long>(fnv64(r.text())));
    std::fflush(stdout);
}

// A fixed pseudo-document (deterministic LCG over a word list): the same bytes on every machine and forever.
// Changing it changes every srv.* reference ("golden update: ..." in the commit).
std::string goldenDoc(std::uint32_t seed, std::size_t words) {
    static const char* W[] = {"cache", "kernel", "token", "prefill", "decode", "buffer", "stream", "thread", "slot", "page",
                              "matrix", "vector", "scale", "block", "layer", "expert", "router", "state", "queue", "batch",
                              "returns", "updates", "reads", "writes", "checks", "stores", "loads", "maps", "splits", "merges",
                              "the", "a", "each", "every", "one", "two", "first", "last", "next", "previous"};
    constexpr std::size_t nw = sizeof(W) / sizeof(W[0]);
    std::uint32_t x = seed * 2654435761u + 12345u;
    std::string o;
    for (std::size_t i = 0; i < words; ++i) {
        x = x * 1664525u + 1013904223u;
        if (i % 12 == 0) o += std::format("\nSection {}.{}: function f_{}_{}() ", seed, i / 12, seed, i / 12);
        o += W[(x >> 16) % nw];
        o += (i % 12 == 11) ? ". " : " ";
    }
    return o;
}

void suiteGolden() {
    logLine("== suite golden");
    const bool precise = serversPrecise();
    std::printf("G header format=1 suite=server\n");
    const std::string P_SHORT = "請用繁體中文說明快速排序的原理，並用 C++ 寫一個泛型的 quicksort 函式，程式碼註解用中文。";
    const std::string P_PAIR_A = "Write a C function that reverses a singly linked list, and explain it.";
    const std::string P_PAIR_B = "請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。";
    const std::string doc_long = goldenDoc(1, 2400) + "\n\nIn one sentence: what does function f_1_7 do?";
    const std::string doc_a = goldenDoc(2, 1500), doc_b = goldenDoc(3, 1500);
    const std::string q1 = "Name the first function above and what it returns, in one sentence.";
    const std::string q2 = "Now name the last function above, in one sentence.";
    auto t1 = [&](const std::string& d) { return arr({msg("user", d + "\n\n" + q1)}); };
    auto t2 = [&](const std::string& d, const ChatResult& a) {
        return arr({msg("user", d + "\n\n" + q1), msg("assistant", a.content), msg("user", q2)});
    };
    const std::map<std::string, std::string> kv = kvPin();
    const std::string ssd = g.work + "\\golden_ssd";
    std::error_code ec;
    fs::remove_all(widen(ssd), ec);
    std::vector<std::string> args = {"--parallel", "1"};
    const bool vis = !g.mmproj.empty() && !g.images.empty() && fs::exists(widen(g.images + "\\shapes.png"));
    if (vis) args.insert(args.end(), {"--mmproj", g.mmproj});
    auto env = kv;
    env["WHIRL_KV_SSD_DIR"] = ssd;
    ChatResult a1, a2_ram, b1;
    {
        ServerLease s("golden_a", args, env, false);
        const ChatResult sh = chat(arr({msg("user", P_SHORT)}), greedy(48) + thinkKw(false));
        gout("short", sh);
        const ChatResult st = chat(arr({msg("user", P_SHORT)}), greedy(48) + thinkKw(false), true);
        check("golden: stream == non-stream (short)", sameText(st, sh), brief(st));
        gout("think", chat(arr({msg("user", P_PAIR_A)}), greedy(64) + thinkKw(true)));
        gout("long", chat(arr({msg("user", doc_long)}), greedy(32) + thinkKw(false)));
        if (vis) {
            const ChatResult v = chat(arr({imgMsg({"shapes.png"}, "Describe this image: the shapes and their colors.")}), greedy(32) + thinkKw(false));
            check("golden: vision request answered", answered(v), brief(v));
            std::printf("G srv.vis.image file=shapes.png bytes=%llu\n",
                        static_cast<unsigned long long>(fs::file_size(widen(g.images + "\\shapes.png"))));
            gout("vis", v);
        }
        a1 = chat(t1(doc_a), greedy(32) + thinkKw(false));
        gout("sessA.t1", a1);
        b1 = chat(t1(doc_b), greedy(32) + thinkKw(false));  // one slot: A's state leaves VRAM (RAM tier)
        gout("sessB.t1", b1);
        const std::size_t ram0 = s->logCount(R"(kv tier: restored \d+ tok from RAM)");
        a2_ram = chat(t2(doc_a, a1), greedy(32) + thinkKw(false));
        gout("sessA.t2.ram", a2_ram);
        check("golden: session A turn 2 restored from the RAM tier (cached > 0, restore logged)",
              a2_ram.cached > 0 && s->waitLog(R"(kv tier: restored \d+ tok from RAM)", ram0 + 1, 10),
              std::format("cached {}, RAM restores {}", a2_ram.cached, s->logCount(R"(kv tier: restored \d+ tok from RAM)")));
        gout("sessB.t2", chat(t2(doc_b, b1), greedy(32) + thinkKw(false)));
        std::this_thread::sleep_for(std::chrono::seconds(2));
        s->waitTierIdle(60);
    }
    ChatResult a2_ssd;
    {
        ServerLease s("golden_restart", args, env, false);
        a2_ssd = chat(t2(doc_a, a1), greedy(32) + thinkKw(false));
        gout("sessA.t2.ssd", a2_ssd);
        check("golden: session A turn 2 after a restart restored from the SSD tier (cached > 0, restore logged)",
              a2_ssd.cached > 0 && s->logCount(R"(kv tier: restored \d+ tok from SSD)") >= 1,
              std::format("cached {}, SSD restores {}", a2_ssd.cached, s->logCount(R"(kv tier: restored \d+ tok from SSD)")));
        check("golden: turn 2 after the SSD restore == after the RAM restore", sameText(a2_ssd, a2_ram),
              std::format("h {:016x} vs {:016x}", fnv64(a2_ssd.text()), fnv64(a2_ram.text())));
    }
    fs::remove_all(widen(ssd), ec);
    if (precise) {
        auto e2 = kv;
        e2["WHIRL_NO_PREFIX_CACHE"] = "1";
        ServerLease s("golden_cold", {"--parallel", "2", "--kv-ram-mb", "0"}, e2);
        const ChatResult a2_cold = chat(t2(doc_a, a1), greedy(32) + thinkKw(false));
        gout("sessA.t2.cold", a2_cold);
        check("golden: precise: turn 2 restored from RAM == no prefix cache (bitwise, COLD-RESUME)", sameText(a2_ram, a2_cold),
              std::format("h {:016x} vs {:016x}", fnv64(a2_ram.text()), fnv64(a2_cold.text())));
        check("golden: precise: turn 2 restored from SSD == no prefix cache (bitwise, COLD-RESUME)", sameText(a2_ssd, a2_cold));
        std::vector<ChatResult> alone = {chat(arr({msg("user", P_PAIR_A)}), greedy(48) + thinkKw(false)),
                                         chat(arr({msg("user", P_PAIR_B)}), greedy(48) + thinkKw(false))};
        gout("solo0", alone[0]);
        gout("solo1", alone[1]);
        using Job = std::function<ChatResult()>;
        const std::vector<Job> jobs = {[&] { return chat(arr({msg("user", P_PAIR_A)}), greedy(48) + thinkKw(false)); },
                                       [&] { return chat(arr({msg("user", P_PAIR_B)}), greedy(48) + thinkKw(false)); }};
        const auto par = parallel(jobs);
        std::string d;
        check("golden: precise: two concurrent requests == each alone (batched decode, no prefix cache)", concSame(par, alone, d), d);
    }
}

int selfTest() {
    auto mk = [](int status, std::string content, std::string reasoning = "") {
        ChatResult r;
        r.status = status;
        r.content = std::move(content);
        r.reasoning = std::move(reasoning);
        return r;
    };
    const ChatResult err = mk(500, ""), err4 = mk(400, ""), ok = mk(200, "hello"), ok2 = mk(200, "world"), think = mk(200, "", "hmm"),
                     empty200 = mk(200, "");
    std::string d;
    // bug-server-exit-av: the exit code after a Ctrl+C stop
    check("selftest: exit code 0 -> no problem", exitProblem(0).empty());
    check("selftest: exit code 0xC0000005 -> access violation", exitProblem(0xC0000005u) == "exit code 0xC0000005 (0xC0000005 = access violation)",
          exitProblem(0xC0000005u));
    check("selftest: exit code 1 -> an error", exitProblem(1) == "exit code 0x00000001 (1)", exitProblem(1));
    {
        SYSTEMTIME t{};
        t.wYear = 2026, t.wMonth = 10, t.wDay = 26, t.wHour = 1, t.wMinute = 2, t.wSecond = 3, t.wMilliseconds = 45;
        const std::string l = exitLogLine(t, 0xC0000005u);
        check("selftest: exit log line in the server log format (server_log_scan set exit)",
              std::regex_search(l, std::regex(R"(^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} E server exit: )")) &&
                  l.rfind("2026-10-26 01:02:03.045 E server exit: code 0xC0000005", 0) == 0,
              l);
    }
    check("selftest: concurrency, equal answers -> same", concSame({ok, ok2}, {ok, ok2}, d));
    d.clear();
    check("selftest: concurrency, one differs -> not same", !concSame({ok, ok}, {ok, ok2}, d) && d == "=x", d);
    d.clear();
    check("selftest: concurrency, all requests failed (500 / 400) -> not same", !concSame({err, err4}, {err, err4}, d), d);
    d.clear();
    check("selftest: concurrency, 200 with empty content -> not same", !concSame({empty200}, {empty200}, d), d);
    d.clear();
    check("selftest: concurrency, reasoning only (no content) -> not same", !concSame({think}, {think}, d), d);
    check("selftest: sameContent of two failed requests -> false", !sameContent(err, err));
    check("selftest: sameContent of a failure and an empty 200 -> false", !sameContent(err, empty200));
    check("selftest: sameContent of equal answers -> true", sameContent(ok, ok));
    check("selftest: sameText of two failed requests -> false", !sameText(err, err4));
    check("selftest: sameText of a reasoning-only reply with itself -> true", sameText(think, think));
    check("selftest: sameText of different answers -> false", !sameText(ok, ok2));
    {
        ChatResult c0 = ok, c1 = ok;
        c1.cached = 34;
        std::string cd;
        check("selftest: allCold of two cached == 0 replies -> true", allCold({c0, c0}, cd), cd);
        check("selftest: allCold with one cached > 0 reply -> false (a warm round tests decode only)", !allCold({c0, c1}, cd), cd);
        check("selftest: allCold of a failed request -> false", !allCold({err}, cd), cd);
        check("selftest: allCold of no replies -> false", !allCold({}, cd), cd);
    }
    using OS = std::optional<std::string>;
    check("selftest: WHIRL_MODE=fast -> relaxacc on", relaxAccOn(OS("fast"), std::nullopt));
    check("selftest: WHIRL_MODE=FAST / ' fast ' -> relaxacc on (the server parses case-insensitively)",
          relaxAccOn(OS("FAST"), std::nullopt) && relaxAccOn(OS(" fast "), std::nullopt));
    check("selftest: WHIRL_MODE=fast:relaxacc,kvq8 -> relaxacc on", relaxAccOn(OS("fast:relaxacc,kvq8"), std::nullopt));
    check("selftest: WHIRL_MODE=fast:kvq8 (no relaxacc) -> off", !relaxAccOn(OS("fast:kvq8"), std::nullopt));
    check("selftest: WHIRL_MODE=fast + WHIRL_RELAX=0 -> off", !relaxAccOn(OS("fast"), OS("0")));
    check("selftest: balance / unset / invalid -> off",
          !relaxAccOn(OS("balance"), std::nullopt) && !relaxAccOn(std::nullopt, std::nullopt) && !relaxAccOn(OS("bogus"), std::nullopt));
    check("selftest: WHIRL_MODE=precise -> precise; balance / fast / unset / invalid -> not",
          preciseMode(OS("precise")) && preciseMode(OS(" PRECISE ")) && !preciseMode(OS("balance")) && !preciseMode(OS("fast")) &&
              !preciseMode(std::nullopt) && !preciseMode(OS("bogus")));
    {
        // CACHEBAL: verify lines as the server writes them (cachebal/e2, e3 server logs, R9700)
        const auto a = parseCacheVerify("2026-10-20 03:22:10.123 I req 2 | slot 0 | prefix-cache verify: cached vs from-scratch prefill "
                                        "logits: max abs diff 2.8730, KL 1.70e-04, top1 same (760 vs 760)");
        const auto b = parseCacheVerify("I req 1 | slot 0 | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff "
                                        "0.0000, KL 0.00e+00, top1 same (760 vs 760)");
        const auto c = parseCacheVerify("I req 4 | slot 0 | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff "
                                        "9.5000, KL 3.10e-01, top1 DIFFERENT (12 vs 760)");
        const auto e = parseCacheVerify("I req 4 | slot 0 | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff "
                                        "0.4000, KL 2.00e-02, top1 same (760 vs 760)");
        check("selftest: verify line balance (KL 1.7e-4, same top-1) -> parsed, ok",
              a && std::abs(a->maxd - 2.873) < 1e-9 && std::abs(a->kl - 1.7e-4) < 1e-12 && a->top1_same && cacheVerifyOk(*a));
        check("selftest: verify line exact (KL 0) -> ok", b && b->kl == 0 && cacheVerifyOk(*b));
        {
            // REVIEW-022 F3: precise needs the bit count; 0.0000 / KL 0 with 3 differing logits is not bitwise
            const auto z = parseCacheVerify("I req 1 | slot 0 | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff "
                                            "0.0000, KL 0.00e+00, top1 same (760 vs 760), bitwise differing logits 0");
            const auto n3 = parseCacheVerify("I req 1 | slot 0 | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff "
                                             "0.0000, KL 0.00e+00, top1 same (760 vs 760), bitwise differing logits 3");
            check("selftest: verify line differing logits 0 -> bitwise; 3 -> not; no count (old line) -> not",
                  z && z->ndiff == 0 && cacheVerifyBitwise(*z) && n3 && n3->ndiff == 3 && !cacheVerifyBitwise(*n3) && b && !cacheVerifyBitwise(*b));
        }
        check("selftest: verify line top-1 DIFFERENT -> not ok", c && !c->top1_same && !cacheVerifyOk(*c));
        check("selftest: verify line KL 2e-2 > 1e-2 (same top-1) -> not ok", e && e->top1_same && !cacheVerifyOk(*e));
        check("selftest: other log lines -> not a verify line",
              !parseCacheVerify("I req 2 | slot 0 | kv tier: restored 5615 tok from SSD in 165.1 ms (2.22 GB/s)") && !parseCacheVerify(""));
    }
    {
        // mt_cache_precise: cached run vs cold run, request by request
        ChatResult a = ok, a2 = ok2, cold = ok, cold2 = ok2;
        a.prompt = cold.prompt = 120;
        a2.prompt = cold2.prompt = 260;
        a2.cached = 118;
        check("selftest: mt_cache_precise cold diff, all equal (cold cached 0) -> \"\"", mtColdDiff({a, a2}, {cold, cold2}).empty());
        check("selftest: mt_cache_precise cold diff, one text differs -> reported",
              mtColdDiff({a, a2}, {cold, cold}).find("#1 text differs") != std::string::npos);
        ChatResult p = cold2;
        p.prompt = 261;
        check("selftest: mt_cache_precise cold diff, prompt count differs -> reported",
              mtColdDiff({a, a2}, {cold, p}).find("#1 prompt 260 vs 261") != std::string::npos);
        ChatResult w = cold2;
        w.cached = 5;
        check("selftest: mt_cache_precise cold diff, the 'cold' reply was cached -> reported",
              mtColdDiff({a, a2}, {cold, w}).find("cold reply has cached 5") != std::string::npos);
        check("selftest: mt_cache_precise cold diff, two failed requests -> not equal", !mtColdDiff({err}, {err}).empty());
        check("selftest: mt_cache_precise cold diff, count mismatch / empty -> not equal",
              !mtColdDiff({a}, {cold, cold2}).empty() && !mtColdDiff({}, {}).empty());
        const CacheVerify exact{0, 0, true, 0}, tiny{0.0001, 1e-9, true, 7}, flip{0, 0, false, 1}, hidden{0, 0, true, 2};
        std::string vd;
        check("selftest: mt_cache_precise verify, 0.0000 / KL 0 but 2 differing logits -> not ok", !mtPreciseVerifyOk({hidden}, 1, vd), vd);
        check("selftest: mt_cache_precise verify, 2 exact lines for 2 hits -> ok", mtPreciseVerifyOk({exact, exact}, 2, vd), vd);
        check("selftest: mt_cache_precise verify, KL 1e-9 (not bitwise) -> not ok", !mtPreciseVerifyOk({exact, tiny}, 2, vd), vd);
        check("selftest: mt_cache_precise verify, top-1 different -> not ok", !mtPreciseVerifyOk({flip}, 1, vd), vd);
        check("selftest: mt_cache_precise verify, a hit without a verify line -> not ok", !mtPreciseVerifyOk({exact}, 2, vd), vd);
        check("selftest: mt_cache_precise verify, no hits at all -> not ok (tests nothing)", !mtPreciseVerifyOk({}, 0, vd), vd);
    }
    {
        // arch025-visfix (N18): suite vis rules on the R9700 lines of 2026-10-28 (n4_r9700 case_B_rg, server_vis_main.log)
        // and on planted faults
        const std::string sizing = "2026-10-28 21:13:23.671 I kv pool sizing: WDDM budget 31.03 GiB, used 19.27 GiB, margin 2101 MiB "
                                   "(...) -> room 9.71 GiB; free VRAM 12.65 GiB minus 768 MiB -> 11.90 GiB; minus 903 MiB for the vision "
                                   "encoder (--mmproj: weights 887 MiB, activations of a 4096-token image beyond the prefill scratch 0 "
                                   "MiB, fixed 16 MiB) -> 8.83 GiB";
        const std::string sizing_text = "2026-10-28 21:12:24.652 I kv pool sizing: WDDM budget 31.03 GiB, used 19.27 GiB, margin 2101 MiB "
                                        "(...) -> room 9.71 GiB; free VRAM 12.65 GiB minus 768 MiB -> 11.90 GiB";
        auto poolLine = [](std::int64_t n) {
            return std::format("2026-10-28 21:13:24.021 I kv pool: {} tokens ({} pages of 256), q8h (int8 + f16 scale / 32, "
                               "Hadamard-rotated q/k) KV (balance, dense model: q8h), 36.1 KiB/token, 8.82 GiB, shared by 4 slots; "
                               "max 131072 tokens per request", n, n / 256);
        };
        const std::string vis_d = "2026-10-28 21:13:24.031 I vision: C:\\m\\mmproj.gguf: 27 layers, 887 pinned MiB in host RAM (1.1 s; "
                                  "VRAM change 0.0 MiB [discrete GPU: hipMemGetInfo; ...]); encoder loaded on demand";
        const std::string vis_u = "2026-10-14 10:00:00.000 I vision: C:\\m\\mmproj.gguf: 27 layers, 887 pinned MiB in host RAM (1.1 s; "
                                  "VRAM change 0.4 MiB [integrated GPU: ...]); encoder loaded on demand";
        std::string pd;
        const auto real = visPoolFacts(281600, {sizing, poolLine(256000), vis_d});
        check("selftest: vis pool, R9700 281600 -> 256000 with the 903 MiB reserve logged -> ok",
              real.reserve_mib == 903 && real.kib_per_tok == 36.1 && !real.uma && visPoolOk(real, pd), pd);
        // fault the check was written for: the encoder's 887 MiB taken from VRAM before the pool, on top of the reserve
        const auto dbl = visPoolFacts(281600, {sizing, poolLine(256000 - 25088), vis_d});
        check("selftest: vis pool, drop = reserve + encoder weights (encoder VRAM before the pool) -> not ok", !visPoolOk(dbl, pd), pd);
        const auto unapplied = visPoolFacts(281600, {sizing, poolLine(281600), vis_d});
        check("selftest: vis pool, reserve logged but the pool not smaller -> not ok", !visPoolOk(unapplied, pd), pd);
        const auto noline = visPoolFacts(281600, {sizing_text, poolLine(281600), vis_d});
        check("selftest: vis pool, discrete GPU with --mmproj and no reserve line -> not ok", !visPoolOk(noline, pd), pd);
        const auto jit = visPoolFacts(281600, {sizing, poolLine(256000 - 1024), vis_d});
        check("selftest: vis pool, reserve + 4 pages of startup jitter -> ok", visPoolOk(jit, pd), pd);
        const auto uma_eq = visPoolFacts(262144, {sizing_text, poolLine(262144), vis_u});
        check("selftest: vis pool, integrated GPU, no reserve, equal -> ok", uma_eq.uma && visPoolOk(uma_eq, pd), pd);
        const auto uma_j = visPoolFacts(262144, {sizing_text, poolLine(262144 - 1024), vis_u});
        check("selftest: vis pool, integrated GPU, 4 pages jitter, vision line after the pool -> ok", visPoolOk(uma_j, pd), pd);
        const auto uma_bad = visPoolFacts(262144, {sizing_text, poolLine(262144 - 3584), vis_u});
        check("selftest: vis pool, integrated GPU, 14 pages smaller (encoder before the pool) -> not ok", !visPoolOk(uma_bad, pd), pd);
        const auto uma_order = visPoolFacts(262144, {vis_u, sizing_text, poolLine(262144 - 1024)});
        check("selftest: vis pool, integrated GPU, jitter but the vision line before the pool -> not ok", !visPoolOk(uma_order, pd), pd);
        check("selftest: vis pool, no pool line -> not ok", !visPoolOk(visPoolFacts(281600, {sizing, vis_d}), pd), pd);

        // second turn: R9700 balance resumed at gen-end @2086 of 2110, the text differs from the cold answer
        ChatResult t2 = ok, cold = ok2, t2same = ok2;
        t2.prompt = t2same.prompt = 2110;
        t2.cached = t2same.cached = 2086;
        check("selftest: vis turn 2, balance, resumed 2086 >= 2066, text != cold -> ok (state gated by the restart verify)",
              visTurn2Ok(false, t2, cold, 2066));
        ChatResult t2short = t2;
        t2short.cached = 2045;
        check("selftest: vis turn 2, balance, resumed only at the image (2045 < 2066) -> not ok", !visTurn2Ok(false, t2short, cold, 2066));
        ChatResult t2err = mk(500, "");
        t2err.cached = 2086;
        check("selftest: vis turn 2, balance, failed request -> not ok", !visTurn2Ok(false, t2err, cold, 2066));
        check("selftest: vis turn 2, precise, text != cold -> not ok (bitwise by COLD-RESUME)", !visTurn2Ok(true, t2, cold, 2066));
        check("selftest: vis turn 2, precise, text == cold, resumed -> ok", visTurn2Ok(true, t2same, cold, 2066));

        // restart: verify lines as the server writes them
        const auto vbal = parseCacheVerify("2026-10-28 21:14:15.700 I req 1 | slot 0 | prefix-cache verify: cached vs from-scratch prefill "
                                           "logits: max abs diff 2.8730, KL 1.70e-04, top1 same (760 vs 760), bitwise differing logits 9000");
        const auto vbad = parseCacheVerify("2026-10-28 21:14:15.700 I req 1 | slot 0 | prefix-cache verify: cached vs from-scratch prefill "
                                           "logits: max abs diff 9.5000, KL 3.10e-01, top1 DIFFERENT (12 vs 760), bitwise differing logits 9000");
        std::string rd;
        check("selftest: vis restart, balance, restored 2110, KL 1.7e-4 same top-1, answer == cold -> ok",
              vbal && visRestartOk(false, 1, 2110, {*vbal}, true, rd), rd);
        check("selftest: vis restart, balance, KL 3.1e-1 top-1 DIFFERENT (wrong restore) -> not ok",
              vbad && !visRestartOk(false, 1, 2110, {*vbad}, true, rd), rd);
        check("selftest: vis restart, balance, no verify line -> not ok", !visRestartOk(false, 1, 2110, {}, true, rd), rd);
        check("selftest: vis restart, balance, no SSD restore logged -> not ok", vbal && !visRestartOk(false, 0, 2110, {*vbal}, true, rd), rd);
        check("selftest: vis restart, balance, cached 0 (cold prefill) -> not ok", vbal && !visRestartOk(false, 1, 0, {*vbal}, true, rd), rd);
        check("selftest: vis restart, balance, answer != cold after the from-scratch continuation -> not ok",
              vbal && !visRestartOk(false, 1, 2110, {*vbal}, false, rd), rd);
        check("selftest: vis restart, precise, answer == cold, restored -> ok", visRestartOk(true, 1, 2110, {}, true, rd), rd);
        check("selftest: vis restart, precise, answer != cold -> not ok", !visRestartOk(true, 1, 2110, {}, false, rd), rd);
    }
    {
        // arch025-k2warn: restart RAM tier vs the largest SSD entry (the 8060S p1 case: 8192 -> 512 MiB, entry 801 MiB)
        const std::string adj512 = "2026-10-25 16:47:40.100 W adjusted: WHIRL_KV_RAM_MB 8192 -> 512 MiB (the RAM tier is pinned "
                                   "host memory; more than half of the 1954 MiB of RAM available now would make the system page, "
                                   "so it is lowered to half)";
        const std::string adj6144 = "2026-10-25 16:48:22.100 W adjusted: WHIRL_KV_RAM_MB 8192 -> 6144 MiB (the RAM tier is pinned "
                                    "host memory; ...)";
        const std::string refused = "2026-10-25 16:47:50.000 W req 3 | slot 0 | kv tier: SSD entry of 5615 tok (801.1 MiB) is "
                                    "larger than the RAM tier (512.0 MiB) and cannot be read back; prefilling from scratch";
        const std::string other = "2026-10-25 16:47:50.000 I req 3 | slot 0 | kv tier: restored 5615 tok from SSD in 120.0 ms (6.70 GB/s)";
        check("selftest: tier restart, adjusted 512 MiB < entry 802 MiB -> INCONCLUSIVE",
              tierRamInconclusive({adj512, other}, 802).find("lowered at startup to 512 MiB") != std::string::npos,
              tierRamInconclusive({adj512, other}, 802));
        check("selftest: tier restart, adjusted 6144 MiB >= entry 802 MiB -> conclusive", tierRamInconclusive({adj6144, other}, 802).empty());
        check("selftest: tier restart, no adjusted line, no refused line -> conclusive", tierRamInconclusive({other}, 802).empty());
        check("selftest: tier restart, refused-restore W line -> INCONCLUSIVE (with both sizes)",
              tierRamInconclusive({refused}, 0).find("1 SSD restore(s) refused by the restart server (entry 801.1 MiB > RAM tier 512.0 MiB)") !=
                  std::string::npos,
              tierRamInconclusive({refused}, 0));
        int runs = 0, inc_fail = 0;
        const int f0 = g_fail, p0 = g_pass;
        // retry: first attempt fails with a small tier, the second passes -> PASS counted once, no FAIL
        restartChecksRamRetry("selftest retry", 802, [&](int a, std::vector<PendCheck>& ck, std::vector<std::string>& l) {
            ++runs;
            ck.push_back({"selftest: (retry probe)", a > 0, ""});
            l = {a == 0 ? adj512 : adj6144};
        });
        const bool retry_ok = runs == 2 && g_fail == f0 && g_pass == p0 + 1;
        // twice INCONCLUSIVE -> counted as FAIL (then taken back: this is the selftest)
        runs = 0;
        restartChecksRamRetry("selftest retry twice", 802, [&](int, std::vector<PendCheck>& ck, std::vector<std::string>& l) {
            ++runs;
            ck.push_back({"selftest: (twice-inconclusive probe, expected FAIL line)", false, ""});
            l = {adj512};
        });
        inc_fail = g_fail - f0;
        const bool twice_ok = runs == 2 && inc_fail == 1;
        // a plain failure with enough RAM: no re-run, FAIL
        runs = 0;
        restartChecksRamRetry("selftest no retry", 802, [&](int, std::vector<PendCheck>& ck, std::vector<std::string>& l) {
            ++runs;
            ck.push_back({"selftest: (real-failure probe, expected FAIL line)", false, ""});
            l = {adj6144};
        });
        const bool plain_ok = runs == 1 && g_fail - f0 == 2;
        g_fail = f0;
        g_pass = p0;
        while (g_failed.size() > 0 && g_failed.back().find("probe, expected FAIL line") != std::string::npos) g_failed.pop_back();
        check("selftest: tier restart retry: INCONCLUSIVE then PASS -> 2 runs, PASS", retry_ok);
        check("selftest: tier restart retry: INCONCLUSIVE twice -> FAIL with the reason", twice_ok);
        check("selftest: tier restart retry: a failure with enough RAM -> no re-run, FAIL", plain_ok);
    }
    // suite golden: its prompts must never change silently (the expected hashes come from an independent Python
    // replica of goldenDoc / fnv64, ARCH-1 2026-10-20); a change here invalidates every srv.* golden reference
    check("selftest: golden fnv64 of \"\" is the FNV-1a offset basis", fnv64("") == 1469598103934665603ull);
    check("selftest: golden document 1 (2400 words) unchanged", fnv64(goldenDoc(1, 2400)) == 0xc1d46b50c56aaea0ull,
          std::format("{:016x}", fnv64(goldenDoc(1, 2400))));
    check("selftest: golden documents 2 and 3 (1500 words) unchanged",
          fnv64(goldenDoc(2, 1500)) == 0x2b9d4e7c2c35cb70ull && fnv64(goldenDoc(3, 1500)) == 0x2efb3273d0550c17ull);
    // bug-gatectrlc (2026-10-26): a gate whose stdout is a file inherited WITHOUT STARTF_USESTDHANDLES (a runner's
    // `Start-Process pwsh -Command "& gate.exe ..." -RedirectStandardOutput F`) lost every line after its 2nd server stop
    // (the 1st stop's AttachConsole(ATTACH_PARENT_PROCESS) swaps STD_OUTPUT_HANDLE to the console, the 2nd stop then
    // judged stdout a console and reopened it on CONOUT$). Two stops (suite basic: basic + basic_cold) under that wrapper:
    // both stop lines and the summary must reach F.
    for (const int n : {1, 2}) {
        const auto [text, code] = consoleStopRun(n);
        int stop_lines = 0;
        for (std::size_t p = 0; (p = text.find(", exited 1", p)) != std::string::npos; ++p) ++stop_lines;
        const std::string want = std::format("== console child: {} of {} stops ok (summary)", n, n);
        check(std::format("selftest: {} Ctrl+C stop(s) under a pwsh-style file-redirect wrapper -> summary line reaches the file", n),
              code == 0 && stop_lines == n && text.find(want) != std::string::npos,
              std::format("wrap exit {}, {} stop line(s) in the file, file tail: {}", code, stop_lines,
                          text.size() > 160 ? text.substr(text.size() - 160) : text));
    }
    std::printf("== whirl-server-gate --selftest: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

}  // namespace

int wmain(int argc, wchar_t** wargv) {
    // bug-gatectrlc: probe the std handles before any console change (sendCtrlC restores the non-console ones)
    captureStdHandles();
    // the servers we start inherit the "ignore Ctrl+C" attribute: clear one inherited
    // from our own parent so that the Ctrl+C stop reaches them
    SetConsoleCtrlHandler(nullptr, FALSE);
    std::vector<std::string> a;
    for (int i = 1; i < argc; ++i) a.push_back(narrow(wargv[i]));
    if (a.size() == 1 && a[0] == "--selftest") return selfTest();
    // internal processes of the --selftest console check (consoleStopRun)
    if (a.size() == 3 && a[0] == "--selftest-console-wrap") return consoleWrap(std::stoi(a[1]), a[2]);
    if (a.size() == 3 && a[0] == "--selftest-console-child") return consoleChild(std::stoi(a[1]), a[2]);
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
    // suite golden is self-contained (fixed prompts); every other suite reads test texts from the text root
    const bool needs_text = std::any_of(g.suites.begin(), g.suites.end(), [](const std::string& x) { return x != "golden"; });
    if (g.text_root.empty() && !g.exe.empty() && !g.model.empty() && needs_text) {
        std::fprintf(stderr, "set WHIRL_GATE_TEXT_ROOT to the root of a llama.cpp source checkout (test texts)\n");
        return 2;
    }
    if ((g.exe.empty() || g.model.empty()) && g.compare.empty()) {
        std::fprintf(stderr,
                     "usage: whirl-server-gate --exe PATH --kind whirl|proto --model GGUF --suite basic,mt_cache,mt_cache_precise,pool,tier,sys,restore_conc,vis,fast_stream\n"
                     "                         [--work DIR] [--port N] [--results OUT.json] [--compare REF.json] [--cli EXE] [--no-lock]\n"
                     "                         [--mmproj MMPROJ.gguf] [--images DIR]   (suite vis)\n"
                     "                         [--server-env NAME=VALUE ...] [--kv q8v|q8|f16|auto]   (default auto)\n"
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
    const std::map<std::string, std::function<void()>> suites = {{"basic", suiteBasic}, {"mt_cache", suiteMtCache}, {"mt_cache_precise", suiteMtCachePrecise}, {"pool", suitePool},
                                                                 {"tier", suiteTier},   {"sys", suiteSys},           {"sys_burst", suiteSysBurst}, {"http_stress", suiteHttpStress}, {"sys_d", suiteSysD}, {"restore_conc", suiteRestoreConc},
                                                                 {"vis", suiteVis},     {"vis_speed", suiteVisSpeed},
                                                                 {"fast_stream", suiteFastStream}, {"golden", suiteGolden}};
    if (!g.exe.empty()) {
        // a gate that runs nothing must not be green (REVIEW-RC4-C P2-1): no --suite is an error
        if (g.suites.empty()) {
            std::fprintf(stderr, "whirl-server-gate: no suites run (pass --suite NAME[,NAME...])\n");
            return 2;
        }
        for (const auto& name : g.suites) {
            if (!suites.contains(name)) {
                std::fprintf(stderr, "unknown suite %s\n", name.c_str());
                return 2;
            }
        }
        for (const auto& name : g.suites) {
            const int before = g_pass + g_fail;
            try {
                suites.at(name)();
            } catch (const std::exception& ex) {
                check("suite " + name + " ran", false, ex.what());
            }
            // a suite that returned without a single check proves nothing
            if (g_pass + g_fail == before) check("suite " + name + " ran at least one check", false, "0 checks");
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
    if (g_pass + g_fail == 0) {
        std::printf("== whirl-server-gate: 0 checks ran (no suites run): refusing to report success\n");
        return 2;
    }
    std::printf("== whirl-server-gate: %d passed, %d failed%s\n", g_pass, g_fail, g_fail ? "" : " (ALL PASS)");
    for (const auto& f : g_failed) std::printf("   FAILED: %s\n", f.c_str());
    return g_fail ? 1 : 0;
}
