// Platform layer, Windows: the launcher's interactive shell (console facts and input, child processes, the system
// tray). See shell.h, docs/dev/launcher.md and progress/feat026-launcher.PRIOR_ART.md.
// SPDX-License-Identifier: Apache-2.0

#include "platform/shell.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>

namespace whirl::platform::shell {

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

bool isConsoleHandle(DWORD which) {
    DWORD mode = 0;
    const HANDLE h = GetStdHandle(which);
    return h != nullptr && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
}

std::mutex g_child_mu;
std::map<std::uint32_t, HANDLE> g_children;  // pid -> process handle (startChild)

HANDLE childHandle(std::uint32_t pid) {
    std::lock_guard<std::mutex> lk(g_child_mu);
    const auto it = g_children.find(pid);
    return it == g_children.end() ? nullptr : it->second;
}

std::string lastErrorText(const char* what) { return std::string(what) + " failed (Windows error " + std::to_string(GetLastError()) + ")"; }

std::function<void()> g_on_close;
BOOL WINAPI onCtrl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) return TRUE;  // the server child handles it
    if (g_on_close) g_on_close();
    return TRUE;
}

// ---- tray state (one tray per process)
constexpr UINT k_tray_msg = WM_APP + 1;
constexpr UINT_PTR k_timer = 1;
std::atomic<HWND> g_tray_hwnd{nullptr};
const TrayConfig* g_cfg = nullptr;
UINT g_taskbar_created = 0;
NOTIFYICONDATAW g_nid{};

bool addIcon() { return Shell_NotifyIconW(NIM_ADD, &g_nid) != FALSE; }

void showMenu(HWND hwnd) {
    HMENU m = CreatePopupMenu();
    if (m == nullptr) return;
    for (const auto& [id, label] : g_cfg->menu) AppendMenuW(m, MF_STRING, static_cast<UINT_PTR>(id), widen(label).c_str());
    POINT pt{};
    GetCursorPos(&pt);
    SetForegroundWindow(hwnd);  // else the menu does not close when the user clicks elsewhere
    const int cmd = TrackPopupMenu(m, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_NONOTIFY, pt.x, pt.y, 0, hwnd, nullptr);
    PostMessageW(hwnd, WM_NULL, 0, 0);
    DestroyMenu(m);
    if (cmd != 0 && g_cfg->on_item) g_cfg->on_item(cmd);
}

LRESULT CALLBACK trayProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (g_taskbar_created != 0 && msg == g_taskbar_created) {  // Explorer restarted: the icon is gone
        addIcon();
        return 0;
    }
    switch (msg) {
    case k_tray_msg:
        if (lp == WM_RBUTTONUP || lp == WM_CONTEXTMENU) showMenu(hwnd);
        else if (lp == WM_LBUTTONDBLCLK && g_cfg->double_click_id != 0 && g_cfg->on_item) g_cfg->on_item(g_cfg->double_click_id);
        return 0;
    case WM_TIMER:
        if (wp == k_timer && g_cfg->on_tick && !g_cfg->on_tick()) DestroyWindow(hwnd);
        return 0;
    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        KillTimer(hwnd, k_timer);
        Shell_NotifyIconW(NIM_DELETE, &g_nid);
        g_tray_hwnd = nullptr;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

void copyTip(wchar_t* dst, std::size_t cap, const std::string& s) {
    const std::wstring w = widen(s);
    const std::size_t n = std::min(w.size(), cap - 1);
    w.copy(dst, n);
    dst[n] = L'\0';
}

}  // namespace

unsigned consoleProcessCount() {
    DWORD list[8];
    return static_cast<unsigned>(GetConsoleProcessList(list, 8));
}

bool stdinIsConsole() { return isConsoleHandle(STD_INPUT_HANDLE); }
bool stdoutIsConsole() { return isConsoleHandle(STD_OUTPUT_HANDLE); }

std::optional<std::string> readConsoleLine() {
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    if (!stdinIsConsole()) {
        // a pipe or file (the chat window's end-to-end test): UTF-8 bytes up to '\n'. The setup never gets here:
        // it runs only with a console stdin (launcher::decideStart / setupAllowed).
        std::string b;
        for (;;) {
            char c = 0;
            DWORD got = 0;
            if (!ReadFile(in, &c, 1, &got, nullptr) || got == 0) {
                if (b.empty()) return std::nullopt;
                break;
            }
            if (c == '\n') break;
            b += c;
        }
        while (!b.empty() && b.back() == '\r') b.pop_back();
        return b;
    }
    static std::wstring pending;  // typed-ahead / pasted text after the first line end
    std::size_t nl;
    while ((nl = pending.find(L'\n')) == std::wstring::npos) {
        wchar_t buf[256];
        DWORD got = 0;
        if (!ReadConsoleW(in, buf, 256, &got, nullptr) || got == 0) return std::nullopt;  // closed input / Ctrl+C
        pending.append(buf, got);
    }
    std::wstring line = pending.substr(0, nl);
    pending.erase(0, nl + 1);
    while (!line.empty() && line.back() == L'\r') line.pop_back();
    if (!line.empty() && line[0] == 0x1A) return std::nullopt;  // Ctrl+Z
    return narrow(line);
}

void setConsoleTitle(const std::string& title) { SetConsoleTitleW(widen(title).c_str()); }

bool consoleMinimized() {
    const HWND h = GetConsoleWindow();
    return h != nullptr && IsWindowVisible(h) && IsIconic(h);
}

bool hideConsole() {
    const HWND h = GetConsoleWindow();
    if (h == nullptr) return false;
    ShowWindow(h, SW_HIDE);
    return !IsWindowVisible(h);
}

void showConsole() {
    const HWND h = GetConsoleWindow();
    if (h == nullptr) return;
    ShowWindow(h, SW_SHOW);
    ShowWindow(h, SW_RESTORE);
    SetForegroundWindow(h);
}

std::wstring exePathWide() {
    std::wstring p(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, p.data(), static_cast<DWORD>(p.size()));
        if (n == 0) return {};
        if (n < p.size()) {
            p.resize(n);
            return p;
        }
        if (p.size() > 32768) return {};
        p.resize(p.size() * 2);
    }
}

std::string exePath() { return narrow(exePathWide()); }

std::string envUtf8(const char* name) {
    const std::wstring wn = widen(name);
    const DWORD n = GetEnvironmentVariableW(wn.c_str(), nullptr, 0);
    if (n == 0) return {};
    std::wstring v(n, L'\0');
    const DWORD m = GetEnvironmentVariableW(wn.c_str(), v.data(), n);
    v.resize(m);
    return narrow(v);
}

std::string localAppData() { return envUtf8("LOCALAPPDATA"); }

std::string downloadsFolder() {
    PWSTR p = nullptr;
    std::string r;
    if (SHGetKnownFolderPath(FOLDERID_Downloads, 0, nullptr, &p) == S_OK && p != nullptr) r = narrow(p);
    CoTaskMemFree(p);
    return r;
}

std::uint32_t startChild(const std::string& cmdline, std::string* why) {
    std::wstring cl = widen(cmdline);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    // same console (the server's log lines are the log panel), own process group (Ctrl+Break reaches it alone; the
    // server turns Ctrl+C handling back on itself: server_main.cpp SetConsoleCtrlHandler(nullptr, FALSE))
    if (!CreateProcessW(nullptr, cl.data(), nullptr, nullptr, FALSE, CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT, nullptr,
                        nullptr, &si, &pi)) {
        if (why) *why = lastErrorText("CreateProcessW");
        return 0;
    }
    CloseHandle(pi.hThread);
    std::lock_guard<std::mutex> lk(g_child_mu);
    g_children[pi.dwProcessId] = pi.hProcess;
    return pi.dwProcessId;
}

bool startInNewConsole(const std::string& cmdline, std::string* why) {
    std::wstring cl = widen(cmdline);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cl.data(), nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT, nullptr, nullptr,
                        &si, &pi)) {
        if (why) *why = lastErrorText("CreateProcessW");
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

bool sendCtrlBreak(std::uint32_t pid) { return GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) != FALSE; }

bool waitChild(std::uint32_t pid, unsigned timeout_ms, int* exit_code) {
    const HANDLE h = childHandle(pid);
    if (h == nullptr) return true;
    if (WaitForSingleObject(h, timeout_ms) != WAIT_OBJECT_0) return false;
    DWORD code = 0;
    GetExitCodeProcess(h, &code);
    if (exit_code) *exit_code = static_cast<int>(code);
    return true;
}

void killChild(std::uint32_t pid) {
    const HANDLE h = childHandle(pid);
    if (h != nullptr) TerminateProcess(h, 1);
}

void ignoreCtrlC(std::function<void()> on_close) {
    g_on_close = std::move(on_close);
    SetConsoleCtrlHandler(onCtrl, TRUE);
}

bool runTray(const TrayConfig& cfg, std::string* why) {
    g_cfg = &cfg;
    const HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = trayProc;
    wc.hInstance = inst;
    wc.lpszClassName = L"WhirlLauncherTray";
    RegisterClassExW(&wc);  // fails harmlessly when already registered (a second runTray)
    // a hidden top-level window, not a message-only one: only top-level windows receive the TaskbarCreated broadcast
    const HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"WHIRL", WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr, inst, nullptr);
    if (hwnd == nullptr) {
        if (why) *why = lastErrorText("CreateWindowExW");
        return false;
    }
    g_tray_hwnd = hwnd;
    g_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    g_nid = NOTIFYICONDATAW{};
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = k_tray_msg;
    g_nid.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));  // the executable's icon (res/whirl.rc.in: "1 ICON")
    if (g_nid.hIcon == nullptr) g_nid.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512))  /* IDI_APPLICATION */;
    copyTip(g_nid.szTip, sizeof(g_nid.szTip) / sizeof(g_nid.szTip[0]), cfg.tooltip);
    if (!addIcon()) {
        if (why) *why = lastErrorText("Shell_NotifyIconW");
        DestroyWindow(hwnd);
        MSG m;
        while (GetMessageW(&m, nullptr, 0, 0) > 0) DispatchMessageW(&m);  // drain to WM_QUIT
        return false;
    }
    SetTimer(hwnd, k_timer, 500, nullptr);
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    g_cfg = nullptr;
    return true;
}

void quitTray() {
    if (const HWND h = g_tray_hwnd.load()) PostMessageW(h, WM_CLOSE, 0, 0);
}

void trayBalloon(const std::string& title, const std::string& text) {
    if (g_tray_hwnd.load() == nullptr) return;
    NOTIFYICONDATAW n = g_nid;
    n.uFlags = NIF_INFO;
    copyTip(n.szInfoTitle, sizeof(n.szInfoTitle) / sizeof(n.szInfoTitle[0]), title);
    copyTip(n.szInfo, sizeof(n.szInfo) / sizeof(n.szInfo[0]), text);
    n.dwInfoFlags = NIIF_INFO;
    Shell_NotifyIconW(NIM_MODIFY, &n);
}

void messageBox(const std::string& title, const std::string& text) {
    MessageBoxW(nullptr, widen(text).c_str(), widen(title).c_str(), MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND);
}

bool openWithDefaultApp(const std::string& path) {
    const HINSTANCE r = ShellExecuteW(nullptr, L"open", widen(path).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(r) > 32;
}

}  // namespace whirl::platform::shell
