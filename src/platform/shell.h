// Platform layer: the interactive shell of the Windows launcher (feat026-launcher, 0.2.6). See docs/dev/platform.md
// and docs/dev/launcher.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is shell_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
// Used only by src/launcher (whirl.exe without arguments, `whirl --setup`, the tray chat window); nothing else
// links these functions, so no other program changes.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace whirl::platform::shell {

// ---- how this process was started
unsigned consoleProcessCount();  // GetConsoleProcessList: processes attached to this console; 0 = no console
bool stdinIsConsole();           // standard input is a console (GetConsoleMode succeeds on it)
bool stdoutIsConsole();          // standard output is a console

// ---- console input / output (UTF-8)
// One line typed at the console (ReadConsoleW, UTF-16 -> UTF-8, without the line end; text typed ahead is kept for
// the next call); nullopt at Ctrl+Z / Ctrl+C / a closed input. A non-console stdin (the chat window's end-to-end
// test) is read as UTF-8 bytes up to the line feed.
std::optional<std::string> readConsoleLine();
void setConsoleTitle(const std::string& title);
// The console window (GetConsoleWindow): minimised / hide / show (restored and in front). Under a Windows Terminal
// that ignores show / hide of its pseudo window, hide() returns false and the window stays (prior art section 2).
bool consoleMinimized();
bool hideConsole();
void showConsole();

// ---- folders and paths (UTF-8)
std::wstring exePathWide();   // this executable (GetModuleFileNameW); empty if unknown
std::string exePath();
std::string localAppData();   // %LOCALAPPDATA% (the tune cache's parent); "" if unset
std::string downloadsFolder();  // the user's Downloads folder (known folder); "" if unknown
std::string envUtf8(const char* name);  // an environment variable read as UTF-16 (non-ASCII user names); "" if unset

// ---- child processes (the server `whirl.exe serve ...`, the chat window)
// Starts cmdline (UTF-8, already quoted) sharing this console, in a new process group (so Ctrl+Break can reach it
// alone). Returns the process id, 0 on failure (*why set).
std::uint32_t startChild(const std::string& cmdline, std::string* why);
// Starts cmdline in a NEW console window (the chat window). Returns false on failure.
bool startInNewConsole(const std::string& cmdline, std::string* why);
// Ctrl+Break to the child's process group (the server's graceful shutdown). false if it could not be sent.
bool sendCtrlBreak(std::uint32_t pid);
// Waits up to timeout_ms; true when the child has exited (*exit_code set).
bool waitChild(std::uint32_t pid, unsigned timeout_ms, int* exit_code);
// TerminateProcess on a child started by startChild (last resort after the server's own timeout).
void killChild(std::uint32_t pid);
// This process ignores Ctrl+C / Ctrl+Break (the server child in the same console handles them and exits; the
// launcher sees the child exit). on_close runs when the console window is closed / at logoff (then the process
// ends when on_close returns, or after a few seconds).
void ignoreCtrlC(std::function<void()> on_close);

// ---- system tray
struct TrayConfig {
    std::string tooltip;                            // e.g. "WHIRL server: 127.0.0.1:8080"
    std::vector<std::pair<int, std::string>> menu;  // id, label in order
    int double_click_id = 0;                        // the item a double click on the icon runs
    std::function<void(int)> on_item;               // runs on the tray thread
    std::function<bool()> on_tick;                  // every 500 ms on the tray thread; false = leave the loop
};
// Adds the icon (the executable's icon resource 1) and runs the message loop on the calling thread until on_tick
// returns false or quitTray() is called; removes the icon before it returns. Re-adds the icon when Explorer
// restarts (TaskbarCreated). false when the icon could not be added (*why set).
bool runTray(const TrayConfig& cfg, std::string* why);
void quitTray();  // from any thread
void trayBalloon(const std::string& title, const std::string& text);  // a notification from the icon (may be ignored)
void messageBox(const std::string& title, const std::string& text);
// Opens a file with its default program (ShellExecuteW "open"; the settings file in Notepad). false on failure.
bool openWithDefaultApp(const std::string& path);

}  // namespace whirl::platform::shell
