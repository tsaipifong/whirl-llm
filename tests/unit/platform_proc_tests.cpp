// WHIRL unit tests of module platform.proc (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-platform-proc-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/gpu_select.h"
#include "platform/args.h"
#include "platform/proc.h"
#include "platform/sys.h"
#include "whirl/common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// arch025 PL4: proc / sys / args.
// (1) The GPU_WAIT mutex name is the contract between every WHIRL process on one GPU (0.2.4 and 0.2.5 servers wait
//     for each other only when the names are equal): the exact strings of v0.2.4 0e35407 (src/model/device.cpp:
//     lockName widened byte by byte) for a uuid, a PCI address and a PCI address with a domain are written out here.
// (2) Each wrapper == the Win32 call it replaced, evaluated here directly; the WaitForSingleObject results lockGpu and
//     onConsoleCtrl test for (WAIT_OBJECT_0, WAIT_ABANDONED) happen on real objects. Test-only object names (never a
//     real GPU's name: holding one would make a WHIRL server on this machine wait).
// (3) widen / narrow keep flags 0 (invalid input -> U+FFFD, not an error) and narrowArgs == narrow per argument.
static void testPlatformProc() {
#ifdef _WIN32
    namespace pproc = whirl::platform::proc;
    namespace psys = whirl::platform::sys;
    namespace pargs = whirl::platform::args;
    // (1) mutex names
    {
        gpusel::GpuRecord u;
        u.uuid = "0A1B2C3D-4E5F-6071-8293-A4B5C6D7E8F9";
        u.pci_bus = 71;
        CHECK(gpusel::lockMutexName(u) == std::wstring(L"Local\\whirl-gpu-uuid-0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"));
        gpusel::GpuRecord p;
        p.pci_bus = 71;
        CHECK(gpusel::lockMutexName(p) == std::wstring(L"Local\\whirl-gpu-pci-71-0-0"));
        p.pci_bus = 196;
        CHECK(gpusel::lockMutexName(p) == std::wstring(L"Local\\whirl-gpu-pci-196-0-0"));
        gpusel::GpuRecord d;
        d.pci_domain = 1;
        d.pci_bus = 3;
        d.pci_device = 2;
        d.pci_function = 1;
        d.index = 5;  // never part of the name
        CHECK(gpusel::lockMutexName(d) == std::wstring(L"Local\\whirl-gpu-pci-3-2-1-d1"));
        const std::string n = gpusel::lockName(d);
        CHECK(gpusel::lockMutexName(d) == std::wstring(n.begin(), n.end()));
    }
    // (2) process calls
    {
        wchar_t buf[1024];
        const DWORD n = GetModuleFileNameW(nullptr, buf, 1024 - 1);
        CHECK(n != 0);
        buf[n] = 0;
        CHECK(pproc::exePath() == std::wstring(buf));
        CHECK(pproc::currentId() == GetCurrentProcessId());
        CHECK(pproc::currentProcess() == GetCurrentProcess());
        std::uint64_t ws = 0;
        CHECK(pproc::workingSetBytes(pproc::currentProcess(), &ws));
        CHECK(ws > 0);
        std::uint64_t total = 0, avail = 0;
        CHECK(psys::memoryStatus(&total, &avail));
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        CHECK(GlobalMemoryStatusEx(&ms) != 0);
        CHECK(total == ms.ullTotalPhys);
        CHECK(avail > 0 && avail <= total);
    }
    {
        // manual-reset event: not signaled, then signaled for every wait
        pproc::Handle ev = pproc::createEvent();
        CHECK(ev != nullptr);
        if (ev != nullptr) {
            CHECK(pproc::wait(ev, 0) == pproc::kWaitTimeout);
            CHECK(pproc::setEvent(ev));
            CHECK(pproc::wait(ev, 0) == pproc::kWaitObject0);
            CHECK(pproc::wait(ev, 0) == pproc::kWaitObject0);
            CloseHandle(static_cast<HANDLE>(ev));
        }
        const auto t0 = std::chrono::steady_clock::now();
        pproc::sleepMs(20);
        CHECK(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(15));
    }
    {
        // named mutex: owned by the first waiter, busy for another thread, abandoned when its owner thread ends
        const std::wstring name = L"Local\\whirl-pl4-test-" + std::to_wstring(GetCurrentProcessId());
        pproc::Handle a = pproc::createMutex(name.c_str());
        CHECK(a != nullptr);
        if (a != nullptr) {
            CHECK(pproc::wait(a, 0) == pproc::kWaitObject0);
            std::uint32_t other = 0;
            std::thread([&] {
                pproc::Handle b = pproc::createMutex(name.c_str());  // the same mutex (bInitialOwner ignored)
                other = b != nullptr ? pproc::wait(b, 0) : 0xFFFFFFFFu;
                if (b != nullptr) CloseHandle(static_cast<HANDLE>(b));
            }).join();
            CHECK(other == pproc::kWaitTimeout);
            ReleaseMutex(static_cast<HANDLE>(a));
            CloseHandle(static_cast<HANDLE>(a));
        }
        const std::wstring name2 = name + L"-abandoned";
        pproc::Handle c = pproc::createMutex(name2.c_str());
        CHECK(c != nullptr);
        if (c != nullptr) {
            std::thread([&] {
                pproc::Handle d = pproc::createMutex(name2.c_str());
                CHECK(d != nullptr && pproc::wait(d, 0) == pproc::kWaitObject0);
                // the thread ends holding it (lockGpu's case: a WHIRL process killed while it holds the GPU)
            }).join();
            CHECK(pproc::wait(c, 1000) == pproc::kWaitAbandoned);
            ReleaseMutex(static_cast<HANDLE>(c));
            CloseHandle(static_cast<HANDLE>(c));
        }
    }
    // (3) conversions
    {
        const std::string u8 = "a\xC3\xA9\xF0\x9F\x98\x80z";  // a, e-acute, U+1F600 (a surrogate pair), z
        CHECK(pargs::wideFromUtf8(u8.data(), static_cast<int>(u8.size()), nullptr, 0) == 5);
        CHECK(widen(u8) == std::wstring(L"aé\U0001F600z"));
        CHECK(narrow(widen(u8)) == u8);
        CHECK(pargs::utf8FromWide(L"é", 1, nullptr, 0) == 2);
        // flags 0: an invalid sequence is replaced, never an empty result (MB_ERR_INVALID_CHARS would fail)
        CHECK(widen(std::string("x\xC3")) == std::wstring(L"x�"));
        const wchar_t lone[] = {L'y', static_cast<wchar_t>(0xD800), 0};
        CHECK(narrow(lone) == std::string("y\xEF\xBF\xBD"));
        // == the direct calls
        const std::string bad = "q\xFF\xFE\x80r";
        const int wn = MultiByteToWideChar(CP_UTF8, 0, bad.data(), static_cast<int>(bad.size()), nullptr, 0);
        std::wstring ref(static_cast<std::size_t>(wn), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, bad.data(), static_cast<int>(bad.size()), ref.data(), wn);
        CHECK(widen(bad) == ref);
        CHECK(widen("").empty() && narrow(L"").empty());
        const wchar_t* wargv[] = {L"whirl", L"--x", L"中文", L""};
        const std::vector<std::string> args = narrowArgs(4, wargv);
        CHECK(args.size() == 4);
        if (args.size() == 4) {
            CHECK(args[0] == "whirl" && args[1] == "--x" && args[2] == "\xE4\xB8\xAD\xE6\x96\x87" && args[3].empty());
            for (int i = 0; i < 4; ++i) CHECK(args[static_cast<std::size_t>(i)] == narrow(wargv[i]));
        }
        CHECK(narrowArgs(0, wargv).empty());
    }
#endif
}

#ifdef _WIN32
// arch025-surv (mutation survivors pl4-exepath-nsize, pl4-ctrl-always, pl4-ws-peak; progress\arch025.MUTATION.md
// class C): the platform::proc wrappers on the inputs the PL4 tests did not reach. Real OS, no seam: the exe-path and
// console-control cases run this program as a child (the --child-lappdata pattern).

// --child-exepath: exit code = platform::proc::exePath().size() (the parent starts a copy from a long path)
static int childExePath() { return static_cast<int>(whirl::platform::proc::exePath().size()); }

// --child-ctrl-passon: a handler registered through addCtrlHandler that returns false must pass the event on to the
// handler registered before it (the OS calls the last-registered handler first). Exit code: 1 = our handler ran,
// + 2 = the earlier handler got the event too. Runs in its own console (the parent's CREATE_NEW_CONSOLE), so the event
// reaches no other process.
static std::atomic<int> g_ctrl_seen{0};
static std::atomic<int> g_ctrl_passed{0};
static BOOL WINAPI ctrlEarlier(DWORD type) {
    if (type == CTRL_BREAK_EVENT) g_ctrl_passed.store(1);
    return TRUE;  // handled: the default handler (ExitProcess) is not reached
}
static bool ctrlPassOn(int type) {
    if (type == whirl::platform::proc::kCtrlBreak) g_ctrl_seen.store(1);
    return false;
}
static int childCtrlPassOn() {
    if (!SetConsoleCtrlHandler(ctrlEarlier, TRUE)) return 10;
    if (!whirl::platform::proc::addCtrlHandler(ctrlPassOn)) return 11;
    if (!GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0)) return 12;
    for (int i = 0; i < 500 && !(g_ctrl_seen.load() && g_ctrl_passed.load()); ++i) Sleep(10);
    Sleep(100);  // a handler called late would still change the result
    return g_ctrl_seen.load() + 2 * g_ctrl_passed.load();
}

// starts `exe` (any path form CreateProcessW takes) with one argument, hidden, its own console; exit code or ~0u
static DWORD runSelfChild(const std::wstring& exe, const wchar_t* arg, bool new_console) {
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + exe + L"\" " + arg;
    const DWORD flags = new_console ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW;
    if (!CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi)) return ~0u;
    if (WaitForSingleObject(pi.hProcess, 20000) != WAIT_OBJECT_0) {
        TerminateProcess(pi.hProcess, 99);
        WaitForSingleObject(pi.hProcess, 5000);  // arch025-surv2: asynchronous; (c) deletes the exe's folder next
    }
    DWORD code = ~0u;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}

static void testPlatformProcSurvivors() {
    namespace pproc = whirl::platform::proc;
    std::vector<wchar_t> selfb(32768);
    const DWORD sn = GetModuleFileNameW(nullptr, selfb.data(), static_cast<DWORD>(selfb.size()));
    CHECK(sn > 0 && sn < selfb.size());
    if (sn == 0 || sn >= selfb.size()) return;
    const std::wstring self(selfb.data(), sn);

    // (a) pl4-ws-peak: workingSetBytes is the CURRENT working set. Touch 256 MiB, release it: the peak keeps it, the
    //     current working set drops again (the old test only checked > 0).
    {
        const std::size_t big = std::size_t{256} << 20;
        void* p = VirtualAlloc(nullptr, big, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        CHECK(p != nullptr);
        if (p != nullptr) {
            std::memset(p, 0x5A, big);
            VirtualFree(p, 0, MEM_RELEASE);
        }
        std::uint64_t ws = 0;
        CHECK(pproc::workingSetBytes(pproc::currentProcess(), &ws));
        PROCESS_MEMORY_COUNTERS pmc{};
        CHECK(K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) != 0);
        CHECK(pmc.PeakWorkingSetSize >= big);
        CHECK(ws + (std::uint64_t{128} << 20) <= pmc.PeakWorkingSetSize);  // the mutant returns the peak itself
        const std::uint64_t cur = pmc.WorkingSetSize;
        CHECK((ws > cur ? ws - cur : cur - ws) < (std::uint64_t{64} << 20));  // == the direct read (moves a little)
    }

    // (b) pl4-ctrl-always: a handler returning false passes the event on (FALSE), true would swallow it
    CHECK_EQ(runSelfChild(self, L"--child-ctrl-passon", true), static_cast<DWORD>(3));

    // (c) pl4-exepath-nsize: started from a path of >= 1024 characters, exePath() is that path truncated to 1022
    //     characters (GetModuleFileNameW(.., 1023) into a 1024-character buffer: 1022 + the terminating null; nSize
    //     1024 would give 1023 and write buf[1024], one past the array). The copy uses the \\?\ form, which bypasses
    //     MAX_PATH whatever LongPathsEnabled says; the start retries without the prefix. arch025-surv2 (REVIEW F4): a
    //     failed copy or start is a FAILED check, not a skip (a skipped case would let the mutant survive unseen).
    {
        namespace fs = std::filesystem;
        const fs::path base = whirl::test::testTmpBase() / (L"pl4-longpath-" + std::to_wstring(GetCurrentProcessId()));
        // \\?\ turns off path normalisation: give it an absolute, normalised, backslash-only path (WHIRL_TEST_TMP may
        // be relative or use forward slashes)
        const std::wstring base_w = fs::absolute(base).lexically_normal().make_preferred().wstring();
        std::wstring dir = L"\\\\?\\" + base_w;
        const std::wstring seg(200, L'd');
        while (dir.size() - 4 + 1 + 16 < 1100) dir += L"\\" + seg;
        std::error_code ec;
        fs::create_directories(dir, ec);
        const std::wstring exe = dir + L"\\whirl-tests.exe";
        const bool copied = !ec && CopyFileW(self.c_str(), exe.c_str(), FALSE) != 0;
        if (!copied) {
            const DWORD le = GetLastError();
            std::printf("  FAIL: testPlatformProcSurvivors (c): copy to a %zu-character path failed (create_directories: %s, "
                        "CopyFileW: %lu; temp folder on FAT / network, or the copy blocked by antivirus?)\n",
                        exe.size() - 4, ec.message().c_str(), le);
            CHECK(copied);
        } else {
            const std::size_t len = exe.size() - 4;  // without the \\?\ prefix: what GetModuleFileNameW returns
            CHECK(len >= 1024);
            DWORD rc = runSelfChild(exe, L"--child-exepath", false);
            if (rc == ~0u) rc = runSelfChild(exe.substr(4), L"--child-exepath", false);
            if (rc == ~0u)
                std::printf("  FAIL: testPlatformProcSurvivors (c): CreateProcessW from a %zu-character path failed (%lu)\n",
                            len, GetLastError());
            CHECK_EQ(rc, static_cast<DWORD>(1022));
        }
        fs::remove_all(L"\\\\?\\" + base_w, ec);
    }
}
#endif

}  // namespace

namespace {
// the old single main()'s helper modes of these tests (the test runs this program as the child)
static int childModes(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 2 && std::string(argv[1]) == "--child-exepath") return childExePath();  // arch025-surv helper mode
    if (argc == 2 && std::string(argv[1]) == "--child-ctrl-passon") return childCtrlPassOn();  // arch025-surv helper mode
#endif
    (void)argc;
    (void)argv;
    return -1;
}
}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_platform_proc() {
    static const whirl::test::Suite s{"platform.proc",
                                      {
                                          {"testPlatformProc", [] { testPlatformProc(); }},
#if defined(_WIN32)
                                          {"testPlatformProcSurvivors", [] { testPlatformProcSurvivors(); }},
#endif
                                      },
                                      childModes};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_platform_proc()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
