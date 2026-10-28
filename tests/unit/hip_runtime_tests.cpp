// WHIRL unit tests of module hip.runtime (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-hip-runtime-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/kernels_abi.h"
#include "hip/alloc_log.h"
#include "hip/launch_log.h"
#include "hip/log_name.h"
#include "platform/env.h"
#include "whirl/common.h"
#include "whirl/uma_mem.h"

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
// REVIEW-ARCH4 P3-2 / arch6e: the launch-log argument digest must be the same for the same argument values,
// whatever memory the argument struct was built in (two runs of one build differed: the GEMV launches' GvArgs
// were built on stack memory that held different bytes), and must still change when an argument changes.
void testLaunchDigest() {
    using whirl::kernels::GvArgs;
    auto build = [](unsigned char fill, unsigned char* buf) {
        std::memset(buf, fill, sizeof(GvArgs));  // what the stack held before (differs from run to run)
        GvArgs* g = new (buf) GvArgs;            // as forward.cpp's gemvKernels builds it: `GvArgs g;` + fields
        g->w[0] = 0x358f40000;
        g->w[1] = 0x359f40000;
        g->y[0] = 0x7dddd0000;
        g->y[1] = 0x7ddde0000;
        g->rb[0] = g->rb[1] = 0x440;
        g->n[0] = g->n[1] = 512;
        g->b1 = 64;
        return g;
    };
    alignas(GvArgs) unsigned char a[sizeof(GvArgs)], b[sizeof(GvArgs)];
    GvArgs* ga = build(0xAA, a);
    GvArgs* gb = build(0x55, b);
    auto digest = [](const GvArgs* g) {
        const std::int32_t ncols = 2048;  // the scalar arguments that follow it in the real launch
        return whirl::hip::launchlog::launchArgDigest(*g, ncols);
    };
    const std::uint64_t da = digest(ga), db = digest(gb);
    CHECK_EQ(da, db);  // same argument values -> same digest
    CHECK(std::memcmp(a, b, sizeof(GvArgs)) == 0);  // ... because the struct carries no undefined bytes
    // still sensitive to a real argument change (one field of one matrix)
    GvArgs* gc = build(0xAA, b);
    gc->n[1] = 513;
    CHECK(digest(gc) != da);
    gc = build(0xAA, b);
    gc->b1 = 65;
    CHECK(digest(gc) != da);
    gc = build(0xAA, b);
    gc->b2 = 96;
    CHECK(digest(gc) != da);
}

// arch6f (REVIEW-ARCH4 P3-2): the launch-log digest of the same argument values must not depend on where the
// device allocations happen to lie. The golden runs E_r97 / E2_r97 (same build) differed in 2 of 8 steps: int32
// pairs of all-integer argument structs (AwGroups first[0..1] = 0,5 -> word 0x500000000, Tok16, RowIdx, the
// row0/nrows of GdnSegs, KvArgs tab0/win) were read as "inside allocation 597" in one run and as a value in the other.
void testLaunchDigestLayout() {
    using namespace whirl::kernels;
    namespace ll = whirl::hip::launchlog;
    struct Layout {
        std::uint64_t base[3];
    };
    // A: the bases put 0x500000000, 0x800000ae0 and 0x4000000000 inside allocations 0..2 (as in the failing runs);
    // B: nowhere near them
    const Layout la{{0x4FE000000ull, 0x7FF000000ull, 0x3F00000000ull}};
    const Layout lb{{0x2000000000ull, 0x2100000000ull, 0x6000000000ull}};
    const std::uint64_t bytes[3] = {0x4000000ull, 0x2000000ull, 0x200000000ull};
    auto with = [&](const Layout& l, auto&& f) {
        ll::testClearAllocs();
        for (int i = 0; i < 3; ++i) ll::testAlloc(l.base[i], bytes[i]);
        const std::uint64_t d = f(l);
        ll::testClearAllocs();
        return d;
    };
    AwGroups aw;
    aw.first[1] = 5;
    aw.count[0] = aw.count[1] = 5;
    Tok16 tk;
    tk.t[4] = 0xae0;
    tk.t[5] = 8;
    RowIdx ri;
    ri.v[1] = 0x40;
    auto dAw = [&](const AwGroups& g) {
        return [&g](const Layout&) { return ll::launchArgDigest(g, std::int32_t{4}); };
    };
    auto dKv = [](std::int32_t win, std::uint64_t koff, int kalloc) {
        return [=](const Layout& l) {
            KvArgs kv;
            kv.k = l.base[kalloc] + koff;
            kv.v = l.base[1] + 0x40;
            kv.tab0 = 0;
            kv.win = win;
            return ll::launchArgDigest(kv, DevPtr{l.base[0] + 0x2000}, std::int32_t{7});
        };
    };
    auto dGdn = [](std::int32_t nrows0) {
        return [=](const Layout& l) {
            GdnSegs s;
            s.s[0].state = l.base[2] + 0x80;
            s.s[0].row0 = 0;
            s.s[0].nrows = nrows0;
            s.s[1].state = l.base[2] + 0x1080;
            s.s[1].row0 = 5;
            s.s[1].nrows = 5;
            return ll::launchArgDigest(s);
        };
    };
    // same values -> same digest in both layouts
    CHECK_EQ(with(la, dAw(aw)), with(lb, dAw(aw)));
    CHECK_EQ(with(la, [&](const Layout&) { return ll::launchArgDigest(tk); }), with(lb, [&](const Layout&) { return ll::launchArgDigest(tk); }));
    CHECK_EQ(with(la, [&](const Layout&) { return ll::launchArgDigest(ri); }), with(lb, [&](const Layout&) { return ll::launchArgDigest(ri); }));
    CHECK_EQ(with(la, dKv(5, 0x100, 0)), with(lb, dKv(5, 0x100, 0)));
    CHECK_EQ(with(la, dGdn(5)), with(lb, dGdn(5)));
    // still sensitive to a real argument change: an integer field, a pointer offset, the allocation pointed into
    AwGroups aw2 = aw;
    aw2.first[1] = 6;
    CHECK(with(la, dAw(aw2)) != with(la, dAw(aw)));
    CHECK(with(lb, dAw(aw2)) != with(lb, dAw(aw)));
    Tok16 tk2 = tk;
    tk2.t[5] = 9;
    CHECK(with(la, [&](const Layout&) { return ll::launchArgDigest(tk2); }) != with(la, [&](const Layout&) { return ll::launchArgDigest(tk); }));
    CHECK(with(la, dKv(6, 0x100, 0)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(la, dKv(5, 0x110, 0)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(lb, dKv(5, 0x110, 0)) != with(lb, dKv(5, 0x100, 0)));
    CHECK(with(la, dKv(5, 0x100, 1)) != with(la, dKv(5, 0x100, 0)));
    CHECK(with(la, dGdn(4)) != with(la, dGdn(5)));
    // a pointer is still normalized: the same (allocation, offset) in both layouts gives the same digest although
    // the raw addresses differ (checked above through KvArgs / GdnSegs), and a raw address change alone does not
    // count as an argument change
}

// arch024-S0a (REVIEW-ARCH6-code C-1): a pointer word whose value is 0 (an unused pointer: single-matrix GvArgs
// w[1..2] / y[1..2], the f16 pool's KvArgs ks / vs) and a small u64 scalar (forward.cpp w.row_bytes) are values. Before
// the fix they fell into the "below the next allocation" rule whenever the lowest allocation was at least as large as
// its base address (one allocation of ~12 GB or more), and were then digested as (ordinal, base - v), which depends on
// where that allocation lies. Also pins the adjacent-allocation case: a pointer inside an allocation that touches the
// next one stays "inside" its own allocation in both layouts.
void testLaunchDigestZeroAndAdjacent() {
    namespace ll = whirl::hip::launchlog;
    using whirl::kernels::KvArgs;
    const auto with = [&](std::initializer_list<std::pair<std::uint64_t, std::uint64_t>> al, auto&& f) {
        ll::testClearAllocs();
        for (const auto& [b, n] : al) ll::testAlloc(b, n);
        const std::uint64_t d = f();
        ll::testClearAllocs();
        return d;
    };
    // the lowest allocation is larger than its base address in both layouts, at different bases
    const std::uint64_t big = 0x400000000ull;  // 16 GiB
    const auto la = {std::pair{0x300000000ull, big}, std::pair{0x800000000ull, 0x1000000ull}};
    const auto lb = {std::pair{0x380000000ull, big}, std::pair{0x900000000ull, 0x1000000ull}};
    const auto zero = [] { return ll::launchArgDigest(std::uint64_t{0}, std::int32_t{3}); };
    CHECK_EQ(with(la, zero), with(lb, zero));
    const auto small = [] { return ll::launchArgDigest(std::uint64_t{0x440}); };
    CHECK_EQ(with(la, small), with(lb, small));
    const auto kv0 = [](std::uint64_t kbase) {
        return [=] {
            KvArgs kv;
            kv.k = kbase + 0x100;
            kv.v = kbase + 0x200;
            kv.ks = 0;  // f16 pool: no scale arrays
            kv.vs = 0;
            return ll::launchArgDigest(kv);
        };
    };
    CHECK_EQ(with(la, kv0(0x300000000ull)), with(lb, kv0(0x380000000ull)));
    // a value of 0 still differs from a real pointer, and from another value
    CHECK(with(la, zero) != with(la, [] { return ll::launchArgDigest(std::uint64_t{0x300000040ull}, std::int32_t{3}); }));
    CHECK(with(la, small) != with(la, [] { return ll::launchArgDigest(std::uint64_t{0x448}); }));
    // adjacent allocations (A ends where B starts) vs a gap between them: a pointer inside A near its end, and one
    // inside B, give the same digest in both layouts
    const auto adj = {std::pair{0x10000000ull, 0x100000ull}, std::pair{0x10100000ull, 0x100000ull}};
    const auto gap = {std::pair{0x20000000ull, 0x100000ull}, std::pair{0x28000000ull, 0x100000ull}};
    const auto two = [](std::uint64_t a, std::uint64_t b) {
        return [=] { return ll::launchArgDigest(std::uint64_t{a + 0xFFF00}, std::uint64_t{b + 0x80}); };
    };
    CHECK_EQ(with(adj, two(0x10000000ull, 0x10100000ull)), with(gap, two(0x20000000ull, 0x28000000ull)));
}

// arch024-S0a (REVIEW-ARCH6-code C-2): the RAW dump keeps at most max_lines lines and counts the rest; the server
// turns it off unless WHIRL_LAUNCH_LOG_RAW=server.
void testLaunchRawCap() {
    namespace ll = whirl::hip::launchlog;
    const ll::RawStats before = ll::rawStats();
    ll::testRaw(true, 3);
    for (int i = 0; i < 5; ++i) ll::testRawPush("x");
    ll::RawStats s = ll::rawStats();
    CHECK(s.on);
    CHECK_EQ(s.kept, std::size_t{3});
    CHECK_EQ(s.dropped, std::uint64_t{2});
    CHECK_EQ(s.max_lines, std::size_t{3});
    // the default cap (no WHIRL_LAUNCH_LOG_RAW_MAX in the test environment) is finite
    CHECK(before.max_lines > 0 && before.max_lines <= 1000000);
    // server process: off unless WHIRL_LAUNCH_LOG_RAW=server (the unit tests run without the variable, or with
    // WHIRL_LAUNCH_LOG_RAW=1 from a gate shell: both are "not server")
    ll::testRaw(true, 3);
    ll::serverProcess();
    const char* rv = std::getenv("WHIRL_LAUNCH_LOG_RAW");
    const bool want_on = rv != nullptr && std::string(rv) == "server";
    CHECK_EQ(ll::rawStats().on, want_on);
    ll::testRaw(before.on, before.max_lines);
}

// arch025-S0a: the allocation log records sizes, kinds and ordinals (no addresses): two runs whose allocations land
// at different addresses give the same text; a size or order change gives a different seqhash; the list is capped;
// whirl-server keeps only the summary unless WHIRL_ALLOC_LOG_RAW=server; off records nothing.
void testAllocLog() {
    namespace al = whirl::hip::alloclog;
    const bool was_on = al::g_on;
    auto run = [](std::uint64_t base, bool swap) {
        al::testEnable(true, true);
        al::onDevAlloc(base + 0x1000, 1u << 20);
        al::onHostAlloc(base + 0x900000, 4096);
        al::onDevAlloc(base + 0x200000, swap ? 512 : 256);
        al::onDevFree(base + 0x1000);
        al::onHostFree(base + 0x900000);
        al::onDevFree(base + 0x777);  // never allocated
        return al::testText();
    };
    const std::string a = run(0x7f0000000000ull, false);
    const std::string b = run(0x100000000ull, false);  // other addresses, same sizes and order
    CHECK(a == b);
    CHECK(a.find("# events 6 seqhash ") == 0);
    CHECK(a.find("0 dev 1048576 #0\n") != std::string::npos);
    CHECK(a.find("1 host 4096 #0\n") != std::string::npos);
    CHECK(a.find("2 dev 256 #1\n") != std::string::npos);
    CHECK(a.find("3 free #0 1048576\n") != std::string::npos);
    CHECK(a.find("4 hostfree #0 4096\n") != std::string::npos);
    CHECK(a.find("5 free #? 0\n") != std::string::npos);
    CHECK(a.find("# unknown frees 1\n") != std::string::npos);
    CHECK(a.find("peak 1048832 bytes") != std::string::npos);
    const std::string c = run(0x7f0000000000ull, true);  // one size differs
    CHECK(c.substr(0, 40) != a.substr(0, 40));          // the seqhash line
    // summary only (the server default): same seqhash, no event lines
    al::testEnable(true, false);
    al::onDevAlloc(0x7f0000001000ull, 1u << 20);
    al::onHostAlloc(0x7f0000900000ull, 4096);
    al::onDevAlloc(0x7f0000200000ull, 256);
    al::onDevFree(0x7f0000001000ull);
    al::onHostFree(0x7f0000900000ull);
    al::onDevFree(0x7f0000000777ull);
    const std::string d = al::testText();
    CHECK(d.substr(0, d.find('\n')) == a.substr(0, a.find('\n')));
    CHECK(d.find(" dev 1048576 #0") == std::string::npos);
    CHECK(d.find("# list off\n") != std::string::npos);
    // cap
    al::testEnable(true, true, 2);
    for (int i = 0; i < 5; ++i) al::onDevAlloc(0x10000ull * (i + 1), 64);
    const std::string e = al::testText();
    CHECK(e.find("# events 5 ") == 0);
    CHECK(e.find("1 dev 64 #1\n") != std::string::npos);
    CHECK(e.find("2 dev 64 #2\n") == std::string::npos);
    CHECK(e.find("# lines dropped 3 (kept the first 2)\n") != std::string::npos);
    // server process: list off unless WHIRL_ALLOC_LOG_RAW=server
    al::testEnable(true, true);
    al::serverProcess();
    al::onDevAlloc(0x10000ull, 64);
    const char* rv = std::getenv("WHIRL_ALLOC_LOG_RAW");
    const bool want_list = rv != nullptr && std::string(rv) == "server";
    CHECK_EQ(al::testText().find("0 dev 64 #0\n") != std::string::npos, want_list);
    // off: nothing recorded
    al::testEnable(false, true);
    al::onDevAlloc(0x10000ull, 64);
    al::onDevFree(0x10000ull);
    CHECK(al::testText().find("# events 0 ") == 0);
    al::testEnable(was_on, true);
}

// bug:log-pid-names: launch_<pid>.txt / alloc_<pid>.txt were opened with "wb", so a later process with the same PID
// overwrote an earlier one's table (arch025 S0b-1 8060S B golden: launch_16468.txt created 21:46:51.112 by moe balance
// pair, rewritten 21:48:08.592 by moe fast single). Two writers with the same PID must leave two files; the names
// follow <prefix><pid>_<start>[_<n>].txt and no existing file is ever opened.
void testLogNamesUnique() {
    namespace fs = std::filesystem;
    namespace ln = whirl::hip::logname;
    const fs::path dir = whirl::test::testTmpBase() / "lognames";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const std::u8string d8 = dir.u8string();  // log_name.h takes UTF-8 folders (bug-envpaths, arch025-int4)
    const std::string d(d8.begin(), d8.end());
    auto count = [&](const char* prefix, bool raw) {
        int n = 0;
        for (const auto& e : fs::directory_iterator(dir)) {
            const std::string f = e.path().filename().string();
            const bool is_raw = f.size() > 8 && f.ends_with(".raw.txt");
            if (f.starts_with(prefix) && f.ends_with(".txt") && is_raw == raw) ++n;
        }
        return n;
    };
    const std::regex name_rx("^(launch|alloc)_[0-9]+_[0-9]{8}T[0-9]{9}(_[0-9]+)?\\.txt$");
    // the helper: the same PID twice (a reused PID, other start time; and the same start time) -> three files
    std::string s1, s2, s3;
    std::FILE* f1 = ln::openUnique(d, "launch_", 16468, "20261025T134651112", ".txt", &s1);
    CHECK(f1 != nullptr);
    if (f1) { std::fputs("first\n", f1); std::fclose(f1); }
    std::FILE* f2 = ln::openUnique(d, "launch_", 16468, "20261025T134808592", ".txt", &s2);
    CHECK(f2 != nullptr);
    if (f2) { std::fputs("second\n", f2); std::fclose(f2); }
    std::FILE* f3 = ln::openUnique(d, "launch_", 16468, "20261025T134651112", ".txt", &s3);
    CHECK(f3 != nullptr);
    if (f3) { std::fputs("third\n", f3); std::fclose(f3); }
    CHECK(s1 == "launch_16468_20261025T134651112");
    CHECK(s2 == "launch_16468_20261025T134808592");
    CHECK(s3 == "launch_16468_20261025T134651112_2");
    CHECK_EQ(count("launch_", false), 3);
    {
        std::FILE* r = _wfopen((dir / (s1 + ".txt")).c_str(), L"rb");  // the first file kept its content
        char b[16] = {};
        if (r) { std::fgets(b, sizeof b, r); std::fclose(r); }
        CHECK(std::string(b) == "first\n");
    }
    CHECK(ln::openExclusive(d, s1, ".txt") == nullptr);  // exclusive: an existing name is refused
    CHECK(ln::openUnique(d + "\\no_such_folder", "launch_", 1, "x", ".txt", nullptr) == nullptr);
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    // the real writers, twice in one process (same PID, same start time) -> two files each
    namespace ll = whirl::hip::launchlog;
    namespace al = whirl::hip::alloclog;
    const bool al_was_on = al::g_on;
    al::testEnable(true, true);
    al::onDevAlloc(0x10000ull, 64);
    al::testWrite(d);
    al::testWrite(d);
    al::testEnable(al_was_on, true);
    CHECK_EQ(count("alloc_", false), 2);
    const ll::RawStats before = ll::rawStats();
    ll::testRaw(true, 4);
    ll::testRawPush("0 k 1x1x1 1x1x1 0");
    ll::testWrite(d);
    ll::testWrite(d);
    ll::testRaw(before.on, before.max_lines);
    CHECK_EQ(count("launch_", false), 2);
    CHECK_EQ(count("launch_", true), 2);  // each table has its own RAW dump, same stem
    int named = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string f = e.path().filename().string();
        if (f.ends_with(".raw.txt")) {
            CHECK(fs::exists(dir / (f.substr(0, f.size() - 8) + ".txt")));
            continue;
        }
        CHECK(std::regex_match(f, name_rx));
        ++named;
    }
    CHECK_EQ(named, 4);
    fs::remove_all(dir, ec);
}

#ifdef _WIN32  // bug-envpaths helpers (the region of the old file)
// bug-envpaths (2026-10-26): the other environment variables whose value becomes a path had the bug-lappdata defect.
// This binary links whirl_hip, not the model library, so it covers class B; classes A (envGet / EnvSnapshot:
// WHIRL_KV_SSD_DIR, WHIRL_CODE_OBJECT) and C (WHIRL_DRAFT_VOCAB file through std::filesystem::path(std::string)) are
// testEnvPathsModelChild in tests/model/model_tests.cpp (whirl-model-tests links whirl_model).
//   B  read by std::getenv at static initialization, file opened with narrow std::fopen: WHIRL_ALLOC_LOG,
//      WHIRL_LAUNCH_LOG (alloc_<pid>_<start>.txt / launch_<pid>_<start>.txt written at exit; read through platform::env
//      since PL1b, getUtf8 + a wide exclusive open since this fix)
// Helper mode --child-envpaths: nothing to do; the atexit writers of the two logs run when it returns.
static int childEnvPaths() { return 0; }

struct EnvChild {
    bool started = false;
    DWORD exit_code = ~0u;
    DWORD pid = 0;
    std::string out;
};

// vars: name / value pairs, sorted by name by the caller's order (the block is passed as given)
static EnvChild runEnvChild(const std::wstring& self, std::vector<std::pair<std::wstring, std::wstring>> vars,
                            const std::filesystem::path& out_file) {
    EnvChild r;
    if (const wchar_t* v = _wgetenv(L"PATH")) vars.emplace_back(L"PATH", v);
    if (const wchar_t* v = _wgetenv(L"SystemRoot")) vars.emplace_back(L"SystemRoot", v);
    std::sort(vars.begin(), vars.end(), [](const auto& a, const auto& b) { return _wcsicmp(a.first.c_str(), b.first.c_str()) < 0; });
    std::wstring blk;
    for (const auto& [n, v] : vars) {
        blk += n;
        blk += L'=';
        blk += v;
        blk += L'\0';
    }
    blk += L'\0';
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE f = CreateFileW(out_file.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return r;
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nullptr;
    si.hStdOutput = f;
    si.hStdError = f;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + self + L"\" --child-envpaths";
    r.started = CreateProcessW(self.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                               blk.data(), nullptr, &si, &pi) != 0;
    CloseHandle(f);
    if (!r.started) return r;
    r.pid = pi.dwProcessId;
    if (WaitForSingleObject(pi.hProcess, 20000) != WAIT_OBJECT_0) TerminateProcess(pi.hProcess, 99);
    GetExitCodeProcess(pi.hProcess, &r.exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    std::ifstream in(out_file, std::ios::binary);
    r.out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return r;
}

static void testEnvPathsChild() {
    namespace fs = std::filesystem;
    wchar_t self[1024];
    const DWORD n = GetModuleFileNameW(nullptr, self, 1024);
    CHECK(n > 0 && n < 1024);
    if (n == 0 || n >= 1024) return;
    const fs::path root = whirl::test::testTmpBase() / L"envpaths";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const fs::path out_file = root / L"child_out.txt";
    // the bug-lappdata folder names (outside code page 950, and inside it) and an ASCII one (bytes must not change)
    for (const wchar_t* name : {L"u蔡é\U0001F600", L"測試", L"ascii dir"}) {
        const fs::path base = root / name;
        fs::create_directories(base / L"alog", ec);
        fs::create_directories(base / L"llog", ec);
        const EnvChild c = runEnvChild(self,
                                       {{L"WHIRL_ALLOC_LOG", (base / L"alog").wstring()}, {L"WHIRL_LAUNCH_LOG", (base / L"llog").wstring()}},
                                       out_file);
        CHECK(c.started && c.exit_code == 0);
        // B: the alloc / launch logs land in the folders the variables name, under the bug:log-pid-names name
        // <prefix><pid>_<start>[_<n>].txt (log_name.h; arch025-int4 merge of bug-envpaths with the lognames port)
        const std::wstring pid = std::to_wstring(c.pid);
        auto logIn = [&](const fs::path& d, const std::wstring& prefix) {
            const std::wregex rx(L"^" + prefix + pid + L"_[0-9]{8}T[0-9]{9}(_[0-9]+)?\\.txt$");
            for (const fs::directory_entry& e : fs::directory_iterator(d, ec))
                if (e.is_regular_file(ec) && std::regex_match(e.path().filename().wstring(), rx)) return true;
            return false;
        };
        const bool b_alloc = logIn(base / L"alog", L"alloc_");
        const bool b_launch = logIn(base / L"llog", L"launch_");
        CHECK(b_alloc);
        CHECK(b_launch);
        if (!b_alloc || !b_launch)
            std::printf("  envpaths B (static-init log folders) %s: alloc=%d launch=%d\n", whirl::narrow(name).c_str(), b_alloc ? 1 : 0, b_launch ? 1 : 0);
    }
    // exactly the three case folders and the child output exist under root (no mangled sibling)
    std::size_t entries = 0;
    for (const fs::directory_entry& e : fs::directory_iterator(root, ec)) {
        const std::wstring fn = e.path().filename().wstring();
        CHECK(fn == L"child_out.txt" || fn == L"u蔡é\U0001F600" || fn == L"測試" || fn == L"ascii dir");
        ++entries;
    }
    CHECK(entries == 4);
    // unset: the child runs and writes no log anywhere
    {
        const EnvChild c = runEnvChild(self, {}, out_file);
        CHECK(c.started && c.exit_code == 0);
        CHECK(c.out.empty());
    }
    fs::remove_all(root, ec);
}
#endif

// arch025 PL1b: the reads moved to platform::env::get give the v0.2.4 result for the same environment. Reference =
// the removed std::getenv code, copied here verbatim (uma_mem.h umaCheckOn, alloc_log.cpp / launch_log.cpp
// serverProcess); each variable is restored to what the test process had.
static void testPl1bEnvReads() {
#ifdef _WIN32
    namespace pe = whirl::platform::env;
    namespace al = whirl::hip::alloclog;
    namespace ll = whirl::hip::launchlog;
    const char* names[] = {"WHIRL_UMA_CHECK", "WHIRL_ALLOC_LOG_RAW", "WHIRL_LAUNCH_LOG_RAW"};
    std::optional<std::string> saved[3];
    for (int i = 0; i < 3; ++i) saved[i] = pe::get(names[i]);
    const bool al_was_on = al::g_on;
    const ll::RawStats ll_before = ll::rawStats();
    // nullptr = unset ("" deletes in-process, so "" itself cannot be set here: N1 HANDOFF)
    const char* vals[] = {nullptr, "0", "1", "00", "0 ", " 0", "server", "Server", "server ", "x=y", "SERVER"};
    int cells = 0;
    for (const char* v : vals) {
        for (const char* n : names) CHECK(pe::set(n, v ? v : "") == 0);
        {
            const char* g = std::getenv("WHIRL_UMA_CHECK");
            const bool ref = !(g && g[0] == '0' && g[1] == 0);
            CHECK_EQ(whirl::hip::umaCheckOn(), ref);
        }
        {
            const char* g = std::getenv("WHIRL_ALLOC_LOG_RAW");
            const bool ref = g != nullptr && std::strcmp(g, "server") == 0;
            al::testEnable(true, true);
            al::serverProcess();
            CHECK_EQ(al::testText().find("# list on\n") != std::string::npos, ref);
        }
        {
            const char* g = std::getenv("WHIRL_LAUNCH_LOG_RAW");
            const bool ref = g != nullptr && std::strcmp(g, "server") == 0;
            ll::testRaw(true, 3);
            ll::serverProcess();
            CHECK_EQ(ll::rawStats().on, ref);
        }
        ++cells;
    }
    CHECK_EQ(cells, 11);
    for (int i = 0; i < 3; ++i) CHECK(pe::set(names[i], saved[i] ? saved[i]->c_str() : "") == 0);
    al::testEnable(al_was_on, true);
    ll::testRaw(ll_before.on, ll_before.max_lines);
#endif
}

// bug-launchlogroot: an empty log folder at exit (WHIRL_LAUNCH_LOG / WHIRL_ALLOC_LOG unset or "", e.g. whirl-tests'
// clearWhirlEnv after the static initializer saw the variable) means "do not write": the exit path is empty instead
// of "\launch_<pid>.txt" (the root of the current drive). A folder gives the bug:log-pid-names name. Only path
// text is checked; nothing is opened.
static void testLogFolderEmpty() {
    namespace pe = whirl::platform::env;
    namespace al = whirl::hip::alloclog;
    namespace ll = whirl::hip::launchlog;
    CHECK(ll::exitPath("", 4242, false).empty());
    CHECK(ll::exitPath("", 4242, true).empty());
    CHECK(al::exitPath("", 4242).empty());
    // with a folder: the first name the writer tries, <pid>_<start> (bug:log-pid-names; arch025-int4 merge)
    const std::string st = whirl::hip::logname::processStart();
    CHECK_EQ(ll::exitPath("D:\\logs", 4242, false), "D:\\logs\\launch_4242_" + st + ".txt");
    CHECK_EQ(ll::exitPath("D:\\logs", 4242, true), "D:\\logs\\launch_4242_" + st + ".raw.txt");
    CHECK_EQ(al::exitPath("D:\\logs", 4242), "D:\\logs\\alloc_4242_" + st + ".txt");
    CHECK_EQ(ll::exitPath("rel dir", 7, false), "rel dir\\launch_7_" + st + ".txt");
    // and the name helpers refuse an empty folder themselves (nothing is created at the drive root)
    CHECK(whirl::hip::logname::openUnique("", "launch_", 4242, st, ".txt", nullptr) == nullptr);
    CHECK(whirl::hip::logname::openExclusive("", "launch_4242", ".raw.txt") == nullptr);
#ifdef _WIN32
    // through the platform::env seam, the read the exit writer makes ("" deletes the variable in-process)
    const char* names[] = {"WHIRL_LAUNCH_LOG", "WHIRL_ALLOC_LOG"};
    std::optional<std::string> saved[2];
    for (int i = 0; i < 2; ++i) saved[i] = pe::get(names[i]);
    for (const char* n : names) CHECK(pe::set(n, "") == 0);
    CHECK(ll::exitPathNow(4242, false).empty());
    CHECK(ll::exitPathNow(4242, true).empty());
    CHECK(al::exitPathNow(4242).empty());
    CHECK(pe::set("WHIRL_LAUNCH_LOG", "D:\\logs") == 0);
    CHECK(pe::set("WHIRL_ALLOC_LOG", "D:\\logs") == 0);
    CHECK_EQ(ll::exitPathNow(4242, false), "D:\\logs\\launch_4242_" + st + ".txt");
    CHECK_EQ(al::exitPathNow(4242), "D:\\logs\\alloc_4242_" + st + ".txt");
    for (int i = 0; i < 2; ++i) CHECK(pe::set(names[i], saved[i] ? saved[i]->c_str() : "") == 0);
#endif
}

}  // namespace

namespace {
// the old single main()'s helper modes of these tests (the test runs this program as the child)
static int childModes(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 2 && std::string(argv[1]) == "--child-envpaths") return childEnvPaths();  // bug-envpaths helper mode
#endif
    (void)argc;
    (void)argv;
    return -1;
}
}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_hip_runtime() {
    static const whirl::test::Suite s{"hip.runtime",
                                      {
                                          {"testLaunchDigest", [] { testLaunchDigest(); }},
                                          {"testLaunchDigestLayout", [] { testLaunchDigestLayout(); }},
                                          {"testLaunchDigestZeroAndAdjacent", [] { testLaunchDigestZeroAndAdjacent(); }},
                                          {"testLaunchRawCap", [] { testLaunchRawCap(); }},
                                          {"testAllocLog", [] { testAllocLog(); }},
                                          {"testLogNamesUnique", [] { testLogNamesUnique(); }},
#if defined(_WIN32)
                                          {"testEnvPathsChild", [] { testEnvPathsChild(); }},
#endif
                                          {"testPl1bEnvReads", [] { testPl1bEnvReads(); }},
                                          {"testLogFolderEmpty", [] { testLogFolderEmpty(); }},
                                      },
                                      childModes};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_hip_runtime()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
