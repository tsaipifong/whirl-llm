// SPDX-License-Identifier: Apache-2.0

#include "support.h"

#include "whirl/common.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace whirl::parity {

namespace fs = std::filesystem;

namespace {

// Quotes one argument following the MSVC CRT command-line parsing rules.
std::wstring quoteArg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    for (std::size_t i = 0;; ++i) {
        std::size_t backslashes = 0;
        while (i < a.size() && a[i] == L'\\') {
            ++i;
            ++backslashes;
        }
        if (i == a.size()) {
            q.append(backslashes * 2, L'\\');
            break;
        }
        if (a[i] == L'"') {
            q.append(backslashes * 2 + 1, L'\\');
            q.push_back(L'"');
        } else {
            q.append(backslashes, L'\\');
            q.push_back(a[i]);
        }
    }
    q.push_back(L'"');
    return q;
}

#ifdef _WIN32
void drain(HANDLE h, std::string* out) {
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(h, buf, sizeof buf, &n, nullptr) && n > 0) out->append(buf, n);
}
#endif

}  // namespace

ProcResult run(const std::vector<std::string>& argv, bool idle) {
    ProcResult r;
#ifdef _WIN32
    std::wstring cmd;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i) cmd.push_back(L' ');
        cmd += quoteArg(widen(argv[i]));
    }
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE out_r, out_w, err_r, err_w;
    if (!CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) return r;
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out_w;
    si.hStdError = err_w;
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    const DWORD flags = CREATE_NO_WINDOW | (idle ? IDLE_PRIORITY_CLASS : 0);
    const BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, flags, nullptr, nullptr, &si, &pi);
    CloseHandle(out_w);
    CloseHandle(err_w);
    if (!ok) {
        CloseHandle(out_r);
        CloseHandle(err_r);
        r.err = "cannot start " + (argv.empty() ? std::string() : argv[0]);
        return r;
    }
    std::thread t(drain, err_r, &r.err);
    drain(out_r, &r.out);
    t.join();
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    r.exit_code = static_cast<int>(code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(out_r);
    CloseHandle(err_r);
#else
    (void)argv;
    (void)idle;
    r.err = "process spawning is implemented for Windows only";
#endif
    return r;
}

void lowerOwnPriority() {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
#endif
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)

namespace {

constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void block(std::uint32_t h[8], const unsigned char* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
               (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
        const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = S0 + mj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

}  // namespace

std::string sha256Hex(std::string_view data) {
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    std::size_t n = data.size();
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) block(h, p + i);
    unsigned char tail[128] = {};
    const std::size_t rem = n - i;
    std::memcpy(tail, p + i, rem);
    tail[rem] = 0x80;
    const std::size_t tl = rem + 9 <= 64 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(n) * 8;
    for (int k = 0; k < 8; ++k) tail[tl - 1 - k] = static_cast<unsigned char>(bits >> (8 * k));
    block(h, tail);
    if (tl == 128) block(h, tail + 64);
    static const char* hex = "0123456789abcdef";
    std::string s;
    for (std::uint32_t v : h)
        for (int k = 28; k >= 0; k -= 4) s.push_back(hex[(v >> k) & 15]);
    return s;
}

// ---------------------------------------------------------------------------

std::vector<std::string> splitLines(std::string_view s) {
    std::vector<std::string> lines;
    std::size_t p = 0;
    while (p < s.size()) {
        std::size_t e = s.find('\n', p);
        if (e == std::string_view::npos) e = s.size();
        std::string_view l = s.substr(p, e - p);
        if (!l.empty() && l.back() == '\r') l.remove_suffix(1);
        lines.emplace_back(l);
        p = e + 1;
    }
    return lines;
}

std::string basename(std::string_view path) {
    const std::size_t s = path.find_last_of("/\\");
    return std::string(s == std::string_view::npos ? path : path.substr(s + 1));
}

std::string joinPath(std::string_view dir, std::string_view name) {
    std::string r(dir);
    if (!r.empty() && r.back() != '/' && r.back() != '\\') r.push_back('\\');
    r += name;
    return r;
}

namespace {
fs::path u8path(const std::string& s) { return fs::path(widen(s)); }
}  // namespace

void makeDirs(const std::string& dir) { fs::create_directories(u8path(dir)); }
bool isDirectory(const std::string& path) { return fs::is_directory(u8path(path)); }
bool fileExists(const std::string& path) { return fs::exists(u8path(path)); }

std::vector<std::string> listFiles(const std::string& dir) {
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(u8path(dir)))
        if (e.is_regular_file()) names.push_back(narrow(e.path().filename().wstring()));
    std::sort(names.begin(), names.end());
    std::vector<std::string> out;
    for (const auto& n : names) out.push_back(joinPath(dir, n));
    return out;
}

}  // namespace whirl::parity
