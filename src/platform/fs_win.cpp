// Platform layer, files: the Windows implementation of platform/fs.h and FakeFs (fs_fake.h).
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL2: the file calls of src/common/util.cpp, src/model/loader.cpp, src/server/log.cpp and
// src/tier/kv_tier.cpp moved here one function per call, each with the arguments it had in v0.2.4 0e35407.

#include "platform/fs.h"
#include "platform/fs_fake.h"

#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <utility>

namespace pfs = whirl::platform::fs;
static_assert(GENERIC_READ == pfs::kGenericRead);
static_assert(GENERIC_WRITE == pfs::kGenericWrite);
static_assert(FILE_WRITE_ATTRIBUTES == pfs::kWriteAttributes);
static_assert(FILE_SHARE_READ == pfs::kShareRead);
static_assert(FILE_SHARE_WRITE == pfs::kShareWrite);
static_assert(FILE_SHARE_DELETE == pfs::kShareDelete);
static_assert(OPEN_EXISTING == pfs::kOpenExisting);
static_assert(OPEN_ALWAYS == pfs::kOpenAlways);
static_assert(FILE_ATTRIBUTE_NORMAL == pfs::kAttrNormal);
static_assert(FILE_FLAG_NO_BUFFERING == pfs::kNoBuffering);
static_assert(FILE_FLAG_OVERLAPPED == pfs::kOverlapped);
static_assert(FILE_FLAG_WRITE_THROUGH == pfs::kWriteThrough);
static_assert(FILE_FLAG_DELETE_ON_CLOSE == pfs::kDeleteOnClose);
static_assert(NO_ERROR == pfs::kNoError);
static_assert(ERROR_FILE_NOT_FOUND == pfs::kErrFileNotFound);
static_assert(ERROR_PATH_NOT_FOUND == pfs::kErrPathNotFound);
static_assert(ERROR_SHARING_VIOLATION == pfs::kErrSharingViolation);
static_assert(ERROR_HANDLE_EOF == pfs::kErrHandleEof);
static_assert(ERROR_ALREADY_EXISTS == pfs::kErrAlreadyExists);
static_assert(ERROR_IO_PENDING == pfs::kErrIoPending);
static_assert(sizeof(OVERLAPPED) <= sizeof(pfs::Overlapped::raw) && alignof(OVERLAPPED) <= 8);
static_assert(sizeof(HANDLE) == sizeof(pfs::Handle));

namespace whirl::platform::fs {

namespace {

HANDLE hh(Handle h) { return static_cast<HANDLE>(h); }
OVERLAPPED* ovp(Overlapped& o) { return reinterpret_cast<OVERLAPPED*>(o.raw); }

OVERLAPPED ovAt(std::uint64_t off) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    return ov;
}

// ---- FakeFs state (off unless a test called fake::begin)

std::atomic<bool> g_on{false};

struct FakeState {
    std::mutex mu;
    std::vector<fake::Open> opens;
    int fail_write = 0;
    int short_read = 0;
    std::uint32_t short_bytes = 0;
};
FakeState& fstate() {
    static FakeState s;
    return s;
}

// true: this write is injected to fail
bool takeWriteFault() {
    if (!g_on.load(std::memory_order_relaxed)) return false;
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    if (f.fail_write <= 0) return false;
    f.fail_write -= 1;
    return true;
}

// the byte count to ask for (n, or the injected short count)
std::uint32_t readCount(std::uint32_t n) {
    if (!g_on.load(std::memory_order_relaxed)) return n;
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    if (f.short_read <= 0) return n;
    f.short_read -= 1;
    return std::min(n, f.short_bytes);
}

}  // namespace

bool valid(Handle h) { return hh(h) != INVALID_HANDLE_VALUE; }

Handle open(const wchar_t* path, std::uint32_t access, std::uint32_t share, std::uint32_t disposition,
            std::uint32_t flags) {
    HANDLE h = CreateFileW(path, access, share, nullptr, disposition, flags, nullptr);
    if (g_on.load(std::memory_order_relaxed)) {
        const DWORD err = GetLastError();  // callers read it right after (ERROR_ALREADY_EXISTS, sharing violation)
        FakeState& f = fstate();
        {
            std::lock_guard<std::mutex> lk(f.mu);
            f.opens.push_back(fake::Open{path, access, share, disposition, flags, h != INVALID_HANDLE_VALUE});
        }
        SetLastError(err);
    }
    return h;
}

bool close(Handle h) { return CloseHandle(hh(h)) != 0; }

int lastError() { return static_cast<int>(GetLastError()); }

bool write(Handle h, const void* buf, std::uint32_t n, std::uint32_t* written) {
    DWORD w = 0;
    const BOOL ok = WriteFile(hh(h), buf, n, &w, nullptr);
    *written = w;
    return ok != 0;
}

bool writeAt(Handle h, const void* buf, std::uint32_t n, std::uint64_t off, std::uint32_t* written) {
    if (takeWriteFault()) {
        *written = 0;
        return false;
    }
    OVERLAPPED ov = ovAt(off);
    DWORD w = 0;
    const BOOL ok = WriteFile(hh(h), buf, n, &w, &ov);
    *written = w;
    return ok != 0;
}

bool readAt(Handle h, void* buf, std::uint32_t n, std::uint64_t off, std::uint32_t* read) {
    OVERLAPPED ov = ovAt(off);
    DWORD r = 0;
    const BOOL ok = ReadFile(hh(h), buf, readCount(n), &r, &ov);
    *read = r;
    return ok != 0;
}

bool flush(Handle h) { return FlushFileBuffers(hh(h)) != 0; }

bool remove(const wchar_t* path) { return DeleteFileW(path) != 0; }

bool setSparse(Handle h) {
    DWORD br = 0;
    return DeviceIoControl(hh(h), FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &br, nullptr) != 0;
}

bool seek(Handle h, std::uint64_t off) {
    LARGE_INTEGER dist;
    dist.QuadPart = static_cast<LONGLONG>(off);
    LARGE_INTEGER np;
    return SetFilePointerEx(hh(h), dist, &np, FILE_BEGIN) != 0;
}

bool setEndOfFile(Handle h) { return SetEndOfFile(hh(h)) != 0; }

bool size(Handle h, std::int64_t* bytes) {
    LARGE_INTEGER sz{};
    if (GetFileSizeEx(hh(h), &sz) == 0) return false;
    *bytes = sz.QuadPart;
    return true;
}

bool lastWriteTime(Handle h, std::uint64_t* ft) {
    FILETIME t;
    if (GetFileTime(hh(h), nullptr, nullptr, &t) == 0) return false;
    *ft = (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    return true;
}

bool setLastWriteTime(Handle h, std::uint64_t ft) {
    ULARGE_INTEGER u;
    u.QuadPart = ft;
    FILETIME t{u.LowPart, u.HighPart};
    return SetFileTime(hh(h), nullptr, nullptr, &t) != 0;
}

bool attributes(const wchar_t* path, Attributes* out) {
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (GetFileAttributesExW(path, GetFileExInfoStandard, &fa) == 0) return false;
    out->size = (static_cast<std::uint64_t>(fa.nFileSizeHigh) << 32) | fa.nFileSizeLow;
    out->last_write = (static_cast<std::uint64_t>(fa.ftLastWriteTime.dwHighDateTime) << 32) | fa.ftLastWriteTime.dwLowDateTime;
    return true;
}

std::wstring longPathForm(const std::wstring& path) {
    constexpr std::size_t kLegacyMax = 248;  // CreateDirectoryW's limit (MAX_PATH - 12); Rust std uses the same
    if (path.size() < kLegacyMax) return path;
    if (path.rfind(L"\\\\?\\", 0) == 0 || path.rfind(L"\\\\.\\", 0) == 0) return path;
    std::wstring full(path.size() + 1, L'\0');
    for (int tries = 0; tries < 4; ++tries) {
        const DWORD n = GetFullPathNameW(path.c_str(), static_cast<DWORD>(full.size()), full.data(), nullptr);
        if (n == 0) return path;
        if (n < full.size()) {
            full.resize(n);
            break;
        }
        full.assign(n, L'\0');  // n = the size needed, with the terminating null
    }
    if (full.empty() || full.back() == L'\0') return path;
    if (full.size() >= 3 && full[1] == L':' && full[2] == L'\\') return L"\\\\?\\" + full;
    if (full.rfind(L"\\\\", 0) == 0) return L"\\\\?\\UNC\\" + full.substr(2);
    return path;
}

bool allocatedSize(const wchar_t* path, std::uint64_t* bytes) {
    DWORD hi = 0;
    const DWORD lo = GetCompressedFileSizeW(path, &hi);
    if (lo != INVALID_FILE_SIZE || GetLastError() == NO_ERROR) {
        *bytes = (static_cast<std::uint64_t>(hi) << 32) | lo;
        return true;
    }
    return false;
}

void forEachMatch(const std::wstring& pattern, const std::function<void(const wchar_t* name, std::uint64_t size)>& fn) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    struct Closer {
        HANDLE h;
        ~Closer() { FindClose(h); }
    } closer{h};
    while (true) {
        fn(fd.cFileName, (static_cast<std::uint64_t>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow);
        if (FindNextFileW(h, &fd) == 0) break;
    }
}

Handle createEvent() { return CreateEventW(nullptr, TRUE, FALSE, nullptr); }

bool resetEvent(Handle ev) { return ResetEvent(hh(ev)) != 0; }

void prepare(Overlapped& ov, std::uint64_t off, Handle ev) {
    std::memset(ov.raw, 0, sizeof ov.raw);
    *ovp(ov) = ovAt(off);
    ovp(ov)->hEvent = hh(ev);
}

bool readAsync(Handle h, void* buf, std::uint32_t n, Overlapped& ov) {
    return ReadFile(hh(h), buf, readCount(n), nullptr, ovp(ov)) != 0;
}

bool overlappedResult(Handle h, Overlapped& ov, std::uint32_t* bytes, bool wait) {
    DWORD r = 0;
    const BOOL ok = GetOverlappedResult(hh(h), ovp(ov), &r, wait ? TRUE : FALSE);
    *bytes = r;
    return ok != 0;
}

Handle mapReadOnly(Handle h) { return CreateFileMappingW(hh(h), nullptr, PAGE_READONLY, 0, 0, nullptr); }

const void* mapView(Handle mapping) { return MapViewOfFile(hh(mapping), FILE_MAP_READ, 0, 0, 0); }

void unmapView(const void* p) { UnmapViewOfFile(p); }

std::FILE* openStdio(const wchar_t* path, const wchar_t* mode) { return _wfopen(path, mode); }

bool utf8ToWideStrict(std::string_view s, std::wstring* out) {
    if (s.empty()) {
        out->clear();
        return true;
    }
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return false;
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), w.data(), n);
    *out = std::move(w);
    return true;
}

// ---- FakeFs control

namespace fake {

void begin() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.opens.clear();
    f.fail_write = 0;
    f.short_read = 0;
    f.short_bytes = 0;
    g_on.store(true, std::memory_order_relaxed);
}

std::vector<Open> end() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    g_on.store(false, std::memory_order_relaxed);
    f.fail_write = 0;
    f.short_read = 0;
    std::vector<Open> out;
    out.swap(f.opens);
    return out;
}

std::vector<Open> snapshot() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    return f.opens;
}

void failWrite(int n) {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.fail_write = n;
}

void shortRead(int n, std::uint32_t bytes) {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.short_read = n;
    f.short_bytes = bytes;
}

int injectedLeft() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    return f.fail_write + f.short_read;
}

}  // namespace fake

}  // namespace whirl::platform::fs
