// SSD file layer for whirl-host-bench --suite ssd (arch025-T1, measure-tools DESIGN §A5). Test code only.
// SPDX-License-Identifier: Apache-2.0
//
// Mirrors the kv_tier file layer (src/tier/kv_tier.cpp, arch025-base 0ced8bf) call for call:
//   write (Tier::runJob JobKind::write, kv_tier.cpp:2402): CreateFileW(GENERIC_READ|GENERIC_WRITE, FILE_SHARE_READ,
//         OPEN_ALWAYS, FILE_FLAG_NO_BUFFERING) + FSCTL_SET_SPARSE; writeAt (kv_tier.cpp:100) = positional WriteFile with an
//         OVERLAPPED offset on a synchronous handle (QD1); FlushFileBuffers barrier (kv_tier.cpp:117).
//   read  (Tier::runJob JobKind::read, kv_tier.cpp:2317): CreateFileW(GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
//         OPEN_EXISTING, FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED); read_qd (kv_tier.h:77) overlapped ReadFile of
//         block_bytes (kv_tier.h:73) in flight, one manual-reset event each, completed in issue order.
// Variants beyond the product: buffered (flags 0), unbuffered_wt (NO_BUFFERING|WRITE_THROUGH, Microsoft's documented
// durable pattern; kv_tier does NOT use WRITE_THROUGH), write at QD4 (the IO thread writes QD1), read at QD1.
// PL2 (platform::fs::PosFile) will own these calls; then this layer should call it instead of mirroring it.
//
// Fs implementations: Win32Fs (real files) and FakeFs (in memory; counts every call, records open flags, tracks the
// in-flight depth, stores bytes) for the CPU selftest.

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// winioctl.h re-expands its DEFINE_GUID lines on every include (not fully guarded); a file that #undef-s far after
// including it (host_bench.cpp) must not see it again
#ifndef IOCTL_STORAGE_QUERY_PROPERTY
#include <winioctl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace whirl::bench::ssd {

enum class Mode { buffered, unbuffered, unbuffered_wt };

inline const char* modeName(Mode m) {
    return m == Mode::buffered ? "buffered" : m == Mode::unbuffered ? "unbuffered" : "unbuffered_wt";
}

// CreateFileW dwFlagsAndAttributes for (mode, qd). qd > 1 needs an overlapped handle; qd 1 uses a synchronous one
// with positional OVERLAPPED offsets (kv_tier writeAt / readAt).
constexpr DWORD openFlags(Mode m, int qd) {
    DWORD f = 0;
    if (m != Mode::buffered) f |= FILE_FLAG_NO_BUFFERING;
    if (m == Mode::unbuffered_wt) f |= FILE_FLAG_WRITE_THROUGH;
    if (qd > 1) f |= FILE_FLAG_OVERLAPPED;
    return f;
}

// the two product combinations (kv_tier.cpp:2403 write, :2318 read)
constexpr DWORD kKvTierWriteFlags = FILE_FLAG_NO_BUFFERING;
constexpr DWORD kKvTierReadFlags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;

struct Counters {
    std::int64_t opens = 0, closes = 0, writes = 0, reads = 0, bytes_w = 0, bytes_r = 0, flushes = 0, removes = 0;
    std::int64_t inflight = 0, max_inflight = 0, errors = 0;
    std::uint64_t off_sum = 0;  // sum of (offset / 4096 + 1) of every submitted I/O (exact order-free check)
    DWORD last_flags = 0;
};

class File {
public:
    virtual ~File() = default;
    // submit one I/O in slot (0..qd-1); slot must be free (waited). Synchronous files complete it here.
    virtual bool submit(int slot, bool write, void* buf, std::uint32_t len, std::uint64_t off) = 0;
    // wait for the I/O in slot; *got = bytes transferred
    virtual bool wait(int slot, std::uint32_t* got) = 0;
    virtual bool flush() = 0;
};

class Fs {
public:
    virtual ~Fs() = default;
    // write = kv_tier's write open (OPEN_ALWAYS, sparse), else its read open (OPEN_EXISTING)
    virtual std::unique_ptr<File> open(const std::wstring& path, bool write, Mode m, int qd) = 0;
    virtual bool remove(const std::wstring& path) = 0;
    virtual bool exists(const std::wstring& path) = 0;
    virtual const char* name() const = 0;
    Counters c;
    std::mutex mu;
    void onSubmit(bool write, std::uint32_t len, std::uint64_t off) {
        std::lock_guard<std::mutex> lk(mu);
        (write ? c.writes : c.reads) += 1;
        (write ? c.bytes_w : c.bytes_r) += len;
        c.off_sum += off / 4096 + 1;
        c.inflight += 1;
        c.max_inflight = std::max(c.max_inflight, c.inflight);
    }
    void onDone() {
        std::lock_guard<std::mutex> lk(mu);
        c.inflight -= 1;
    }
    void onError() {
        std::lock_guard<std::mutex> lk(mu);
        c.errors += 1;
    }
    void resetMax() {
        std::lock_guard<std::mutex> lk(mu);
        c.max_inflight = c.inflight;
    }
};

// ------------------------------------------------------------------ Win32 (real files)
class Win32File final : public File {
public:
    Win32File(Fs& fs, HANDLE h, int qd) : fs_(fs), h_(h), qd_(qd), ov_(static_cast<std::size_t>(qd)), got_(static_cast<std::size_t>(qd)),
                                          res_(static_cast<std::size_t>(qd)), ev_(static_cast<std::size_t>(qd), nullptr) {
        if (qd_ > 1)
            for (auto& e : ev_) e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    ~Win32File() override {
        for (HANDLE e : ev_)
            if (e != nullptr) CloseHandle(e);
        CloseHandle(h_);
        std::lock_guard<std::mutex> lk(fs_.mu);
        fs_.c.closes += 1;
    }
    bool ok() const {
        for (HANDLE e : ev_)
            if (qd_ > 1 && e == nullptr) return false;
        return true;
    }
    bool submit(int slot, bool write, void* buf, std::uint32_t len, std::uint64_t off) override {
        fs_.onSubmit(write, len, off);
        OVERLAPPED& ov = ov_[static_cast<std::size_t>(slot)];
        ov = OVERLAPPED{};
        ov.Offset = static_cast<DWORD>(off);
        ov.OffsetHigh = static_cast<DWORD>(off >> 32);
        if (qd_ <= 1) {
            // synchronous handle: kv_tier writeAt / readAt (the call returns when the I/O is done)
            DWORD n = 0;
            const BOOL r = write ? WriteFile(h_, buf, len, &n, &ov) : ReadFile(h_, buf, len, &n, &ov);
            got_[static_cast<std::size_t>(slot)] = n;
            res_[static_cast<std::size_t>(slot)] = r != 0 || GetLastError() == ERROR_HANDLE_EOF;
            return true;
        }
        ov.hEvent = ev_[static_cast<std::size_t>(slot)];
        ResetEvent(ov.hEvent);
        const BOOL r = write ? WriteFile(h_, buf, len, nullptr, &ov) : ReadFile(h_, buf, len, nullptr, &ov);
        res_[static_cast<std::size_t>(slot)] = true;
        if (r == 0) {
            const DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) {
                fs_.onDone();
                fs_.onError();
                res_[static_cast<std::size_t>(slot)] = false;
                return false;
            }
        }
        return true;
    }
    bool wait(int slot, std::uint32_t* got) override {
        const std::size_t k = static_cast<std::size_t>(slot);
        bool ok = res_[k];
        if (qd_ <= 1) {
            *got = got_[k];
        } else {
            DWORD n = 0;
            if (GetOverlappedResult(h_, &ov_[k], &n, TRUE) == 0 && GetLastError() != ERROR_HANDLE_EOF) ok = false;
            *got = n;
        }
        fs_.onDone();
        if (!ok) fs_.onError();
        return ok;
    }
    bool flush() override {
        {
            std::lock_guard<std::mutex> lk(fs_.mu);
            fs_.c.flushes += 1;
        }
        return FlushFileBuffers(h_) != 0;
    }

private:
    Fs& fs_;
    HANDLE h_;
    int qd_;
    std::vector<OVERLAPPED> ov_;
    std::vector<DWORD> got_;
    std::vector<bool> res_;
    std::vector<HANDLE> ev_;
};

class Win32Fs final : public Fs {
public:
    std::unique_ptr<File> open(const std::wstring& path, bool write, Mode m, int qd) override {
        const DWORD fl = openFlags(m, qd);
        HANDLE h = write ? CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, fl, nullptr)
                         : CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, fl, nullptr);
        {
            std::lock_guard<std::mutex> lk(mu);
            c.last_flags = fl;
            if (h == INVALID_HANDLE_VALUE) {
                c.errors += 1;
                return nullptr;
            }
            c.opens += 1;
        }
        if (write) {
            DWORD br = 0;  // kv_tier.cpp:2414 (sparse: dedup holes)
            DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &br, nullptr);
        }
        auto f = std::make_unique<Win32File>(*this, h, qd);
        if (!f->ok()) return nullptr;
        return f;
    }
    bool remove(const std::wstring& path) override {
        {
            std::lock_guard<std::mutex> lk(mu);
            c.removes += 1;
        }
        if (DeleteFileW(path.c_str()) != 0) return true;
        const DWORD e = GetLastError();
        return e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND;
    }
    bool exists(const std::wstring& path) override { return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES; }
    const char* name() const override { return "win32"; }
};

// ------------------------------------------------------------------ FakeFs (in memory, CPU selftest)
class FakeFs;
class FakeFile final : public File {
public:
    FakeFile(FakeFs& fs, std::vector<std::uint8_t>& data, int qd, bool write, DWORD flags)
        : fs_(fs), data_(data), qd_(qd), write_(write), flags_(flags), pend_(static_cast<std::size_t>(qd)) {}
    ~FakeFile() override;
    bool submit(int slot, bool write, void* buf, std::uint32_t len, std::uint64_t off) override;
    bool wait(int slot, std::uint32_t* got) override;
    bool flush() override;

private:
    struct Pend {
        bool busy = false, write = false;
        void* buf = nullptr;
        std::uint32_t len = 0;
        std::uint64_t off = 0;
    };
    FakeFs& fs_;
    std::vector<std::uint8_t>& data_;
    int qd_;
    bool write_;
    DWORD flags_;
    std::vector<Pend> pend_;
};

class FakeFs final : public Fs {
public:
    std::map<std::wstring, std::vector<std::uint8_t>> files;
    int contract_violations = 0;  // misaligned unbuffered I/O, write on a read handle, busy slot, ...
    std::unique_ptr<File> open(const std::wstring& path, bool write, Mode m, int qd) override {
        const DWORD fl = openFlags(m, qd);
        std::lock_guard<std::mutex> lk(mu);
        c.last_flags = fl;
        auto it = files.find(path);
        if (it == files.end()) {
            if (!write) {  // OPEN_EXISTING
                c.errors += 1;
                return nullptr;
            }
            it = files.emplace(path, std::vector<std::uint8_t>{}).first;  // OPEN_ALWAYS
        }
        c.opens += 1;
        return std::make_unique<FakeFile>(*this, it->second, qd, write, fl);
    }
    bool remove(const std::wstring& path) override {
        std::lock_guard<std::mutex> lk(mu);
        c.removes += 1;
        files.erase(path);
        return true;
    }
    bool exists(const std::wstring& path) override {
        std::lock_guard<std::mutex> lk(mu);
        return files.count(path) != 0;
    }
    const char* name() const override { return "fake"; }
};

inline FakeFile::~FakeFile() {
    std::lock_guard<std::mutex> lk(fs_.mu);
    fs_.c.closes += 1;
    for (const Pend& p : pend_)
        if (p.busy) fs_.contract_violations += 1;  // closed with I/O in flight
}

inline bool FakeFile::submit(int slot, bool write, void* buf, std::uint32_t len, std::uint64_t off) {
    Pend& p = pend_[static_cast<std::size_t>(slot)];
    {
        std::lock_guard<std::mutex> lk(fs_.mu);
        const bool unbuf = (flags_ & FILE_FLAG_NO_BUFFERING) != 0;
        if (p.busy || slot >= qd_ || (write && !write_) ||
            (unbuf && ((off % 4096) != 0 || (len % 4096) != 0 || (reinterpret_cast<std::uintptr_t>(buf) % 4096) != 0)))
            fs_.contract_violations += 1;
    }
    fs_.onSubmit(write, len, off);
    p = Pend{true, write, buf, len, off};
    return true;
}

inline bool FakeFile::wait(int slot, std::uint32_t* got) {
    Pend& p = pend_[static_cast<std::size_t>(slot)];
    if (!p.busy) {
        std::lock_guard<std::mutex> lk(fs_.mu);
        fs_.contract_violations += 1;
        *got = 0;
        return false;
    }
    if (p.write) {
        if (data_.size() < p.off + p.len) data_.resize(static_cast<std::size_t>(p.off + p.len));
        std::memcpy(data_.data() + p.off, p.buf, p.len);
        *got = p.len;
    } else {
        const std::uint64_t n = p.off >= data_.size() ? 0 : std::min<std::uint64_t>(p.len, data_.size() - p.off);
        if (n) std::memcpy(p.buf, data_.data() + p.off, static_cast<std::size_t>(n));
        *got = static_cast<std::uint32_t>(n);
    }
    p.busy = false;
    fs_.onDone();
    return true;
}

inline bool FakeFile::flush() {
    std::lock_guard<std::mutex> lk(fs_.mu);
    fs_.c.flushes += 1;
    for (const Pend& p : pend_)
        if (p.busy) fs_.contract_violations += 1;  // a barrier with I/O in flight orders nothing
    return true;
}

// ------------------------------------------------------------------ one pass (the timed op)
// blocks I/Os of len bytes at offsets 0, len, 2 len, ...; slot i % qd uses bufs[i % qd]; at most qd in flight,
// completed in issue order (kv_tier read loop). write: one flush at the end (kv_tier's per-entry barrier).
inline bool runPass(File& f, bool write, const std::vector<void*>& bufs, std::uint64_t blocks, std::uint32_t len, int qd,
                    double* flush_ns) {
    std::uint64_t issued = 0, done = 0;
    bool ok = true;
    while (done < blocks) {
        while (ok && issued < blocks && issued - done < static_cast<std::uint64_t>(qd)) {
            const int k = static_cast<int>(issued % static_cast<std::uint64_t>(qd));
            if (!f.submit(k, write, bufs[static_cast<std::size_t>(k)], len, issued * len)) {
                ok = false;
                break;
            }
            issued += 1;
        }
        if (done >= issued) break;
        std::uint32_t got = 0;
        if (!f.wait(static_cast<int>(done % static_cast<std::uint64_t>(qd)), &got) || got != len) ok = false;
        done += 1;
        if (!ok) {  // let the rest land before the buffers can go
            for (; done < issued; ++done) f.wait(static_cast<int>(done % static_cast<std::uint64_t>(qd)), &got);
            break;
        }
    }
    if (ok && write) {
        LARGE_INTEGER t0, t1, fq;
        QueryPerformanceCounter(&t0);
        ok = f.flush();
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&fq);
        if (flush_ns) *flush_ns = static_cast<double>(t1.QuadPart - t0.QuadPart) * 1e9 / static_cast<double>(fq.QuadPart);
    }
    return ok;
}

// ------------------------------------------------------------------ bench directory rules
// Real I/O only below an absolute directory, never a drive root, never D: / E: (USB4 backup drives that share the
// R9700's link). Returns "" when allowed, else the reason.
inline std::string dirRefusal(const std::wstring& dir) {
    if (dir.size() < 3 || dir[1] != L':' || (dir[2] != L'\\' && dir[2] != L'/')) return "not an absolute drive path (X:\\...)";
    const wchar_t d = static_cast<wchar_t>(towupper(dir[0]));
    if (d == L'D' || d == L'E') return "drives D: and E: are backup-only (USB4, shared with the R9700)";
    std::wstring rest = dir.substr(3);
    while (!rest.empty() && (rest.back() == L'\\' || rest.back() == L'/')) rest.pop_back();
    if (rest.empty()) return "a drive root is refused; use a scratch directory";
    if (rest.find(L"..") != std::wstring::npos) return "\"..\" is refused";
    return "";
}

// The physical drive behind a volume (product id), "" when it cannot be read (no admin rights needed).
inline std::string driveModel(wchar_t letter) {
    wchar_t vp[] = L"\\\\.\\X:";
    vp[4] = letter;
    HANDLE h = CreateFileW(vp, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return "";
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceProperty;
    q.QueryType = PropertyStandardQuery;
    std::vector<std::uint8_t> buf(4096);
    DWORD br = 0;
    std::string out;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof q, buf.data(), static_cast<DWORD>(buf.size()), &br, nullptr) != 0 &&
        br >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        const auto* d = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR*>(buf.data());
        if (d->ProductIdOffset != 0 && d->ProductIdOffset < br) out = reinterpret_cast<const char*>(buf.data() + d->ProductIdOffset);
        while (!out.empty() && out.back() == ' ') out.pop_back();
        while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    }
    CloseHandle(h);
    for (char& ch : out)
        if (ch == '"' || ch == ';' || ch == '\\' || static_cast<unsigned char>(ch) < 0x20) ch = '_';
    return out;
}

}  // namespace whirl::bench::ssd
