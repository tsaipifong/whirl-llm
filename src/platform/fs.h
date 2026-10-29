// Platform layer: files (arch025 PL2). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only (a file
// handle is the bits of a Win32 HANDLE as void*); the Windows implementation is fs_win.cpp, chosen by
// cmake/platform.cmake. No virtual interface.
//
// One function per Win32 / CRT call (as llama.cpp's llama_file / llama_mmap pimpl does per OS): the callers keep
// their logic and so their call order. The CreateFileW arguments stay with the callers, as named constants whose
// values fs_win.cpp static_asserts against <windows.h>; every combination of v0.2.4 0e35407 (incl.
// FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED; 0e35407 uses no FILE_FLAG_WRITE_THROUGH) is pinned by the tier
// test testFsOpenTable through FakeFs (platform/fs_fake.h), which records every open.
// Return values are Win32's (true = the call returned nonzero); error codes are GetLastError values as plain ints.
// The callers:
//   - MappedFile, readFile, writeFile (src/common/util.cpp); the GGUF reader (src/model/loader.cpp); the server log
//     (src/server/log.cpp)
//   - the SSD tier's files (src/tier/kv_tier.cpp): directory lock, positional unbuffered writes, the overlapped QD-4
//     restore read, startup index, mtime stamps, sparse files, deletes, the pinned-memory declaration
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>

namespace whirl::platform::fs {

using Handle = void*;  // a Win32 HANDLE
bool valid(Handle h);  // h != INVALID_HANDLE_VALUE

// CreateFileW dwDesiredAccess
inline constexpr std::uint32_t kGenericRead = 0x80000000u;
inline constexpr std::uint32_t kGenericWrite = 0x40000000u;
inline constexpr std::uint32_t kWriteAttributes = 0x00000100u;  // FILE_WRITE_ATTRIBUTES
// dwShareMode
inline constexpr std::uint32_t kShareNone = 0;
inline constexpr std::uint32_t kShareRead = 0x1u;
inline constexpr std::uint32_t kShareWrite = 0x2u;
inline constexpr std::uint32_t kShareDelete = 0x4u;
// dwCreationDisposition
inline constexpr std::uint32_t kOpenExisting = 3;
inline constexpr std::uint32_t kOpenAlways = 4;
// dwFlagsAndAttributes
inline constexpr std::uint32_t kAttrNormal = 0x00000080u;      // FILE_ATTRIBUTE_NORMAL
inline constexpr std::uint32_t kNoBuffering = 0x20000000u;     // FILE_FLAG_NO_BUFFERING
inline constexpr std::uint32_t kOverlapped = 0x40000000u;      // FILE_FLAG_OVERLAPPED
inline constexpr std::uint32_t kWriteThrough = 0x80000000u;    // FILE_FLAG_WRITE_THROUGH (unused by 0e35407)
inline constexpr std::uint32_t kDeleteOnClose = 0x04000000u;   // FILE_FLAG_DELETE_ON_CLOSE

// GetLastError values the callers test for
inline constexpr int kNoError = 0;                 // NO_ERROR
inline constexpr int kErrFileNotFound = 2;         // ERROR_FILE_NOT_FOUND
inline constexpr int kErrPathNotFound = 3;         // ERROR_PATH_NOT_FOUND
inline constexpr int kErrSharingViolation = 32;    // ERROR_SHARING_VIOLATION
inline constexpr int kErrHandleEof = 38;           // ERROR_HANDLE_EOF
inline constexpr int kErrAlreadyExists = 183;      // ERROR_ALREADY_EXISTS
inline constexpr int kErrIoPending = 997;          // ERROR_IO_PENDING

// CreateFileW(path, access, share, nullptr, disposition, flags, nullptr). Recorded by FakeFs.
Handle open(const wchar_t* path, std::uint32_t access, std::uint32_t share, std::uint32_t disposition,
            std::uint32_t flags);
bool close(Handle h);  // CloseHandle
int lastError();       // GetLastError

// WriteFile(h, buf, n, &written, nullptr) (a synchronous handle at its file pointer)
bool write(Handle h, const void* buf, std::uint32_t n, std::uint32_t* written);
// WriteFile / ReadFile(h, buf, n, &count, &ov) with an OVERLAPPED holding only the offset (positional I/O on a
// synchronous handle). FakeFs: an injected write failure returns false without writing; an injected short read
// asks for fewer bytes (a sector multiple).
bool writeAt(Handle h, const void* buf, std::uint32_t n, std::uint64_t off, std::uint32_t* written);
bool readAt(Handle h, void* buf, std::uint32_t n, std::uint64_t off, std::uint32_t* read);
bool flush(Handle h);                     // FlushFileBuffers
bool remove(const wchar_t* path);         // DeleteFileW
bool setSparse(Handle h);                 // DeviceIoControl(FSCTL_SET_SPARSE, no buffers)
bool seek(Handle h, std::uint64_t off);   // SetFilePointerEx(off, FILE_BEGIN)
bool setEndOfFile(Handle h);              // SetEndOfFile
bool size(Handle h, std::int64_t* bytes);  // GetFileSizeEx
// GetFileTime / SetFileTime of the last-write time (FILETIME as 100 ns ticks since 1601)
bool lastWriteTime(Handle h, std::uint64_t* ft);
bool setLastWriteTime(Handle h, std::uint64_t ft);

// GetFileAttributesExW(GetFileExInfoStandard): size and last-write time (100 ns ticks since 1601)
struct Attributes {
    std::uint64_t size = 0;
    std::uint64_t last_write = 0;
};
bool attributes(const wchar_t* path, Attributes* out);
// GetCompressedFileSizeW: bytes on disk (sparse files: without the holes); false on failure (INVALID_FILE_SIZE with
// an error set), as 0e35407 tested it.
bool allocatedSize(const wchar_t* path, std::uint64_t* bytes);

// FindFirstFileW(pattern) / FindNextFileW / FindClose: fn(file name, size) for every match, in the order Windows
// returns them.
void forEachMatch(const std::wstring& pattern, const std::function<void(const wchar_t* name, std::uint64_t size)>& fn);

// Overlapped reads on a handle opened with kOverlapped. Overlapped is an OVERLAPPED (opaque storage here).
struct Overlapped {
    alignas(8) unsigned char raw[32];
};
// the event of an overlapped read: platform::proc::createEvent (manual reset, not signaled; one definition since
// arch025-N6 lane 1 leg 11, it was written in fs_win.cpp and proc_win.cpp)
bool resetEvent(Handle ev);    // ResetEvent
void prepare(Overlapped& ov, std::uint64_t off, Handle ev);  // zeroed, offset, hEvent
// ReadFile(h, buf, n, nullptr, &ov); false: see lastError() (kErrIoPending = in flight). FakeFs: short read.
bool readAsync(Handle h, void* buf, std::uint32_t n, Overlapped& ov);
bool overlappedResult(Handle h, Overlapped& ov, std::uint32_t* bytes, bool wait);  // GetOverlappedResult

// Read-only file mapping: CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr) (nullptr on failure),
// MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0), UnmapViewOfFile.
Handle mapReadOnly(Handle h);
const void* mapView(Handle mapping);
void unmapView(const void* p);

// _wfopen
std::FILE* openStdio(const wchar_t* path, const wchar_t* mode);

// UTF-8 -> UTF-16 with MB_ERR_INVALID_CHARS (MultiByteToWideChar); false on invalid UTF-8 (out untouched). "" -> "".
bool utf8ToWideStrict(std::string_view s, std::wstring* out);

// arch025-d7 (D7.1): the form of path a Win32 file call accepts whatever its length (Rust std's maybe_verbatim): a path
// shorter than 248 characters, one that already starts with \\?\ or \\.\, or one that is neither drive-absolute nor UNC
// after GetFullPathNameW comes back unchanged; a longer one is made absolute (GetFullPathNameW) and gets \\?\ (UNC:
// \\?\UNC\server\share\...). A process that is not long-path aware (no manifest, or LongPathsEnabled = 0, the Windows
// default) cannot open a path past MAX_PATH in the normal form.
std::wstring longPathForm(const std::wstring& path);

}  // namespace whirl::platform::fs
