// Platform layer: files (arch025 PL2). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is fs_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

namespace whirl::platform::fs {

// PL2 moves here (CreateFileW access / share / disposition / flags kept bit for bit, pinned by a table test against
// 0e35407, incl. FILE_FLAG_NO_BUFFERING / FILE_FLAG_WRITE_THROUGH / FILE_FLAG_OVERLAPPED):
//   - MappedFile: CreateFileW, GetFileSizeEx, CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, CloseHandle;
//     _wfopen and the UTF-8 <-> UTF-16 path conversion (src/common/util.cpp)
//   - the SSD tier files: CreateFileW, ReadFile / WriteFile (OVERLAPPED, unbuffered), GetOverlappedResult,
//     SetFilePointerEx, SetEndOfFile, FlushFileBuffers, DeleteFileW, FindFirstFileW / FindNextFileW / FindClose,
//     GetFileAttributesExW, GetCompressedFileSizeW, GetFileTime / SetFileTime, DeviceIoControl, the OVERLAPPED
//     events (CreateEventW / ResetEvent), GetLastError, _wfopen, the UTF-8 -> UTF-16 path conversion
//     (src/tier/kv_tier.cpp)
//   - _wfopen of the GGUF reader (src/model/loader.cpp) and of the server log (src/server/log.cpp)
// The only place with a fake (FakeFs: injected write failure and short read for tier-tests).

}  // namespace whirl::platform::fs
