# The platform layer (`src/platform`)

This page is for people who add or move code that calls the operating system: files, sockets, the environment,
the console, clocks, the process itself. WHIRL runs on Windows only; the platform layer is where every Win32,
Winsock and CRT-platform call lives, so the rest of the code sees plain C++ and the tests can inject IO and socket
errors. Decision record: ADR-0004 Decision 5 (link-time per-concern headers, no virtual interface).

State (0.2.5): PL0 made the skeleton (headers, CMake target, check). **PL2 (fs) has moved its calls**
(`fs.h` / `fs_win.cpp` / `fs_fake.h`, section below); PL1, PL3, PL4 move the rest (table below), PL5 makes the check
blocking in `release_gate.ps1`.

## Shape

| Header | Namespace | Concern | Step that fills it |
|---|---|---|---|
| `src/platform/env.h` | `whirl::platform::env` | process environment (the fingerprint's full block read, single reads, the one write) | PL1 |
| `src/platform/paths.h` | `whirl::platform::paths` | well-known folders (tune cache under LOCALAPPDATA, default cache / log folders) | PL1 |
| `src/platform/console.h` | `whirl::platform::console` | UTF-8 code page and its restore, console mode, binary stdout | PL1 |
| `src/platform/time.h` | `whirl::platform::time` | 1 ms timer resolution, local wall-clock time of log lines | PL1 |
| `src/platform/fs.h` | `whirl::platform::fs` | mapped GGUF files, the SSD tier's unbuffered / overlapped files, `_wfopen` | PL2 (with FakeFs) |
| `src/platform/net.h` | `whirl::platform::net` | the HTTP server's Winsock calls, the release front-end's port pre-check | PL3 (with FakeNet) |
| `src/platform/proc.h` | `whirl::platform::proc` | executable path, the per-GPU named mutex, Ctrl+C / shutdown events, process id / working set | PL4 |
| `src/platform/sys.h` | `whirl::platform::sys` | physical RAM; the wmain UTF-16 -> UTF-8 conversion ("args") | PL4 |

Each header lists, in a comment, the exact calls and files its step moves. The full per-file list with counts is
`tools/gates/platform_pending.txt`.

Rules:
- **Headers are platform-free.** No `<windows.h>`, Winsock or any other platform header in `src/platform/*.h`, and
  no Win32 types (`HANDLE`, `SOCKET`, `DWORD`) in their declarations: handles become opaque WHIRL types, error codes
  plain integers. Checked twice: `src/platform/headers_win.cpp` includes every header first and stops the build if
  `<windows.h>` / Winsock got pulled in, and the platform check counts a platform include in a `src/platform/*.h` as
  a violation.
- **One Windows implementation file per concern**, `src/platform/<concern>_win.cpp`, listed in
  `cmake/platform.cmake`. It includes `<windows.h>` itself. A future POSIX port adds `<concern>_posix.cpp` next to
  it and chooses in CMake; nothing outside `src/platform` changes.
- **No virtual interface, no runtime registry.** The implementation is chosen at link time. The only fakes are for
  fs (PL2: write failure, short read for the tier tests) and net (PL3: accept / recv errors); how they are injected
  is decided in those steps.
- **Moving a call keeps it bit for bit**: same flags (`CreateFileW` access / share / disposition / flags, socket
  options), same call order, same strings (the SSD fingerprint, `exeIdentity`, mutex names, paths). Each step pins
  them with a table test against v0.2.4 0e35407 (arch025.PLAN.md risk R7).
- **Standard C++ stays standard.** `std::thread`, `std::mutex`, `std::chrono`, `std::filesystem` are not platform
  code and do not move.
- **Include path.** Callers write `#include "platform/fs.h"` (the target puts `src/` on the include path). Never put
  `src/platform` itself on an include path: `time.h` there would shadow the C library's `<time.h>`. Inside
  `whirl::platform`, `time` is a namespace: platform code calls `std::time`, not bare `time()`.

Not part of this layer (whitelisted by the check): `src/hip/` (the HIP runtime's own `LoadLibrary`, DXGI and PDH
code, ADR-0004 #5), `src/release/dll_deps.cpp` (the DLL dependency diagnosis), and the vendored
`src/vision/stb_image.h`.

## fs (PL2)

`platform/fs.h` has one function per Win32 / CRT file call (`open` = CreateFileW, `writeAt` / `readAt` = WriteFile /
ReadFile with an offset-only OVERLAPPED, `readAsync` / `overlappedResult` / `createEvent` / `prepare` for the SSD
restore's QD-4 overlapped reads, `flush`, `remove`, `setSparse`, `seek` / `setEndOfFile`, file times, attributes,
`forEachMatch` = FindFirstFileW / FindNextFileW / FindClose, the read-only mapping of `MappedFile`, `openStdio` =
_wfopen, `utf8ToWideStrict`). A file handle is `fs::Handle` (the bits of a HANDLE as `void*`). The CreateFileW
arguments stay at the call sites as named constants (`kGenericRead`, `kShareRead`, `kOpenAlways`, `kNoBuffering`,
`kOverlapped`, `kDeleteOnClose`, ...) whose values `fs_win.cpp` static_asserts against `<windows.h>`.

Callers: `src/common/util.cpp` (MappedFile, readFile / writeFile), `src/model/loader.cpp` (GGUF reader),
`src/server/log.cpp` (log file), `src/tier/kv_tier.cpp` (the SSD tier). Left for PL4: util.cpp's widen / narrow
(args) and kv_tier.cpp's GetCurrentProcessId / GetModuleFileNameW (proc).

**FakeFs** (`platform/fs_fake.h`, implemented inside `fs_win.cpp`, the PL3 FakeNet pattern): off unless a test calls
`fake::begin()`; then it records every `fs::open` (path, access, share, disposition, flags, ok) and injects
`failWrite(n)` (the next n `writeAt` return false and write nothing) and `shortRead(n, bytes)` (the next n `readAt` /
`readAsync` ask Windows for at most `bytes`, a sector multiple so unbuffered handles stay legal). Off, each of those
calls costs one relaxed atomic load more.

The flag pin (arch025 risk R7 / A5: a lost `FILE_FLAG_NO_BUFFERING` halves SSD write speed and changes no output):
`whirl-tier-tests` `testFsOpenTable` runs every SSD-tier open (directory lock, spill write, restart index, mtime stamp,
restore read) and compares each recorded open with the v0.2.4 0e35407 table written in literal Win32 values (no
`FILE_FLAG_WRITE_THROUGH` anywhere in 0e35407); `whirl-tests` `testFsMappedOpen` does the same for `MappedFile`.
`testFsWriteFail`, `testFsShortRead` and `testFsVerifyNotBlind` cover the injected faults and the restore check.

The SSD speed tool uses the same layer: `tests/bench/ssd_io.h` (`whirl-host-bench --suite ssd`, arch025 T1)
opens, writes, reads, flushes and deletes its files through `platform::fs` with the CreateFileW integers its pre-PL2
version passed to Win32 directly; `--selftest --suite ssd --ssd-dir DIR` prints the "ssd open table" (12 opens,
recorded by FakeFs, compared with literal values) and fails on any difference. Only its non-product QD4 *write*
variant calls WriteFile on the fs handle itself (kv_tier never writes overlapped, so `fs.h` has no such call).

## CMake

`cmake/platform.cmake` defines `whirl_platform` (STATIC). It is the bottom layer: it links no other WHIRL target.
The core library links it (`target_link_libraries(whirl PUBLIC whirl_platform)` in `CMakeLists.txt`), so every
executable can reach it and PL1-PL4 only add sources to `cmake/platform.cmake`. `whirl_tier_core` (kv_tier.cpp, built without
`whirl`) links `whirl_platform` itself (`cmake/tier.cmake`, PL2). Winsock (`ws2_32`) and `winmm`
move to `whirl_platform` with PL3 / PL1. The library holds `headers_win.cpp` (referenced by nothing, dropped by the
linker) and, since PL2, `fs_win.cpp`.

## The check

`tools/gates/platform_check.ps1` (rule `platform` of `tools/gates/pending_scan.py`) scans `src/` and `include/`
(comments and string literals are not code) for:
1. a platform header include (`<windows.h>`, `<winsock2.h>`, `<ws2tcpip.h>`, `<io.h>`, `<psapi.h>`, `<timeapi.h>`,
   `<winioctl.h>`, ...);
2. a Win32 / Winsock / CRT-platform call: any CamelCase name with the Unicode suffix `W` / `ExW` / `ExA`
   (`CreateFileW`, `GetFileAttributesExW`), a fixed list of other names (`CloseHandle`, `ReadFile`, `GetLastError`,
   `WSAStartup`, `setsockopt`, `closesocket`, `_wfopen`, `_setmode`, ...: `PLATFORM_NAMES`), and the BSD socket
   names only when written with the global qualifier (`::recv`, `::send`); an unqualified `bind(` / `listen(` is
   covered by the file's `<winsock2.h>` include. Member and qualified calls (`x.send()`, `std::bind`) do not count.

Outside the allowed places each site counts. `tools/gates/platform_pending.txt` holds, per file, the number of
sites of the tree and the step that removes them. A file with more sites than its entry (GROWN), a file without an
entry (NEW) or an entry with fewer sites than its count (STALE: lower the entry in the same commit) fails block
mode. The list only shrinks; at PL5 it is empty.

```
pwsh -NoProfile -File tools\gates\platform_check.ps1                 # report (exit 0 always)
pwsh -NoProfile -File tools\gates\platform_check.ps1 -Mode block     # exit 1 on NEW / GROWN / STALE
pwsh -NoProfile -File tools\gates\platform_check.ps1 -SelfTest       # rule selftest + block / pass on a copy of the tree
```

Adding OS code: put it in `src/platform/<concern>_win.cpp` behind a platform-free declaration in the concern's
header, and call that from the rest of the code. If a name the check should know is missing from `PLATFORM_NAMES`,
add it in the same commit and add its site to the pending list only if it is an existing site being recorded.
