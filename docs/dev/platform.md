# The platform layer (`src/platform`)

This page is for people who add or move code that calls the operating system: files, sockets, the environment,
the console, clocks, the process itself. WHIRL runs on Windows only; the platform layer is where every Win32,
Winsock and CRT-platform call lives, so the rest of the code sees plain C++ and the tests can inject IO and socket
errors. Decision record: ADR-0004 Decision 5 (link-time per-concern headers, no virtual interface).

State (0.2.5): PL0 made the skeleton (headers, CMake target, check). **PL3 filled `net`**: every Winsock call of
the HTTP server, the release pre-check and the server tests' client is in `src/platform/net_win.cpp`, with FakeNet.
**PL1 filled env / paths / console / time** and **PL2 (fs) has moved its calls** (`fs.h` / `fs_win.cpp` /
`fs_fake.h`; sections below). PL4 (2026-10-27) moved the rest (proc / sys / args, below): the tree has no platform call
left outside the whitelist, and PL5 makes the check blocking in `release_gate.ps1`.

## Shape

| Header | Namespace | Concern | Step that fills it |
|---|---|---|---|
| `src/platform/env.h` | `whirl::platform::env` | process environment (the fingerprint's full block read, single reads, the one write) | PL1 |
| `src/platform/paths.h` | `whirl::platform::paths` | well-known folders (tune cache under LOCALAPPDATA, default cache / log folders) | PL1 |
| `src/platform/console.h` | `whirl::platform::console` | UTF-8 code page and its restore, console mode, binary stdout | PL1 |
| `src/platform/time.h` | `whirl::platform::time` | 1 ms timer resolution, local wall-clock time of log lines | PL1 |
| `src/platform/fs.h` | `whirl::platform::fs` | mapped GGUF files, the SSD tier's unbuffered / overlapped files, `_wfopen` | PL2 (with FakeFs) |
| `src/platform/net.h` | `whirl::platform::net` | the HTTP server's Winsock calls, the release front-end's port pre-check (`probeListen`), the test client | PL3 (done; FakeNet in `net_fake.h`) |
| `src/platform/proc.h` | `whirl::platform::proc` | executable path, the per-GPU named mutex, Ctrl+C / shutdown events, process id / working set | PL4 |
| `src/platform/sys.h` | `whirl::platform::sys` | physical RAM | PL4 |
| `src/platform/args.h` | `whirl::platform::args` | UTF-8 <-> UTF-16 (behind `widen` / `narrow` / `narrowArgs`, so every path and wmain's argv) | PL4 |

Each header lists, in a comment, the exact calls and files its step moves. There is no pending list (PL5: whitelist
only, see "The check").

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
  fs (PL2: write failure, short read for the tier tests) and net (PL3: accept / recv errors, see "FakeNet" below);
  how the fs fake is injected is decided in PL2.
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
ReadFile with an offset-only OVERLAPPED, `readAsync` / `overlappedResult` / `prepare` (the event: `proc::createEvent`) for the SSD
restore's QD-4 overlapped reads, `flush`, `remove`, `setSparse`, `seek` / `setEndOfFile`, file times, attributes,
`forEachMatch` = FindFirstFileW / FindNextFileW / FindClose, the read-only mapping of `MappedFile`, `openStdio` =
_wfopen, `utf8ToWideStrict`). A file handle is `fs::Handle` (the bits of a HANDLE as `void*`). The CreateFileW
arguments stay at the call sites as named constants (`kGenericRead`, `kShareRead`, `kOpenAlways`, `kNoBuffering`,
`kOverlapped`, `kDeleteOnClose`, ...) whose values `fs_win.cpp` static_asserts against `<windows.h>`.

Callers: `src/common/util.cpp` (MappedFile, readFile / writeFile), `src/model/qwen35/weights.cpp` (GGUF reader; was loader.cpp before arch025-M1),
`src/server/log.cpp` (log file), `src/tier/kv_tier.cpp` (the SSD tier). util.cpp's widen / narrow (args) and
kv_tier.cpp's GetCurrentProcessId / GetModuleFileNameW (proc) moved in PL4.

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

## CMake

`cmake/platform.cmake` defines `whirl_platform` (STATIC). It is the bottom layer: it links no other WHIRL target.
The core library links it (`target_link_libraries(whirl PUBLIC whirl_platform)` in `CMakeLists.txt`), so every
executable can reach it and PL1-PL4 only add sources to `cmake/platform.cmake`. `whirl_tier_core` (kv_tier.cpp, built
without `whirl`) links `whirl_platform` itself (`cmake/tier.cmake`, PL2); so does `whirl_hip` (PRIVATE, PL1b).
Winsock (`ws2_32`) and `winmm` move to `whirl_platform` with PL3 / PL1. The library holds `headers_win.cpp`
(referenced by nothing, dropped by the linker), `net_win.cpp` (PL3), `env_win.cpp` / `paths_win.cpp` /
`console_win.cpp` / `time_win.cpp` (PL1) and `fs_win.cpp` (PL2), and it is the only target that links `ws2_32`
(PRIVATE; `whirl_server_core` and `whirl_release` no longer name it).

## FakeNet (PL3)

`src/platform/net_fake.h`, implemented in `net_win.cpp`. It is a pass-through fake (the model is Boost.Corosio's
`mocket` fuse; MSVC has no `ld --wrap`, and there is no virtual interface). It is off unless a test calls
`fake::begin()`; off, each net call costs one relaxed atomic load more. Once on:
- it tracks the sockets a server makes: those opened on a `Lookup::passive_numeric` address and those `accept()`
  returns on them. The test client's sockets in the same process are never tracked.
- it records every call on a tracked socket with its arguments as integers (`setsockopt(65535,4102,4,300)`). The
  tags are `L1`… / `C1`…, and calls on no socket get `-`. The server test `net_call_table` compares these records with
  the call sequences and socket options of v0.2.4 0e35407.
- `failAccept(n, err)` / `failRecv(n, err)` make the next n accepts on a tracked listener, or the next n recvs on a
  tracked accepted socket, fail with that Winsock error, without calling Winsock. The tests `net_accept_error` and
  `net_recv_error` use them: the connection closes, and the server goes on.

## env / paths / console / time (PL1)

PL1 (2026-10-26) filled the four small concerns. Each function wraps exactly the call it replaced; the logic stays
with the caller.
- `env::get(name)` is the one `_dupenv_s` read: the VRAM cap (src/common/vram_limit.cpp) and the kernel table
  switches (src/kernels_host/kernels_abi.cpp), both ASCII numbers / names. LOCALAPPDATA moved to `env::getUtf8` (below);
  bug-envpaths (2026-10-26) moved `envGet` / `EnvSnapshot::capture` (src/config: every WHIRL_* of the model layer and
  whirl-server, among them the paths KV_SSD_DIR, DRAFT_MODEL, DRAFT_VOCAB, MMPROJ, CODE_OBJECT and the dump files),
  WHIRL_ALLOC_LOG / WHIRL_LAUNCH_LOG (src/hip, now opened with `_wfopen`) and the vis-encode files (src/vision/vis_cli.cpp)
  to `env::getUtf8` too, and made the draft-vocabulary paths (src/model/spec.cpp) UTF-8. Rule: an environment value that
  becomes a path is read with `env::getUtf8` and opened through `widen()`; ASCII values keep their bytes
  (child-process test `testEnvPathsChild`).
  arch025-env61 (PLAN 6.1) moved the last raw read, WHIRL_LEAK_CHECK in src/hip/leak_count.cpp (`g_on`'s static
  initializer, `leakcount::switchFrom(env::get)`), so no file outside src/platform reads the environment directly.
  `cpu_platform_check` enforces it: a raw read (`getenv` / `_wgetenv` / `getenv_s` / `_dupenv_s` / `_wdupenv_s` /
  `GetEnvironmentVariable*` / `GetEnvironmentStrings*` / `ExpandEnvironmentStrings*` / `_environ` / `_wenviron`,
  `std::`-qualified too) is allowed in `src/platform/*_win.cpp` only; the src/hip / dll_deps.cpp / stb_image.h whitelist
  does not exempt it (environment writes keep that whitelist: src/hip/gpu_restrict.cpp writes into the HIP runtime's
  own CRT copy). The env scan (`env_read_check.ps1`, `env_pending.txt`) is a separate list: it counts the
  `envGet` / `envFlag` readers of the WHIRL_* settings outside config.settings, which are not raw OS reads.
  `env::set` is the one `_putenv_s` of the product (GPU_RESOURCE_CACHE_SIZE, server_main.cpp). `env::block()`
  returns the raw `GetEnvironmentStringsW` entries in the OS order; the SSD fingerprint (src/cache/fingerprint.cpp)
  still converts (`narrow`), filters and sorts them itself, so the fingerprint is unchanged.
- `paths::localAppData()` reads LOCALAPPDATA from the CRT's wide table and converts it to UTF-8
  (`env::getUtf8`, `_wdupenv_s`; bug-lappdata 2026-10-26, as llama.cpp's `common_get_path_from_env` with `_wgetenv`).
  PL1 kept the narrow read, which is the ANSI code page: every caller hands the text to `widen()` (UTF-8), so a
  non-ASCII user folder was mangled in any program without the UTF-8 activeCodePage manifest (whirl.exe and
  whirl-server.exe have it, so on Windows 10 1903+ their paths do not change). ASCII values give the same bytes as
  before (child-process test `testLocalAppDataChild`). Still not `SHGetKnownFolderPath`: tests and users point
  LOCALAPPDATA elsewhere. The path builders (`whirlDir`,
  `serverLogPath`, `kvCacheDir`, `tuneCachePath`, `tuneReadPaths4`, `tuneReadPathsOld`) are pure string functions,
  pinned by `testPlatformPaths` with the literal v0.2.4 strings. `declarePinned` (kv_tier.cpp) still builds its own
  `pinned` folder from the pointer it is given.
- `console::isConsole` / `outputCodePage` / `setOutputCodePage` / `setStdoutBinary`; the saved code page and the
  atexit restore stay in `initConsole` (src/release/release.cpp, which no longer includes `<windows.h>`; the two
  LoadLibrary error codes are `dlldeps::k_error_mod_not_found` / `k_error_proc_not_found`).
- `time::beginPeriod1ms` / `endPeriod1ms` (winmm is linked by `whirl_platform`), `time::localTime` for log lines.

Pins: `testFingerprintPlatformEnv` (cache tests: `fingerprintEnv()` equals a verbatim copy of the base loop, and a
fixed environment set hashes to a constant computed outside C++), `testPlatformPaths`, `testPlatformEnv`,
`testPlatformConsoleTime` (unit tests). Mutations of each are in `scratch\a25_pl1\pl1_extra.ps1`.

## proc / sys / args (PL4)

PL4 (2026-10-27) moved the last calls, one function per Win32 call, arguments as in v0.2.4 0e35407:
- `proc::exePath()` is GetModuleFileNameW into a 1024-character buffer with nSize 1023 and `buf[n] = 0`, exactly the
  two copies it replaced (`exeDirectory` in src/spec/policy.cpp, `exeIdentity` in src/tier/kv_tier.cpp). arch025-d7
  (X4 K7) fixed the cut of a path of 1023 characters or more (it came back as 1022, so `exeIdentity` was 0): the first
  try is still v0.2.4's (so every path up to 1022 characters, and with it every SSD-cache key, is unchanged), and a
  truncated read (return value == nSize) is read again into a doubled buffer. `exeIdentityOf` reads the file through
  `fs::longPathForm` (`\\?\` for a path of 248+ characters), so it works without long-path awareness; the server turns
  the SSD tier off with a warning when the identity is still 0 (0 is shared by every build). `exeIdentity()` is now `exeIdentityOf(exePath())`; `testExeIdentityPinned` (tier tests) pins
  `exeIdentityOf` on files with a fixed size and last-write time to constants computed in Python, and `exeIdentity()`
  of the test exe to the v0.2.4 body evaluated with direct Win32 calls.
- The GPU_WAIT mutex (src/model/device.cpp `lockGpu`): `proc::createMutex` (CreateMutexW(nullptr, FALSE, name)) and
  `proc::wait` (WaitForSingleObject, its result as is; `kWaitObject0` / `kWaitAbandoned` / `kWaitTimeout`
  static_asserted). The name is `gpusel::lockMutexName` (lockName widened byte by byte, as before); every WHIRL version
  on one GPU must use the same name or they stop waiting for each other. `testPlatformProc` (unit tests) writes the
  uuid / PCI / PCI-with-domain names out literally and runs the wait results on real (test-named) objects, including
  an abandoned mutex.
- Shutdown (src/server/server_main.cpp): `proc::createEvent` (manual reset, not signaled), `setEvent`, `wait`,
  `sleepMs`; `proc::restoreCtrlC()` (SetConsoleCtrlHandler(nullptr, FALSE)) then `proc::addCtrlHandler(onConsoleCtrl)`
  (SetConsoleCtrlHandler(trampoline, TRUE); the trampoline turns the handler's bool into BOOL). `kCtrlC` / `kCtrlBreak` /
  `kCtrlClose` are static_asserted. No CPU test sends a console event (it would reach the whole console group).
- `proc::currentId` (the SSD directory lock owner and the pinned-memory declaration), `proc::currentProcess` +
  `proc::workingSetBytes` (K32GetProcessMemoryInfo, the WDDM-LOG RAM figure).
- `sys::memoryStatus` (GlobalMemoryStatusEx: total / available physical RAM of the RAM-tier sizing,
  src/tier/ram_size.h).
- `args::wideFromUtf8` / `utf8FromWide` (MultiByteToWideChar / WideCharToMultiByte with CP_UTF8 and flags 0: invalid
  input becomes U+FFFD, as before). `widen` / `narrow` (src/common/util.cpp) keep their "length, then convert" logic;
  the two wmain loops became `whirl::narrowArgs`.

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

Blocking since PL5 (whitelist only, any mode, no pending list). Allowed places: the implementation files of
`src/platform/` (`*_win.cpp`; the headers `src/platform/*.h` are checked and must stay platform-free), `src/hip/` (the
HIP runtime's own DXGI / PDH / LoadLibrary code), `src/release/dll_deps.cpp` and the vendored
`src/vision/stb_image.h`. `tests/` and `tools/` are not scanned; the `wmain` entry points call `whirl::narrowArgs`
and need no entry. Every other site in `src/` or `include/` fails the check in report mode too, and a line in a
`-Pending` file is itself a FAIL (`platform_pending.txt` is deleted). The release gate runs it as the CPU step
`cpu_platform_check` (`-Mode block`); `pending_scan.py platform --tree-test` proves block and pass on a copy of the tree.

```
pwsh -NoProfile -File tools\gates\platform_check.ps1 -Mode block   # exit 1 on any site outside the whitelist
pwsh -NoProfile -File tools\gates\platform_check.ps1 -SelfTest     # rule selftest + block / pass on a copy of the tree
```

Adding OS code: put it in `src/platform/<concern>_win.cpp` behind a platform-free declaration in the concern's
header, and call that from the rest of the code. If a name the check should know is missing from `PLATFORM_NAMES`,
add it in the same commit (there is no pending list to record a site in).
