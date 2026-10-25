# The platform layer (`src/platform`)

This page is for people who add or move code that calls the operating system: files, sockets, the environment,
the console, clocks, the process itself. WHIRL runs on Windows only; the platform layer is where every Win32,
Winsock and CRT-platform call lives, so the rest of the code sees plain C++ and the tests can inject IO and socket
errors. Decision record: ADR-0004 Decision 5 (link-time per-concern headers, no virtual interface).

State (0.2.5): PL0 made the skeleton (headers, CMake target, check). **PL3 filled `net`**: every Winsock call of
the HTTP server, the release pre-check and the server tests' client is in `src/platform/net_win.cpp`, with FakeNet.
PL1, PL2 and PL4 move the other calls (table below), and PL5 makes the check blocking in `release_gate.ps1`.

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

## CMake

`cmake/platform.cmake` defines `whirl_platform` (STATIC). It is the bottom layer: it links no other WHIRL target.
The core library links it (`target_link_libraries(whirl PUBLIC whirl_platform)` in `CMakeLists.txt`), so every
executable can reach it and PL1-PL4 only add sources to `cmake/platform.cmake`. Winsock (`ws2_32`) and `winmm`
move to `whirl_platform` with PL3 / PL1. Since PL3 the library holds `headers_win.cpp` (referenced by nothing) and
`net_win.cpp`, and it is the only target that links `ws2_32` (PRIVATE; `whirl_server_core` and `whirl_release` no
longer name it).

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
