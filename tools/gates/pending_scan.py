#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""pending_scan.py - the four structure checks of arch025 (S0a), each with a pending list of the sites other 0.2.5
steps remove. Report mode now; a later step switches a check to block mode (include blocks since M3, arch since M6, platform since PL5; env still reports).
A rule's "blocked" scopes fail in every mode and take no pending entry (arch025-F7: env reads in forward / engine / ops).

Rules (sources under src/ and include/; comments and string literals are not code):
  platform  a Windows / CRT-platform header (#include <windows.h>, <winsock2.h>, <shlobj.h>, <io.h>, ...) or (PL0) a
            Win32 / Winsock / CRT-platform call (PLATFORM_CALL) outside the platform layer: allowed in the
            implementation files of src/platform/ (*_win.cpp), src/hip/ (the HIP runtime's own DXGI / PDH /
            LoadLibrary code, ADR-0004 #5), src/release/dll_deps.cpp and the vendored src/vision/stb_image.h; the
            headers src/platform/*.h are checked too (they must stay platform-free, ADR-0004 #5).
            Blocking since PL5 (any mode, no pending list).
  include   whirl_server_core (cmake/server.cmake) reaches include/whirl/model.h through its sources or the repo
            headers they include (transitively); one site per (core source, header chain). Blocking since M3 (any mode).
  env       a process-environment read (getenv / _dupenv_s / _wgetenv / GetEnvironmentVariable / GetEnvironmentStrings
            / envGet / envFlag / envOn / env("...") / dump::envOn); allowed in src/platform/ and src/config/env_snapshot.*.
            The pending counts of the src/cli/ command files (arch025-C2 split of src/cli/main.cpp) and
            src/server/server_main.cpp stay at 0.2.5's end (PLAN Noted).
  arch      a host-side GPU-architecture branch (Arch::gfx<N>) outside the device layer: allowed in src/hip/,
            include/whirl/hip.h, include/whirl/device_profile.h, src/model/device.cpp. Blocking since M6 (any mode, no
            pending list): host code names a family through device::Profile / k_default_arch / k_uma_estimate_arch.

Pending list (tools/gates/<rule>_pending.txt): path<TAB>count<TAB>step per line ('#' comments). In block mode a file's
hits must equal its entry's count: more = FAIL (a new site), fewer = STALE (the step landed: lower or delete the
entry, so the list only shrinks), a file without an entry = FAIL. Report mode prints the same and exits 0.

  python -I tools/gates/pending_scan.py <rule> [--root R] [--pending F] [--mode report|block] [--write-pending]
  python -I tools/gates/pending_scan.py --selftest
Last line: "<rule> check: ok|report|FAIL (...)"; exit 0 / 1 (block mode only) / 2 usage.
--write-pending writes the current hits as the pending list (step "?" for new entries, existing steps kept).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile

EXTS = {".cpp", ".h", ".hpp", ".cc", ".inl", ".hip", ".cu", ".cuh"}
GATES = os.path.dirname(os.path.abspath(__file__))

PLATFORM_HDR = re.compile(
    r"^\s*#\s*include\s*<(windows|winsock2|ws2tcpip|mswsock|shlobj|shlobj_core|shellapi|shlwapi|knownfolders|psapi|pdh|"
    r"pdhmsg|dxgi[0-9_]*|d3dkmthk|tlhelp32|winternl|processthreadsapi|synchapi|sysinfoapi|fileapi|handleapi|"
    r"io|direct|conio|wincrypt|bcrypt|iphlpapi|mmsystem|timeapi|winioctl|ntstatus|dbghelp|objbase|combaseapi)\.h>",
    re.I)
# arch025-PL0: Win32 / CRT-platform calls, so a pending file that already includes <windows.h> cannot grow a new OS
# call unnoticed (S0a matched include lines only). Code text only (comments / strings stripped). Three parts:
#  1. any CamelCase identifier ending in the Win32 Unicode suffix W / ExW / ExA called as a function (CreateFileW,
#     GetFileAttributesExW, ...): no WHIRL function is named like that (0 false hits on 62b9e9c);
#  2. a fixed list of Win32 / Winsock / CRT-platform names without that suffix (extend it when PL1-PL4 meet one);
#  3. the BSD socket names that are also ordinary words (socket, bind, listen, accept, recv, send, select, shutdown,
#     connect) only when written with the global qualifier (::recv); unqualified ones are covered by the
#     <winsock2.h> include of the file (a std::bind / a lambda named accept is not an OS call).
# A member call (x.send(), p->ReadFile()) or a qualified one (std::bind, Foo::Sleep) is not counted.
PLATFORM_NAMES = (
    "CloseHandle|ReadFile|WriteFile|ReadFileEx|WriteFileEx|FlushFileBuffers|SetEndOfFile|SetFilePointerEx|GetFileSizeEx|"
    "GetFileTime|SetFileTime|GetFileType|LockFileEx|UnlockFileEx|GetOverlappedResult|CancelIoEx|DeviceIoControl|"
    "MapViewOfFile|UnmapViewOfFile|FindClose|GetLastError|SetLastError|WaitForSingleObject|WaitForMultipleObjects|"
    "SetEvent|ResetEvent|ReleaseMutex|ReleaseSemaphore|Sleep|SleepEx|SwitchToThread|GetCurrentProcess|"
    "GetCurrentProcessId|GetCurrentThread|GetCurrentThreadId|OpenProcess|TerminateProcess|ExitProcess|"
    "GetExitCodeProcess|DuplicateHandle|SetPriorityClass|SetThreadPriority|CreateThread|GetStdHandle|GetConsoleMode|"
    "SetConsoleMode|GetConsoleOutputCP|SetConsoleOutputCP|GetConsoleCP|SetConsoleCP|SetConsoleCtrlHandler|"
    "GlobalMemoryStatusEx|GetSystemInfo|GetNativeSystemInfo|K32GetProcessMemoryInfo|GetProcessMemoryInfo|"
    "QueryPerformanceCounter|QueryPerformanceFrequency|GetTickCount|GetTickCount64|GetLocalTime|GetSystemTime|"
    "GetSystemTimeAsFileTime|timeBeginPeriod|timeEndPeriod|timeGetTime|MultiByteToWideChar|WideCharToMultiByte|"
    "VirtualAlloc|VirtualFree|VirtualLock|VirtualUnlock|VirtualQuery|HeapAlloc|HeapFree|LocalFree|GlobalFree|"
    "LoadLibrary|GetProcAddress|FreeLibrary|CoTaskMemFree|InitializeCriticalSection|EnterCriticalSection|"
    "LeaveCriticalSection|DeleteCriticalSection|CreateIoCompletionPort|GetQueuedCompletionStatus|"
    "PostQueuedCompletionStatus|MAKEWORD|"
    "WSA[A-Z]\\w*|closesocket|setsockopt|getsockopt|ioctlsocket|getaddrinfo|freeaddrinfo|GetAddrInfoW|FreeAddrInfoW|"
    "inet_pton|inet_ntop|getpeername|getsockname|"
    "_setmode|_fileno|_isatty|_get_osfhandle|_open_osfhandle|_wfopen|_wfopen_s|_wopen|_wstat|_wstat64|_wstati64|"
    "_waccess|_wmkdir|_wrmdir|_wremove|_wrename|_wunlink|_wputenv|_putenv|_putenv_s|_commit|_chsize_s|_locking|"
    "_getpid|_beginthreadex")
PLATFORM_CALL = re.compile(
    r"(?<![\w.>:])(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z0-9]*)*[a-z0-9](?:W|ExW|ExA)"
    r"|(?:" + PLATFORM_NAMES + r")"
    r"|::(?:socket|bind|listen|accept|recv|send|select|shutdown|connect)"
    r")\s*\(")
PLATFORM_HEADER_EXTS = {".h", ".hpp", ".inl", ".cuh"}
ENV_READ = re.compile(
    r"\b(?:getenv|_wgetenv|_dupenv_s|_wdupenv_s|GetEnvironmentVariable[AW]?|GetEnvironmentStrings[AW]?|envGet|envFlag|"
    r"envOn)\s*\(|(?<![\w.:>])env\s*\(\s*\"")
ARCH = re.compile(r"\bArch::gfx\d+\b")

RULES = {
    # blocked since arch025-PL5: whitelist only (src/platform *_win.cpp, src/hip, dll_deps.cpp, the vendored stb_image.h; tests/
    # and tools/ are not scanned, the wmain entry points call whirl::narrowArgs); no pending list (exit 1 in report mode too)
    "platform": {"allowed": ["src/platform/", "src/hip/", "src/release/dll_deps.cpp", "src/vision/stb_image.h"],
                 "what": "platform header / OS call", "blocked": ["src/", "include/"]},
    # blocked since arch025-M3 (R-8c): the server core sees whirl/model_types.h only; no core source may reach model.h
    # and the pending list is gone (exit 1 in report mode too)
    "include": {"allowed": [], "what": "server core reaches model.h", "blocked": ["src/", "include/"]},
    # blocked (arch025-F7, plan X-1): no env read and no pending entry, in report mode too (exit 1). The forward and the
    # server engine read the load's EnvSnapshot / their Model members only; src/ops (the forward's choices since
    # F5 / F6) is pure.
    "env": {"allowed": ["src/platform/", "src/config/env_snapshot."], "what": "env read",
            "blocked": ["src/model/qwen35/forward/", "src/model/qwen35/forward_internal.h", "src/model/forward.cpp", "src/server/engine",
                        "src/model/qwen35/server_model.cpp", "src/ops/", "include/whirl/ops/"]},
    # blocked since arch025-M6: host Arch:: values only in the device layer (DeviceProfile and its named defaults); the
    # pending list is gone (exit 1 in report mode too)
    "arch": {"allowed": ["src/hip/", "include/whirl/hip.h", "include/whirl/device_profile.h", "src/model/device.cpp"],
             "what": "Arch:: branch", "blocked": ["src/", "include/"]},
}


def strip_code(text: str) -> str:
    """Comments removed and string literal contents blanked, line structure kept. Include lines keep their <...>."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            body = text[i + 1:j]
            # keep the first word of a literal: env("KV") must still look like a call with a string argument
            out.append('"' + re.sub(r"[^\n]", " ", body) + ('"' if j < n and text[j] == '"' else ""))
            i = j + 1 if j < n and text[j] == '"' else j
        elif c == "'":
            j = i + 1
            while j < n and text[j] != "'" and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append("' '" if j < n and text[j] == "'" else "'")
            i = j + 1 if j < n and text[j] == "'" else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def rel(root: str, p: str) -> str:
    return os.path.relpath(p, root).replace("\\", "/")


def sources(root: str, dirs=("src", "include")) -> list[str]:
    res = []
    for d in dirs:
        base = os.path.join(root, d)
        for dp, _, fs in os.walk(base):
            for f in fs:
                if os.path.splitext(f)[1].lower() in EXTS:
                    res.append(rel(root, os.path.join(dp, f)))
    return sorted(res)


def read(root: str, r: str) -> str:
    with open(os.path.join(root, r), encoding="utf-8", errors="replace") as f:
        return f.read()


def allowed(rule: str, r: str) -> bool:
    return any(r == a or (a.endswith("/") and r.startswith(a)) or (a.endswith(".") and r.startswith(a))
               for a in RULES[rule]["allowed"])


def line_hits(root: str, rule: str) -> dict[str, list[tuple[int, str]]]:
    """path -> [(line, text)] for the regex rules."""
    hits: dict[str, list[tuple[int, str]]] = {}
    for r in sources(root):
        # the platform layer's own headers are not exempt: they must not include a platform header or call the OS
        platform_hdr = (rule == "platform" and r.startswith("src/platform/")
                        and os.path.splitext(r)[1].lower() in PLATFORM_HEADER_EXTS)
        if allowed(rule, r) and not platform_hdr:
            continue
        text = read(root, r)
        raw = text.split("\n")
        if rule == "platform":
            # include lines are matched on the raw text (strip_code keeps <...> but blanks "..." headers; the
            # platform headers are all <...>), comment-only lines are skipped via the stripped text; OS calls are
            # matched on the stripped text (one site per call)
            code = strip_code(text).split("\n")
            for i, l in enumerate(raw):
                if PLATFORM_HDR.match(l) and code[i].strip():
                    hits.setdefault(r, []).append((i + 1, l.strip()))
                for _ in PLATFORM_CALL.finditer(code[i]):
                    hits.setdefault(r, []).append((i + 1, l.strip()))
            continue
        pat = ENV_READ if rule == "env" else ARCH
        code = strip_code(text).split("\n")
        for i, l in enumerate(code):
            for _ in pat.finditer(l):
                hits.setdefault(r, []).append((i + 1, raw[i].strip()))
    return hits


def cmake_sources(root: str, target: str) -> list[str]:
    """The source list of add_library(<target> ...) in cmake/*.cmake or CMakeLists.txt."""
    files = [os.path.join(root, "CMakeLists.txt")]
    cdir = os.path.join(root, "cmake")
    if os.path.isdir(cdir):
        files += [os.path.join(cdir, f) for f in sorted(os.listdir(cdir)) if f.endswith(".cmake")]
    for f in files:
        if not os.path.exists(f):
            continue
        t = open(f, encoding="utf-8", errors="replace").read()
        m = re.search(r"add_library\(\s*" + re.escape(target) + r"\b([^)]*)\)", t)
        if m:
            out = []
            for tok in m.group(1).split():
                tok = tok.replace("${CMAKE_SOURCE_DIR}/", "")
                if os.path.splitext(tok)[1].lower() in EXTS:
                    out.append(tok)
            return out
    return []


def include_hits(root: str) -> dict[str, list[tuple[int, str]]]:
    """Core source -> [(line, chain)] for every repo-header chain from it to include/whirl/model.h."""
    target = "include/whirl/model.h"
    inc_re = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
    cache: dict[str, list[tuple[int, str]]] = {}

    def resolve(frm: str, name: str) -> str | None:
        for base in (os.path.dirname(frm), "include", "src"):
            c = os.path.normpath(os.path.join(base, name)).replace("\\", "/")
            if os.path.exists(os.path.join(root, c)):
                return c
        return None

    def includes(f: str) -> list[tuple[int, str]]:
        if f not in cache:
            res = []
            raw = read(root, f).split("\n")
            code = strip_code("\n".join(raw)).split("\n")
            for i, l in enumerate(raw):
                m = inc_re.match(l)
                if m and code[i].strip():
                    p = resolve(f, m.group(1))
                    if p:
                        res.append((i + 1, p))
            cache[f] = res
        return cache[f]

    hits: dict[str, list[tuple[int, str]]] = {}
    for src in cmake_sources(root, "whirl_server_core"):
        if not os.path.exists(os.path.join(root, src)):
            continue
        seen = {src}
        stack = [(src, [src], 0)]
        while stack:
            f, chain, first_line = stack.pop()
            for ln, p in includes(f):
                fl = first_line or ln
                if p == target:
                    hits.setdefault(src, []).append((fl, " -> ".join(chain + [p])))
                    continue
                if p in seen:
                    continue
                seen.add(p)
                stack.append((p, chain + [p], fl))
    for v in hits.values():
        v.sort()
    return hits


def load_pending(path: str):
    entries = {}
    if not path or not os.path.exists(path):
        return entries, None
    for n, l in enumerate(open(path, encoding="utf-8").read().split("\n"), 1):
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        c = l.split("\t")
        if len(c) < 3 or not c[1].strip().isdigit():
            return None, f"{path}:{n} is not path<TAB>count<TAB>step"
        entries[c[0].strip()] = (int(c[1]), c[2].strip(), n)
    return entries, None


def run(rule: str, root: str, pending: str, mode: str, out=print) -> int:
    if rule not in RULES:
        out(f"{rule} check: FAIL (usage: unknown rule; one of {', '.join(RULES)})")
        return 2
    if not os.path.isdir(os.path.join(root, "src")):
        out(f"{rule} check: FAIL (usage: no src/ under {root})")
        return 2
    entries, err = load_pending(pending)
    if err:
        out(f"{rule} check: FAIL (usage: {err})")
        return 2
    hits = include_hits(root) if rule == "include" else line_hits(root, rule)
    new = grown = stale = 0
    blocked_pre = RULES[rule].get("blocked", [])
    is_blocked = lambda p: any(p.startswith(b) for b in blocked_pre)
    blocked = 0
    for path in sorted(set(hits) | set(entries)):
        if is_blocked(path):
            h = hits.get(path, [])
            if path in entries:
                blocked += 1
                out(f"  BLOCKED pending line {entries[path][2]}: {path} is in a blocked scope (no pending entry allowed)")
            if h:
                blocked += 1
                out(f"  BLOCKED {RULES[rule]['what']} {path}: {len(h)} site(s) in a blocked scope (any mode)")
                for ln, t in h:
                    out(f"    {path}:{ln}: {t}")
            continue
        h = hits.get(path, [])
        if path in entries:
            cnt, step, n = entries[path]
            if len(h) > cnt:
                grown += 1
                out(f"  GROWN {path}: {len(h)} site(s), pending allows {cnt} ({step}, line {n})")
                for ln, t in h:
                    out(f"    {path}:{ln}: {t}")
            elif len(h) < cnt:
                stale += 1
                out(f"  STALE pending line {n}: {path} x{cnt} ({step}) has {len(h)} site(s): lower or delete the entry")
            else:
                out(f"  pending {step}: {path} x{cnt}")
        else:
            new += 1
            out(f"  NEW {RULES[rule]['what']} {path}: {len(h)} site(s) (no pending entry)")
            for ln, t in h:
                out(f"    {path}:{ln}: {t}")
    total = sum(len(v) for v in hits.values())
    summary = f"{total} site(s) in {len(hits)} file(s), {len(entries)} pending entr(y/ies); {new} new, {grown} grown, {stale} stale"
    if blocked_pre:
        summary += f", {blocked} blocked"
    bad = new + grown + stale
    if blocked:
        out(f"{rule} check: FAIL ({summary})")
        return 1
    if mode == "block":
        if bad:
            out(f"{rule} check: FAIL ({summary})")
            return 1
        out(f"{rule} check: ok ({summary})")
        return 0
    out(f"{rule} check: report ({summary}{'; block mode would FAIL' if bad else ''})")
    return 0


def write_pending(rule: str, root: str, pending: str) -> int:
    entries, err = load_pending(pending)
    if err:
        print(err)
        return 2
    hits = include_hits(root) if rule == "include" else line_hits(root, rule)
    head = [f"# arch025-S0a {rule} check (tools/gates/pending_scan.py): path<TAB>count<TAB>step that removes the sites.",
            "# The list may only shrink: a step that removes sites lowers or deletes its entry (block mode: STALE otherwise)."]
    if os.path.exists(pending):
        head = [l for l in open(pending, encoding="utf-8").read().split("\n") if l.startswith("#")] or head
    lines = head + [f"{p}\t{len(hits[p])}\t{(entries or {}).get(p, (0, '?', 0))[1]}" for p in sorted(hits)]
    with open(pending, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {pending}: {len(hits)} entr(y/ies)")
    return 0


def selftest() -> int:
    fails = 0
    cases = 0
    base = tempfile.mkdtemp(prefix="whirl_pending_scan_")

    def tree(files: dict[str, str]):
        shutil.rmtree(base, ignore_errors=True)
        for k, v in files.items():
            p = os.path.join(base, k)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(v)

    def case(name, rule, files, pend, mode, want):
        nonlocal fails, cases
        cases += 1
        tree(files)
        pf = ""
        if pend is not None:
            pf = os.path.join(base, "pending.txt")
            with open(pf, "w", encoding="utf-8") as f:
                f.write(pend)
        lines: list[str] = []
        rc = run(rule, base, pf, mode, out=lines.append)
        ok = rc == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} [{rule}/{mode}] {name}: rc {rc} (want {want}) | {lines[-1] if lines else ''}")

    T = "\t"
    CORE = ("add_library(whirl_server_core STATIC\n  ${CMAKE_SOURCE_DIR}/src/server/engine.cpp\n"
            "  ${CMAKE_SOURCE_DIR}/src/server/http.cpp)\n")
    # platform
    case("clean tree passes", "platform", {"src/a.cpp": "#include <vector>\nint x;\n"}, None, "block", 0)
    case("windows.h outside the platform layer blocked", "platform", {"src/a.cpp": "#include <windows.h>\n"}, None, "block", 1)
    case("same in report mode exits 1 (PL5)", "platform", {"src/a.cpp": "#include <windows.h>\n"}, None, "report", 1)
    case("allowed dirs pass", "platform", {"src/hip/a.cpp": "#include <windows.h>\n", "src/platform/env_win.cpp": "#include <windows.h>\n",
                                           "src/release/dll_deps.cpp": "#  include <psapi.h>\n"}, None, "block", 0)
    case("commented include is not code", "platform", {"src/a.cpp": "// #include <windows.h>\n/* #include <io.h> */\n"}, None, "block", 0)
    case("malformed pending is a usage error", "platform", {"src/a.cpp": "\n"}, "src/a.cpp two PL1\n", "block", 2)
    # platform, PL0: OS calls and the platform layer's own headers
    case("new OS call without a platform include blocked", "platform",
         {"src/model/x.cpp": "void f() { auto h = CreateFileW(p, 0, 0, nullptr, 3, 0, nullptr); }\n"}, None, "block", 1)
    case("same in report mode exits 1 (PL5)", "platform",
         {"src/model/x.cpp": "void f() { CreateFileW(p, 0, 0, nullptr, 3, 0, nullptr); }\n"}, None, "report", 1)
    case("not OS calls: member, qualified, std::bind, a lambda named accept, comment, string", "platform",
         {"src/a.cpp": "x.send(b); p->ReadFile(); Foo::Sleep(1); auto g = std::bind(f, 1); if (accept(c)) {}\n"
                       "// CreateFileW(p)\nconst char* s = \"GetLastError()\"; recv(s, b, 1, 0);\n"}, None, "block", 0)
    case("OS calls in src/platform/*_win.cpp pass", "platform",
         {"src/platform/fs_win.cpp": "#include <windows.h>\nvoid* o(const wchar_t* p) { return CreateFileW(p, 0, 0, nullptr, 3, 0, nullptr); }\n"},
         None, "block", 0)
    case("a src/platform header including windows.h blocked", "platform",
         {"src/platform/fs.h": "#pragma once\n#include <windows.h>\n"}, None, "block", 1)
    case("pass: skeleton shape (platform-free header, *_win.cpp, caller through the header)", "platform",
         {"src/platform/fs.h": "#pragma once\nnamespace whirl::platform::fs { bool exists(const char* p); }\n",
          "src/platform/fs_win.cpp": "#include \"platform/fs.h\"\n#include <windows.h>\n"
                                     "bool whirl::platform::fs::exists(const char*) { return GetFileAttributesW(L\"x\") != 0; }\n",
          "src/tier/t.cpp": "#include \"platform/fs.h\"\nbool e() { return whirl::platform::fs::exists(\"x\"); }\n"},
         None, "block", 0)
    # arch025-PL5: blocking in any mode, no pending list
    case("PL5: windows.h outside the layer blocked in report mode", "platform", {"src/server/s.cpp": "#include <windows.h>\n"}, None, "report", 1)
    case("PL5: an OS call outside the layer blocked in report mode", "platform", {"src/tier/t.cpp": "void f(void* h) { CloseHandle(h); }\n"}, None, "report", 1)
    case("PL5: a pending entry does not allow it", "platform", {"src/a.cpp": "#include <windows.h>\n"}, f"src/a.cpp{T}1{T}PL1\n", "block", 1)
    case("PL5: a pending entry for a clean file is itself a FAIL", "platform", {"src/a.cpp": "int x;\n"}, f"src/a.cpp{T}0{T}PL1\n", "report", 1)
    case("PL5: include/ header blocked", "platform", {"include/whirl/x.h": "#include <windows.h>\n", "src/a.cpp": "int x;\n"}, None, "report", 1)
    case("PL5: whitelist (hip, dll_deps, stb_image, platform *_win.cpp) passes in report mode", "platform",
         {"src/hip/a.cpp": "#include <windows.h>\n", "src/release/dll_deps.cpp": "#include <windows.h>\nvoid f(void* h){CloseHandle(h);}\n",
          "src/vision/stb_image.h": "#include <io.h>\n", "src/platform/p_win.cpp": "#include <windows.h>\nvoid f(void* h){CloseHandle(h);}\n"}, None, "report", 0)
    case("PL5: an entry point calling the layer only passes", "platform",
         {"src/cli/main.cpp": "#include \"platform/args.h\"\nint wmain(int c, wchar_t** v) { return run(whirl::narrowArgs(c, v)); }\n"}, None, "report", 0)
    case("env reads blocked", "env", {"src/model/f.cpp": 'int a = envFlag("X", true);\nauto b = std::getenv("WHIRL_Y");\n'}, None, "block", 1)
    case("env helper with a string argument counts", "env", {"src/server/s.cpp": 'const auto kv = env("KV");\n'}, None, "block", 1)
    case("not env reads: fingerprintEnv(), x.env(), a comment, a string", "env",
         {"src/a.cpp": 'auto e = fingerprintEnv();\nauto f = cfg.env("x");\n// getenv("A")\nconst char* s = "getenv(";\n'}, None, "block", 0)
    case("env pending counts exactly", "env", {"src/cli/main.cpp": 'envGet("A"); envGet("B");\nstd::getenv("C");\n'},
         f"src/cli/main.cpp{T}3{T}C2\n", "block", 0)
    case("env snapshot file allowed", "env", {"src/config/env_snapshot.cpp": 'std::getenv("A");\n'}, None, "block", 0)
    # arch025-F7: forward / engine / ops blocked in any mode, even with a pending entry
    case("F7: forward env read blocked in report mode", "env", {"src/model/qwen35/forward/dispatch.cpp": 'bool a = envFlag("HEAD_GEMVX", false);\n'},
         None, "report", 1)
    case("F7: forward pending entry does not allow it", "env", {"src/model/qwen35/forward/gdn.cpp": 'auto v = std::getenv("WHIRL_X");\n'},
         f"src/model/qwen35/forward/gdn.cpp{T}1{T}F7\n", "block", 1)
    case("F7: a pending entry for a blocked path is itself a FAIL", "env", {"src/model/qwen35/forward/gdn.cpp": "int x;\n"},
         f"src/model/qwen35/forward/gdn.cpp{T}0{T}F7\n", "report", 1)
    case("F7: engine_thread env read blocked", "env", {"src/server/engine_thread.cpp": 'auto v = envGet("KV");\n'}, None, "report", 1)
    case("F7: forward_internal.h env read blocked", "env", {"src/model/qwen35/forward_internal.h": 'inline bool f() { return envFlag("A"); }\n'},
         None, "report", 1)
    case("F7: ops env read blocked", "env", {"include/whirl/ops/gdn.h": 'const char* v = getenv("WHIRL_A");\n', "src/ops/gdn.cpp": "int x;\n"}, None, "report", 1)
    case("F7: forward comment / string / snapshot field are not reads", "env",
         {"src/model/qwen35/forward/dispatch.cpp": '// envFlag("HEAD_GEMVX")\nconst char* s = "getenv(";\nbool h = head_gemvx;\n'}, None, "block", 0)
    case("F7: the rest of the tree stays report mode", "env", {"src/server/server_main.cpp": 'envGet("A");\n'}, None, "report", 0)
    # arch
    case("Arch:: branch blocked", "arch", {"src/model/config.cpp": "if (a == hip::Arch::gfx1151) {}\n"}, None, "block", 1)
    case("device layer allowed", "arch", {"include/whirl/device_profile.h": "Arch::gfx1151\n", "src/hip/h.cpp": "Arch::gfx1201\n"}, None, "block", 0)
    case("Arch::gfx in a comment is not code", "arch", {"src/a.cpp": "// Arch::gfx1151 only\n"}, None, "block", 0)
    # arch025-M6: blocking in any mode, a pending entry no longer covers a site
    case("M6: Arch:: branch blocked in report mode", "arch", {"src/vision/kernel_image.cpp": "case hip::Arch::gfx1201: break;\n"},
         None, "report", 1)
    case("M6: a pending entry does not allow it", "arch", {"src/model/config.cpp": "auto a = hip::Arch::gfx1201;\n"},
         f"src/model/config.cpp{T}1{T}M6\n", "block", 1)
    case("M6: a pending entry for a clean file is itself a FAIL", "arch", {"src/model/config.cpp": "int x;\n"},
         f"src/model/config.cpp{T}0{T}M6\n", "report", 1)
    case("M6: an include/ header outside the device layer blocked", "arch", {"include/whirl/model.h": "hip::Arch a = hip::Arch::gfx1201;\n",
                                                                                 "src/a.cpp": "int x;\n"}, None, "report", 1)
    case("M6: named defaults from the device layer pass", "arch",
         {"include/whirl/device_profile.h": "inline constexpr hip::Arch k_default_arch = hip::Arch::gfx1201;\n",
          "src/model/config.cpp": "auto g = device::profile(device::k_default_arch).gemv_r;\n",
          "src/server/server_main.cpp": "auto d = mtpDefaults(moe, device::k_uma_estimate_arch);\n",
          "src/model/device.cpp": "Arch::gfx1151\n", "src/hip/hip_runtime.cpp": "Arch::gfx1151\n"}, None, "report", 0)
    case("M6: a string or comment naming gfx1151 is not code", "arch",
         {"src/vision/kernel_image.cpp": '{"gfx1151", d, &n},  // was case hip::Arch::gfx1151\n'}, None, "block", 0)
    # include
    inc_tree = {"cmake/server.cmake": CORE, "src/server/engine.cpp": '#include "server/engine.h"\n',
                "src/server/engine.h": '#include "whirl/model.h"\n', "src/server/http.cpp": '#include "whirl/common.h"\n',
                "include/whirl/model.h": "#pragma once\n", "include/whirl/common.h": "#pragma once\n"}
    case("server core reaching model.h through a header blocked", "include", inc_tree, None, "block", 1)
    case("since M3 a pending entry does not cover it", "include", inc_tree, f"src/server/engine.cpp{T}1{T}M3\n", "block", 1)
    case("since M3 report mode fails too", "include", inc_tree, None, "report", 1)
    clean = dict(inc_tree)
    clean["src/server/engine.h"] = '#include "whirl/model_types.h"\n'
    clean["include/whirl/model_types.h"] = "#pragma once\n"
    case("after M3: pending entry STALE", "include", clean, f"src/server/engine.cpp{T}1{T}M3\n", "block", 1)
    case("after M3: clean", "include", clean, None, "block", 0)
    other = dict(clean)
    other["src/model/qwen35/server_model.cpp"] = '#include "whirl/model.h"\n'  # not a core source
    case("a non-core source may include model.h", "include", other, None, "block", 0)
    case("a non-core source may include model.h (report)", "include", other, None, "report", 0)
    shutil.rmtree(base, ignore_errors=True)
    if fails:
        print(f"pending_scan selftest: FAIL ({fails} of {cases})")
        return 1
    print(f"pending_scan selftest: ok ({cases} cases)")
    return 0


def tree_test(root: str, rule: str = "platform") -> int:
    """block / pass on a copy of the real tree with the real pending list (PL5: platform has none). Pass: the copy as is.
    Block: a raw OS call / platform include added outside the whitelist; pass: the same inside it."""
    pending = os.path.join(GATES, f"{rule}_pending.txt")
    base = tempfile.mkdtemp(prefix="whirl_pending_tree_")
    fails = 0
    n = 0
    try:
        for d in ("src", "include"):
            shutil.copytree(os.path.join(root, d), os.path.join(base, d))

        def one(name, want, mode="block"):
            nonlocal fails
            lines: list[str] = []
            rc = run(rule, base, pending, mode, out=lines.append)
            ok = rc == want
            fails += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} [{rule}/tree/{mode}] {name}: rc {rc} (want {want}) | {lines[-1] if lines else ''}")

        one("the tree with its pending list passes", 0)
        if rule == "env":
            # arch025-F7: forward / engine blocked in both modes; the copy as is passes in both
            one("the tree passes in report mode", 0, "report")
            n = 2
            for rel, line in (("src/model/qwen35/forward/dispatch.cpp", '\nstatic bool f7Probe() { return envFlag("HEAD_GEMVX", false); }\n'),
                              ("src/server/engine.cpp", '\nstatic const char* f7Probe() { return std::getenv("WHIRL_KV"); }\n')):
                f = os.path.join(base, *rel.split("/"))
                orig = open(f, "rb").read()
                with open(f, "a", encoding="utf-8") as h:
                    h.write(line)
                one(f"an env read added to {rel} is blocked", 1)
                one(f"an env read added to {rel} is blocked in report mode", 1, "report")
                n += 2
                with open(f, "wb") as h:
                    h.write(orig)
            one("restored tree passes", 0)
            n += 1
            if fails:
                print(f"pending_scan tree test: FAIL ({fails} of {n})")
                return 1
            print(f"pending_scan tree test: ok ({n} cases)")
            return 0
        if rule == "arch":
            # arch025-M6: blocking in both modes; the copy as is passes in both, a site added anywhere is blocked in both
            one("the tree passes in report mode", 0, "report")
            n = 2
            for rel, line in (("src/model/config.cpp", "\nstatic const auto m6Probe = hip::Arch::gfx1201;\n"),
                              ("src/server/server_main.cpp", "\nstatic bool m6Probe(whirl::hip::Arch a) { return a == whirl::hip::Arch::gfx1151; }\n"),
                              ("include/whirl/model.h", "\ninline constexpr auto m6_probe = whirl::hip::Arch::gfx1151;\n")):
                f = os.path.join(base, *rel.split("/"))
                orig = open(f, "rb").read()
                with open(f, "a", encoding="utf-8") as h:
                    h.write(line)
                one(f"an Arch:: value added to {rel} is blocked", 1)
                one(f"an Arch:: value added to {rel} is blocked in report mode", 1, "report")
                n += 2
                with open(f, "wb") as h:
                    h.write(orig)
            one("restored tree passes", 0)
            n += 1
            if fails:
                print(f"pending_scan tree test: FAIL ({fails} of {n})")
                return 1
            print(f"pending_scan tree test: ok ({n} cases)")
            return 0
        # arch025-PL5: blocking in both modes with no pending list; a raw Win32 call or platform include outside the
        # whitelist blocks anywhere (src/, include/), the same inside src/platform/*_win.cpp, src/hip, dll_deps.cpp passes
        one("the tree passes in report mode", 0, "report")
        n = 2
        for rel, line in (("src/model/pl5_probe.cpp", "unsigned long long now() { return GetTickCount64(); }\n"),
                          ("src/server/http.cpp", "\nstatic void pl5Probe(void* h) { CloseHandle(h); }\n"),
                          ("src/cli/main.cpp", "\n#include <windows.h>\n"),
                          ("include/whirl/common.h", "\nstatic inline void pl5Probe(void* h) { CloseHandle(h); }\n")):
            f = os.path.join(base, *rel.split("/"))
            had = os.path.exists(f)
            orig = open(f, "rb").read() if had else None
            with open(f, "a", encoding="utf-8") as h:
                h.write(line)
            one(f"a raw Win32 call / platform include added to {rel} is blocked", 1)
            one(f"the same in {rel} is blocked in report mode", 1, "report")
            n += 2
            if had:
                with open(f, "wb") as h:
                    h.write(orig)
            else:
                os.remove(f)
        for rel in ("src/platform/pl5_probe_win.cpp", "src/hip/pl5_probe.cpp", "src/release/dll_deps.cpp"):
            f = os.path.join(base, *rel.split("/"))
            had = os.path.exists(f)
            orig = open(f, "rb").read() if had else None
            with open(f, "a", encoding="utf-8") as h:
                h.write("\n#include <windows.h>\nstatic void pl5Probe(void* h) { CloseHandle(h); }\n")
            one(f"the same raw call inside {rel} (whitelist) passes", 0)
            n += 1
            if had:
                with open(f, "wb") as h:
                    h.write(orig)
            else:
                os.remove(f)
        hdr = os.path.join(base, "src", "platform", "pl5_probe.h")
        with open(hdr, "w", encoding="utf-8") as h:
            h.write("#pragma once\n#include <windows.h>\n")
        one("a src/platform header including windows.h is blocked", 1)
        n += 1
        os.remove(hdr)
        one("restored tree passes", 0)
        n += 1
        pend = os.path.join(base, "pl5_pending.txt")
        with open(pend, "w", encoding="utf-8") as h:
            h.write("src/model/config.cpp\t0\tPL5\n")
        lines: list[str] = []
        rc = run(rule, base, pend, "report", out=lines.append)
        n += 1
        if rc != 1:
            fails += 1
        print(f"  {'ok  ' if rc == 1 else 'FAIL'} [{rule}/tree/report] a pending line is itself a FAIL: rc {rc} (want 1)")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if fails:
        print(f"pending_scan tree test: FAIL ({fails} of {n})")
        return 1
    print(f"pending_scan tree test: ok ({n} cases)")
    return 0


def main(argv: list[str], rule_default: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rule", nargs="?", default=rule_default)
    ap.add_argument("--root", default=os.path.normpath(os.path.join(GATES, "..", "..")))
    ap.add_argument("--pending", default=None)
    ap.add_argument("--mode", choices=["report", "block"], default="report")
    ap.add_argument("--write-pending", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tree-test", action="store_true", help="block / pass on a copy of --root (platform, env, arch rules)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.tree_test:
        return tree_test(a.root, a.rule or "platform")
    if not a.rule:
        print("check: FAIL (usage: name a rule: " + ", ".join(RULES) + ")")
        return 2
    pending = a.pending if a.pending is not None else os.path.join(GATES, f"{a.rule}_pending.txt")
    if a.write_pending:
        return write_pending(a.rule, a.root, pending)
    return run(a.rule, a.root, pending, a.mode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
