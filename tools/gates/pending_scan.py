#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""pending_scan.py - the four structure checks of arch025 (S0a), each with a pending list of the sites other 0.2.5
steps remove. Report mode now; a later step switches a check to block mode (PL5 platform, M3 include, F7 env reads
in forward / engine, M6 arch).

Rules (sources under src/ and include/; comments and string literals are not code):
  platform  a Windows / CRT-platform header (#include <windows.h>, <winsock2.h>, <shlobj.h>, <io.h>, ...) outside
            the platform layer: allowed in src/platform/, src/hip/ (the HIP runtime's own DXGI / PDH / LoadLibrary
            code, ADR-0004 #5) and src/release/dll_deps.cpp.
  include   whirl_server_core (cmake/server.cmake) reaches include/whirl/model.h through its sources or the repo
            headers they include (transitively); one site per (core source, header chain). M3 makes it block.
  env       a process-environment read (getenv / _dupenv_s / _wgetenv / GetEnvironmentVariable / GetEnvironmentStrings
            / envGet / envFlag / envOn / env("...") / dump::envOn); allowed in src/platform/ and src/config/env_snapshot.*.
            The pending counts of src/cli/main.cpp and src/server/server_main.cpp stay at 0.2.5's end (PLAN Noted).
  arch      a host-side GPU-architecture branch (Arch::gfx<N>) outside the device layer: allowed in src/hip/,
            include/whirl/hip.h, include/whirl/device_profile.h, src/model/device.cpp.

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
ENV_READ = re.compile(
    r"\b(?:getenv|_wgetenv|_dupenv_s|_wdupenv_s|GetEnvironmentVariable[AW]?|GetEnvironmentStrings[AW]?|envGet|envFlag|"
    r"envOn)\s*\(|(?<![\w.:>])env\s*\(\s*\"")
ARCH = re.compile(r"\bArch::gfx\d+\b")

RULES = {
    "platform": {"allowed": ["src/platform/", "src/hip/", "src/release/dll_deps.cpp"], "what": "platform header"},
    "include": {"allowed": [], "what": "server core reaches model.h"},
    "env": {"allowed": ["src/platform/", "src/config/env_snapshot."], "what": "env read"},
    "arch": {"allowed": ["src/hip/", "include/whirl/hip.h", "include/whirl/device_profile.h", "src/model/device.cpp"],
             "what": "Arch:: branch"},
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
        if allowed(rule, r):
            continue
        text = read(root, r)
        raw = text.split("\n")
        if rule == "platform":
            # include lines are matched on the raw text (strip_code keeps <...> but blanks "..." headers; the
            # platform headers are all <...>), comment-only lines are skipped via the stripped text
            code = strip_code(text).split("\n")
            for i, l in enumerate(raw):
                if PLATFORM_HDR.match(l) and code[i].strip():
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
    for path in sorted(set(hits) | set(entries)):
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
    bad = new + grown + stale
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
    case("same in report mode exits 0", "platform", {"src/a.cpp": "#include <windows.h>\n"}, None, "report", 0)
    case("allowed dirs pass", "platform", {"src/hip/a.cpp": "#include <windows.h>\n", "src/platform/env_win.cpp": "#include <windows.h>\n",
                                           "src/release/dll_deps.cpp": "#  include <psapi.h>\n"}, None, "block", 0)
    case("commented include is not code", "platform", {"src/a.cpp": "// #include <windows.h>\n/* #include <io.h> */\n"}, None, "block", 0)
    case("pending entry covers its file", "platform", {"src/a.cpp": "#include <windows.h>\n#include <shlobj.h>\n"},
         f"src/a.cpp{T}2{T}PL1\n", "block", 0)
    case("one more than pending: GROWN", "platform", {"src/a.cpp": "#include <windows.h>\n#include <io.h>\n#include <shlobj.h>\n"},
         f"src/a.cpp{T}2{T}PL1\n", "block", 1)
    case("fewer than pending: STALE", "platform", {"src/a.cpp": "#include <windows.h>\n"}, f"src/a.cpp{T}2{T}PL1\n", "block", 1)
    case("malformed pending is a usage error", "platform", {"src/a.cpp": "\n"}, "src/a.cpp two PL1\n", "block", 2)
    # env
    case("env reads blocked", "env", {"src/model/f.cpp": 'int a = envFlag("X", true);\nauto b = std::getenv("WHIRL_Y");\n'}, None, "block", 1)
    case("env helper with a string argument counts", "env", {"src/server/s.cpp": 'const auto kv = env("KV");\n'}, None, "block", 1)
    case("not env reads: fingerprintEnv(), x.env(), a comment, a string", "env",
         {"src/a.cpp": 'auto e = fingerprintEnv();\nauto f = cfg.env("x");\n// getenv("A")\nconst char* s = "getenv(";\n'}, None, "block", 0)
    case("env pending counts exactly", "env", {"src/cli/main.cpp": 'envGet("A"); envGet("B");\nstd::getenv("C");\n'},
         f"src/cli/main.cpp{T}3{T}C2\n", "block", 0)
    case("env snapshot file allowed", "env", {"src/config/env_snapshot.cpp": 'std::getenv("A");\n'}, None, "block", 0)
    # arch
    case("Arch:: branch blocked", "arch", {"src/model/config.cpp": "if (a == hip::Arch::gfx1151) {}\n"}, None, "block", 1)
    case("device layer allowed", "arch", {"include/whirl/device_profile.h": "Arch::gfx1151\n", "src/hip/h.cpp": "Arch::gfx1201\n"}, None, "block", 0)
    case("Arch::gfx in a comment is not code", "arch", {"src/a.cpp": "// Arch::gfx1151 only\n"}, None, "block", 0)
    # include
    inc_tree = {"cmake/server.cmake": CORE, "src/server/engine.cpp": '#include "server/engine.h"\n',
                "src/server/engine.h": '#include "whirl/model.h"\n', "src/server/http.cpp": '#include "whirl/common.h"\n',
                "include/whirl/model.h": "#pragma once\n", "include/whirl/common.h": "#pragma once\n"}
    case("server core reaching model.h through a header blocked", "include", inc_tree, None, "block", 1)
    case("pending covers it", "include", inc_tree, f"src/server/engine.cpp{T}1{T}M3\n", "block", 0)
    clean = dict(inc_tree)
    clean["src/server/engine.h"] = '#include "whirl/model_types.h"\n'
    clean["include/whirl/model_types.h"] = "#pragma once\n"
    case("after M3: pending entry STALE", "include", clean, f"src/server/engine.cpp{T}1{T}M3\n", "block", 1)
    case("after M3: clean", "include", clean, None, "block", 0)
    other = dict(inc_tree)
    other["src/server/backend_qwen35.cpp"] = '#include "whirl/model.h"\n'  # not a core source
    case("a non-core source may include model.h", "include", other, f"src/server/engine.cpp{T}1{T}M3\n", "block", 0)
    shutil.rmtree(base, ignore_errors=True)
    if fails:
        print(f"pending_scan selftest: FAIL ({fails} of {cases})")
        return 1
    print(f"pending_scan selftest: ok ({cases} cases)")
    return 0


def main(argv: list[str], rule_default: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rule", nargs="?", default=rule_default)
    ap.add_argument("--root", default=os.path.normpath(os.path.join(GATES, "..", "..")))
    ap.add_argument("--pending", default=None)
    ap.add_argument("--mode", choices=["report", "block"], default="report")
    ap.add_argument("--write-pending", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.rule:
        print("check: FAIL (usage: name a rule: " + ", ".join(RULES) + ")")
        return 2
    pending = a.pending if a.pending is not None else os.path.join(GATES, f"{a.rule}_pending.txt")
    if a.write_pending:
        return write_pending(a.rule, a.root, pending)
    return run(a.rule, a.root, pending, a.mode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
