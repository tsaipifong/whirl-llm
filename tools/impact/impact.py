# SPDX-License-Identifier: Apache-2.0
"""Given a git diff, list the product paths it touches and what to re-run (tests, gates, golden cases, release
cells). Reads tools/impact_map.json (tools/impact/impact_build.py); the functions a hunk lands in are found by
scanning the file at that revision, so it also works on old commits.

    python tools/impact/impact.py --commit SHA [--commit SHA ...]     one or more commits (their own diffs)
    python tools/impact/impact.py --range A..B                         everything between two revisions
    python tools/impact/impact.py --worktree | --staged                uncommitted changes
    options: --json (machine output)  --map PATH  --root REPO  --why (print the reason for every path)

Rules, most specific first:
  * a hunk inside a function -> that function's paths, models and tests (from the map; matched by file + name,
    else by name, else the file's paths)
  * a hunk outside any function in a header -> the paths of every file that includes the header
  * a .hip kernel file -> its archs (from the code-object guards), the host functions that dispatch its kernels
  * test / gate scripts, golden references, docs, CMake: their own rules (see FILE_RULES)
GPU scope: both GPUs unless every touched kernel / branch is arch-specific. Mode scope: all three modes always
(a change is proven mode-confined only by the golden hashes: precise must stay bitwise identical).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cxxscan  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# files that are not product code: (pattern, paths, extra re-runs, note)
FILE_RULES = [
    ("tests/golden/*", [], ["cpu_golden_xcheck", "golden(all, both GPUs)"], "golden references changed: the commit must say 'golden update: <reason>'"),
    ("tools/golden/*", [], ["cpu_golden_selftest", "cpu_golden_xcheck", "golden(one model, one GPU) to prove the runner"], "golden runner"),
    # arch025-n17: the module table and the iso manifests are build descriptions too: cpu_cmake_modules compares them with
    # the CMake targets (before the generic tools/gates/* and tests/* rules)
    ("tools/gates/modules.json", [], ["cpu_modules_check", "cpu_census", "cpu_cmake_modules"], "module table"),
    ("tests/*/isolation.txt", [], ["cpu_iso_<module>", "cpu_cmake_modules"], "isolation manifest"),
    ("tools/gates/*", [], ["release_gate.ps1 -List", "the changed step on one GPU"], "release gate script"),
    ("tools/klgate/*", [], ["tools/klgate/test_klgate_judge.py", "klgate one model one GPU"], "KL gate"),
    ("tools/impact/*", [], ["python tools/impact/impact_build.py --check", "python tools/impact/validate.py"], "impact map tooling"),
    ("tools/impact_map.json", [], [], "generated map"),
    ("tools/package_release.ps1", ["startup"], ["package smoke: unzip to a clean folder, run whirl devices + whirl-server --help"], "packaging"),
    ("redist/*", ["startup"], ["package smoke: unzip to a clean folder, run whirl devices + whirl-server --help"], "shipped DLLs"),
    ("tests/server/server_gate.cpp", [], ["cpu_gate_selftest", "the changed suite on one GPU"], "server gate"),
    ("tests/*", [], ["the test executable that changed"], "test code"),
    ("docs/*", [], [], "docs only"),
    ("*.md", [], [], "docs only"),
    ("CMakeLists.txt", ["build"], ["full build both archs", "release gate CPU steps"], "build"),
    ("cmake/*", ["build"], ["full build both archs", "release gate CPU steps"], "build"),
    ("build.bat", ["build"], ["full build"], "build"),
]
# identifiers on a changed declaration line that say nothing about who uses it
DECL_NOISE = {"const", "constexpr", "static", "inline", "struct", "class", "enum", "public", "private", "protected", "virtual",
              "override", "void", "bool", "int", "unsigned", "char", "float", "double", "auto", "std", "uint8_t", "uint16_t",
              "uint32_t", "uint64_t", "int32_t", "int64_t", "size_t", "string", "vector", "optional", "array", "true", "false",
              "nullptr", "return", "if", "else", "for", "namespace", "using", "typename", "template", "u32", "u64", "f32",
              "default", "delete", "noexcept", "explicit", "operator", "mutable", "volatile", "extern", "static_assert",
              "sizeof", "include", "define", "string_view", "map", "set", "pair", "function", "atomic", "mutex", "this", "whirl"}
PATH_GOLDEN = {  # path -> golden suites that exercise it (CLI suites from whirl golden, 'server' from whirl-server-gate --suite golden)
    "cli.prefill": ["single", "pair"], "cli.decode": ["single"], "cli.mtp": ["single", "pair", "ngram"], "cli.dflash": [],
    "server.prefill": ["server"], "server.decode": ["server"], "server.mtp": ["server"], "server.sampling": ["server"],
    "server.loop": ["server"], "cache.prefix": ["server"], "cache.restore": ["server"], "cache.spill": ["server"],
    "cache.fingerprint": ["server"], "vision": ["server"], "load": ["single", "pair", "ngram", "server"], "http": ["server"],
    "tokenizer": ["single", "server"], "startup": ["server"], "build": ["single", "pair", "ngram", "server"],
}


def git(*a, root=ROOT):
    r = subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"impact: git {' '.join(a)} failed: {r.stderr.strip()}")
    return r.stdout


def show(rev, path):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def parse_diff(diff):
    """unified diff (-U0) -> {path: {"new": [lines], "old": [lines], "old_path": p, "status": ...}}"""
    files = {}
    cur = None
    for l in diff.splitlines():
        if l.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)", l)
            cur = {"old_path": m.group(1), "new": [], "old": []}
            files[m.group(2)] = cur
        elif l.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", l)
            o, on, n, nn = int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
            cur["old"] += list(range(o, o + on))
            cur["new"] += list(range(n, n + nn))
    return files


def analyze(changes, m, new_rev, old_rev):
    """changes: parse_diff output. Returns the impact report dict."""
    F = m["functions"]
    by_name = defaultdict(list)
    for nid, d in F.items():
        by_name[d["name"]].append(nid)
    paths = defaultdict(set)    # path -> reasons
    models = set()
    archs = set()
    tests = set()
    reruns = defaultdict(set)   # extra -> reasons
    notes = []
    hits = []                   # (file, what)
    host_code = False
    # arch024-S12 (hold-out misses 1e18b8f, a5fe163, 65b84f6): a host function that dispatches kernels (a host_user of a
    # .hip file: it selects / launches them through KernelTable fields) is covered by those kernels' tests too. Before,
    # only a change to the .hip file itself listed them, so Model::matmul / moeGuR / precChoice changes missed
    # testGemv / testMoeMx / benchGemvh.
    dispatch_tests = defaultdict(set)
    for h in m["hip"].values():
        for u in h["host_users"]:
            dispatch_tests[u] |= set(h["tests"])
    for path, ch in sorted(changes.items()):
        rule = next((r for r in FILE_RULES if fnmatch.fnmatch(path, r[0])), None)
        if rule and not path.endswith((".cpp", ".h", ".hip")) or (rule and path.startswith(("tests/", "tools/"))):
            for p in rule[1]:
                paths[p].add(f"{path} ({rule[3]})")
            for x in rule[2]:
                reruns[x].add(path)
            hits.append((path, rule[3]))
            if path.startswith("tests/") and path.endswith(".cpp"):
                exe = [e for e, fl in m["executables"].items() if path in fl]
                for e in exe:
                    reruns[f"{e} (test source changed)"].add(path)
            continue
        if path.endswith(".hip"):
            h = m["hip"].get(path)
            if not h:
                archs |= {"gfx1201", "gfx1151"} if "/gfx1151/" not in path else {"gfx1151"}
                notes.append(f"{path}: not in the map (new file?): both GPUs, path load")
                paths["load"].add(path)
                continue
            archs |= set(h["arch"])
            models |= set(h["models"])
            for p in h["paths"]:
                paths[p].add(f"{path} (kernel; dispatched by {', '.join(h['host_users'][:4])}{'...' if len(h['host_users']) > 4 else ''})")
            tests |= set(h["tests"])
            hits.append((path, f"kernel file, arch {'/'.join(h['arch'])}"))
            continue
        if not path.endswith((".cpp", ".h", ".inc", ".in")):
            if path.startswith(("src/", "include/", "kernels/")):
                host_code = True
                notes.append(f"{path}: unrecognized source type, treated as build-wide")
                paths["build"].add(path)
            else:
                hits.append((path, "no rule (not product code)"))
            continue
        host_code = True
        archs |= {"gfx1201", "gfx1151"}
        touched = set()
        outside_ids = set()   # identifiers on changed lines outside any function (declarations, tables)
        outside_types = set()  # the class / struct bodies those lines are in (fallback when no function uses the ids)
        outside = False
        code_changed = False
        for rev, key in ((new_rev, "new"), (old_rev, "old")):
            lines = ch[key]
            if not lines:
                continue
            if rev:
                src = show(rev, path if key == "new" else ch["old_path"])
            elif key == "new" and os.path.exists(os.path.join(ROOT, path)):
                src = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
            else:
                src = None
            if src is None:
                continue
            sc = cxxscan.scan(src)
            clean = cxxscan.blank_code(src).split("\n")
            for ln in lines:
                txt = clean[ln - 1] if 0 < ln <= len(clean) else ""
                if not txt.strip():
                    continue  # comment / blank line: no code changed here
                code_changed = True
                f = cxxscan.func_at(sc, ln)
                if f is None:
                    outside = True
                    outside_ids |= {x for x in re.findall(r"[A-Za-z_]\w*", txt) if len(x) > 2} - DECL_NOISE
                    if key == "new":
                        outside_types |= set(cxxscan.classes_at(sc, ln)) - DECL_NOISE
                else:
                    touched.add(f.name)
        if not code_changed:
            hits.append((path, "comments / blank lines only: no product path"))
            continue
        for name in sorted(touched):
            nid = f"{path}::{name}"
            cands = [nid] if nid in F else by_name.get(name, [])
            if not cands:
                notes.append(f"{path}::{name}: function not in the map (new or renamed): using the file's paths")
                for p in m["files"].get(path, []):
                    paths[p].add(f"{path} (file of new function {name})")
                continue
            for c in cands:
                d = F[c]
                for p in d["paths"]:
                    paths[p].add(f"{name}")
                models |= set(d["models"])
                tests |= set(d["tests"])
                tests |= dispatch_tests.get(d["name"], set())
            hits.append((path, f"function {name}"))
        if outside:
            # declarations / tables: the functions (in this file and, for a header, in every includer) that use one of
            # the changed identifiers; if none is found, every function of those files
            scope = [path] + (m["includers"].get(path, []) if path.endswith(".h") else [])
            users = {}
            scans = {}
            for f in scope:
                src = show(new_rev, f) if new_rev else (open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
                                                        if os.path.exists(os.path.join(ROOT, f)) else None)
                if src is None:
                    continue
                scans[f] = cxxscan.scan(src)
                for fn in scans[f].funcs:
                    if fn.idents & outside_ids:
                        users[(f, fn.name)] = fn.idents & outside_ids
            # arch024-S12: a member added to a struct that no function reads yet (a test lands before its fix, e.g.
            # a4626b9 PinnedGpu::adapter_shared) changes every user of the struct: fall back to the enclosing types
            struct_fallback = False
            if not users and outside_types:
                struct_fallback = True
                for f, sc2 in scans.items():
                    for fn in sc2.funcs:
                        if fn.idents & outside_types:
                            users[(f, fn.name)] = fn.idents & outside_types
                if users:
                    notes.append(f"{path}: no function uses {', '.join(sorted(outside_ids)[:4])}: using the users of "
                                 f"{', '.join(sorted(outside_types))}")
            prod = [(f, n) for f, n in users if f"{f}::{n}" in F]
            # a test function that reads a changed declaration itself (e.g. a table test of a constexpr row) covers it
            # even when no product function's test list reaches it (ARCH3: device_profile.h field edits)
            for f, n in users:
                for exe, srcs in m["executables"].items():
                    if f in srcs and f"{exe}:{n}" in m["tests"]:
                        tests.add(f"{exe}:{n}")
            if prod:
                for f, n in sorted(prod):
                    d = F[f"{f}::{n}"]
                    for p in d["paths"]:
                        paths[p].add(f"{n} (uses {', '.join(sorted(users[(f, n)])[:3])} of {path})")
                    models |= set(d["models"])
                    tests |= set(d["tests"])
                    tests |= dispatch_tests.get(d["name"], set())
                hits.append((path, f"declarations ({', '.join(sorted(outside_ids)[:6])}{'...' if len(outside_ids) > 6 else ''}): "
                                   f"{len(prod)} function(s) in {len({f for f, _ in prod})} file(s) use them"))
            if not prod or struct_fallback:
                # no function uses the changed identifiers: every path of the file and its includers (fail closed).
                # The struct-user fallback above only ADDS tests / paths to this (review A1: a4626b9 lost 5 GPU steps
                # when it replaced it)
                for f in scope:
                    for p in m["files"].get(f, []):
                        paths[p].add(f"{path} (file-scope change; no user function found)")
                hits.append((path, f"file-scope change, no user function found: all paths of {len(scope)} file(s)"))
        if not touched and not outside:
            hits.append((path, "deleted / renamed file"))
            for p in m["files"].get(path, []):
                paths[p].add(f"{path} (file removed)")
    if not models:
        models = {"dense", "moe"}
    if not archs and paths:
        archs = {"gfx1201", "gfx1151"}
    rep = build_report(m, paths, models, archs, tests, reruns, notes, hits)
    pins = pins_files(changes, m)
    if pins:
        rep["release_gate_steps"] = sorted(set(rep["release_gate_steps"]) | {"cpu_pins"})
        rep["notes"].append(f"cpu_pins: {', '.join(pins[:4])}{' ...' if len(pins) > 4 else ''} linked by whirl-pins / whirl-startup-replay")
    sst, snotes = safety_steps(changes, new_rev, old_rev)
    if sst:
        rep["release_gate_steps"] = sorted(set(rep["release_gate_steps"]) | sst)
        rep["notes"] += snotes
    if changes:
        add_all_cpu(rep, m)
    rep.update(module_gate(changes, rep=rep, m=m))
    return rep


def all_cpu(m):
    """(every CPU release step, every CPU test per executable) of map m. arch025-srctext, owner 2026-10-28 (「98/103，沒選到的
    那5項，如果不耗什麼時，那乾脆都跑了吧?」): the CPU steps (P-cpu ~6-9 min; the C++ safety steps longer) always run in
    full, so a CPU test the map cannot reach (source-text tests such as testForwardSplit, destructors, data tables) is
    never left out. Only GPU steps are selected by the map. An old map without "cpu_steps" falls back to the cpu_* names
    it does have (release_gate.ps1 literals + the declared RELEASE_STEPS + one cpu_<exe> per test executable)."""
    steps = set(m.get("cpu_steps") or [])
    if not steps:
        steps = {s for s in m.get("release_gate_steps", []) if s.startswith("cpu_") and "<" not in s}
        steps |= {s for s in (m.get("declared", {}).get("release_steps") or {}) if s.startswith("cpu_")}
        for d in m.get("tests", {}).values():
            if not d.get("gpu") and d.get("exe", "").startswith("whirl-"):
                short = d["exe"].replace("whirl-", "", 1)
                short = short[:-len("-tests")] if short.endswith("-tests") else short
                steps.add(f"cpu_{short}")
    tests = defaultdict(set)
    for tid, d in m.get("tests", {}).items():
        if d.get("gpu") or d.get("exe") in ("whirl", "whirl-kernel-test"):
            continue
        exe, name = tid.split(":", 1)
        tests[exe].add(name)
    return steps, tests


def add_all_cpu(rep, m):
    """adds every CPU step and every CPU test to report rep (only ADDS). The map's own pick is kept in rep["map_steps"] /
    rep["map_cpu_tests"] (diagnostics: what the rules alone would select)."""
    steps, tests = all_cpu(m)
    rep["map_steps"] = list(rep["release_gate_steps"])
    rep["map_cpu_tests"] = {e: list(v) for e, v in rep["cpu_tests"].items()}
    rep["release_gate_steps"] = sorted(set(rep["release_gate_steps"]) | steps)
    cpu = {e: set(v) for e, v in rep["cpu_tests"].items()}
    for e, v in tests.items():
        cpu.setdefault(e, set()).update(v)
    rep["cpu_tests"] = {e: sorted(v) for e, v in sorted(cpu.items())}
    rep["cpu_all"] = True


# arch025-Q1 / Q3 (PLAN (e)): every C / C++ change -> cpu_analyze; a lifetime-class change -> cpu_asan; the safety tools themselves
# -> all of them. Rules in tools/gates/modules.json "safety" (cxx / lifetime / own).
def load_safety(root=None):
    p = os.path.join(root or ROOT, "tools", "gates", "modules.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        return d.get("safety"), d.get("modules") or {}
    except (OSError, ValueError):
        return None, {}


def changed_text(path, ch, new_rev, old_rev):
    """the changed lines (new and old side) of one file; None when a side cannot be read (callers fail closed)"""
    out = []
    for rev, key, p in ((new_rev, "new", path), (old_rev, "old", ch.get("old_path") or path)):
        if not ch.get(key):
            continue
        if rev:
            src = show(rev, p)
        else:
            try:
                src = open(os.path.join(ROOT, p), encoding="utf-8", errors="replace").read()
            except OSError:
                src = None
        if src is None:
            return None
        lines = src.splitlines()
        out += [lines[i - 1] for i in ch[key] if 0 < i <= len(lines)]
    return out


def safety_steps(changes, new_rev, old_rev, safety=None, mods=None):
    if safety is None:
        safety, mods = load_safety()
    if not safety:
        return set(), []
    mods = mods or {}
    steps, notes = set(), []
    cx, lt, own = safety.get("cxx", {}), safety.get("lifetime", {}), safety.get("own", {})
    th = safety.get("threads", {})  # arch025-Q4: cpu_tsa
    th_tok = re.compile(th.get("tokens", r"(?!)"))
    ext = tuple(cx.get("ext", []))
    tok = re.compile(lt.get("tokens", r"(?!)"))
    lt_globs = list(lt.get("globs", []))
    for n in lt.get("modules", []):
        lt_globs += list((mods.get(n) or {}).get("paths", []))
    why_a, why_l, why_o, why_t = [], [], [], []
    for path, ch in sorted(changes.items()):
        if any(fnmatch.fnmatch(path, g) for g in own.get("globs", [])):
            steps |= set(own.get("steps", []))
            why_o.append(path)
        is_cxx = path.endswith(ext) and any(fnmatch.fnmatch(path, g) for g in cx.get("globs", []))
        if not is_cxx:
            continue
        steps |= set(cx.get("steps", []))
        why_a.append(path)
        if th and any(fnmatch.fnmatch(path, g) for g in th.get("globs", [])):
            steps |= set(th.get("steps", []))
            why_t.append(f"{path} (compiled by the cpu_tsa build)")
        elif th:
            ttxt = changed_text(path, ch, new_rev, old_rev)
            thit = None if ttxt is None else sorted({mt.group(0).strip() for l in ttxt for mt in th_tok.finditer(l)})
            if thit is None or thit:
                steps |= set(th.get("steps", []))
                why_t.append(f"{path} ({'text not readable: fail closed' if thit is None else 'changed lines use ' + ', '.join(thit[:4])})")
        if any(fnmatch.fnmatch(path, g) for g in lt_globs):
            steps |= set(lt.get("steps", []))
            why_l.append(f"{path} (lifetime module / path)")
            continue
        txt = changed_text(path, ch, new_rev, old_rev)
        if txt is None:
            steps |= set(lt.get("steps", []))
            why_l.append(f"{path} (text not readable: fail closed)")
            continue
        hit = sorted({mt.group(0).strip() for l in txt for mt in tok.finditer(l)})
        if hit:
            steps |= set(lt.get("steps", []))
            why_l.append(f"{path} (changed lines use {', '.join(hit[:4])})")
    if why_a:
        notes.append(f"cpu_analyze: C / C++ change in {', '.join(why_a[:4])}{' ...' if len(why_a) > 4 else ''}")
    if why_l:
        notes.append(f"cpu_asan: lifetime-class change: {'; '.join(why_l[:4])}{' ...' if len(why_l) > 4 else ''}")
    if why_t:
        notes.append(f"cpu_tsa: lock / thread change: {'; '.join(why_t[:4])}{' ...' if len(why_t) > 4 else ''}")
    if why_o:
        notes.append(f"cpu_asan + cpu_analyze: safety tools changed ({', '.join(why_o[:4])})")
    return steps, notes


# arch024-S11c (F-3, S11b P1 / P4: template changes caught only by the prompt token pin, which no step ran): the CPU pins
# (tests/pins/check.ps1) cover every product source whirl-pins / whirl-startup-replay link (tokenizer, chat template,
# server engine, cache, tier, startup decisions) and every header such a source includes; tests/pins/* themselves too.
PIN_EXES = ("whirl-pins", "whirl-startup-replay")


def pins_files(changes, m):
    linked = set()
    for e in PIN_EXES:
        linked |= set(m.get("executables", {}).get(e, []))
    out = []
    for path in sorted(changes):
        if path.startswith("tests/pins/"):
            out.append(path)
            continue
        if not path.startswith(("src/", "include/")) or not path.endswith((".cpp", ".h", ".inc")):
            continue
        if path in linked or any(x in linked for x in m.get("includers", {}).get(path, [])):
            out.append(path)
    return out


def load_modules(root=None):
    """tools/gates/modules.json of the tree (arch024-S12), or None when the tree has none"""
    p = os.path.join(root or ROOT, "tools", "gates", "modules.json")
    try:
        return json.load(open(p, encoding="utf-8"))["modules"]
    except (OSError, ValueError, KeyError):
        return None


BUILD_WIDE = ("CMakeLists.txt", "cmake/*", "build.bat", "redist/*", "tools/package_release.ps1")  # review C6: full gate


def expand_steps(steps):
    """release-step names of a report as release_gate.ps1 names them (<mode> expanded; the gate has seq MoE precise /
    balance only; dfl_* are Q4_K_M compatibility steps, not product steps)"""
    out = set()
    for x in steps:
        x = x.split(" ")[0]
        if x.startswith("dfl_"):
            continue
        for md in (["precise", "balance", "fast"] if "<mode>" in x else [None]):
            y = x.replace("<mode>", md) if md else x
            if not (y.startswith("seq_moe_") and md == "fast"):
                out.add(y)
    return out


def module_steps(mods, n):
    d = mods[n]
    return set(d.get("steps") or []) | {p.get("step") for p in d.get("pending") or []}


def module_gate(changes, mods=None, rep=None, m=None):
    """Which modules (tools/gates/modules.json) the changed files belong to, and the dev-time module gate to run.
    Full gate: product code in no module, two modules, a contract module, a build-wide file (C6), or a header whose
    includers lie outside every module (rdeps, A4). A module with "callers": "rdeps" (M1 handles) is run with the modules
    of every file that includes it; "callers": [names] (M8 threads) with the named modules. Steps the impact map selects
    beyond the module's table (PLAN c-2, e.g. server.mtp -> srv_balance + srv_fast) are added as -Steps (A3).
    Dev-time fast feedback only until the owner signs off O1 / O2 (arch024 PLAN): releases run the full gate."""
    mods = load_modules() if mods is None else mods
    if mods is None:
        return {"modules": {}, "module_gate": None, "module_extra_steps": []}
    hit = defaultdict(list)
    loose, why = [], []

    def owner(path):
        return [n for n, d in mods.items() if any(fnmatch.fnmatch(path, g) for g in d.get("paths", []))]

    for path in sorted(changes):
        if any(fnmatch.fnmatch(path, g) for g in BUILD_WIDE):
            loose.append(path)
            why.append(f"{path} is build-wide")
            continue
        if not path.startswith(("src/", "include/", "kernels/")):
            continue
        ms = owner(path)
        for n in ms:
            hit[n].append(path)
        if not ms:
            loose.append(path)
    # a module's `when` file (e.g. prefill_schedule.h for M3 / M4, PLAN C6) adds steps instead of forcing the full gate
    loose = [f for f in loose if not any(f in (mods[n].get("when") or {}) for n in hit)]
    callers = defaultdict(set)  # module -> modules of its callers
    for n in list(hit):
        c = mods[n].get("callers")
        if c == "rdeps":
            for f in hit[n]:
                inc = [x for x in ((m or {}).get("includers", {}).get(f, [])) if x.startswith(("src/", "include/", "kernels/"))]
                if m is None:
                    why.append(f"{n}: callers by includers need the impact map")
                    loose.append(f)
                for x in inc:
                    om = [k for k in owner(x) if k != n]
                    if om:
                        callers[n] |= set(om)
                    elif not owner(x):
                        loose.append(x)
                        why.append(f"{x} includes {f} and is in no module")
        elif isinstance(c, list):
            callers[n] |= set(c)
    when = sorted({st for n in hit for f, sts in (mods[n].get("when") or {}).items() if f in changes for st in sts})
    full = [n for n in hit if mods[n].get("full")] + [k for n in callers for k in callers[n] if mods.get(k, {}).get("full")]
    direct = sorted(hit)
    extra = []
    if not hit and not loose:
        gate = "none (no product code)"
    elif full:
        gate = f"full gate, both GPUs (contract module {', '.join(sorted(set(full)))})"
    elif loose:
        lz = sorted(set(loose))
        gate = (f"full gate or the release steps above: {len(lz)} file(s) in no module ({', '.join(lz[:4])}{' ...' if len(lz) > 4 else ''})"
                + (f"; {why[0]}" if why else ""))
    elif len(direct) > 1:
        gate = f"full gate, both GPUs (cross-module: {', '.join(direct)})"
    else:
        n = direct[0]
        run = [n] + sorted(callers[n] - {n})
        have = set(when)
        for k in run:
            have |= module_steps(mods, k)
        extra = sorted(expand_steps((rep or {}).get("release_gate_steps", [])) - have)
        add = when + extra
        gate = f"release_gate.ps1 -Module {','.join(run)}" + (f" -Steps {','.join(add)}" if add else "")
        # review K1: the map's steps are for every GPU it lists (both, unless the change is arch-specific)
        two = [x for x in add if not x.startswith("cpu_")]
        if two and len((rep or {}).get("gpus", ["gfx1201", "gfx1151"])) > 1:
            gate += f" [both GPUs, one after the other: {','.join(two)}]"
    return {"modules": {n: v for n, v in sorted(hit.items())}, "module_gate": gate, "module_extra_steps": extra,
            "module_callers": {n: sorted(v) for n, v in callers.items()}}


# server-gate suites that are flake-repro tools with no release step by design (progress cbflake.HANDOFF: sys_burst,
# http_stress, sys_d; sys_d runs 30 reps by default). Every path they cover is also covered by a gated suite (basic: prefix,
# loop, http; mt_cache_precise / tier / sys: restore), so a change that selects them selects those steps too. arch025-Q6:
# reported as "repro", not as a GAP.
REPRO_SUITES = {"sys_burst", "sys_d", "http_stress"}


def build_report(m, paths, models, archs, tests, reruns, notes, hits):
    decl = m["declared"]
    P = sorted(paths)
    # tests grouped by executable
    cpu = defaultdict(list)
    gpu_cli = set()
    kt = set()
    for t in sorted(tests):
        exe, name = t.split(":", 1)
        if exe == "gpu":
            gpu_cli.add(name)
        elif exe == "whirl-kernel-test":
            kt.add(name)
        else:
            cpu[exe].append(name)
    # server-gate suites from the paths
    suites = sorted({s for s, ps in decl["gate_suites"].items() for p in P if p in ps})
    # golden cases
    gsuites = sorted({g for p in P for g in PATH_GOLDEN.get(p, [])} | {n.split(".", 1)[1] for n in gpu_cli if n.startswith("golden.")})
    golden = []
    if gsuites:
        golden = [f"{a}: models {','.join(sorted(models))} x modes precise,balance,fast x suites {','.join(gsuites)}" for a in sorted(archs)]
    # release gate steps
    steps = set()
    for exe in cpu:
        short = exe.replace("whirl-", "").replace("-tests", "")
        if short.startswith("iso-"):
            # arch025-isoall: whirl-iso-<module>-tests runs in release_gate step cpu_iso_<module> (tools/gates/iso_check.py)
            short = short.replace("-", "_")
        steps.add({"tests": "cpu_tests"}.get(short, f"cpu_{short}"))
    if kt:
        steps.add("kt_full")
    for n in gpu_cli:
        if n.startswith("golden."):
            steps |= {f"golden_{'swift' if md == 'dense' else 'moe'}" for md in models}
        elif n == "seqtest":
            steps |= {f"seq_{'swift' if md == 'dense' else 'moe'}_<mode>" for md in models}
        elif n == "selftest":
            steps |= {f"st_{'swift' if md == 'dense' else 'moe'}_<mode>" for md in models}
        elif n == "dflash_bench":
            steps.add("dfl_<mode> (R9700, Q4_K_M compatibility only)")
    not_in_gate, repro = [], []
    for s in suites:
        hit = [st for st, d in m["release_srv_steps"].items() if s in d["suites"]]
        if s == "golden":  # suite server of golden.ps1 (default suites, ARCH2): run by the golden_<model> steps
            hit += [f"golden_{'swift' if md == 'dense' else 'moe'}" for md in models]
        steps |= set(hit)
        if not hit:
            (repro if s in REPRO_SUITES else not_in_gate).append(s)
    # matrix cells
    cells = set()
    for p in P:
        metric = decl["paths"].get(p, {}).get("metric")
        for met in decl["metric_cells"].get(str(metric), []):
            for a in sorted(archs):
                g = decl["gpus"][a]
                if met == "agent":
                    cells.add(f"{g}.agent.*.wall.whirl")
                    continue
                for md in sorted(models):
                    cells.add(f"{g}.{decl['models'][md]}.{met}.*.whirl.*")
    return {
        "paths": {p: sorted(paths[p])[:6] for p in P},
        "gpus": sorted(archs), "models": sorted(models), "modes": decl["modes"] if P else [],
        "cpu_tests": {e: v for e, v in sorted(cpu.items())},
        "kernel_tests": sorted(kt),
        "gpu_cli_gates": sorted(gpu_cli),
        "server_gate_suites": suites,
        "golden": golden,
        "release_gate_steps": sorted(steps),
        "not_in_release_gate": not_in_gate,
        "repro_suites": repro,
        "extra": {k: sorted(v) for k, v in sorted(reruns.items())},
        "matrix_cells": sorted(cells),
        "touched": hits,
        "notes": notes,
    }


def print_report(title, r, why=False):
    print(f"== {title}")
    for f, w in r["touched"]:
        print(f"   touched  {f}: {w}")
    if not r["paths"]:
        print("   paths    none (no product path)")
    for p, why_l in r["paths"].items():
        print(f"   path     {p}" + (f"   <- {'; '.join(why_l)}" if why else ""))
    if r["paths"]:
        print(f"   scope    GPUs {','.join(r['gpus'])} | models {','.join(r['models'])} | modes {','.join(r['modes'])}")
    if r.get("cpu_all"):
        ncs = len([s for s in r["release_gate_steps"] if s.startswith("cpu_")])
        print(f"   cpu      ALL {ncs} CPU release steps and all {sum(len(v) for v in r['cpu_tests'].values())} CPU tests "
              f"({len(r['cpu_tests'])} executables) run on every change (owner 2026-10-28); only GPU steps are map-selected")
    for e, v in (r.get("map_cpu_tests") if r.get("cpu_all") else r["cpu_tests"]).items():
        print(f"   cpu      {e}: {len(v)} test(s){' the map points at' if r.get('cpu_all') else ''}: {', '.join(v[:8])}{' ...' if len(v) > 8 else ''}")
    if r["kernel_tests"]:
        print(f"   kernel   whirl-kernel-test: {', '.join(r['kernel_tests'][:10])}{' ...' if len(r['kernel_tests']) > 10 else ''}")
    if r["gpu_cli_gates"]:
        print(f"   gpu-cli  {', '.join(r['gpu_cli_gates'])}")
    if r["server_gate_suites"]:
        print(f"   srv-gate {', '.join(r['server_gate_suites'])}")
    for g in r["golden"]:
        print(f"   golden   {g}")
    if r["release_gate_steps"]:
        print(f"   release  {', '.join(r['release_gate_steps'])}")
    if r["not_in_release_gate"]:
        print(f"   GAP      server-gate suites with no release_gate.ps1 step: {', '.join(r['not_in_release_gate'])}")
    if r.get("repro_suites"):
        print(f"   repro    flake-repro server-gate suites, no release step by design (their paths are in the steps above): {', '.join(r['repro_suites'])}")
    for k, v in r["extra"].items():
        print(f"   also     {k}  ({', '.join(v[:3])})")
    if r["matrix_cells"]:
        print(f"   cells    {', '.join(r['matrix_cells'])}")
    if r.get("module_gate"):
        for n, fl in r["modules"].items():
            print(f"   module   {n}: {', '.join(fl[:4])}{' ...' if len(fl) > 4 else ''}")
        for n, cl in (r.get("module_callers") or {}).items():
            print(f"   callers  {n}: run with {', '.join(cl)}")
        if r.get("module_extra_steps"):
            print(f"   mod-add  steps the map selects beyond the module table (PLAN c-2): {', '.join(r['module_extra_steps'])}")
        print(f"   mod-gate {r['module_gate']}  (dev-time only until owner sign-off O1/O2; releases run the full gate)")
    for n in r["notes"]:
        print(f"   note     {n}")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="append", default=[])
    ap.add_argument("--range")
    ap.add_argument("--worktree", action="store_true")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--why", action="store_true")
    ap.add_argument("--map")
    ap.add_argument("--root")
    a = ap.parse_args()
    if a.root:
        ROOT = a.root
    mp = a.map or os.path.join(ROOT, "tools", "impact_map.json")
    import impact_build
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except (OSError, ValueError):
        m = None
    if a.map is None and (m is None or m.get("tree_hash") != impact_build.tree_hash(ROOT, impact_build.list_files(ROOT))):
        print(f"impact: {mp} is missing or stale for this tree: rebuilding it in memory (tools/impact/impact_build.py writes it)",
              file=sys.stderr)
        m = impact_build.build(ROOT)
    jobs = []
    for c in a.commit:
        jobs.append((f"commit {git('log', '-1', '--format=%h %s', c).strip()}", git("diff", "-U0", "--no-color", "-M", f"{c}^", c), c, f"{c}^"))
    if a.range:
        x, y = a.range.split("..")
        base = git("merge-base", x, y).strip() or x  # A..B = what B adds since it left A (not the reverse of commits A gained)
        jobs.append((f"range {a.range} (from merge-base {base[:9]})", git("diff", "-U0", "--no-color", "-M", base, y), y, base))
    if a.staged:
        jobs.append(("staged", git("diff", "-U0", "--no-color", "-M", "--cached"), None, "HEAD"))
    if a.worktree:
        jobs.append(("worktree", git("diff", "-U0", "--no-color", "-M", "HEAD"), None, "HEAD"))
    if not jobs:
        ap.error("give --commit, --range, --staged or --worktree")
    out = {}
    for title, diff, new_rev, old_rev in jobs:
        r = analyze(parse_diff(diff), m, new_rev, old_rev)
        out[title] = r
        if not a.json:
            print_report(title, r, a.why)
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
