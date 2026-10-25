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
import declscan  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# files that are not product code: (pattern, paths, extra re-runs, note)
FILE_RULES = [
    ("tests/golden/*", [], ["cpu_golden_xcheck", "golden(all, both GPUs)"], "golden references changed: the commit must say 'golden update: <reason>'"),
    ("tools/golden/*", [], ["cpu_golden_selftest", "cpu_golden_xcheck", "golden(one model, one GPU) to prove the runner"], "golden runner"),
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
    k = (ROOT, rev, path)
    if k not in _show_cache:
        if len(_show_cache) > 4000:
            _show_cache.clear()
        r = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True)
        _show_cache[k] = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
    return _show_cache[k]


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


def analyze(changes, m, new_rev, old_rev, legacy=()):
    """changes: parse_diff output. Returns the impact report dict.
    legacy (validation only, review A4): "ids" = the 0.2.4 rule for declaration lines (every identifier on the line selects
    the functions naming it; removed declarations fall back), "moved" = no moved-declaration rule. validate.py runs both
    to attribute every GPU step the P1 rules drop to the rule that dropped it."""
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
    # measure-tools P1: what each changed C++ file changes (functions, declarations, includes, conditions), parsed
    # once; the declaration statements of both sides (sigs) tell a moved declaration from a changed one
    FC = {p: file_changes(p, c, new_rev, old_rev) for p, c in changes.items()
          if p.endswith((".cpp", ".h", ".inc", ".in")) and not (p.startswith(("tests/", "tools/")))}
    sigs = {"new": defaultdict(set), "old": defaultdict(set)}  # side -> statement signature -> files
    for fc in FC.values():
        for x in fc["decls"]:
            if x["sig"]:
                sigs[x["side"]][x["sig"]].add(x["file"])
    beh = defaultdict(set)      # behaviour class -> reasons (B1.3)
    decl_rows = []              # declared names per changed declaration line (B1.1)
    defects = []                # map defects (B1.4)
    dropped_ids = {}            # review A4: identifier -> kind (type / param / init / qualifier / other) the new rule dropped
    test_steps = set()          # release steps of test executables whose sources changed
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
                    # measure-tools P1 (map defect found on 605281b..9844fde: tier_tests.cpp changed, cpu_tier not
                    # selected): the release step that runs the changed test executable
                    st = next((k for k, v in (m["declared"].get("release_steps") or {}).items() if v == f"exe:{e}"), None)
                    if st:
                        test_steps.add(st)
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
        fc = FC[path]
        touched = fc["touched"]
        outside = fc["outside"]
        outside_types = fc["types_at"]  # the class / struct bodies the changed lines are in (fallback when no function uses the names)
        for t, why_b in fc["behaviour"]:
            beh[t].add(why_b)
        if not fc["code"]:
            hits.append((path, "comments / blank lines only: no product path"))
            continue
        # measure-tools P1 (B1.1): a declaration selects the users of the DECLARED names, not of every identifier on
        # the line (type names such as DevBuf / EventH are type_changed information). A declaration that appears
        # unchanged on the other side of the diff (moved between headers, re-indented) selects nothing.
        # mt-p1 review A1: an identical statement on the other side is a MOVE only when it is in another file, or in
        # the same file with the order of its class / enum unchanged (a diff-alignment edit); a reorder inside a struct
        # or enum changes the layout / the enumerator values: its declarations stay, and the file fails closed
        order_changed, shifted = scope_changes(fc, ch)
        pair = {path, ch["old_path"]}

        def is_moved(x):
            fs = sigs["old" if x["side"] == "new" else "new"].get(x["sig"]) if x["sig"] else None
            if not fs:
                return False
            if any(f not in pair for f in fs):
                return True
            return x["cls"] is None or x["cls"] not in order_changed
        decls = [x for x in fc["decls"] if "moved" in legacy or not is_moved(x)]
        moved = len(fc["decls"]) - len(decls)
        member_names = {(x["cls"], x["name"]) for x in decls if x["cls"] and x["what"] in ("member", "function", "alias")}
        free_names = {x["name"] for x in decls if (x["cls"], x["name"]) not in member_names}
        outside_ids = free_names | {n for _, n in member_names}
        # review A4: the identifiers the 0.2.4 rule took from these lines that the declared-name rule drops, by kind
        declared_all = {x["name"] for x in fc["decls"]}
        for side, ln, txt, whats in fc["legacy"]:
            for x, c in id_categories(txt, declared_all, in_function="function" in whats).items():
                if dropped_ids.get(x) != "other":
                    dropped_ids[x] = c
        if "ids" in legacy:
            member_names = set()
            free_names = {x for _, _, txt, _ in fc["legacy"] for x in re.findall(r"[A-Za-z_]\w*", txt) if len(x) > 2} - DECL_NOISE
            outside_ids = set(free_names)
        for x in decls:
            decl_rows.append({"file": path, "side": x["side"], "line": x["line"], "name": (x["cls"] + "::" if x["cls"] else "") + x["name"],
                              "what": x["what"], "types": x["types"]})
        if moved:
            notes.append(f"{path}: {moved} declaration(s) unchanged on the other side of the diff (moved / re-indented): build only")
        for side, ln, tg in fc["includes"]:
            reruns[f"build: compile {path} and every file that includes it (#include changed)"].add(f"{path}:{ln} {tg}")
        fail_closed = []
        for side, ln, k in fc["cond"]:
            fail_closed.append(f"{side}:{ln} {'preprocessor condition' if k == 'cond' else 'using namespace'}")
        for side, ln, txt in fc["unparsed"]:
            fail_closed.append(f"{side}:{ln} unparsed")
            defects.append(f"{path}:{ln} ({side}): declaration line the parser does not understand: {txt}")
        for side, nb in fc["unbalanced"]:
            fail_closed.append(f"{side}: braces / namespace lines changed without their block ({nb})")
        if "braces" in legacy and fc.get("brace_lines"):
            fail_closed.append("legacy: brace / namespace lines (0.2.4: no identifier -> file scope)")
        for c in sorted(order_changed):
            fail_closed.append(f"declaration order inside {c} changed (layout / enumerator values)")
            beh["layout"].add(f"{c}: declaration order changed")
        for c in sorted(shifted - order_changed):
            fail_closed.append(f"enumerator values of {c} shift (enumerator added / removed before others)")
        for c in sorted(shifted | {c for c in order_changed if c in fc.get("enums", set())}):
            beh["numeric"].add(f"{c}: enumerator values / order changed")
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
        if outside and (decls or fail_closed or ("ids" in legacy and free_names)):
            # declarations / tables: the functions (in this file and, for a header, in every includer) that use one of
            # the declared names; if none is found, every function of those files
            scope = [path] + (m["includers"].get(path, []) if path.endswith(".h") else [])
            users = {}
            scans = {}
            fsrc = {}
            for f in scope:
                src = show(new_rev, f) if new_rev else (open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
                                                        if os.path.exists(os.path.join(ROOT, f)) else None)
                if src is None:
                    continue
                scans[f] = cxxscan.scan(src)
                fsrc[f] = src
            # mt-p1 review C2 (R7, R8): a changed constant / macro also changes the macros defined with it and the
            # classes whose member declarations name it (array bound, default member value): their users too
            via = indirect_names({x["name"] for x in decls if x["cls"] is None and x["what"] in ("variable", "macro")}, fsrc, scans)
            if via:
                notes.append(f"{path}: also the users of {', '.join(sorted(via)[:6])} (macros / classes defined with the changed names)")
                free_names = free_names | via
            derived = derived_classes({c for c, _ in member_names}, fsrc.values())
            known_cls = {c for sc2 in scans.values() for c, _, _ in sc2.classes}
            for f, sc2 in scans.items():
                for fn in sc2.funcs:
                    u = decl_users(fn, member_names, free_names, derived, known_cls)
                    if u:
                        users[(f, fn.name)] = u
            # arch024-S12: a member added to a struct that no function reads yet (a test lands before its fix, e.g.
            # a4626b9 PinnedGpu::adapter_shared) changes every user of the struct: fall back to the enclosing types
            struct_fallback = False
            outside_types = outside_types | {c for c, _ in member_names}
            # a declaration removed with all its users (old side only, no user left in the new tree): the changed users
            # are touched functions already; nothing else can use it (the build would fail), so no fallback (S7 kNoTag)
            # mt-p1 review A2: only namespace-scope functions / variables / aliases / macros; a removed member or
            # enumerator changes sizeof / layout / later enumerator values even with no user left: it stays live
            new_names = {(x["cls"], x["name"]) for x in decls if x["side"] == "new"}
            live = [x for x in decls if "ids" in legacy or x["side"] == "new" or (x["cls"], x["name"]) in new_names
                    or x["cls"] is not None or x["what"] not in ("function", "variable", "alias", "macro")]
            if decls and not live and not users:
                notes.append(f"{path}: removed declaration(s) {', '.join(sorted({x['name'] for x in decls})[:4])} with no user left: build only")
            if live and not users and outside_types:
                struct_fallback = True
                for f, sc2 in scans.items():
                    for fn in sc2.funcs:
                        q = fn.name.split("::")
                        own = {q[-2]} & outside_types if len(q) >= 2 else set()  # P1: member functions of the type too
                        if fn.idents & outside_types or own:
                            users[(f, fn.name)] = (fn.idents & outside_types) | own
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
            if (live and (not prod or struct_fallback)) or fail_closed:
                # no function uses the declared names (or a line the rule cannot place: preprocessor condition,
                # unparsed declaration, braces without their block): every path of the file and its includers (fail
                # closed). The struct-user fallback above only ADDS tests / paths to this (review A1: a4626b9 lost 5 GPU
                # steps when it replaced it). measure-tools P1 (B1.4): reported as a map defect when it is a gap of the
                # map (no user found / unparsed), as a note when the rule is conservative by design (#if, using namespace)
                why_fc = "; ".join(fail_closed[:3]) if fail_closed else "no user function found"
                for f in scope:
                    for p in m["files"].get(f, []):
                        paths[p].add(f"{path} (file-scope change; {why_fc})")
                hits.append((path, f"file-scope change ({why_fc}): all paths of {len(scope)} file(s)"))
                if live and (not prod or struct_fallback):
                    defects.append(f"{path}: declared {', '.join(sorted(outside_ids)[:4])}{' ...' if len(outside_ids) > 4 else ''} "
                                   f"has no user function in the map: file-scope fallback over {len(scope)} file(s)")
        elif outside and not decls and not fail_closed and fc["includes"]:
            hits.append((path, f"#include lines only ({len(fc['includes'])}): build only"))
        elif outside and not decls and not fail_closed and moved:
            hits.append((path, f"{moved} declaration(s) moved unchanged: build only"))
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
    if test_steps:
        rep["release_gate_steps"] = sorted(set(rep["release_gate_steps"]) | test_steps)
    behaviour_classes(changes, rep, beh, decl_rows, new_rev)
    rep["declarations"] = decl_rows
    rep["dropped_ids"] = dropped_ids
    rep["map_defects"] = defects
    rep.update(module_gate(changes, rep=rep, m=m))
    return rep


_show_cache: dict = {}


def file_changes(path, ch, new_rev, old_rev):
    """measure-tools P1: one changed C++ file, both sides: touched functions, declarations outside functions (declscan:
    declared name, class, type names, statement), #include lines, preprocessor conditions, unparsed lines, behaviour
    hints from the declarations (layout / lifetime)."""
    out = {"touched": set(), "decls": [], "includes": [], "cond": [], "unparsed": [], "unbalanced": [], "code": False,
           "outside": False, "types_at": set(), "behaviour": [], "src": {}, "legacy": []}
    for rev, key in ((new_rev, "new"), (old_rev, "old")):
        lines = ch[key]
        if rev:
            src = show(rev, path if key == "new" else ch["old_path"])
        elif key == "new" and os.path.exists(os.path.join(ROOT, path)):
            src = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
        else:
            src = None
        if src is None:
            continue
        out["src"][key] = src  # both sides, also the side without changed lines (review A1: declaration order)
        if not lines:
            continue
        sc = cxxscan.scan(src)
        clean = cxxscan.blank_code(src).split("\n")
        raw = src.split("\n")
        lone = []  # changed brace lines whose partner brace line is unchanged
        lset = set(lines)
        for ln in lines:
            txt = clean[ln - 1] if 0 < ln <= len(clean) else ""
            f = cxxscan.func_at(sc, ln)
            d = declscan.declared(src, ln) if (f is None or not txt.strip()) else None
            if d is not None and d["kind"] == "pragma":
                continue
            if f is not None:
                if txt.strip() or (d and d["kind"] != "punct"):  # code, or a preprocessor line inside the function
                    out["code"] = True
                    out["touched"].add(f.name)
                continue
            k = d["kind"]
            if k == "punct" and not txt.strip():
                continue  # comment / blank line: no code changed here
            out["code"] = True
            out["outside"] = True
            out["legacy"].append((key, ln, txt, {x["what"] for x in d["decls"]}))
            if k == "punct" and ("{" in txt or "}" in txt or re.search(r"\bnamespace\b", txt)):
                out["brace_lines"] = out.get("brace_lines", 0) + 1
            if key == "new":
                out["types_at"] |= set(cxxscan.classes_at(sc, ln)) - DECL_NOISE
            if k == "punct":
                # a lone '}' / namespace '{' moves a block edge over unchanged lines (a type opener names its type)
                if ("{" in txt or "}" in txt) and any(q is None or q not in lset for q in declscan.brace_partners(src, ln)):
                    lone.append(ln)
                continue
            if k in ("decl", "enumerator", "macro"):
                for x in d["decls"]:
                    out["decls"].append(dict(x, side=key, line=ln, file=path if key == "new" else ch["old_path"]))
            elif k == "include":
                out["includes"].append((key, ln, d["target"]))
            elif k in ("cond", "scope"):
                out["cond"].append((key, ln, k))
            elif k == "unparsed":
                out["unparsed"].append((key, ln, (raw[ln - 1] if ln <= len(raw) else "").strip()[:80]))
        if lone:
            out["unbalanced"].append((key, f"line(s) {', '.join(map(str, lone[:4]))}"))
    # layout / lifetime hints: a member present on one side only, or whose declared type changed
    sides = defaultdict(dict)
    for x in out["decls"]:
        if x["cls"] and x["what"] == "member":
            sides[(x["cls"], x["name"])][x["side"]] = tuple(x["types"])
    for (c, n), sd in sorted(sides.items()):
        if len(sd) == 1:
            out["behaviour"].append(("layout", f"{c}::{n} {'added' if 'new' in sd else 'removed'}"))
        elif sd.get("new") != sd.get("old"):
            tt = set(sd.get("new", ())) | set(sd.get("old", ()))
            out["behaviour"].append(("lifetime" if any(HANDLE_TYPES.search(t) for t in tt) else "layout",
                                     f"{c}::{n} type {'/'.join(sd.get('old', ())) or '-'} -> {'/'.join(sd.get('new', ())) or '-'}"))
    for x in out["decls"]:
        if x["name"].startswith("~"):
            out["behaviour"].append(("lifetime", f"{x['cls']}::{x['name']} declaration"))
    return out


def id_categories(text, declared, in_function=False):
    """review A4: the identifiers of a declaration line the 0.2.4 rule used (len > 2, not DECL_NOISE) minus the declared
    names, each with its kind: init (after the top-level '='), qualifier (before '::'), param (inside top-level
    parentheses: parameter names and types), type (template arguments, decltype, the rest of the declaration's type),
    other (in a declarator position, i.e. followed by = ; { [ ( , : at top level, yet not declared: the parser may have
    missed the declared name; never accepted without a manual check)"""
    if in_function:  # a line of a function declaration (also a continuation line): parameters, their types, defaults
        return {x: "param" for x in re.findall(r"[A-Za-z_]\w*", text) if len(x) > 2 and x not in DECL_NOISE and x not in declared}
    if re.match(r"\s*(?:inline\s+)?namespace\b", text):
        return {x: "qualifier" for x in re.findall(r"[A-Za-z_]\w*", text) if len(x) > 2 and x not in DECL_NOISE and x not in declared}
    t = declscan._mask(text)
    eq = declscan._top(t, "=")
    cut = eq[0] if eq else len(text)
    depth, d = [], 0
    for c in text:
        if c == "(":
            d += 1
        depth.append(d)
        if c == ")":
            d -= 1
    rank = {"other": 4, "type": 3, "param": 2, "qualifier": 1, "init": 0}
    out = {}
    for mm in re.finditer(r"[A-Za-z_]\w*", text):
        x = mm.group(0)
        if len(x) <= 2 or x in DECL_NOISE or x in declared:
            continue
        p = mm.start()
        rest = text[mm.end():].lstrip()
        if p >= cut:
            c = "init"
        elif rest.startswith("::"):
            c = "qualifier"
        elif depth[p] > 0:
            c = "param"
        elif t[p] == " ":
            c = "type"
        elif rest[:1] in ("=", ";", "{", "[", ",", ":", "(", ""):
            c = "other"
        else:
            c = "type"
        if x not in out or rank[c] > rank[out[x]]:
            out[x] = c
    return out


def indirect_names(names, fsrc, scans):
    """macros whose body names one of names (transitively), and classes whose body (outside member function bodies)
    names one of them (review C2: R8 TILE2 = TILE * 2, R7 float v[kSlots])"""
    if not names:
        return set()
    macros = {}
    for src in fsrc.values():
        for mm in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)((?:[^\n]*\\\n)*[^\n]*)", src, re.M):
            macros[mm.group(1)] = set(re.findall(r"[A-Za-z_]\w*", mm.group(2)))
    got, todo = set(names), list(names)
    while todo:
        n = todo.pop()
        for mname, ids in macros.items():
            if n in ids and mname not in got:
                got.add(mname)
                todo.append(mname)
    for f, sc in scans.items():
        lines = fsrc[f].split("\n")
        for cname, a, b in sc.classes:
            body = set()
            for ln in range(a, min(b, len(lines)) + 1):
                if not cxxscan.func_at(sc, ln):
                    body |= set(re.findall(r"[A-Za-z_]\w*", lines[ln - 1].split("//")[0]))
            if body & got and cname not in got:
                got.add(cname)
    return got - set(names)


def scope_changes(fc, ch):
    """mt-p1 review A1 / C2: (classes / enums of the changed declarations whose declaration order differs between the
    two sides, enums whose common enumerators sit at other positions: values shift)"""
    order_changed, shifted = set(), set()
    enums = {x["cls"] for x in fc["decls"] if x["what"] == "enumerator" and x["cls"]}
    fc["enums"] = enums
    a, b = fc["src"].get("old"), fc["src"].get("new")
    if a is None or b is None:
        return order_changed, shifted
    for c in sorted({x["cls"] for x in fc["decls"] if x["cls"]}):
        oa, ob = declscan.scope_order(a, c), declscan.scope_order(b, c)
        sa, sb = set(oa), set(ob)
        if [n for n in oa if n in sb] != [n for n in ob if n in sa]:
            order_changed.add(c)
        elif c in enums and any(oa.index(n) != ob.index(n) for n in sa & sb):
            shifted.add(c)
    return order_changed, shifted


HANDLE_TYPES = re.compile(r"^(DevBuf|EventH|StreamH|PinnedBuf|unique_ptr|shared_ptr|\w*(Owner|Guard|Handle|Lock)\w*)$")


def derived_classes(bases, sources):
    """class -> the classes in sources that derive from it (directly or not; `struct X : public Y {`)"""
    edges = defaultdict(set)
    for t in sources:
        for mm in re.finditer(r"\b(?:class|struct)\s+(\w+)\s*(?:final\s*)?:\s*([^{;()]+)\{", t):
            for b in re.findall(r"(\w+)\s*(?:<[^<>]*>)?\s*(?:,|$)", mm.group(2).strip()):
                edges[b].add(mm.group(1))
    out = {}
    for c in bases:
        seen, todo = set(), [c]
        while todo:
            for y in edges.get(todo.pop(), ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        out[c] = seen
    return out


def decl_users(fn, member_names, free_names, derived, known_cls):
    """measure-tools P1 (B1.1): the declared names fn uses. A member C::n: accessed as .n / ->n, or named inside a member
    function of C (or of a class derived from C), or C and n both named (C::n, designated initializers), or fn is the
    definition C::n itself. A free name n: named in fn, or fn is its (namespace-scope) definition."""
    got = set()
    q = fn.name.split("::")
    fcls = q[-2] if len(q) >= 2 else None
    for c, n in member_names:
        if n in fn.members or (fcls is not None and (fcls == c or fcls in derived.get(c, ())) and (n in fn.idents or fn.short == n)) \
                or (c in fn.idents and n in fn.idents):
            got.add(f"{c}::{n}")
    for n in free_names:
        if n in fn.idents or (fn.short == n and (fcls is None or fcls not in known_cls)):
            got.add(n)
    return got


# ------------------------------------------------------------------------------------------------
# measure-tools P1 (B1.3): behaviour classes. A FLOOR (hipfire ar/gate/routing.py): they add measurement kinds and
# the GPU checks that live outside release_gate (arch024 blast GAP list: E2, kvram719, visab, dense/dfl, E6, version),
# and the release steps srv_gpu_select / srv_env_* that no path selected before (s15-runner.HANDOFF.md:19). They never
# remove a step.
BEHAVIOUR = {
    "lifetime": {"measure": ["leak / pending-free tests (NullBackend tracking: cpu_device, cpu_server)", "ASan build of the CPU tests",
                             "A3 alloc churn (whirl-host-bench, P3)"],
                 "speed": "none when the work traces are equal (B3 / ADR-0003)"},
    "layout": {"measure": ["CPU tests of the users", "golden of the touched paths (bitwise)"]},
    "dispatch": {"measure": ["A1 kernel cells of the touched kernels (whirl-kernel-test --bench, P2)", "golden", "KL for balance / fast",
                             "A2 stage bench of the op (P4)"]},
    "numeric": {"measure": ["A1 kernel cells", "golden (precise bitwise)", "KL for balance / fast"]},
    "schedule": {"measure": ["A4 scheduler bench (whirl-host-bench --suite sched, P3)", "server gate suites",
                             "server work trace (launch_table.py --server)"]},
    "io": {"measure": ["A5 tier bandwidth (whirl-host-bench --suite tier)", "tier gate suites"],
           "checks": ["kvram719: tools/repro/kvram719_tier.ps1 (each GPU, one after the other)"]},
    "sizing": {"measure": ["startup decision table (cpu_pins: whirl-startup-replay)", "P-size: tools/gates/startup_lines.py over the gate's server logs"],
               "checks": ["E2: tools/gates/env_matrix.ps1 -Cells E2 (8060S: WHIRL_VRAM_LIMIT_MB 49152 / 32768 / 24576 + fake 0.5 GB carve-out)",
                          "kvram719: tools/repro/kvram719_tier.ps1 (each GPU, one after the other)"],
               "steps": ["srv_env_e1", "srv_env_e3", "srv_env_e4", "srv_env_e7"]},
    "pinned": {"measure": ["NullBackend pinned tracking (cpu_device)"], "steps": ["srv_gpu_select"],
               "checks": ["kvram719: tools/repro/kvram719_tier.ps1 (each GPU, one after the other)"]},
    "device": {"measure": ["whirl-device-tests"], "steps": ["srv_gpu_select"],
               "checks": ["E6 hot-plug: progress/arch024-S14.E6_CHECKLIST.md (owner, manual)"]},
    "protocol": {"measure": ["whirl-server-tests / HTTP suites", "cpu_pins token hash"]},
    "vision": {"measure": ["cpu_vision", "golden server vision rows"],
               "checks": ["visab: CLI vision A/B on both GPUs (embedding bitwise + answer; no script in the repo yet: "
                          "scratch/s11/case_visab_*)"]},
    "compat": {"measure": [], "checks": ["dense: tools/golden/golden.ps1 check -Models dense (Q4_K_M, WHIRL_GATE_DENSE)",
                                         "dfl_<mode>: WHIRL_GATE_DFLASH (R9700, Q4_K_M DFlash)"]},
    "version": {"measure": [], "checks": ["version: every exe sha changes; release_check b_r1_gates needs gate summaries on the "
                                          "shipped exe (rebuild + the selected steps on it, or the ADR-0003 work-equivalence proof)"]},
    "build": {"measure": ["full build both archs"]},
    "doc": {"measure": []},
}
PATH_BEHAVIOUR = {"server.loop": "schedule", "server.prefill": "schedule", "server.decode": "schedule", "server.mtp": "schedule",
                  "cache.prefix": "schedule", "server.sampling": "numeric", "cache.spill": "io", "cache.restore": "io",
                  "cache.fingerprint": "io", "http": "protocol", "tokenizer": "protocol", "vision": "vision",
                  "startup": "sizing", "load": "compat", "cli.dflash": "compat", "build": "build"}
FILE_BEHAVIOUR = [  # (glob, class); paths as in the tree (git ls-files src include, 2026-10-25)
    ("src/device/*", "device"), ("src/hip/*", "device"), ("include/whirl/device*.h", "device"), ("include/whirl/hip*.h", "device"),
    ("src/model/device.cpp", "device"), ("src/server/server_main.cpp", "device"),
    ("src/device/handles.h", "lifetime"),
    ("src/common/startup_fit*", "sizing"), ("include/whirl/startup_fit.h", "sizing"), ("src/server/engine_gov*", "sizing"),
    ("src/common/mem_governor*", "sizing"), ("include/whirl/mem_governor.h", "sizing"), ("include/whirl/kv_layout.h", "sizing"),
    ("src/tier/ram_size.h", "sizing"), ("src/server/server_main.cpp", "sizing"),
    ("src/tier/*", "io"), ("src/server/tier_bridge*", "io"), ("src/cache/tier_adapter*", "io"),
    # arch024 PLAN P-719 (v2, A3): owners of pinned host memory -> srv_gpu_select + kvram719
    ("src/tier/kv_tier*", "pinned"), ("src/server/tier_bridge*", "pinned"), ("src/cache/ckpt_store*", "pinned"),
    ("src/server/engine_gov*", "pinned"), ("src/device/handles.h", "pinned"), ("src/hip/wddm_mem.cpp", "pinned"),
    ("src/hip/device_mem.cpp", "pinned"), ("src/hip/uma_mem.cpp", "pinned"), ("src/server/server_main.cpp", "pinned"),
    ("src/common/numerics*", "numeric"), ("include/whirl/numerics.h", "numeric"), ("src/model/tune*", "numeric"),
    ("src/kernels_host/*", "dispatch"), ("src/ops/*", "dispatch"), ("src/vision/*", "vision"), ("include/whirl/vision.h", "vision"),
    ("src/server/engine_vision.cpp", "vision"), ("src/server/vision_iface.h", "vision"),
    ("src/server/protocol*", "protocol"), ("src/server/http*", "protocol"), ("src/server/stream*", "protocol"),
    ("src/chat/*", "protocol"), ("src/tokenizer/*", "protocol"), ("src/server/engine*", "schedule"), ("src/cache/*", "schedule"),
    ("kernels/*", "dispatch"), ("kernels/*", "numeric"), ("docs/*", "doc"), ("*.md", "doc"), ("cmake/*", "build"),
]
NAME_BEHAVIOUR = [  # (regex on a touched / user function name, class)
    (re.compile(r"(?i)alloc|free|release|~|adopt|destroy|teardown"), "lifetime"),
    (re.compile(r"(?i)fit|sizing|budget|vram|pinned|carve|kvBytes|reserve|StateMem|allocState"), "sizing"),
    (re.compile(r"(?i)device|adapter|gpuPreflight|hipBackend|selectGpu|probe"), "device"),
    (re.compile(r"(?i)pinned|hostMalloc|hostAlloc|hostRegister"), "pinned"),
]


def behaviour_classes(changes, rep, beh, decl_rows, new_rev):
    """fills rep["behaviour"] {class: reasons}, rep["measure"] {class: kinds}, rep["outside_gate_checks"]; adds the
    classes' release steps (srv_gpu_select, srv_env_*) to rep["release_gate_steps"]"""
    for path in changes:
        for g, c in FILE_BEHAVIOUR:
            if fnmatch.fnmatch(path, g):
                beh[c].add(f"file {path}")
        if path.endswith(".hip"):
            beh["dispatch"].add(f"kernel {path}")
            beh["numeric"].add(f"kernel {path}")
        if path.startswith("tests/"):
            beh.setdefault("test", set()).add(path)
    for p, why_l in rep["paths"].items():
        c = PATH_BEHAVIOUR.get(p)
        if c:
            beh[c].add(f"path {p}")
    for f, w in rep["touched"]:
        if w.startswith("function "):
            nm = w[len("function "):]
            for rx, c in NAME_BEHAVIOUR:
                if rx.search(nm.split("::")[-1]):
                    beh[c].add(f"function {nm}")
    for r in decl_rows:
        if any(t in ("PinnedBuf",) for t in r["types"]):
            beh["pinned"].add(f"declaration {r['name']} (PinnedBuf)")
        for rx, c in NAME_BEHAVIOUR:
            if rx.search(r["name"].split("::")[-1]):
                beh[c].add(f"declaration {r['name']}")
    ver = changes.get("CMakeLists.txt")
    if ver:
        src = (show(new_rev, "CMakeLists.txt") if new_rev else show_side("CMakeLists.txt"))
        src = src.split("\n") if isinstance(src, str) else src
        if src and any(re.search(r"\bproject\s*\(.*VERSION", src[i - 1]) for i in ver["new"] if 0 < i <= len(src)):
            beh["version"].add("CMakeLists.txt project(... VERSION ...) changed")
    beh = {k: v for k, v in beh.items() if v}
    rep["behaviour"] = {k: sorted(v)[:8] for k, v in sorted(beh.items())}
    rep["measure"] = {k: BEHAVIOUR.get(k, {}).get("measure", []) for k in sorted(beh)}
    checks = sorted({x for k in beh for x in BEHAVIOUR.get(k, {}).get("checks", [])})
    rep["outside_gate_checks"] = checks
    add = sorted({x for k in beh for x in BEHAVIOUR.get(k, {}).get("steps", [])})
    if {"vision", "numeric", "dispatch", "layout"} & set(beh):
        # golden server has the vision rows (the only GPU vision correctness check in release_gate); validation hold-out
        # f6e4236 (vision::imageKey) lost golden when type names stopped selecting it. Review C1: numeric / dispatch /
        # layout name golden (bitwise) as their measurement, so they add the golden steps too
        add = sorted(set(add) | {f"golden_{'swift' if md == 'dense' else 'moe'}" for md in rep.get("models") or ["dense", "moe"]})
    if add and (rep["paths"] or any(p.startswith(("src/", "include/", "kernels/")) for p in changes)):
        rep["release_gate_steps"] = sorted(set(rep["release_gate_steps"]) | set(add))


def show_side(path):
    """lines of a file in the worktree, or None"""
    p = os.path.join(ROOT, path)
    try:
        return open(p, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        return None


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
        steps.add({"tests": "cpu_tests"}.get(short, f"cpu_{short}"))
    if kt:
        steps.add("kt_full")
    if gsuites:
        # measure-tools P1 (map defect found by the arch024 ground truth, B0: golden server listed, no golden step): the
        # golden cases the report lists are run by the golden_<model> steps
        steps |= {f"golden_{'swift' if md == 'dense' else 'moe'}" for md in models}
    for n in gpu_cli:
        if n.startswith("golden."):
            steps |= {f"golden_{'swift' if md == 'dense' else 'moe'}" for md in models}
        elif n == "seqtest":
            steps |= {f"seq_{'swift' if md == 'dense' else 'moe'}_<mode>" for md in models}
        elif n == "selftest":
            steps |= {f"st_{'swift' if md == 'dense' else 'moe'}_<mode>" for md in models}
        elif n == "dflash_bench":
            steps.add("dfl_<mode> (R9700, Q4_K_M compatibility only)")
    not_in_gate = []
    for s in suites:
        hit = [st for st, d in m["release_srv_steps"].items() if s in d["suites"]]
        if s == "golden":  # suite server of golden.ps1 (default suites, ARCH2): run by the golden_<model> steps
            hit += [f"golden_{'swift' if md == 'dense' else 'moe'}" for md in models]
        steps |= set(hit)
        if not hit:
            not_in_gate.append(s)
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
    for e, v in r["cpu_tests"].items():
        print(f"   cpu      {e}: {len(v)} test(s): {', '.join(v[:8])}{' ...' if len(v) > 8 else ''}")
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
    for k, v in r["extra"].items():
        print(f"   also     {k}  ({', '.join(v[:3])})")
    if r["matrix_cells"]:
        print(f"   cells    {', '.join(r['matrix_cells'])}")
    for k, v in (r.get("behaviour") or {}).items():
        meas = (r.get("measure") or {}).get(k) or []
        print(f"   class    {k}: {'; '.join(v[:3])}{' ...' if len(v) > 3 else ''}" + (f"  -> measure: {'; '.join(meas)}" if meas else ""))
    for c in r.get("outside_gate_checks") or []:
        print(f"   check    {c}  (GPU check outside release_gate)")
    if r.get("declarations") and why:
        for d in r["declarations"]:
            print(f"   decl     {d['file']}:{d['line']} ({d['side']}) {d['name']} [{d['what']}]" + (f" type {'/'.join(d['types'])}" if d["types"] else ""))
    for d in r.get("map_defects") or []:
        print(f"   DEFECT   {d}")
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
