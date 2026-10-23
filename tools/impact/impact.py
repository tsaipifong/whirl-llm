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
            if not users and outside_types:
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
                hits.append((path, f"declarations ({', '.join(sorted(outside_ids)[:6])}{'...' if len(outside_ids) > 6 else ''}): "
                                   f"{len(prod)} function(s) in {len({f for f, _ in prod})} file(s) use them"))
            else:
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
    rep.update(module_gate(changes))
    return rep


def load_modules(root=None):
    """tools/gates/modules.json of the tree (arch024-S12), or None when the tree has none"""
    p = os.path.join(root or ROOT, "tools", "gates", "modules.json")
    try:
        return json.load(open(p, encoding="utf-8"))["modules"]
    except (OSError, ValueError, KeyError):
        return None


def module_gate(changes, mods=None):
    """Which modules (tools/gates/modules.json) the changed product files belong to, and the dev-time module gate
    to run. Product code outside every module, two modules, or a contract file -> the full gate. Dev-time fast
    feedback only until the owner signs off O1 / O2 (arch024 PLAN): releases run the full gate."""
    mods = load_modules() if mods is None else mods
    if mods is None:
        return {"modules": {}, "module_gate": None}
    hit = defaultdict(list)
    loose = []
    for path in sorted(changes):
        if not path.startswith(("src/", "include/", "kernels/")):
            continue
        ms = [n for n, d in mods.items() if any(fnmatch.fnmatch(path, g) for g in d.get("paths", []))]
        for n in ms:
            hit[n].append(path)
        if not ms:
            loose.append(path)
    when = sorted({st for n in hit for f, sts in (mods[n].get("when") or {}).items() if f in changes for st in sts})
    # a module's `when` file (e.g. prefill_schedule.h for M3 / M4, PLAN C6) adds steps instead of forcing the full gate
    loose = [f for f in loose if not any(f in (mods[n].get("when") or {}) for n in hit)]
    full = [n for n in hit if mods[n].get("full")]
    if not hit and not loose:
        gate = "none (no product code)"
    elif full:
        gate = f"full gate, both GPUs (contract module {', '.join(full)})"
    elif loose:
        gate = f"full gate or the release steps above: {len(loose)} product file(s) in no module ({', '.join(loose[:4])}{' ...' if len(loose) > 4 else ''})"
    elif len(hit) > 1:
        gate = f"full gate, both GPUs (cross-module: {', '.join(sorted(hit))})"
    else:
        n = next(iter(hit))
        gate = f"release_gate.ps1 -Module {n}" + (f" -Steps {','.join(when)}" if when else "")
    return {"modules": {n: v for n, v in sorted(hit.items())}, "module_gate": gate}


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
    if r.get("module_gate"):
        for n, fl in r["modules"].items():
            print(f"   module   {n}: {', '.join(fl[:4])}{' ...' if len(fl) > 4 else ''}")
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
