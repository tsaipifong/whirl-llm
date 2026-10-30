# SPDX-License-Identifier: Apache-2.0
"""arch025-emut: run emut.exe (clang's ExprMutationAnalyzer, official LLVM) on the census TUs and write ast.json in the
schema of tools/gates/engine_split_ast.py, so engine_split_census.py could consume either analyzer.

  python -I emut_run.py --root R --out ast.json --classes Engine,JobQueue,... --tu src/server/engine.cpp ...
         [--cache DIR] [--jobs N] [--callable-ok a,b] [--emut PATH] [--llvm DIR]

Same TUs, flags (engine_split_ast.py DEFINES / STDFLAGS / -I src, include, generated version.h) and tracked classes as
the Python analyzer.  The LLVM folder is --llvm, else %WHIRL_LLVM%, else C:/ClaudeWorkspace/tools/llvm/21.1.8; emut.exe
is --emut, else %WHIRL_EMUT%.  No PATH lookup.  --callable-ok is accepted for CLI compatibility and ignored: the
analyzer has no such list (std::function::operator() is const, so it calls every std::function member call a read -
the census's callable_reads rule is stricter than the analyzer there).

Output: {"clang": emut path, "analyzer": "...", "tus": [{"tu", "errors", "rc", "deps", "hits", "calls", "funcs",
"odd", "nopos", "unmapped", "outside"}]}
  hits  [rel, index, member, class, "w"|"r", why, weak, returned_from]   (why starts with "write: " / "pointee-write: "
        / "read"; index = character offset in the file read as text, as engine_split_ast.py)
  calls [rel, index, callee_mangled, "w"|"r", why, returned_from, weak]
Exit 0 ok, 3 tool / LLVM missing.
"""
import hashlib, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "emut-1"
DEFINES = ["-DNOMINMAX", "-DWHIRL_WITH_HIP=1", "-DWIN32_LEAN_AND_MEAN", "-D_CRT_SECURE_NO_WARNINGS"]
STDFLAGS = ["--target=x86_64-pc-windows-msvc", "-std=c++20", "-fsyntax-only", "-Wno-everything"]
VERSION_H = ('#pragma once\n#define WHIRL_VERSION_MAJOR 0\n#define WHIRL_VERSION_MINOR 0\n#define WHIRL_VERSION_PATCH 0\n'
             '#define WHIRL_VERSION_STRING "0.0.0"\n#define WHIRL_GPU_ARCHS_STRING "census"\n')


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a else default


def llvm_dir():
    return arg("--llvm") or os.environ.get("WHIRL_LLVM") or r"C:\ClaudeWorkspace\tools\llvm\21.1.8"


def resource_dir(llvm):
    base = os.path.join(llvm, "lib", "clang")
    if not os.path.isdir(base): return None
    vs = sorted(os.listdir(base))
    return os.path.join(base, vs[-1]) if vs else None


class PosMap:
    """byte offset -> character index of the file as Python text mode reads it (universal newlines); as engine_split_ast.py"""
    cache = {}

    @classmethod
    def get(cls, path):
        if path in cls.cache: return cls.cache[path]
        raw = open(path, "rb").read()
        mp = [0] * (len(raw) + 1)
        ci, bi, n = 0, 0, len(raw)
        while bi < n:
            b = raw[bi]
            if b == 0x0D and bi + 1 < n and raw[bi + 1] == 0x0A:
                mp[bi] = ci; mp[bi + 1] = ci; bi += 2; ci += 1; continue
            ln = 1 if b < 0x80 else 2 if b >> 5 == 6 else 3 if b >> 4 == 14 else 4 if b >> 3 == 30 else 1
            for q in range(ln):
                if bi + q < n: mp[bi + q] = ci
            bi += ln; ci += 1
        mp[n] = ci
        cls.cache[path] = mp
        return mp


def rel_of(root, f):
    p = os.path.normcase(os.path.normpath(f))
    for sub in ("src", "include"):
        b = os.path.normcase(os.path.normpath(os.path.join(root, sub))) + os.sep
        if p.startswith(b):
            r = os.path.relpath(p, os.path.normcase(os.path.normpath(os.path.join(root, sub)))).replace("\\", "/")
            return r if sub == "src" else "include:" + r
    return None


def real_case(root, rel):
    """the tool lower-cases paths; give the file's spelling from the tree (rel paths are compared case-sensitively)"""
    sub, r = ("include", rel[8:]) if rel.startswith("include:") else ("src", rel)
    cur = os.path.join(root, sub)
    out = []
    for part in r.split("/"):
        try:
            m = next((x for x in os.listdir(cur) if x.lower() == part.lower()), part)
        except OSError:
            m = part
        out.append(m); cur = os.path.join(cur, m)
    return ("include:" if sub == "include" else "") + "/".join(out)


def run_tu(emut, llvm, root, tu_rel, classes, gen, work):
    src = os.path.join(root, tu_rel)
    out = os.path.join(work, re.sub(r"[^\w.]+", "_", tu_rel) + f".{os.getpid()}.json")
    flags = STDFLAGS + DEFINES + ["-I" + os.path.join(root, "src"), "-I" + os.path.join(root, "include"), "-I" + gen]
    rd = resource_dir(llvm)
    if rd: flags = ["-resource-dir", rd] + flags
    cmd = [emut, "--classes=" + ",".join(sorted(classes)), "--root=" + root, "--out=" + out, src, "--"] + flags
    r = subprocess.run(cmd, capture_output=True)
    err = r.stderr.decode("utf-8", errors="replace")
    errors = [l for l in err.splitlines() if re.search(r"\berror\b:", l)][:20]
    res = {"tu": tu_rel, "errors": errors, "rc": r.returncode, "deps": [tu_rel], "hits": [], "calls": [], "funcs": {},
           "odd": [], "nopos": 0, "unmapped": [], "outside": 0}
    if not os.path.exists(out):
        res["errors"] = res["errors"] or [f"emut wrote no output (rc {r.returncode}): {err[-300:]}"]
        return res
    d = json.load(open(out, encoding="utf-8"))
    os.remove(out)
    res["odd"], res["nopos"], res["outside"] = sorted(set(d.get("odd", []))), d.get("nopos", 0), d.get("outside", 0)
    rels = {}

    def rel(f):
        if f not in rels:
            q = rel_of(root, f)
            rels[f] = real_case(root, q) if q else None
        return rels[f]

    def cpos(f, off):
        q = rel(f)
        if q is None: return None
        full = os.path.join(root, "src", q) if not q.startswith("include:") else os.path.join(root, "include", q[8:])
        try:
            mp = PosMap.get(full)
            return [q, mp[min(off, len(mp) - 1)]]
        except OSError:
            return None
    deps = set()
    for f in d.get("deps", []):
        q = rel(f)
        if q: deps.add(("src/" + q) if not q.startswith("include:") else ("include/" + q[8:]))
    res["deps"] = sorted(deps | {tu_rel})
    for f, off, mem, cls, v, why, weak, rf in d.get("hits", []):
        q = cpos(f, off)
        if q: res["hits"].append(q + [mem, cls, v, why, weak, rf])
        elif v == "w": res["unmapped"].append(f"{f}:{off} {mem} ({why})")
    for f, off, callee, v, why, rf, weak in d.get("calls", []):
        q = cpos(f, off)
        if q: res["calls"].append(q + [callee, v, why, rf, weak])
    res["funcs"] = d.get("funcs", {})
    res["analyzer"] = d.get("analyzer")
    return res


def main():
    root = arg("--root")
    out = arg("--out")
    classes = set(x for x in (arg("--classes", "Engine") or "").split(",") if x)
    tus = [sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--tu"]
    cache = arg("--cache")
    jobs = int(arg("--jobs", "6"))
    llvm = llvm_dir()
    emut = arg("--emut") or os.environ.get("WHIRL_EMUT")
    if not emut or not os.path.exists(emut) or not os.path.isdir(llvm):
        print(f"emut_run: NO TOOL (emut {emut}, llvm {llvm}; set WHIRL_EMUT / WHIRL_LLVM)"); return 3
    work = cache or os.path.join(os.path.dirname(os.path.abspath(out)), "emut_work")
    gen = os.path.join(work, "_generated")
    os.makedirs(os.path.join(gen, "whirl"), exist_ok=True)
    vh = os.path.join(gen, "whirl", "version.h")
    if not os.path.exists(vh): open(vh, "w").write(VERSION_H)
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(lambda t: run_tu(emut, llvm, root, t, classes, gen, work), tus))
    an = next((r.get("analyzer") for r in res if r.get("analyzer")), None)
    json.dump({"clang": emut, "analyzer": an, "version": VERSION,
               "emut_sha256": hashlib.sha256(open(emut, "rb").read()).hexdigest(), "tus": res},
              open(out, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
