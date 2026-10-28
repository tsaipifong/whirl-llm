# SPDX-License-Identifier: Apache-2.0
"""modules.json <-> CMake targets (arch025 D11 N17, DISCOVERY review item 11): the module table and the build stay in step.

    python tools/gates/cmake_modules_check.py --configure DIR [--root REPO] [--modules FILE]
    python tools/gates/cmake_modules_check.py --build DIR     [--root REPO] [--modules FILE]

The target -> source lists come from CMake itself: the File API codemodel-v2 reply (cmake-file-api(7)) of a configured
build directory. --configure DIR creates DIR (a directory this tool owns: it holds the marker file .cmake_modules_check;
an existing DIR without it is refused), writes the codemodel-v2 query and runs a configure-only
tools/safety/build_cfg.bat (WHIRL_CFG_ONLY=1, default options = HIP on, both code objects), ~1 min; nothing is compiled.
--build DIR reads the reply of a directory configured with that query already (the self-test's synthetic trees).
The device code objects are custom commands, which the codemodel does not describe: their device translation units are
the inputs of the `-x hip` commands in DIR/build.ninja (generator output, read only for that).

Checks (each FAIL line names the file / target and what to change):
  1 a source file a module lists (its paths globs over the census' product files; .cpp / .c / .cu / .hip) is compiled
    by a CMake target, or is a device translation unit, or is #included (transitively) by one of those; headers
    (.h / .hpp / .inc / .cuh) need nothing
  2 every product source a target compiles (src/ include/ kernels/, and the device translation units) has exactly one
    module: the census' own rule (census_check.owners, no second list); generated sources outside the iso copies
    (bin2c arrays, embedded_objects.cpp) are counted, not checked
  3 the iso build targets match the isolation manifests (tests/sampling/isolation.txt, tests/iso/*/isolation.txt):
    per manifest the object library (lib line, else whirl_iso_<module>) compiles exactly its src lines and the test
    program (exe line) exactly its testsrc lines (as the copies under DIR/iso/<module>/{mod,test}), the program depends
    on the library, the module is a modules.json module; and every target that compiles a file under DIR/iso/ belongs
    to a manifest
Last line "cmake modules: ok (...)" or "cmake modules: FAIL (N problem(s))". Exit 0 ok, 1 FAIL, 2 setup (no reply, refused DIR,
configure failed).
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import census_check  # noqa: E402  (product_files + the module ownership rule)

REPO = os.path.dirname(os.path.dirname(HERE))
MARKER = ".cmake_modules_check"
SRC_EXTS = (".cpp", ".c", ".cu", ".hip")
INC_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def norm(p):
    return os.path.normcase(os.path.normpath(p)).replace("\\", "/")


def rel_to(p, base):
    """repo-relative '/' path of p when p is under base, else None"""
    a, b = norm(p), norm(base)
    if a == b or not a.startswith(b + "/"):
        return None
    return os.path.normpath(p).replace("\\", "/")[len(os.path.normpath(base)) + 1:]


def read_reply(build):
    """{target name: {"type", "compiled": [abs paths], "generated": set(abs), "deps": [names]}}, source dir"""
    rdir = os.path.join(build, ".cmake", "api", "v1", "reply")
    idx = sorted(glob.glob(os.path.join(rdir, "index-*.json")))
    if not idx:
        raise RuntimeError(f"no File API reply in {rdir} (configure with the codemodel-v2 query first, or use --configure)")
    ix = json.load(open(idx[-1], encoding="utf-8"))
    cm = [o for o in ix.get("objects", []) if o.get("kind") == "codemodel"]
    if not cm:
        raise RuntimeError(f"{idx[-1]}: no codemodel object (the query asks for codemodel-v2)")
    c = json.load(open(os.path.join(rdir, cm[0]["jsonFile"]), encoding="utf-8"))
    src_dir = c["paths"]["source"]
    conf = c["configurations"][0]
    ids = {t["id"]: t["name"] for t in conf["targets"]}
    out = {}
    for t in conf["targets"]:
        tj = json.load(open(os.path.join(rdir, t["jsonFile"]), encoding="utf-8"))
        comp, gen = [], set()
        for s in tj.get("sources", []):
            p = s["path"] if os.path.isabs(s["path"]) else os.path.join(src_dir, s["path"])
            if s.get("isGenerated"):
                gen.add(norm(p))
            if "compileGroupIndex" in s:
                comp.append(p)
        out[tj["name"]] = {"type": tj.get("type"), "compiled": comp, "generated": gen,
                           "deps": [ids.get(d["id"], d["id"]) for d in tj.get("dependencies", [])]}
    return out, src_dir


def device_roots(build, root):
    """repo-relative inputs of the `-x hip` device compiles in build.ninja"""
    p = os.path.join(build, "build.ninja")
    if not os.path.isfile(p):
        return []
    roots = set()
    for line in open(p, encoding="utf-8", errors="replace"):
        if " -x hip " not in line:
            continue
        toks = line.replace('"', " ").split()
        for i, t in enumerate(toks):
            if t == "-o" and i + 2 < len(toks):
                r = rel_to(toks[i + 2].replace("$:", ":"), root)
                if r:
                    roots.add(r)
    return sorted(roots)


def include_closure(starts, root, extra_dirs=("kernels", "include", "src")):
    seen, todo = set(), list(starts)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        try:
            text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for inc in INC_RE.findall(text):
            for base in [os.path.dirname(f)] + list(extra_dirs):
                cand = os.path.normpath(os.path.join(base, inc)).replace("\\", "/")
                if os.path.isfile(os.path.join(root, cand)):
                    todo.append(cand)
                    break
    return seen


def manifests(root):
    out = []
    lists = [os.path.join(root, "tests", "sampling", "isolation.txt")] + sorted(glob.glob(os.path.join(root, "tests", "iso", "*", "isolation.txt")))
    for path in lists:
        if not os.path.isfile(path):
            continue
        m = {"file": os.path.relpath(path, root).replace("\\", "/"), "module": None, "exe": None, "lib": None, "src": [], "testsrc": []}
        for line in open(path, encoding="utf-8"):
            mm = re.match(r"^([a-z]+)[ \t]+([^ \t#\r\n]+)", line)
            if not mm:
                continue
            k, v = mm.group(1), mm.group(2)
            if k in ("module", "exe", "lib"):
                m[k] = v
            elif k in ("src", "testsrc"):
                m[k].append(v)
        if not m["module"] and m["file"] == "tests/sampling/isolation.txt":
            m["module"] = "sampling"  # cmake/iso.cmake: the one manifest without a module line
        if m["module"] and not m["lib"]:
            m["lib"] = re.sub(r"[^A-Za-z0-9]", "_", "whirl_iso_" + m["module"])
        out.append(m)
    return out


def iso_copy(kind, path):
    rel = re.sub(r"^(src|include|tests)/", "", path)
    return ("test/" if kind == "testsrc" else "mod/") + rel


def check(root, build, mods, targets, roots, pending=(), release=False):
    errs, notes, warns = [], [], []
    files = census_check.product_files(root)
    fileset = set(files)
    iso_dir = os.path.join(build, "iso")
    compiled = set()  # repo-relative product / tests / tools files compiled by a non-iso target
    n_gen = 0
    for name, t in targets.items():
        for p in t["compiled"]:
            if rel_to(p, iso_dir) is not None:
                continue
            r = rel_to(p, root)
            if r is not None:
                compiled.add(r)
            elif norm(p) in t["generated"] or rel_to(p, build) is not None:
                n_gen += 1
            else:
                errs.append(f"target {name} compiles {p}, outside the source tree and not generated")
    compiled.update(roots)
    reach = include_closure(sorted(compiled), root)
    # 1 module-listed sources are built
    n_listed = 0
    for mname, d in sorted(mods.items()):
        for f in files:
            if not any(fnmatch.fnmatch(f, g) for g in d.get("paths") or []):
                continue
            if not f.endswith(SRC_EXTS):
                continue
            n_listed += 1
            if f not in compiled and f not in reach:
                errs.append(f"module {mname}: {f} is compiled by no CMake target and included by no compiled source "
                            f"(add it to a target in CMakeLists.txt / cmake/*.cmake, or drop it from the module)")
    # 2 compiled product sources have exactly one module (census rule)
    n_prod = 0
    for f in sorted(compiled):
        if f.split("/", 1)[0] not in census_check.TOPS:
            continue
        n_prod += 1
        if f not in fileset:
            errs.append(f"{f} is compiled but is not a census product file (extension?)")
            continue
        own = census_check.owners(f, mods)
        pend = [p for p in pending if not own and fnmatch.fnmatch(f, p.get("glob") or "")]
        if pend and not release:
            warns.append(f"{f} is compiled and still census-pending (owner {pend[0].get('owner')})")
            continue
        if len(own) != 1:
            errs.append(f"{f} is compiled by CMake and is in {len(own)} modules{': ' + ', '.join(own) if own else ''} (census: exactly one)")
    # 3 iso targets <-> manifests
    mans = manifests(root)
    claimed = set()
    for m in mans:
        tag = f"{m['file']} (module {m['module']})"
        if not m["module"] or not m["exe"]:
            errs.append(f"{m['file']}: needs 'module' and 'exe' lines")
            continue
        if m["module"] not in mods:
            errs.append(f"{tag}: module is not in tools/gates/modules.json")
        base = os.path.join(iso_dir, m["module"])
        for tname, kind, want_type, lines in ((m["lib"], "src", "OBJECT_LIBRARY", m["src"]), (m["exe"], "testsrc", "EXECUTABLE", m["testsrc"])):
            claimed.add(tname)
            t = targets.get(tname)
            if t is None:
                errs.append(f"{tag}: no CMake target {tname} (cmake/iso.cmake builds one per manifest)")
                continue
            if t["type"] != want_type:
                errs.append(f"{tag}: target {tname} is {t['type']}, want {want_type}")
            want = {iso_copy(kind, p) for p in lines}
            got = set()
            for p in t["compiled"]:
                r = rel_to(p, base)
                got.add(r if r is not None else "<outside iso/" + m["module"] + ">: " + p)
            for x in sorted(want - got):
                errs.append(f"{tag}: target {tname} does not compile {x} (manifest {kind} line)")
            for x in sorted(got - want):
                errs.append(f"{tag}: target {tname} compiles {x}, which the manifest does not list")
        if m["exe"] in targets and m["lib"] not in targets[m["exe"]]["deps"]:
            errs.append(f"{tag}: test program {m['exe']} does not depend on {m['lib']}")
    for name, t in sorted(targets.items()):
        if name in claimed:
            continue
        if any(rel_to(p, iso_dir) is not None for p in t["compiled"]):
            errs.append(f"target {name} compiles iso copies but no isolation manifest names it")
    notes.append(f"{len(warns)} census-pending, " if warns else "")
    notes.append(f"{len(targets)} targets, {n_listed} module source files, {n_prod} compiled product sources "
                 f"({len(roots)} device TU), {n_gen} generated, {len(mans)} iso manifests")
    return errs, notes, warns


def configure(root, d):
    if os.path.isdir(d):
        if not os.path.isfile(os.path.join(d, MARKER)):
            raise RuntimeError(f"{d} exists and is not this tool's directory (no {MARKER}); pass a new directory")
        shutil.rmtree(d)
    q = os.path.join(d, ".cmake", "api", "v1", "query")
    os.makedirs(q)
    open(os.path.join(d, MARKER), "w").write("configure-only build directory of tools/gates/cmake_modules_check.py\n")
    open(os.path.join(q, "codemodel-v2"), "w").close()
    env = dict(os.environ, WHIRL_CFG_ONLY="1")
    env.pop("WHIRL_QUIET_WAIT", None)
    bat = os.path.join(root, "tools", "safety", "build_cfg.bat")
    r = subprocess.run(["cmd", "/c", bat, d, ""], env=env, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"configure failed (exit {r.returncode}): " + " | ".join((r.stdout + r.stderr).strip().splitlines()[-5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--modules", default=None)
    ap.add_argument("--pending", default=None)
    ap.add_argument("--release", action="store_true", help="census-pending compiled files fail too (census_check --release)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--configure", help="new / own build directory: configure-only with the File API query")
    g.add_argument("--build", help="configured build directory with a codemodel-v2 reply")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    try:
        mods = json.load(open(a.modules or os.path.join(root, "tools", "gates", "modules.json"), encoding="utf-8"))["modules"]
        ppath = a.pending or os.path.join(root, "tools", "gates", "census_pending.json")
        pending = (json.load(open(ppath, encoding="utf-8")).get("pending") or []) if os.path.isfile(ppath) else []
        build = os.path.abspath(a.configure or a.build)
        if a.configure:
            configure(root, build)
        targets, src_dir = read_reply(build)
        if norm(src_dir) != norm(root):
            raise RuntimeError(f"the reply describes source dir {src_dir}, not {root}")
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"cmake modules: SETUP ({e})")
        return 2
    errs, notes, warns = check(root, build, mods, targets, device_roots(build, root), pending, a.release)
    for w in warns:
        print("   WARN " + w)
    for e in errs:
        print("   FAIL " + e)
    tail = "".join(notes)
    print(f"cmake modules: {'ok' if not errs else f'FAIL ({len(errs)} problem(s))'} ({tail})")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
