# SPDX-License-Identifier: Apache-2.0
"""CMake build-graph diff for test selection (arch025-modsel3, PLAN 3.8).

A CMakeLists.txt / cmake/*.cmake edit that is not a pure source-list edit (impact.cmake_list_edit) - a new add_library /
add_executable, a target_link_libraries edit, a compile definition, a new include() - used to plan the full matrix. This
tool configures both revisions (CMake, Ninja generator, configure only, ~40 s each; nothing is compiled) and compares the
generated Ninja action graphs, bazel-diff style (progress\\arch025-modsel3.PRIOR_ART.md): every action is fingerprinted
from its rule command, its variables (paths normalized: <S> = source root, <B> = build directory) and the fingerprints of
the actions that produce its inputs; configure-time files (configure_file / file(COPY): generated headers, the iso copies)
count by content for the actions that read them (as an input or under an include directory).

Per executable / DLL it compares what the link is made of:
  * the objects (each = the source it compiles + its fingerprint): an object whose fingerprint is new or gone is a
    "changed source" of that executable (test_change then uses that source's module / test executable rules);
  * the rest of the link (rule, flags, external libraries, manifests): a change there cannot be attributed to a source.
Result (compare()): {"ok", "full": [reasons], "sources": {artifact: [repo paths]}, "exe_changes": {artifact: [reasons]},
"notes": [...]} for the artifacts a GPU step runs (gpu_exes); every other executable is a CPU test (CPU all runs them).
Fail closed ("full"): a generated source (custom command output, e.g. device code objects) of a GPU executable changed; a
product executable's link changed beyond its objects; a GPU executable disappeared; a DLL or another terminal build
output that no executable links changed (a copied runtime DLL, ...); a configure-time file that no action reads changed;
either configure failed (analyze_revs "ok": False); a product executable links the same objects in another order (moving
sources into a new library: MSVC's static-initialization order between objects follows the link order, not proven
irrelevant for WHIRL). The same objects in another order in a test executable select the GPU steps that run it.

    python tools/impact/cmake_graph.py --root REPO --old REV [--new REV] --work DIR [--gpu-exes a,b]   (JSON to stdout)
"""
import argparse
import collections
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# per-target paths / descriptions: not part of what an action does
EXCL = {"OBJECT_DIR", "OBJECT_FILE_DIR", "TARGET_COMPILE_PDB", "TARGET_PDB", "TARGET_SUPPORT_DIR", "TARGET_FILE",
        "TARGET_IMPLIB", "DESC", "DEP_FILE", "RESTAT", "description", "depfile", "deps", "restat", "pool"}
SKIP_KINDS = {"phony", "CLEAN", "HELP", "RERUN_CMAKE", "VERIFY_GLOBS"}
SKIP_OUTS = re.compile(r"^CMakeFiles/(edit_cache|rebuild_cache|install|install/local|install/strip|list_install_components)\.util$")
# configure-directory entries that are CMake's own bookkeeping, not inputs of any action
NOT_GEN = {"CMakeFiles", ".cmake", "build.ninja", "CMakeCache.txt", "cmake_install.cmake", ".cmake_modules_check",
           ".ninja_log", ".ninja_deps", "CTestTestfile.cmake", ".cmake_graph", "whirl_codepage.txt"}


def _root_rx(p):
    p = p.rstrip("\\/")
    return re.compile("".join("[\\\\/]" if c in "\\/" else re.escape(c) for c in p) + r"(?=[\\/\s\"';:,)]|$)", re.I)


def norm(text, S, B):
    """text with the source root -> <S>, the build directory -> <B>, '\\' -> '/'"""
    t = text.replace("$:", ":")
    for root, tok in sorted(((S, "<S>"), (B, "<B>")), key=lambda x: -len(x[0])):
        t = _root_rx(root).sub(tok, t)
    t = t.replace("\\", "/")
    # CMake names a custom command's transformed depfile by a hash of its absolute paths: same command, other name
    return re.sub(r"CMakeFiles/d/[0-9a-f]{64}\.d", "CMakeFiles/d/<H>.d", t)


def _tokens(s):
    """ninja path list -> tokens ('$ ' escaped space, '$$' dollar)"""
    s = s.replace("$$", "\0D").replace("$ ", "\0S")
    return [x.replace("\0S", " ").replace("\0D", "$") for x in s.split()]


def _key(p):
    """canonical output key: relative to the build directory"""
    return p[4:] if p.startswith("<B>/") else p


def _kind(rule):
    m = re.match(r"^([A-Z_]+?)(?:__.*)?$", rule)
    return m.group(1) if m else rule


def parse(build_text, rules_text, S, B, gen):
    """(build.ninja text, CMakeFiles/rules.ninja text, source root, build dir, {configure-time file rel path: content})
    -> graph {"edges", "producer", "fp", "artifacts", "gen", "gen_read", "terminals"}"""
    rules, cur = {}, None
    for l in rules_text.splitlines():
        m = re.match(r"^rule (\S+)", l)
        if m:
            cur = m.group(1)
            rules[cur] = ""
        elif cur and re.match(r"^\s+command = ", l):
            rules[cur] = norm(l.split("=", 1)[1].strip(), S, B)
    edges, e = [], None
    for l in build_text.splitlines():
        if l.startswith("build "):
            body = l[6:].replace("$:", "\0C")
            i = body.index(":")
            outs = _tokens(body[:i].replace("\0C", ":").split(" | ")[0])
            rest = body[i + 1:].replace("\0C", ":")
            rest, _, _order = rest.partition(" || ")
            rest, _, impl = rest.partition(" | ")
            tk = _tokens(rest)
            rule = tk[0] if tk else ""
            e = {"outs": [_key(norm(o, S, B)) for o in outs], "rule": rule, "kind": _kind(rule),
                 "ins": [norm(x, S, B) for x in tk[1:]], "impl": [norm(x, S, B) for x in _tokens(impl)], "vars": {}}
            edges.append(e)
        elif e is not None and l.startswith("  ") and " = " in l:
            k, v = l.strip().split(" = ", 1) if " = " in l.strip() else (l.strip().rstrip(" ="), "")
            e["vars"][k] = norm(v, S, B)
        elif not l.strip() or l.startswith("#"):
            e = None if not l.strip() else e
    edges = [x for x in edges if x["kind"] not in SKIP_KINDS and not any(SKIP_OUTS.match(o) for o in x["outs"])]
    producer = {}
    for x in edges:
        for o in x["outs"]:
            producer[o] = x
    genh = {k: hashlib.sha256(norm(v, S, B).encode("utf-8") if isinstance(v, str) else v).hexdigest()[:16] for k, v in gen.items()}
    gen_read = set()
    memo = {}

    def inp(p):
        k = _key(p)
        if k in producer:
            return "E:" + fp(producer[k])
        if k in genh:
            gen_read.add(k)
            return f"G:{k}:{genh[k]}"
        return p

    gkeys, pkeys, under = sorted(genh), sorted(producer), {}

    def files_under(rel):
        """configure-time files and action outputs under <B>/rel (an include directory or a copy root)"""
        if rel not in under:
            g = [k for k in gkeys if k.startswith(rel)]
            gen_read.update(g)
            under[rel] = ([f"{k}:{genh[k]}" for k in g], [k for k in pkeys if k.startswith(rel)])
        return under[rel]

    def incdirs(x):
        """what a compile action can #include from the build directory: everything under its <B> include directories,
        and, for a source that is itself a copy under <B> (cmake/iso.cmake: <B>/iso/<module>/...), everything under the
        first two directories of that copy (an #include relative to the including file)"""
        rels = []
        for d in re.findall(r"-I\s*(\S+)", x["vars"].get("INCLUDES", "")):
            d = d.strip('"')
            if d.startswith("<B>/") or not d.startswith("<"):
                rels.append(_key(d).rstrip("/") + "/")
        if "COMPILER" in x["kind"] and x["ins"] and not x["ins"][0].startswith("<S>/"):
            parts = _key(x["ins"][0]).split("/")
            if len(parts) > 2:
                rels.append("/".join(parts[:2]) + "/")
        out = []
        for rel in sorted(set(rels)):
            g, ps = files_under(rel)
            out += g + ["E:" + fp(producer[k]) for k in ps if producer[k] is not x]
        return out

    def fp(x):
        i = id(x)
        if i not in memo:
            memo[i] = "?"  # cycle guard (ninja graphs are acyclic)
            v = {k: s for k, s in sorted(x["vars"].items()) if k not in EXCL}
            blob = json.dumps([x["kind"], rules.get(x["rule"], ""), v, [inp(p) for p in x["ins"]],
                               sorted(inp(p) for p in x["impl"]), incdirs(x)], sort_keys=True)
            memo[i] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
        return memo[i]

    def src_of(o):
        x = producer.get(_key(o))
        if x is None or not x["ins"]:
            return _key(o)
        s = x["ins"][0]
        return s[4:] if s.startswith("<S>/") else "<B>/" + _key(s)

    def objects(x, seen_libs):
        out = []
        for p in x["ins"]:
            k = _key(p)
            y = producer.get(k)
            if y is not None and "COMPILER" in y["kind"]:
                out.append((src_of(k), fp(y)))
            elif y is not None and "STATIC_LIBRARY" in y["kind"] and k not in seen_libs:
                seen_libs.add(k)
                out += objects(y, seen_libs)
        return out

    for x in edges:
        fp(x)  # every action once: fills gen_read / under (what include directories expose) before the terminal scan
    artifacts, consumed = {}, set()
    for x in edges:
        for p in x["ins"] + x["impl"]:
            consumed.add(_key(p))
    for x in edges:
        if "EXECUTABLE_LINKER" not in x["kind"] and "SHARED_LIBRARY_LINKER" not in x["kind"] and "MODULE_LIBRARY_LINKER" not in x["kind"]:
            continue
        seen = set()
        objs = objects(x, seen)
        ext = []
        for t in x["vars"].get("LINK_LIBRARIES", "").split():
            k = _key(t)
            y = producer.get(k)
            if y is not None and "STATIC_LIBRARY" in y["kind"]:
                if k not in seen:
                    seen.add(k)
                    objs += objects(y, seen)
            elif y is not None:
                ext.append("E:" + fp(y))  # an import library of a DLL built here
            else:
                ext.append(t)
    # external (system / SDK import) libraries count as a set: their order only matters for a symbol two of them
    # define, which link.exe resolves the same way for import libraries of different DLLs; order changes are a note
        v = {k: s for k, s in sorted(x["vars"].items()) if k not in EXCL and k != "LINK_LIBRARIES"}
        other = sorted(inp(p) for p in x["impl"] if producer.get(_key(p)) is None or "STATIC_LIBRARY" not in producer[_key(p)]["kind"])
        link = hashlib.sha256(json.dumps([x["kind"], rules.get(x["rule"], ""), v, sorted(set(ext)), other], sort_keys=True).encode()).hexdigest()[:20]
        name = os.path.basename(x["outs"][0])
        base, dot, sfx = name.rpartition(".")
        artifacts[base if dot else name] = {"kind": "exe" if "EXECUTABLE" in x["kind"] else "dll", "out": x["outs"][0],
                                            "objects": objs, "sources": sorted({s for s, _ in objs}), "link": link, "ext": ext}
    for rel, (_, ps) in under.items():
        consumed.update(ps)
    terminals = {}
    for x in edges:
        if "LINKER" in x["kind"] or "COMPILER" in x["kind"]:
            continue
        for o in x["outs"]:
            if o not in consumed:
                terminals[o] = fp(x)
    return {"edges": edges, "producer": producer, "artifacts": artifacts, "gen": genh, "gen_read": gen_read,
            "terminals": terminals}


def _is_product(a):
    return not any(s.startswith(("tests/", "tools/")) for s, _ in a["objects"])


def compare(a, b, gpu_exes):
    """graph a (old) vs graph b (new) for the artifacts gpu_exes names (basenames without .exe)"""
    r = {"ok": True, "full": [], "sources": {}, "exe_changes": {}, "notes": []}
    for name in sorted(set(a["artifacts"]) | set(b["artifacts"])):
        ea, eb = a["artifacts"].get(name), b["artifacts"].get(name)
        kind = (eb or ea)["kind"]
        if kind == "dll":
            if ea is None or eb is None or ea["objects"] != eb["objects"] or ea["link"] != eb["link"]:
                r["full"].append(f"DLL {name}: {'new' if ea is None else 'removed' if eb is None else 'changed'} (who loads it is not in the graph)")
            continue
        if name not in gpu_exes:
            continue
        if eb is None:
            r["full"].append(f"executable {name} that a GPU step runs is no longer built")
            continue
        if ea is None:
            r["exe_changes"][name] = ["new executable (a GPU step runs it)"]
            continue
        ca = collections.Counter(f for _, f in ea["objects"])
        cb = collections.Counter(f for _, f in eb["objects"])
        diff = (ca - cb) + (cb - ca)
        srcs = sorted({s for s, f in ea["objects"] + eb["objects"] if f in diff})
        prod = _is_product(eb) and _is_product(ea)
        why = []
        if srcs:
            gen = [s for s in srcs if s.startswith("<B>/")]
            if gen:
                r["full"].append(f"{name}: generated source(s) {', '.join(gen[:3])} built differently (custom command / configure output)")
            r["sources"][name] = [s for s in srcs if not s.startswith("<B>/")]
            if not r["sources"][name]:
                del r["sources"][name]
            why.append(f"{len(srcs)} object(s) differ")
        elif [f for _, f in ea["objects"]] != [f for _, f in eb["objects"]]:
            # MSVC runs the dynamic initializers of different objects in link order, and nothing proves that no WHIRL
            # initializer depends on another object's (main session 2026-10-29: fail closed; a scan of the objects'
            # .CRT$XCU initializers could narrow this later)
            if prod:
                r["full"].append(f"product executable {name}: same objects, other link order (static-initialization order not proven irrelevant)")
            why.append("same objects, other link order (static-initialization order)")
        if ea["link"] == eb["link"] and ea.get("ext") != eb.get("ext"):
            r["notes"].append(f"{name}: same external libraries in another order / repeated")
        if ea["link"] != eb["link"]:
            if prod:
                r["full"].append(f"product executable {name}: link rule / flags / external libraries changed")
            why.append("link rule / flags / external libraries changed")
        if why and not prod:
            r["exe_changes"][name] = why
    ta, tb = a["terminals"], b["terminals"]
    for o in sorted(set(ta) | set(tb)):
        if o.endswith((".lib", ".obj")):
            continue  # a library / object nothing links: no runtime effect
        if ta.get(o) != tb.get(o):
            r["full"].append(f"build output {o} that no executable links {'is new' if o not in ta else 'is gone' if o not in tb else 'changed'} (its user is not in the graph)")
    for k in sorted(set(a["gen"]) | set(b["gen"])):
        if a["gen"].get(k) != b["gen"].get(k) and k not in a["gen_read"] and k not in b["gen_read"]:
            r["full"].append(f"configure-time file <B>/{k} changed and no build action reads it")
    return r


# ---------------------------------------------------------------------------------------------------------------------
# configure + load

def load(build_dir, S):
    bn = open(os.path.join(build_dir, "build.ninja"), encoding="utf-8", errors="replace").read()
    rn = open(os.path.join(build_dir, "CMakeFiles", "rules.ninja"), encoding="utf-8", errors="replace").read()
    gen = {}
    for dp, dns, fns in os.walk(build_dir):
        rel = os.path.relpath(dp, build_dir).replace("\\", "/")
        if rel == ".":
            dns[:] = [d for d in dns if d not in NOT_GEN]
            fns = [f for f in fns if f not in NOT_GEN]
        for f in fns:
            k = (f if rel == "." else f"{rel}/{f}")
            raw = open(os.path.join(dp, f), "rb").read()
            try:
                gen[k] = raw.decode("utf-8")
            except UnicodeDecodeError:
                gen[k] = raw
    return parse(bn, rn, S, build_dir, gen)


def export_rev(root, rev, dest):
    """the tree of rev into dest (git archive: reads the repository, writes nothing to it)"""
    r = subprocess.run(["git", "-C", root, "archive", "--format=tar", rev], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git archive {rev}: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
        t.extractall(dest, filter="data")


def configure(src, bdir, tool_root=REPO):
    os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, ".cmake_graph"), "w").write("configure-only directory of tools/impact/cmake_graph.py\n")
    bat = os.path.join(bdir, "..", os.path.basename(bdir) + "_cfg.bat")
    vc = os.path.join(tool_root, "tools", "safety", "vcenv.bat")
    with open(bat, "w") as fh:
        fh.write(f'@echo off\r\ncall "{vc}" >nul\r\nif errorlevel 1 exit /b 1\r\n'
                 f'cmake -S "{src}" -B "{bdir}" -G Ninja -DCMAKE_BUILD_TYPE=Release >"{bdir}.cfg.log" 2>&1\r\nexit /b %ERRORLEVEL%\r\n')
    rc = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace").returncode
    if rc != 0:
        tail = ""
        try:
            tail = " | ".join(open(f"{bdir}.cfg.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-4:])
        except OSError:
            pass
        raise RuntimeError(f"configure of {src} failed (exit {rc}): {tail[:300]}")


def analyze_revs(root, old_rev, new_rev, work, gpu_exes, keep=False):
    """configure old_rev and new_rev (None = the working tree of root) under work and compare(); never raises:
    {"ok": False, "why": ...} when either side cannot be configured (test_change keeps the full matrix then)"""
    os.makedirs(work, exist_ok=True)
    made = []
    try:
        sides = []
        for tag, rev in (("a", old_rev), ("b", new_rev)):
            bdir = os.path.join(work, f"cfg_{tag}")
            for p in (bdir, os.path.join(work, f"src_{tag}")):
                if os.path.isdir(p):
                    shutil.rmtree(p)
            if rev is None:
                src = root
            else:
                src = os.path.join(work, f"src_{tag}")
                os.makedirs(src)
                made.append(src)
                export_rev(root, rev, src)
            made.append(bdir)
            configure(src, bdir)
            sides.append(load(bdir, src))
        r = compare(sides[0], sides[1], set(gpu_exes))
        r["configured"] = [old_rev, new_rev or "worktree"]
        return r
    except Exception as e:  # noqa: BLE001 - any failure = cannot be sure = full
        return {"ok": False, "why": f"{type(e).__name__}: {e}", "full": [], "sources": {}, "exe_changes": {}, "notes": []}
    finally:
        if not keep:
            for p in made:
                shutil.rmtree(p, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new")
    ap.add_argument("--work", required=True)
    ap.add_argument("--gpu-exes", default="whirl,whirl-server,whirl-server-gate,whirl-kernel-test")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    r = analyze_revs(os.path.abspath(a.root), a.old, a.new, os.path.abspath(a.work), a.gpu_exes.split(","), a.keep)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
