# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests for tools/gates/cmake_modules_check.py on synthetic trees with a synthetic CMake File API reply."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "cmake_modules_check.py")


def tree(files, mods, targets, ninja="", pending=None, manifests=None):
    """files: {path: text}; targets: {name: (type, [compiled paths], [dependency names])}; a path starting with "@/" is
    under the build directory, every other compiled path is under the source tree"""
    base = tempfile.mkdtemp(prefix="cmkmod_")
    root, build = os.path.join(base, "src_tree"), os.path.join(base, "build")
    for f, text in files.items():
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    for f, text in (manifests or {}).items():
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
    os.makedirs(os.path.join(root, "tools", "gates"), exist_ok=True)
    json.dump({"modules": mods}, open(os.path.join(root, "tools", "gates", "modules.json"), "w"))
    if pending is not None:
        json.dump({"pending": pending}, open(os.path.join(root, "tools", "gates", "census_pending.json"), "w"))
    rdir = os.path.join(build, ".cmake", "api", "v1", "reply")
    os.makedirs(rdir)
    tl = []
    for i, (name, (typ, comp, deps)) in enumerate(targets.items()):
        srcs = []
        for p in comp:
            if p.startswith("@/"):
                srcs.append({"path": os.path.join(build, p[2:]).replace("\\", "/"), "compileGroupIndex": 0, "isGenerated": "/iso/" not in p})
            else:
                srcs.append({"path": p, "compileGroupIndex": 0})
        tj = {"name": name, "id": f"{name}::@x", "type": typ, "sources": srcs, "dependencies": [{"id": f"{d}::@x"} for d in deps]}
        fn = f"target-{i}.json"
        json.dump(tj, open(os.path.join(rdir, fn), "w"))
        tl.append({"name": name, "id": f"{name}::@x", "jsonFile": fn})
    json.dump({"paths": {"source": root.replace("\\", "/"), "build": build.replace("\\", "/")}, "configurations": [{"name": "Release", "targets": tl}]},
              open(os.path.join(rdir, "codemodel-v2-0.json"), "w"))
    json.dump({"objects": [{"kind": "codemodel", "version": {"major": 2}, "jsonFile": "codemodel-v2-0.json"}]},
              open(os.path.join(rdir, "index-0.json"), "w"))
    open(os.path.join(build, "build.ninja"), "w").write(ninja.replace("@ROOT", root.replace("\\", "/")))
    return base, root, build


def run(root, build, *extra):
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root, "--build", build] + list(extra), capture_output=True, text=True)
    return p.returncode, p.stdout


MODS = {"a": {"paths": ["src/a/*"]}, "k": {"paths": ["kernels/*"]}, "iso_a": {"paths": ["src/iso_a.cpp"]}}
FILES = {"src/a/x.cpp": "// x\n", "src/a/x.h": "// h\n", "src/iso_a.cpp": "// i\n", "kernels/root.hip": '#include "frag.hip"\n',
         "kernels/frag.hip": "// f\n", "tests/t.cpp": "// t\n"}
NINJA = 'build k.hsaco: CUSTOM_COMMAND\n  COMMAND = cmd /C "clang.exe -x hip --offload-arch=gfx1201 -o @ROOT/../build/k.hsaco @ROOT/kernels/root.hip"\n'
MAN = {"tests/iso/iso_a/isolation.txt": "module iso_a\nexe whirl-iso-a-tests\nsrc src/iso_a.cpp\nhdr src/a/x.h\ntestsrc tests/t.cpp\n"}
ISO_T = {"whirl_iso_iso_a": ("OBJECT_LIBRARY", ["@/iso/iso_a/mod/iso_a.cpp"], []),
         "whirl-iso-a-tests": ("EXECUTABLE", ["@/iso/iso_a/test/t.cpp"], ["whirl_iso_iso_a"])}


def T(**kw):
    t = {"lib": ("STATIC_LIBRARY", ["src/a/x.cpp", "src/iso_a.cpp", "@/kernels/gen.cpp"], []), "tst": ("EXECUTABLE", ["tests/t.cpp"], ["lib"])}
    t.update(ISO_T)
    t.update(kw)
    return t


def drop(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


CASES = [
    # (name, files, mods, targets, ninja, pending, manifests, extra args, want rc, substring)
    ("pass: every module source compiled, kernel fragments reached from the device TU, iso targets match",
     FILES, MODS, T(), NINJA, None, MAN, [], 0, "cmake modules: ok"),
    ("block: a module .cpp no target compiles", dict(FILES, **{"src/a/y.cpp": "// y\n"}), MODS, T(), NINJA, None, MAN, [], 1,
     "module a: src/a/y.cpp is compiled by no CMake target"),
    ("pass: a module .cpp #included by a compiled source", dict(FILES, **{"src/a/y.cpp": "// y\n", "src/a/x.cpp": '#include "y.cpp"\n'}),
     MODS, T(), NINJA, None, MAN, [], 0, "cmake modules: ok"),
    ("block: a kernel fragment no device TU includes", dict(FILES, **{"kernels/lost.hip": "// l\n"}), MODS, T(), NINJA, None, MAN, [], 1,
     "module k: kernels/lost.hip is compiled by no CMake target"),
    ("block: no device compile in build.ninja (the fragments are then unbuilt)", FILES, MODS, T(), "", None, MAN, [], 1,
     "kernels/root.hip is compiled by no CMake target"),
    ("pass: a module header no target compiles", dict(FILES, **{"src/a/z.h": "// z\n"}), MODS, T(), NINJA, None, MAN, [], 0, "cmake modules: ok"),
    ("block: a compiled product source in no module", dict(FILES, **{"src/b.cpp": "// b\n"}), MODS,
     T(lib=("STATIC_LIBRARY", ["src/a/x.cpp", "src/iso_a.cpp", "src/b.cpp"], [])), NINJA, None, MAN, [], 1, "src/b.cpp is compiled by CMake and is in 0 modules"),
    ("pass: a compiled source in no module but census-pending (WARN)", dict(FILES, **{"src/b.cpp": "// b\n"}), MODS,
     T(lib=("STATIC_LIBRARY", ["src/a/x.cpp", "src/iso_a.cpp", "src/b.cpp"], [])), NINJA, [{"glob": "src/b.cpp", "owner": "lane 9"}], MAN, [], 0,
     "WARN src/b.cpp is compiled and still census-pending"),
    ("block: a census-pending compiled source in release mode", dict(FILES, **{"src/b.cpp": "// b\n"}), MODS,
     T(lib=("STATIC_LIBRARY", ["src/a/x.cpp", "src/iso_a.cpp", "src/b.cpp"], [])), NINJA, [{"glob": "src/b.cpp", "owner": "lane 9"}], MAN,
     ["--release"], 1, "src/b.cpp is compiled by CMake and is in 0 modules"),
    ("block: a compiled source in two modules", FILES, dict(MODS, a2={"paths": ["src/a/x.cpp"]}), T(), NINJA, None, MAN, [], 1,
     "src/a/x.cpp is compiled by CMake and is in 2 modules: a, a2"),
    ("block: a compiled tool source outside the tree", FILES, MODS, T(odd=("EXECUTABLE", ["C:/elsewhere/z.cpp"], [])), NINJA, None, MAN, [], 1,
     "outside the source tree"),
    ("block: iso library compiles a file its manifest does not list", dict(FILES, **{"src/a/w.cpp": "// w\n"}), MODS,
     T(whirl_iso_iso_a=("OBJECT_LIBRARY", ["@/iso/iso_a/mod/iso_a.cpp", "@/iso/iso_a/mod/a/w.cpp"], []),
       lib=("STATIC_LIBRARY", ["src/a/x.cpp", "src/a/w.cpp", "src/iso_a.cpp"], [])), NINJA, None, MAN, [], 1,
     "compiles mod/a/w.cpp, which the manifest does not list"),
    ("block: iso library misses a manifest src line", FILES, MODS, T(whirl_iso_iso_a=("OBJECT_LIBRARY", [], [])), NINJA, None,
     {"tests/iso/iso_a/isolation.txt": MAN["tests/iso/iso_a/isolation.txt"] + "src src/a/x.cpp\n"}, [], 1, "does not compile mod/iso_a.cpp"),
    ("block: manifest exe target missing", FILES, MODS, drop(T(), "whirl-iso-a-tests"), NINJA, None, MAN, [], 1, "no CMake target whirl-iso-a-tests"),
    ("block: iso test program does not depend on the iso library", FILES, MODS,
     T(**{"whirl-iso-a-tests": ("EXECUTABLE", ["@/iso/iso_a/test/t.cpp"], [])}), NINJA, None, MAN, [], 1, "does not depend on whirl_iso_iso_a"),
    ("block: an iso target no manifest names", FILES, MODS, T(whirl_iso_ghost=("OBJECT_LIBRARY", ["@/iso/ghost/mod/g.cpp"], [])), NINJA, None, MAN,
     [], 1, "target whirl_iso_ghost compiles iso copies but no isolation manifest names it"),
    ("block: manifest module not in modules.json", FILES, drop(MODS, "iso_a"), T(), NINJA, None, MAN, [], 1,
     "module is not in tools/gates/modules.json"),
    # arch025 lane 2 merge (2026-10-29): one manifest may isolate a group of modules (tests/iso/forward covers
    # forward.dispatch .. forward.vision_rows + forward.state): its module line names the group
    ("pass: manifest module names a group of modules (iso_a.x, iso_a.y)", FILES,
     dict(drop(MODS, "iso_a"), **{"iso_a.x": {"paths": ["src/iso_a.cpp"]}, "iso_a.y": {"paths": []}}), T(), NINJA, None, MAN, [], 0,
     "cmake modules: ok"),
    ("block: a module that only shares the prefix is no group (iso_ab)", FILES,
     dict(drop(MODS, "iso_a"), **{"iso_ab": {"paths": ["src/iso_a.cpp"]}}), T(), NINJA, None, MAN, [], 1,
     "module is not in tools/gates/modules.json"),
    ("block: wrong target type for the iso library", FILES, MODS, T(whirl_iso_iso_a=("STATIC_LIBRARY", ["@/iso/iso_a/mod/iso_a.cpp"], [])), NINJA,
     None, MAN, [], 1, "is STATIC_LIBRARY, want OBJECT_LIBRARY"),
]


def setup_cases():
    out = []
    base, root, build = tree(FILES, MODS, T(), NINJA, None, MAN)
    shutil.rmtree(os.path.join(build, ".cmake"))
    rc, o = run(root, build)
    out.append(("setup: no File API reply -> exit 2", rc == 2 and "no File API reply" in o, o))
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root, "--configure", build], capture_output=True, text=True)
    out.append(("setup: --configure refuses an existing directory it does not own", p.returncode == 2 and "is not this tool's directory" in p.stdout,
                p.stdout))
    shutil.rmtree(base, ignore_errors=True)
    return out


fails = 0
n = 0
for name, files, mods, targets, ninja, pending, mans, extra, want, sub in CASES:
    base, root, build = tree(files, mods, targets, ninja, pending, mans)
    rc, out = run(root, build, *extra)
    ok = rc == want and sub in out
    fails += not ok
    n += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  rc {rc} out {out[-400:]!r}"))
    shutil.rmtree(base, ignore_errors=True)
for name, ok, out in setup_cases():
    fails += not ok
    n += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  out {out[-300:]!r}"))
print(f"cmake modules check test: {n - fails}/{n} ok")
sys.exit(1 if fails else 0)
