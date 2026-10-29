# SPDX-License-Identifier: Apache-2.0
"""Tests of tools/impact/cmake_graph.py (arch025-modsel3, PLAN 3.8): a non-list CMake edit is attributed through the
generated Ninja action graph of the two revisions (what each executable is linked from, how each object is compiled);
anything the graph cannot attribute stays full. Synthetic build.ninja texts in the CMake Ninja generator's shape; CPU only,
no CMake, no git, no GPU.

    python -I tools/impact/cmake_graph_test.py        exit 0 = all pass
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmake_graph as cg  # noqa: E402

fails = 0
n = 0


def check(name, cond, info=""):
    global fails, n
    n += 1
    if not cond:
        fails += 1
        print(f"FAIL {name} {info}")


S = "C:\\R\\whirl"          # source root of both revisions (the parser normalizes it to <S>)
B1, B2 = "C:\\T\\cfg_a", "C:\\T\\cfg_b"  # two configure directories (normalized to <B>)
RULES = """rule CXX_COMPILER__{t}_unscanned_Release
  deps = msvc
  command = cl.exe /nologo /TP $DEFINES $INCLUDES $FLAGS /showIncludes /Fo$out /Fd$TARGET_COMPILE_PDB /FS -c $in
  description = Building CXX object $out

rule CXX_STATIC_LIBRARY_LINKER__{t}_Release
  command = cmd.exe /C "$PRE_LINK && lib.exe /nologo $LINK_FLAGS /out:$TARGET_FILE $in && $POST_BUILD"

rule CXX_EXECUTABLE_LINKER__{t}_Release
  command = cmd.exe /C "$PRE_LINK && link.exe /nologo $in /out:$TARGET_FILE $LINK_FLAGS $LINK_LIBRARIES && $POST_BUILD"
"""


def esc(p):
    return p.replace(":", "$:")


def obj(t, src, defines="-DNOMINMAX", B=None):
    o = f"CMakeFiles\\{t}.dir\\{src.replace('/', chr(92))}.obj"
    return o, (f"build {o}: CXX_COMPILER__{t}_unscanned_Release {esc(S)}\\{src.replace('/', chr(92))} || cmake_object_order_depends_target_{t}\n"
               f"  CONFIG = Release\n  DEFINES = {defines}\n  FLAGS = /O2 -std:c++20\n  INCLUDES = -I{S}\\include -I{S}\\src -I{B}\\generated\n"
               f"  OBJECT_DIR = CMakeFiles\\{t}.dir\n  OBJECT_FILE_DIR = CMakeFiles\\{t}.dir\\x\n  TARGET_COMPILE_PDB = CMakeFiles\\{t}.dir\\\n"
               f"  TARGET_PDB = {t}.pdb\n\n")


def lib(t, srcs, B, defines=None):
    txt, objs = "", []
    for s in srcs:
        o, e = obj(t, s, (defines or {}).get(s, "-DNOMINMAX"), B)
        objs.append(o)
        txt += e
    txt += (f"build {t}.lib: CXX_STATIC_LIBRARY_LINKER__{t}_Release {' '.join(objs)}\n  CONFIG = Release\n  LINK_FLAGS = /machine:x64\n"
            f"  OBJECT_DIR = CMakeFiles\\{t}.dir\n  POST_BUILD = cd .\n  PRE_LINK = cd .\n  TARGET_FILE = {t}.lib\n\n")
    return txt, RULES.format(t=t)


def exe(t, srcs, libs, B, out=None, ext="kernel32.lib", defines=None, flags="/machine:x64"):
    txt, objs = "", []
    for s in srcs:
        o, e = obj(t, s, (defines or {}).get(s, "-DNOMINMAX"), B)
        objs.append(o)
        txt += e
    out = out or f"{t}.exe"
    txt += (f"build {out}: CXX_EXECUTABLE_LINKER__{t}_Release {' '.join(objs)} | {' '.join(libs)}\n  CONFIG = Release\n"
            f"  LINK_FLAGS = {flags}\n  LINK_LIBRARIES = {'  '.join(libs)}  {ext}\n  OBJECT_DIR = CMakeFiles\\{t}.dir\n"
            f"  POST_BUILD = cd .\n  PRE_LINK = cd .\n  TARGET_FILE = {out}\n  TARGET_IMPLIB = {t}.lib\n\n")
    return txt, RULES.format(t=t)


def graph(parts, B, gen=None):
    bt = "".join(p[0] for p in parts) + "build all: phony whirl.exe\n"
    rt = "".join(p[1] for p in parts)
    return cg.parse(bt, rt, S, B, gen if gen is not None else {"generated/whirl/version.h": "v1"})


def base(B, **kw):
    return [lib("whirl_model", ["src/model/loader.cpp", "src/model/config.cpp"], B, kw.get("mdef")),
            exe("whirl-cli", kw.get("cli", ["src/cli/main.cpp"]), kw.get("cli_libs", ["whirl_model.lib"]), B, out="whirl.exe",
                ext=kw.get("ext", "kernel32.lib"), defines=kw.get("cdef")),
            exe("whirl-server-gate", ["tests/server/server_gate.cpp"], kw.get("sg_libs", []), B),
            exe("whirl-tests", ["tests/unit/unit_tests.cpp"], ["whirl_model.lib"], B)] + kw.get("extra", [])


GPU = {"whirl", "whirl-server-gate"}  # artifact names the GPU steps run (release_gate.ps1)


def cmp(a, b):
    return cg.compare(a, b, GPU)


# 1. the same graph from two configure directories: nothing changed (paths normalized)
r = cmp(graph(base(B1), B1), graph(base(B2), B2))
check("identical graphs", r["ok"] and not r["full"] and not r["sources"] and not r["exe_changes"], r)
# 2. a removed compile definition of one source in a lib whirl.exe links: that source counts as changed (its module)
r = cmp(graph(base(B1), B1), graph(base(B2, mdef={"src/model/config.cpp": "-DNOMINMAX -DX=1"}), B2))
check("define change -> source of whirl.exe", r["ok"] and r["sources"].get("whirl") == ["src/model/config.cpp"] and not r["full"], r)
check("define change in a CPU-only exe is not reported for GPU", "whirl-tests" not in r["sources"], r)
# 3. sources moved from the exe into a new static library with the same flags: same objects, no source changed
moved = base(B2, cli=["src/cli/main.cpp"], cli_libs=["whirl_cli_core.lib", "whirl_model.lib"],
             extra=[lib("whirl_cli_core", ["src/cli/cmd_chat.cpp"], B2)])
before = base(B1, cli=["src/cli/main.cpp", "src/cli/cmd_chat.cpp"])
r = cmp(graph(before, B1), graph(moved, B2))
check("new add_library with moved sources: no source, no full", r["ok"] and not r["sources"] and not r["full"] and not r["exe_changes"], r)
check("moved sources, same order: no note", not r["notes"], r["notes"])
moved2 = base(B2, cli=["src/cli/main.cpp"], cli_libs=["whirl_model.lib", "whirl_cli_core.lib"], extra=[lib("whirl_cli_core", ["src/cli/cmd_chat.cpp"], B2)])
r = cmp(graph(before, B1), graph(moved2, B2))
check("moved sources, other link order: product exe -> full (static-initialization order)", r["ok"] and not r["sources"] and any("link order" in x for x in r["full"]), r)
# 4. a new test executable no GPU step runs: nothing for the GPU
r = cmp(graph(base(B1), B1), graph(base(B2, extra=[exe("whirl-new-tests", ["tests/unit/new_tests.cpp"], ["whirl_model.lib"], B2)]), B2))
check("new CPU test exe: no GPU change", r["ok"] and not r["full"] and not r["sources"] and not r["exe_changes"], r)
# 5. target_link_libraries edit of a GPU test exe: that exe changed (its GPU steps), not full
r = cmp(graph(base(B1), B1), graph(base(B2, sg_libs=["whirl_model.lib"]), B2))
check("link edit of a GPU test exe -> exe change", r["ok"] and "whirl-server-gate" in r["exe_changes"] and not r["full"], r)
check("its new objects are product sources of that exe", r["sources"].get("whirl-server-gate") == ["src/model/config.cpp", "src/model/loader.cpp"], r)
# 6. a product exe links another system library: cannot attribute -> full
r = cmp(graph(base(B1), B1), graph(base(B2, ext="kernel32.lib dxgi.lib"), B2))
check("product exe link libraries -> full", r["ok"] and any("whirl" in x for x in r["full"]), r)
# 7. a configure-time generated header in an include dir of whirl.exe's objects changes: every such object is changed,
# and a generated source of a GPU exe is full
r = cmp(graph(base(B1), B1, {"generated/whirl/version.h": "v1"}), graph(base(B2), B2, {"generated/whirl/version.h": "v2"}))
check("generated header -> objects that see it", r["ok"] and "src/cli/main.cpp" in r["sources"].get("whirl", []), r)
cc = (f"build kernels\\k.cpp | ${{cmake_ninja_workdir}}kernels\\k.cpp: CUSTOM_COMMAND {esc(S)}\\kernels\\a.hip\n"
      f"  COMMAND = cmd.exe /C \"cd /D {B1} && clang.exe -x hip --offload-arch=gfx1201 {S}\\kernels\\a.hip -o {B1}\\kernels\\k.cpp\"\n  DESC = gen\n\n")


def with_kernel(B, arch="gfx1201"):
    t = cc.replace(B1, B).replace("gfx1201", arch)
    o = (f"build CMakeFiles\\whirl-cli.dir\\kernels\\k.cpp.obj: CXX_COMPILER__whirl-cli_unscanned_Release kernels\\k.cpp\n"
         f"  DEFINES = -DNOMINMAX\n  FLAGS = /O2\n  INCLUDES = -I{S}\\include\n  OBJECT_DIR = CMakeFiles\\whirl-cli.dir\n\n")
    parts = base(B)
    parts[1] = (parts[1][0].replace("CXX_EXECUTABLE_LINKER__whirl-cli_Release ", "CXX_EXECUTABLE_LINKER__whirl-cli_Release CMakeFiles\\whirl-cli.dir\\kernels\\k.cpp.obj "), parts[1][1])
    return [(t + o, "rule CUSTOM_COMMAND\n  command = $COMMAND\n")] + parts


r = cmp(graph(with_kernel(B1), B1), graph(with_kernel(B2), B2))
check("custom command, same command: nothing", r["ok"] and not r["full"] and not r["sources"], r)
r = cmp(graph(with_kernel(B1), B1), graph(with_kernel(B2, "gfx1200"), B2))
check("custom command change feeding a GPU exe -> full (generated source)", r["ok"] and r["full"], r)
# 8. an exe a GPU step runs disappears -> full; one appears (a GPU step names it) -> exe change
r = cmp(graph(base(B1), B1), graph([p for p in base(B2) if "whirl-server-gate" not in p[0]], B2))
check("GPU exe removed -> full", r["ok"] and any("whirl-server-gate" in x for x in r["full"]), r)
r = cmp(graph([p for p in base(B1) if "whirl-server-gate" not in p[0]], B1), graph(base(B2), B2))
check("GPU exe added -> exe change", r["ok"] and "whirl-server-gate" in r["exe_changes"] and not r["full"], r)
# 9. a product exe gets a new source: that source
r = cmp(graph(base(B1), B1), graph(base(B2, cli=["src/cli/main.cpp", "src/cli/cmd_new.cpp"]), B2))
check("new source in whirl.exe", r["ok"] and r["sources"].get("whirl") == ["src/cli/cmd_new.cpp"] and not r["full"], r)
# 10. product exe link flags -> full; test exe link flags -> exe change
r = cmp(graph(base(B1), B1), graph(base(B2) [:1] + [exe("whirl-cli", ["src/cli/main.cpp"], ["whirl_model.lib"], B2, out="whirl.exe", flags="/machine:x64 /STACK:8000000")] + base(B2)[2:], B2))
check("product exe link flags -> full", r["ok"] and r["full"], r)
# 11. a terminal artifact that is no exe (e.g. a copied DLL) changes -> full (nothing tells who uses it)
dll = lambda B, v: (f"build rocm_x.dll: CUSTOM_COMMAND {esc(S)}\\redist\\rocm_x.dll\n  COMMAND = copy {S}\\redist\\rocm_x.dll {B}\\rocm_x.dll {v}\n\n", "rule CUSTOM_COMMAND\n  command = $COMMAND\n")
r = cmp(graph(base(B1) + [dll(B1, "")], B1), graph(base(B2) + [dll(B2, "/Y")], B2))
check("changed terminal non-exe artifact -> full", r["ok"] and any("rocm_x.dll" in x for x in r["full"]), r)
r = cmp(graph(base(B1) + [dll(B1, "")], B1), graph(base(B2) + [dll(B2, "")], B2))
check("same terminal non-exe artifact -> nothing", r["ok"] and not r["full"], r)
# 12. a configure-time file nothing reads (by input or include dir) that changes -> full
r = cmp(graph(base(B1), B1, {"generated/whirl/version.h": "v1", "data/x.txt": "1"}), graph(base(B2), B2, {"generated/whirl/version.h": "v1", "data/x.txt": "2"}))
check("unread configure-time file changed -> full", r["ok"] and any("data/x.txt" in x for x in r["full"]), r)
# 13. parser: escapes, implicit / order-only inputs, path normalization (both slash forms, drive-letter case)
g = graph(base(B1), B1)
e = g["artifacts"].get("whirl")
check("parser finds whirl.exe from whirl-cli", e is not None and e["kind"] == "exe", sorted(g["artifacts"]))
check("normalized source keys", "src/cli/main.cpp" in (e or {}).get("sources", []), (e or {}).get("sources"))
check("norm both slash forms", cg.norm("c:/r/whirl/src/a.cpp and C:\\R\\whirl\\src\\b.cpp", S, B1) == "<S>/src/a.cpp and <S>/src/b.cpp",
      cg.norm("c:/r/whirl/src/a.cpp and C:\\R\\whirl\\src\\b.cpp", S, B1))
# 14. a test source of a GPU test exe compiled with other flags: reported as that exe's source (test_change maps it)
sg2 = base(B2)
sg2[2] = exe("whirl-server-gate", ["tests/server/server_gate.cpp"], [], B2, defines={"tests/server/server_gate.cpp": "-DNOMINMAX -DSG=2"})
r = cmp(graph(base(B1), B1), graph(sg2, B2))
check("test source flags of a GPU test exe", r["ok"] and r["sources"].get("whirl-server-gate") == ["tests/server/server_gate.cpp"], r)
# 15. found on real commits (1dad884): CMake names a custom command's depfile by a hash of absolute paths (two configure
# directories -> two names for one command); the same system libraries in another order are a note, not a link change
h1, h2 = "a" * 64, "b" * 64
check("depfile hash normalized", cg.norm(f"{B1}\\CMakeFiles\\d\\{h1}.d", S, B1) == cg.norm(f"{B2}\\CMakeFiles\\d\\{h2}.d", S, B2),
      (cg.norm(f"{B1}\\CMakeFiles\\d\\{h1}.d", S, B1), cg.norm(f"{B2}\\CMakeFiles\\d\\{h2}.d", S, B2)))
r = cmp(graph(base(B1, ext="ws2_32.lib kernel32.lib"), B1), graph(base(B2, ext="kernel32.lib ws2_32.lib"), B2))
check("system library order -> note only", r["ok"] and not r["full"] and any("external libraries" in x for x in r["notes"]), r)
# 16. an action output under an include directory of an object (a custom command copying a header into <B>/iso/...):
# consumed, not a terminal; its change changes the objects that see it
cp = lambda B, v: (f"build iso\\m\\whirl\\version.h: CUSTOM_COMMAND generated\\whirl\\version.h\n  COMMAND = copy {v}\n\n", "rule CUSTOM_COMMAND\n  command = $COMMAND\n")
def iso(B, v):
    parts = base(B)
    o, e = obj("whirl-server-gate", "tests/server/server_gate.cpp", B=B)
    e = e.replace(f"-I{B}\\generated", f"-I{B}\\generated -I{B}\\iso\\m")
    parts[2] = (e + parts[2][0].split("\n\n", 1)[1], parts[2][1])
    return parts + [cp(B, v)]
r = cmp(graph(iso(B1, "1"), B1), graph(iso(B2, "1"), B2))
check("copied header under an include dir: not terminal, same -> nothing", r["ok"] and not r["full"] and not r["sources"], r)
r = cmp(graph(iso(B1, "1"), B1), graph(iso(B2, "2"), B2))
check("copied header changes -> the objects that include it", r["ok"] and not r["full"] and r["sources"].get("whirl-server-gate") == ["tests/server/server_gate.cpp"], r)
print(f"cmake_graph_test: {n - fails}/{n} passed")
sys.exit(1 if fails else 0)
