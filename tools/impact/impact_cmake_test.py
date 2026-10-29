# SPDX-License-Identifier: Apache-2.0
"""Tests of the CMake source-list rule (arch025-modsel2, PLAN 3.3): a CMakeLists.txt / cmake/*.cmake edit whose changed
lines are ONLY source-path entries (plus blank / comment lines) counts as the module(s) owning those sources; any other
CMake edit stays build-wide (full gate). CPU only, no git, no GPU.

    python -I tools/impact/impact_cmake_test.py        exit 0 = all pass
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impact  # noqa: E402

fails = 0
n = 0


def check(name, cond, info=""):
    global fails, n
    n += 1
    if not cond:
        fails += 1
        print(f"FAIL {name} {info}")


L = impact.cmake_list_edit
F = "cmake/server.cmake"
# --- line classifier: pass cases (list-only)
check("add one source", L(F, [], ["  ${CMAKE_SOURCE_DIR}/src/cache/x.cpp"]) == ["src/cache/x.cpp"], L(F, [], ["  ${CMAKE_SOURCE_DIR}/src/cache/x.cpp"]))
check("remove one source", L(F, ["  src/cache/x.cpp"], []) == ["src/cache/x.cpp"])
check("move the closing paren", L(F, ["  src/a/x.cpp)"], ["  src/a/x.cpp", "  src/a/y.cpp)"]) == ["src/a/x.cpp", "src/a/y.cpp"],
      L(F, ["  src/a/x.cpp)"], ["  src/a/x.cpp", "  src/a/y.cpp)"]))
check("two sources on one line + trailing comment", L(F, [], ["  src/a/x.cpp src/a/y.h  # new"]) == ["src/a/x.cpp", "src/a/y.h"])
check("comment / blank only", L(F, ["# old words"], ["# new words", ""]) == [])
check("CMAKE_CURRENT_LIST_DIR/.. in cmake/", L(F, [], ["  ${CMAKE_CURRENT_LIST_DIR}/../src/a/x.cpp"]) == ["src/a/x.cpp"])
check("test source", L("CMakeLists.txt", [], ["  tests/cache/cache_tests.cpp"]) == ["tests/cache/cache_tests.cpp"])
# --- block cases (not list-only -> None = build-wide)
check("new target", L(F, [], ["add_library(whirl_x STATIC src/a/x.cpp)"]) is None)
check("link libraries", L(F, [], ["target_link_libraries(a PRIVATE b)"]) is None)
check("flag", L(F, ["  -DFOO=1"], ["  -DFOO=2"]) is None)
check("option word in list", L(F, [], ["  PRIVATE"]) is None)
check("bracket comment opens", L(F, [], ["#[[ src/a/x.cpp"]) is None)
check("bracket comment closes", L(F, ["]]"], []) is None)
check("kernel source", L(F, [], ["  kernels/gemv.hip"]) is None)
check("hip extension under src", L(F, [], ["  src/a/x.hip"]) is None)
check("paren balance changes", L(F, ["  src/a/x.cpp)"], ["  src/a/x.cpp"]) is None)
check("quoted path", L(F, [], ['  "src/a/x.cpp"']) is None)
check("other variable", L(F, [], ["  ${WHIRL_SRC}/a/x.cpp"]) is None)
check("CURRENT_LIST_DIR/.. from the root file leaves the tree", L("CMakeLists.txt", [], ["  ${CMAKE_CURRENT_LIST_DIR}/../src/a/x.cpp"]) is None)
check("unreadable side", L(F, None, ["  src/a/x.cpp"]) is None)
check("dot-dot path", L(F, [], ["  src/../kernels/a.cpp"]) is None)

# --- module_gate with the report's cmake_lists
MODS = {
    "sampling": {"paths": ["src/sampling/*"], "steps": ["cpu_server", "srv_balance"]},
    "protocol": {"paths": ["src/server/stream.*"], "steps": ["cpu_server", "golden_swift"]},
    "contract.server": {"paths": ["include/whirl/server_model.h"], "steps": ["cpu_server"], "full": True},
}


def ch(*fs, renames=None):
    out = {f: {"old_path": f, "new": [1], "old": []} for f in fs}
    for new, old in (renames or {}).items():
        out[new] = {"old_path": old, "new": [1], "old": [1]}
    return out


def gate(changes, lists):
    r = impact.module_gate(changes, MODS, rep={"cmake_lists": lists, "gpus": ["gfx1201", "gfx1151"]})
    return r["module_gate"], r["modules"]


g, mm = gate(ch("cmake/server.cmake", "src/sampling/new.cpp"), {"cmake/server.cmake": ["src/sampling/new.cpp"]})
check("list-only edit of one module = that module", g.startswith("release_gate.ps1 -Module sampling") and mm == {"sampling": ["cmake/server.cmake", "src/sampling/new.cpp"]}, (g, mm))
g, mm = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": ["src/sampling/gone.cpp"]})
check("removing a module's source (file kept) = that module", g.startswith("release_gate.ps1 -Module sampling"), g)
g, _ = gate(ch("cmake/server.cmake"), None)
check("same file, not list-only (no cmake_lists entry) = build-wide full", g.startswith("full gate") and "cmake/server.cmake" in g, g)
g, _ = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": ["src/nowhere/x.cpp"]})
check("listed source in no module = full", g.startswith("full gate") and "src/nowhere/x.cpp" in g, g)
g, mm = gate(ch("cmake/server.cmake", renames={"src/sampling/b.cpp": "src/old/b.cpp"}),
             {"cmake/server.cmake": ["src/old/b.cpp", "src/sampling/b.cpp"]})
check("moved source: old path follows the rename", g.startswith("release_gate.ps1 -Module sampling"), (g, mm))
g, _ = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": ["src/sampling/a.cpp", "src/server/stream.cpp"]})
check("two modules' sources = cross-module (impact.py rule)", "cross-module" in g, g)
g, _ = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": ["include/whirl/server_model.h"]})
check("contract module's source = full (contract)", "contract module" in g, g)
g, mm = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": ["tests/cache/cache_tests.cpp"]})
check("test source only = no product code", g.startswith("none") and not mm, (g, mm))
g, _ = gate(ch("cmake/server.cmake"), {"cmake/server.cmake": []})
check("comment-only edit = no product code", g.startswith("none"), g)
g, _ = gate(ch("build.bat"), {"build.bat": ["src/sampling/a.cpp"]})
check("build.bat never list-only (cmake_lists entry ignored)", g.startswith("full gate"), g)
g, _ = gate(ch("CMakeLists.txt", "cmake/server.cmake"), {"cmake/server.cmake": ["src/sampling/a.cpp"]})
check("one list-only + one other CMake file = full", g.startswith("full gate") and "CMakeLists.txt" in g, g)

print(f"impact cmake test: {n - fails}/{n} ok")
sys.exit(1 if fails else 0)
