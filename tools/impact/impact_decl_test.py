# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests of impact.py's measure-tools P1 rules (CPU, < 1 s): declared-name selection (B1.1), moved
declarations, #include / #define / #if lines (B1.2), behaviour classes and the GPU checks outside release_gate (B1.3),
map defects (B1.4). Each case runs impact.analyze() in-process on a two-revision fake repo (impact.show patched).

    python -I tools/impact/impact_decl_test.py        exit 0 = "impact decl test: N/N ok"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impact  # noqa: E402

ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


HDR_OLD = """#pragma once
struct Model {
    std::vector<DevPtr> allocations;
    int other = 0;
};
"""
HDR_NEW = """#pragma once
struct Model {
    std::vector<DevBuf> allocations;
    int other = 0;
};
"""
CPP = """#include "model.h"
void Model::alloc(int n) {
  allocations.push_back(n);
}
void useBuf(DevBuf b) {
  DevBuf c = b;
  c.adopt();
}
void readOther(Model& m) {
  int x = m.other;
}
void readAlloc(Model& m) {
  auto y = m.allocations.size();
}
"""


def fmap(funcs):
    """minimal impact map: funcs = {id: (file, name, paths)}"""
    F = {nid: {"file": f, "name": n, "lines": [1, 2], "paths": ps, "models": ["dense", "moe"], "tests": [f"whirl-tests:t_{n.split('::')[-1]}"]}
         for nid, (f, n, ps) in funcs.items()}
    tests = {t: {"exe": "whirl-tests", "gpu": False} for d in F.values() for t in d["tests"]}
    files = {}
    for d in F.values():
        files.setdefault(d["file"], set()).update(d["paths"])
    return {"functions": F, "hip": {}, "files": {k: sorted(v) for k, v in files.items()},
            "includers": {"src/model.h": ["src/model.cpp"], "src/other.h": ["src/model.cpp"]},
            "executables": {"whirl-tests": ["src/model.cpp"]}, "tests": tests, "release_srv_steps": {},
            "declared": {"paths": {p: {"metric": None} for p in ("load", "cli.decode", "startup", "vision")}, "gate_suites": {},
                         "metric_cells": {"None": []}, "gpus": {"gfx1201": "r97", "gfx1151": "s8"},
                         "models": {"dense": "swift", "moe": "ornith"}, "modes": ["precise", "balance", "fast"]}}


M = fmap({"src/model.cpp::Model::alloc": ("src/model.cpp", "Model::alloc", ["load"]),
          "src/model.cpp::useBuf": ("src/model.cpp", "useBuf", ["cli.decode"]),
          "src/model.cpp::readOther": ("src/model.cpp", "readOther", ["vision"]),
          "src/model.cpp::readAlloc": ("src/model.cpp", "readAlloc", ["startup"])})
REPO = {}


def fake_show(rev, path):
    return REPO.get((rev, path))


def run(changes):
    return impact.analyze(changes, M, "B", "A")


def tests_of(r):
    return {t for v in r["cpu_tests"].values() for t in v}


def git_lines(a, b):
    """changed line numbers (old, new) exactly as `git diff --no-index -U0` reports them"""
    import re
    import shutil
    import subprocess
    import tempfile
    d = tempfile.mkdtemp()
    try:
        fa, fb = os.path.join(d, "a"), os.path.join(d, "b")
        with open(fa, "w", newline="\n") as f:
            f.write(a)
        with open(fb, "w", newline="\n") as f:
            f.write(b)
        out = subprocess.run(["git", "diff", "--no-index", "-U0", fa, fb], capture_output=True, text=True).stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
    o, n = [], []
    for mm in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", out, re.M):
        os_, oc, ns, nc = int(mm.group(1)), int(mm.group(2) or 1), int(mm.group(3)), int(mm.group(4) or 1)
        o += list(range(os_, os_ + oc))
        n += list(range(ns, ns + nc))
    return o, n


def pmap(funcs, includers, hip=None):
    """map for the probes (as scratch/mt_p1_rr/probe_rr.py): every function its own path p_<name> and test t_<name>"""
    F = {}
    for f, n in funcs:
        short = n.split("::")[-1]
        F[f"{f}::{n}"] = {"file": f, "name": n, "lines": [1, 2], "paths": [f"p_{short}"], "models": ["dense", "moe"],
                          "tests": [f"whirl-tests:t_{short}"]}
    tests = {t: {"exe": "whirl-tests", "gpu": False} for d in F.values() for t in d["tests"]}
    files = {}
    for d in F.values():
        files.setdefault(d["file"], set()).update(d["paths"])
    hip = hip or {}
    paths = sorted({p for d in F.values() for p in d["paths"]} | {p for h in hip.values() for p in h["paths"]})
    return {"functions": F, "hip": hip, "files": {k: sorted(v) for k, v in files.items()}, "includers": includers,
            "executables": {"whirl-tests": sorted(files)}, "tests": tests, "release_srv_steps": {},
            "declared": {"paths": {p: {"metric": None} for p in paths}, "gate_suites": {}, "metric_cells": {"None": []},
                         "gpus": {"gfx1201": "r97", "gfx1151": "s8"}, "models": {"dense": "swift", "moe": "ornith"},
                         "modes": ["precise", "balance", "fast"]}}


def probe(A, B, funcs, need, includers=None, hip=None):
    """A, B: {path: text} old / new; returns (report, needed functions not selected, selected function names)"""
    REPO.clear()
    for p, t in A.items():
        REPO[("A", p)] = t
    for p, t in B.items():
        REPO[("B", p)] = t
    mp = pmap(funcs, includers if includers is not None else {"src/x.h": ["src/x.cpp"], "src/y.h": ["src/x.cpp"], "src/z.h": ["src/x.cpp"]}, hip)
    changes = {}
    for p in sorted(set(A) | set(B)):
        o, n = git_lines(A.get(p, ""), B.get(p, ""))
        if o or n:
            changes[p] = {"old_path": p, "new": n, "old": o}
    r = impact.analyze(changes, mp, "B", "A")
    sel = {p[2:] for p in r["paths"] if p.startswith("p_")}
    return r, sorted(set(need) - sel), sel


CPP13 = '#include "x.h"\nint tileRows(int n) {\n  return n / kTile;\n}\nint other() {\n  return 1;\n}\n'
HA13b = ("#pragma once\n#if defined(WHIRL_GFX1151)\nconstexpr int kTile = 32;\nconstexpr int kRows = 4;\n#else\n"
         "constexpr int kTile = 64;\nconstexpr int kRows = 4;\n#endif\n")
HA14b = ("#pragma once\nstruct Blk {\n    int n;\n#ifdef WHIRL_TRACE\n    int pad[8];\n    int t;\n#else\n    int pad[2];\n    int t;\n#endif\n};\n")
CPP14 = '#include "x.h"\nvoid sendBlk(Blk& b) {\n  hipMemcpy(dst, &b, sizeof(Blk));\n}\nint getN(Blk& b) {\n  return b.n;\n}\n'
HA15 = "#pragma once\nnamespace gfx1151 {\nconstexpr int kWaves = 4;\n}\nnamespace gfx1201 {\n}\n"
HB15 = "#pragma once\nnamespace gfx1151 {\n}\nnamespace gfx1201 {\nconstexpr int kWaves = 4;\n}\n"
CPP15 = ('#include "x.h"\nint waves1151() {\n  return gfx1151::kWaves;\n}\nint waves1201() {\n  using namespace gfx1201;\n  return kWaves;\n}\n')
HA16 = ("#pragma once\ntemplate <int A> struct Cfg;\ntemplate <> struct Cfg<1151> {\n    static constexpr int kTile = 32;\n    int pad;\n};\n"
        "template <> struct Cfg<1201> {\n    static constexpr int kTile = 64;\n    int pad;\n};\n")
CPP16 = '#include "x.h"\nint rows1201(int n) {\n  return n / Cfg<1201>::kTile;\n}\nint other() {\n  return 1;\n}\n'
HA17 = ("#pragma once\nstruct Engine {\n    struct Config {\n        int threads = 4;\n        int q = 1;\n    };\n};\n"
        "struct Server {\n    struct Config {\n        int threads = 8;\n        int q = 1;\n    };\n};\n")
CPP17 = '#include "x.h"\nint engThreads(Engine::Config& c) {\n  return c.threads;\n}\nint other() {\n  return 1;\n}\n'
HA18a = "#pragma once\nnamespace g1151 {\nconstexpr int kTile = 32;\n}\n"
HA18b = "#pragma once\nnamespace g1201 {\nconstexpr int kTile = 64;\n}\n"
CPP18 = '#include "x.h"\nint rows1151(int n) {\n  return n / g1151::kTile;\n}\nint other() {\n  return 1;\n}\n'
HA2 = "#pragma once\nvoid emit(long v);\ntemplate <class T> void put(T v) {\n  emit(v);\n}\nvoid emit(int v);\n"
HB2 = "#pragma once\nvoid emit(long v);\nvoid emit(int v);\ntemplate <class T> void put(T v) {\n  emit(v);\n}\n"
CPP2P = '#include "x.h"\nvoid writeTok(int t) {\n  put(t);\n}\nvoid emit(long v) {\n  sinkL(v);\n}\nvoid emit(int v) {\n  sinkI(v);\n}\n'
CPP1A = ('#include "x.h"\nextern std::string g_base;\nstd::string g_base = makeBase();\nstd::string g_path = g_base + "/kv";\n'
         'std::string kvDir() {\n  return g_path;\n}\nstd::string baseDir() {\n  return g_base;\n}\n')
CPP1B = ('#include "x.h"\nextern std::string g_base;\nstd::string g_path = g_base + "/kv";\nstd::string g_base = makeBase();\n'
         'std::string kvDir() {\n  return g_path;\n}\nstd::string baseDir() {\n  return g_base;\n}\n')
# D2 (device_profile.h shape): a designated-initializer row of a per-arch table; vision reads the field through profile()
PROF = ("#pragma once\nstruct Profile {\n    unsigned gemm_tile;\n    unsigned vision_attn_threads;\n};\n"
        "inline constexpr Profile k_profile_gfx1151 = {\n    .gemm_tile = 32,\n    .vision_attn_threads = %s,\n};\n"
        "inline constexpr Profile k_profile_gfx1201 = {\n    .gemm_tile = 64,\n    .vision_attn_threads = %s,\n};\n")
CPPD2 = ('#include "x.h"\nconst Profile& profile(int a) {\n  return a == 1151 ? k_profile_gfx1151 : k_profile_gfx1201;\n}\n'
         'void Vision::init(int arch) {\n  attn_threads_ = profile(arch).vision_attn_threads;\n}\n'
         'int gemmTile(int arch) {\n  return profile(arch).gemm_tile;\n}\n')
# a struct moved unchanged between two headers of the same namespace (include guards differ): still a move
MV_A = ("#ifndef X_H\n#define X_H\nnamespace whirl {\nstruct Opt {\n    int ctx = 8;\n};\nint keep();\n}\n#endif\n",
        "#ifndef Y_H\n#define Y_H\nnamespace whirl {\nint other();\n}\n#endif\n")
MV_B = ("#ifndef X_H\n#define X_H\nnamespace whirl {\nint keep();\n}\n#endif\n",
        "#ifndef Y_H\n#define Y_H\nnamespace whirl {\nint other();\nstruct Opt {\n    int ctx = 8;\n};\n}\n#endif\n")
CPPMV = '#include "x.h"\nint readCtx(whirl::Opt& o) {\n  return o.ctx;\n}\nint other() {\n  return 1;\n}\n'


def swap(t, a, b):
    return t.replace(a, "\x00").replace(b, a).replace("\x00", b)


# name: (old files, new files, map functions, needed (selected), block (must NOT be selected))
PROBES = {
    "P13b #if/#else kTile values swapped": ({"src/x.h": HA13b, "src/x.cpp": CPP13},
                                            {"src/x.h": swap(HA13b, "kTile = 32", "kTile = 64"), "src/x.cpp": CPP13},
                                            [("src/x.cpp", "tileRows"), ("src/x.cpp", "other")], ["tileRows"], []),
    "P14b #ifdef/#else padding swapped in a struct": ({"src/x.h": HA14b, "src/x.cpp": CPP14},
                                                      {"src/x.h": swap(HA14b, "pad[8]", "pad[2]"), "src/x.cpp": CPP14},
                                                      [("src/x.cpp", "sendBlk"), ("src/x.cpp", "getN")], ["sendBlk"], []),
    "P15 constant moved to another namespace in one header": ({"src/x.h": HA15, "src/x.cpp": CPP15}, {"src/x.h": HB15, "src/x.cpp": CPP15},
                                                              [("src/x.cpp", "waves1151"), ("src/x.cpp", "waves1201")], ["waves1201"], []),
    "P16 Cfg<1151>/Cfg<1201> kTile swapped": ({"src/x.h": HA16, "src/x.cpp": CPP16},
                                              {"src/x.h": swap(HA16, "kTile = 32", "kTile = 64"), "src/x.cpp": CPP16},
                                              [("src/x.cpp", "rows1201"), ("src/x.cpp", "other")], ["rows1201"], []),
    "P17 Engine::Config/Server::Config defaults swapped": ({"src/x.h": HA17, "src/x.cpp": CPP17},
                                                           {"src/x.h": swap(HA17, "threads = 4", "threads = 8"), "src/x.cpp": CPP17},
                                                           [("src/x.cpp", "engThreads"), ("src/x.cpp", "other")], ["engThreads"], []),
    "P18 two headers changed in place to swapped values": ({"src/x.h": HA18a, "src/y.h": HA18b, "src/x.cpp": CPP18},
                                                           {"src/x.h": HA18a.replace("32", "64"), "src/y.h": HA18b.replace("64", "32"),
                                                            "src/x.cpp": CPP18},
                                                           [("src/x.cpp", "rows1151"), ("src/x.cpp", "other")], ["rows1151"], []),
    "P2 overload moved before the template that calls it": ({"src/x.h": HA2, "src/x.cpp": CPP2P}, {"src/x.h": HB2, "src/x.cpp": CPP2P},
                                                            [("src/x.cpp", "writeTok"), ("src/x.cpp", "emit")], ["writeTok"], []),
    "P1 namespace-scope globals swapped in one .cpp": ({"src/x.h": "#pragma once\n", "src/x.cpp": CPP1A},
                                                       {"src/x.h": "#pragma once\n", "src/x.cpp": CPP1B},
                                                       [("src/x.cpp", "kvDir"), ("src/x.cpp", "baseDir")], ["kvDir"], []),
    "D2 designated initializer .vision_attn_threads swapped": ({"src/x.h": PROF % ("128", "256"), "src/x.cpp": CPPD2},
                                                               {"src/x.h": PROF % ("256", "128"), "src/x.cpp": CPPD2},
                                                               [("src/x.cpp", "profile"), ("src/x.cpp", "Vision::init"),
                                                                ("src/x.cpp", "gemmTile")], ["profile", "init"], ["gemmTile"]),
    "count: a statement removed from two headers, added to one (occurrences 2 -> 1): not a move":
        ({"src/x.h": HA18a, "src/y.h": HA18a.replace("#pragma once", "#pragma once\n// y"), "src/z.h": "#pragma once\n", "src/x.cpp": CPP18},
         {"src/x.h": "#pragma once\n", "src/y.h": "#pragma once\n// y\n", "src/z.h": HA18a, "src/x.cpp": CPP18},
         [("src/x.cpp", "rows1151"), ("src/x.cpp", "other")], ["rows1151"], []),
    "block: struct moved unchanged between headers (guards differ)": ({"src/x.h": MV_A[0], "src/y.h": MV_A[1], "src/x.cpp": CPPMV},
                                                                      {"src/x.h": MV_B[0], "src/y.h": MV_B[1], "src/x.cpp": CPPMV},
                                                                      [("src/x.cpp", "readCtx"), ("src/x.cpp", "other")], [], ["readCtx", "other"]),
}

real = impact.show
impact.show = fake_show
impact.load_modules = lambda root=None: None
try:
    # B1.1 declared name: the member's users, not the users of its type
    REPO.clear()
    REPO.update({("A", "src/model.h"): HDR_OLD, ("B", "src/model.h"): HDR_NEW, ("B", "src/model.cpp"): CPP, ("A", "src/model.cpp"): CPP})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [3], "old": [3]}})
    t = tests_of(r)
    expect("pass: users of the declared member (Model::alloc in-class, readAlloc via m.allocations)",
           {"t_alloc", "t_readAlloc"} <= t, t)
    expect("block: a function that only names the type DevBuf is not selected", "t_useBuf" not in t, t)
    expect("block: a function using another member is not selected", "t_readOther" not in t, t)
    expect("declarations row: Model::allocations with type DevBuf recorded",
           any(d["name"] == "Model::allocations" and "DevBuf" in d["types"] for d in r["declarations"]), r["declarations"])
    expect("behaviour: DevPtr -> DevBuf member type = lifetime (handle type)", "lifetime" in r["behaviour"], r["behaviour"])
    expect("startup user -> sizing class -> srv_env_* steps + E2 check",
           "srv_env_e1" in r["release_gate_steps"] and any(c.startswith("E2") for c in r["outside_gate_checks"]),
           (r["release_gate_steps"], r["outside_gate_checks"]))
    expect("no map defect for a declared member with users", not r["map_defects"], r["map_defects"])

    # a member added that no function reads yet -> struct fallback + file scope (fail closed) + map defect
    HDR_ADD = HDR_NEW.replace("    int other = 0;\n", "    int other = 0;\n    int fresh_ = 1;\n")
    REPO[("B", "src/model.h")] = HDR_ADD
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [5], "old": []}})
    t = tests_of(r)
    expect("fail closed: unread new member -> users of Model (struct fallback: its member functions)", "t_alloc" in t, t)
    expect("fail closed: unread new member -> file-scope paths", "cli.decode" in r["paths"], sorted(r["paths"]))
    expect("map defect reported for a declaration with no user", any("fresh_" in d for d in r["map_defects"]), r["map_defects"])
    expect("behaviour: member added = layout", "layout" in r["behaviour"], r["behaviour"])

    # mt-p1 review A2: an unused MEMBER removed changes sizeof / layout -> fail closed (file scope), as the 0.2.4 tool did
    REPO[("A", "src/model.h")] = HDR_ADD
    REPO[("B", "src/model.h")] = HDR_NEW
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [], "old": [5]}})
    expect("A2 removed unused member -> file scope + layout", "cli.decode" in r["paths"] and "layout" in r["behaviour"],
           (sorted(r["paths"]), r["behaviour"]))
    # A2: an unused enumerator removed in the middle shifts later values -> fail closed
    EN_A = "#pragma once\nenum Kind {\n    kA,\n    kOld,\n    kB,\n};\n"
    EN_B = "#pragma once\nenum Kind {\n    kA,\n    kB,\n};\n"
    REPO.update({("A", "src/model.h"): EN_A, ("B", "src/model.h"): EN_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [], "old": [4]}})
    expect("A2 removed unused enumerator -> file scope", "cli.decode" in r["paths"], sorted(r["paths"]))
    # the S7 rule stays for namespace-scope functions / variables (kNoTag): removed with its last user -> build only
    NS_A = "#pragma once\nconstexpr int kNoTag = 0xff;\nint keep(int x);\n"
    NS_B = "#pragma once\nint keep(int x);\n"
    REPO.update({("A", "src/model.h"): NS_A, ("B", "src/model.h"): NS_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [], "old": [2]}})
    expect("removed namespace constant with no user left: no file scope, no defect", "cli.decode" not in r["paths"] and not r["map_defects"],
           (sorted(r["paths"]), r["map_defects"]))
    # mt-p1 review A1: two members swapped inside one struct (kv_tier.h Header n_tok / nck): layout change, not a move
    SW_A = "#pragma once\nstruct Model {\n    std::vector<DevBuf> allocations;\n    int other = 0;\n};\n"
    SW_B = "#pragma once\nstruct Model {\n    int other = 0;\n    std::vector<DevBuf> allocations;\n};\n"
    REPO.update({("A", "src/model.h"): SW_A, ("B", "src/model.h"): SW_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [3, 4], "old": [3, 4]}})
    t = tests_of(r)
    expect("A1 member swap -> users of both members + file scope + layout", {"t_alloc", "t_readAlloc", "t_readOther"} <= t
           and "cli.decode" in r["paths"] and "layout" in r["behaviour"], (t, sorted(r["paths"]), r["behaviour"]))
    # A1: two enumerators swapped (numerics.h Item kvq4 / relaxacc, k_items is positional) -> fail closed
    EQ_A = "#pragma once\nenum class Item {\n    kvq4,\n    relaxacc,\n    other,\n};\n"
    EQ_B = "#pragma once\nenum class Item {\n    relaxacc,\n    kvq4,\n    other,\n};\n"
    REPO.update({("A", "src/model.h"): EQ_A, ("B", "src/model.h"): EQ_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [3, 4], "old": [3, 4]}})
    expect("A1 enumerator swap -> file scope", "cli.decode" in r["paths"], sorted(r["paths"]))
    # review C2 (R6): an enumerator inserted mid-enum shifts the later values -> fail closed; appended at the end -> not.
    # The new enumerator has a user (otherwise the no-user fallback alone fails closed)
    M["functions"]["src/model.cpp::useTurbo"] = {"file": "src/model.cpp", "name": "useTurbo", "lines": [1, 2], "paths": ["load"],
                                                 "models": ["dense", "moe"], "tests": ["whirl-tests:t_useTurbo"]}
    M["tests"]["whirl-tests:t_useTurbo"] = {"exe": "whirl-tests", "gpu": False}
    REPO[("B", "src/model.cpp")] = CPP + "int useTurbo() {\n  return (int)Item::turbo;\n}\n"
    EI_B = "#pragma once\nenum class Item {\n    kvq4,\n    turbo,\n    relaxacc,\n    other,\n};\n"
    REPO.update({("A", "src/model.h"): EQ_A, ("B", "src/model.h"): EI_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": []}})
    expect("C2 enumerator inserted mid-enum -> file scope", "cli.decode" in r["paths"], sorted(r["paths"]))
    EE_B = "#pragma once\nenum class Item {\n    kvq4,\n    relaxacc,\n    other,\n    turbo,\n};\n"
    REPO.update({("A", "src/model.h"): EQ_A, ("B", "src/model.h"): EE_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [6], "old": []}})
    expect("block: enumerator appended at the end -> only its users", "cli.decode" not in r["paths"] and "t_useTurbo" in tests_of(r),
           (sorted(r["paths"]), tests_of(r)))
    REPO[("B", "src/model.cpp")] = CPP
    # review C2 (R7 / R8): a constant used as an array bound in a struct -> the struct's users; a macro used by another
    # macro -> that macro's users
    M["functions"]["src/model.cpp::copyRing"] = {"file": "src/model.cpp", "name": "copyRing", "lines": [1, 2], "paths": ["load"],
                                                 "models": ["dense", "moe"], "tests": ["whirl-tests:t_copyRing"]}
    M["functions"]["src/model.cpp::tile2"] = {"file": "src/model.cpp", "name": "tile2", "lines": [1, 2], "paths": ["load"],
                                              "models": ["dense", "moe"], "tests": ["whirl-tests:t_tile2"]}
    M["tests"]["whirl-tests:t_copyRing"] = {"exe": "whirl-tests", "gpu": False}
    M["tests"]["whirl-tests:t_tile2"] = {"exe": "whirl-tests", "gpu": False}
    CPP_C2 = CPP + "void copyRing(Ring& r, void* d) {\n  memcpy(d, &r, sizeof(Ring));\n}\nint tile2() {\n  return TILE2;\n}\n"
    REPO.update({("A", "src/model.cpp"): CPP_C2, ("B", "src/model.cpp"): CPP_C2,
                 ("A", "src/model.h"): "#pragma once\nconstexpr int kSlots = 8;\nstruct Ring {\n    float v[kSlots];\n};\n",
                 ("B", "src/model.h"): "#pragma once\nconstexpr int kSlots = 16;\nstruct Ring {\n    float v[kSlots];\n};\n"})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [2], "old": [2]}})
    expect("C2 R7 constant used as a struct array bound -> users of the struct", "t_copyRing" in tests_of(r), tests_of(r))
    REPO.update({("A", "src/model.h"): "#pragma once\n#define TILE 64\n#define TILE2 (TILE * 2)\n",
                 ("B", "src/model.h"): "#pragma once\n#define TILE 32\n#define TILE2 (TILE * 2)\n"})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [2], "old": [2]}})
    expect("C2 R8 macro used through another macro -> its users", "t_tile2" in tests_of(r), tests_of(r))
    REPO.update({("A", "src/model.cpp"): CPP, ("B", "src/model.cpp"): CPP})
    # block: a diff-alignment edit (the same statement on both sides, order of the struct unchanged) stays a no-op
    REPO.update({("A", "src/model.h"): SW_A, ("B", "src/model.h"): SW_A})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": [4]}})
    expect("block: same statement, same order (diff alignment) -> nothing selected", not r["paths"], sorted(r["paths"]))
    # moved declaration (other.h -> model.h, identical statement): build only; changed while moving: selected
    REPO.update({("A", "src/other.h"): "#pragma once\nstruct Model {\n    int other = 0;\n};\n", ("B", "src/other.h"): "#pragma once\n"})
    REPO[("B", "src/model.h")] = HDR_NEW
    REPO[("A", "src/model.h")] = "#pragma once\nstruct Model {\n    std::vector<DevBuf> allocations;\n};\n"
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": []},
             "src/other.h": {"old_path": "src/other.h", "new": [], "old": [2, 3, 4]}})
    expect("moved unchanged declaration selects no user (block)", "t_readOther" not in tests_of(r), tests_of(r))
    expect("moved declaration noted as build only", any("moved" in n for n in r["notes"]), r["notes"])
    REPO[("A", "src/other.h")] = "#pragma once\nstruct Model {\n    int other = 7;\n};\n"
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": []},
             "src/other.h": {"old_path": "src/other.h", "new": [], "old": [2, 3, 4]}})
    expect("moved AND changed declaration selects its users (pass)", "t_readOther" in tests_of(r), tests_of(r))

    # B1.2 preprocessor lines (were 'comments only' before P1)
    REPO[("A", "src/model.h")] = HDR_NEW
    REPO[("B", "src/model.h")] = HDR_NEW.replace("#pragma once\n", "#pragma once\n#include <vector>\n")
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [2], "old": []}})
    expect("#include added -> build-only rerun, no path", not r["paths"] and any(k.startswith("build: compile src/model.h") for k in r["extra"]),
           (r["paths"], r["extra"]))
    REPO[("B", "src/model.h")] = HDR_NEW.replace("#pragma once\n", "#pragma once\n#define WITH_ALLOC 1\n")
    CPP2 = CPP.replace("auto y = m.allocations.size();", "auto y = WITH_ALLOC;")
    REPO[("B", "src/model.cpp")] = CPP2
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [2], "old": []}})
    expect("#define -> users of the macro", tests_of(r) == {"t_readAlloc"}, tests_of(r))
    REPO[("B", "src/model.cpp")] = CPP
    REPO[("B", "src/model.h")] = HDR_NEW.replace("    int other = 0;\n", "#if defined(__gfx1151__)\n    int other = 0;\n#endif\n")
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4, 6], "old": []}})
    expect("#if around a member -> file scope (fail closed, not a defect)", "cli.decode" in r["paths"] and not r["map_defects"],
           (sorted(r["paths"]), r["map_defects"]))

    # a lone closing brace moves a block edge over unchanged lines -> file scope
    SRC_A = "namespace {\nint helperA(int x);\n}\nint helperB(int x);\n"
    SRC_B = "namespace {\nint helperA(int x);\nint helperB(int x);\n}\n"
    REPO.update({("A", "src/model.h"): SRC_A, ("B", "src/model.h"): SRC_B})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": [3]}})
    expect("lone '}' moved -> file scope", "cli.decode" in r["paths"], sorted(r["paths"]))

    # comment-only change still selects nothing
    REPO.update({("A", "src/model.h"): HDR_NEW, ("B", "src/model.h"): HDR_NEW.replace("int other = 0;", "int other = 0;  // why")})
    r = run({"src/model.h": {"old_path": "src/model.h", "new": [4], "old": [4]}})
    expect("block: a changed member line whose code is equal still selects its users only", tests_of(r) == {"t_readOther"} or not r["paths"],
           tests_of(r))

    # B1.3 behaviour classes from files
    REPO.update({("A", "CMakeLists.txt"): "project(whirl VERSION 0.2.3)\n", ("B", "CMakeLists.txt"): "project(whirl VERSION 0.2.4)\n"})
    r = run({"CMakeLists.txt": {"old_path": "CMakeLists.txt", "new": [1], "old": [1]}})
    expect("CMake project VERSION -> version class + its check", "version" in r["behaviour"] and any(c.startswith("version")
           for c in r["outside_gate_checks"]), (r["behaviour"], r["outside_gate_checks"]))
    REPO.update({("A", "src/device/backend.cpp"): "void f(){\n int a;\n}\n", ("B", "src/device/backend.cpp"): "void f(){\n int b;\n}\n"})
    M["functions"]["src/device/backend.cpp::f"] = {"file": "src/device/backend.cpp", "name": "f", "lines": [1, 3], "paths": ["startup"],
                                                   "models": ["dense", "moe"], "tests": []}
    r = run({"src/device/backend.cpp": {"old_path": "src/device/backend.cpp", "new": [2], "old": [2]}})
    expect("src/device file -> device class: srv_gpu_select + E6 check", "srv_gpu_select" in r["release_gate_steps"]
           and any(c.startswith("E6") for c in r["outside_gate_checks"]), (r["release_gate_steps"], r["outside_gate_checks"]))
    # P-719 (arch024 PLAN): a pinned host memory owner -> srv_gpu_select + kvram719 (hold-out miss S6b, TierBridge)
    REPO.update({("A", "src/server/tier_bridge.cpp"): "void g(){\n int a;\n}\n", ("B", "src/server/tier_bridge.cpp"): "void g(){\n int b;\n}\n"})
    M["functions"]["src/server/tier_bridge.cpp::g"] = {"file": "src/server/tier_bridge.cpp", "name": "g", "lines": [1, 3],
                                                       "paths": ["load"], "models": ["dense", "moe"], "tests": []}
    r = run({"src/server/tier_bridge.cpp": {"old_path": "src/server/tier_bridge.cpp", "new": [2], "old": [2]}})
    expect("pinned owner (tier_bridge.cpp) -> srv_gpu_select + kvram719", "srv_gpu_select" in r["release_gate_steps"]
           and any(c.startswith("kvram719") for c in r["outside_gate_checks"]), (r["release_gate_steps"], r["outside_gate_checks"]))
    # golden cases listed by a path -> the golden_<model> steps (hold-out miss B0: golden server listed, no step)
    M["functions"]["src/server/tier_bridge.cpp::g"]["paths"] = ["startup"]
    impact.PATH_GOLDEN["startup"] = ["server"]
    r = run({"src/server/tier_bridge.cpp": {"old_path": "src/server/tier_bridge.cpp", "new": [2], "old": [2]}})
    expect("golden server listed -> golden_swift + golden_moe steps", {"golden_swift", "golden_moe"} <= set(r["release_gate_steps"]),
           r["release_gate_steps"])
    # vision class -> golden_<model> (golden server holds the vision rows; hold-out f6e4236 vision::imageKey)
    REPO.update({("A", "src/vision/vision.cpp"): "void h(){\n int a;\n}\n", ("B", "src/vision/vision.cpp"): "void h(){\n int b;\n}\n"})
    M["functions"]["src/vision/vision.cpp::h"] = {"file": "src/vision/vision.cpp", "name": "h", "lines": [1, 3], "paths": [],
                                                  "models": ["dense", "moe"], "tests": ["whirl-tests:t_h"]}
    M["tests"]["whirl-tests:t_h"] = {"exe": "whirl-tests", "gpu": False}
    r = run({"src/vision/vision.cpp": {"old_path": "src/vision/vision.cpp", "new": [2], "old": [2]}})
    expect("vision file -> golden_swift + golden_moe + visab check", {"golden_swift", "golden_moe"} <= set(r["release_gate_steps"])
           and any(c.startswith("visab") for c in r["outside_gate_checks"]), (r["release_gate_steps"], r["outside_gate_checks"]))
    # review C1: numeric / dispatch / layout classes name golden -> they add golden_<model> (also with no path reached)
    REPO.update({("A", "src/common/numerics.cpp"): "void q(){\n int a;\n}\n", ("B", "src/common/numerics.cpp"): "void q(){\n int b;\n}\n"})
    M["functions"]["src/common/numerics.cpp::q"] = {"file": "src/common/numerics.cpp", "name": "q", "lines": [1, 3], "paths": [],
                                                    "models": ["dense", "moe"], "tests": []}
    r = run({"src/common/numerics.cpp": {"old_path": "src/common/numerics.cpp", "new": [2], "old": [2]}})
    expect("C1 numeric class -> golden_swift + golden_moe", {"golden_swift", "golden_moe"} <= set(r["release_gate_steps"]),
           r["release_gate_steps"])
    # a changed test source selects the release step that runs its executable (map defect found on 605281b..9844fde)
    M["executables"]["whirl-tier-tests"] = ["tests/server/tier_tests.cpp"]
    M["declared"]["release_steps"] = {"cpu_tier": "exe:whirl-tier-tests"}
    r = run({"tests/server/tier_tests.cpp": {"old_path": "tests/server/tier_tests.cpp", "new": [1], "old": [1]}})
    expect("test source changed -> its release step (cpu_tier)", "cpu_tier" in r["release_gate_steps"], r["release_gate_steps"])
    REPO.update({("A", "docs/x.md"): "a\n", ("B", "docs/x.md"): "b\n"})
    r = run({"docs/x.md": {"old_path": "docs/x.md", "new": [1], "old": [1]}})
    expect("block: docs only -> no GPU check, no step", not r["outside_gate_checks"] and not r["release_gate_steps"],
           (r["outside_gate_checks"], r["release_gate_steps"]))

    # ---------------------------------------------------------------------------------------------------------------
    # mt-p1 pass 3 (progress/mt-p1.REREVIEW.md act-on 1-3, C-a): the reviewer's probes (scratch/mt_p1_rr/probe_rr.py),
    # changed lines from `git diff --no-index -U0` (difflib aligns swaps differently and hid act-on 1)
    for nm, (a, b, fns, need, block) in sorted(PROBES.items()):
        r, miss, sel = probe(a, b, fns, need)
        expect(f"pass3 {nm}: needed {need} selected", not miss, (miss, sel, r["notes"][:3]))
        if block:
            expect(f"pass3 {nm}: block, not selected {block}", not (set(block) & sel), sel)
    # review C-a, promoted (K1, hipError 719 header): a header included by a .hip selects the includers' kernel work
    HIP = {"kernels/gemm_small.hip": {"arch": ["gfx1201"], "models": ["dense", "moe"], "paths": ["p_prefill"], "host_users": ["gemmSmall"],
                                      "tests": ["whirl-kernel-test:testGemmSmall"], "fields": []}}
    KH = "#pragma once\n__device__ inline int gemms_stage_chunks(int sb) { return (sb + 15) / 16 + 1; }\n"
    inc = {"kernels/gemm_small_addr.h": ["kernels/gemm_small.hip", "tests/unit/unit_tests.cpp"]}
    r, _, _ = probe({"kernels/gemm_small_addr.h": KH}, {"kernels/gemm_small_addr.h": KH.replace(" + 1;", ";")}, [], [], inc, HIP)
    expect("pass3 K1 .hip-included header -> includer kernel tests + kt_full + path", "testGemmSmall" in r["kernel_tests"]
           and "kt_full" in r["release_gate_steps"] and "p_prefill" in r["paths"], (r["kernel_tests"], r["release_gate_steps"], sorted(r["paths"])))
    expect("pass3 K1 -> P-719 checks (srv_gpu_select + kvram719)", "srv_gpu_select" in r["release_gate_steps"]
           and any(c.startswith("kvram719") for c in r["outside_gate_checks"]), (r["release_gate_steps"], r["outside_gate_checks"]))
    expect("pass3 K1 kernel_header class + golden", "kernel_header" in r["behaviour"] and "golden_swift" in r["release_gate_steps"],
           (r["behaviour"], r["release_gate_steps"]))
    r, _, _ = probe({"kernels/gemm_small_addr.h": KH}, {"kernels/gemm_small_addr.h": KH.replace("#pragma once\n", "#pragma once\n// note\n")},
                    [], [], inc, HIP)
    expect("pass3 K1 block: comment-only edit of the header -> no kernel work", "kt_full" not in r["release_gate_steps"], r["release_gate_steps"])
    r, _, _ = probe({"kernels/other.h": KH}, {"kernels/other.h": KH.replace(" + 1;", ";")}, [], [], {"kernels/other.h": ["src/a.cpp"]}, HIP)
    expect("pass3 K1 block: header with no .hip includer -> no kt_full", "kt_full" not in r["release_gate_steps"], r["release_gate_steps"])
finally:
    impact.show = real
print(f"impact decl test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
