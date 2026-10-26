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
# (pass 5: x.cpp includes y.h in the source too, as the map says - the closure now comes from the #include lines)
CPPMV = '#include "x.h"\n#include "y.h"\nint readCtx(whirl::Opt& o) {\n  return o.ctx;\n}\nint other() {\n  return 1;\n}\n'


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

# mt-p1 pass 4 probes (scratch/mt_p1_rv3/probe_rv3.py): name: (old, new, functions, needed, block, includers)
INC4 = {"src/a.h": ["src/b.h", "src/u.cpp", "src/v.cpp"], "src/b.h": ["src/v.cpp"]}
FN4 = [("src/u.cpp", "useU"), ("src/v.cpp", "useV")]
A1h = "#pragma once\nnamespace w {\nint score(long v);\nint score(int v);\n}  // namespace w\n"
B1h = "#pragma once\nnamespace w {\nint score(long v);\n}  // namespace w\n"
Ab1 = "#pragma once\n#include \"a.h\"\nnamespace w {\nint more();\n}  // namespace w\n"
Bb1 = "#pragma once\n#include \"a.h\"\nnamespace w {\nint more();\nint score(int v);\n}  // namespace w\n"
U1 = '#include "a.h"\nint useU() {\n  return w::score(3);\n}\n'
V1 = '#include "b.h"\nint useV() {\n  return w::score(4);\n}\n'
A2h = ("#pragma once\nnamespace w {\ntemplate <int A> struct Cfg {\n    static constexpr int kTile = 16;\n};\n"
       "template <> struct Cfg<1151> {\n    static constexpr int kTile = 32;\n};\n}  // namespace w\n")
B2h = "#pragma once\nnamespace w {\ntemplate <int A> struct Cfg {\n    static constexpr int kTile = 16;\n};\n}  // namespace w\n"
Bb2 = ("#pragma once\n#include \"a.h\"\nnamespace w {\nint more();\ntemplate <> struct Cfg<1151> {\n    static constexpr int kTile = 32;\n};\n"
       "}  // namespace w\n")
U2 = '#include "a.h"\nint useU() {\n  return w::Cfg<1151>::kTile;\n}\n'
V2 = '#include "b.h"\nint useV() {\n  return w::Cfg<1151>::kTile;\n}\n'
A3h = "#pragma once\n#define WHIRL_SAFE_COPY 1\nint other();\n"
B3h = "#pragma once\nint other();\n"
Ab3 = "#pragma once\n#include \"a.h\"\nint more();\n"
Bb3 = "#pragma once\n#include \"a.h\"\nint more();\n#define WHIRL_SAFE_COPY 1\n"
U3c = ('#include "a.h"\nvoid useU(float* d, const float* s, int n) {\n#if WHIRL_SAFE_COPY\n  memmove(d, s, n * 4);\n'
       '#else\n  memcpy(d, s, n * 4);\n#endif\n}\n')
A6h = ("#pragma once\nstruct Profile {\n    int vision_attn_threads;\n    int other;\n};\n"
       "inline constexpr Profile k_p{\n    .vision_attn_threads =\n        128,\n    .other = 1,\n};\n")
U6 = '#include "a.h"\nint useU(const Profile& p) {\n  return p.vision_attn_threads;\n}\n'
V6 = '#include "a.h"\nconst Profile& prof() {\n  return k_p;\n}\n'
A7b = ("#pragma once\nstruct Profile {\n    int gemv_r[4];\n    int other;\n};\n"
       "inline constexpr Profile k_p{\n    .gemv_r = {1, 1,\n               2, 2},\n    .other = 1,\n};\n")
U7b = '#include "a.h"\nint useU(const Profile& p, int b) {\n  return p.gemv_r[b];\n}\n'
A5h = "#pragma once\nconstexpr int kTile = 32;\nconstexpr int kRows = kTile * 2;\n"
U5 = '#include "a.h"\nint useU() {\n  return kRows;\n}\n'
V5 = '#include "a.h"\nint useV() {\n  return kTile;\n}\n'
T4i = "static const int kMaxFold = 3;\nstatic const unsigned kTbl[] = {1, 2, 3};\n"
T4c = '#include "t.inc"\nint lookup(int c) {\n  return c < kMaxFold ? kTbl[c] : 0;\n}\nint other() {\n  return 1;\n}\n'
INC1 = {"src/a.h": ["src/u.cpp", "src/v.cpp"]}
PROBES4 = {
    "V1 overload moved a.h -> b.h, u.cpp sees only a.h (score(3) binds score(long))":
        ({"src/a.h": A1h, "src/b.h": Ab1, "src/u.cpp": U1, "src/v.cpp": V1}, {"src/a.h": B1h, "src/b.h": Bb1, "src/u.cpp": U1, "src/v.cpp": V1},
         FN4, ["useU"], [], INC4),
    "V2 explicit specialisation moved a.h -> b.h, u.cpp silently uses the primary":
        ({"src/a.h": A2h, "src/b.h": Ab1, "src/u.cpp": U2, "src/v.cpp": V2}, {"src/a.h": B2h, "src/b.h": Bb2, "src/u.cpp": U2, "src/v.cpp": V2},
         FN4, ["useU"], [], INC4),
    "V3c macro moved a.h -> b.h, u.cpp tests it with #if":
        ({"src/a.h": A3h, "src/b.h": Ab3, "src/u.cpp": U3c, "src/v.cpp": V1}, {"src/a.h": B3h, "src/b.h": Bb3, "src/u.cpp": U3c, "src/v.cpp": V1},
         FN4, ["useU"], [], INC4),
    "V3 macro moved a.h -> b.h, u.cpp tests it with #ifdef (known gap P3)":
        ({"src/a.h": A3h, "src/b.h": Ab3, "src/u.cpp": U3c.replace("#if WHIRL", "#ifdef WHIRL"), "src/v.cpp": V1},
         {"src/a.h": B3h, "src/b.h": Bb3, "src/u.cpp": U3c.replace("#if WHIRL", "#ifdef WHIRL"), "src/v.cpp": V1}, FN4, ["useU"], [], INC4),
    "V6 designator value on its own line (C-1)":
        ({"src/a.h": A6h, "src/u.cpp": U6, "src/v.cpp": V6}, {"src/a.h": A6h.replace("128,", "256,"), "src/u.cpp": U6, "src/v.cpp": V6},
         [("src/u.cpp", "useU"), ("src/v.cpp", "prof")], ["useU"], [], INC1),
    "V7b designated array value continued on the next line (C-1)":
        ({"src/a.h": A7b, "src/u.cpp": U7b, "src/v.cpp": V6}, {"src/a.h": A7b.replace("2, 2}", "2, 4}"), "src/u.cpp": U7b, "src/v.cpp": V6},
         [("src/u.cpp", "useU"), ("src/v.cpp", "prof")], ["useU"], [], INC1),
    "V5 namespace-scope kRows = kTile * 2: kTile changed -> kRows readers too (noted)":
        ({"src/a.h": A5h, "src/u.cpp": U5, "src/v.cpp": V5}, {"src/a.h": A5h.replace("= 32", "= 64"), "src/u.cpp": U5, "src/v.cpp": V5},
         FN4, ["useU", "useV"], [], INC1),
    "V4b .inc constant, user in the including .cpp, no files[] row for the .inc (noted)":
        ({"src/t.inc": T4i, "src/t.cpp": T4c}, {"src/t.inc": T4i.replace("= 3;", "= 4;"), "src/t.cpp": T4c},
         [("src/t.cpp", "lookup"), ("src/t.cpp", "other")], ["lookup"], ["other"], {"src/t.inc": ["src/t.cpp"]}),
}
# G3: kGemmsCfgs row (identifier-less) -> default of KernelTable::gemms_geom -> precChoice reads k.gemms_geom through a
# parameter; loadK names the class (pass 3 selected only these loaders)
G3H = ("#pragma once\nstruct GemmCfg {\n    int bm, bn, threads;\n};\ninline constexpr std::array<GemmCfg, 3> kGemmsCfgs = {{\n"
       "    {64, 32, 128}, {128, 32, 256},\n    {64, 128, 256},\n}};\nstruct KernelTable {\n    int gemms{};\n"
       "    std::array<GemmCfg, kGemmsCfgs.size()> gemms_geom = kGemmsCfgs;\n};\n")
G3C = ('#include "k.h"\nvoid loadK(KernelTable* t) {\n  KernelTable z;\n  *t = z;\n}\n'
       'int precChoice(const KernelTable& k, int si) {\n  return k.gemms_geom[si].bn;\n}\nint other() {\n  return 1;\n}\n')
G3 = ({"src/k.h": G3H, "src/m.cpp": G3C}, {"src/k.h": G3H.replace("{64, 128, 256}", "{64, 64, 256}"), "src/m.cpp": G3C},
      [("src/m.cpp", "loadK"), ("src/m.cpp", "precChoice"), ("src/m.cpp", "other")], ["precChoice"], ["other"], {"src/k.h": ["src/m.cpp"]})
# block: the same overload move where every includer of a.h also includes b.h: a real move, build only
# (pass 5: u.cpp includes b.h in the source too, as the map says; the closure now comes from the #include lines)
MV_IN = ({"src/a.h": A1h, "src/b.h": Ab1, "src/u.cpp": U1.replace('"a.h"', '"b.h"'), "src/v.cpp": V1}, {"src/a.h": B1h, "src/b.h": Bb1, "src/u.cpp": U1.replace('"a.h"', '"b.h"'), "src/v.cpp": V1},
         FN4, [], [], {"src/a.h": ["src/b.h", "src/u.cpp", "src/v.cpp"], "src/b.h": ["src/u.cpp", "src/v.cpp"]})

# mt-p1 pass 5 (review4 A1 / A2, progress/mt-p1-pass4.PRIOR_ART.md "Review 4" / "Pass 5"; probes
# scratch/mt_p1_review4/probe_r4_cases.py, probe_r4b_cases.py): the translation units that see a moved statement are
# taken from each revision's own #include lines (impact.tree_includes, patched here to read the fake repo), and must be
# equal in both trees for macros / free-function overloads / explicit specialisations (gain direction too)
FNV5 = [("src/u.cpp", "useU"), ("src/v.cpp", "useV")]
V5_ = '#include "b.h"\nint useV() {\n  return 0;\n}\n'
U51ma = '#define WHIRL_SAFE_COPY 1\nint useU() {\n#ifdef WHIRL_SAFE_COPY\n  return 1;\n#else\n  return 2;\n#endif\n}\n'
U51mb = 'int useU() {\n#ifdef WHIRL_SAFE_COPY\n  return 1;\n#else\n  return 2;\n#endif\n}\n'
B51a = "#pragma once\nnamespace w {\nint more();\n}\n"
B51mb = "#pragma once\nnamespace w {\nint more();\n}\n#define WHIRL_SAFE_COPY 1\n"
A52a = "#pragma once\n#define WHIRL_SAFE_COPY 1\nint other();\n"
A52b = "#pragma once\nint other();\n"
C52a = "#pragma once\nint more();\n"
C52b = "#pragma once\nint more();\n#define WHIRL_SAFE_COPY 1\n"
U52 = '#include "c.h"\nint useU() {\n#ifdef WHIRL_SAFE_COPY\n  return 1;\n#else\n  return 2;\n#endif\n}\n'
V52 = '#include "c.h"\nint useV() {\n#ifdef WHIRL_SAFE_COPY\n  return 1;\n#else\n  return 2;\n#endif\n}\n'
A52oa = "#pragma once\nnamespace w {\nint score(int v);\n}\n"
A52ob = "#pragma once\nnamespace w {\n}\n"
C52oa = "#pragma once\nnamespace w {\nint score(long v);\n}\n"
C52ob = "#pragma once\nnamespace w {\nint score(long v);\nint score(int v);\n}\n"
U52o = '#include "c.h"\nint useU() {\n  return w::score(3);\n}\n'
V52o = '#include "c.h"\nint useV() {\n  return w::score(4);\n}\n'
# explicit specialisation gained by v.cpp (new probe, same shape as R4-2o)
A52sa = "#pragma once\n#include \"p.h\"\nnamespace w {\ntemplate <> struct Cfg<1151> {\n    static constexpr int kTile = 32;\n};\n}\n"
A52sb = "#pragma once\n#include \"p.h\"\nnamespace w {\n}\n"
C52sa = "#pragma once\n#include \"p.h\"\nint more();\n"
C52sb = ("#pragma once\n#include \"p.h\"\nint more();\nnamespace w {\ntemplate <> struct Cfg<1151> {\n"
         "    static constexpr int kTile = 32;\n};\n}\n")
P52s = "#pragma once\nnamespace w {\ntemplate <int A> struct Cfg {\n    static constexpr int kTile = 16;\n};\n}\n"
U52s = '#include "c.h"\nint useU() {\n  return w::Cfg<1151>::kTile;\n}\n'
V52s = '#include "c.h"\nint useV() {\n  return w::Cfg<1151>::kTile;\n}\n'
# plain type moved into a header with MORE includers: still a move (the 6 real cross-file move commits)
A52ta = "#pragma once\nstruct Ring {\n    int head;\n    int tail;\n};\n"
A52tb = "#pragma once\n"
C52tb = "#pragma once\nint more();\nstruct Ring {\n    int head;\n    int tail;\n};\n"
U52t = '#include "a.h"\n#include "c.h"\nint useU(Ring& r) {\n  return r.head;\n}\n'
# (pass 6, review5 A-1: v.cpp gains Ring and must not spell it, else lookup of Ring in v.cpp changes: a change)
V52t = '#include "c.h"\nint useV(int r) {\n  return r + 1;\n}\n'
# R4-18 header split, a.h deleted; R4-19 include switched in the same commit (old-tree includers)
A518 = "#pragma once\nnamespace w {\nint score(int v);\nint score(long v);\n}\n"
B518a = "#pragma once\nint more();\n"
B518b = "#pragma once\nint more();\nnamespace w {\nint score(long v);\n}\n"
C518a = "#pragma once\nint more2();\n"
C518b = "#pragma once\nint more2();\nnamespace w {\nint score(int v);\n}\n"
U518a = '#include "a.h"\nint useU() {\n  return w::score(3);\n}\n'
U518b = '#include "b.h"\nint useU() {\n  return w::score(3);\n}\n'
V518 = '#include "c.h"\nint useV() {\n  return w::score(4);\n}\n'
A519a = "#pragma once\nnamespace w {\nint score(int v);\nint score(long v);\n}\n"
A519b = "#pragma once\nnamespace w {\nint score(long v);\n}\n"
B519 = "#pragma once\nnamespace w {\nint score(long v);\n}\n"
C519a = "#pragma once\n#include \"a.h\"\nint more2();\n"
C519b = "#pragma once\n#include \"a.h\"\nint more2();\nnamespace w {\nint score(int v);\n}\n"
V519 = '#include "c.h"\nint useV() {\n  return w::score(4);\n}\n'
# name: (old, new, functions, needed, block, includers of the NEW tree's map)
PROBES5 = {
    "R4-1m macro moved u.cpp -> b.h (u.cpp does not include b.h; the old file is a TU)":
        ({"src/u.cpp": U51ma, "src/b.h": B51a, "src/v.cpp": V5_}, {"src/u.cpp": U51mb, "src/b.h": B51mb, "src/v.cpp": V5_},
         FNV5, ["useU"], [], {"src/b.h": ["src/v.cpp"]}),
    "R4-2 macro moved a.h -> c.h with more includers (v.cpp gains it)":
        ({"src/a.h": A52a, "src/c.h": C52a, "src/u.cpp": U52, "src/v.cpp": V52}, {"src/a.h": A52b, "src/c.h": C52b, "src/u.cpp": U52, "src/v.cpp": V52},
         FNV5, ["useV"], [], {"src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
    "R4-2o overload score(int) moved a.h -> c.h with more includers (v.cpp score(4) rebinds)":
        ({"src/a.h": A52oa, "src/c.h": C52oa, "src/u.cpp": U52o, "src/v.cpp": V52o}, {"src/a.h": A52ob, "src/c.h": C52ob, "src/u.cpp": U52o, "src/v.cpp": V52o},
         FNV5, ["useV"], [], {"src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
    "R4-2s explicit specialisation moved a.h -> c.h with more includers (v.cpp leaves the primary)":
        ({"src/a.h": A52sa, "src/c.h": C52sa, "src/p.h": P52s, "src/u.cpp": U52s, "src/v.cpp": V52s},
         {"src/a.h": A52sb, "src/c.h": C52sb, "src/p.h": P52s, "src/u.cpp": U52s, "src/v.cpp": V52s},
         FNV5, ["useV"], [], {"src/p.h": ["src/a.h", "src/c.h", "src/u.cpp", "src/v.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
    "R4-18 header split, a.h deleted, u.cpp now includes b.h only (old-tree includers of a.h)":
        ({"src/a.h": A518, "src/b.h": B518a, "src/c.h": C518a, "src/u.cpp": U518a, "src/v.cpp": V518},
         {"src/b.h": B518b, "src/c.h": C518b, "src/u.cpp": U518b, "src/v.cpp": V518},
         FNV5, ["useU"], [], {"src/b.h": ["src/u.cpp"], "src/c.h": ["src/v.cpp"]}),
    "R4-19 score(int) a.h -> c.h while u.cpp switches #include a.h -> b.h (old-tree includers of a.h)":
        ({"src/a.h": A519a, "src/b.h": B519, "src/c.h": C519a, "src/u.cpp": U518a, "src/v.cpp": V519},
         {"src/a.h": A519b, "src/b.h": B519, "src/c.h": C519b, "src/u.cpp": U518b, "src/v.cpp": V519},
         FNV5, ["useU"], [], {"src/a.h": ["src/c.h", "src/v.cpp"], "src/b.h": ["src/u.cpp"], "src/c.h": ["src/v.cpp"]}),
}
# blocks: still moves (build only, no function selected)
MOVES5 = {
    "plain type moved a.h -> c.h with more includers (types: loss direction only)":
        ({"src/a.h": A52ta, "src/c.h": C52a, "src/u.cpp": U52t, "src/v.cpp": V52t},
         {"src/a.h": A52tb, "src/c.h": C52tb, "src/u.cpp": U52t, "src/v.cpp": V52t},
         FNV5, {"src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
    "macro moved a.h -> c.h, both seen by exactly u.cpp and v.cpp":
        ({"src/a.h": A52a, "src/c.h": C52a, "src/u.cpp": '#include "a.h"\n' + U52, "src/v.cpp": '#include "a.h"\n' + V52},
         {"src/a.h": A52b, "src/c.h": C52b, "src/u.cpp": '#include "a.h"\n' + U52, "src/v.cpp": '#include "a.h"\n' + V52},
         FNV5, {"src/a.h": ["src/u.cpp", "src/v.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
}
# review4 R4-9 / R4-17: a changed / removed #define line names its macro: not an identifier-less table row
A59 = "#pragma once\n#define WHIRL_SAFE_COPY 1\nint other();\n"
U59 = '#include "a.h"\nint useU() {\n#if defined(WHIRL_FAST) || \\\n    defined(WHIRL_SAFE_COPY)\n  return 1;\n#else\n  return 2;\n#endif\n}\n'
A517 = "#pragma once\n#define WHIRL_ROWS 4\n"
U517 = '#include "a.h"\nstatic const int kLocal[] = {WHIRL_ROWS, 1};\nint useU() {\n  return kLocal[0];\n}\n'
V5n = '#include "a.h"\nint useV() {\n  return 0;\n}\n'
NOID5 = {
    "R4-9 macro removed (#define line)": ({"src/a.h": A59, "src/u.cpp": U59, "src/v.cpp": V5n},
                                         {"src/a.h": "#pragma once\nint other();\n", "src/u.cpp": U59, "src/v.cpp": V5n}),
    "R4-17 macro value changed (#define line)": ({"src/a.h": A517, "src/u.cpp": U517, "src/v.cpp": V5n},
                                                {"src/a.h": A517.replace("4", "8"), "src/u.cpp": U517, "src/v.cpp": V5n}),
}


def fake_tree_includes(rev):
    """impact.tree_includes on the fake repo: path -> #include targets of that revision"""
    import re as _re
    return {p: _re.findall(r'^\s*#\s*include\s*[<"]([^">]+)[">]', t, _re.M) for (r_, p), t in REPO.items() if r_ == rev}


# mt-p1 pass 6 (review5 A-1, probes scratch/mt_p1_review5/probe_r5.py): a moved statement of ANY kind that a TU newly
# sees is a change when that TU (or a file it reads) spells the declared name: partial specialisation, using-declaration,
# namespace constant / type shadowing a global. Each fails on a8bc01e (scratch/mt_p1_p6/before_impact_decl_test.txt)
P63 = "#pragma once\nnamespace w {\ntemplate <typename T> struct Cfg {\n    static constexpr int kTile = 16;\n};\n}\n"
A63a = "#pragma once\n#include \"p.h\"\nnamespace w {\ntemplate <typename T> struct Cfg<T*> {\n    static constexpr int kTile = 32;\n};\n}\n"
A63b = "#pragma once\n#include \"p.h\"\nnamespace w {\n}\n"
C63a = "#pragma once\n#include \"p.h\"\nint more();\n"
C63b = ("#pragma once\n#include \"p.h\"\nint more();\nnamespace w {\ntemplate <typename T> struct Cfg<T*> {\n"
        "    static constexpr int kTile = 32;\n};\n}\n")
U63 = '#include "a.h"\n#include "c.h"\nint useU() {\n  return w::Cfg<int*>::kTile;\n}\n'
V63 = '#include "c.h"\nint useV() {\n  return w::Cfg<int*>::kTile;\n}\n'
INC63 = {"src/p.h": ["src/a.h", "src/c.h", "src/u.cpp", "src/v.cpp"], "src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}
B65 = "#pragma once\nnamespace w {\nint score(long v);\n}\nnamespace w2 {\nint score(int v);\n}\n"
A65a = "#pragma once\n#include \"b.h\"\nnamespace w {\nusing w2::score;\n}\n"
A65b = "#pragma once\n#include \"b.h\"\nnamespace w {\n}\n"
C65a = "#pragma once\n#include \"b.h\"\nint more();\n"
C65b = "#pragma once\n#include \"b.h\"\nint more();\nnamespace w {\nusing w2::score;\n}\n"
U65 = '#include "a.h"\n#include "c.h"\nint useU() {\n  return w::score(3);\n}\n'
V65 = '#include "c.h"\nint useV() {\n  return w::score(4);\n}\n'
INC65 = {"src/b.h": ["src/a.h", "src/c.h", "src/u.cpp", "src/v.cpp"], "src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}
B67 = "#pragma once\nconstexpr int kTile = 16;\n"
A67a = "#pragma once\nnamespace w {\nconstexpr int kTile = 32;\n}\n"
A67b = "#pragma once\nnamespace w {\n}\n"
C67a = "#pragma once\nnamespace w {\nint more();\n}\n"
C67b = "#pragma once\nnamespace w {\nint more();\nconstexpr int kTile = 32;\n}\n"
U67 = '#include "a.h"\n#include "c.h"\nnamespace w {\nint useU() {\n  return kTile;\n}\n}\n'
V67 = '#include "b.h"\n#include "c.h"\nnamespace w {\nint useV() {\n  return kTile;\n}\n}\n'
FNW6 = [("src/u.cpp", "w::useU"), ("src/v.cpp", "w::useV")]
INC67 = {"src/b.h": ["src/v.cpp"], "src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}
B68 = "#pragma once\nstruct Tile {\n    int n = 16;\n};\n"
A68a = "#pragma once\nnamespace w {\nstruct Tile {\n    int n = 32;\n};\n}\n"
C68b = "#pragma once\nnamespace w {\nint more();\nstruct Tile {\n    int n = 32;\n};\n}\n"
U68 = '#include "a.h"\n#include "c.h"\nnamespace w {\nint useU() {\n  Tile t;\n  return t.n;\n}\n}\n'
V68 = '#include "b.h"\n#include "c.h"\nnamespace w {\nint useV() {\n  Tile t;\n  return t.n;\n}\n}\n'
T68 = '#pragma once\nnamespace w {\ninline int useT() {\n  Tile t;\n  return t.n;\n}\n}\n'
V68i = '#include "b.h"\n#include "c.h"\n#include "t.h"\nnamespace w {\nint useV() {\n  return useT();\n}\n}\n'
PROBES6 = {
    "R5-8i type w::Tile a.h -> c.h, gained v.cpp spells it only through its included t.h":
        ({"src/b.h": B68, "src/a.h": A68a, "src/c.h": C67a, "src/t.h": T68, "src/u.cpp": U68, "src/v.cpp": V68i},
         {"src/b.h": B68, "src/a.h": A67b, "src/c.h": C68b, "src/t.h": T68, "src/u.cpp": U68, "src/v.cpp": V68i},
         # (useT itself is not found: the user search covers c.h and its includers, not headers a TU includes beside
         # it - a pre-existing scope limit; the TU's function useV, which calls useT, is selected and covers its paths)
         FNW6 + [("src/t.h", "w::useT")], ["useV"], dict(INC67, **{"src/t.h": ["src/v.cpp"]})),
    "R5-3 partial specialisation Cfg<T*> a.h -> c.h, v.cpp gains it":
        ({"src/p.h": P63, "src/a.h": A63a, "src/c.h": C63a, "src/u.cpp": U63, "src/v.cpp": V63},
         {"src/p.h": P63, "src/a.h": A63b, "src/c.h": C63b, "src/u.cpp": U63, "src/v.cpp": V63}, FNV5, ["useV"], INC63),
    "R5-5 using w2::score a.h -> c.h, v.cpp gains it":
        ({"src/b.h": B65, "src/a.h": A65a, "src/c.h": C65a, "src/u.cpp": U65, "src/v.cpp": V65},
         {"src/b.h": B65, "src/a.h": A65b, "src/c.h": C65b, "src/u.cpp": U65, "src/v.cpp": V65}, FNV5, ["useV"], INC65),
    "R5-7 namespace constant w::kTile a.h -> c.h, v.cpp gains it (shadows ::kTile)":
        ({"src/b.h": B67, "src/a.h": A67a, "src/c.h": C67a, "src/u.cpp": U67, "src/v.cpp": V67},
         {"src/b.h": B67, "src/a.h": A67b, "src/c.h": C67b, "src/u.cpp": U67, "src/v.cpp": V67}, FNW6, ["useV"], INC67),
    "R5-8 type w::Tile a.h -> c.h, v.cpp gains it (shadows ::Tile)":
        ({"src/b.h": B68, "src/a.h": A68a, "src/c.h": C67a, "src/u.cpp": U68, "src/v.cpp": V68},
         {"src/b.h": B68, "src/a.h": A67b, "src/c.h": C68b, "src/u.cpp": U68, "src/v.cpp": V68}, FNW6, ["useV"], INC67),
}
# blocks (the 6 real cross-file move commits 51de103 / 3bfaac7 / dadae41 / 920ae63 / ef40a1f / 43205f5 stay moves,
# scratch/mt_p1_p6/moves_p6_vs_p5.txt): the gained TU spells a member name only (fields are reached through their
# class), or the name only in a comment, or the gained TU is a file added by the commit (no earlier meaning)
V68m = '#include "c.h"\nint useV() {\n  // a Tile is moved here\n  struct Q { int n = 1; } q;\n  return q.n;\n}\n'
MOVES6 = {
    "type w::Tile moved, gained v.cpp spells only the member name n and Tile in a comment":
        ({"src/a.h": A68a, "src/c.h": C67a, "src/u.cpp": U68, "src/v.cpp": V68m},
         {"src/a.h": A67b, "src/c.h": C68b, "src/u.cpp": U68, "src/v.cpp": V68m}, FNW6,
         {"src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
    "type w::Tile moved, the gained TU v.cpp is added by the commit":
        ({"src/a.h": A68a, "src/c.h": C67a, "src/u.cpp": U68},
         {"src/a.h": A67b, "src/c.h": C68b, "src/u.cpp": U68, "src/v.cpp": V68.replace('#include "b.h"\n', "")}, FNW6,
         {"src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}),
}


# mt-p1 pass 6b (p6_vs_024 failing rows): identifier-less continuation lines of a multi-line statement.
# T6b: string table, 2nd row changed (de40446 numerics.cpp:264 shape); the head line is unchanged, so the row alone must
# select the table's reader AND fail closed to the file scope (other() too: 0.2.4 selected the file's paths)
T6B = ('const char* const k_sw[] = {"FD_SPLITS", "GEMV_MAX",\n'
       '                            "GDN_SEQ", "NAIVE_ATTN", nullptr};\n'
       'int noteSw(const char* v) {\n  return k_sw[0] == v;\n}\nint other() {\n  return 1;\n}\n')
T6BN = T6B.replace('"GDN_SEQ", "NAIVE_ATTN"', '"GDN_SEQ", "NAIVE_ATTN", "MOE_BN"')
# D6b: wrapped declaration of a new function in a header (c80b27a vision.h:133 shape), only the continuation line
# changes: the callers are selected, the line is a declaration continuation (decl_cont), not an identifier-less row,
# and no file-scope fallback (the reader of another function in the includer is not selected)
D6BH = ("#pragma once\nstd::uint64_t placeFit(const std::array<std::uint64_t, 7>& need,\n"
        "                       std::array<std::pair<int, std::uint64_t>, 7>& at);\nint keep();\n")
D6BN = D6BH.replace("std::array<std::pair<int, std::uint64_t>, 7>& at);", "std::array<std::pair<int, std::uint64_t>, 8>& at);")
D6BC = ('#include "x.h"\nint useFit() {\n  std::array<std::pair<int, std::uint64_t>, 7> at;\n  return (int)placeFit({}, at);\n}\n'
        'int other() {\n  return keep();\n}\n')
real = impact.show
impact.show = fake_show
real_tree = getattr(impact, "tree_includes", None)
impact.tree_includes = fake_tree_includes
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
    # (pass 6, review5 A-1: model.cpp, the TU that spells Model, saw other.h in the old tree too; without that include
    # it newly sees the moved member and spells its class: a change, not a move)
    REPO.update({("A", "src/model.cpp"): '#include "other.h"\n' + CPP, ("B", "src/model.cpp"): '#include "other.h"\n' + CPP})
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

    # ---------------------------------------------------------------------------------------------------------------
    # mt-p1 pass 4 (progress/mt-p1.REVIEW3.md act-on 1-3, C-1, noted V4b / V5): the reviewer's probes
    # (scratch/mt_p1_rv3/probe_rv3.py, realmut_rv3.py G3) as tests; each fails on 14c576a
    for nm, (a, b, fns, need, block, inc) in sorted(PROBES4.items()):
        r, miss, sel = probe(a, b, fns, need, inc)
        expect(f"pass4 {nm}: needed {need} selected", not miss, (miss, sel, r["notes"][:3]))
        if block:
            expect(f"pass4 {nm}: block, not selected {block}", not (set(block) & sel), sel)
    # G3 (kernels_abi.h:453 kGemmsCfgs row = default of KernelTable::gemms_geom, the R9700 gemms launch geometry): the
    # reader precChoice gets golden (its path lists golden server) and the changed row is reported as identifier-less
    impact.PATH_GOLDEN["p_precChoice"] = ["server"]
    r, miss, sel = probe(*G3[:4], G3[5])
    expect("pass4 G3 table row -> member default -> .gemms_geom reader + golden_swift / golden_moe",
           not miss and {"golden_swift", "golden_moe"} <= set(r["release_gate_steps"]), (miss, sel, r["release_gate_steps"]))
    expect("pass4 G3 identifier-less row reported (noid_lines, for validate R-init-noid)", r.get("noid_lines"), r.get("noid_lines"))
    del impact.PATH_GOLDEN["p_precChoice"]
    r, _, sel = probe(*MV_IN[:4], MV_IN[5])
    expect("pass4 block: overload moved a.h -> b.h with every includer of a.h including b.h -> still a move (build only)",
           not sel, (sel, r["notes"][:3]))
    # mt-p1 pass 4b (hold LESS 4fae76e / c3601d4 / 7078143, src/cli/main.cpp WHIRL_HELP_MODES): a macro used only inside
    # namespace-scope help strings; the strings are read by commandHelp alone (wmain returns right after printHelp), so
    # the users are commandHelp, not every function of the file. Pass 3 found no user function, reported a map defect
    # and fell back to the file's paths (cmdChat = cli.decode / prefill / mtp / dflash): an over-selection, not ground truth
    HELP = ('#define HELP_MODES                 \\\n    "  --precise | --balance\\n"  \\\n    "    numerics mode %s\\n"\n\n'
            'const char* k_help_chat =\n    "usage: chat\\n"\n    HELP_MODES\n    "end\\n";\n'
            'const char* k_help_bench =\n    "usage: bench\\n"\n    HELP_MODES;\n'
            'const char* commandHelp(int c) {\n  if (c == 1) return k_help_chat;\n  if (c == 2) return k_help_bench;\n  return nullptr;\n}\n'
            'int cmdChat(int n) {\n  return n + 1;\n}\n')
    r, miss, sel = probe({"src/main.cpp": HELP % "(a)"}, {"src/main.cpp": HELP % "(b)"},
                         [("src/main.cpp", "commandHelp"), ("src/main.cpp", "cmdChat")], ["commandHelp"], {})
    expect("pass4b help-text macro -> namespace help strings -> commandHelp selected", not miss, (miss, sel, r["notes"][:3]))
    expect("pass4b block: help-text macro -> not cmdChat (no file-scope fallback)", "cmdChat" not in sel, (sel, r["notes"][:3]))
    expect("pass4b help-text macro: no map defect (the user is found)", not r["map_defects"], r["map_defects"])
    # ---------------------------------------------------------------------------------------------------------------
    # mt-p1 pass 5 (review4 A1 / A2 / A3 noid): each probe fails on a63f7ee
    for nm, (a, b, fns, need, block, inc) in sorted(PROBES5.items()):
        r, miss, sel = probe(a, b, fns, need, inc)
        expect(f"pass5 {nm}: needed {need} selected", not miss, (miss, sel, r["notes"][:3]))
    for nm, (a, b, fns, inc) in sorted(MOVES5.items()):
        r, miss, sel = probe(a, b, fns, [], inc)
        expect(f"pass5 block: {nm} -> still a move (build only)", not sel and any("moved / re-indented" in n for n in r["notes"]),
               (sel, r["notes"][:3]))
    # pass 3 occurrence count, with the TU closure satisfied (x.cpp and y.cpp include z.h as well), so the count check
    # alone must make it a change (the pass-3 fixture is now also caught by the closure)
    A5c = {"src/x.h": HA18a, "src/y.h": HA18a.replace("#pragma once", "#pragma once\n// y"), "src/z.h": "#pragma once\n",
           "src/x.cpp": '#include "z.h"\n' + CPP18, "src/y.cpp": '#include "y.h"\n#include "z.h"\nint yy() {\n  return 2;\n}\n'}
    B5c = dict(A5c, **{"src/x.h": "#pragma once\n", "src/y.h": "#pragma once\n// y\n", "src/z.h": HA18a})
    r, miss, sel = probe(A5c, B5c, [("src/x.cpp", "rows1151"), ("src/x.cpp", "other"), ("src/y.cpp", "yy")], ["rows1151"],
                         {"src/x.h": ["src/x.cpp"], "src/y.h": ["src/y.cpp"], "src/z.h": ["src/x.cpp", "src/y.cpp"]})
    expect("pass5 count 2 -> 1 with the TU closure satisfied: not a move", not miss, (miss, sel, r["notes"][:3]))
    for nm, (a, b) in sorted(NOID5.items()):
        r, miss, sel = probe(a, b, FNV5, ["useU"], {"src/a.h": ["src/u.cpp", "src/v.cpp"]})
        expect(f"pass5 {nm}: useU selected", not miss, (miss, sel))
        expect(f"pass5 block: {nm} -> no identifier-less line reported", not r.get("noid_lines"), r.get("noid_lines"))
    # mt-p1 pass 6 (review5 A-1): every kind is checked for TUs that newly see it and spell its name
    for nm, (a, b, fns, need, inc) in sorted(PROBES6.items()):
        r, miss, sel = probe(a, b, fns, need, inc)
        expect(f"pass6 {nm}: needed {need} selected", not miss, (miss, sel, r["notes"][:3]))
    for nm, (a, b, fns, inc) in sorted(MOVES6.items()):
        r, miss, sel = probe(a, b, fns, [], inc)
        # (the added v.cpp of the second block is itself a change: its own function is selected, the move is not)
        nosel = {"useU", "w::useU"} | (set() if "added" in nm else {"useV", "w::useV"})
        expect(f"pass6 block: {nm} -> still a move", not (sel & nosel)
               and any("moved / re-indented" in n for n in r["notes"] if n.startswith("src/c.h")), (sel, r["notes"][:4]))
    # mt-p1 pass 6b: identifier-less continuation lines (see T6B / D6BH)
    r, miss, sel = probe({"src/t.cpp": T6B}, {"src/t.cpp": T6BN}, [("src/t.cpp", "noteSw"), ("src/t.cpp", "other")], ["noteSw", "other"], {})
    expect("pass6b de40446 shape: changed identifier-less row of a table (head unchanged) -> its reader + 0.2.4 file scope",
           not miss, (miss, sel, r["notes"][:3]))
    expect("pass6b de40446 shape: the row stays an identifier-less row (noid_lines, validate C-3), not a continuation",
           r.get("noid_lines") and not r.get("decl_cont"), (r.get("noid_lines"), r.get("decl_cont")))
    expect("pass6b de40446 shape: file-scope reason names the initializer row",
           "identifier-less initializer row of k_sw" in str(r.get("paths")), str(r.get("paths"))[:300])
    r, miss, sel = probe({"src/x.h": D6BH, "src/x.cpp": D6BC}, {"src/x.h": D6BN, "src/x.cpp": D6BC},
                         [("src/x.cpp", "useFit"), ("src/x.cpp", "other")], ["useFit"], {"src/x.h": ["src/x.cpp"]})
    expect("pass6b c80b27a shape: changed continuation line of a function declaration -> its caller selected",
           not miss, (miss, sel, r["notes"][:3]))
    expect("pass6b c80b27a shape: reported as decl_cont (placeFit), not as an identifier-less row",
           not r.get("noid_lines") and sorted((c[1], c[2], tuple(c[3])) for c in r.get("decl_cont", [])) == [("new", 3, ("placeFit",)), ("old", 3, ("placeFit",))], (r.get("noid_lines"), r.get("decl_cont")))
    expect("pass6b block: c80b27a shape -> no file-scope fallback (other not selected)", "other" not in sel, sel)
    # mt-p1 pass 6c: since pass 6b an identifier-less initializer row also fails closed to the file scope, so the V6 / V7b
    # and G3 selection checks above hold even with the designator / member-default rules off (mut_p6b MISSED both).
    # These pin the rules themselves: the attribution (declarations row, reason of the reader's path) for V6 / V7b, and
    # for the member-default rule a NAMED constant (no identifier-less row, no file scope) read only through a member.
    for nm, fld in (("V6 designator value on its own line (C-1)", "Profile::vision_attn_threads"),
                    ("V7b designated array value continued on the next line (C-1)", "Profile::gemv_r")):
        a, b, fns, need, _, inc = PROBES4[nm]
        r, miss, sel = probe(a, b, fns, need, inc)
        expect(f"pass6c {nm}: the continuation row declares the field in force ({fld})",
               any(d["name"] == fld and d["what"] == "field" for d in r["declarations"]), r["declarations"])
        why = sorted(r["paths"].get("p_useU", [])) if isinstance(r["paths"], dict) else []
        expect(f"pass6c {nm}: useU selected as a reader of {fld} (not only by the file scope)",
               any(f"uses {fld}" in w for w in why), why)
    KD6 = "#pragma once\nconstexpr int kBn = 32;\nstruct KT {\n    int bn = kBn;\n};\nstruct Holder {\n    KT kt;\n};\n"
    KC6 = ('#include "k.h"\nint readBn(const Holder& h) {\n  return h.kt.bn;\n}\nint other() {\n  return 1;\n}\n')
    r, miss, sel = probe({"src/k.h": KD6, "src/m.cpp": KC6}, {"src/k.h": KD6.replace("= 32;", "= 64;"), "src/m.cpp": KC6},
                         [("src/m.cpp", "readBn"), ("src/m.cpp", "other")], ["readBn"], {"src/k.h": ["src/m.cpp"]})
    expect("pass6c member default from a named constant (kBn -> KT::bn = kBn) -> reader of .bn selected",
           not miss, (miss, sel, r["notes"][:3]))
    expect("pass6c block: member default from a named constant -> no file scope (other not selected)",
           "other" not in sel and not r.get("noid_lines"), (sel, r.get("noid_lines"), r["notes"][:3]))
    # mt-p1 pass 7 (review6 A-1, rows.txt: 3fa73e5 attention.h:32 `std::uint32_t n = 1;`, 709d575 hip.h:136
    # `bool ok = false;`, 54c4c6c `std::uint32_t lo = 0, hi = 0;`, ac7d33c `std::vector<float> lg;`): a row that spells
    # its declared name, however short, is that declarator: its readers are selected by the declared-name rule, it is
    # not an identifier-less row (no noid_lines, no file scope). Each fails on e8e495a (scratch/mt_p1_p7/before_*.txt)
    for nm, old_, new_, rd in (("bool ok = false -> true", "    bool ok = false;\n", "    bool ok = true;\n", "ok"),
                               ("std::uint32_t n = 1 -> 2", "    std::uint32_t n = 1;\n", "    std::uint32_t n = 2;\n", "n"),
                               ("lo = 0, hi = 0 -> hi = 1", "    std::uint32_t lo = 0, hi = 0;\n", "    std::uint32_t lo = 0, hi = 1;\n", "hi"),
                               ("member lg added (no initializer)", "", "    std::vector<float> lg;\n", "lg")):
        SH7 = "#pragma once\nstruct St {\n    int keep_me = 0;\n%s};\n"
        SC7 = ('#include "s.h"\nint readIt(const St& s) {\n  return int(s.%s);\n}\nint other() {\n  return 1;\n}\n' % rd)
        r, miss, sel = probe({"src/s.h": SH7 % old_, "src/m.cpp": SC7}, {"src/s.h": SH7 % new_, "src/m.cpp": SC7},
                             [("src/m.cpp", "readIt"), ("src/m.cpp", "other")], ["readIt"], {"src/s.h": ["src/m.cpp"]})
        expect(f"pass7 short member {nm}: reader of St::{rd} selected", not miss, (miss, sel, r["notes"][:3]))
        expect(f"pass7 block: short member {nm} -> not an identifier-less row, no file scope (other not selected)",
               "other" not in sel and not r.get("noid_lines") and r.get("noid_named"),
               (sel, r.get("noid_lines"), r.get("noid_named")))
    # the data rows of a table stay identifier-less and fail closed (de40446 / b13854f / k_env shapes), even when the
    # table's declared name is short: a row that does not spell it is not the declarator
    T7 = "#pragma once\nstatic const int kt[] = {\n    1, 2,\n    %s,\n};\nint other();\n"
    T7C = '#include "t.h"\nint readKt() {\n  return kt[3];\n}\nint other() {\n  return 1;\n}\n'
    r, miss, sel = probe({"src/t.h": T7 % "3, 4", "src/m.cpp": T7C}, {"src/t.h": T7 % "3, 5", "src/m.cpp": T7C},
                         [("src/m.cpp", "readKt"), ("src/m.cpp", "other")], ["readKt", "other"], {"src/t.h": ["src/m.cpp"]})
    expect("pass7 data row of the short-named table kt (3, 4 -> 3, 5) -> reader + file scope (fails closed)",
           not miss and r.get("noid_lines") and not r.get("noid_named"), (miss, sel, r.get("noid_lines"), r.get("noid_named")))
    # only the names declared ON the row count: a data row spelling a short name declared on another line (n) is still
    # an identifier-less row of kt
    T7n = "#pragma once\nconstexpr int n = 4;\nstatic const int kt[] = {\n    1, 2,\n    n, %s,\n};\nint other();\n"
    r, miss, sel = probe({"src/t.h": T7n % "4", "src/m.cpp": T7C}, {"src/t.h": T7n % "5", "src/m.cpp": T7C},
                         [("src/m.cpp", "readKt"), ("src/m.cpp", "other")], ["readKt", "other"], {"src/t.h": ["src/m.cpp"]})
    expect("pass7 data row `n, 4` -> `n, 5` of kt (n declared on another line) -> still identifier-less, file scope",
           not miss and r.get("noid_lines") and not r.get("noid_named"), (miss, sel, r.get("noid_lines"), r.get("noid_named")))
    # mt-p1 pass 8 (mut_p7 MISSED "declared names of the whole file, not of the row"): above, n's own line is unchanged,
    # so n is not among the change's declarations and the mutant cannot tell. Here the same commit also changes n's
    # declaration (n = 4 -> 5): n is now a declaration of the change, but on line 2, not on the data row (line 5), so
    # the row must stay identifier-less (noid_lines, not noid_named) and fail closed to the file scope (other selected)
    r, miss, sel = probe({"src/t.h": T7n % "4", "src/m.cpp": T7C},
                         {"src/t.h": (T7n % "5").replace("n = 4;", "n = 5;"), "src/m.cpp": T7C},
                         [("src/m.cpp", "readKt"), ("src/m.cpp", "other")], ["readKt", "other"], {"src/t.h": ["src/m.cpp"]})
    row5 = [x for x in (r.get("noid_lines") or []) if x.startswith("src/t.h:5 ")]
    named5 = [x for x in (r.get("noid_named") or []) if x.startswith("src/t.h:5 ")]
    expect("pass8 data row `n, 4` -> `n, 5` with n's declaration changed too (n declared on line 2, not on the row) "
           "-> row still identifier-less + file scope", not miss and row5 and not named5,
           (miss, sel, r.get("noid_lines"), r.get("noid_named")))
    # a closing line of the statement (`};` / `}();`, 8158dc3 device_profile.h:76 / :95, 703c666 launch_log.cpp:115):
    # no value of its own; it stays an identifier-less row for validate (C-3, noid_lines / noid_at) but does not fail
    # closed: a new table added with designated rows selects its readers only
    # mt-p1 pass 9 (review p7p8 F1): the designated rows use a long field name (block_n): a short `.bn = 64,` is a
    # data row that fails closed (designators are not declarators, see the pass9 tests below); with `.bn` this test
    # passed only because the designator was wrongly taken as the declarator of bn
    P7 = ("#pragma once\nstruct Pf {\n    int block_n;\n};\ninline constexpr Pf k_pa = {\n    .block_n = 32,\n};\n%sint other();\n")
    P7N = "inline constexpr Pf k_pb = {\n    .block_n = 64,\n};\n"
    P7C = '#include "p.h"\nint readPb() {\n  return k_pb.block_n;\n}\nint other() {\n  return 1;\n}\n'
    r, miss, sel = probe({"src/p.h": P7 % "", "src/m.cpp": P7C}, {"src/p.h": P7 % P7N, "src/m.cpp": P7C},
                         [("src/m.cpp", "readPb"), ("src/m.cpp", "other")], ["readPb"], {"src/p.h": ["src/m.cpp"]})
    expect("pass7 table k_pb added (designated row + closing `};`) -> reader selected", not miss, (miss, sel, r["notes"][:3]))
    expect("pass7 block: closing `};` -> noid_close (validate C-3 keeps it), no file scope (other not selected)",
           "other" not in sel and r.get("noid_close") and r.get("noid_lines") and not
           any("identifier-less initializer row" in w for v in r["paths"].values() for w in v),
           (sel, r.get("noid_close"), r.get("noid_lines")))
    # mt-p1 pass 9 (review p7p8 F1, probe P1): a designated data row with a SHORT field name in an existing table
    # (`.bn = 32,` -> `.bn = 48,` of k_pa; pick() returns a reference to the table, viaPick reads it through pick):
    # the designator names a member of Pf, it does not declare bn ([dcl.init.aggr]), so the row is an identifier-less
    # data row: noid_lines (not noid_named) and the 0.2.4 file scope (viaPick + other selected). Fails on the pass 7/8
    # tree (scratch/mt_p1_p9/before_p9.txt): only readPa selected, viaPick missed
    P9F = ("#pragma once\nstruct Pf {\n    int bn;\n    int bm;\n};\ninline constexpr Pf k_pa = {\n    .bn = %s,\n"
           "    .bm = 4,\n};\ninline const Pf& pick() { return k_pa; }\n")
    P9FC = ('#include "p.h"\nint readPa() {\n  return k_pa.bm;\n}\nint viaPick() {\n  return pick().bm * 2;\n}\n'
            'int other() {\n  return 1;\n}\n')
    r, miss, sel = probe({"src/p.h": P9F % "32", "src/m.cpp": P9FC}, {"src/p.h": P9F % "48", "src/m.cpp": P9FC},
                         [("src/m.cpp", "readPa"), ("src/m.cpp", "viaPick"), ("src/m.cpp", "other")],
                         ["readPa", "viaPick", "other"], {"src/p.h": ["src/m.cpp"]})
    expect("pass9 F1 designated short row `.bn = 32` -> 48 of k_pa -> data row, file scope (viaPick via pick() selected)",
           not miss and r.get("noid_lines") and not r.get("noid_named"), (miss, sel, r.get("noid_lines"), r.get("noid_named")))
    # control: a row that declares a short variable and also has a designator of the same spelling
    # (`Pf bn = {.bn = 1};` declares the variable bn) stays the declarator (noid_named), its reader is selected
    P9G = "#pragma once\nstruct Pf {\n    int bn;\n};\ninline constexpr Pf bn = {.bn = %s};\n"
    P9GC = '#include "p.h"\nint readBn() {\n  return bn.bn;\n}\nint other() {\n  return 1;\n}\n'
    r, miss, sel = probe({"src/p.h": P9G % "1", "src/m.cpp": P9GC}, {"src/p.h": P9G % "2", "src/m.cpp": P9GC},
                         [("src/m.cpp", "readBn"), ("src/m.cpp", "other")], ["readBn"], {"src/p.h": ["src/m.cpp"]})
    expect("pass9 F1 control `Pf bn = {.bn = 1}` -> 2: the variable bn is declared on the row -> noid_named, reader selected",
           not miss and r.get("noid_named") and not r.get("noid_lines"), (miss, sel, r.get("noid_lines"), r.get("noid_named")))
    # mt-p1 pass 9 (review p7p8 F2, probes P2 / P2L / P3 / P6 / P6L / P7): a value inside the braces of a single-line
    # brace initializer / lambda changed; the statement text up to `{` is identical on both sides, so the old signature
    # called it moved / re-indented and nothing was selected. The signature now includes the brace contents: the
    # declared name's readers are selected. Each fails on the pass 7/8 tree (scratch/mt_p1_p9/before_p9.txt)
    INC9 = {"src/p.h": ["src/m.cpp"]}
    for nm, hdr, old_, new_, rdr in (
            ("single-line kTable[] = {1, 2, 3} -> 4", "static const int kTable[] = {1, 2, %s};\n", "3", "4", "kTable[2]"),
            ("single-line short kt[] = {1, 2, 3} -> 4", "static const int kt[] = {1, 2, %s};\n", "3", "4", "kt[2]"),
            ("first line of a multi-line table kt[] = {1, 2, -> 9", "static const int kt[] = {1, %s,\n    3, 4};\n", "2", "9", "kt[1]"),
            ("lambda body twice: arg * 2 -> 3", "inline auto twice = [](int arg) { return arg * %s; };\n", "2", "3", "twice(2)"),
            ("short lambda f: a * 2 -> 3", "inline auto f = [](int a) { return a * %s; };\n", "2", "3", "f(2)"),
            ("brace-init Pr p{1} -> {2}", "struct Pr { int v; };\ninline Pr p{%s};\n", "1", "2", "p.v"),
            ('string table file_magic[8] = {"WHIRL01"} -> 02', 'inline const char file_magic[8] = {"WHIRL0%s"};\n', "1", "2", "file_magic[0]")):
        H9 = "#pragma once\n" + hdr
        C9 = '#include "p.h"\nint readIt() {\n  return int(%s);\n}\nint other() {\n  return 1;\n}\n' % rdr
        r, miss, sel = probe({"src/p.h": H9 % old_, "src/m.cpp": C9}, {"src/p.h": H9 % new_, "src/m.cpp": C9},
                             [("src/m.cpp", "readIt"), ("src/m.cpp", "other")], ["readIt"], INC9)
        expect(f"pass9 F2 {nm} -> not a move, reader selected",
               not miss and not any("moved / re-indented" in n for n in r["notes"]), (miss, sel, r["notes"][:3]))
    # controls: identical brace contents re-spaced, or moved to another header that every includer also sees, are
    # still a move (build only, readIt not selected); moved AND a value changed is not
    C9 = '#include "p.h"\nint readIt() {\n  return kTable[2];\n}\nint other() {\n  return 1;\n}\n'
    H9a = "#pragma once\nstatic const int kTable[] = {1, 2, 3};\nint keep();\n"
    r, miss, sel = probe({"src/p.h": H9a, "src/m.cpp": C9}, {"src/p.h": H9a.replace("{1, 2, 3}", "{1,  2,   3}"), "src/m.cpp": C9},
                         [("src/m.cpp", "readIt"), ("src/m.cpp", "other")], [], INC9)
    expect("pass9 F2 control: single-line table with re-spaced brace contents -> still a move (nothing selected)",
           not sel and any("moved / re-indented" in n for n in r["notes"]), (sel, r["notes"][:3]))
    Q9a = ('#pragma once\n#include "q.h"\nstatic const int kTable[] = {1, 2, 3};\n', "#pragma once\nint keep();\n")
    Q9b = ('#pragma once\n#include "q.h"\n', "#pragma once\nint keep();\nstatic const int kTable[] = {1, 2, 3};\n")
    INCQ = {"src/p.h": ["src/m.cpp"], "src/q.h": ["src/p.h", "src/m.cpp"]}
    r, miss, sel = probe({"src/p.h": Q9a[0], "src/q.h": Q9a[1], "src/m.cpp": C9}, {"src/p.h": Q9b[0], "src/q.h": Q9b[1], "src/m.cpp": C9},
                         [("src/m.cpp", "readIt"), ("src/m.cpp", "other")], [], INCQ)
    expect("pass9 F2 control: single-line table moved p.h -> q.h (q.h included by p.h), same contents -> still a move",
           "readIt" not in sel and any("moved / re-indented" in n for n in r["notes"]), (sel, r["notes"][:3]))
    r, miss, sel = probe({"src/p.h": Q9a[0], "src/q.h": Q9a[1], "src/m.cpp": C9},
                         {"src/p.h": Q9b[0], "src/q.h": Q9b[1].replace("2, 3}", "2, 4}"), "src/m.cpp": C9},
                         [("src/m.cpp", "readIt"), ("src/m.cpp", "other")], ["readIt"], INCQ)
    expect("pass9 F2 moved p.h -> q.h AND a value changed (3 -> 4) -> not a move, reader selected", not miss, (miss, sel, r["notes"][:3]))
    # mt-p1 pass 7 (review6 A-2, probe R6-1m): an enumerator of an UNSCOPED enum is declared in the enclosing scope; a TU
    # that newly sees it and spells the ENUMERATOR's name (not the type name) changes meaning (v.cpp's unqualified
    # kTile now finds w::kTile = 32 instead of ::kTile = 16): not a move. enum class needs Mode::, so it stays a move
    B7 = "#pragma once\nconstexpr int kTile = 16;\n"
    A7b = "#pragma once\nnamespace w {\n}\n"
    C7a = "#pragma once\nnamespace w {\nint more();\n}\n"
    U7 = '#include "a.h"\n#include "c.h"\nnamespace w {\nint useU() {\n  return kTile;\n}\n}\n'
    V7 = '#include "b.h"\n#include "c.h"\nnamespace w {\nint useV() {\n  return kTile;\n}\n}\n'
    A7m = "#pragma once\nnamespace w {\nenum Mode {\n    kTile = 32,\n};\n}\n"
    C7m = "#pragma once\nnamespace w {\nint more();\nenum Mode {\n    kTile = 32,\n};\n}\n"
    INC7 = {"src/b.h": ["src/v.cpp"], "src/a.h": ["src/u.cpp"], "src/c.h": ["src/u.cpp", "src/v.cpp"]}
    for nm, rep in (("enum Mode", "enum Mode"), ("enum Mode : int (release.h:17 shape)", "enum Mode : int")):
        r, miss, sel = probe({"src/b.h": B7, "src/a.h": A7m.replace("enum Mode", rep), "src/c.h": C7a, "src/u.cpp": U7, "src/v.cpp": V7},
                             {"src/b.h": B7, "src/a.h": A7b, "src/c.h": C7m.replace("enum Mode", rep), "src/u.cpp": U7, "src/v.cpp": V7},
                             FNW6, ["useV"], INC7)
        expect(f"pass7 R6-1m multi-line unscoped {nm} a.h -> c.h, v.cpp spells only kTile -> useV selected",
               not miss, (miss, sel, r["notes"][:3]))
    for nm, rep in (("enum class Mode", "enum class Mode"), ("enum struct Mode", "enum struct Mode")):
        r, miss, sel = probe({"src/b.h": B7, "src/a.h": A7m.replace("enum Mode", rep), "src/c.h": C7a, "src/u.cpp": U7, "src/v.cpp": V7},
                             {"src/b.h": B7, "src/a.h": A7b, "src/c.h": C7m.replace("enum Mode", rep), "src/u.cpp": U7, "src/v.cpp": V7},
                             FNW6, [], INC7)
        expect(f"pass7 block: R6-1s multi-line {nm} moved, v.cpp never names Mode -> still a move",
               not sel and any("moved / re-indented" in n for n in r["notes"] if n.startswith("src/c.h")), (sel, r["notes"][:4]))
    for sc, want in (((("ns:namespace w", "enum:enum Mode"), (), "kT = 1,"), True),
                     ((("enum:enum Exit : int",), (), "kOk = 0,"), True),
                     ((("enum:typedef enum",), (), "kA = 1,"), True),
                     ((("class:struct S", "enum:enum"), (), "kB = 2,"), True),
                     ((("ns:namespace w", "enum:enum class Mode"), (), "kT = 1,"), False),
                     ((("enum:enum struct Mode : unsigned",), (), "kT = 1,"), False)):
        expect(f"pass7 unscoped_enum({sc[0][-1]}) == {want}",
               impact.unscoped_enum({"name": "k", "cls": "E", "what": "enumerator", "sig": sc}) == want)
finally:
    impact.show = real
    if real_tree is not None:
        impact.tree_includes = real_tree
# mt-p1 pass 6 (review5 C-2): the git path of tree_includes / tree_tus, unpatched, on this repository's HEAD: the CLI
# translation unit includes whirl/common.h, so tree_tus must list src/cli/main.cpp as a TU of include/whirl/common.h,
# and a failing git grep (exit 2) must fall back to the map (None), never return a tree with no #include lines
if real_tree is not None:
    _inc = impact.tree_includes("HEAD")
    expect("pass6 C-2 git tree_includes(HEAD): src/cli/main.cpp includes whirl/common.h",
           _inc is not None and "whirl/common.h" in _inc.get("src/cli/main.cpp", []), (_inc or {}).get("src/cli/main.cpp"))
    _tt = impact.tree_tus("HEAD", {"includers": {}})
    expect("pass6 C-2 git tree_tus(HEAD): include/whirl/common.h is seen by src/cli/main.cpp (source tree)",
           _tt[1] == "tree" and "src/cli/main.cpp" in _tt[0]("include/whirl/common.h")
           and "include/whirl/common.h" in _tt[3]("src/cli/main.cpp"), _tt[1])
    _run = impact.subprocess.run

    def _grep_fails(cmd, *a, **k):
        r_ = _run(cmd, *a, **k)
        if "grep" in cmd:
            r_.returncode, r_.stdout = 2, ""
        return r_
    impact.subprocess.run = _grep_fails
    try:
        expect("pass6 C-2 git grep exit 2 -> tree_includes None (map fallback, not an empty include graph)",
               impact.tree_includes("HEAD") is None)
    finally:
        impact.subprocess.run = _run
# mt-p1 pass 6c: the same C-2 return-code rule with an injected fake runner (no git, no repository needed): the check
# above runs real git first and is vacuous where impact.ROOT is not a repository (a temp copy: ls-tree already fails)
if real_tree is not None:
    import types

    def _fake_git(grep_rc, grep_out):
        def _r(cmd, *a, **k):
            if "ls-tree" in cmd:
                return types.SimpleNamespace(returncode=0, stdout="src/a.cpp\nsrc/a.h\ndocs/x.md\n", stderr="")
            if "grep" in cmd:
                return types.SimpleNamespace(returncode=grep_rc, stdout=grep_out, stderr="fatal: fake" if grep_rc > 1 else "")
            raise AssertionError(f"unexpected command {cmd}")
        return _r
    _run = impact.subprocess.run
    try:
        impact.subprocess.run = _fake_git(0, 'R:src/a.cpp:#include "a.h"\n')
        _inc = impact.tree_includes("R")
        expect("pass6c C-2 fake runner: grep exit 0 -> the #include lines parsed",
               _inc == {"src/a.cpp": ["a.h"], "src/a.h": []}, _inc)
        impact.subprocess.run = _fake_git(1, "")
        _inc = impact.tree_includes("R")
        expect("pass6c C-2 fake runner: grep exit 1 (no match) -> files with no #include, not None",
               _inc == {"src/a.cpp": [], "src/a.h": []}, _inc)
        for rc_ in (2, 128):
            impact.subprocess.run = _fake_git(rc_, "")
            _inc = impact.tree_includes("R")
            expect(f"pass6c C-2 fake runner: grep exit {rc_} -> None (map fallback, fail closed)", _inc is None, _inc)
    finally:
        impact.subprocess.run = _run
print(f"impact decl test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
