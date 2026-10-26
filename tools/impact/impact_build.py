# SPDX-License-Identifier: Apache-2.0
"""Builds tools/impact_map.json: source -> product paths -> tests / gates -> release-matrix cells.

    python tools/impact/impact_build.py [--root REPO] [--out tools/impact_map.json] [--check]

Everything that can be read from the code is read from the code:
  * CMake: which source files each executable links (add_library / add_executable / target_sources /
    target_link_libraries, followed transitively), so a CPU test only covers code it really links.
  * C++ / HIP: function definitions and call sites (tools/impact/cxxscan.py); the call graph links a function
    to every function of the called name (over-approximation, the safe side). Generic names are cut.
  * Kernel dispatch: kernels_abi.cpp binds KernelTable fields to kernel names (L.req / L.opt / the
    {&KernelTable::f, "name"} table); a .hip file defining a name binds to the field; host functions that
    touch the field bind to the .hip file. The arch of a .hip file comes from the #if guards in
    kernels/whirl_kernels.hip (gfx1201 set vs gfx1151 set) and the CMake per-arch source lists.
  * Product paths: a few named root functions per path (PATHS below); a function belongs to a path when it is
    reachable from a root. A test belongs to a function when its test function reaches it.
What is declared, not derived (and says so in the JSON, "declared": true):
  * the root functions of each path, the server-gate suites' paths (the gate drives a server over HTTP, so
    there is no call edge), the path -> release-matrix metric, and the model-type narrowing name rules.
--check: rebuild in memory and exit 1 if the committed map differs (stale map).
Exit 1 in every mode (map not written) when a declared root (PATHS, CLI_GATES) resolves to no function
(arch024 A9); tools/impact/impact_build_test.py tests the block and pass paths.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cxxscan  # noqa: E402

SRC_EXT = (".cpp", ".h", ".hip", ".inc", ".in")
SCAN_DIRS = ("src", "include", "kernels", "tests", "tools")
DEBUG = {}  # internals for --why
MAX_FANOUT = 8  # a called name defined in more places than this is too generic to resolve
# arch024-S12: "check" is no longer cut: only two functions define it (Watch::check, hip_runtime check) and cutting it
# hid testWddmWatch -> Watch::check (392874b); MAX_FANOUT still cuts names that are really generic
STOP_NAMES = cxxscan.STD_CALLS | {"main", "wmain", "CHECK", "fmt", "out", "log", "run", "ti", "I", "alignUp",
                                  "msSince", "nowMs", "unixNow", "env", "die"}

# ------------------------------------------------------------------------------------------------
# declared: product paths and their root functions (short or qualified names; every root must resolve,
# the builder fails otherwise, so a rename cannot silently empty a path)
PATHS = {
    "cli.prefill":       {"roots": ["prefillLikeServer", "Model::prefill", "Model::prefillSegs", "Model::run"], "metric": "pp",
                          "desc": "CLI / model prefill (chunked forward over the prompt)"},
    "cli.decode":        {"roots": ["plainDecode", "Model::decodeStep", "Model::step"], "metric": "tg",
                          "desc": "CLI plain decode (one token per step)"},
    "cli.mtp":           {"roots": ["specDecode", "Model::mtpBatchStepEx", "Model::verifyBatchEnqueue", "Model::mtpEnqueue",
                                    "Model::verify"], "metric": "tg", "desc": "MTP draft / verify (CLI and the model side of the server)"},
    "cli.dflash":        {"roots": ["dflashDecode"], "metric": "tg", "desc": "DFlash2 drafter (CLI)"},
    "server.prefill":    {"roots": ["Engine::startJob", "Engine::prefillGroup", "Engine::prefillStep", "Engine::finishPrefill"],
                          "metric": "pp", "desc": "server prefill on the engine schedule (chunks, segments, groups)"},
    "server.decode":     {"roots": ["Engine::decodeCycle", "Engine::finishJob"], "metric": "tg",
                          "desc": "server decode cycle (batched slots)"},
    "server.mtp":        {"roots": ["Engine::pickDrafts", "Engine::splitDrafts", "planSlotDrafts", "Engine::draftCap"], "metric": "tg",
                          "desc": "server MTP / n-gram draft allocation and acceptance"},
    "server.sampling":   {"roots": ["Engine::sampleRow", "filterCands", "Engine::prepareRow", "Engine::probOf", "Engine::sampleRace"],
                          "metric": "tg", "desc": "sampling and speculative sampling"},
    "server.loop":       {"roots": ["Engine::runLoop", "Engine::submitAndWait"], "metric": None, "depth": 0,
                          "desc": "engine scheduler loop / admission"},
    "cache.prefix":      {"roots": ["PrefixStore::cacheMatch", "PrefixStore::speMatch", "Engine::saveCkpt", "CkptStore::save", "CkptStore::load", "Engine::createSpe", "Engine::attachSpe",
                                    "PrefixCacheImpl::matchSlot", "PrefixCacheImpl::matchShared", "PrefixCacheImpl::ckptSlotFor",
                                    "Engine::maybeSplitCkpt"],
                          "metric": "agent", "desc": "in-VRAM prefix cache: checkpoints, shared system prompts, page COW"},
    "cache.restore":     {"roots": ["Engine::tryRestore", "Engine::pollRestores", "TierBridge::restoreMatch", "TierBridge::startRestore",
                                    "TierBridge::pumpRestore", "TierBridge::restoreDone", "TierBridge::restoreFailed",
                                    "TierBridge::finishTail", "TierBridge::cutTail", "TierBridge::queuedReuse", "TierBridge::prefetchCandidate",
                                    "Tier::beginRestore", "Tier::lookup", "Tier::pumpRestore", "Tier::endRestore"],
                          "metric": "agent", "desc": "RAM / SSD tier restore into VRAM (warm: RAM, cold: SSD after a restart)"},
    "cache.spill":       {"roots": ["TierBridge::spillSlot", "TierBridge::spillSpe", "TierBridge::supersedeApply", "TierBridge::flushOnStop", "Engine::flushTierOnStop", "Tier::startWrite", "Tier::tick",
                                    "Tier::runJob", "Tier::waitForDedupBase"],
                          "metric": "agent", "desc": "VRAM -> RAM -> SSD spill, dedup, flush on stop"},
    "cache.fingerprint": {"roots": ["tierFingerprint", "serverFingerprintInputs", "fingerprintEnv", "Tier::index", "Tier::indexFile"],
                          "metric": None, "desc": "SSD tier identity: which saved states a new server may reuse"},
    "vision":            {"roots": ["Engine::visPrepare", "Engine::attachVision", "Qwen35Vision::encode", "Model::visRow",
                                    "Model::injectRows"], "metric": None, "desc": "image encoder and image-token injection"},
    "http":              {"roots": ["Engine::writeError", "StreamState::send"], "metric": None,
                          "desc": "HTTP transport, OpenAI protocol, streaming"},
    "load":              {"roots": ["Model::load", "loadOrTune", "Model::autotuneGemm", "Model::checkPreciseDecode"], "metric": "all",
                          "desc": "model load, kernel table, auto-tune, precise self-checks"},
    "startup":           {"roots": ["serveMain", "gpuPreflight", "diagnose", "probeTable", "loadOrThrow", "requireModelFile", "requirePortFree"],
                          "metric": None, "depth": 1,
                          "desc": "server composition root, GPU / DLL preflight (packaging)"},
}
# file-pattern roots for code with no function root (HTTP front end, protocol, tokenizer, template)
PATH_FILES = {
    "http": ["src/server/http.cpp", "src/server/http.h", "src/server/protocol.cpp", "src/server/protocol.h", "src/server/stream.cpp",
             "src/server/stream.h", "src/server/engine_api.h"],
    "tokenizer": ["src/tokenizer/*", "src/chat/*", "include/whirl/tokenizer.h", "include/whirl/chat*.h", "src/server/tokens.*"],
    "startup": ["src/release/*", "tools/package_release.ps1", "redist/*", "cmake/release.cmake", "src/hip/runtime_loader.cpp",
                "src/hip/hip_api.h", "include/whirl/hip_runtime_loader.h", "tests/hiprt/*"],
    "build": ["CMakeLists.txt", "cmake/*.cmake", "build.bat"],
}
PATHS.setdefault("tokenizer", {"roots": [], "metric": "all", "desc": "tokenizer, chat template, JSON"})
PATHS.setdefault("build", {"roots": [], "metric": "all", "desc": "build system (every binary)"})

# declared: model-type narrowing by name (Swift = dense qwen35, Ornith = MoE qwen35moe; both hybrid GDN + attention)
MOE_NAME = re.compile(r"moe|expert|router", re.I)
DENSE_ONLY_FUNCS = {"Model::ffnBlock"}  # the dense FFN; MoE layers run moeBlock instead

# declared: server-gate suites -> paths (tests/server/server_gate.cpp suite functions; HTTP-driven, no call edge)
GATE_SUITES = {
    "basic":        ["server.prefill", "server.decode", "server.mtp", "server.sampling", "server.loop", "http", "cache.prefix", "tokenizer"],
    "mt_cache":     ["cache.prefix", "server.prefill"],
    "mt_cache_precise": ["cache.prefix", "cache.restore", "server.prefill", "server.decode"],  # 0.2.2: precise multi-turn cache == cold
    "pool":         ["cache.prefix", "server.loop", "server.prefill"],
    "tier":         ["cache.spill", "cache.restore", "cache.fingerprint", "cache.prefix"],
    "sys":          ["cache.prefix", "cache.spill", "cache.restore"],
    "restore_conc": ["cache.restore", "cache.spill"],
    "sys_burst":    ["cache.prefix", "server.loop", "http"],      # CB-FLAKE repro suites (diagnostic, not in release_gate)
    "sys_d":        ["cache.prefix", "cache.restore", "server.loop", "http"],
    "http_stress":  ["http", "server.loop"],
    "vis":          ["vision"],
    "fast_stream":  ["server.decode", "server.mtp", "http"],
    "golden":       ["server.prefill", "server.decode", "server.mtp", "server.loop", "cache.prefix", "cache.restore", "cache.spill",
                     "cache.fingerprint", "vision"],
}
# declared: release_gate.ps1 steps -> what they run (step names are parsed from the script and checked below)
RELEASE_STEPS = {
    "cpu_tests": "exe:whirl-tests", "cpu_model": "exe:whirl-model-tests", "cpu_cache": "exe:whirl-cache-tests",
    "cpu_vision": "exe:whirl-vision-tests", "cpu_tier": "exe:whirl-tier-tests", "cpu_server": "exe:whirl-server-tests",
    "cpu_gate_selftest": "tool", "cpu_gate_nosuite": "tool", "cpu_server_only_typo": "tool", "cpu_golden_selftest": "tool",
    "cpu_golden_xcheck": "tool", "kt_full": "exe:whirl-kernel-test", "cpu_hiprt": "exe:whirl-hiprt-tests", "cpu_hiprt_nofakes": "exe:whirl-hiprt-tests",
    "cpu_device": "exe:whirl-device-tests",
    # REVIEW-ARCH3 P3-3: device::profile() does not compile with an hip::Arch enumerator without a row (listed with
    # the device tests: both guard include/whirl/device_profile.h)
    "cpu_profile_switch": "exe:whirl-device-tests",
    "cpu_ops": "exe:whirl-ops-tests",  # kernel choice per op (ARCH-1 K-1..K-5)
    # arch024-S11c (F-3): tests/pins/check.ps1 runs whirl-pins (prompt token hash, engine pins on the mock model) and
    # whirl-startup-replay (startup decision table); impact.py selects it for every product file those two link
    "cpu_pins": "tool",
}
CLI_GATES = {  # GPU gates run through whirl.exe subcommands: their roots are CLI functions (derived reach)
    "golden.single": ["goldenSingle"], "golden.pair": ["goldenPair"], "golden.ngram": ["goldenSingle"],
    "seqtest": ["cmdSeqtest"], "selftest": ["cmdSelftest"], "dflash_bench": ["dflashDecode"], "bench": ["cmdBench"],
}
METRIC_CELLS = {"pp": ["pp"], "tg": ["tg"], "agent": ["agent"], "all": ["pp", "tg", "agent"], None: []}
GPUS = {"gfx1201": "r97", "gfx1151": "s8"}
MODELS = {"dense": "swift", "moe": "ornith"}
MODES = ["precise", "balance", "fast"]


def rel(root, p):
    return os.path.relpath(p, root).replace("\\", "/")


def git(root, *a):
    return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def list_files(root):
    # arch025-M5: the working tree, not only the index: tracked files that still exist + untracked, not ignored files.
    # A refactor agent moves files without git (the main session stages them), so `git ls-files` alone listed the old
    # paths (src/model/forward/*.cpp, gone) and missed the new ones: 12 roots "not found". Same list as before on a
    # clean checkout.
    out = git(root, "ls-files", "--cached", "--others", "--exclude-standard")
    files = sorted({l.strip() for l in out.splitlines() if l.strip() and os.path.isfile(os.path.join(root, l.strip()))})
    if not files:  # not a git checkout
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".git", "build")]
            files += [rel(root, os.path.join(dp, f)) for f in fns]
    return sorted(files)


# ------------------------------------------------------------------------------------------------
# CMake: target -> sources, links

def parse_cmake(root, files):
    texts = []
    for f in files:
        if f == "CMakeLists.txt" or (f.startswith("cmake/") and f.endswith(".cmake")) or f.endswith("/CMakeLists.txt"):
            texts.append((f, open(os.path.join(root, f), encoding="utf-8", errors="replace").read()))
    srcs = defaultdict(set)
    links = defaultdict(set)
    globs = {}
    exes = set()
    var_globs = {}
    for f, t in texts:
        t = re.sub(r"#[^\n]*", "", t)
        for m in re.finditer(r"file\(GLOB\s+(\w+)\s+(?:CONFIGURE_DEPENDS\s+)?([^)]*)\)", t):
            var_globs[m.group(1)] = [g.replace("${CMAKE_SOURCE_DIR}/", "") for g in m.group(2).split()]
        for m in re.finditer(r"\b(add_library|add_executable|target_sources)\(\s*([\w-]+)([^)]*)\)", t):
            kind, name, body = m.group(1), m.group(2), m.group(3)
            if kind == "add_executable":
                exes.add(name)
            for tok in body.split():
                tok = tok.replace("${CMAKE_SOURCE_DIR}/", "").replace("${CMAKE_CURRENT_LIST_DIR}/../", "")
                vm = re.fullmatch(r"\$\{(\w+)\}", tok)
                if vm and vm.group(1) in var_globs:
                    for g in var_globs[vm.group(1)]:
                        srcs[name].update(x for x in files if fnmatch.fnmatch(x, g))
                elif tok.endswith((".cpp", ".h", ".hip")) and "$" not in tok:
                    srcs[name].add(tok)
        for m in re.finditer(r"target_link_libraries\(\s*([\w-]+)([^)]*)\)", t):
            for tok in m.group(2).split():
                if tok not in ("PRIVATE", "PUBLIC", "INTERFACE") and not tok.startswith(("$", '"')):
                    links[m.group(1)].add(tok)
    targets = set(srcs) | set(links)

    def closure(tg, seen=None):
        seen = seen if seen is not None else set()
        if tg in seen:
            return set()
        seen.add(tg)
        s = set(srcs.get(tg, ()))
        for l in links.get(tg, ()):
            if l in targets:
                s |= closure(l, seen)
        return s

    return {e: sorted(closure(e)) for e in sorted(exes)}, {t: sorted(srcs[t]) for t in sorted(srcs)}


# ------------------------------------------------------------------------------------------------
# kernels: arch of each .hip file, KernelTable fields <-> kernel names <-> .hip files

def hip_arch(root, files, scans):
    """kernels/*.hip -> set of archs whose code object compiles it (from the #if guards of the include tree)."""
    arch = defaultdict(set)

    def walk(f, archs):
        if f not in scans:
            return
        for inc, guard in scans[f].includes:
            a = set(archs)
            if guard == "gfx1151":
                a &= {"gfx1151"}
            elif guard == "not_gfx1151":
                a -= {"gfx1151"}
            elif guard == "gfx1201":
                a &= {"gfx1201"}
            if not a:
                continue
            p = os.path.normpath(os.path.join(os.path.dirname(f), inc)).replace("\\", "/")
            if p in scans and not a <= arch[p]:
                arch[p] |= a
                walk(p, a)

    top = "kernels/whirl_kernels.hip"
    arch[top] = {"gfx1201", "gfx1151"}
    walk(top, {"gfx1201", "gfx1151"})
    vis = "kernels/vision/vision.hip"
    if vis in scans:
        arch[vis] = {"gfx1201", "gfx1151"}
        walk(vis, {"gfx1201", "gfx1151"})
    return arch


def kernel_fields(root, scans):
    """KernelTable / Caps field -> kernel-name literals (from kernels_abi.cpp)."""
    p = "src/kernels_host/kernels_abi.cpp"
    t = open(os.path.join(root, p), encoding="utf-8", errors="replace").read()
    fields = defaultdict(set)
    for m in re.finditer(r"\b[kc]\.(\w+)[^=;\n]*=\s*[^;]*?L\.(?:req|opt)\(\s*([^;]*?)\)\s*[;!=)]", t):
        for lit in re.findall(r'"([A-Za-z0-9_]+)"', m.group(2)):
            fields[m.group(1)].add(lit)
    for m in re.finditer(r"\{\s*&KernelTable::(\w+)\s*,\s*\"([A-Za-z0-9_]+)\"", t):
        fields[m.group(1)].add(m.group(2))
    # name builders, e.g. gemvqNtName(...) -> "gemvq_nt", used as k.gemvq_nt[...] = L.req(gemvqNtName(...))
    builders = {}
    for m in re.finditer(r"std::string\s+(\w+Name)\([^)]*\)\s*\{([^}]*)\}", t):
        builders[m.group(1)] = set(re.findall(r'"([A-Za-z0-9_]+)"', m.group(2)))
    for m in re.finditer(r"\b[kc]\.(\w+)[^=;\n]*=\s*[^;]*?L\.(?:req|opt)\(\s*(\w+Name)\(", t):
        fields[m.group(1)] |= builders.get(m.group(2), set())
    return {f: sorted(v) for f, v in fields.items()}


def hip_fields(root, hip_files, fields):
    """.hip file -> KernelTable fields whose kernel-name literal occurs in it (as an identifier prefix)."""
    out = defaultdict(set)
    for hf in hip_files:
        t = open(os.path.join(root, hf), encoding="utf-8", errors="replace").read()
        idents = set(re.findall(r"[A-Za-z_]\w*", t))
        for fld, lits in fields.items():
            for lit in lits:
                stem = lit.rstrip("_")
                if len(stem) < 4:
                    continue
                if stem in idents or any(i.startswith(stem) for i in idents) or re.search(re.escape(stem) + r"\s*##", t):
                    out[hf].add(fld)
                    break
    return out


# ------------------------------------------------------------------------------------------------

def build(root):
    files = list_files(root)
    exe_files, lib_srcs = parse_cmake(root, files)
    scans = {}
    for f in files:
        if f.endswith(SRC_EXT) and f.split("/")[0] in SCAN_DIRS:
            try:
                scans[f] = cxxscan.scan(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
            except Exception as e:  # never silently: a file the scanner cannot read is reported
                print(f"impact_build: cannot scan {f}: {e}", file=sys.stderr)
    is_test = lambda f: f.startswith("tests/") or f.startswith("tools/")
    # function nodes
    nodes = {}  # id -> (file, Func)
    by_short = defaultdict(list)
    by_qual = defaultdict(list)
    for f, s in scans.items():
        for fn in s.funcs:
            nid = f"{f}::{fn.name}"
            if nid in nodes:  # overloads: merge into one node
                o = nodes[nid][1]
                o.calls |= fn.calls
                o.members |= fn.members
                o.idents |= fn.idents
                o.start, o.end = min(o.start, fn.start), max(o.end, fn.end)
                continue
            nodes[nid] = (f, fn)
            by_short[fn.short].append(nid)
            by_qual[fn.name].append(nid)
            q2 = "::".join(fn.name.split("::")[-2:])
            if q2 != fn.name:
                by_qual[q2].append(nid)
    # class bases (class X : public Y) and the identifiers each file can see (itself + its local includes)
    bases = defaultdict(set)
    classes = set()
    raw_idents = {}
    for f in scans:
        t = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        raw_idents[f] = set(re.findall(r"[A-Za-z_]\w*", t))
        for m in re.finditer(r"\b(?:class|struct)\s+(\w+)\s*(?:final\s*)?:\s*([^{;]+)\{", t):
            classes.add(m.group(1))
            for b in re.findall(r"(?:public|private|protected)?\s*(?:[\w]+::)*(\w+)", m.group(2)):
                if b not in ("public", "private", "protected", "virtual", ""):
                    bases[m.group(1)].add(b)
        for m in re.finditer(r"\b(?:class|struct)\s+(\w+)\s*(?:final\s*)?[:{]", t):
            classes.add(m.group(1))
    visible = {}
    for f, s in scans.items():
        v = set(raw_idents[f])
        for inc, _ in s.includes:
            for cand in (os.path.normpath(os.path.join(os.path.dirname(f), inc)).replace("\\", "/"), "include/" + inc, "src/" + inc):
                if cand in raw_idents:
                    v |= raw_idents[cand]
                    break
        visible[f] = v

    def class_seen(c, vis, depth=0):
        if c in vis:
            return True
        return depth < 4 and any(class_seen(b, vis, depth + 1) for b in bases.get(c, ()))

    # call edges. A name links to every function of that name, except: a target in the caller's own file wins
    # (file-local helpers), and a method of class C is linked only when C (or a base of C) is visible in the
    # caller's file (TpsTrace::tick in the CLI is not Tier::tick).
    edges = defaultdict(set)
    near_cut = {}  # called name -> largest visible fan-out seen, for names within 2 of MAX_FANOUT (or over it)
    for nid, (f, fn) in nodes.items():
        caller_cls = fn.name.split("::")[-2] if "::" in fn.name else None
        # constructions (`TestServer srv(su);`, `Engine e{...}`): a class named in the body links to its constructors
        for c in fn.idents & classes:
            for t in by_qual.get(f"{c}::{c}", []):
                if t != nid and not (is_test(nodes[t][0]) and not is_test(f)):
                    edges[nid].add(t)
        for c in fn.calls:
            if c in STOP_NAMES:
                continue
            tg = [t for t in by_short.get(c, []) if t != nid and not (is_test(nodes[t][0]) and not is_test(f))]
            local = [t for t in tg if nodes[t][0] == f]
            if local:
                tg = local
            # arch024-S12 (review C2): drop methods of classes the caller cannot see BEFORE counting the fan-out, so
            # a name is cut for being generic only by targets that could really be meant
            vis_tg = []
            for t in tg:
                tq = nodes[t][1].name.split("::")
                cls = tq[-2] if len(tq) >= 2 else None
                if cls and cls in classes and cls != caller_cls and nodes[t][0] != f and not class_seen(cls, visible[f]):
                    continue
                vis_tg.append(t)
            if vis_tg and len(vis_tg) >= MAX_FANOUT - 2:
                near_cut[c] = max(near_cut.get(c, 0), len(vis_tg))
            if not vis_tg or len(vis_tg) > MAX_FANOUT:
                continue
            for t in vis_tg:
                edges[nid].add(t)
    # kernel dispatch layer
    hip_files = [f for f in scans if f.endswith(".hip")]
    arch = hip_arch(root, files, scans)
    kfields = kernel_fields(root, scans)
    hf = hip_fields(root, hip_files, kfields)
    field_users = defaultdict(set)  # field -> host nodes touching it
    host_dirs = ("src/model/", "src/server/backend", "src/vision/", "tests/kernels/", "src/kernels_host/")
    for nid, (f, fn) in nodes.items():
        if not f.startswith(host_dirs) or f.endswith(".hip"):
            continue
        for fld in fn.members & set(kfields):
            field_users[fld].add(nid)
    # reachability helpers
    def resolve(name):
        r = by_qual.get(name) or by_short.get(name) or []
        return [x for x in r if not is_test(nodes[x][0])] or r

    def reach(roots, allow_tests=False, depth=None):
        seen = {r: 0 for r in roots}
        dq = deque(roots)
        while dq:
            x = dq.popleft()
            if depth is not None and seen[x] >= depth:
                continue
            for y in edges.get(x, ()):
                if y not in seen and (allow_tests or not is_test(nodes[y][0])):
                    seen[y] = seen[x] + 1
                    dq.append(y)
        return set(seen)

    DEBUG["edges"], DEBUG["nodes"], DEBUG["resolve"] = edges, nodes, resolve
    errors = []
    path_nodes = {}
    for p, d in PATHS.items():
        roots = []
        for r in d["roots"]:
            got = resolve(r)
            if not got:
                errors.append(f"path {p}: root '{r}' not found in the code")
            roots += got
        path_nodes[p] = reach(roots, depth=d.get("depth"))
    # moe / dense narrowing: a node is MoE-only when every path to it from the model roots passes a MoE-named node
    model_roots = [x for r in ["Model::run", "Model::decodeStep", "Model::mtpBatchStepEx", "Model::verifyBatchEnqueue", "Model::forward"]
                   for x in resolve(r)]
    allm = reach(model_roots)

    def reach_avoid(roots, avoid):
        seen = set(r for r in roots if not avoid(r))
        dq = deque(seen)
        while dq:
            x = dq.popleft()
            for y in edges.get(x, ()):
                if y not in seen and not avoid(y) and not is_test(nodes[y][0]):
                    seen.add(y)
                    dq.append(y)
        return seen

    no_moe = reach_avoid(model_roots, lambda n: bool(MOE_NAME.search(nodes[n][1].short)))
    no_dense = reach_avoid(model_roots, lambda n: nodes[n][1].name in DENSE_ONLY_FUNCS)
    def models_of(nid):
        if nid in allm and nid not in no_moe:
            return ["moe"]
        if nid in allm and nid not in no_dense:
            return ["dense"]
        return ["dense", "moe"]

    # tests: every test function (short name test*) of every test executable; GPU kernel-test families too
    tests = {}
    for exe, fl in exe_files.items():
        if not (exe.endswith("-tests") or exe == "whirl-kernel-test"):
            continue
        linked = set(fl)
        for f in fl:
            if not is_test(f) or f not in scans:
                continue
            for fn in scans[f].funcs:
                if not re.match(r"(kt::)?(test|bench)[A-Z]", fn.name.split("::")[-1] if "::" not in fn.name else fn.short) and not fn.short.startswith("test"):
                    continue
                nid = f"{f}::{fn.name}"
                r = reach([nid], allow_tests=True)
                r = {x for x in r if nodes[x][0] in linked or nodes[x][0].endswith(".h")}
                tid = f"{exe}:{fn.short}"
                tests[tid] = {"exe": exe, "gpu": exe == "whirl-kernel-test", "reach": r}
    for g, roots in CLI_GATES.items():
        rr = []
        for r in roots:
            got = resolve(r)
            if not got:
                errors.append(f"cli gate {g}: root '{r}' not found")
            rr += got
        tests[f"gpu:{g}"] = {"exe": "whirl", "gpu": True, "reach": reach(rr)}

    # per-function record
    funcs = {}
    for nid, (f, fn) in nodes.items():
        if is_test(f):
            continue
        paths = sorted(p for p, s in path_nodes.items() if nid in s)
        funcs[nid] = {"file": f, "name": fn.name, "lines": [fn.start, fn.end], "paths": paths, "models": models_of(nid),
                      "tests": sorted(t for t, d in tests.items() if nid in d["reach"])}
    # per-.hip record: arch, fields, host users -> paths and tests
    hip_idents = {h: set(re.findall(r"[A-Za-z_]\w*", open(os.path.join(root, h), encoding="utf-8", errors="replace").read()))
                  for h in hip_files}
    kt_lits = defaultdict(set)  # kernel-test id -> string literals inside the test function (and the helpers it reaches in tests/kernels)
    for t, d in tests.items():
        if d["exe"] != "whirl-kernel-test":
            continue
        for x in d["reach"]:
            f, fn = nodes[x]
            if not f.startswith("tests/kernels/"):
                continue
            for ln, lit in scans[f].strings:
                if fn.start <= ln <= fn.end and re.fullmatch(r"[a-z][a-z0-9_]+", lit):
                    kt_lits[t].add(lit.rstrip("_"))
    hips = {}
    for h in sorted(hip_files):
        if h.startswith("tests/"):
            continue
        flds = sorted(hf.get(h, ()))
        users = set()
        for fl in flds:
            users |= field_users.get(fl, set())
        prod_users = {u for u in users if not is_test(nodes[u][0])}
        paths = sorted({p for u in prod_users for p in funcs.get(u, {}).get("paths", [])})
        ktests = sorted({f"whirl-kernel-test:{nodes[u][1].short}" for u in users if nodes[u][0].startswith("tests/kernels/")})
        # kernel-test functions that reach a user of these fields
        for t, d in tests.items():
            if d["exe"] == "whirl-kernel-test" and d["reach"] & users:
                ktests.append(t)
        # kernel-test functions that name one of this file's kernels in a string literal (kernels fetched by name)
        hid = hip_idents.get(h, set())
        for t, lits in kt_lits.items():
            if any(len(l) >= 5 and (l in hid or any(i.startswith(l) for i in hid)) for l in lits):
                ktests.append(t)
        mdl = ["moe"] if MOE_NAME.search(os.path.basename(h)) else ["dense", "moe"]
        a = sorted(arch.get(h, set())) or (["gfx1151"] if "/gfx1151/" in h else ["gfx1201", "gfx1151"])
        hips[h] = {"arch": a, "fields": flds, "host_users": sorted(nodes[u][1].name for u in prod_users),
                   "paths": paths or ["load"], "models": mdl, "tests": sorted(set(ktests))}
    # file index
    file_paths = defaultdict(set)
    for nid, d in funcs.items():
        file_paths[d["file"]].update(d["paths"])
    for p, pats in PATH_FILES.items():
        for f in files:
            if any(fnmatch.fnmatch(f, pt) for pt in pats):
                file_paths[f].add(p)
    # reverse include closure: header -> every scanned file that includes it (directly or not)
    inc_by = defaultdict(set)
    for f, sc in scans.items():
        for inc, _ in sc.includes:
            for cand in (os.path.normpath(os.path.join(os.path.dirname(f), inc)).replace("\\", "/"), "include/" + inc, "src/" + inc,
                         "kernels/" + inc):
                if cand in scans:
                    inc_by[cand].add(f)
                    break
    includers = {}
    for h in inc_by:
        seen, dq = set(), deque([h])
        while dq:
            x = dq.popleft()
            for y in inc_by.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        includers[h] = sorted(seen)
    # release gate steps (parsed from the script, so a renamed step is noticed)
    rg = os.path.join(root, "tools/gates/release_gate.ps1")
    rg_steps = []
    srv_steps = {}
    if os.path.exists(rg):
        t = open(rg, encoding="utf-8", errors="replace").read()
        rg_steps = sorted(set(re.findall(r'AddStep\s+"([^"$]+)"', t)) | {"st_<model>_<mode>", "golden_<model>", "seq_<model>_<mode>", "dfl_<mode>", "cpu_<exe>"})
        # server-gate steps: AddStep "srv_x" $true (GateCmd $MODEL "suite,suite" "mode" ...)
        for mm in re.finditer(r'AddStep\s+"(srv_\w+)"\s+\$true\s+\(GateCmd\s+\$(\w+)\s+"([^"]*)"\s+"([^"]*)"', t):
            srv_steps[mm.group(1)] = {"model": {"SW": "dense", "MX": "moe", "Q": "dense-q4"}.get(mm.group(2), mm.group(2)),
                                      "suites": mm.group(3).split(","), "mode": mm.group(4) or "balance"}
        for s in RELEASE_STEPS:
            if s.startswith("cpu_") and s not in t and "cpu_$(" not in t:
                errors.append(f"release step {s} not in release_gate.ps1")
    # every server-gate suite must be declared (a new suite with no paths would never be asked for)
    sg = os.path.join(root, "tests/server/server_gate.cpp")
    if os.path.exists(sg):
        for su in sorted(set(re.findall(r'\{"(\w+)",\s*suite[A-Z]\w*\}', open(sg, encoding="utf-8", errors="replace").read()))):
            if su not in GATE_SUITES and su not in ("vis_speed",):
                errors.append(f"server-gate suite '{su}' is not declared in GATE_SUITES (tools/impact/impact_build.py)")
    # gates per path (declared suites + derived CLI gates + derived CPU tests)
    gates = {}
    for p in PATHS:
        g = set()
        for su, ps in GATE_SUITES.items():
            if p in ps:
                g.add(f"server-gate:{su}")
        for t, d in tests.items():
            if d["reach"] & path_nodes[p] and t.startswith("gpu:"):
                g.add(t)
        gates[p] = sorted(g)
    out = {
        "_doc": __doc__.strip().splitlines()[0] + " Generated by tools/impact/impact_build.py; do not edit by hand.",
        "source_commit": git(root, "rev-parse", "HEAD").strip(),
        "tree_hash": tree_hash(root, files),
        "declared": {"paths": {p: {"roots": d["roots"], "files": PATH_FILES.get(p, []), "metric": d["metric"], "desc": d["desc"]}
                               for p, d in PATHS.items()},
                     "gate_suites": GATE_SUITES, "cli_gates": CLI_GATES, "release_steps": RELEASE_STEPS,
                     "moe_name_rule": MOE_NAME.pattern, "dense_only": sorted(DENSE_ONLY_FUNCS), "metric_cells": {str(k): v for k, v in METRIC_CELLS.items()},
                     "gpus": GPUS, "models": MODELS, "modes": MODES},
        "executables": exe_files,
        "kernel_fields": kfields,
        "hip": hips,
        "functions": funcs,
        "files": {f: sorted(v) for f, v in sorted(file_paths.items())},
        "tests": {t: {"exe": d["exe"], "gpu": d["gpu"]} for t, d in sorted(tests.items())},
        "path_gates": gates,
        "release_gate_steps": rg_steps,
        "release_srv_steps": srv_steps,
        "includers": includers,
        "fanout_near_cut": {k: v for k, v in sorted(near_cut.items())},  # arch024-S12 C2: > MAX_FANOUT = cut
        "stats": {"files_scanned": len(scans), "functions": len(nodes), "edges": sum(len(v) for v in edges.values()),
                  "tests": len(tests), "hip_files": len(hips), "kernel_fields": len(kfields)},
        "errors": errors,
    }
    return out


def tree_hash(root, files):
    h = hashlib.sha256()
    for f in files:
        if f.endswith(SRC_EXT + (".cmake", ".txt", ".ps1")) and f.split("/")[0] in SCAN_DIRS + ("cmake", "CMakeLists.txt"):
            try:
                h.update(f.encode() + b"\0" + open(os.path.join(root, f), "rb").read())
            except OSError:
                pass
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--why", nargs=2, metavar=("FROM", "TO"), help="print the call chain the map uses from function FROM to TO")
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.root, "tools", "impact_map.json")
    m = build(a.root)
    # arch024 A9: a root (path or CLI gate) that resolves to nothing means a declared path lost its entry point
    # (renamed / moved / deleted function): the map would silently drop that path's functions and tests. Every mode
    # fails with exit 1 and nothing is written, so the step that moved the symbol must update the roots.
    if m["errors"]:
        for e in m["errors"]:
            print("impact_build: ERROR " + e, file=sys.stderr)
        print(f"impact_build: {len(m['errors'])} error(s); map not written. Update the roots in tools/impact/impact_build.py "
              "(PATHS / CLI_GATES) in the same commit that renames, moves or deletes the function.", file=sys.stderr)
        return 1
    if a.why:
        e, nd, res = DEBUG["edges"], DEBUG["nodes"], DEBUG["resolve"]
        src, dst = res(a.why[0]), set(res(a.why[1]))
        prev = {s_: None for s_ in src}
        dq = deque(src)
        while dq:
            x = dq.popleft()
            for y in e.get(x, ()):
                if y not in prev:
                    prev[y] = x
                    dq.append(y)
        hit = [d_ for d_ in dst if d_ in prev]
        if not hit:
            print(f"no call chain from {a.why[0]} to {a.why[1]}")
            return 1
        ch, x = [], hit[0]
        while x is not None:
            ch.append(x)
            x = prev[x]
        print("\n  -> ".join(reversed(ch)))
        return 0
    text = json.dumps(m, separators=(",", ":"), ensure_ascii=False) + "\n"
    if a.check:
        try:
            old = json.load(open(out_path, encoding="utf-8"))
        except (OSError, ValueError):
            print(f"impact_build --check: no map at {out_path}")
            return 1
        if old.get("tree_hash") != m["tree_hash"]:
            print(f"impact_build --check: map is stale (map tree {old.get('tree_hash')}, sources {m['tree_hash']}); rerun impact_build.py")
            return 1
        print("impact_build --check: map is current")
        return 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(text)
    s = m["stats"]
    print(f"impact_build: {out_path}: {s['files_scanned']} files, {s['functions']} functions, {s['edges']} call edges, "
          f"{s['tests']} tests, {s['hip_files']} .hip files, {s['kernel_fields']} kernel-table fields, 0 error(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
