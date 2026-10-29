# SPDX-License-Identifier: Apache-2.0
"""Block / pass test of impact.py's release-step selection for the arch024-S11c rules (CPU, a few seconds).

    python tools/impact/impact_steps_test.py [--map tools/impact_map.json]

Each case runs impact.analyze() in-process on a synthetic change (one changed line inside a named function, taken
from the committed map) and checks the selected release_gate steps:
  F-3 pass : chat template / tokenizer / protocol / engine / startup decisions -> cpu_pins
  F-3 block: a kernel .hip file, docs -> no cpu_pins
  F-4 pass : spec::raceE (the race key the server samples with) -> srv_balance + srv_fast (suite basic)
  F-4 note : spec::sampleRace is the host reference of the GPU race draw (tests only): cpu_tests + kt_full, no srv_*
  (the rule cases check rep["map_steps"], what the map's rules select; since arch025-srctext every selection also holds
  EVERY CPU step and CPU test, checked by the CPU-all cases for every change above)
Prints "impact steps test: N/N ok" and exits 0 only when every case holds.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impact  # noqa: E402

mp = sys.argv[sys.argv.index("--map") + 1] if "--map" in sys.argv else os.path.join(HERE, "..", "impact_map.json")
m = json.load(open(mp, encoding="utf-8"))


def fn_line(path, name):
    d = m["functions"].get(f"{path}::{name}")
    if d is None:
        raise SystemExit(f"impact steps test: SETUP {path}::{name} not in the map")
    return d["lines"][0] + 1


full_reps = []  # (change, full report): every one must contain every CPU step and every CPU test (arch025-srctext)


def steps(path, line=None):
    """the steps the map's RULES select (rep["map_steps"]); the selection itself is that plus every CPU step"""
    ch = {path: {"old_path": path, "new": [line] if line else [], "old": []}}
    r = impact.analyze(ch, m, None, "HEAD")
    full_reps.append((f"{path}:{line}", r))
    return set(r.get("map_steps", r["release_gate_steps"]))


cases = []


def case(name, got, want_in=(), want_out=()):
    bad = [f"missing {s}" for s in want_in if s not in got] + [f"unexpected {s}" for s in want_out if s in got]
    cases.append((name, not bad, bad, sorted(got)))


tpl = next(d["name"] for d in m["functions"].values() if d["file"] == "src/chat/chat_template.cpp")
case("F-3 pass: chat template -> cpu_pins", steps("src/chat/chat_template.cpp", fn_line("src/chat/chat_template.cpp", tpl)), ["cpu_pins"])
tk = next(d["name"] for d in m["functions"].values() if d["file"] == "src/tokenizer/tokenizer.cpp")
case("F-3 pass: tokenizer -> cpu_pins", steps("src/tokenizer/tokenizer.cpp", fn_line("src/tokenizer/tokenizer.cpp", tk)), ["cpu_pins"])
pr = next(d["name"] for d in m["functions"].values() if d["file"] == "src/server/protocol.cpp")
case("F-3 pass: protocol -> cpu_pins", steps("src/server/protocol.cpp", fn_line("src/server/protocol.cpp", pr)), ["cpu_pins"])
case("F-3 pass: engine -> cpu_pins", steps("src/server/engine.cpp", fn_line("src/server/engine.cpp", "Engine::allocState")), ["cpu_pins"])
case("F-3 pass: tests/pins/requests.jsonl -> cpu_pins", steps("tests/pins/requests.jsonl"), ["cpu_pins"])
hip = next(h for h in sorted(m["hip"]) if h.startswith("kernels/"))
case(f"F-3 block: {hip} -> no cpu_pins", steps(hip, 1), [], ["cpu_pins"])
case("F-3 block: docs -> no cpu_pins", steps("docs/building.md", 1), [], ["cpu_pins"])
case("F-4 pass: spec::raceE -> srv_balance + srv_fast",
     steps("src/common/spec_sample.cpp", fn_line("src/common/spec_sample.cpp", "raceE")), ["srv_balance", "srv_fast", "cpu_tests"])
case("F-4 note: spec::sampleRace (test reference) -> cpu_tests + kt_full only",
     steps("src/common/spec_sample.cpp", fn_line("src/common/spec_sample.cpp", "sampleRace")), ["cpu_tests", "kt_full"],
     ["srv_balance", "srv_fast"])
# arch025-Q1 / Q3 (PLAN (e)): C / C++ change -> cpu_analyze; lifetime-class change -> cpu_asan (tools/gates/modules.json "safety")
safety, smods = impact.load_safety()
tok = impact.re.compile(safety["lifetime"]["tokens"])
ROOT = impact.ROOT


def line_where(path, want_token):
    """first non-blank, non-comment line of a file that does (or does not) match the lifetime tokens"""
    for i, l in enumerate(open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read().splitlines(), 1):
        if l.strip() and not l.strip().startswith(("//", "#")) and bool(tok.search(l)) == want_token:
            return i
    raise SystemExit(f"impact steps test: SETUP no {'token' if want_token else 'plain'} line in {path}")


case("Q1 pass: page_pool (lifetime module) -> cpu_asan + cpu_analyze",
     steps("src/cache/page_pool.cpp", fn_line("src/cache/page_pool.cpp", "PagePool::takeFree")), ["cpu_asan", "cpu_analyze", "cpu_safety_selftest"])
case("Q1 pass: protocol.cpp line with a lifetime token -> cpu_asan",
     steps("src/server/protocol.cpp", line_where("src/server/protocol.cpp", True)), ["cpu_asan", "cpu_analyze"])
case("Q1 block: protocol.cpp line without a lifetime token -> cpu_analyze only",
     steps("src/server/protocol.cpp", line_where("src/server/protocol.cpp", False)), ["cpu_analyze"], ["cpu_asan"])
case("Q3 pass: test source -> cpu_analyze", steps("tests/unit/unit_tests.cpp", line_where("tests/unit/unit_tests.cpp", False)), ["cpu_analyze"])
case("Q1/Q3 pass: safety tools -> all three", steps("tools/safety/run_asan.bat", 1), ["cpu_asan", "cpu_analyze", "cpu_safety_selftest"])
case(f"Q3 block: {hip} -> no cpu_analyze / cpu_asan", steps(hip, 1), [], ["cpu_analyze", "cpu_asan"])
case("Q3 block: docs -> no cpu_analyze", steps("docs/building.md", 1), [], ["cpu_analyze", "cpu_asan"])
# arch025-Q4: threads class -> cpu_tsa
case("Q4 pass: engine_queue.cpp (compiled by the tsa build; Engine::stop moved there in N4e) -> cpu_tsa", steps("src/server/engine_queue.cpp", fn_line("src/server/engine_queue.cpp", "Engine::stop")),
     ["cpu_tsa", "cpu_analyze"])
case("Q4 pass: kv_tier.h -> cpu_tsa", steps("src/tier/kv_tier.h", 1), ["cpu_tsa"])
case("Q4 pass: safety tools -> cpu_tsa", steps("tools/safety/tsa/tsa_check.py", 1), ["cpu_tsa", "cpu_safety_selftest"])
case("Q4 block: unit test line without a lock token -> no cpu_tsa",
     steps("tests/unit/unit_tests.cpp", line_where("tests/unit/unit_tests.cpp", False)), ["cpu_analyze"], ["cpu_tsa"])
case("Q4 block: docs -> no cpu_tsa", steps("docs/building.md", 1), [], ["cpu_tsa"])
# arch025-n12 (PLAN N12, impact.py fall-through): a file the build compiles INTO the exes that is not C / C++ source (a
# bin2c input, res/*.rc.in / manifest, the build-time generators, CMake-configured .in templates) has an owner module or
# is build-wide (impact.BUILD_INPUTS). Before N12 a change to data/draft_vocab/subset_64k.bin, res/* or tools/bin2c
# selected no GPU step and module gate "none (no product code)".
def report(path, line=None):
    ch = {path: {"old_path": path, "new": [line] if line else [], "old": []}}
    r = impact.analyze(ch, m, None, "HEAD")
    full_reps.append((f"{path}:{line}", r))
    return r


def gate_of(r):
    return r.get("module_gate") or ""


GPU_ALL = impact.expand_steps([s.replace("<model>", md) for s in m["release_gate_steps"] if not s.startswith("cpu_")
                               for md in ("swift", "moe")])
r = report("data/draft_vocab/subset_64k.bin", 1)
case("N12 pass: draft vocab .bin (bin2c -> whirl_model) -> MTP paths + model.core (contract: full gate)",
     set(r["paths"]) | set(r["modules"]) | ({"full"} if gate_of(r).startswith("full gate") else set())
     | set(r.get("map_cpu_tests", {}).get("whirl-model-tests", [])),
     ["cli.mtp", "server.mtp", "model.core", "full", "testDraftVocab"])
r = report("res/whirl.rc.in", 12)
case("N12 pass: res/whirl.rc.in (VERSIONINFO of whirl.exe / whirl-server.exe) -> startup + module release (pkg_smoke)",
     set(r["paths"]) | set(r["modules"]) | ({"mod-release"} if gate_of(r).startswith("release_gate.ps1 -Module release") else set()),
     ["startup", "release", "mod-release"])
r = report("tools/gen-icon/main.cpp", 20)
case("N12 pass: tools/gen-icon (icon rendered at build time into the .rc) -> startup + module release",
     set(r["paths"]) | set(r["modules"]), ["startup", "release"])
r = report("res/whirl.manifest", 24)
case("N12 pass: res/whirl.manifest (process-wide UTF-8 code page / long paths / asInvoker) -> build-wide: full gate + every GPU step",
     ({"full"} if gate_of(r).startswith("full gate") else set()) | ({"all-gpu"} if GPU_ALL <= impact.expand_steps(r["map_steps"]) else set())
     | set(r["paths"]), ["full", "all-gpu", "startup"])
r = report("tools/bin2c/bin2c.cpp", 20)
case("N12 pass: tools/bin2c (writes every embedded code object / vision / vocab array) -> build-wide: full gate + every GPU step",
     ({"full"} if gate_of(r).startswith("full gate") else set()) | ({"all-gpu"} if GPU_ALL <= impact.expand_steps(r["map_steps"]) else set())
     | set(r["paths"]), ["full", "all-gpu", "build"])
r = report("tools/gen-unicode-tables/main.cpp", 20)
case("N12 pass: tools/gen-unicode-tables (not run by the build; its output unicode_tables.inc is checked in) -> regenerate note, no GPU step",
     {k.split(" ")[0] for k in r["extra"]} | {s for s in r["map_steps"] if not s.startswith("cpu_")}, ["regenerate"],
     sorted(GPU_ALL))
r = report("src/release/version.h.in", 8)
case("N12 pass: src/release/version.h.in (-> generated whirl/version.h: release.cpp, server/http.cpp) -> startup + http, callers protocol",
     set(r["paths"]) | {k for v in (r.get("module_callers") or {}).values() for k in v} | set(r["modules"]),
     ["startup", "http", "release", "protocol"])
r = report("src/hip/embedded_objects.cpp.in", 13)
case("N12 pass: src/hip/embedded_objects.cpp.in (configure_file -> whirl_hip) is in module hip.runtime, not 'in no module'",
     set(r["modules"]) | ({"no-module"} if "in no module" in gate_of(r) else set()), ["hip.runtime"], ["no-module"])
r = report("docs/building.md", 1)
case("N12 block: docs still select no module and no GPU step",
     set(r["modules"]) | {s for s in r["map_steps"] if not s.startswith("cpu_")}, [], ["model.core", "release", "golden_swift"])
r = report("src/tokenizer/unicode_tables.inc", 20)
case("N12 block: unicode_tables.inc keeps its source rule (tokenizer path, module tokenizer, not build-wide)",
     set(r["paths"]) | set(r["modules"]) | ({"full"} if gate_of(r).startswith("full gate") else set()), ["tokenizer"], ["full", "build"])


def cmake_inputs(root):
    """every repo path CMakeLists.txt / cmake/*.cmake name (a path literal, ${CMAKE_SOURCE_DIR}/..., or a relative one)"""
    import glob
    import re as _re
    pat = _re.compile(r"(?:\$\{(?:CMAKE_SOURCE_DIR|PROJECT_SOURCE_DIR|CMAKE_CURRENT_SOURCE_DIR)\}/|\$\{CMAKE_CURRENT_LIST_DIR\}/\.\./"
                      r"|(?<![\w/$}.-]))((?:src|include|kernels|tools|tests|data|res|redist)/[A-Za-z0-9_./*+-]+)")
    out = set()
    for f in [os.path.join(root, "CMakeLists.txt")] + sorted(glob.glob(os.path.join(root, "cmake", "*.cmake"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            for mt in pat.finditer(line.split("#", 1)[0]):
                out.add(mt.group(1).rstrip(")\""))
    return sorted(out)


# census: every path the build names that is not C / C++ / HIP source of src / include / kernels / tests (those have
# the source rules, the census and cmake_modules_check), and not a test folder / manifest (tests/* rule), needs a
# BUILD_INPUTS entry. A new data / res / generator input added to CMake without a rule fails here (the llama.cpp
# '**/.cmake' path-filter lesson, progress\arch025-n12.PRIOR_ART.md).
bi = getattr(impact, "build_input", lambda p: None)
SRC_EXT = (".cpp", ".h", ".hip", ".inc", ".c", ".hpp")
unruled = []
for p in cmake_inputs(ROOT):
    if p.startswith(("src/", "include/", "kernels/", "tests/")) and p.endswith(SRC_EXT):
        continue
    if p.startswith("tests/"):
        continue  # test folders / isolation manifests: FILE_RULES tests/* and tests/*/isolation.txt
    if p.startswith("tools/safety/"):
        continue  # inputs of the C++ safety steps: modules.json safety.own
    if bi(p) is None:
        unruled.append(p)
case("N12 census: every non-source path CMake names has a BUILD_INPUTS rule", set(unruled), [], unruled or ["x"])
case("N12 census setup: the CMake scan finds the known inputs",
     set(cmake_inputs(ROOT)), ["data/draft_vocab/subset_64k.bin", "res/whirl.rc.in", "res/whirl.manifest", "tools/bin2c/bin2c.cpp",
                               "src/hip/embedded_objects.cpp.in", "src/release/version.h.in"])
# every BUILD_INPUTS owner is a module, and agrees with modules.json where modules.json also covers the path
mods_all = impact.load_modules() or {}
bad_owner = []
for e in getattr(impact, "BUILD_INPUTS", []):
    own = e.get("owner")
    if own is not None and own not in mods_all:
        bad_owner.append(f"{e['glob']}: owner {own} is not a module")
    mj = [n for n, d in mods_all.items() if any(impact.fnmatch.fnmatch(e["glob"], g) for g in d.get("paths", []))]
    if mj and own not in mj:
        bad_owner.append(f"{e['glob']}: owner {own}, modules.json says {mj}")
    for u in e.get("users", []):
        if not os.path.exists(os.path.join(ROOT, u)):
            bad_owner.append(f"{e['glob']}: user {u} does not exist")
case("N12 census: BUILD_INPUTS owners are modules and match modules.json; users exist",
     set(bad_owner) | ({"table"} if getattr(impact, "BUILD_INPUTS", None) else set()), ["table"], bad_owner)
# .in templates under src / include / kernels are product files: census_check counts them (each in one module)
sys.path.insert(0, os.path.join(ROOT, "tools", "gates"))
import census_check  # noqa: E402
ins = [f for f in census_check.product_files(ROOT) if f.endswith(".in")]
case("N12 census: census_check.product_files lists the .in templates (embedded_objects.cpp.in, version.h.in)",
     set(ins), ["src/hip/embedded_objects.cpp.in", "src/release/version.h.in"])
# arch025-srctext (owner 2026-10-28 「98/103，沒選到的那5項，如果不耗什麼時，那乾脆都跑了吧?」): every CPU step and every
# CPU test is in EVERY selection (docs, kernel file, test file, product function alike); only GPU steps are map-selected.
cpu_steps, cpu_tests = impact.all_cpu(m)
case("CPU-all setup: the map lists CPU steps (cpu_steps parsed from release_gate.ps1) and CPU tests",
     {"cpu_steps"} if m.get("cpu_steps") and len(cpu_steps) >= 20 and sum(len(v) for v in cpu_tests.values()) >= 100 else set(),
     ["cpu_steps"])
for where, r in full_reps:
    miss = sorted(cpu_steps - set(r["release_gate_steps"]))
    miss_t = sorted(f"{e}:{t}" for e, v in cpu_tests.items() for t in v if t not in r["cpu_tests"].get(e, []))
    gpu_lost = sorted(set(r.get("map_steps", [])) - set(r["release_gate_steps"]))
    case(f"CPU-all pass: {where} selects every CPU step + test, keeps the map's steps",
         {"all"} if not miss and not miss_t and not gpu_lost else set(), ["all"])
    if miss or miss_t or gpu_lost:
        print(f"     left out: steps {miss[:6]} tests {miss_t[:6]} map steps lost {gpu_lost[:6]}")
# block: a map without the CPU-all rule would leave CPU steps out for docs (proves the case above can fail)
r0 = impact.analyze({"docs/building.md": {"old_path": "docs/building.md", "new": [1], "old": []}}, m, None, "HEAD")
case("CPU-all block: docs map pick alone has no CPU test step (the rule, not the map, adds them)",
     {"cpu_model"} if "cpu_model" in r0["release_gate_steps"] and "cpu_model" not in r0.get("map_steps", ["cpu_model"]) else set(),
     ["cpu_model"])
ok = 0
for name, good, bad, got in cases:
    ok += good
    print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f": {'; '.join(bad)} (got {', '.join(got)})"))
print(f"impact steps test: {ok}/{len(cases)} ok")
sys.exit(0 if ok == len(cases) else 1)
