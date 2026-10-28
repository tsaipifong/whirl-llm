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
case("Q4 pass: engine.cpp (compiled by the tsa build) -> cpu_tsa", steps("src/server/engine.cpp", fn_line("src/server/engine.cpp", "Engine::stop")),
     ["cpu_tsa", "cpu_analyze"])
case("Q4 pass: kv_tier.h -> cpu_tsa", steps("src/tier/kv_tier.h", 1), ["cpu_tsa"])
case("Q4 pass: safety tools -> cpu_tsa", steps("tools/safety/tsa/tsa_check.py", 1), ["cpu_tsa", "cpu_safety_selftest"])
case("Q4 block: unit test line without a lock token -> no cpu_tsa",
     steps("tests/unit/unit_tests.cpp", line_where("tests/unit/unit_tests.cpp", False)), ["cpu_analyze"], ["cpu_tsa"])
case("Q4 block: docs -> no cpu_tsa", steps("docs/building.md", 1), [], ["cpu_tsa"])
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
