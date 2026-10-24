# SPDX-License-Identifier: Apache-2.0
"""Block / pass test of impact.py's release-step selection for the arch024-S11c rules (CPU, a few seconds).

    python tools/impact/impact_steps_test.py [--map tools/impact_map.json]

Each case runs impact.analyze() in-process on a synthetic change (one changed line inside a named function, taken
from the committed map) and checks the selected release_gate steps:
  F-3 pass : chat template / tokenizer / protocol / engine / startup decisions -> cpu_pins
  F-3 block: a kernel .hip file, docs -> no cpu_pins
  F-4 pass : spec::raceE (the race key the server samples with) -> srv_balance + srv_fast (suite basic)
  F-4 note : spec::sampleRace is the host reference of the GPU race draw (tests only): cpu_tests + kt_full, no srv_*
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


def steps(path, line=None):
    ch = {path: {"old_path": path, "new": [line] if line else [], "old": []}}
    return set(impact.analyze(ch, m, None, "HEAD")["release_gate_steps"])


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
ok = 0
for name, good, bad, got in cases:
    ok += good
    print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f": {'; '.join(bad)} (got {', '.join(got)})"))
print(f"impact steps test: {ok}/{len(cases)} ok")
sys.exit(0 if ok == len(cases) else 1)
