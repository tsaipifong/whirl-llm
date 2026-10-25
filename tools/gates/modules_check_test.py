# SPDX-License-Identifier: Apache-2.0
"""Block and pass tests of tools/gates/modules_check.py and of impact.py's module mapping (arch024-S12). CPU only.

    python tools/gates/modules_check_test.py        exit 0 = "modules check test: N/N ok"
"""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "impact"))
import modules_check  # noqa: E402
import impact  # noqa: E402

STEPS = {"cpu_server", "cpu_cache", "golden_swift", "srv_balance", "seq_swift_precise"}
FILES = ["src/server/engine.cpp", "src/cache/ckpt_store.cpp", "src/cache/ckpt_store.h", "include/whirl/prefill_schedule.h"]
BASE = {"status": "dev-only: test", "modules": {
    "a": {"id": "MA", "paths": ["src/cache/ckpt_store.*"], "steps": ["cpu_cache", "golden_swift"],
          "when": {"include/whirl/prefill_schedule.h": ["seq_swift_precise"]}},
    "later": {"id": "MB", "paths": ["src/server/later.*"], "lands_on": "branch-x", "steps": ["cpu_server"],
              "pending": [{"step": "cpu_future", "lands_in": "S99"}]},
    "x": {"id": "X", "paths": ["src/device/backend.h"], "lands_on": "n/a", "full": True}}}


def mut(f):
    d = copy.deepcopy(BASE)
    f(d["modules"])
    return d


CASES = [  # name, modules json, expect ok, substring of an error (block cases)
    ("pass: base (declared module not on this branch, pending step)", BASE, True, None),
    ("block: status not dev-only", dict(BASE, status="release"), False, "dev-only"),
    ("block: unknown step", mut(lambda m: m["a"]["steps"].append("no_such")), False, "step no_such"),
    ("block: stale pending", mut(lambda m: m["later"]["pending"].append({"step": "cpu_server", "lands_in": "S1"})), False, "pending step cpu_server"),
    ("block: unknown when-step", mut(lambda m: m["a"]["when"].update({"x.h": ["nope"]})), False, "step nope"),
    ("pass: safety rules with gate steps and known modules", dict(BASE, safety={"_doc": [], "lifetime": {"modules": ["a"], "steps": ["cpu_cache"]}}), True, None),
    ("block: safety step not in the gate (arch025-Q1)", dict(BASE, safety={"cxx": {"steps": ["cpu_analyze_typo"]}}), False, "safety cxx: step cpu_analyze_typo"),
    ("block: safety lifetime module unknown (arch025-Q1)", dict(BASE, safety={"lifetime": {"modules": ["nope"], "steps": []}}), False, "safety lifetime: module nope"),
    ("block: paths missing, no lands_on", mut(lambda m: m["later"].pop("lands_on")), False, "no lands_on"),
    ("block: file in two modules", mut(lambda m: m.update(b={"id": "MC", "paths": ["src/cache/*"], "steps": ["cpu_cache"]})), False, "two modules"),
    ("block: no steps, not full", mut(lambda m: m["a"].update(steps=[])), False, "no steps"),
    ("block: no id", mut(lambda m: m["a"].pop("id")), False, "no id"),
    ("pass: callers rdeps / list", mut(lambda m: (m["a"].update(callers="rdeps"), m["later"].update(callers=["a"]))), True, None),
    ("block: manual free text names a gate step (arch025-kvram719tag)", mut(lambda m: m["a"].update(manual=["P-719: srv_balance R9700"])), False, "names gate step srv_balance"),
    ("block: manual free text names a step without its prefix (arch025-kvram719tag)", mut(lambda m: m["a"].update(manual=["P-719: balance R9700"])), False, "names gate step srv_balance"),
    ("pass: manual free text with no gate step", mut(lambda m: m["a"].update(manual=["P-size: startup sizing lines", "vram_hog scenario once"])), True, None),
    ("block: callers names an unknown module", mut(lambda m: m["a"].update(callers=["nope"])), False, "callers must be"),
]


def main():
    ok = bad = 0
    for name, mj, want_ok, sub in CASES:
        errs, _ = modules_check.check(mj, FILES, STEPS)
        good = (not errs) if want_ok else (bool(errs) and any(sub in e for e in errs))
        print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f": errors {errs}"))
        ok, bad = ok + good, bad + (not good)
    # impact.py module mapping (dev-time module gate suggestion)
    mods = BASE["modules"]
    ch = lambda *fs: {f: {"new": [1], "old": [], "old_path": f} for f in fs}
    exp = [
        ("map: one module", ch("src/cache/ckpt_store.cpp"), "release_gate.ps1 -Module a"),
        ("map: one module + its when file", ch("src/cache/ckpt_store.h", "include/whirl/prefill_schedule.h"), "release_gate.ps1 -Module a -Steps seq_swift_precise"),
        ("map: contract -> full", ch("src/device/backend.h"), "full gate, both GPUs (contract"),
        ("map: product file in no module -> full", ch("src/cache/ckpt_store.cpp", "src/server/engine.cpp"), "full gate or"),
        ("map: two modules -> full", ch("src/cache/ckpt_store.cpp", "src/server/later.cpp"), "full gate, both GPUs (cross-module"),
        ("map: docs only", ch("docs/dev/testing.md"), "none"),
    ]
    for name, c, want in exp:
        g = impact.module_gate(c, mods)["module_gate"]
        good = g.startswith(want)
        print(f"{'ok  ' if good else 'FAIL'} {name}: {g}")
        ok, bad = ok + good, bad + (not good)
    # a module with a when-file and only its own files: -Steps added
    mods2 = {"a": dict(BASE["modules"]["a"], paths=["src/cache/ckpt_store.*", "include/whirl/prefill_schedule.h"])}
    g = impact.module_gate(ch("src/cache/ckpt_store.cpp", "include/whirl/prefill_schedule.h"), mods2)["module_gate"]
    good = g.startswith("release_gate.ps1 -Module a -Steps seq_swift_precise")
    print(f"{'ok  ' if good else 'FAIL'} map: when-file adds -Steps: {g}")
    ok, bad = ok + good, bad + (not good)
    # review A3: steps the impact map selects beyond the module table are added as -Steps (PLAN c-2)
    rep = {"release_gate_steps": ["cpu_cache", "golden_swift", "srv_balance", "srv_fast", "seq_moe_<mode>", "dfl_<mode> (R9700, Q4_K_M)"]}
    g = impact.module_gate(ch("src/cache/ckpt_store.cpp"), mods, rep=rep)["module_gate"]
    good = g.startswith("release_gate.ps1 -Module a -Steps seq_moe_balance,seq_moe_precise,srv_balance,srv_fast")
    print(f"{'ok  ' if good else 'FAIL'} map: map steps beyond the table are added: {g}")
    ok, bad = ok + good, bad + (not good)
    # review K1: GPU steps added beyond the table carry the both-GPU hint; one-GPU (arch-specific) changes do not
    good = g.endswith("[both GPUs, one after the other: seq_moe_balance,seq_moe_precise,srv_balance,srv_fast]")
    print(f"{'ok  ' if good else 'FAIL'} map: added GPU steps say both GPUs: {g}")
    ok, bad = ok + good, bad + (not good)
    g1 = impact.module_gate(ch("src/cache/ckpt_store.cpp"), mods, rep=dict(rep, gpus=["gfx1151"]))["module_gate"]
    good = "both GPUs" not in g1 and "-Steps" in g1
    print(f"{'ok  ' if good else 'FAIL'} map: an 8060S-only change gets no both-GPU hint: {g1}")
    ok, bad = ok + good, bad + (not good)
    # review A4: callers by includers (rdeps): includer in a module joins, includer in no module = full gate
    mods3 = {"h": {"id": "H", "paths": ["src/device/handles.h"], "callers": "rdeps", "steps": ["cpu_device"]},
             "c": {"id": "C", "paths": ["src/cache/*"], "steps": ["cpu_cache"]},
             "t": {"id": "T", "paths": [], "callers": ["c"], "steps": ["cpu_server"]}}
    for name, inc, want in (("map: rdeps includer in a module joins", ["src/cache/ckpt_store.h", "tests/x.cpp"], "release_gate.ps1 -Module h,c"),
                            ("map: rdeps includer in no module = full", ["src/cache/ckpt_store.h", "src/server/engine.h"], "full gate or"),
                            ("map: rdeps without the map = full", None, "full gate or")):
        mm = None if inc is None else {"includers": {"src/device/handles.h": inc}}
        g = impact.module_gate(ch("src/device/handles.h"), mods3, m=mm)["module_gate"]
        good = g.startswith(want)
        print(f"{'ok  ' if good else 'FAIL'} {name}: {g}")
        ok, bad = ok + good, bad + (not good)
    # review C6: a build-wide file with a module file = full gate
    g = impact.module_gate(ch("src/cache/ckpt_store.cpp", "cmake/server.cmake"), mods)["module_gate"]
    good = g.startswith("full gate or") and "build-wide" in g
    print(f"{'ok  ' if good else 'FAIL'} map: build-wide file = full gate: {g}")
    ok, bad = ok + good, bad + (not good)
    print(f"modules check test: {ok}/{ok + bad} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
