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
    ("block: paths missing, no lands_on", mut(lambda m: m["later"].pop("lands_on")), False, "no lands_on"),
    ("block: file in two modules", mut(lambda m: m.update(b={"id": "MC", "paths": ["src/cache/*"], "steps": ["cpu_cache"]})), False, "two modules"),
    ("block: no steps, not full", mut(lambda m: m["a"].update(steps=[])), False, "no steps"),
    ("block: no id", mut(lambda m: m["a"].pop("id")), False, "no id"),
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
    good = g == "release_gate.ps1 -Module a -Steps seq_swift_precise"
    print(f"{'ok  ' if good else 'FAIL'} map: when-file adds -Steps: {g}")
    ok, bad = ok + good, bad + (not good)
    print(f"modules check test: {ok}/{ok + bad} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
