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
    "x": {"id": "X", "paths": ["src/device/backend.h"], "lands_on": "n/a", "full": True}},
    # arch025-Q7: standards (kinds / classes / per-module metric keys + proof kinds; no numbers)
    "standards": {
        "a": {"metrics": ["host.ckpt.save"], "rel": "mt-p3 entry (numbers allowed outside the Q7 parts)"},
        "kinds": {"exit_rule": {"compare": "rule", "bitwise": False}, "golden_exact": {"compare": "exact", "bitwise": True},
                  "work_trace": {"compare": "exact", "bitwise": True}, "stage_time": {"compare": "distribution", "bitwise": False},
                  "task_accuracy": {"compare": "task_accuracy", "bitwise": False}, "kl_guard": {"compare": "diagnostic", "bitwise": False}},
        "classes": {"refactor": {"modes": ["precise", "balance", "fast"], "allow_bitwise": True, "requires_bitwise": True},
                    "precise": {"modes": ["precise"], "allow_bitwise": True, "requires": ["golden_exact"]},
                    "balance": {"modes": ["balance"], "allow_bitwise": False, "requires": ["task_accuracy"]},
                    "fast": {"modes": ["fast"], "allow_bitwise": False, "requires": ["task_accuracy"]}},
        "modules": {
            "a": {"metrics": ["host.ckpt.save", "stage.layer.attn"],
                  "proof": {"refactor": ["exit_rule", "golden_exact", "work_trace", "stage_time"],
                            "precise": ["golden_exact", "stage_time"],
                            "balance": ["task_accuracy", "kl_guard", "stage_time"],
                            "fast": ["task_accuracy", "kl_guard", "stage_time"]}},
            "later": {"metrics": [], "metrics_none": "no bench key", "proof": {"refactor": ["exit_rule", "golden_exact"]}},
            "x": {"metrics": ["stage.decode_step"], "proof": {"refactor": ["golden_exact", "work_trace"]}},
            "plat.*": {"metrics": [], "metrics_none": "family default", "proof": {"refactor": ["exit_rule", "golden_exact"]}}}}}


def mut(f):
    d = copy.deepcopy(BASE)
    f(d["modules"])
    return d


def smut(f):
    """mutate the standards section (arch025-Q7 cases)"""
    d = copy.deepcopy(BASE)
    f(d["standards"], d["modules"])
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
    ("block: callers names an unknown module", mut(lambda m: m["a"].update(callers=["nope"])), False, "callers must be"),
    # arch025-Q7 standards: block + pass
    ("pass: complete standards (refactor / precise / balance / fast, glob entry declared early)", BASE, True, None),
    ("block: no standards section", dict(BASE, standards={}), False, "needs kinds, classes and modules"),
    ("block: module without a standards entry", smut(lambda s, m: s["modules"].pop("later")), False, "later: no standards entry"),
    ("block: new module without a standards entry", smut(lambda s, m: m.update(n={"id": "MN", "paths": [], "steps": ["cpu_server"]})), False, "n: no standards entry"),
    ("pass: glob entry covers a module (platform.* style)", smut(lambda s, m: m.update({"plat.fs": {"id": "PL2", "paths": [], "steps": ["cpu_server"]}})), True, None),
    ("block: entry without a refactor proof", smut(lambda s, m: s["modules"]["later"]["proof"].pop("refactor")), False, "no refactor proof"),
    ("block: unknown kind", smut(lambda s, m: s["modules"]["a"]["proof"]["refactor"].append("vibes")), False, "unknown kind vibes"),
    ("block: unknown class", smut(lambda s, m: s["modules"]["a"]["proof"].update(turbo=["exit_rule"])), False, "unknown class turbo"),
    ("block: balance proved bitwise (golden_exact)", smut(lambda s, m: s["modules"]["a"]["proof"]["balance"].append("golden_exact")), False, "class balance lists bitwise"),
    ("block: fast proved by launch-table equality", smut(lambda s, m: s["modules"]["a"]["proof"]["fast"].append("work_trace")), False, "class fast lists bitwise"),
    ("block: balance class allows bitwise", smut(lambda s, m: s["classes"]["balance"].update(allow_bitwise=True)), False, "balance: allow_bitwise must be false"),
    ("block: a non-precise class allows bitwise", smut(lambda s, m: s["classes"].update(speedy={"modes": ["balance"], "allow_bitwise": True})), False, "only refactor or a precise-only"),
    ("block: class fast missing", smut(lambda s, m: (s["classes"].pop("fast"), s["modules"]["a"]["proof"].pop("fast"))), False, "class fast missing"),
    ("block: refactor without a bitwise kind", smut(lambda s, m: s["modules"]["later"]["proof"].update(refactor=["exit_rule"])), False, "lists no bitwise kind"),
    ("block: balance without task_accuracy", smut(lambda s, m: s["modules"]["a"]["proof"].update(balance=["kl_guard", "stage_time"])), False, "must list kind task_accuracy"),
    ("block: precise without golden_exact", smut(lambda s, m: s["modules"]["a"]["proof"].update(precise=["stage_time"])), False, "must list kind golden_exact"),
    ("block: numeric threshold in a Q7 field", smut(lambda s, m: s["modules"]["a"].update(note="time within 5% of base")), False, "numeric threshold"),
    ("block: numeric bound in a kind", smut(lambda s, m: s["kinds"]["stage_time"].update(desc="median <= 3 MAD")), False, "numeric threshold"),
    ("block: bad metric key", smut(lambda s, m: s["modules"]["a"]["metrics"].append("decode tok/s")), False, "is not a host.*"),
    ("block: no metrics and no reason", smut(lambda s, m: s["modules"]["later"].pop("metrics_none")), False, "no metrics and no metrics_none"),
    ("block: mt-p3 metric missing from the module standard", smut(lambda s, m: s["modules"]["a"].update(metrics=["stage.layer.attn"])), False, "mt-p3 metric host.ckpt.save"),
    ("block: kind without bitwise flag", smut(lambda s, m: s["kinds"]["stage_time"].pop("bitwise")), False, "kind stage_time: needs compare and bitwise"),
    ("block: class with an unknown mode", smut(lambda s, m: s["classes"]["fast"].update(modes=["turbo"])), False, "modes must be"),
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
