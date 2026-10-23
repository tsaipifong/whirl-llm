# SPDX-License-Identifier: Apache-2.0
"""Block and pass tests of validate.py's pair scoring and of impact.py's kernel-dispatch tests (arch024-S12). CPU, < 1 s.

    python -I tools/impact/validate_test.py        exit 0 = "validate test: N/N ok"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impact  # noqa: E402
import validate  # noqa: E402

ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


# review K3: a touched test that no executable registers is a miss, not a dropped pair
m = {"executables": {"whirl-tests": ["tests/unit/unit_tests.cpp"]}, "tests": {"whirl-tests:testA": {"exe": "whirl-tests"}}}
regs, unp = validate.pair(m, {("tests/unit/unit_tests.cpp", "testA"), ("tests/new/new_tests.cpp", "testB")})
expect("pair: registered test pairs", regs == {"whirl-tests:testA": "whirl-tests"}, regs)
expect("pair: test of an exe missing from the map is unpaired", unp == [("tests/new/new_tests.cpp", "testB")], unp)
rep = {"cpu_tests": {"whirl-tests": ["testA"]}, "kernel_tests": []}
sc = dict(validate.score_pairs(regs, unp, rep))
expect("score: listed test = HIT", sc.get("whirl-tests:testA") == "HIT", sc)
expect("score: unpaired counts as a miss (block)", sc.get("tests/new/new_tests.cpp:testB") == "UNPAIRED" and len(sc) == 2, sc)
sc2 = dict(validate.score_pairs(regs, [], {"cpu_tests": {}, "kernel_tests": []}))
expect("score: not listed = MISS", sc2 == {"whirl-tests:testA": "MISS"}, sc2)

# hold-out root cause (1e18b8f, a5fe163, 65b84f6): a host function that dispatches a kernel lists that kernel's tests
src = "void Model::matmul() {\n  x = 1;\n}\n"
mp = {"functions": {"src/model/forward.cpp::Model::matmul": {"name": "Model::matmul", "paths": ["cli.decode"], "models": ["dense"],
                                                             "tests": []}},
      "hip": {"kernels/gemv_q.hip": {"arch": ["gfx1201"], "models": [], "paths": [], "host_users": ["Model::matmul"],
                                     "tests": ["whirl-kernel-test:testGemv"]}},
      "files": {}, "includers": {}, "executables": {}, "tests": {}, "release_srv_steps": {},
      "declared": {"paths": {"cli.decode": {"metric": "tg"}}, "gate_suites": {}, "metric_cells": {"tg": ["tg"]}, "gpus": {"gfx1201": "r97", "gfx1151": "s8"},
                   "models": {"dense": "swift", "moe": "ornith"}, "modes": ["precise", "balance", "fast"]}}
real_show = impact.show
try:
    impact.show = lambda rev, path: src
    r = impact.analyze({"src/model/forward.cpp": {"new": [2], "old": [2], "old_path": "src/model/forward.cpp"}}, mp, "x", "y")
    expect("dispatch: changed dispatcher lists its kernel's test", "testGemv" in r["kernel_tests"] and "kt_full" in r["release_gate_steps"], r["kernel_tests"])
    mp["hip"]["kernels/gemv_q.hip"]["host_users"] = ["Model::other"]
    r = impact.analyze({"src/model/forward.cpp": {"new": [2], "old": [2], "old_path": "src/model/forward.cpp"}}, mp, "x", "y")
    expect("dispatch: a function that does not dispatch it does not (block)", "testGemv" not in r["kernel_tests"], r["kernel_tests"])
finally:
    impact.show = real_show
print(f"validate test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
