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
# measure-tools P1 + review A4: a narrowed commit passes only when EVERY lost GPU step is shown unneeded by a named rule:
# the step comes back with the legacy identifier rule (R-decl) and every identifier the new rule dropped is a type /
# parameter / initializer reference / qualifier, or no identifier was dropped (R-member: member-access precision), or
# the step comes back with the moved rule off (R-moved: identical statement in another file). Ground truth: never lose
# a needed step.
gt = {"aaaaaaa": {"golden_swift", "srv_cache_balance"}}
nar = [("aaaaaaa", {"steps": ["srv_cache_balance"], "paths": []}),      # ground truth loses a needed step -> fail
       ("bbbbbbb", {"steps": ["kt_full"], "paths": ["cli.decode"]}),     # legacy ids restore it, dropped = types -> pass
       ("ccccccc", {"steps": ["srv_fast"], "paths": []}),                # no variant restores it -> unexplained -> fail
       ("ddddddd", {"steps": ["srv_balance"], "paths": []}),             # restored, but a dropped id of kind 'other' -> fail
       ("eeeeeee", {"steps": ["golden_moe"], "paths": []}),              # restored with the moved rule off -> pass
       ("fffffff", {"steps": ["seq_swift_<mode>"], "paths": []}),        # restored, nothing dropped -> R-member -> pass
       ("ggggggg", {"steps": ["cpu_tests"], "paths": ["load"]})]         # no GPU step lost -> pass (CPU / path only)
attrib = {"aaaaaaa": {"ids": ["srv_cache_balance"], "moved": [], "dropped": {"DevBuf": "type"}},
          "bbbbbbb": {"ids": ["kt_full"], "moved": [], "dropped": {"DevBuf": "type", "n": "param"}},
          "ccccccc": {"ids": [], "moved": [], "dropped": {}},
          "ddddddd": {"ids": ["srv_balance"], "moved": [], "dropped": {"kFoo": "other", "Bar": "type"}},
          "eeeeeee": {"ids": [], "moved": ["golden_moe"], "dropped": {}},
          "fffffff": {"ids": ["seq_swift_balance", "seq_swift_precise", "seq_swift_fast"], "moved": [], "dropped": {}},
          "ggggggg": {"ids": [], "moved": [], "dropped": {}}}
w = {k: {"decl": ["x"]} for k in attrib}
fails, lines, expl, gpu_expl = validate.classify_narrowed(nar, w, gt, attrib)
expect("narrowed: ground truth losing a needed step fails (block)", "aaaaaaa" in fails, lines)
expect("narrowed: lost step no variant restores = unexplained (block)", "ccccccc" in fails, lines)
expect("narrowed: a dropped identifier of kind 'other' is not a named rule (block)", "ddddddd" in fails, lines)
expect("narrowed: R-decl with only type / parameter ids dropped (pass)", "bbbbbbb" in expl, lines)
expect("narrowed: R-moved (pass)", "eeeeeee" in expl, lines)
expect("narrowed: R-member, nothing dropped (pass)", "fffffff" in expl, lines)
expect("narrowed: CPU-only loss needs no rule (pass)", "ggggggg" in expl and gpu_expl == 3, (expl, gpu_expl))
fails2, lines2, _, _ = validate.classify_narrowed([("bbbbbbb", {"steps": ["kt_full"], "paths": []})], w, {}, None)
expect("narrowed: no attribution available = fail (block)", fails2 == ["bbbbbbb"], lines2)
# mt-p1 pass 4 (REVIEW3 act-on 2, G3 kGemmsCfgs row): restored by the 0.2.4 identifier rule with nothing dropped, but
# the changed line had no identifier (an initializer row): 0.2.4 reached it by its file-scope fallback, not by a
# same-named local, so it is no evidence of member-access precision -> R-init-noid, failing
fn, ln, en, _ = validate.classify_narrowed([("iiiiiii", {"steps": ["golden_moe", "srv_balance"], "paths": []})], {"iiiiiii": {"decl": []}}, {},
                                           {"iiiiiii": {"ids": ["golden_moe", "srv_balance"], "moved": [], "both": [], "braces": [],
                                                        "dropped": {}, "noid": ["include/whirl/kernels_abi.h:453"]}})
expect("narrowed: identifier-less initializer row restored only by legacy ids = R-init-noid, failing (block)",
       fn == ["iiiiiii"] and any("R-init-noid" in x for x in ln), ln)
# the identifier categories of a declaration line (what the new rule dropped and why)
cats = validate.impact.id_categories("    std::vector<device::DevBuf> allocations = makeBufs(kNum);", ["allocations"])
expect("categories: DevBuf type, device qualifier, makeBufs / kN initializer",
       cats.get("DevBuf") == "type" and cats.get("device") == "qualifier" and cats.get("makeBufs") == "init" and cats.get("kNum") == "init", cats)
cats = validate.impact.id_categories("    bool restoreIsShared(std::uint8_t ck_kind, std::uint32_t ck_pos);", ["restoreIsShared"])
expect("categories: parameter names", cats.get("ck_kind") == "param" and cats.get("ck_pos") == "param", cats)
cats = validate.impact.id_categories("        const std::wstring& path, const char* driver_advice);", ["messageMissingSymbol"], in_function=True)
expect("categories: continuation line of a function declaration = parameters (7d8c0bf)", cats == {"wstring": "param", "path": "param",
       "driver_advice": "param"}, cats)
fk, lk, ek, _ = validate.classify_narrowed([("hhhhhhh", {"steps": ["golden_moe"], "paths": []})], {"hhhhhhh": {"decl": []}}, {},
                                           {"hhhhhhh": {"ids": [], "moved": [], "both": [], "braces": ["golden_moe"], "dropped": {}}})
expect("narrowed: R-block (a namespace block added whole, a4db20b) passes", ek == ["hhhhhhh"] and not fk, lk)
cats = validate.impact.id_categories("namespace whirl::device::detail {", [])
expect("categories: namespace names on a namespace line are qualifiers", cats == {"device": "qualifier", "detail": "qualifier"}, cats)
cats = validate.impact.id_categories("    int fresh_count = 0;", [])
expect("categories: an undeclared identifier in declarator position is 'other' (block)", cats.get("fresh_count") == "other", cats)
# covered(): release steps with <mode>, golden, checks, P-size via the sizing class
rep = {"release_gate_steps": ["seq_swift_<mode>", "golden_moe"], "outside_gate_checks": ["kvram719: tools/repro/kvram719_tier.ps1"],
       "behaviour": {"sizing": ["path startup"]}, "server_gate_suites": [], "golden": [], "extra": {}, "cpu_tests": {}, "kernel_tests": []}
expect("covered: seq_swift_<mode> expands to seq_swift_balance", validate.covered(rep, "release:seq_swift_balance"))
expect("covered: golden = either golden_<model>", validate.covered(rep, "golden"))
expect("covered: named check", validate.covered(rep, "check:kvram719") and not validate.covered(rep, "check:E2"))
expect("covered: P-size = sizing class", validate.covered(rep, "check:P-size"))
expect("covered: missing step (block)", not validate.covered(rep, "release:srv_gpu_select"))
# pass 4: run_baseline replays one at a time (never Popen: WinError 1450 / 1455) and reuses finished replay.py --out files
import json as _json  # noqa: E402
import tempfile as _tf  # noqa: E402
_td = _tf.mkdtemp(prefix="vt_base_")
_row = {"steps": ["golden_moe"], "paths": ["cli.decode"], "cpu": [], "kt": [], "decl": []}
_bj, _nj = os.path.join(_td, "b.json"), os.path.join(_td, "n.json")
_json.dump({"aaaaaaa": _row}, open(_bj, "w", encoding="utf-8"))
_json.dump({"aaaaaaa": dict(_row, cpu=["whirl-tests:t"])}, open(_nj, "w", encoding="utf-8"))
_calls = []


def _no_popen(*x, **k):
    raise AssertionError("Popen used")


def _fake_run(cmd, *x, **k):
    _calls.append(cmd)
    out = cmd[cmd.index("--out") + 1]
    _json.dump({"aaaaaaa": _row}, open(out, "w", encoding="utf-8"))
    return type("R", (), {"returncode": 0, "stdout": b""})()


_sv = (validate.subprocess.Popen, validate.subprocess.run, validate.attribute)
validate.subprocess.Popen, validate.subprocess.run, validate.attribute = _no_popen, _fake_run, lambda nar, rev: {}
try:
    r1 = validate.run_baseline("0e35407", 5, "HEAD", 0, base_json=_bj, new_json=_nj)
    expect("baseline: both jsons given = no replay, 0 failing", r1 == 0 and not _calls, (r1, _calls))
    r2 = validate.run_baseline("0e35407", 5, "HEAD", 0, base_json=_bj)
    expect("baseline: base json given = ONE replay (these tools), sequential", r2 == 0 and len(_calls) == 1
           and _calls[0][_calls[0].index("--tools") + 1] == os.path.dirname(os.path.abspath(validate.__file__)), _calls)
    _json.dump({"bbbbbbb": _row}, open(_nj, "w", encoding="utf-8"))
    r3 = validate.run_baseline("0e35407", 5, "HEAD", 0, base_json=_bj, new_json=_nj)
    expect("baseline: replays of different commits are refused (block)", r3 is None, r3)
finally:
    validate.subprocess.Popen, validate.subprocess.run, validate.attribute = _sv
    validate.shutil.rmtree(_td, ignore_errors=True)
# mt-p1 pass 4b: pass-to-pass LESS (hold 4fae76e / c3601d4 / 7078143): only paths lost, the earlier pass's file-scope
# fallback of src/cli/main.cpp ("no user function") resolved -> R-fallback-resolved; any other LESS fails
_fb = "src/cli/main.cpp: declared WHIRL_HELP_MODES has no user function in the map: file-scope fallback over 1 file(s)"
_r = lambda steps=(), paths=(), cpu=(), defects=(): {"steps": list(steps), "paths": list(paths), "cpu": list(cpu), "kt": [],
                                                    "decl": [], "defects": list(defects)}
_mp = {"files": {"src/cli/main.cpp": ["cli.decode", "cli.mtp"], "src/a.h": ["load"]}, "includers": {"src/a.h": ["src/cli/main.cpp"]}}
_b = {"aaaaaaa": _r(paths=["cli.decode", "cli.mtp", "load"], defects=[_fb]),
      "bbbbbbb": _r(steps=["golden_swift"], paths=["cli.decode", "load"], defects=[_fb]),
      "ccccccc": _r(paths=["cli.decode", "load"]),
      "ddddddd": _r(paths=["cli.decode", "server.decode"], defects=[_fb]),
      "eeeeeee": _r(paths=["load"], cpu=["whirl-tests:t1"]),
      "fffffff": _r(paths=["load"])}
_w = {"aaaaaaa": _r(paths=["load"]),
      "bbbbbbb": _r(paths=["load"]),
      "ccccccc": _r(paths=["load"]),
      "ddddddd": _r(paths=[]),
      "eeeeeee": _r(paths=["load"]),
      "fffffff": _r(paths=["load", "startup"])}
_f, _l, _e = validate.compare_passes(_b, _w, _mp)
expect("passes: resolved file-scope fallback, only that file's paths lost = R-fallback-resolved (pass)", _e == ["aaaaaaa"], _l)
expect("passes: a lost release step fails even with the fallback resolved (block)", "bbbbbbb" in _f, _l)
expect("passes: paths lost with no resolved fallback fail (block)", "ccccccc" in _f, _l)
expect("passes: a lost path outside the resolved file's paths fails (block)", "ddddddd" in _f, _l)
expect("passes: a lost CPU test fails (block)", "eeeeeee" in _f, _l)
expect("passes: more is listed, not failing", "fffffff" not in _f and any(x.startswith("   more fffffff") for x in _l), _l)
_w2 = dict(_w, aaaaaaa=_r(paths=["load"], defects=[_fb]))
_f2, _l2, _e2 = validate.compare_passes(_b, _w2, _mp)
expect("passes: fallback still reported by this pass = not resolved (block)", "aaaaaaa" in _f2, _l2)
print(f"validate test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
