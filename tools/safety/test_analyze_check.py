#!/usr/bin/env python
"""safe025: tests of the /analyze baseline gate (analyze_check.py). CPU only, no compiler: synthetic build directories with
hand-written SARIF files. Run: python tools/safety/test_analyze_check.py   (exit 0 = all pass)

Review A2: the keys must not depend on the tree path, the line, or the message language.
Review A3: an incomplete run (TU not recompiled, failed build, no prepare) must FAIL, not PASS.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_check as ac  # noqa: E402


def sarif(src_root, results):
    """results: list of (relpath, ruleId, function, line, message)"""
    arts, res = [], []
    for rel, rule, fn, line, msg in results:
        uri = "file:///" + os.path.join(src_root, rel).replace("\\", "/").lower()
        if uri not in arts:
            arts.append(uri)
        res.append({"ruleId": rule, "message": {"text": msg},
                    "locations": [{"physicalLocation": {"artifactLocation": {"index": arts.index(uri)}, "region": {"startLine": line, "startColumn": 5}},
                                   "logicalLocations": [{"fullyQualifiedName": fn, "kind": "function"}]}]})
    return {"version": "2.1.0", "runs": [{"results": res, "artifacts": [{"location": {"uri": u}} for u in arts]}]}


class Tree:
    """A source tree with a build directory laid out like run_analyze.bat leaves it."""

    def __init__(self, root, tus, rc=0, line_shift=0, lang="en"):
        self.src = os.path.join(root, "src_tree")
        self.b = os.path.join(self.src, "build-analyze")
        os.makedirs(self.b, exist_ok=True)
        with open(os.path.join(self.b, "CMakeCache.txt"), "w") as f:
            f.write(f"CMAKE_HOME_DIRECTORY:INTERNAL={self.src}\n")
        with open(os.path.join(self.b, ac.START), "w") as f:
            f.write("targets t\n")
        t0 = os.path.getmtime(os.path.join(self.b, ac.START))
        objs = []
        for obj, results in tus.items():
            fo = os.path.join(self.b, obj)
            os.makedirs(os.path.dirname(fo), exist_ok=True)
            open(fo, "w").close()
            msgs = [(r, c, fn, ln + line_shift, ("使用的記憶體 'x' 尚未初始化" if lang == "zh" else "Using uninitialized memory 'x'") + f" '{ln * 7}'")
                    for r, c, fn, ln in results]
            with open(ac.sarif_of(fo), "w", encoding="utf-8") as f:
                json.dump(sarif(self.src, msgs), f, ensure_ascii=False)
            for p in (fo, ac.sarif_of(fo)):
                os.utime(p, (t0 + 5, t0 + 5))
            objs.append(obj)
        with open(os.path.join(self.b, ac.EXPECTED), "w") as f:
            f.write("\n".join(objs) + "\n")
        with open(os.path.join(self.b, "DONE"), "w") as f:
            f.write(f"rc={rc}\n")


TUS = {
    "CMakeFiles/a.dir/src/a.cpp.obj": [("src/a.cpp", "C6001", "whirl::a", 10), ("include/whirl/json.h", "C26495", "whirl::json::Value::Value", 30)],
    "CMakeFiles/b.dir/src/b.cpp.obj": [("src/b.cpp", "C6262", "whirl::b::big", 200), ("include/whirl/json.h", "C26495", "whirl::json::Value::Value", 30)],
}


def run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ac.main(["analyze_check.py", *argv])
    return rc, buf.getvalue()


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="anchk_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def baseline(self, tree):
        bl = os.path.join(self.tmp, "baseline.txt")
        rc, out = run("update", tree.b, bl)
        self.assertEqual(rc, 0, out)
        return bl

    def test_complete_run_passes(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        rc, out = run("check", t.b, bl)
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS", out)
        # the header warning seen by two TUs at the same place counts once
        self.assertIn("1\tinclude/whirl/json.h|C26495|whirl::json::Value::Value", open(bl, encoding="utf-8").read())

    def test_other_tree_path_lines_language_pass(self):  # A2
        t1 = Tree(os.path.join(self.tmp, "first"), TUS)
        bl = self.baseline(t1)
        t2 = Tree(os.path.join(self.tmp, "second copy", "deeper"), TUS, line_shift=37, lang="zh")
        rc, out = run("check", t2.b, bl)
        self.assertEqual(rc, 0, out)

    def test_new_warning_fails(self):
        t1 = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t1)
        more = dict(TUS)
        more["CMakeFiles/a.dir/src/a.cpp.obj"] = TUS["CMakeFiles/a.dir/src/a.cpp.obj"] + [("src/a.cpp", "C6011", "whirl::a", 12)]
        t2 = Tree(os.path.join(self.tmp, "t2"), more)
        rc, out = run("check", t2.b, bl)
        self.assertEqual(rc, 1, out)
        self.assertIn("NEW x1: src/a.cpp|C6011|whirl::a", out)

    def test_second_same_key_in_function_fails(self):
        t1 = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t1)
        more = dict(TUS)
        more["CMakeFiles/a.dir/src/a.cpp.obj"] = TUS["CMakeFiles/a.dir/src/a.cpp.obj"] + [("src/a.cpp", "C6001", "whirl::a", 14)]
        rc, out = run("check", Tree(os.path.join(self.tmp, "t2"), more).b, bl)
        self.assertEqual(rc, 1, out)

    def test_tu_not_recompiled_fails(self):  # A3: incremental build
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        old = time.time() - 3600
        for p in ("CMakeFiles/b.dir/src/b.cpp.obj",):
            fo = os.path.join(t.b, p)
            os.utime(fo, (old, old))
            os.utime(ac.sarif_of(fo), (old, old))
        rc, out = run("check", t.b, bl)
        self.assertEqual(rc, 1, out)
        self.assertIn("INCOMPLETE", out)

    def test_tu_missing_fails(self):  # A3: partial build
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        os.remove(os.path.join(t.b, "CMakeFiles/b.dir/src/b.cpp.obj"))
        rc, out = run("check", t.b, bl)
        self.assertEqual(rc, 1, out)

    def test_failed_build_fails(self):  # A3: build error
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        t_bad = Tree(os.path.join(self.tmp, "t2"), TUS, rc=1)
        rc, out = run("check", t_bad.b, bl)
        self.assertEqual(rc, 1, out)
        self.assertIn("build failed", out)

    def test_done_missing_or_empty_fails(self):  # the old "echo 0>" wrote an empty DONE
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        open(os.path.join(t.b, "DONE"), "w").close()
        rc, out = run("check", t.b, bl)
        self.assertEqual(rc, 1, out)

    def test_no_prepare_fails(self):  # A3: plain ninja run
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        os.remove(os.path.join(t.b, ac.START))
        rc, out = run("check", t.b, bl)
        self.assertEqual(rc, 1, out)

    def test_update_refuses_incomplete(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS, rc=2)
        rc, out = run("update", t.b, os.path.join(self.tmp, "bl.txt"))
        self.assertEqual(rc, 1, out)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "bl.txt")))

    def test_fixed_warning_still_passes_and_is_listed(self):
        t1 = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t1)
        fewer = dict(TUS)
        fewer["CMakeFiles/b.dir/src/b.cpp.obj"] = [("include/whirl/json.h", "C26495", "whirl::json::Value::Value", 30)]
        rc, out = run("check", Tree(os.path.join(self.tmp, "t2"), fewer).b, bl)
        self.assertEqual(rc, 0, out)
        self.assertIn("fixed since baseline", out)

    def test_sdk_header_ignored(self):
        t1 = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t1)
        t2 = Tree(os.path.join(self.tmp, "t2"), TUS)
        # a warning in a file outside the source tree (SDK / STL) is not ours
        fo = os.path.join(t2.b, "CMakeFiles/a.dir/src/a.cpp.obj")
        d = json.load(open(ac.sarif_of(fo), encoding="utf-8"))
        d["runs"][0]["artifacts"].append({"location": {"uri": "file:///c:/program%20files/windows%20kits/10/include/x.h"}})
        d["runs"][0]["results"].append({"ruleId": "C6001", "locations": [{"physicalLocation": {"artifactLocation": {"index": len(d["runs"][0]["artifacts"]) - 1}, "region": {"startLine": 1}}}]})
        json.dump(d, open(ac.sarif_of(fo), "w", encoding="utf-8"))
        t0 = os.path.getmtime(os.path.join(t2.b, ac.START))
        os.utime(ac.sarif_of(fo), (t0 + 5, t0 + 5))
        rc, out = run("check", t2.b, bl)
        self.assertEqual(rc, 0, out)

    def test_analyze_off_tu_not_expected(self):  # arch025-Q3: kernel byte arrays get /analyze- (C1060 out of heap)
        self.assertTrue(ac.analyze_on('cl.exe /c /analyze /analyze:external- /analyze:log:format:sarif /Fo"a.obj" a.cpp'))
        self.assertFalse(ac.analyze_on('cl.exe /c /analyze /analyze:external- /analyze:log:format:sarif /analyze- /Fo"k.obj" k.cpp'))
        self.assertFalse(ac.analyze_on('cl.exe /c /analyze:external- /Fo"b.obj" b.cpp'))

    # arch025-Q3: the whitelist (file:line + reason per baseline key)
    def whitelist(self, rows):
        p = os.path.join(self.tmp, "wl.tsv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("# header\n" + "".join("\t".join(r) + "\n" for r in rows))
        return p

    GOOD = [("include/whirl/json.h|C26495|whirl::json::Value::Value", "include/whirl/json.h:30", "member left uninitialized by design (test)"),
            ("src/a.cpp|C6001|whirl::a", "src/a.cpp:10", "false positive: x is written on every path (test)"),
            ("src/b.cpp|C6262|whirl::b::big", "src/b.cpp:200", "large stack frame, main thread only (test)")]

    def test_reasons_complete_pass(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD), t.b)
        self.assertEqual(rc, 0, out)
        self.assertIn("analyze reasons: ok", out)

    def test_reasons_missing_key_fails(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD[:2]))
        self.assertEqual(rc, 1, out)
        self.assertIn("without a whitelist reason: src/b.cpp|C6262", out)

    def test_reasons_stale_row_and_empty_reason_fail(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        rows = self.GOOD[:2] + [("src/b.cpp|C6262|whirl::b::big", "src/b.cpp:200", "TODO"), ("src/c.cpp|C6001|c", "src/c.cpp:1", "gone from the baseline")]
        rc, out = run("reasons", bl, self.whitelist(rows))
        self.assertEqual(rc, 1, out)
        self.assertIn("no reason for src/b.cpp|C6262", out)
        self.assertIn("not in the baseline (stale): src/c.cpp|C6001|c", out)

    def test_reasons_line_drift_is_info_not_fail(self):
        # arch025-Q3: unrelated code above moved every warning 69 lines down (server_tests.cpp 7009 -> 7078 on arch025-next):
        # same file|rule|function and count -> PASS, the drift is printed as INFO
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        t2 = Tree(os.path.join(self.tmp, "t2"), TUS, line_shift=69)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD), t2.b)
        self.assertEqual(rc, 0, out)
        self.assertIn("INFO line drift: src/a.cpp|C6001|whirl::a: whitelist line 10, cl reports line(s) 79 now", out)
        self.assertIn("analyze reasons: ok", out)
        self.assertNotIn("FAIL", out)

    def test_reasons_other_function_fails(self):
        # the same warning now reported in another function = another key: no reason for it -> FAIL
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        moved = dict(TUS)
        moved["CMakeFiles/a.dir/src/a.cpp.obj"] = [("src/a.cpp", "C6001", "whirl::a2", 10), TUS["CMakeFiles/a.dir/src/a.cpp.obj"][1]]
        t2 = Tree(os.path.join(self.tmp, "t2"), moved)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD), t2.b)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL src/a.cpp|C6001|whirl::a2: cl reports it 1 time(s) and the whitelist has no reason", out)

    def test_reasons_extra_count_fails(self):
        # a second occurrence of a whitelisted key (other line, same function) has no reason -> FAIL, even with drifted lines
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        more = dict(TUS)
        more["CMakeFiles/a.dir/src/a.cpp.obj"] = TUS["CMakeFiles/a.dir/src/a.cpp.obj"] + [("src/a.cpp", "C6001", "whirl::a", 25)]
        t2 = Tree(os.path.join(self.tmp, "t2"), more, line_shift=4)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD), t2.b)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL src/a.cpp|C6001|whirl::a: cl reports it 2 time(s), the baseline accepts 1", out)

    def test_reasons_wrong_file_fails(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        rows = [self.GOOD[0], ("src/a.cpp|C6001|whirl::a", "src/other.cpp:10", "points into another file (test)"), self.GOOD[2]]
        rc, out = run("reasons", bl, self.whitelist(rows))
        self.assertEqual(rc, 1, out)
        self.assertIn("is not in the key's file src/a.cpp", out)

    def test_reasons_incomplete_build_fails(self):
        t = Tree(os.path.join(self.tmp, "t1"), TUS)
        bl = self.baseline(t)
        t2 = Tree(os.path.join(self.tmp, "t2"), TUS, rc=2)
        rc, out = run("reasons", bl, self.whitelist(self.GOOD), t2.b)
        self.assertEqual(rc, 1, out)
        self.assertIn("build dir incomplete", out)


class FaultProofTest(unittest.TestCase):
    """arch025-Q3: analyze_check.py faultproof (verdict of analyze_fault.bat). Synthetic case dirs written like the bat leaves them;
    the real-compiler run is in cpu_safety_selftest / cpu_analyze."""

    F = ac.FAULT_FILE
    ACC = (F, "C6001", "acceptedOld", 14)
    PLANTED = [(F, "C6001", "usesUninitialized", 29), (F, "C6011", "derefsNull", 35), (F, "C6201", "overrun", 42)]
    SAME = (F, "C6001", "acceptedOld", 19)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="afp_")
        self.bl = os.path.join(self.tmp, "bl.txt")
        with open(self.bl, "w", encoding="utf-8") as f:
            f.write(f"1\t{ac.FAULT_ACCEPTED}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def case(self, name, results, done_rc=0, run_gate=True):
        """One case dir: the Tree layout (CMakeCache, start, expected, SARIF), then the real 'check' writes gate.log + GATE."""
        t = Tree(os.path.join(self.tmp, "src_" + name), {"af.cpp.obj": results}, rc=done_rc)
        d = os.path.join(self.tmp, "proof", name)
        shutil.copytree(t.b, d)
        with open(os.path.join(d, "CMakeCache.txt"), "w") as f:
            f.write(f"CMAKE_HOME_DIRECTORY:INTERNAL={t.src}\n")
        if run_gate:
            rc, out = run("check", d, self.bl)
            with open(os.path.join(d, "gate.log"), "w", encoding="utf-8") as f:
                f.write(out)
            with open(os.path.join(d, "GATE"), "w") as f:
                f.write(f"rc={rc}\n")

    def proof(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ac.faultproof(os.path.join(self.tmp, "proof"))
        return rc, buf.getvalue()

    def good(self, skip=None):
        cases = {"clean": [self.ACC], "new": [self.ACC] + self.PLANTED, "samekey": [self.ACC, self.SAME]}
        for k, v in cases.items():
            if k != skip:
                self.case(k, v)

    def test_all_three_cases_ok(self):
        self.good()
        rc, out = self.proof()
        self.assertEqual(rc, 0, out)
        self.assertIn("analyze fault proof: ok", out)

    def test_samekey_not_caught_fails(self):
        self.good(skip="samekey")
        self.case("samekey", [self.ACC])  # cl reported the key only once: the gate PASSES, so the proof must FAIL
        rc, out = self.proof()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL samekey:", out)

    def test_clean_without_accepted_warning_fails(self):
        self.good(skip="clean")
        self.case("clean", [])  # PASS, but with no warning at all it proves nothing about the baseline
        rc, out = self.proof()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL clean: the accepted warning", out)

    def test_clean_with_extra_warning_fails(self):
        self.good(skip="clean")
        self.case("clean", [self.ACC, self.PLANTED[0]])
        rc, out = self.proof()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL clean: the baseline-only TU must PASS", out)

    def test_new_with_too_few_planted_fails(self):
        self.good(skip="new")
        self.case("new", [self.ACC] + self.PLANTED[:2])
        rc, out = self.proof()
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL new:", out)

    def test_compile_failed_or_gate_missing_fails(self):
        self.good(skip="new")
        self.case("new", [self.ACC] + self.PLANTED, done_rc=2)
        rc, out = self.proof()
        self.assertEqual(rc, 1, out)
        self.assertIn("new: compile DONE rc=2", out)
        shutil.rmtree(os.path.join(self.tmp, "proof", "samekey"))
        self.case("samekey", [self.ACC, self.SAME], run_gate=False)
        rc, out = self.proof()
        self.assertIn("samekey: gate.log missing or empty", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
