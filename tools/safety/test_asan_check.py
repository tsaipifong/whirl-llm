#!/usr/bin/env python
"""arch025-Q1: block / pass tests of asan_check.py (the cpu_asan verdict). CPU only, no compiler: synthetic build-asan logs and a
synthetic source tree. Run: python tools/safety/test_asan_check.py   (exit 0 = all pass)
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asan_check as ak  # noqa: E402

SUMMARY = {
    "whirl-device-tests": "LEAKCHECK: 19 NullBackend instances checked, 0 leaked\ndevice tests: 178 passed, 0 failed\n",
    "whirl-cache-tests": "cache tests: 158 passed, 0 failed\n",
    "whirl-tier-tests": "tier tests: 1746 passed, 0 failed\nLEAKCHECK: 6 NullBackend instances checked, 0 leaked\n",
    "whirl-tests": "unit tests: 6185 passed, 0 failed\n",
    "whirl-model-tests": "model tests: 40 passed, 0 failed\n",
    "whirl-ops-tests": "ops tests: 300 passed, 0 failed\n",
    "whirl-vision-tests": "vision tests: 20 passed, 0 failed\n",
    "whirl-server-tests": "server tests: 15633 passed, 0 failed\n",
    "pins": "SAME tokens_swift (10 lines)\npins: ok (outputs in x)\n",
}
FAULT_LOG = {
    "clean": "LEAKCHECK: 1 NullBackend instances checked, 0 leaked\n",
    "missing_free": "LEAKCHECK: NullBackend destroyed with 1 allocations\nLEAKCHECK: 1 NullBackend instances checked, 1 leaked\n",
    "missing_host": "LEAKCHECK: 1 NullBackend instances checked, 1 leaked\n",
    "missing_stream": "LEAKCHECK: 1 NullBackend instances checked, 1 leaked\n",
    "uaf": "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n",
    "uaf_vector": "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n",
    "overflow": "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n",
    "double_free": "==1==ERROR: AddressSanitizer: attempting double-free on 0x1 in thread T0:\n",
}
EXIT = {"clean": 0, "missing_free": 97, "missing_host": 97, "missing_stream": 97}


class Run:
    def __init__(self, root):
        self.b = os.path.join(root, "build-asan")
        self.L = os.path.join(self.b, "logs")
        self.src = os.path.join(root, "src_tree")
        os.makedirs(self.L)
        os.makedirs(os.path.join(self.src, "src", "a"))
        with open(os.path.join(self.b, "RUN_START"), "w") as f:
            f.write("start\n")
        t0 = os.path.getmtime(os.path.join(self.b, "RUN_START"))
        os.utime(os.path.join(self.b, "RUN_START"), (t0 - 60, t0 - 60))
        self.write_src("src/a/x.cpp", "int f() { return 1; }\n")
        self.write_src("tools/safety/asan_allow.txt", "# header\n")
        steps = ["[1] build start", "[2] build exit 0 expect 0 ok start=1"]
        for n, t in SUMMARY.items():
            self.log(f"{n}.log", t)
            steps.append(f"[3] {n} exit 0 expect 0 ok start=2")
        for m, t in FAULT_LOG.items():
            self.log(f"fault_{m}.log", f"safety_faults mode={m}\n" + t)
            e = EXIT.get(m, 1)
            steps.append(f"[4] fault_{m} exit {e} expect {e if m in EXIT else 'nz'} ok start=3")
        self.log("steps.log", "\n".join(steps) + "\n")
        with open(os.path.join(self.b, "RUN_DONE"), "w") as f:
            f.write("rc=0\n")

    def log(self, name, text):
        with open(os.path.join(self.L, name), "w", encoding="utf-8") as f:
            f.write(text)

    def write_src(self, rel, text):
        p = os.path.join(self.src, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)

    def step(self, name, line):
        p = os.path.join(self.L, "steps.log")
        t = open(p, encoding="utf-8").read().splitlines()
        t = [line if (x.split()[1:2] == [name]) else x for x in t]
        if line is None:
            t = [x for x in t if x is not None]
        self.log("steps.log", "\n".join(x for x in t if x is not None) + "\n")


def check(r):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ak.main(["asan_check.py", "check", r.b, r.src])
    return rc, buf.getvalue()


class AsanCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="asanchk_")
        self.r = Run(os.path.join(self.tmp, "t"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_run_passes(self):
        rc, out = check(self.r)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"asan check: ok ({len(ak.SUITES)}/{len(ak.SUITES)} suites clean, 8/8 fault modes caught, 0 whitelisted", out)

    def test_asan_report_in_suite_fails(self):  # the server-tests CHECK_EQ dangling reference of safe025
        self.r.log("whirl-server-tests.log", "==9==ERROR: AddressSanitizer: stack-use-after-scope on address 0x1\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whirl-server-tests: AddressSanitizer report(s): stack-use-after-scope", out)

    def test_report_with_exit_0_and_summary_still_fails(self):
        self.r.log("whirl-ops-tests.log", "ops tests: 300 passed, 0 failed\n==9==ERROR: AddressSanitizer: heap-buffer-overflow on x\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whirl-ops-tests: AddressSanitizer report(s): heap-buffer-overflow", out)

    def test_suite_not_run_fails(self):
        self.r.step("whirl-vision-tests", None)
        os.remove(os.path.join(self.r.L, "whirl-vision-tests.log"))
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("step whirl-vision-tests: not in steps.log", out)
        self.assertIn("whirl-vision-tests: no log", out)

    def test_skipped_server_suite_fails(self):
        self.r.step("whirl-server-tests", "[5] whirl-server-tests exit skipped expect 0 UNEXPECTED start=4")
        self.r.log("whirl-server-tests.log", "SKIPPED: no tokenizer GGUF\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("step whirl-server-tests: exit skipped, expected 0", out)

    def test_failed_test_fails(self):
        self.r.log("whirl-tests.log", "unit tests: 6184 passed, 1 failed\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whirl-tests: no summary line", out)

    def test_suite_leak_fails(self):
        self.r.log("whirl-device-tests.log", "LEAKCHECK: 19 NullBackend instances checked, 1 leaked\ndevice tests: 178 passed, 0 failed\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whirl-device-tests: leak check reports leaked", out)

    def test_uaf_not_caught_fails(self):  # ASan off (a plain build): the UAF exits non-zero for another reason, or 0
        self.r.log("fault_uaf.log", "safety_faults mode=uaf\naccess violation\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("fault uaf: expected heap-use-after-free, got no ASan report", out)

    def test_overflow_wrong_kind_fails(self):
        self.r.log("fault_overflow.log", "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("fault overflow: expected heap-buffer-overflow", out)

    def test_leak_not_caught_fails(self):  # WHIRL_LEAKCHECK off
        self.r.log("fault_missing_free.log", "safety_faults mode=missing_free\nno fault detected inline\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("fault missing_free: expected a leaked instance", out)

    def test_run_done_missing_or_bad_fails(self):
        os.remove(os.path.join(self.r.b, "RUN_DONE"))
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("RUN_DONE missing", out)
        with open(os.path.join(self.r.b, "RUN_DONE"), "w") as f:
            f.write("rc=2\n")
        rc, out = check(self.r)
        self.assertIn("RUN_DONE rc=2", out)

    def test_stale_log_and_no_run_start_fail(self):  # arch025-Q135 run 2: an earlier run's vision log was read as current
        self.r.step("whirl-vision-tests", None)
        p = os.path.join(self.r.L, "whirl-vision-tests.log")
        t0 = os.path.getmtime(os.path.join(self.r.b, "RUN_START"))
        os.utime(p, (t0 - 30, t0 - 30))
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whirl-vision-tests: log whirl-vision-tests.log is older than RUN_START", out)
        self.assertIn("step whirl-vision-tests: not in steps.log", out)
        os.remove(os.path.join(self.r.b, "RUN_START"))
        rc, out = check(self.r)
        self.assertIn("RUN_START missing", out)

    def test_unlisted_opt_out_fails_listed_passes(self):
        self.r.write_src("src/a/y.cpp", "// x\n__declspec(no_sanitize_address) void g() {}\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("ASan opt-out not in the whitelist: src/a/y.cpp:2", out)
        self.r.write_src("tools/safety/asan_allow.txt", "# h\nsrc/a/y.cpp:2\treads a guard page on purpose (test)\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 0, out)
        self.assertIn("1 whitelisted opt-out site(s)", out)

    def test_stale_or_reasonless_whitelist_fails(self):
        self.r.write_src("tools/safety/asan_allow.txt", "src/a/x.cpp:1\tno opt-out there any more (test)\nsrc/a/x.cpp:9\n")
        rc, out = check(self.r)
        self.assertEqual(rc, 1, out)
        self.assertIn("whitelist entry src/a/x.cpp:1 points at no ASan opt-out", out)
        self.assertIn("need 'file:line<TAB>reason'", out)


if __name__ == "__main__":
    r = unittest.main(verbosity=2, exit=False).result
    n = r.testsRun
    bad = len(r.failures) + len(r.errors)
    print(f"asan check test: {n - bad}/{n} ok")
    sys.exit(0 if bad == 0 else 1)
