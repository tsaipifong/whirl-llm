# SPDX-License-Identifier: Apache-2.0
"""Self-check of whirl-server-tests' per-case watchdog (arch025-t915, PLAN 9.15). CPU, ~10 s.

    python -I tools/gates/server_watchdog_test.py --exe whirl-server-tests.exe --gguf SWIFT.gguf

The hidden case watchdog_selfcheck_hang never returns. With WHIRL_TEST_CASE_TIMEOUT_S=3 the program must end by itself, fast,
with exit 124 and a line "FAIL watchdog_selfcheck_hang: TIMEOUT"; without the variable it must keep running (the outer
timeout here kills it: that is today's behaviour, kept as the default). A real case must still pass with the watchdog on.
exit 0 = "server watchdog test: N/N ok".
"""
import argparse
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--exe", required=True)
ap.add_argument("--gguf", required=True)
a = ap.parse_args()
ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


def run(args, env_extra, outer):
    env = dict(os.environ)
    env.pop("WHIRL_TEST_CASE_TIMEOUT_S", None)
    env.update(env_extra)
    p = subprocess.Popen([a.exe, "--gguf", a.gguf, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    t0 = time.time()
    try:
        out, _ = p.communicate(timeout=outer)
        return p.returncode, out, time.time() - t0, False
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)  # by PID
        out, _ = p.communicate()
        return None, out or "", time.time() - t0, True


rc, out, sec, hung = run(["--only", "watchdog_selfcheck_hang"], {"WHIRL_TEST_CASE_TIMEOUT_S": "3"}, 60)
expect("watchdog on: the hanging case does not hang the program", not hung, f"still running after {sec:.0f} s")
expect("watchdog on: exit code 124", rc == 124, rc)
expect("watchdog on: the failing case is named", "FAIL watchdog_selfcheck_hang: TIMEOUT" in out, out[-300:])
expect("watchdog on: ended near the limit", sec < 30, sec)

rc, out, sec, hung = run(["--only", "watchdog_selfcheck_hang"], {}, 8)
expect("watchdog off (default): behaviour unchanged, the case still blocks", hung, (rc, out[-200:]))

rc, out, sec, hung = run(["--only", "protocol"], {"WHIRL_TEST_CASE_TIMEOUT_S": "120"}, 120)
expect("watchdog on: a passing case is not disturbed", rc == 0 and "server tests:" in out and " 0 failed" in out, (rc, out[-200:]))

rc, out, sec, hung = run(["--only", "protocol"], {"WHIRL_TEST_CASE_TIMEOUT_S": "abc"}, 120)
expect("bad value: usage error exit 2, not silent", rc == 2, (rc, out[-200:]))

print(f"server watchdog test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
