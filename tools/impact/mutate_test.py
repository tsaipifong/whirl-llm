# SPDX-License-Identifier: Apache-2.0
"""Tests of mutate.py's hang handling (arch025-t915, PLAN 9.15). CPU, ~10 s.

    python -I tools/impact/mutate_test.py        exit 0 = "mutate test: N/N ok"

A test command that outlives its timeout must become a recorded verdict CAUGHT(timeout): the whole process tree is killed
by PID (a grandchild too), and the next mutant still runs.
"""
import io
import re
import json
import os
import subprocess
import sys
import tempfile
import time
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate  # noqa: E402

ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


def alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def wait_gone(pid, secs=5.0):
    t0 = time.time()
    while alive(pid) and time.time() - t0 < secs:
        time.sleep(0.2)
    return not alive(pid)


tmp = tempfile.mkdtemp(prefix="mutate_test_")
py = sys.executable
# a "test" that spawns a grandchild and then sleeps far past the timeout. The PARENT writes the grandchild's pid right after
# Popen (int7 verify 10-30: when the grandchild wrote it itself, a loaded machine killed the tree at the 3 s timeout before the
# grandchild's interpreter had started, so the pid file was missing; the timeouts below are 15 s for the same reason).
parent = os.path.join(tmp, "hang.py")
SPAWN = ["import subprocess, sys, time",
         "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])",
         "open(PIDFILE, chr(119)).write(str(p.pid))",
         "time.sleep(300)"]
NL = chr(10)


def spawn(pidfile):
    return NL.join(SPAWN).replace("PIDFILE", repr(pidfile)) + NL

open(parent, "w").write(spawn(os.path.join(tmp, "child.pid")))

# 1. sh(): a hang is rc 124 + TIMEOUT text, the tree is gone
t0 = time.time()
try:
    rc, out, sec = mutate.sh(f'"{py}" "{parent}"', timeout=15)
    err = None
except Exception as e:  # before the fix: subprocess.TimeoutExpired
    rc, out, sec, err = None, "", 0, repr(e)
expect("sh: a hanging command returns instead of raising", err is None, err)
expect("sh: rc 124 and TIMEOUT in the output", rc == 124 and "TIMEOUT" in out, (rc, out[:80]))
expect("sh: returned near the timeout (not 300 s)", time.time() - t0 < 30, time.time() - t0)
cpid = None
try:
    cpid = int(open(os.path.join(tmp, "child.pid")).read())
except Exception as e:
    expect("sh: the grandchild started", False, repr(e))
if cpid:
    gone = wait_gone(cpid)
    if not gone:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(cpid)], capture_output=True)
    expect("sh: the grandchild process is killed with the tree", gone, f"pid {cpid} still alive")

# 2. a normal command is unchanged
rc, out, sec = mutate.sh(f'"{py}" -c "print(1)"', timeout=30)
expect("sh: normal command rc 0 + output", rc == 0 and out.strip() == "1", (rc, out))
rc, out, sec = mutate.sh(f'"{py}" -c "import sys; sys.exit(3)"', timeout=30)
expect("sh: failing command keeps its rc", rc == 3, rc)

# 3. main(): hang mutant -> CAUGHT(timeout) (per-row timeout_s), next mutant still runs, source restored
repo = os.path.join(tmp, "repo")
os.makedirs(os.path.join(repo, "src"))
for n in ("a.txt", "b.txt"):
    open(os.path.join(repo, "src", n), "w", newline="\n").write("good\n")
g = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)
g("init", "-q")
g("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
chk = os.path.join(tmp, "chk.py")
# exit 0 while the file says good, hang when src/a.txt says hang, exit 1 when src/b.txt says bad
fa, fb = os.path.join(repo, "src", "a.txt"), os.path.join(repo, "src", "b.txt")
chk_lines = ["import sys",
             f"a = open({fa!r}).read(); b = open({fb!r}).read()",
             "if 'hang' in a:"]
chk_lines += ["    " + l for l in spawn(os.path.join(tmp, "child2.pid")).splitlines()]
chk_lines += ["sys.exit(1 if 'bad' in b else 0)"]
open(chk, "w").write(NL.join(chk_lines) + NL)
spec = os.path.join(tmp, "spec.json")
json.dump({"mutations": [
    {"id": "m-hang", "module": "x", "kind": "replace", "file": "src/a.txt", "find": "good", "replace": "hang", "what": "w",
     "run": "k", "expect_test": "e:t", "timeout_s": 15},
    {"id": "m-bad", "module": "x", "kind": "replace", "file": "src/b.txt", "find": "good", "replace": "bad", "what": "w",
     "run": "k", "expect_test": "e:t", "timeout_s": 60}]}, open(spec, "w"))
tsv = os.path.join(tmp, "out.tsv")
mutate.ROOT = repo
mutate.impact_lists = lambda expect_: (True, "stub")
sys.argv = ["mutate.py", "--cmd", f'k="{py}" "{chk}"', "--spec", spec, "--tsv", tsv]
buf = io.StringIO()
t0 = time.time()
try:
    with contextlib.redirect_stdout(buf):
        mrc = mutate.main()
    merr = None
except BaseException as e:
    mrc, merr = None, repr(e)
text = buf.getvalue()
expect("main: a hanging mutant does not end the run", merr is None, merr)
expect("main: m-hang is CAUGHT(timeout)", "CAUGHT(timeout) m-hang" in text, text[-400:])
expect("main: the next mutant ran after the hang", "m-bad" in text and "CAUGHT" in text.split("m-bad")[0][-40:] + "CAUGHT", text[-300:])
expect("main: both caught, exit 0", mrc == 0, (mrc, text[-200:]))
expect("main: run took seconds, not the 1800 s default", time.time() - t0 < 60, time.time() - t0)
expect("main: hung mutant's source restored", open(os.path.join(repo, "src", "a.txt")).read() == "good\n")
expect("main: git tree clean", g("status", "--short").stdout.strip() == "", g("status", "--short").stdout)
tsvtext = open(tsv).read() if os.path.exists(tsv) else ""
expect("main: tsv records CAUGHT(timeout)", "CAUGHT(timeout)" in tsvtext, tsvtext)
try:
    c2 = int(open(os.path.join(tmp, "child2.pid")).read())
    expect("main: hung mutant's grandchild killed", wait_gone(c2), f"pid {c2} alive")
except Exception as e:
    expect("main: grandchild pid file", False, repr(e))

# 4. v0252-tool3 (PLAN 9.57): a command that exits 125 (= its BUILD failed, git bisect's "cannot test") is run once more;
# 125 again = verdict BUILD-FAIL (never CAUGHT) and main() exits 4; a build that fails once and then builds is judged on its
# test. Evidence: ornkv chain3 recorded ornkv-moe-f16 CAUGHT on a whirl-tests.exe LINK failure.
repo2 = os.path.join(tmp, "repo2")
os.makedirs(os.path.join(repo2, "src"))
for n in ("b.txt", "f.txt", "t.txt"):
    open(os.path.join(repo2, "src", n), "w", newline="\n").write("good\n")
g2 = lambda *a: subprocess.run(["git", "-C", repo2, *a], capture_output=True, text=True)
g2("init", "-q")
g2("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
g2("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
cnt = os.path.join(tmp, "flaky.count")
chk2 = os.path.join(tmp, "chk2.py")
open(chk2, "w").write(NL.join([
    "import os, sys",
    f"W = {os.path.join(repo2, 'src')!r}",
    "r = lambda n: open(os.path.join(W, n)).read()",
    "if 'broken' in r('b.txt'):",
    "    print('FAILED: [code=4294967295] whirl-tests.exe (stub link failure)'); sys.exit(125)",
    "if 'flaky' in r('f.txt'):",
    f"    c = {cnt!r}",
    "    n = int(open(c).read()) if os.path.exists(c) else 0",
    "    open(c, 'w').write(str(n + 1))",
    "    if n == 0:",
    "        sys.exit(125)",
    "    print('FAIL testFlaky'); sys.exit(1)",
    "sys.exit(1 if 'bad' in r('t.txt') else 0)"]) + NL)
spec2 = os.path.join(tmp, "spec2.json")
json.dump({"mutations": [
    {"id": "m-build", "module": "x", "kind": "replace", "file": "src/b.txt", "find": "good", "replace": "broken", "what": "w",
     "run": "k", "expect_test": "e:t"},
    {"id": "m-flaky", "module": "x", "kind": "replace", "file": "src/f.txt", "find": "good", "replace": "flaky", "what": "w",
     "run": "k", "expect_test": "e:t"},
    {"id": "m-bad", "module": "x", "kind": "replace", "file": "src/t.txt", "find": "good", "replace": "bad", "what": "w",
     "run": "k", "expect_test": "e:t"}]}, open(spec2, "w"))
tsv2 = os.path.join(tmp, "out2.tsv")
mutate.ROOT = repo2
sys.argv = ["mutate.py", "--cmd", f'k="{py}" "{chk2}"', "--spec", spec2, "--tsv", tsv2]
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        mrc = mutate.main()
    merr = None
except BaseException as e:
    mrc, merr = None, repr(e)
text = buf.getvalue()
rows = {l.split("\t")[1]: l.split("\t")[4] for l in open(tsv2).read().splitlines()[1:]} if os.path.exists(tsv2) else {}
expect("build: no exception", merr is None, merr)
expect("build: a build failure is BUILD-FAIL in the TSV, never CAUGHT", rows.get("m-build") == "BUILD-FAIL", rows)
expect("build: BUILD-FAIL line printed", "BUILD-FAIL" in text and "m-build" in text.split("BUILD-FAIL", 1)[-1][:60], text[-500:])
expect("build: a build that fails once is retried and then judged (m-flaky CAUGHT)",
       rows.get("m-flaky") == "CAUGHT" and os.path.exists(cnt) and open(cnt).read() == "2", (rows, open(cnt).read() if os.path.exists(cnt) else None))
expect("build: a test failure stays CAUGHT", rows.get("m-bad") == "CAUGHT", rows)
expect("build: main exits 4 when a mutant could not be built", mrc == 4, (mrc, text[-300:]))
expect("build: the summary counts it apart from caught", re.search(r"2/3 caught.*1 could not be built", text) is not None, text[-300:])
expect("build: sources restored", all(open(os.path.join(repo2, "src", n)).read() == "good\n" for n in ("b.txt", "f.txt", "t.txt")))

print(f"mutate test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
