# SPDX-License-Identifier: Apache-2.0
"""Tests of mutate.py's hang handling (arch025-t915, PLAN 9.15). CPU, ~10 s.

    python -I tools/impact/mutate_test.py        exit 0 = "mutate test: N/N ok"

A test command that outlives its timeout must become a recorded verdict CAUGHT(timeout): the whole process tree is killed
by PID (a grandchild too), and the next mutant still runs.
"""
import io
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
# a "test" that spawns a grandchild (writes its pid to a file) and then sleeps far past the timeout
parent = os.path.join(tmp, "hang.py")
SPAWN = ["import subprocess, sys, time",
         "subprocess.Popen([sys.executable, '-c', 'import os,time,sys; open(sys.argv[1],chr(119)).write(str(os.getpid())); time.sleep(300)', PIDFILE])",
         "time.sleep(300)"]
NL = chr(10)


def spawn(pidfile):
    return NL.join(SPAWN).replace("PIDFILE", repr(pidfile)) + NL

open(parent, "w").write(spawn(os.path.join(tmp, "child.pid")))

# 1. sh(): a hang is rc 124 + TIMEOUT text, the tree is gone
t0 = time.time()
try:
    rc, out, sec = mutate.sh(f'"{py}" "{parent}"', timeout=3)
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
     "run": "k", "expect_test": "e:t", "timeout_s": 3},
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

print(f"mutate test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
