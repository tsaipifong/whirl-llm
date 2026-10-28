# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests for tools/gates/census_check.py on synthetic trees."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "census_check.py")


def tree(files, mods, pending=None):
    root = tempfile.mkdtemp(prefix="census_")
    for f in files:
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("// x\n")
    os.makedirs(os.path.join(root, "tools", "gates"), exist_ok=True)
    json.dump({"modules": mods}, open(os.path.join(root, "tools", "gates", "modules.json"), "w"))
    if pending is not None:
        json.dump({"pending": pending}, open(os.path.join(root, "tools", "gates", "census_pending.json"), "w"))
    return root


def run(root, *extra):
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root] + list(extra), capture_output=True, text=True)
    return p.returncode, p.stdout


A = {"a": {"paths": ["src/a/*"]}, "b": {"paths": ["src/b.cpp", "include/whirl/b.h"]}}
CASES = [
    ("pass: every product file in one module", ["src/a/x.cpp", "src/a/y.h", "src/b.cpp", "include/whirl/b.h", "tests/t.cpp"],
     A, None, [], 0, "census: ok"),
    ("block: a product file in no module", ["src/a/x.cpp", "src/c.cpp"], A, None, [], 1, "src/c.cpp is in no module"),
    ("block: a kernel file in no module", ["src/a/x.cpp", "kernels/k.hip"], A, None, [], 1, "kernels/k.hip is in no module"),
    ("pass: an unregistered file listed as pending (WARN)", ["src/a/x.cpp", "src/c.cpp"], A,
     [{"glob": "src/c.cpp", "owner": "lane 9"}], [], 0, "WARN pending 'src/c.cpp'"),
    ("block: a stale pending entry", ["src/a/x.cpp", "src/b.cpp"], A, [{"glob": "src/b.cpp", "owner": "lane 9"}], [], 1,
     "stale pending entry"),
    ("block: release mode with a pending entry left", ["src/a/x.cpp", "src/c.cpp"], A,
     [{"glob": "src/c.cpp", "owner": "lane 9"}], ["--release"], 1, "release mode"),
    ("block: a file in two modules", ["src/a/x.cpp"], {"a": {"paths": ["src/a/*"]}, "a2": {"paths": ["src/a/x.cpp"]}},
     None, [], 1, "is in 2 modules"),
    ("pass: non-product files are not counted", ["src/a/x.cpp", "docs/d.md", "tools/t.py", "src/a/readme.txt"], A, None,
     [], 0, "census: ok (1 product files"),
]
fails = 0
for name, files, mods, pending, extra, want, sub in CASES:
    root = tree(files, mods, pending)
    rc, out = run(root, *extra)
    ok = rc == want and sub in out
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  rc {rc} out {out[-300:]!r}"))
    shutil.rmtree(root, ignore_errors=True)
print(f"census check test: {len(CASES) - fails}/{len(CASES)} ok")
sys.exit(1 if fails else 0)
