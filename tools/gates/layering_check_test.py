# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests for tools/gates/layering_check.py on synthetic trees."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "layering_check.py")
MODS = {"lo": {"paths": ["src/lo/*"]}, "hi": {"paths": ["src/hi/*"]}, "mid": {"paths": ["src/mid/*"]}}


def tree(files):
    root = tempfile.mkdtemp(prefix="layer_")
    for f, body in files.items():
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(body)
    os.makedirs(os.path.join(root, "tools", "gates"), exist_ok=True)
    json.dump({"modules": MODS}, open(os.path.join(root, "tools", "gates", "modules.json"), "w"))
    return root


def run(root, *extra):
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root] + list(extra), capture_output=True, text=True)
    return p.returncode, p.stdout


BASE = {"src/lo/lo.h": "int lo();\n", "src/mid/mid.h": '#include "../lo/lo.h"\n', "src/hi/hi.cpp": '#include "../mid/mid.h"\n'}
fails = 0
total = 0


def case(name, ok, detail):
    global fails, total
    total += 1
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + detail))


root = tree(BASE)
rc, out = run(root, "--update")
case("update writes the baseline", rc == 0 and "baseline written (2 edges, 0 cycle(s))" in out, out)
rc, out = run(root)
case("pass: tree equal to its baseline", rc == 0 and "layering: ok (2 edges" in out, out)
open(os.path.join(root, "src", "hi", "x.cpp"), "w").write('#include "../lo/lo.h"\n')
rc, out = run(root)
case("block: a new cross-module include", rc == 1 and "new cross-module include hi -> lo" in out, out)
os.remove(os.path.join(root, "src", "hi", "x.cpp"))
open(os.path.join(root, "src", "lo", "back.cpp"), "w").write('#include "../hi/hi.h"\n')
open(os.path.join(root, "src", "hi", "hi.h"), "w").write('#include "../mid/mid.h"\n')
rc, out = run(root)
case("block: a new cycle (and its new edge)", rc == 1 and "include cycle not allowed: hi, lo, mid" in out, out)
run(root, "--update")
rc, out = run(root)
case("pass: the same cycle once declared", rc == 0 and "1 cycle(s)" in out, out)
os.remove(os.path.join(root, "src", "lo", "back.cpp"))
rc, out = run(root)
case("pass: removing an edge shrinks, prints gone", rc == 0 and "gone lo -> hi" in out, out)
open(os.path.join(root, "src", "free.cpp"), "w").write('#include "lo/lo.h"\n')
rc, out = run(root)
case("block: an unregistered file is its own node (file:src/free.cpp -> lo)", rc == 1 and "file:src/free.cpp -> lo" in out, out)
shutil.rmtree(root, ignore_errors=True)
root = tree(BASE)
rc, out = run(root)
case("block: no baseline file", rc == 1 and "cannot read" in out, out)
shutil.rmtree(root, ignore_errors=True)
print(f"layering check test: {total - fails}/{total} ok")
sys.exit(1 if fails else 0)
