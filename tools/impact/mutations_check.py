# SPDX-License-Identifier: Apache-2.0
"""Stale-mutant check for tools/impact/mutations.json (CPU, a second): every mutant must still apply.

    python tools/impact/mutations_check.py [--root REPO] [--spec FILE]    check the committed tree
    python tools/impact/mutations_check.py --selftest                     synthetic cases

A "replace" mutant is stale when its "find" text does not occur exactly once in its file (mutate.py refuses
it, so a refactor that moves or rewrites the code silently turns the mutant into a no-op in every report that
only counts the survivors). A "revert" mutant is stale when its file is gone or the commit's diff of that file
no longer reverse-applies (git apply -R --check). Exit 0 = every mutant applies, 1 = stale mutants listed
(id, kind, file, why), 2 = bad input. Run it in the commit that moves / rewrites code a mutant targets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def stale(root, muts):
    out = []
    for m in muts:
        path = os.path.join(root, m.get("file", ""))
        kind = m.get("kind")
        if not os.path.isfile(path):
            out.append((m["id"], kind, m.get("file"), "file does not exist"))
            continue
        if kind == "replace":
            n = open(path, "rb").read().decode("utf-8", errors="replace").count(m["find"])
            if n != 1:
                out.append((m["id"], kind, m["file"], f"find text occurs {n} times (want 1)"))
        elif kind == "revert":
            d = subprocess.run(["git", "-C", root, "show", "--format=", m["commit"], "--", m["file"]], capture_output=True)
            if d.returncode != 0 or not d.stdout.strip():
                out.append((m["id"], kind, m["file"], f"commit {m['commit']} has no diff of this file"))
                continue
            r = subprocess.run(["git", "-C", root, "apply", "-R", "--check", "--whitespace=nowarn", "-"], input=d.stdout, capture_output=True)
            if r.returncode != 0:
                out.append((m["id"], kind, m["file"], f"cannot reverse-apply {m['commit']}: {r.stderr.decode(errors='replace').strip()[:120]}"))
        else:
            out.append((m["id"], kind, m["file"], "unknown kind"))
    return out


def selftest():
    cases = []
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.cpp"), "w", encoding="utf-8", newline="\n").write("int f() { return 1; }\nint g() { return 1; }\nint h() { return 2; }\n")
        mk = lambda i, find: {"id": i, "kind": "replace", "file": "a.cpp", "find": find, "replace": "x"}
        got = {s[0]: s[3] for s in stale(d, [mk("once", "return 2;"), mk("twice", "return 1;"), mk("zero", "return 3;"),
                                              {"id": "nofile", "kind": "replace", "file": "b.cpp", "find": "x", "replace": "y"},
                                              {"id": "badkind", "kind": "zap", "file": "a.cpp"}])}
        cases.append(("applying mutant passes", "once" not in got))
        cases.append(("find twice is stale", "twice" in got and "2 times" in got["twice"]))
        cases.append(("find zero is stale", "zero" in got and "0 times" in got["zero"]))
        cases.append(("missing file is stale", "nofile" in got))
        cases.append(("unknown kind is stale", "badkind" in got))
        cases.append(("revert of a non-repo commit is stale", "rv" in {s[0] for s in stale(d, [{"id": "rv", "kind": "revert", "file": "a.cpp", "commit": "deadbee"}])}))
    ok = 0
    for n, g in cases:
        ok += g
        print(f"{'ok  ' if g else 'FAIL'} {n}")
    print(f"mutations check selftest: {ok}/{len(cases)} ok")
    return 0 if ok == len(cases) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--spec")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    spec = a.spec or os.path.join(a.root, "tools", "impact", "mutations.json")
    try:
        muts = json.load(open(spec, encoding="utf-8"))["mutations"]
    except (OSError, ValueError, KeyError) as e:
        print(f"mutations_check: cannot read {spec}: {e}")
        return 2
    bad = stale(a.root, muts)
    for i, k, f, why in bad:
        print(f"STALE {i} ({k}, {f}): {why}")
    print(f"mutations check: {'ok' if not bad else 'FAIL'} ({len(muts) - len(bad)}/{len(muts)} mutants apply)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
