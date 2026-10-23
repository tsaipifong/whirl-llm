# SPDX-License-Identifier: Apache-2.0
"""Checks tools/gates/modules.json against a source tree and release_gate.ps1's step list (arch024-S12).

    python tools/gates/modules_check.py [--modules PATH] [--tree DIR | --rev REV] [--steps a,b,...]

--tree: a checked-out tree (default: this repo); --rev: a git revision of this repo, read with `git archive` into a
temp folder (read-only: no worktree is touched, e.g. --rev arch024-s6b). --steps: the gate's step names (default:
`pwsh release_gate.ps1 -List` of this repo's gate, which modules.json belongs to).
FAIL (exit 1): status not marked dev-only; a module without id / steps (and not full); a step that is neither in the
gate nor pending; a pending step that is in the gate (stale); a when-step not in the gate; a module with paths of which
none is in the tree and no lands_on; a file of the tree claimed by two modules. Tolerated (printed): a module whose files
are not in the tree but that has lands_on, a module with no paths yet. Exit 0 = "modules check: ok".
modules.json is dev-time fast feedback only until the owner signs off O1 / O2 (arch024 PLAN).
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def tree_files(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", "build", "__pycache__")]
        for f in files:
            out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return out


def rev_files(rev):
    data = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", rev], capture_output=True)
    if data.returncode != 0:
        raise SystemExit(f"modules check: git archive {rev} failed: {data.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(data.stdout)) as t:
        return [m.name for m in t.getmembers() if m.isfile()]


def gate_steps():
    r = subprocess.run(["pwsh", "-NoProfile", "-File", os.path.join(HERE, "release_gate.ps1"), "-Device", "r9700", "-List"],
                       capture_output=True, text=True)
    steps = [l.split()[0] for l in r.stdout.splitlines() if len(l.split()) == 2 and l.split()[1] in ("CPU", "GPU")]
    if r.returncode != 0 or len(steps) < 10:
        raise SystemExit(f"modules check: release_gate.ps1 -List failed (exit {r.returncode}, {len(steps)} steps)")
    # 8060s-only / r9700-only steps: none differ today except dfl_* (needs WHIRL_GATE_DENSE); modules do not use them
    return set(steps)


def check(mj, files, steps):
    errs, notes = [], []
    if "dev-only" not in str(mj.get("status", "")):
        errs.append("status must say dev-only (not for releases until owner sign-off O1/O2)")
    mods = mj.get("modules") or {}
    if not mods:
        errs.append("no modules")
    owner = {}
    for n, d in mods.items():
        if not d.get("id"):
            errs.append(f"{n}: no id")
        st = list(d.get("steps") or [])
        if not st and not d.get("full"):
            errs.append(f"{n}: no steps and not full")
        pend = {p.get("step") for p in d.get("pending") or []}
        for p in pend:
            if p in steps:
                errs.append(f"{n}: pending step {p} is in the gate now: move it to steps")
        for s in st:
            if s not in steps and s not in pend:
                errs.append(f"{n}: step {s} is not in release_gate.ps1 and not pending")
        for f, sts in (d.get("when") or {}).items():
            for s in sts:
                if s not in steps:
                    errs.append(f"{n}: when {f}: step {s} is not in release_gate.ps1")
        c = d.get("callers")
        if c is not None and c != "rdeps" and not (isinstance(c, list) and all(x in mods and x != n for x in c)):
            errs.append(f"{n}: callers must be \"rdeps\" or a list of other module names, got {c!r}")
        globs = list(d.get("paths") or [])
        present = [f for f in files if any(fnmatch.fnmatch(f, g) for g in globs)]
        for f in present:
            if f in owner and owner[f] != n:
                errs.append(f"{f} is in two modules: {owner[f]} and {n}")
            owner[f] = n
        if globs and not present:
            if d.get("lands_on"):
                notes.append(f"{n} ({d.get('id')}): declared, not in this tree (lands on {d['lands_on']})")
            else:
                errs.append(f"{n}: none of {', '.join(globs)} is in the tree and no lands_on")
        elif not globs:
            notes.append(f"{n} ({d.get('id')}): no paths yet (lands on {d.get('lands_on')})")
        else:
            notes.append(f"{n} ({d.get('id')}): {len(present)} file(s): {', '.join(present[:4])}{' ...' if len(present) > 4 else ''}")
    return errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default=os.path.join(HERE, "modules.json"))
    ap.add_argument("--tree")
    ap.add_argument("--rev")
    ap.add_argument("--steps")
    a = ap.parse_args()
    try:
        mj = json.load(open(a.modules, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"modules check: FAIL: cannot read {a.modules}: {e}")
        return 1
    files = rev_files(a.rev) if a.rev else tree_files(a.tree or REPO)
    steps = set(a.steps.split(",")) if a.steps else gate_steps()
    errs, notes = check(mj, files, steps)
    where = f"rev {a.rev}" if a.rev else (a.tree or REPO)
    for n in notes:
        print(f"   {n}")
    for e in errs:
        print(f"   FAIL {e}")
    print(f"modules check: {'ok' if not errs else f'FAIL ({len(errs)} problem(s))'} ({len(mj.get('modules') or {})} modules, {where}, {len(steps)} gate steps)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
