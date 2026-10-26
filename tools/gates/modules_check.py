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
Standards (arch025-Q7, check_standards): FAIL when a module has no standards.modules entry (exact name or a glob such as
platform.*), an entry has no refactor proof, names an undeclared kind / class or a metric key that is not host.* /
stage.* / tier.* / kern.*, has no metrics and no metrics_none reason, misses a mt-p3 metric of the same module, a kind
has no compare / bitwise, a class required by name (refactor / precise / balance / fast) is missing, balance or fast
allow bitwise, a class that allows bitwise is neither refactor nor precise-only, a no-bitwise class lists a bitwise
kind (three-mode standard: bit-identity only for precise and behaviour-preserving refactors), refactor lists no bitwise
kind, a class's required kind is missing, or a Q7 field holds a numeric threshold (numbers stay in the measuring tools).
modules.json is dev-time fast feedback only until the owner signs off O1 / O2 (arch024 PLAN).
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
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


MODES = {"precise", "balance", "fast"}
NAMED_CLASSES = ("refactor", "precise", "balance", "fast")
METRIC_RE = re.compile(r"^(host|stage|tier|kern)\.[a-z0-9_]+(\.[a-z0-9_]+)*$")
NUMBER_RES = (re.compile(r"\d\s*%"), re.compile(r"[<>]=?\s*\d"),
              re.compile(r"\b\d+(\.\d+)?\s*(ms|us|ns|tok/s|GB/s|GB|MB|KiB|MiB|GiB|x)\b"))


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield str(k)
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def _is_glob(k):
    return any(c in k for c in "*?[")


def standard_of(mods_std, name):
    """standards.modules entry of a module: the exact name first, then the first glob key that matches"""
    if name in mods_std:
        return name, mods_std[name]
    for k, v in mods_std.items():
        if _is_glob(k) and fnmatch.fnmatch(name, k):
            return k, v
    return None, None


def check_standards(mj, mods):
    """arch025-Q7: every module has a standard (metric keys; per behaviour class the measurement kinds); no numbers"""
    errs, notes = [], []
    std = mj.get("standards") or {}
    kinds, classes, mstd = std.get("kinds") or {}, std.get("classes") or {}, std.get("modules") or {}
    if not kinds or not classes or not mstd:
        return ["standards: needs kinds, classes and modules (arch025-Q7)"], notes
    for k, d in kinds.items():
        if not isinstance(d, dict) or not d.get("compare") or not isinstance(d.get("bitwise"), bool):
            errs.append(f"standards kind {k}: needs compare and bitwise (true / false)")
    bitwise = {k for k, d in kinds.items() if isinstance(d, dict) and d.get("bitwise") is True}
    for c in NAMED_CLASSES:
        if c not in classes:
            errs.append(f"standards: class {c} missing")
    for c, d in classes.items():
        modes = set(d.get("modes") or [])
        if not modes or not modes <= MODES:
            errs.append(f"standards class {c}: modes must be a non-empty subset of {sorted(MODES)}")
        ab = d.get("allow_bitwise")
        if not isinstance(ab, bool):
            errs.append(f"standards class {c}: allow_bitwise must be true / false")
        if c in ("balance", "fast") and ab is not False:
            errs.append(f"standards class {c}: allow_bitwise must be false (three-mode standard: {c} is judged by task accuracy)")
        if ab is True and c != "refactor" and modes != {"precise"}:
            errs.append(f"standards class {c}: only refactor or a precise-only class may be proved bitwise")
        for r in d.get("requires") or []:
            if r not in kinds:
                errs.append(f"standards class {c}: requires unknown kind {r}")
    for name in mods:
        key, e = standard_of(mstd, name)
        if e is None:
            errs.append(f"{name}: no standards entry (standards.modules.{name} or a glob): metric keys + refactor proof kinds")
            continue
        if key != name:
            notes.append(f"{name}: standard from standards.modules.{key}")
        p3 = std.get(name)
        if isinstance(p3, dict) and key == name and isinstance(e, dict):
            have = set(e.get("metrics") or [])
            for m in p3.get("metrics") or []:
                m0 = str(m).split(" (")[0].strip()
                if m0 not in have:
                    errs.append(f"{name}: mt-p3 metric {m0} (standards.{name}) missing from standards.modules.{name}.metrics")
    for key, e in mstd.items():
        if not isinstance(e, dict):
            errs.append(f"standards.modules.{key}: must be an object")
            continue
        if key not in mods and not (_is_glob(key) and any(fnmatch.fnmatch(n, key) for n in mods)):
            notes.append(f"standards.modules.{key}: declared before its module (no module of that name yet)")
        ms = e.get("metrics")
        if not isinstance(ms, list):
            errs.append(f"standards.modules.{key}: metrics must be a list")
            ms = []
        for m in ms:
            if not isinstance(m, str) or not METRIC_RE.match(m):
                errs.append(f"standards.modules.{key}: metric {m!r} is not a host.* / stage.* / tier.* / kern.* key")
        if not ms and not e.get("metrics_none"):
            errs.append(f"standards.modules.{key}: no metrics and no metrics_none reason")
        proof = e.get("proof") or {}
        if "refactor" not in proof:
            errs.append(f"standards.modules.{key}: no refactor proof (0.2.5 modules are behaviour-preserving moves)")
        for c, ks in proof.items():
            if c not in classes:
                errs.append(f"standards.modules.{key}: unknown class {c}")
                continue
            ks = list(ks or [])
            if not ks:
                errs.append(f"standards.modules.{key}: class {c} lists no kinds")
            for k in ks:
                if k not in kinds:
                    errs.append(f"standards.modules.{key}: class {c}: unknown kind {k}")
            if classes[c].get("allow_bitwise") is not True:
                bad = [k for k in ks if k in bitwise]
                if bad:
                    errs.append(f"standards.modules.{key}: class {c} lists bitwise kind(s) {bad}: {c} is never judged bitwise (three-mode standard)")
            if classes[c].get("requires_bitwise") and ks and not any(k in bitwise for k in ks):
                errs.append(f"standards.modules.{key}: class {c} lists no bitwise kind (a behaviour-preserving change is proved bit-identical)")
            for r in classes[c].get("requires") or []:
                if r not in ks:
                    errs.append(f"standards.modules.{key}: class {c} must list kind {r}")
    for part in ("kinds", "classes", "modules", "_doc_q7"):
        for t in _strings(std.get(part)):
            for rx in NUMBER_RES:
                mt = rx.search(t)
                if mt:
                    errs.append(f"standards.{part}: numeric threshold {mt.group(0)!r} in {t[:60]!r} (numbers stay in the measuring tools / results.db)")
                    break
    return errs, notes


def check(mj, files, steps):
    errs, notes = [], []
    if "dev-only" not in str(mj.get("status", "")):
        errs.append("status must say dev-only (not for releases until owner sign-off O1/O2)")
    mods = mj.get("modules") or {}
    if not mods:
        errs.append("no modules")
    # arch025-Q1 / Q3: the safety impact rules name gate steps and lifetime modules that must exist
    for cls, d in (mj.get("safety") or {}).items():
        if cls.startswith("_") or not isinstance(d, dict):
            continue
        for s in d.get("steps") or []:
            if s not in steps:
                errs.append(f"safety {cls}: step {s} is not in release_gate.ps1")
        for n in d.get("modules") or []:
            if n not in mods:
                errs.append(f"safety {cls}: module {n} is not in modules")
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
    if mods:
        e2, n2 = check_standards(mj, mods)
        errs += e2
        notes += n2
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
