# SPDX-License-Identifier: Apache-2.0
"""Census (arch025 D11 N2): every product file belongs to exactly one module of tools/gates/modules.json.

Product files: src/, include/, kernels/ with a C / C++ / HIP extension. A file in no module fails, unless a glob in
tools/gates/census_pending.json lists it (files a lane is still registering; printed as WARN with the owner). A pending
entry that matches no unregistered file is stale and fails, so the list only shrinks. --release fails while any pending
entry is left (release_check: every product file in a module before a release).

    python tools/gates/census_check.py [--root DIR] [--modules FILE] [--pending FILE] [--release]
"""
import argparse
import fnmatch
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOPS = ("src", "include", "kernels")
EXTS = (".cpp", ".h", ".hpp", ".hip", ".inc", ".c", ".cu", ".cuh")


def product_files(root):
    out = []
    for t in TOPS:
        for d, dirs, files in os.walk(os.path.join(root, t)):
            dirs[:] = [x for x in dirs if x not in (".git", "build", "__pycache__")]
            for f in files:
                if f.endswith(EXTS):
                    out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(out)


def owners(f, mods):
    """the modules whose paths globs match product file f (census: exactly one; cmake_modules_check.py uses the same rule)"""
    return [n for n, d in mods.items() if any(fnmatch.fnmatch(f, g) for g in d.get("paths") or [])]


def check(files, mods, pending, release=False):
    errs, warns = [], []
    unreg = []
    for f in files:
        own = owners(f, mods)
        if len(own) > 1:
            errs.append(f"{f} is in {len(own)} modules: {', '.join(own)}")
        elif not own:
            unreg.append(f)
    covered = set()
    for p in pending:
        g = p.get("glob") or ""
        hit = [f for f in unreg if fnmatch.fnmatch(f, g)]
        if not hit:
            errs.append(f"stale pending entry {g!r} ({p.get('owner')}): it matches no unregistered file - remove it")
            continue
        covered.update(hit)
        (errs if release else warns).append(f"pending {g!r}: {len(hit)} file(s) not in a module yet, owner {p.get('owner')}")
    for f in unreg:
        if f not in covered:
            errs.append(f"{f} is in no module (add it to a module in tools/gates/modules.json)")
    return errs, warns, len(unreg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--modules", default=None)
    ap.add_argument("--pending", default=None)
    ap.add_argument("--release", action="store_true")
    a = ap.parse_args()
    mpath = a.modules or os.path.join(a.root, "tools", "gates", "modules.json")
    ppath = a.pending or os.path.join(a.root, "tools", "gates", "census_pending.json")
    try:
        mods = json.load(open(mpath, encoding="utf-8"))["modules"]
        pending = json.load(open(ppath, encoding="utf-8")).get("pending") or [] if os.path.isfile(ppath) else []
    except (OSError, ValueError, KeyError) as e:
        print(f"census: FAIL: cannot read the registry: {e}")
        return 1
    files = product_files(a.root)
    errs, warns, n_unreg = check(files, mods, pending, a.release)
    for w in warns:
        print("   WARN " + w)
    for e in errs:
        print("   FAIL " + e)
    tail = f"{len(files)} product files, {len(mods)} modules, {n_unreg} pending in {len(pending)} entr(y/ies)"
    print(f"census: {'ok' if not errs else f'FAIL ({len(errs)} problem(s))'} ({tail}{', release mode' if a.release else ''})")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
