# SPDX-License-Identifier: Apache-2.0
"""iso_includes_check.py (arch025 PLAN 9.13): every quoted #include of every file an isolation manifest stages must itself be
staged by that manifest.

Why: iso_check.py builds a module from the files its tests/**/isolation.txt lists (src / hdr / testsrc / testhdr lines) and
nothing else. A merge that adds `#include "x.h"` to a staged file but not `hdr x.h` to the manifest fails only in that ~1 min
per module build (C1083), i.e. after a 2 h verify has started (arch025-int5b: config.device env_get.h, then 8 more misses in
hip.mem / hip.restrict / hip.runtime / vision / release). This is the seconds-long static version of the same rule (Bazel's
layering_check "a target may only include headers it declares", on the manifests).

Resolution as the product build does it: the including file's own folder, then src/, then include/, then tests/. System headers
(<...>) and includes that resolve to no repo file (generated or external) are ignored; a `gen` line of the manifest names a
build-generated header. A staged file that is missing is iso_check.py's business (it reports "no such file").

Usage:  python -I tools/gates/iso_includes_check.py [--root DIR | --rev COMMIT] [--repo DIR]   |   --selftest
  --rev reads the tree of COMMIT read-only (git archive of tests / src / include into a temp dir) - for the red proofs.
Exit 0 = ok, 1 = misses (one "MISS <manifest>: <file> includes "<h>" -> <repo path> (not staged)" line each), 2 = usage.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = {"src", "hdr", "testsrc", "testhdr", "mutate", "declbase"}
INC_RX = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def manifests(root):
    out = []
    for dp, dn, fn in os.walk(os.path.join(root, "tests")):
        dn.sort()
        if "isolation.txt" in fn:
            out.append(os.path.join(dp, "isolation.txt"))
    return sorted(out)


def parse(man):
    staged, gens = set(), set()
    for ln in open(man, encoding="utf-8", errors="replace"):
        parts = ln.split("#", 1)[0].split()
        if len(parts) >= 2 and parts[0] in KINDS:
            staged.add(parts[1].replace("\\", "/"))
        elif len(parts) >= 2 and parts[0] == "gen":
            gens.add(parts[1].replace("\\", "/"))
    return staged, gens


def check(root):
    """-> (list of miss lines, n manifests, n staged files, n resolved includes)"""
    misses, nman, nfiles, ninc = [], 0, 0, 0
    for man in manifests(root):
        nman += 1
        mname = os.path.relpath(man, root).replace("\\", "/")[len("tests/"):-len("/isolation.txt")]
        staged, gens = parse(man)
        for f in sorted(staged):
            p = os.path.join(root, *f.split("/"))
            if not os.path.isfile(p):
                continue
            nfiles += 1
            text = open(p, encoding="utf-8", errors="replace").read()
            here = os.path.dirname(f)
            for inc in INC_RX.findall(text):
                cands = [os.path.normpath(os.path.join(here, inc)).replace("\\", "/"), "src/" + inc, "include/" + inc, "tests/" + inc]
                found = [c for c in cands if os.path.isfile(os.path.join(root, *c.split("/")))]
                if not found:
                    continue
                ninc += 1
                if any(c in staged for c in cands) or inc in gens or any(g.endswith(inc) for g in gens):
                    continue
                misses.append(f'MISS {mname}: {f} includes "{inc}" -> {found[0]} (not staged)')
    return misses, nman, nfiles, ninc


def export_rev(repo, rev, dest):
    p = subprocess.run(["git", "-C", repo, "archive", "--format=tar", rev, "tests", "src", "include"], capture_output=True)
    if p.returncode:
        raise SystemExit(f"iso includes check: git archive {rev} failed: {p.stderr.decode(errors='replace').strip()}")
    tmp = os.path.join(dest, "t.tar")
    open(tmp, "wb").write(p.stdout)
    with tarfile.open(tmp) as t:
        t.extractall(dest)
    os.remove(tmp)


def run(root):
    misses, nman, nfiles, ninc = check(root)
    for m in misses:
        print(m)
    if misses:
        print(f"iso includes check: FAIL ({len(misses)} miss(es): add the header to the manifest's hdr / testhdr lines)")
        return 1
    print(f"iso includes check: ok ({nman} manifests, {nfiles} staged files, {ninc} repo includes)")
    return 0


def selftest():
    total = fails = 0

    def case(name, ok, detail=""):
        nonlocal total, fails
        total += 1
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + detail))

    def tree(manifest, files):
        root = tempfile.mkdtemp(prefix="isoinc_")
        for f, body in dict(files, **{"tests/iso/m/isolation.txt": manifest}).items():
            p = os.path.join(root, *f.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(body)
        return root

    def go(root):
        misses, *_ = check(root)
        return misses

    files = {"src/a/a.cpp": '#include "a.h"\n#include <vector>\n#include "../b/b.h"\n', "src/a/a.h": '#include "whirl/c.h"\n',
             "src/b/b.h": "\n", "include/whirl/c.h": '#include "whirl/d.h"\n', "include/whirl/d.h": "\n"}
    full = "module m\nsrc src/a/a.cpp\nhdr src/a/a.h\nhdr src/b/b.h\nhdr include/whirl/c.h\nhdr include/whirl/d.h\n"
    r = tree(full, files)
    case("pass: whole include closure staged", go(r) == [], str(go(r)))
    shutil.rmtree(r)
    r = tree(full.replace("hdr include/whirl/d.h\n", ""), files)
    m = go(r)
    case("block: a transitive header (include/whirl/d.h) not staged", len(m) == 1 and 'include/whirl/c.h includes "whirl/d.h"' in m[0], str(m))
    shutil.rmtree(r)
    r = tree(full.replace("hdr src/b/b.h\n", ""), files)
    m = go(r)
    case("block: a relative include (../b/b.h) not staged", len(m) == 1 and '"../b/b.h"' in m[0], str(m))
    shutil.rmtree(r)
    r = tree(full + "# hdr src/zzz.h\n", dict(files, **{"src/a/a.cpp": files["src/a/a.cpp"] + '#include "gen.h"\n#include "nowhere.h"\n'}))
    case("pass: an include that is no repo file (generated / external) is ignored", go(r) == [], str(go(r)))
    shutil.rmtree(r)
    r = tree(full + "gen gen.h\n", dict(files, **{"src/a/a.cpp": files["src/a/a.cpp"] + '#include "gen.h"\n', "src/gen.h": "\n"}))
    case("pass: a gen line covers a header that also exists in the repo", go(r) == [], str(go(r)))
    shutil.rmtree(r)
    r = tree(full + "src src/a/gone.cpp\n", files)
    case("pass: a staged file that does not exist is not this check's business", go(r) == [], str(go(r)))
    shutil.rmtree(r)
    r = tree(full.replace("hdr src/a/a.h\n", "").replace("src src/a/a.cpp", "src src/a/a.cpp\n# hdr src/a/a.h  (commented out)"), files)
    m = go(r)
    case("block: a commented-out hdr line stages nothing", any('"a.h"' in x for x in m), str(m))
    shutil.rmtree(r)
    r = tree(full.replace("hdr include/whirl/d.h\n", ""), files)
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--root", r], capture_output=True, text=True)
    case("exit code 1 and the MISS line on the command line", p.returncode == 1 and "MISS iso/m:" in p.stdout and "FAIL (1 miss" in p.stdout, p.stdout)
    shutil.rmtree(r)
    print(f"iso includes check selftest: {'PASS' if not fails else 'FAIL'} ({total - fails}/{total})")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.normpath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--rev")
    ap.add_argument("--repo")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.rev:
        tmp = tempfile.mkdtemp(prefix="isoinc_rev_")
        try:
            export_rev(a.repo or a.root, a.rev, tmp)
            return run(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return run(a.root)


if __name__ == "__main__":
    sys.exit(main())
