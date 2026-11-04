# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests for tools/gates/pathlen_check.py (PATH-256 P6, progress\\pathlen256.DISCOVERY.md §2a / §2d-4).

Rules under test: an .exe whose full path is over 150 bytes in UTF-8 fails (Sir Pifeng: no GPU program from a path over
~150 characters; 346 crashed, 150 was measured safe); any other file or folder whose full path is 240 UTF-16 units or more
fails (15 units of room below 255). Lengths are of the full path without a \\\\?\\ prefix. Every case builds its own tree
under one short temp folder, so the lengths are exact for whatever folder the test runs in.
    python -I tools/gates/pathlen_check_test.py
Last line "pathlen check test: N/M as expected"; exit 0 all as expected, 1 not.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "pathlen_check.py")
LP = "\\\\?\\"


def lp(p):
    return p if p.startswith(LP) else LP + p


def make_file(path):
    """create path (full, absolute) with its folders, through \\\\?\\ (no LongPathsEnabled needed)"""
    os.makedirs(lp(os.path.dirname(path)), exist_ok=True)
    with open(lp(path), "wb") as f:
        f.write(b"x")


def padded(base, total_units, ext, name_char="a", unit_of=lambda s: len(s.encode("utf-16-le")) // 2):
    """a path base\\d\\<name><ext> whose length (by unit_of) is exactly total_units; folders of <= 100 chars"""
    parts = []
    cur = base
    # leave room for the file name: split the rest into 100-char folders
    while unit_of(cur) + 1 + 100 + 1 + 20 < total_units:
        seg = "d" * 100
        parts.append(seg)
        cur = cur + "\\" + seg
    stem_units = total_units - unit_of(cur) - 1 - unit_of(ext)
    one = unit_of(name_char)
    if stem_units <= 0 or stem_units % one:
        raise ValueError("cannot pad to %d" % total_units)
    p = cur + "\\" + name_char * (stem_units // one) + ext
    assert unit_of(p) == total_units, (unit_of(p), total_units)
    return p


def u8(s):
    return len(s.encode("utf-8"))


def run(*args):
    p = subprocess.run([sys.executable, "-I", TOOL] + list(args), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, p.stdout + p.stderr


def main():
    tmp = tempfile.mkdtemp(prefix="pl_")
    results = []

    def case(name, files, want_rc, want_text, extra=()):
        root = os.path.join(tmp, "c%d" % len(results))
        os.makedirs(root)
        for f in files:
            make_file(f(root))
        rc, out = run("--root", root, *extra)
        ok = rc == want_rc and all(t in out for t in want_text)
        results.append(ok)
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  (rc %d, want %d; output:\n%s)" % (rc, want_rc, out)))

    case("pass: exe of exactly 150 bytes, other file of 239 units", [lambda r: padded(r, 150, ".exe", unit_of=u8),
                                                                     lambda r: padded(r, 239, ".obj")], 0,
         ["pathlen check: ok", "longest exe 150 bytes", "longest path 239"])
    case("block: exe of 151 bytes", [lambda r: padded(r, 151, ".exe", unit_of=u8)], 1,
         ["FAIL exe 151 bytes > 150", "pathlen check: FAIL (1 path(s))"])
    case("block: upper-case .EXE of 151 bytes", [lambda r: padded(r, 151, ".EXE", unit_of=u8)], 1,
         ["FAIL exe 151 bytes > 150"])
    case("block: exe with a 51-character Chinese name: under 150 characters, 153+ bytes (UTF-8 bytes, not characters)",
         [lambda r: r + "\\" + "中" * 51 + ".exe"], 1, ["FAIL exe"])
    case("block: other file of 240 units", [lambda r: padded(r, 240, ".obj")], 1,
         ["FAIL path 240 >= 240", "pathlen check: FAIL (1 path(s))"])
    root = os.path.join(tmp, "dir240")
    os.makedirs(lp(padded(root, 240, "")))
    rc, out = run("--root", root)
    ok = rc == 1 and "FAIL path 240 >= 240" in out
    results.append(ok)
    print(("ok   " if ok else "FAIL ") + "block: a folder of 240 units (empty)" + ("" if ok else "\n" + out))
    case("pass: an .exe.manifest or .exe.pdb is not an exe (240 rule only)",
         [lambda r: padded(r, 200, ".exe.pdb", unit_of=u8)], 0, ["pathlen check: ok"])
    rc, out = run("--root", os.path.join(tmp, "no_such_root"))
    ok = rc == 2 and "pathlen check: SETUP" in out
    results.append(ok)
    print(("ok   " if ok else "FAIL ") + "setup: a missing root -> exit 2" + ("" if ok else "\n" + out))
    # two roots, one inside the other: each path counted once
    r2 = os.path.join(tmp, "nest")
    make_file(padded(os.path.join(r2, "sub"), 151, ".exe", unit_of=u8))
    rc, out = run("--root", r2, "--root", os.path.join(r2, "sub"))
    ok = rc == 1 and "pathlen check: FAIL (1 path(s))" in out
    results.append(ok)
    print(("ok   " if ok else "FAIL ") + "nested roots: one hit counted once" + ("" if ok else "\n" + out))
    shutil.rmtree(lp(tmp), ignore_errors=True)
    n = sum(1 for r in results if r)
    print("pathlen check test: %d/%d as expected" % (n, len(results)))
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
