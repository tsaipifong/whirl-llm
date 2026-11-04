# SPDX-License-Identifier: Apache-2.0
"""PATH-256 P6 (progress\\pathlen256.DISCOVERY.md §2a / §2d-4): path lengths of a dev / test tree.

Sir Pifeng 2026-11-03: every WHIRL path stays under 256 (Windows' limit), and no GPU program runs from an exe path longer
than about 150 characters (a 0.2.5 exe at 346 characters stopped with 0xC0000409, once with a 0x3B blue screen; 150 was
measured safe: memory amd-long-exe-path-crash). Rules, on the full path without a \\\\?\\ prefix:
  - a file ending in .exe (any case) over 150 bytes in UTF-8 -> FAIL ("FAIL exe N bytes > 150: <path>")
    (bytes: the AMD runtime reads the exe path through the ANSI API, UTF-8 by the manifest; ASCII bytes = characters)
  - any other file or folder of 240 or more UTF-16 units -> FAIL ("FAIL path N >= 240: <path>") (15 units of room)
Symbolic links / junctions are listed but not followed. A root given twice, or inside another root, is walked once.
    python -I tools/gates/pathlen_check.py --root DIR [--root DIR ...]
    python -I tools/gates/pathlen_check.py --selftest          (tools/gates/pathlen_check_test.py)
Last line: "pathlen check: ok (...)" exit 0 | "pathlen check: FAIL (N path(s))" exit 1 | "pathlen check: SETUP ..." exit 2.
release_gate step cpu_pathlen runs the selftest, then this over the worktree and its build folder.
"""
import argparse
import os
import stat
import subprocess
import sys

EXE_MAX_BYTES = 150   # .exe: UTF-8 bytes, fail when greater
PATH_FAIL_UNITS = 240  # everything else: UTF-16 units, fail when greater or equal
LP = "\\\\?\\"


def strip_lp(p):
    if p.startswith(LP + "UNC\\"):
        return "\\\\" + p[len(LP) + 4:]
    return p[len(LP):] if p.startswith(LP) else p


def units16(p):
    return len(p.encode("utf-16-le", "surrogatepass")) // 2


def bytes8(p):
    return len(p.encode("utf-8", "surrogatepass"))


def walk(root):
    """yield (full path without prefix, is_dir) for every entry under root (root itself excluded); links not followed"""
    stack = [LP + os.path.abspath(root) if not root.startswith(LP) else root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError:
            continue
        for e in entries:
            full = os.path.join(d, e.name)
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                yield strip_lp(full), False
                continue
            is_link = stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)  # reparse point
            is_dir = stat.S_ISDIR(st.st_mode)
            yield strip_lp(full), is_dir
            if is_dir and not is_link:
                stack.append(full)


def check(roots):
    hits, seen = [], set()
    n_files = n_dirs = 0
    longest_exe = longest_path = 0
    for root in roots:
        for p, is_dir in walk(root):
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            if is_dir:
                n_dirs += 1
            else:
                n_files += 1
            if not is_dir and p.lower().endswith(".exe"):
                b = bytes8(p)
                longest_exe = max(longest_exe, b)
                if b > EXE_MAX_BYTES:
                    hits.append("FAIL exe %d bytes > %d: %s" % (b, EXE_MAX_BYTES, p))
                continue
            u = units16(p)
            longest_path = max(longest_path, u)
            if u >= PATH_FAIL_UNITS:
                hits.append("FAIL path %d >= %d: %s" % (u, PATH_FAIL_UNITS, p))
    return hits, n_files, n_dirs, longest_exe, longest_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathlen_check_test.py")
        return subprocess.run([sys.executable, "-I", test]).returncode
    if not a.root:
        print("pathlen check: SETUP no --root given")
        return 2
    missing = [r for r in a.root if not os.path.isdir(LP + os.path.abspath(r))]
    if missing:
        print("pathlen check: SETUP root(s) not found: %s" % ", ".join(missing))
        return 2
    hits, nf, nd, le, lpth = check(a.root)
    for h in hits:
        print(h)
    if hits:
        print("pathlen check: FAIL (%d path(s))" % len(hits))
        return 1
    print("pathlen check: ok (%d files, %d folders; longest exe %d bytes (limit %d), longest path %d (fail at %d))"
          % (nf, nd, le, EXE_MAX_BYTES, lpth, PATH_FAIL_UNITS))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
