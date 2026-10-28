#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find (and optionally merge) identical top-level code between the two kernel sets (arch025 N5).

A kernel source is split into top-level units: one declaration / definition (from its first code line to the line
where braces and parentheses close at depth 0 with `}` or `;`), or one preprocessor directive at depth 0. Comments and
blank lines before a unit belong to it as leading trivia but are not part of its key; the key is the code with comments
removed and whitespace collapsed. Runs of consecutive units with equal keys in a gfx1201 file (kernels/*.hip) and a
gfx1151 file (kernels/gfx1151/*.hip) are the same code token for token.

    kshare.py scan [--min N]            list the runs (>= N code lines) that are still duplicated; exit 1 if any
    kshare.py merge [--min N] [--dry]   move every run into kernels/shared/<stem>_<ident>.h and #include it at the
                                        run's place in both files (the preprocessed translation unit of each arch
                                        stays token-identical: prove it with kco.sh pp / the code objects)

Units are compared only between the sets (a run inside one set is one translation unit and would be a redefinition).
Preprocessor conditionals must balance inside a run, or the run is shrunk until they do.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(REPO, "kernels")
SHARED = "shared"


class Unit:
    __slots__ = ("start", "code_start", "end", "key", "ncode", "pp")

    def __init__(self, start, code_start, end, key, ncode, pp):
        self.start, self.code_start, self.end, self.key, self.ncode, self.pp = start, code_start, end, key, ncode, pp


def strip_comments(lines):
    """per line: the code with comments and string / char contents kept verbatim except comments removed"""
    out = []
    in_block = False
    for ln in lines:
        res = []
        i = 0
        n = len(ln)
        while i < n:
            if in_block:
                j = ln.find("*/", i)
                if j < 0:
                    i = n
                else:
                    in_block = False
                    i = j + 2
                    res.append(" ")
                continue
            c = ln[i]
            if c == "/" and i + 1 < n and ln[i + 1] == "/":
                break
            if c == "/" and i + 1 < n and ln[i + 1] == "*":
                in_block = True
                i += 2
                continue
            if c in "\"'":
                j = i + 1
                while j < n and ln[j] != c:
                    j += 2 if ln[j] == "\\" else 1
                res.append(ln[i:j + 1])
                i = j + 1
                continue
            res.append(c)
            i += 1
        out.append("".join(res))
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def split_units(lines):
    code = strip_comments(lines)
    units = []
    i = 0
    n = len(lines)
    depth = 0
    paren = 0
    start = 0
    code_start = None
    keyparts = []
    ncode = 0
    pp_cont = False
    while i < n:
        c = code[i]
        cs = c.strip()
        if code_start is None and depth == 0 and paren == 0:
            if not cs and not pp_cont:
                i += 1
                continue
            if cs.startswith("#") or pp_cont:
                # preprocessor directive at top level (with continuation lines)
                j = i
                parts = []
                while True:
                    parts.append(code[j].rstrip())
                    if not code[j].rstrip().endswith("\\") or j + 1 >= n:
                        break
                    j += 1
                units.append(Unit(start, i, j, norm(" ".join(parts)), j - i + 1, True))
                i = j + 1
                start = i
                continue
            code_start = i
        if cs:
            ncode += 1
            keyparts.append(c)
        # strings were kept; count braces outside them
        for m in re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}()]', c):
            t = m.group(0)
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
            elif t == "(":
                paren += 1
            elif t == ")":
                paren -= 1
        if depth == 0 and paren == 0 and cs and (cs.endswith("}") or cs.endswith(";") or cs.endswith("};")) \
                and not cs.startswith("#"):
            units.append(Unit(start, code_start, i, norm(" ".join(keyparts)), ncode, False))
            start = i + 1
            code_start = None
            keyparts = []
            ncode = 0
        i += 1
    if code_start is not None:
        units.append(Unit(start, code_start, n - 1, norm(" ".join(keyparts)), ncode, False))
    return units


def pp_balance(units):
    d = 0
    for u in units:
        if not u.pp:
            continue
        k = u.key
        if re.match(r"#\s*(if|ifdef|ifndef)\b", k):
            d += 1
        elif re.match(r"#\s*endif\b", k):
            d -= 1
            if d < 0:
                return False
    return d == 0


def ident_of(units, lines=None):
    """a readable name for a run: its first kernel, else its first device function, else macro / struct"""
    keys = [u.key for u in units if not u.pp]
    pats = (r"__global__ void (?:__launch_bounds__ ?\([^)]*\) ?)?(\w+) ?\(",
            r"__device__[^(;{]*?([a-z_]\w*) ?\(",
            r"(?:struct|union|class) (\w+)")
    for pat in pats:
        for k in keys:
            m = re.search(pat, k)
            if m and m.group(1) not in ("__launch_bounds__", "__attribute__", "if", "for", "while", "sizeof"):
                return m.group(1)
    for u in units:
        m = re.match(r"# ?define (\w+)", u.key)
        if m:
            return m.group(1).lower()
    return "part"


def load(path):
    with open(path, encoding="utf-8", newline="") as fh:
        s = fh.read()
    nl = "\r\n" if "\r\n" in s else "\n"
    return s.split(nl), nl


def files():
    a = sorted(os.path.join(KDIR, f) for f in os.listdir(KDIR) if f.endswith(".hip"))
    b = sorted(os.path.join(KDIR, "gfx1151", f) for f in os.listdir(os.path.join(KDIR, "gfx1151")) if f.endswith(".hip"))
    return a, b


def find_runs(min_code):
    fa, fb = files()
    data = {}
    for p in fa + fb:
        lines, nl = load(p)
        data[p] = (lines, nl, split_units(lines))
    used = {p: set() for p in data}
    cands = []
    for pa in fa:
        ua = data[pa][2]
        for pb in fb:
            ub = data[pb][2]
            sm = difflib.SequenceMatcher(None, [u.key for u in ua], [u.key for u in ub], autojunk=False)
            for blk in sm.get_matching_blocks():
                if blk.size == 0:
                    continue
                # trim leading / trailing preprocessor-only units and shrink until conditionals balance
                lo, hi = 0, blk.size
                while True:
                    while lo < hi and ua[blk.a + lo].pp and not pp_balance(ua[blk.a + lo:blk.a + hi]):
                        lo += 1
                    while hi > lo and ua[blk.a + hi - 1].pp and not pp_balance(ua[blk.a + lo:blk.a + hi]):
                        hi -= 1
                    if lo >= hi or pp_balance(ua[blk.a + lo:blk.a + hi]):
                        break
                    hi -= 1
                if lo >= hi:
                    continue
                run_a = ua[blk.a + lo:blk.a + hi]
                nc = sum(u.ncode for u in run_a if not u.pp)
                if nc < min_code:
                    continue
                cands.append((nc, pa, blk.a + lo, blk.a + hi, pb, blk.b + lo, blk.b + hi))
    cands.sort(key=lambda c: -c[0])
    runs = []
    for nc, pa, a0, a1, pb, b0, b1 in cands:
        ia = set(range(a0, a1))
        ib = set(range(b0, b1))
        if ia & used[pa] or ib & used[pb]:
            continue
        used[pa] |= ia
        used[pb] |= ib
        runs.append((nc, pa, a0, a1, pb, b0, b1))
    return data, runs


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("scan", "merge"))
    ap.add_argument("--min", type=int, default=8, help="minimum code lines of a run")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    data, runs = find_runs(a.min)
    total = sum(r[0] for r in runs)
    for nc, pa, a0, a1, pb, b0, b1 in runs:
        ua, ub = data[pa][2], data[pb][2]
        print(f"{nc:5d}  {rel(pa)}:{ua[a0].code_start + 1}-{ua[a1 - 1].end + 1}  ==  "
              f"{rel(pb)}:{ub[b0].code_start + 1}-{ub[b1 - 1].end + 1}")
    print(f"runs: {len(runs)}, code lines: {total}")
    if a.mode == "scan":
        return 1 if runs else 0
    # merge: edit bottom-up per file so earlier line numbers stay valid
    sdir = os.path.join(KDIR, SHARED)
    os.makedirs(sdir, exist_ok=True)
    edits = {}  # path -> list of (first_line, last_line, replacement)
    names = set(os.listdir(sdir))
    for nc, pa, a0, a1, pb, b0, b1 in runs:
        la, nla, ua = data[pa]
        lb, nlb, ub = data[pb]
        stem = os.path.splitext(os.path.basename(pa))[0]
        base = f"{stem}_{ident_of(ua[a0:a1], la)}"
        name = base + ".h"
        k = 2
        while name in names:
            name = f"{base}_{k}.h"
            k += 1
        names.add(name)
        s0, s1 = ua[a0].code_start, ua[a1 - 1].end
        t0, t1 = ub[b0].code_start, ub[b1 - 1].end
        body = la[s0:s1 + 1]
        hdr = [f"// WHIRL device code, shared by both kernel sets (gfx1201 {rel(pa)}, gfx1151 {rel(pb)}).",
               "// SPDX-License-Identifier: Apache-2.0",
               "// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,",
               "// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).",
               ""] + body + [""]
        inc = f'#include "{SHARED}/{name}"  // shared with {rel(pb) if True else ""}'
        if not a.dry:
            with open(os.path.join(sdir, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(nla.join(hdr))
        edits.setdefault(pa, []).append((s0, s1, f'#include "{SHARED}/{name}"  // also in the gfx1151 set ({rel(pb)})'))
        edits.setdefault(pb, []).append((t0, t1, f'#include "{SHARED}/{name}"  // also in the gfx1201 set ({rel(pa)})'))
        del inc
    if a.dry:
        return 0
    for p, ed in edits.items():
        lines, nl, _ = data[p]
        for s0, s1, r in sorted(ed, key=lambda e: -e[0]):
            lines[s0:s1 + 1] = [r]
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(nl.join(lines))
        print("edited", rel(p), len(ed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
