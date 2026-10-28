# SPDX-License-Identifier: Apache-2.0
"""Layering ratchet (arch025 D11 N2): no new cross-module include and no new include cycle without declaring it.

Builds the include graph of the product files (src/, include/, kernels/; quoted includes) with one node per module of
tools/gates/modules.json and one node per file that is in no module ("file:<path>"). tools/gates/layering_baseline.json
lists the declared cross-module edges and the allowed cycles (strongly connected components). It fails on an edge that
is not declared and on a cycle that is not inside an allowed one; edges that disappeared are printed so the baseline
only shrinks. --update rewrites the baseline from the tree (the main session, after reviewing the printed new edges).

    python tools/gates/layering_check.py [--root DIR] [--modules FILE] [--baseline FILE] [--update]
"""
import argparse
import collections
import fnmatch
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOPS = ("src", "include", "kernels")
EXTS = (".cpp", ".h", ".hpp", ".hip", ".inc", ".c", ".cu", ".cuh")
INC = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def product_files(root):
    out = []
    for t in TOPS:
        for d, dirs, files in os.walk(os.path.join(root, t)):
            dirs[:] = [x for x in dirs if x not in (".git", "build", "__pycache__")]
            for f in files:
                if f.endswith(EXTS):
                    out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(out)


def graph(root, mods):
    files = product_files(root)
    fileset = set(files)
    byname = collections.defaultdict(list)
    for f in files:
        byname[os.path.basename(f)].append(f)
    owner = {}
    for f in files:
        own = [n for n, d in mods.items() if any(fnmatch.fnmatch(f, g) for g in d.get("paths") or [])]
        owner[f] = own[0] if own else "file:" + f

    def resolve(src, inc):
        for c in [os.path.join(os.path.dirname(src), inc)] + [os.path.join(b, inc) for b in ("include", "src", "", "kernels")]:
            c = os.path.normpath(c).replace(os.sep, "/")
            if c in fileset:
                return c
        b = byname.get(os.path.basename(inc), [])
        return b[0] if len(b) == 1 else None

    edges = {}
    for f in files:
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        for m in INC.finditer(text):
            r = resolve(f, m.group(1))
            if r and owner[f] != owner[r]:
                line = text.count("\n", 0, m.start()) + 1
                edges.setdefault((owner[f], owner[r]), f"{f}:{line} -> {r}")
    return edges


def sccs(edges):
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj.setdefault(b, set())
    index, low, onst, st, out, counter = {}, {}, set(), [], [], [0]
    for start in sorted(adj):
        if start in index:
            continue
        work = [(start, iter(sorted(adj[start])))]
        index[start] = low[start] = counter[0]
        counter[0] += 1
        st.append(start)
        onst.add(start)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is None:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        x = st.pop()
                        onst.discard(x)
                        comp.append(x)
                        if x == v:
                            break
                    if len(comp) > 1:
                        out.append(sorted(comp))
            elif w not in index:
                index[w] = low[w] = counter[0]
                counter[0] += 1
                st.append(w)
                onst.add(w)
                work.append((w, iter(sorted(adj[w]))))
            elif w in onst:
                low[v] = min(low[v], index[w])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--modules", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    mpath = a.modules or os.path.join(a.root, "tools", "gates", "modules.json")
    bpath = a.baseline or os.path.join(a.root, "tools", "gates", "layering_baseline.json")
    try:
        mods = json.load(open(mpath, encoding="utf-8"))["modules"]
    except (OSError, ValueError, KeyError) as e:
        print(f"layering: FAIL: cannot read {mpath}: {e}")
        return 1
    edges = graph(a.root, mods)
    cyc = sccs(edges)
    if a.update:
        json.dump({"_doc": ["Declared cross-module include edges and allowed include cycles (arch025 D11 N2). Written by "
                            "tools/gates/layering_check.py --update after the main session reviewed the new edges; the "
                            "check fails on any edge or cycle not listed here."],
                   "edges": sorted([x, y] for x, y in edges), "cycles": cyc},
                  open(bpath, "w", encoding="utf-8", newline="\n"), indent=1)
        print(f"layering: baseline written ({len(edges)} edges, {len(cyc)} cycle(s)) -> {bpath}")
        return 0
    try:
        base = json.load(open(bpath, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"layering: FAIL: cannot read {bpath}: {e}")
        return 1
    known = {tuple(e) for e in base.get("edges") or []}
    allowed = [set(c) for c in base.get("cycles") or []]
    errs = []
    for (x, y), where in sorted(edges.items()):
        if (x, y) not in known:
            errs.append(f"new cross-module include {x} -> {y} ({where}): declare it (layering_check.py --update after review) or remove it")
    for c in cyc:
        if not any(set(c) <= al for al in allowed):
            errs.append(f"include cycle not allowed: {', '.join(c)}")
    gone = sorted(known - set(edges))
    for x, y in gone:
        print(f"   gone {x} -> {y} (the baseline can shrink: --update)")
    for e in errs:
        print("   FAIL " + e)
    print(f"layering: {'ok' if not errs else f'FAIL ({len(errs)} problem(s))'} ({len(edges)} edges, {len(cyc)} cycle(s), "
          f"{len(gone)} baseline edge(s) gone)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
