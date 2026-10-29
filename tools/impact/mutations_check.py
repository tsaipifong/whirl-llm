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


import re

# arch025-dtools D5.5: "maybe equivalent" without a reason is a hole in the evidence (x12-f4b-settokens had one). A mutant text may say
# "equivalent" only with its argument ("equivalent by design", "NOT equivalent by construction ...", "equivalent because ...").
HEDGE = re.compile(r"\b(maybe|possibly|probably|perhaps|might be)\s+(output[- ])?equivalent", re.I)


def unargued(muts):
    return [(m["id"], "hedged equivalence", m.get("file"), "says '" + HEDGE.search(m.get(k, "")).group(0) + "' with no argument")
            for m in muts for k in ("expect_test", "what") if HEDGE.search(m.get(k, ""))]


CARDS = ("r9700", "8060s")


def labels(muts, modules, extra=None):
    """arch025-n11b: a mutant's "module" must be a tools/gates/modules.json module or a declared non-product label
    (mutations.json "labels": {label: reason}); test_change.py's GPU evidence rule attributes kills by this label"""
    extra = extra or {}
    return [(m["id"], "label", m.get("file"), f"module {m.get('module')!r} is not in tools/gates/modules.json (nor in the catalog's \"labels\")")
            for m in muts if m.get("module") not in modules and m.get("module") not in extra]


def gfx1201_only(root):
    """kernels/*.hip files compiled into the gfx1201 (R9700) code object only: included by kernels/whirl_kernels.hip
    inside its `#if !defined(__gfx1151__)` block and not included ("../x.hip") by the gfx1151 set"""
    try:
        txt = open(os.path.join(root, "kernels", "whirl_kernels.hip"), encoding="utf-8").read().splitlines()
    except OSError:
        return set()
    out, inside = set(), False
    for ln in txt:
        s = ln.strip()
        if s.startswith("#if") and "!defined(__gfx1151__)" in s.replace(" ", ""):
            inside = True
        elif s.startswith("#endif"):
            inside = False
        elif inside:
            mm = re.match(r'#include\s+"([^"/]+\.hip)"', s)
            if mm:
                out.add("kernels/" + mm.group(1))
    d = os.path.join(root, "kernels", "gfx1151")
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        if f.endswith(".hip"):
            for mm in re.finditer(r'#include\s+"\.\./([^"/]+\.hip)"', open(os.path.join(d, f), encoding="utf-8", errors="replace").read()):
                out.discard("kernels/" + mm.group(1))
    return out


def cards_bad(muts, only1201=()):
    """arch025-n11b: "cards" (optional) = the cards a one-code-object mutant applies to; a non-empty list of r9700 / 8060s.
    A kernels/gfx1151/ file is compiled into the 8060S code object only, so its mutant must say cards ["8060s"]; a file
    of gfx1201_only() must say cards ["r9700"]."""
    out = []
    for m in muts:
        c = m.get("cards")
        if c is not None and not (isinstance(c, list) and c and len(set(c)) == len(c) and all(x in CARDS for x in c)):
            out.append((m["id"], "cards", m.get("file"), f"cards {c!r}: want a non-empty list of {', '.join(CARDS)}"))
        elif (m.get("file") or "").startswith("kernels/gfx1151/") and c != ["8060s"]:
            out.append((m["id"], "cards", m.get("file"), "a kernels/gfx1151/ file is the 8060S code object only: cards [\"8060s\"]"))
        elif m.get("file") in only1201 and c != ["r9700"]:
            out.append((m["id"], "cards", m.get("file"), "a gfx1201-only kernel file (kernels/whirl_kernels.hip !__gfx1151__ block): cards [\"r9700\"]"))
    return out


def equiv_bad(muts):
    """arch025-n11b: "equivalent" (optional) = the written argument why no test can kill the mutant (it is then left out
    of the GPU evidence rule); it must be an argument (>= 40 characters) and the mutant must expect no catching test"""
    out = []
    for m in muts:
        if "equivalent" not in m:
            continue
        e = m.get("equivalent")
        if not (isinstance(e, str) and len(e.strip()) >= 40):
            out.append((m["id"], "equivalent", m.get("file"), "equivalent needs its argument (>= 40 characters)"))
        elif not str(m.get("expect_test", "")).lower().startswith("none"):
            out.append((m["id"], "equivalent", m.get("file"), f"equivalent but expect_test names a catcher: {m.get('expect_test')!r}"))
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
    cases.append(("hedged 'maybe equivalent' is flagged", [x[0] for x in unargued([{"id": "h1", "expect_test": "GPU: golden (maybe equivalent in output)"}])] == ["h1"]))
    cases.append(("'possibly output-equivalent' is flagged (what text)", [x[0] for x in unargued([{"id": "h2", "what": "possibly output equivalent"}])] == ["h2"]))
    cases.append(("an argued claim passes", unargued([{"id": "ok1", "expect_test": "equivalent by design: WHIRL_MODE unset"}, {"id": "ok2", "expect_test": "NOT equivalent by construction"}]) == []))
    # arch025-n11b: the module label must be a tools/gates/modules.json module (or a declared non-product label): the
    # GPU evidence rule attributes kills by it (hiprt-* named hip.runtime_loader, a module that no longer exists)
    MODS = {"hip.runtime": {}, "kernels.common": {}}
    got = [x[0] for x in labels([{"id": "l1", "module": "hip.runtime_loader"}, {"id": "l2", "module": "hip.runtime"},
                                 {"id": "l3", "module": "gates"}], MODS, {"gates": "tools/gates scripts"})]
    cases.append(("label not in modules.json is flagged; module / declared label pass", got == ["l1"]))
    # cards: a non-empty subset of the two cards; a kernels/gfx1151/ file is the 8060S code object only
    got = {x[0]: x[3] for x in cards_bad([{"id": "c1", "file": "kernels/common.hip", "cards": ["r9700"]},
                                          {"id": "c2", "file": "kernels/common.hip", "cards": []},
                                          {"id": "c3", "file": "kernels/common.hip", "cards": ["gfx1201"]},
                                          {"id": "c4", "file": "kernels/gfx1151/quant.hip"},
                                          {"id": "c5", "file": "kernels/gfx1151/quant.hip", "cards": ["8060s"]},
                                          {"id": "c6", "file": "src/a.cpp", "cards": "r9700"}])}
    cases.append(("cards: valid one-card list passes", "c1" not in got and "c5" not in got))
    cases.append(("cards: empty / unknown card / not a list flagged", {"c2", "c3", "c6"} <= set(got)))
    cases.append(("cards: kernels/gfx1151/ file without cards [8060s] flagged", "c4" in got))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "kernels", "gfx1151"))
        open(os.path.join(d, "kernels", "whirl_kernels.hip"), "w").write(
            '#include "smoke.hip"\n#if !defined(__gfx1151__)\n#include "common.hip"\n#include "moe_f.hip"\n#endif\n#include "dflash.hip"\n')
        open(os.path.join(d, "kernels", "gfx1151", "w.hip"), "w").write('#include "common.hip"\n#include "../moe_f.hip"\n')
        only = gfx1201_only(d)
        cases.append(("gfx1201_only: the !__gfx1151__ block minus files the gfx1151 set includes", only == {"kernels/common.hip"}))
        got = [x[0] for x in cards_bad([{"id": "g1", "file": "kernels/common.hip"}, {"id": "g2", "file": "kernels/common.hip", "cards": ["r9700"]},
                                        {"id": "g3", "file": "kernels/moe_f.hip"}, {"id": "g4", "file": "kernels/dflash.hip"}], only)]
        cases.append(("cards: a gfx1201-only file without cards [r9700] flagged; shared files pass", got == ["g1"]))
    # equivalent: an argument (>= 40 chars) and no test expected to catch it
    got = [x[0] for x in equiv_bad([{"id": "e1", "equivalent": "yes", "expect_test": "none"},
                                    {"id": "e2", "equivalent": "a static_assert whose condition holds generates no code", "expect_test": "x:testY"},
                                    {"id": "e3", "equivalent": "a static_assert whose condition holds generates no code", "expect_test": "none: equivalent"}])]
    cases.append(("equivalent: short argument or a test expectation flagged; argued + none passes", got == ["e1", "e2"]))
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
        cat = json.load(open(spec, encoding="utf-8"))
        muts = cat["mutations"]
        mods = json.load(open(os.path.join(a.root, "tools", "gates", "modules.json"), encoding="utf-8"))["modules"]
    except (OSError, ValueError, KeyError) as e:
        print(f"mutations_check: cannot read {spec} / tools/gates/modules.json: {e}")
        return 2
    bad = stale(a.root, muts) + unargued(muts) + labels(muts, mods, cat.get("labels")) + cards_bad(muts, gfx1201_only(a.root)) + equiv_bad(muts)
    for i, k, f, why in bad:
        print(f"STALE {i} ({k}, {f}): {why}")
    print(f"mutations check: {'ok' if not bad else 'FAIL'} ({len(muts) - len(bad)}/{len(muts)} mutants apply)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
