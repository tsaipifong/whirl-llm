# SPDX-License-Identifier: Apache-2.0
"""Reverse text proof P1 of a behaviour-preserving move step (arch025-engine S0; DESIGN 5 P1, review F11 / R1).

For every entry of the OLD tree's engine census (tools/gates/engine_split_census.py: Engine and the cut's component
classes, free functions, nested types), find the same entry in the NEW tree's census, map the new body back with the
step's rules (`SlotPages::` -> `Engine::`, `sc_.` -> ``, renamed entries ...), drop comments and whitespace, and compare
the two bodies token by token (string literals included, so a changed log text is a difference). Every difference
left must be listed in the rules' allowed_diffs; an entry missing in the new tree must be listed in removed.

  python -I tools/impact/revtext.py --old <tree> --new <tree> [--rules step_rules.json] [--cut tools/gates/engine_cut.json]
                                    [--work DIR]
  python -I tools/impact/revtext.py --selftest [--work DIR]     fault cases on this repo's tree
rules (JSON): {"replace": [[new_text, old_text], ...], "regex": [[pattern, repl], ...],
               "renames": {"new entry": "old entry"}, "removed": {"old entry": "why"},
               "allowed_diffs": [{"entry": e, "old": "<token text>", "new": "<token text>", "why": "..."}]}
Exit 0 "revtext: ok (N entries identical after the rules, K allowed differences)", 1 a difference, 2 usage / crash.
"""
import json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CENSUS = os.path.join(REPO, "tools", "gates", "engine_split_census.py")
TOK = re.compile(r'R"([^(]*)\((?:.|\n)*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d[\w.]*|::|->|<<=|>>=|[-+*/%&|^!=<>]=|&&|\|\||\+\+|--|<<|>>|\S')


def arg(n, d=None):
    a = sys.argv[1:]
    return a[a.index(n) + 1] if n in a else d


def strip_comments(s):
    out, i, n = [], 0, len(s)
    while i < n:
        if s.startswith("//", i):
            j = s.find("\n", i); i = n if j < 0 else j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2); i = n if j < 0 else j + 2; out.append(" ")
        elif s[i] in "\"'":
            q, j = s[i], i + 1
            while j < n and s[j] != q:
                j += 2 if s[j] == "\\" else 1
            out.append(s[i:j + 1]); i = j + 1
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def census(tree, out, cut):
    r = subprocess.run([sys.executable, "-I", CENSUS, tree, out, "--cut", cut], capture_output=True, text=True)
    if r.returncode not in (0, 2):
        raise SystemExit(f"revtext: census crashed on {tree}: {r.stderr[-400:]}")
    rows = {}
    for l in list(open(os.path.join(out, "funcs.tsv"), encoding="utf-8"))[1:]:
        p = l.rstrip("\n").split("\t")
        rows.setdefault(p[0].split("#")[0], []).append((p[1], int(p[2]), int(p[3]), p[8]))
    return rows


def body_text(tree, rows):
    """raw text (comments dropped, strings kept) of every part of an entry, in census order"""
    parts = []
    for f, ln, n, dfn in sorted(rows, key=lambda x: (x[0], x[1])):
        if "in-class initializer" in dfn:   # synthetic: the class body's initializers (checked by the member-order pin)
            continue
        lines = open(os.path.join(tree, "src", f), encoding="utf-8", errors="replace").read().split("\n")
        parts.append("\n".join(lines[ln - 1:ln - 1 + n]))
    return strip_comments("\n".join(parts))


def tokens(s):
    return [m.group(0) for m in TOK.finditer(s)]


def run(old, new, rules, cut, work):
    os.makedirs(work, exist_ok=True)
    ro, rn = census(old, os.path.join(work, "old"), cut), census(new, os.path.join(work, "new"), cut)
    ren = rules.get("renames", {})
    back = {ren.get(k, k): k for k in rn}
    allowed = list(rules.get("allowed_diffs", []))
    used, bad, same = [], [], 0
    for e in sorted(ro):
        if e not in back:
            if e in rules.get("removed", {}): continue
            bad.append(f"MISSING {e}: not found in the new tree (not in rules.removed)"); continue
        to = tokens(body_text(old, ro[e]))
        nt = body_text(new, rn[back[e]])
        for a, b in rules.get("replace", []): nt = nt.replace(a, b)
        for p, r in rules.get("regex", []): nt = re.sub(p, r, nt)
        tn = tokens(nt)
        if to == tn:
            same += 1; continue
        # allowed differences of this entry: replace the new token text by the old one, then compare again
        jo, jn = " ".join(to), " ".join(tn)
        for ad in allowed:
            if ad.get("entry") in (e, "*") and ad.get("new") and ad["new"] in jn:
                jn = jn.replace(ad["new"], ad.get("old", ""), 1); used.append(ad)
        if " ".join(jn.split()) == " ".join(jo.split()):
            continue
        k = next((i for i, (x, y) in enumerate(zip(jo.split(), jn.split())) if x != y), min(len(jo.split()), len(jn.split())))
        ctx_o, ctx_n = " ".join(jo.split()[max(0, k - 6):k + 8]), " ".join(jn.split()[max(0, k - 6):k + 8])
        bad.append(f"DIFF {e} at token {k}: old [{ctx_o}] new [{ctx_n}]")
    unused = [a for a in allowed if a not in used]
    for l in bad: print("  " + l)
    for a in unused: print(f"  note: allowed_diff not used: {a.get('entry')}: {a.get('why', '')}")
    if bad:
        print(f"revtext: FAIL ({len(bad)} entries differ / missing, {same} identical)"); return 1
    print(f"revtext: ok ({same} entries identical after the rules, {len(used)} allowed differences)"); return 0


def selftest(work):
    cut = os.path.join(REPO, "tools", "gates", "engine_cut.json")
    eh = "server/engine_pages.cpp"
    src = open(os.path.join(REPO, "src", eh), encoding="utf-8").read()
    m = re.search(r"void Engine::dropCache\(Slot& sl\) \{\n((?:    [^\n]*\n){2})", src)
    cases = []

    def tree(name, text, extra=None):
        d = os.path.join(work, re.sub(r"\W+", "_", name))
        if os.path.isdir(d): shutil.rmtree(d)
        for sub in ("src", "include"):
            shutil.copytree(os.path.join(REPO, sub), os.path.join(d, sub))
        open(os.path.join(d, "src", eh), "w", encoding="utf-8").write(text)
        if extra: extra(d)
        return d

    def case(name, text, rules, want, extra=None):
        d = tree(name, text, extra)
        print(f"-- {name}")
        rc = run(REPO, d, rules, cut, d + "_w")
        cases.append((name, rc == want))

    case("identity", src, {}, 0)
    if m:
        a, b = m.group(1).split("\n")[:2]
        case("two statements of dropCache swapped", src.replace(m.group(1), b + "\n" + a + "\n", 1), {}, 1)
    else:
        cases.append(("dropCache anchor", False))
    lg = re.search(r'\n(    [^\n]*log[IWE]\("[^\n]*\n)', src)
    if lg:
        case("a log line deleted", src.replace(lg.group(1), "", 1), {}, 1)
        t2 = src.replace(lg.group(1), lg.group(1).replace('("', '("x', 1), 1)
        case("a log text changed (string literal)", t2, {}, 1)
        lit = re.search(r'\("([^"]*)"', lg.group(1)).group(1)
        case("the same change listed in allowed_diffs", t2,
             {"allowed_diffs": [{"entry": "*", "old": '"' + lit + '"', "new": '"x' + lit + '"', "why": "selftest"}]}, 0)
    else:
        cases.append(("log anchor", False))
    # a move with rules: dropCache becomes SlotPages::dropCache with this-> -> sc_. (rules map it back)
    mv = src.replace("void Engine::dropCache(Slot& sl) {", "void SlotPages::dropCache(Slot& sl) {", 1)
    case("a move SlotPages::dropCache with the mapping rule", mv, {"replace": [["SlotPages::dropCache", "Engine::dropCache"]]}, 0)
    case("the same move without the rule", mv, {}, 1)
    ok = all(g for _, g in cases)
    for n, g in cases: print(f"  {'ok  ' if g else 'FAIL'} {n}")
    print(f"revtext selftest: {sum(g for _, g in cases)}/{len(cases)} as expected")
    return 0 if ok else 1


def main():
    work = arg("--work", os.path.join(REPO, "build", "revtext"))
    if "--selftest" in sys.argv: return selftest(work)
    old, new = arg("--old"), arg("--new")
    if not old or not new: print(__doc__); return 2
    rules = json.load(open(arg("--rules"), encoding="utf-8")) if arg("--rules") else {}
    return run(old, new, rules, arg("--cut", os.path.join(REPO, "tools", "gates", "engine_cut.json")), work)


if __name__ == "__main__":
    sys.exit(main())
