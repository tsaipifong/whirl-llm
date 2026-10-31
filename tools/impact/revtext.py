# SPDX-License-Identifier: Apache-2.0
"""Reverse text proof P1 of a behaviour-preserving move step (arch025-engine S0; DESIGN 5 P1, review F11 / R1).

For every entry of the OLD tree's engine census (tools/gates/engine_split_census.py: Engine and the cut's component
classes, free functions, nested types), find the same entry in the NEW tree's census, map the new body back with the
step's rules (`SlotPages::` -> `Engine::`, `sc_.` -> ``, renamed entries ...), drop comments and whitespace, and compare
the two bodies token by token (string literals included, so a changed log text is a difference). Every difference
left must be listed in the rules' allowed_diffs; an entry missing in the new tree must be listed in removed.

  python -I tools/impact/revtext.py --old <tree> --new <tree> [--rules step_rules.json] [--cut tools/gates/engine_cut.json]
                                    [--work DIR]
  python -I tools/impact/revtext.py --selftest [--work DIR] [--only REGEX]     fault cases on this repo's tree (--only: the
                                    cases whose name matches, for red / green runs; acceptance is the full run)
rules (JSON; keys starting with "_" are comments, any other unknown key is a rules error, exit 2):
  "replace": [[new_text, old_text], ...]   applied to the NEW text (comments already dropped), in order, left to right,
        non-overlapping, ON TOKEN BOUNDARIES (arch025-revtextfix, S0 review C-6): new_text is tokenized with the same
        tokenizer and matches only a whole run of tokens (whitespace between them free); a string / char literal is one
        token, so a rule never rewrites inside a literal, and `dc_.` never cuts the identifier `sdc_`. old_text is put
        in its place as text (a space is added where it would glue two identifiers), so the regex rules still see text.
  "regex": [[pattern, repl], ...]          applied after replace, to the text (Python re; the rule writer owns it).
  "renames": {"new entry": "old entry"}    one to one: every value at most once, and no other new entry may already be
        called that old name (two new entries onto one old entry = a copy: declare the second one in "added"); the
        key must be a new-tree entry and the value an old-tree entry.
  "removed": {"old entry": "why"}          old entries that are gone; listing an entry the new tree still has fails.
  "added": {"new entry": {"equals": "<token text>", "sig": "<token text>" | "is": "<token text>" | "planned": true, "why": "..."}}
        entries ONLY in the new tree (functions, `member:<name>` declarations). Every new-tree entry that maps to no
        old entry must be listed here, or the run fails (`NEW <entry>`). After the replace / regex rules:
          equals  - a function: the tokens inside its body braces AND (sig, required for a function; arch025-engine-rt2,
                    S6 review act-on 1) the tokens before the body brace: return type, name, parameters, qualifiers
                    (out of class: the definition's line, e.g. "void SlotPages::f(Slot& sl)"); a member: its whole
                    declaration statement (no sig)
          is      - all tokens of the entry (signature + body, or the declaration)
          planned - the name is a planned row of the cut (new_funcs `<name>` / `Engine.<name>`; new_data for members);
                    use it only for code the step writes new, the review reads it (the text is not checked)
        e.g. S6: "added": {"noteSpeNew": {"equals": "stat_spe_new_ += 1;", "sig": "void noteSpeNew()", "why": "the former statement of createSpe"}}
        An in-class entry starts at its first code character (census, rt2), so `is` works on a one-line inline method.
  "allowed_diffs": [{"entry": "<entry>" | ["<entry>", ...], "old": "<token text>", "new": "<token text>", "why": "..."}]
        a difference left after the rules: in each named entry the first whole-token run equal to `new` is put back
        to `old` (once per entry), then the entry is compared again. The entry must be named: "*" or any wildcard is a
        rules error (S0b review C-4: `"*", false -> true` would apply once in every member entry). `new` must not
        be empty, `why` must be given.
Exit 0 "revtext: ok (N entries identical after the rules, K allowed differences, A added entries as declared)",
1 a difference / missing / new entry, 2 usage / rules error / crash.
"""
import contextlib, io, json, os, re, shutil, subprocess, sys

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
    # S0b (S0 review act-on 4): every data member declaration of a tracked class is an entry "member:<name>" (its whole
    # statement: type, annotations, in-class initializer), so `vram_idle_s_ = 60.0` -> `30.0` or a type change is a DIFF;
    # a member moved into a component keeps its entry name (rules.renames maps a renamed one back). A census without
    # member_decls.tsv fails (the comparison would silently skip the declarations).
    md = os.path.join(out, "member_decls.tsv")
    if not os.path.exists(md):
        raise SystemExit(f"revtext: census of {tree} wrote no member_decls.tsv (old census?): FAIL")
    for l in list(open(md, encoding="utf-8"))[1:]:
        c, n, f, ln, a0, a1 = l.rstrip("\n").split("\t")
        rows.setdefault("member:" + n, []).append((f, int(ln), -1, f"decl {c}::{n} @{a0}:{a1}"))
    return rows


def body_text(tree, rows):
    """raw text (comments dropped, strings kept) of every part of an entry, in census order"""
    parts = []
    for f, ln, n, dfn in sorted(rows, key=lambda x: (x[0], x[1], x[3])):
        if "in-class initializer" in dfn:   # synthetic: the initializers are compared as member:<name> entries (S0b)
            continue
        raw = open(os.path.join(tree, "src", f), encoding="utf-8", errors="replace").read()
        if n < 0:   # S0b: a member declaration, by offsets (the census's strip() keeps offsets)
            a0, a1 = (int(x) for x in dfn.rsplit("@", 1)[1].split(":"))
            parts.append(re.sub(r"^\s*(?:public|private|protected)\s*:", "", raw[a0:a1]))
            continue
        lines = raw.split("\n")
        # rt2: the census starts an in-class entry at its first code character; an access specifier on that same line
        # (`public: void f() {..}`) is not part of the entry (members drop it the same way)
        parts.append(re.sub(r"^\s*(?:(?:public|private|protected)\s*:(?!:)\s*)+", "", "\n".join(lines[ln - 1:ln - 1 + n])))
    return strip_comments("\n".join(parts))


def tokens(s):
    return [m.group(0) for m in TOK.finditer(s)]


class RulesError(Exception):
    pass


RULE_KEYS = {"replace", "regex", "renames", "removed", "added", "allowed_diffs"}


def check_rules(rules):
    """structural checks that need no tree (arch025-revtextfix); raises RulesError"""
    if not isinstance(rules, dict): raise RulesError("rules must be a JSON object")
    bad = sorted(k for k in rules if not k.startswith("_") and k not in RULE_KEYS)
    if bad: raise RulesError(f"unknown key(s) {bad} (known: {sorted(RULE_KEYS)}; '_...' keys are comments)")
    for i, pr in enumerate(rules.get("replace", [])):
        if not (isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, str) for x in pr)):
            raise RulesError(f"replace[{i}] must be [new_text, old_text]")
        if not tokens(pr[0]): raise RulesError(f"replace[{i}]: new_text {pr[0]!r} has no tokens")
    for i, pr in enumerate(rules.get("regex", [])):
        if not (isinstance(pr, list) and len(pr) == 2): raise RulesError(f"regex[{i}] must be [pattern, repl]")
    ren = rules.get("renames", {})
    if not isinstance(ren, dict): raise RulesError("renames must be an object {new entry: old entry}")
    seen = {}
    for k, v in ren.items():
        if not isinstance(v, str) or not v: raise RulesError(f"renames[{k!r}] must name an old entry")
        if v in seen: raise RulesError(f"renames are not one to one: {seen[v]!r} and {k!r} both rename to {v!r}")
        seen[v] = k
    if not isinstance(rules.get("removed", {}), dict): raise RulesError("removed must be an object {old entry: why}")
    added = rules.get("added", {})
    if not isinstance(added, dict): raise RulesError("added must be an object {new entry: {...}}")
    for k, v in added.items():
        if not isinstance(v, dict): raise RulesError(f"added[{k!r}] must be an object")
        modes = [m for m in ("equals", "is", "planned") if m in v]
        if len(modes) != 1: raise RulesError(f"added[{k!r}] needs exactly one of equals / is / planned (has {modes})")
        if "planned" in v and v["planned"] is not True: raise RulesError(f"added[{k!r}]: planned must be true")
        if not str(v.get("why", "")).strip(): raise RulesError(f"added[{k!r}] needs a why")
        if k in ren: raise RulesError(f"{k!r} is in both added and renames")
        # rt2 (S6 review act-on 1): equals on a function compares its body AND its signature (the tokens before the
        # body brace: return type, name, parameters, qualifiers); before, a function's equals was body-only
        if "sig" in v and not ("equals" in v and not k.startswith("member:")):
            raise RulesError(f"added[{k!r}]: sig belongs to a function's equals (is compares all tokens; a member's equals is its whole declaration)")
        if "equals" in v and not k.startswith("member:") and (not isinstance(v.get("sig"), str) or not tokens(v["sig"])):
            raise RulesError(f"added[{k!r}]: a function's equals needs sig (the tokens before the body brace, e.g. 'std::uint64_t speNew() const')")
    for i, ad in enumerate(rules.get("allowed_diffs", [])):
        ents = ad.get("entry")
        ents = [ents] if isinstance(ents, str) else ents
        if not ents or not isinstance(ents, list) or not all(isinstance(e, str) and e for e in ents):
            raise RulesError(f"allowed_diffs[{i}] needs an entry name (or a list of names)")
        wild = [e for e in ents if re.search(r"[*?\[\]]", e)]
        if wild: raise RulesError(f"allowed_diffs[{i}]: entry {wild} is a wildcard; name every entry it applies to")
        if not tokens(ad.get("new", "") or ""): raise RulesError(f"allowed_diffs[{i}] ({ents}): new is empty")
        if not str(ad.get("why", "")).strip(): raise RulesError(f"allowed_diffs[{i}] ({ents}) needs a why")


def _w(c):
    return bool(c) and (c.isalnum() or c == "_")


def tok_replace(text, a, b):
    """replace every whole-token run equal to tokens(a) by the text b (left to right, non-overlapping)"""
    ta = tokens(a)
    sp = [(m.start(), m.end(), m.group(0)) for m in TOK.finditer(text)]
    out, pos, i, k = [], 0, 0, len(ta)
    while i + k <= len(sp):
        if [x[2] for x in sp[i:i + k]] == ta:
            out.append(text[pos:sp[i][0]])
            prev = out[-1][-1:] if out[-1] else (out[-2][-1:] if len(out) > 1 else "")
            end = sp[i + k - 1][1]
            nxt = text[end:end + 1]
            out.append((" " if _w(prev) and _w(b[:1]) else "") + b + (" " if _w(b[-1:]) and _w(nxt) else ""))
            pos = end; i += k
        else:
            i += 1
    out.append(text[pos:])
    return "".join(out)


def tok_sub_once(seq, a, b):
    """first whole-token run of a in the token list seq -> b; (new list, True) or (seq, False)"""
    k = len(a)
    for i in range(len(seq) - k + 1):
        if seq[i:i + k] == a:
            return seq[:i] + b + seq[i + k:], True
    return seq, False


def apply_rules(text, rules):
    for a, b in rules.get("replace", []): text = tok_replace(text, a, b)
    for p, r in rules.get("regex", []): text = re.sub(p, r, text)
    return text


def body_tokens(tt, name):
    """tokens inside the body braces of function `name` (the first `{` after `name (`), or None"""
    sb = sig_body(tt, name)
    return None if sb is None else sb[1]


def sig_body(tt, name):
    """rt2: (the tokens before the body brace = the signature, the tokens inside the body braces) of function `name`"""
    short = name.split("::")[-1]
    for i in range(len(tt) - 1):
        if tt[i] == short and tt[i + 1] == "(":
            j = next((x for x in range(i, len(tt)) if tt[x] == "{"), None)
            if j is None: return None
            d = 0
            for x in range(j, len(tt)):
                d += (tt[x] == "{") - (tt[x] == "}")
                if d == 0: return tt[:j], tt[j + 1:x]
            return None
    return None


def planned_names(cut):
    try:
        c = json.load(open(cut, encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    for k in c.get("new_funcs", {}):
        if not k.startswith("_"): out.add(k.split(".", 1)[1] if k.startswith("Engine.") else k)
    for k in c.get("new_data", {}):
        if not k.startswith("_"): out.add("member:" + k)
    return out


def run(old, new, rules, cut, work, ro=None):
    """ro: the old tree's census rows when the caller already has them (the selftest compares many trees to one old)"""
    try:
        check_rules(rules)
    except RulesError as ex:
        print(f"revtext: rules error: {ex}"); return 2
    os.makedirs(work, exist_ok=True)
    if ro is None: ro = census(old, os.path.join(work, "old"), cut)
    rn = census(new, os.path.join(work, "new"), cut)
    ren, added, removed = rules.get("renames", {}), rules.get("added", {}), rules.get("removed", {})
    allowed = list(rules.get("allowed_diffs", []))
    used, bad, same, nadd = [], [], 0, 0
    # arch025-revtextfix (S0 review C-6): walk BOTH trees; the new -> old map is one to one
    back = {}   # old entry -> new entry
    for k in sorted(ren):
        if k not in rn: bad.append(f"RENAME {k} -> {ren[k]}: {k} is not an entry of the new tree")
        if ren[k] not in ro: bad.append(f"RENAME {k} -> {ren[k]}: {ren[k]} is not an entry of the old tree")
    for k in sorted(rn):
        if k in added:
            if k in ro: bad.append(f"ADDED {k}: the old tree has an entry of that name (compare it; renames if it moved)")
            continue
        o = ren.get(k, k)
        if o in back:
            bad.append(f"RENAME COLLISION {o}: new entries {back[o]} and {k} both map to it (one is a copy: list it in added)"); continue
        if o not in ro:
            if k not in ren: bad.append(f"NEW {k}: only in the new tree (not in rules.added)")
            continue
        back[o] = k
    for k in sorted(added):
        if k not in rn: bad.append(f"ADDED {k}: not an entry of the new tree")
    for e in sorted(removed):
        if e in back: bad.append(f"REMOVED {e}: the new tree still has it (as {back[e]})")
        elif e not in ro: bad.append(f"REMOVED {e}: not an entry of the old tree")
    for e in sorted(ro):
        if e not in back:
            if e in removed: continue
            bad.append(f"MISSING {e}: not found in the new tree (not in rules.removed)"); continue
        to = tokens(body_text(old, ro[e]))
        tn = tokens(apply_rules(body_text(new, rn[back[e]]), rules))
        if to == tn:
            same += 1; continue
        # allowed differences of this entry: the first whole-token run of `new` back to `old`, then compare again
        for ad in allowed:
            ents = [ad["entry"]] if isinstance(ad["entry"], str) else ad["entry"]
            if e in ents:
                tn, hit = tok_sub_once(tn, tokens(ad["new"]), tokens(ad.get("old", "") or ""))
                if hit: used.append(ad)
        if tn == to:
            continue
        k = next((i for i, (x, y) in enumerate(zip(to, tn)) if x != y), min(len(to), len(tn)))
        bad.append(f"DIFF {e} at token {k}: old [{' '.join(to[max(0, k - 6):k + 8])}] new [{' '.join(tn[max(0, k - 6):k + 8])}]")
    plan = planned_names(cut)
    for k in sorted(added):
        if k not in rn or k in ro: continue
        spec = added[k]
        if spec.get("planned"):
            if k in plan: nadd += 1
            else: bad.append(f"ADDED {k}: planned, but the cut ({os.path.basename(cut)}) has no new_funcs / new_data row {k}")
            continue
        tt = tokens(apply_rules(body_text(new, rn[k]), rules))
        parts = [r for r in rn[k] if "in-class initializer" not in r[3]]
        if "is" in spec:
            got, want, what = tt, tokens(spec["is"]), "all tokens"
        elif k.startswith("member:"):
            got, want, what = tt, tokens(spec["equals"]), "declaration"
        elif len(parts) != 1:
            bad.append(f"ADDED {k}: {len(parts)} census parts (overloads?): use is, not equals"); continue
        else:
            sb = sig_body(tt, k)
            if sb is not None and sb[0] != tokens(spec["sig"]):   # rt2: the signature (return type, parameters, const)
                g, want = sb[0], tokens(spec["sig"])
                j = next((i for i, (x, y) in enumerate(zip(want, g)) if x != y), min(len(want), len(g)))
                bad.append(f"ADDED {k}: the signature is not the declared text at token {j}: declared [{' '.join(want[max(0, j - 6):j + 8])}] "
                           f"new [{' '.join(g[max(0, j - 6):j + 8])}]"); continue
            got, want, what = (None if sb is None else sb[1]), tokens(spec["equals"]), "body"
        if got == want:
            nadd += 1
        else:
            g = got if got is not None else []
            j = next((i for i, (x, y) in enumerate(zip(want, g)) if x != y), min(len(want), len(g)))
            bad.append(f"ADDED {k}: the {what} is not the declared text at token {j}: declared [{' '.join(want[max(0, j - 6):j + 8])}] "
                       f"new [{' '.join(g[max(0, j - 6):j + 8]) if got is not None else '(no body found)'}]")
    unused = [a for a in allowed if a not in used]
    for l in bad: print("  " + l)
    for a in unused: print(f"  note: allowed_diff not used: {a.get('entry')}: {a.get('why', '')}")
    if bad:
        print(f"revtext: FAIL ({len(bad)} entries differ / missing / new, {same} identical)"); return 1
    print(f"revtext: ok ({same} entries identical after the rules, {len(used)} allowed differences, {nadd} added entries as declared)"); return 0


def selftest(work):
    cut = os.path.join(REPO, "tools", "gates", "engine_cut.json")
    eh = "server/engine_pages.cpp"
    src = open(os.path.join(REPO, "src", eh), encoding="utf-8").read()
    # arch025-engine S7: dropCache is SlotPages's since S7 (engine_pages.cpp); the cases anchor on the definition the
    # tree has (DC / DCLS), so they mean the same on trees before and after S7
    DCLS = next((c for c in ("SlotPages", "Engine") if f"void {c}::dropCache(Slot& sl) {{" in src), "Engine")
    DC = f"void {DCLS}::dropCache(Slot& sl) {{"
    m = re.search(re.escape(DC) + r"\n((?:    [^\n]*\n){2})", src)
    cases = []

    def tree(name, text, extra=None, rel=eh):
        d = os.path.join(work, re.sub(r"\W+", "_", name))
        if os.path.isdir(d): shutil.rmtree(d)
        for sub in ("src", "include"):
            shutil.copytree(os.path.join(REPO, sub), os.path.join(d, sub))
        open(os.path.join(d, "src", rel), "w", encoding="utf-8").write(text)
        if extra: extra(d)
        return d

    # revtextfix: the old tree is this repo for every case: one census (the selftest took ~45 s a case with two)
    ro = census(REPO, os.path.join(work, "_old"), cut)

    only = arg("--only")   # rt2: a regex on the case names (red / green runs of a few cases; the full run has no --only)

    def case(name, text, rules, want, extra=None, rel=eh, expect=None):
        """want = exit code; expect = a regex the run's output must match (the case is red for the right reason)"""
        if only and not re.search(only, name): return
        d = tree(name, text, extra, rel)
        print(f"-- {name}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run(REPO, d, rules, cut, d + "_w", ro=ro)
        out = buf.getvalue(); print(out, end="")
        good = rc == want and (expect is None or re.search(expect, out, re.M) is not None)
        if rc == want and not good: print(f"  (exit {rc} as expected, but no line matches {expect!r})")
        cases.append((name, good))

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
        # revtextfix: allowed_diffs name their entry ("*" is a rules error); the entry is the function around the log line
        lent = re.findall(r"(?m)^[^\s#/][^\n]*?\b(?:(?:Engine|SlotPages)::)?(\w+)\([^\n]*\{\s*$", src[:lg.start() + 1])[-1]
        case("the same change listed in allowed_diffs", t2,
             {"allowed_diffs": [{"entry": lent, "old": '"' + lit + '"', "new": '"x' + lit + '"', "why": "selftest"}]}, 0)
        # revtextfix (S0 review C-6 / S0b review C-4): replace rules work on token boundaries - a string literal is one
        # token (the old `str.replace` rewrote `"zq_.x..."` back to `"x..."`); zq_ is a holder name the code does not use
        case("T1 a replace rule must not rewrite inside a string literal", src.replace(lg.group(1), lg.group(1).replace('("', '("zq_.', 1), 1),
             {"replace": [["zq_.", ""]]}, 1, expect=rf"^  DIFF {lent} at token")
    else:
        cases.append(("log anchor", False))
    # a move with rules: dropCache becomes PrefillPhase::dropCache (before S7: Engine -> SlotPages; rules map it back)
    mv = src.replace(DC, "void PrefillPhase::dropCache(Slot& sl) {", 1)
    case("a move PrefillPhase::dropCache with the mapping rule", mv, {"replace": [["PrefillPhase::dropCache", f"{DCLS}::dropCache"]]}, 0)
    case("the same move without the rule", mv, {}, 1)
    # S0b (S0 review act-on 4, scratch/a25_engine/s0review/rt_new): in-class initializers / member types were not compared
    hh = "server/engine.h"
    hs = open(os.path.join(REPO, "src", hh), encoding="utf-8").read()
    for old, new, nm in (("    double vram_idle_s_ = 60.0;", "    double vram_idle_s_ = 30.0;", "an in-class initializer changed (vram_idle_s_ 60 -> 30)"),
                         ("    bool gov_hold_ = false;", "    bool gov_hold_ = true;", "an in-class initializer changed (gov_hold_ false -> true)"),
                         ("    std::uint32_t ctx_new_ = 0xffffffffu;", "    std::uint16_t ctx_new_ = 0xffffffffu;", "a member type changed (ctx_new_ uint32 -> uint16)")):
        if hs.count(old) == 1: case(nm, hs.replace(old, new, 1), {}, 1, rel=hh)
        else: cases.append((nm + ": anchor", False))
    if hs.count("    double vram_idle_s_ = 60.0;") == 1:
        case("the initializer change listed in allowed_diffs (control)", hs.replace("    double vram_idle_s_ = 60.0;", "    double vram_idle_s_ = 30.0;", 1),
             {"allowed_diffs": [{"entry": "member:vram_idle_s_", "old": "60.0", "new": "30.0", "why": "selftest"}]}, 0, rel=hh)
    # ---- revtextfix (S0 review C-6 = S0b review C-4) ----
    if hs.count("    double vram_idle_s_ = 60.0;") == 1 and hs.count("    bool gov_hold_ = false;") == 1:
        # allowed_diffs match whole tokens: `16` is not a token of `160.0` (the old joined-string replace made 160.0 -> 60.0)
        case("A1 an allowed_diff that is not on a token boundary (16 -> 6 inside 160.0)", hs.replace("    double vram_idle_s_ = 60.0;", "    double vram_idle_s_ = 160.0;", 1),
             {"allowed_diffs": [{"entry": "member:vram_idle_s_", "old": "6", "new": "16", "why": "selftest"}]}, 1, rel=hh,
             expect=r"^  DIFF member:vram_idle_s_ ")
        # entry "*" would apply `false -> true` once in every member entry (S0b review C-4): a rules error now
        case("A2 an allowed_diff with entry * is a rules error", hs.replace("    bool gov_hold_ = false;", "    bool gov_hold_ = true;", 1),
             {"allowed_diffs": [{"entry": "*", "old": "false", "new": "true", "why": "selftest"}]}, 2, rel=hh,
             expect=r"^revtext: rules error: allowed_diffs\[0\]: entry \['\*'\] is a wildcard")
        case("A3 an allowed_diff naming a list of entries (control)", hs.replace("    bool gov_hold_ = false;", "    bool gov_hold_ = true;", 1),
             {"allowed_diffs": [{"entry": ["member:gov_hold_"], "old": "false", "new": "true", "why": "selftest"}]}, 0, rel=hh)
        # a new data member only in the new tree (it was never looked at)
        nm_ = hs.replace("    bool gov_hold_ = false;", "    bool gov_hold_ = false;\n    int extra_s_ = 0;", 1)
        case("N6 a new member only in the new tree", nm_, {}, 1, rel=hh, expect=r"^  NEW member:extra_s_: only in the new tree")
        case("N7 the new member declared in rules.added (control)", nm_,
             {"added": {"member:extra_s_": {"equals": "int extra_s_ = 0;", "why": "selftest"}}}, 0, rel=hh)
    else:
        cases.append(("A/N member anchors", False))
    dm = re.search(re.escape(DC) + r"\n((?:    [^\n]*\n)+)\}\n", src)
    if dm:
        whole, inner = dm.group(0), dm.group(1)
        cp = src.replace(whole, whole + "\n" + whole.replace("dropCache(", "dropCache2(", 1), 1)
        # new entries only in the new tree: a copy of a body into a new function, the old one left as it was
        case("N1 a new function only in the new tree (copy of dropCache as dropCache2)", cp, {}, 1,
             expect=r"^  NEW dropCache2: only in the new tree")
        dsig = f"void {DCLS}::dropCache2(Slot& sl)"   # rt2: a function's equals declares its signature too
        case("N2 the copy declared in rules.added with its body (control)", cp,
             {"added": {"dropCache2": {"equals": inner, "sig": dsig, "why": "selftest"}}}, 0, expect=r", 1 added entries as declared\)")
        case("N3 rules.added with a body that is not the new body", cp,
             {"added": {"dropCache2": {"equals": inner.split("\n")[0], "sig": dsig, "why": "selftest"}}}, 1,
             expect=r"^  ADDED dropCache2: the body is not the declared text")
        case("N4 rules.added planned for a name not in the cut's new_funcs", cp,
             {"added": {"dropCache2": {"planned": True, "why": "selftest"}}}, 1, expect=r"^  ADDED dropCache2: planned, but the cut")
        ga = src.replace(whole, whole + "\nbool Engine::govActive() const {\n    return gov_hold_;\n}\n", 1)
        case("N5 rules.added planned for govActive (in the cut's new_funcs) (control)", ga,
             {"added": {"govActive": {"planned": True, "why": "selftest"}}}, 0, expect=r", 1 added entries as declared\)")
        # renames are one to one: two new entries onto one old entry was a silent dict overwrite (with the name rule the
        # copy passed as "identical")
        case("R1 renames a copy onto the old entry it was copied from", cp,
             {"renames": {"dropCache2": "dropCache"}, "replace": [["dropCache2", "dropCache"]]}, 1,
             expect=r"^  RENAME COLLISION dropCache: new entries dropCache and dropCache2")
        case("R1b two renames onto one old entry is a rules error", cp,
             {"renames": {"dropCache2": "dropCache", "dropCache": "dropCache"}}, 2, expect=r"^revtext: rules error: renames are not one to one")
        rn_ = src.replace(DC, f"void {DCLS}::dropCacheX(Slot& sl) {{", 1)
        case("R2 a real rename dropCache -> dropCacheX with its rules (control)", rn_,
             {"renames": {"dropCacheX": "dropCache"}, "replace": [["dropCacheX", "dropCache"]]}, 0, expect=r"^revtext: ok \(")
        case("R3 a rename onto a name the old tree does not have", cp, {"renames": {"dropCache2": "noSuchEntry"}}, 1,
             expect=r"^  RENAME dropCache2 -> noSuchEntry: noSuchEntry is not an entry of the old tree")
        case("R4 rules.removed lists an entry the new tree still has", src, {"removed": {"dropCache": "selftest"}}, 1,
             expect=r"^  REMOVED dropCache: the new tree still has it")
        # boundaries: `szq_` is one identifier, the rule `zq_.` must not cut it (old: `szq_.l.` -> `sl.`)
        case("T2 a replace rule must not cut an identifier", src.replace(whole, whole.replace("sl.", "szq_.l.", 1), 1),
             {"replace": [["zq_.", ""]]}, 1, expect=r"^  DIFF dropCache at token")
        # control: a token-run rule matches across whitespace / line breaks and keeps the text for the regex rules
        sp_ = src.replace(whole, whole.replace("sl.", "zq_ .\n        sl.", 1), 1)
        case("T3 a replace rule matches a token run across whitespace (control)", sp_,
             {"replace": [["zq_.", ""]]}, 0,
             expect=r"^revtext: ok \(")
    else:
        cases.append(("dropCache body anchor", False))
    case("K1 an unknown rules key (typo allowed_diff) is a rules error", src, {"allowed_diff": []}, 2,
         expect=r"^revtext: rules error: unknown key")
    # ---- rt2 (S6 review act-on 1): added.equals compared only the body; the signature (return type, parameters,
    # const) of a new function was never looked at (probe_ret / probe_par passed); `is` on a single-line inline method
    # failed because the census started an in-class entry on the line before it ----
    ph = "server/engine_pages.h"
    ps = open(os.path.join(REPO, "src", ph), encoding="utf-8").read()
    SN = "    std::uint64_t speNew() const { return stat_spe_new_; }\n"
    if ps.count(SN) == 1:
        def addl(line):   # a new inline method right after speNew (the line before it is speNew's)
            return ps.replace(SN, SN + line + "\n", 1)
        rok = "    std::uint64_t speNew2() const { return stat_spe_new_; }"
        case("S1 a new read method declared with equals + sig (control)", addl(rok),
             {"added": {"speNew2": {"equals": "return stat_spe_new_;", "sig": "std::uint64_t speNew2() const", "why": "selftest"}}}, 0,
             rel=ph, expect=r", 1 added entries as declared\)")
        case("S2 equals: the new method's return type narrowed (uint64 -> uint32)", addl(rok.replace("std::uint64_t", "std::uint32_t", 1)),
             {"added": {"speNew2": {"equals": "return stat_spe_new_;", "sig": "std::uint64_t speNew2() const", "why": "selftest"}}}, 1,
             rel=ph, expect=r"^  ADDED speNew2: the signature is not the declared text")
        case("S3 equals: the new method's parameter type narrowed (uint32 -> uint16)",
             addl("    void noteSpeTok2(std::uint16_t n) { stat_spe_tok_ += n; }"),
             {"added": {"noteSpeTok2": {"equals": "stat_spe_tok_ += n;", "sig": "void noteSpeTok2(std::uint32_t n)", "why": "selftest"}}}, 1,
             rel=ph, expect=r"^  ADDED noteSpeTok2: the signature is not the declared text")
        case("S4 equals on a function without sig is a rules error", addl(rok),
             {"added": {"speNew2": {"equals": "return stat_spe_new_;", "why": "selftest"}}}, 2,
             rel=ph, expect=r"^revtext: rules error: added\['speNew2'\]: a function's equals needs sig")
        case("I1 is on a single-line inline method (control: the census entry is that one line)",
             addl("    void noteSpeNew2() { stat_spe_new_ += 1; }"),
             {"added": {"noteSpeNew2": {"is": "void noteSpeNew2() { stat_spe_new_ += 1; }", "why": "selftest"}}}, 0,
             rel=ph, expect=r", 1 added entries as declared\)")
        case("I2 is on a single-line inline method whose return type changed", addl(rok.replace("std::uint64_t", "std::uint32_t", 1)),
             {"added": {"speNew2": {"is": rok.strip(), "why": "selftest"}}}, 1,
             rel=ph, expect=r"^  ADDED speNew2: the all tokens is not the declared text at token 2:")
    else:
        cases.append(("S/I speNew anchor", False))
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
