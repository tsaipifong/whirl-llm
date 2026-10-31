# SPDX-License-Identifier: Apache-2.0
"""Member-order pin of class Engine (arch025-engine S0; DESIGN 6 C2): declaration order == destruction order.

Flattens Engine's data members (a member whose type is one of the pin's `groups:` structs is expanded into that
struct's data members, recursively, as `holder.member`; the struct may live in any header under src/server) and
compares the sequence with tests/pins/engine_member_order.txt. A step that moves a member changes the pin file in
the same commit AND adds a row to the pin file's audit table (member | type | what its destructor does | why nobody
else sees the move), DESIGN 2.5. Generalised from scratch/a25_lane1b/pin_member_order.py (lane 1 leg 4).

  python -I tools/gates/member_order_check.py [--root R] [--pin P]      exit 0 ok, 1 order differs, 2 parse error
  python -I tools/gates/member_order_check.py --write                   rewrite the member list (keeps the header)
  python -I tools/gates/member_order_check.py --selftest [--work W]     fault cases (swap last_tick_ / last_reap_,
                                                                        a group member moved, S0b: tb_ moved with / without
                                                                        an audit row, initializer / type changed, origin
                                                                        edited) must fail, base and the control pass
  python -I tools/gates/member_order_check.py --write-origin            (S0b, once) the origin list + its sha256
S0b: every change against tests/pins/engine_member_order_origin.txt (moved / added / removed / declaration changed) needs
an audit-table row naming the member; see origin_check().
"""
import difflib, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def arg(n, d=None):
    a = sys.argv[1:]
    return a[a.index(n) + 1] if n in a else d


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def class_body(text, name):
    m = re.search(r"\b(?:class|struct)\s+" + name + r"\b(?:\s+final)?\s*(?::(?!:)[^;{]*)?\{", text)
    if not m: return None
    i, depth = m.end(), 1
    while depth:
        c = text[i]; depth += c == "{"; depth -= c == "}"; i += 1
    return text[m.end():i - 1]


def detpl(x):
    while True:
        y = re.sub(r"<[^<>]*>", "", x)
        if y == x: return x
        x = y


def statements(body):
    out, cur, depth = [], [], 0
    for c in body:
        if c == "{":
            depth += 1; cur.append(c); continue
        if c == "}":
            depth -= 1; cur.append(c)
            if depth == 0:
                st = "".join(cur); head = detpl(st.split("{")[0])
                if "(" in head or re.search(r"\b(struct|class|enum|union)\b", head):
                    out.append(st); cur = []
            continue
        if c == ";" and depth == 0:
            out.append("".join(cur)); cur = []; continue
        cur.append(c)
    return out


def data_members(body, sigs=False):
    res = []
    for st in statements(body):
        s = re.sub(r"\b(public|private|protected)\s*:", "", st).strip()
        sig = " ".join(s.split())   # S0b: the whole declaration (type, annotations, initializer), whitespace-normalised
        s = re.sub(r"\bWH_\w+\((?:[^()]|\([^()]*\))*\)", "", s)
        if not s or re.match(r"(using|static|friend|typedef|template|struct|class|enum|union)\b", s): continue
        buf, depth = [], 0
        for c in s:
            if c == "{": depth += 1
            elif c == "}": depth -= 1; continue
            if depth == 0: buf.append(c)
        s = "".join(buf)
        na = detpl(s)
        if "(" in na or "operator" in na: continue
        parts, depth, cur = [], 0, []
        for c in s:
            depth += c == "<"; depth -= c == ">"
            if c == "," and depth == 0: parts.append("".join(cur)); cur = []
            else: cur.append(c)
        parts.append("".join(cur))
        m = re.match(r"(.*?)([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*$", re.sub(r"=.*$", "", parts[0]).strip(), flags=re.S)
        if not m or not m.group(1).strip(): continue
        typ = m.group(1).strip()
        for n in [m.group(2)] + [re.sub(r"=.*$", "", p).strip() for p in parts[1:]]:
            n = re.sub(r"\[.*$", "", n).strip()
            if n: res.append((typ, n, sig) if sigs else (typ, n))
    return res


def flatten(texts, cls, groups, prefix="", sigs=False):
    body = next((b for b in (class_body(t, cls) for t in texts) if b is not None), None)
    if body is None: raise ValueError(f"class {cls} not found")
    out = []
    for typ, n, sig in data_members(body, True):
        base = re.sub(r"^(const\s+)?", "", typ).split("::")[-1].strip().rstrip("&* ").strip()
        if base in groups and base != cls and not re.search(r"[&*]\s*$", typ):
            out += flatten(texts, base, groups, prefix + n + ".", sigs)
        else:
            out.append((prefix + n, sig) if sigs else prefix + n)
    return out


# S0b (S0 review C-1 / the C1 fifth shape): the pin list is edited by the step that moves a member, so the list alone cannot
# show an unaudited move. Every member is also compared with the ORIGIN list (tests/pins/engine_member_order_origin.txt:
# name <TAB> whole declaration, written at a17493b = d6e83ed + S0 + C1; same 120 names, same order as c1ca7ce / abe3512); its sha256 is fixed here.
# A member that moved (not on the longest common subsequence of the origin order and the tree order), was added or removed,
# or whose declaration changed (type, annotation, in-class initializer: `vram_idle_s_ = 60.0` -> `30.0`) needs a row in the
# pin's audit table whose first column names it (comma list, `*` globs; holder prefixes are ignored).
ORIGIN = "engine_member_order_origin.txt"
ORIGIN_SHA = "20b171e44898532e25e8a5f57633204ecb36b4a01ee397a7d42bf58bea91f9c9"


def sha_text(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def read_origin(p):
    out = []
    for l in open(p, encoding="utf-8"):
        if l.startswith("#") or not l.strip(): continue
        n, sig = l.rstrip("\n").split("\t", 1)
        out.append((n, sig))
    return out


def audit_names(pin):
    """first column of every audit-table row of the pin header: `# | a_, b_ (new) | ...`"""
    pats = []
    for l in open(pin, encoding="utf-8"):
        m = re.match(r"#\s*\|([^|]+)\|(?:[^|]*\|){3,}", l)
        if not m or m.group(1).strip() in ("member", "callback") or set(m.group(1).strip()) <= set("-: "): continue
        for x in m.group(1).split(","):
            x = re.sub(r"\([^)]*\)", "", x).strip()
            # S0c (S0b review C-1 (a)): a glob needs at least two literal name characters (a lone `*` covered everything)
            if re.fullmatch(r"[\w*.]+", x) and len(re.sub(r"[*.]", "", x)) >= 2: pats.append(x)
    return pats


def lcs_keep(a, b):
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    keep, i, j = set(), 0, 0
    while i < n and j < m:
        if a[i] == b[j]: keep.add(a[i]); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]: i += 1
        else: j += 1
    return keep


def origin_check(texts, groups, pin, origin_p):
    import fnmatch
    if not os.path.exists(origin_p): return 1, [f"member order: ORIGIN MISSING {origin_p}"]
    if sha_text(origin_p) != ORIGIN_SHA:
        return 1, [f"member order: ORIGIN CHANGED ({origin_p} sha256 {sha_text(origin_p)[:12]} != {ORIGIN_SHA[:12]} fixed in member_order_check.py)"]
    org = read_origin(origin_p)
    have = flatten(texts, "Engine", set(groups), sigs=True)
    last = lambda n: n.split(".")[-1]
    ol, hl = [last(n) for n, _ in org], [last(n) for n, _ in have]
    dup = {x for x in ol + hl if ol.count(x) > 1 or hl.count(x) > 1}
    ident = lambda n: n if last(n) in dup else last(n)
    oi, hi = [ident(n) for n, _ in org], [ident(n) for n, _ in have]
    osig, hsig = {ident(n): s for n, s in org}, {ident(n): s for n, s in have}
    keep = lcs_keep(oi, hi)
    need = {}
    for i in hi:
        if i not in osig: need[i] = "added"
        elif i not in keep: need[i] = "moved"
        elif hsig[i] != osig[i]: need[i] = f"declaration changed [{osig[i][:70]}] -> [{hsig[i][:70]}]"
    for i in oi:
        if i not in hsig: need[i] = "removed"
    pats = audit_names(pin)
    full = {ident(n): n for n, _ in have}
    covered = lambda i: any(fnmatch.fnmatchcase(i, p) or fnmatch.fnmatchcase(full.get(i, i), p) for p in pats)
    bad = [f"member order: NO AUDIT ROW for {i} ({why}; vs the origin list)" for i, why in sorted(need.items()) if not covered(i)]
    ok = [f"member order: audited vs origin: {i} ({why.split(' [')[0]})" for i, why in sorted(need.items()) if covered(i)]
    return (1 if bad else 0), bad + ok


def read_pin(p):
    head, groups, members = [], [], []
    for l in open(p, encoding="utf-8"):
        s = l.rstrip("\n")
        if s.startswith("#") or not s.strip():
            if not members: head.append(s)
            continue
        if s.startswith("groups:"): groups = s.split()[1:]; head.append(s); continue
        members.append(s.strip())
    return head, groups, members


def tree_texts(root, eng_h=None):
    d = os.path.join(root, "src", "server")
    eh = eng_h if eng_h is not None else open(os.path.join(d, "engine.h"), encoding="utf-8").read()
    texts = [strip_comments(eh)]
    for f in sorted(os.listdir(d)):
        if f.endswith(".h") and f != "engine.h":
            texts.append(strip_comments(open(os.path.join(d, f), encoding="utf-8").read()))
    return texts


def check(texts, groups, want, pin=None, origin=None):
    have = flatten(texts, "Engine", set(groups))
    orc, olines = origin_check(texts, groups, pin, origin) if pin else (0, [])
    otxt = "".join("\n" + l for l in olines)
    if have == want and not orc: return 0, f"member order: ok ({len(have)} members, same order; origin list: {len(olines)} audited changes)" + otxt
    if have == want: return 1, "member order: AUDIT MISSING (the pin list matches the tree, but a change against the origin has no audit row)" + otxt
    diff = "\n".join(difflib.unified_diff(want, have, "pin", "tree", lineterm="", n=1))
    return 1, "member order: DIFFERS\n" + diff + otxt


# S0b selftest mutants (S0 review: r5-tb-before-ps = scratch/a25_engine/s0review/mut_r5.json, mo_pin_tbmoved.txt, rt_new)
TB, CK = "    TierBridge tb_;\n", "    CkptStore ck_;  // checkpoint buffers (arch024 M2)\n"


def selftest(root, groups, want, pin, origin, work):
    eh = open(os.path.join(root, "src", "server", "engine.h"), encoding="utf-8").read()
    head = [l.rstrip("\n") for l in open(pin, encoding="utf-8") if l.startswith("#") or l.startswith("groups:") or not l.strip()]
    head = head[:next((k for k, l in enumerate(head) if l.startswith("groups:")), len(head)) + 1]
    os.makedirs(work, exist_ok=True)
    cases = []

    def pin_with(name, members, extra_rows=()):
        p2 = os.path.join(work, name + ".txt")
        hd = list(head)
        k = next(k for k, l in enumerate(hd) if l.startswith("# | member |")) + 1
        hd[k + 1:k + 1] = list(extra_rows)   # after the table's |---| line
        open(p2, "w", encoding="utf-8", newline="\n").write("\n".join(hd) + "\n" + "\n".join(members) + "\n")
        return p2

    def case(name, text, expect_rc, expect_re, pin_p=None, origin_p=None):
        try:
            rc, msg = check(tree_texts(root, text), groups, want if pin_p is None else read_pin(pin_p)[2],
                            pin_p or pin, origin_p or origin)
        except ValueError as e:
            rc, msg = 2, f"parse error {e}"
        good = rc == expect_rc and (expect_re is None or re.search(expect_re, msg, re.M) is not None)
        cases.append((name, good, msg.splitlines()[0] if good or not msg else msg[:300]))

    case("base tree passes", eh, 0, r"^member order: ok ")
    a, b = "    TimePoint last_tick_{};", "    TimePoint last_reap_{};"
    if eh.count(a) == 1 and eh.count(b) == 1:
        case("swap last_tick_ / last_reap_ (two trivial members) fails", eh.replace(a, "@@A@@").replace(b, a).replace("@@A@@", b), 1, r"^member order: DIFFERS")
    else:
        cases.append(("swap last_tick_ / last_reap_: anchors missing", False, ""))
    sb = re.search(r"\n(    SampleBuffers sb_[^\n]*\n)", eh)
    sl = re.search(r"\n(    std::vector<Slot> slots_;[^\n]*\n)", eh)
    if sb and sl:
        case("sb_ moved after slots_ (DESIGN v2 6 C1 case a) fails",
             eh.replace(sb.group(1), "", 1).replace(sl.group(1), sl.group(1) + sb.group(1), 1), 1, r"^member order: DIFFERS")
    else:
        cases.append(("sb_ / slots_ anchors missing", False, ""))
    if eh.count(TB) == 1 and eh.count(CK) == 1:
        tbm = eh.replace(TB, "", 1).replace(CK, CK + TB, 1)
        case("r5: tb_ moved after ck_ (outlives ps_ / kv_arrays_), pin list unchanged: fails", tbm, 1, r"^member order: DIFFERS")
        moved = flatten(tree_texts(root, tbm), "Engine", set(groups))
        p_nr = pin_with("pin_tb_moved_no_row", moved)
        case("r5b: tb_ moved AND the pin list edited to match, no audit row: fails (the C1 fifth shape)", tbm, 1,
             r"^member order: NO AUDIT ROW for tb_ \(moved", p_nr)
        p_r = pin_with("pin_tb_moved_row", moved, ["# | tb_ | TierBridge | selftest row | after ck_ | selftest control |"])
        case("r5c: control: the same move with an audit row for tb_ passes", tbm, 0, r"^member order: audited vs origin: tb_ \(moved\)", p_r)
        # S0c (S0b review C-1 (a), r_mo.py case a): a lone `*` row would cover every change
        p_star = pin_with("pin_tb_moved_star", moved, ["# | * | any | any | any | any |"])
        case("C-1a: the same move with only a lone '*' audit row: fails", tbm, 1, r"^member order: NO AUDIT ROW for tb_ \(moved", p_star)
    else:
        cases.append(("tb_ / ck_ anchors missing", False, ""))
    for old, new, nm in (("    double vram_idle_s_ = 60.0;", "    double vram_idle_s_ = 30.0;", "vram_idle_s_"),
                         # S8 moved gov_hold_ / ctx_new_ into GovState: the same two shapes (value, type) on Engine members
                         ("    bool vram_warn_ = false;", "    bool vram_warn_ = true;", "vram_warn_"),
                         ("    bool spec_on_ = false;", "    int spec_on_ = 0;", "spec_on_")):
        if eh.count(old) == 1:
            case(f"in-class declaration changed ({old.strip()} -> {new.strip()}), names / order unchanged: fails",
                 eh.replace(old, new, 1), 1, r"^member order: NO AUDIT ROW for " + nm + r" \(declaration changed")
        else:
            cases.append((f"{nm} anchor missing", False, ""))
    o2 = os.path.join(work, "origin_edited.txt")
    open(o2, "w", encoding="utf-8", newline="\n").write(open(origin, encoding="utf-8").read().replace("\ttb_", "\ttb_ ", 1) + "# edited\n")
    case("the origin list edited: fails", eh, 1, r"^member order: ORIGIN CHANGED", None, o2)
    for n, good, msg in cases:
        print(f"  {'ok  ' if good else 'FAIL'} {n}" + ("" if good else f"\n       got: {msg}"))
    print(f"member order selftest: {sum(g for _, g, _ in cases)}/{len(cases)} as expected")
    return 0 if all(g for _, g, _ in cases) else 1


def main():
    root = arg("--root", REPO)
    pin = arg("--pin", os.path.join(REPO, "tests", "pins", "engine_member_order.txt"))
    origin = arg("--origin", os.path.join(REPO, "tests", "pins", ORIGIN))
    head, groups, want = read_pin(pin)
    if "--write" in sys.argv:
        have = flatten(tree_texts(root), "Engine", set(groups))
        open(pin, "w", encoding="utf-8", newline="\n").write("\n".join(head) + "\n" + "\n".join(have) + "\n")
        print(f"member order: wrote {pin} ({len(have)} members)"); return 0
    if "--write-origin" in sys.argv:   # once (S0b); the sha256 printed goes into ORIGIN_SHA above
        have = flatten(tree_texts(root), "Engine", set(groups), sigs=True)
        open(origin, "w", encoding="utf-8", newline="\n").write(
            "# arch025-engine S0b: origin of the member-order pin (name <TAB> whole declaration), written at a17493b (d6e83ed + S0 + C1) by\n"
            "# member_order_check.py --write-origin; never edited (its sha256 is fixed in member_order_check.py ORIGIN_SHA).\n"
            + "".join(f"{n}\t{sg}\n" for n, sg in have))
        print(f"member order: wrote {origin} ({len(have)} members), sha256 {sha_text(origin)}"); return 0
    if len(want) < 50:
        print(f"member order: parse error (pin has {len(want)} members)"); return 2
    if "--selftest" in sys.argv:
        return selftest(root, groups, want, pin, origin, arg("--work", os.path.join(REPO, "build", "member_order_selftest")))
    try:
        rc, msg = check(tree_texts(root), groups, want, pin, origin)
    except ValueError as e:
        print(f"member order: parse error ({e})"); return 2
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
