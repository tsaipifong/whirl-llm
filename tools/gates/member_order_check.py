# SPDX-License-Identifier: Apache-2.0
"""Member-order pin of class Engine (arch025-engine S0; DESIGN 6 C2): declaration order == destruction order.

Flattens Engine's data members (a member whose type is one of the pin's `groups:` structs is expanded into that
struct's data members, recursively, as `holder.member`; the struct may live in any header under src/server) and
compares the sequence with tests/pins/engine_member_order.txt. A step that moves a member changes the pin file in
the same commit AND adds a row to the pin file's audit table (member | type | what its destructor does | why nobody
else sees the move), DESIGN 2.5. Generalised from scratch/a25_lane1b/pin_member_order.py (lane 1 leg 4).

  python -I tools/gates/member_order_check.py [--root R] [--pin P]      exit 0 ok, 1 order differs, 2 parse error
  python -I tools/gates/member_order_check.py --write                   rewrite the member list (keeps the header)
  python -I tools/gates/member_order_check.py --selftest                fault cases (swap last_tick_ / last_reap_,
                                                                        a group member moved) must fail, base passes
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


def data_members(body):
    res = []
    for st in statements(body):
        s = re.sub(r"\b(public|private|protected)\s*:", "", st).strip()
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
            if n: res.append((typ, n))
    return res


def flatten(texts, cls, groups, prefix=""):
    body = next((b for b in (class_body(t, cls) for t in texts) if b is not None), None)
    if body is None: raise ValueError(f"class {cls} not found")
    out = []
    for typ, n in data_members(body):
        base = re.sub(r"^(const\s+)?", "", typ).split("::")[-1].strip().rstrip("&* ").strip()
        if base in groups and base != cls and not re.search(r"[&*]\s*$", typ):
            out += flatten(texts, base, groups, prefix + n + ".")
        else:
            out.append(prefix + n)
    return out


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


def check(texts, groups, want):
    have = flatten(texts, "Engine", set(groups))
    if have == want: return 0, f"member order: ok ({len(have)} members, same order)"
    diff = "\n".join(difflib.unified_diff(want, have, "pin", "tree", lineterm="", n=1))
    return 1, "member order: DIFFERS\n" + diff


def main():
    root = arg("--root", REPO)
    pin = arg("--pin", os.path.join(REPO, "tests", "pins", "engine_member_order.txt"))
    head, groups, want = read_pin(pin)
    if "--write" in sys.argv:
        have = flatten(tree_texts(root), "Engine", set(groups))
        open(pin, "w", encoding="utf-8", newline="\n").write("\n".join(head) + "\n" + "\n".join(have) + "\n")
        print(f"member order: wrote {pin} ({len(have)} members)"); return 0
    if len(want) < 50:
        print(f"member order: parse error (pin has {len(want)} members)"); return 2
    if "--selftest" in sys.argv:
        eh = open(os.path.join(root, "src", "server", "engine.h"), encoding="utf-8").read()
        cases, ok = [], True
        rc, _ = check(tree_texts(root), groups, want)
        cases.append(("base tree passes", rc == 0))
        a, b = "    TimePoint last_tick_{};", "    TimePoint last_reap_{};"
        if eh.count(a) == 1 and eh.count(b) == 1:
            m = eh.replace(a, "@@A@@").replace(b, a).replace("@@A@@", b)
            rc, _ = check(tree_texts(root, m), groups, want)
            cases.append(("swap last_tick_ / last_reap_ (two trivial members) fails", rc == 1))
        else:
            cases.append(("swap last_tick_ / last_reap_: anchors missing", False))
        sb = re.search(r"\n(    SampleBuffers sb_[^\n]*\n)", eh)
        sl = re.search(r"\n(    std::vector<Slot> slots_;[^\n]*\n)", eh)
        if sb and sl:
            m = eh.replace(sb.group(1), "", 1).replace(sl.group(1), sl.group(1) + sb.group(1), 1)
            rc, _ = check(tree_texts(root, m), groups, want)
            cases.append(("sb_ moved after slots_ (DESIGN v2 6 C1 case a) fails", rc == 1))
        else:
            cases.append(("sb_ / slots_ anchors missing", False))
        for n, good in cases:
            print(f"  {'ok  ' if good else 'FAIL'} {n}"); ok &= good
        print(f"member order selftest: {sum(g for _, g in cases)}/{len(cases)} as expected")
        return 0 if ok else 1
    try:
        rc, msg = check(tree_texts(root), groups, want)
    except ValueError as e:
        print(f"member order: parse error ({e})"); return 2
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
