# SPDX-License-Identifier: Apache-2.0
"""Declared-name parser for a changed line outside any function (measure-tools P1, DESIGN B1.1-1.2).

impact.py used to take every identifier on such a line as "changed", type names included, so a member
`std::vector<device::DevBuf> allocations;` selected every function that mentions DevBuf (605281b..9844fde: model.h
147, engine_api.h 101, vision.h 43 users). This module returns the DECLARED entity instead:

    declared(src, line) -> {"kind": ..., "decls": [{"name", "cls", "what", "types"}], "target": include target}

kind: "decl" (declarators found), "enumerator", "include", "macro", "cond" (#if / #else / #endif ...: file scope),
"pragma", "assert" (static_assert: compile time only), "scope" (using namespace: name lookup of the file), "punct"
(braces, access specifiers, namespace openers), "unparsed" (code the parser does not understand: impact.py fails
closed and reports it as a map defect).
decls: name = the declared name (member, function, variable, alias, type, enumerator, macro); cls = the class /
struct / union / enum the line is in (innermost), None at namespace scope; what = member | function | variable |
type | alias | enumerator | macro; types = the other identifiers of the declaration (type_changed information,
never selection). Text scanner (stdlib only), like cxxscan; it over-approximates where it is unsure.
"""
from __future__ import annotations

import bisect
import re

import cxxscan

NOISE = {"const", "constexpr", "consteval", "constinit", "static", "inline", "struct", "class", "enum", "union", "public",
         "private", "protected", "virtual", "override", "final", "void", "bool", "int", "unsigned", "signed", "char",
         "short", "long", "float", "double", "auto", "std", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
         "int8_t", "int16_t", "int32_t", "int64_t", "string", "vector", "optional", "array", "map", "set", "pair",
         "unique_ptr", "shared_ptr", "function", "atomic", "mutex", "string_view", "true", "false", "nullptr", "typename",
         "template", "noexcept", "explicit", "operator", "mutable", "volatile", "extern", "friend", "default", "delete",
         "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast", "sizeof", "alignof", "decltype", "alignas",
         "this", "return", "new", "using", "namespace", "typedef", "register", "thread_local", "chrono", "whirl"}
_ident = re.compile(r"~?[A-Za-z_]\w*")
_access = re.compile(r"^\s*(?:public|private|protected)\s*:(?!:)\s*")
_cache: dict = {}


def _prep(src: str):
    """per source: blanked text (comments / strings, preprocessor lines too), line offsets, the brace frames open at
    every line start, the matching brace of every '{', and the preprocessor directive each line belongs to"""
    key = hash(src)
    if key in _cache:
        return _cache[key]
    clean0, _ = cxxscan._blank(src)  # comments and strings blanked, '#' lines kept
    raw_lines = src.split("\n")
    c0_lines = clean0.split("\n")
    pp = {}  # line -> (start line, directive text of the start line)
    i = 0
    while i < len(c0_lines):
        if c0_lines[i].lstrip().startswith("#"):
            start = i
            while True:
                pp[i + 1] = (start + 1, raw_lines[start])
                if not raw_lines[i].rstrip().endswith("\\") or i + 1 >= len(c0_lines):
                    break
                i += 1
        i += 1
    clines = [(" " * len(l) if (n + 1) in pp else l) for n, l in enumerate(c0_lines)]
    clean = "\n".join(clines)
    starts = [0]
    for l in clines:
        starts.append(starts[-1] + len(l) + 1)
    frames_at = {}  # line -> tuple of frames (kind, name, open pos, header)
    match = {}
    stack = []
    seg = 0
    line = 1
    frames_at[1] = ()
    for p, c in enumerate(clean):
        if c == "\n":
            line += 1
            frames_at[line] = tuple(stack)
        elif c == ";":
            seg = p + 1
        elif c == "{":
            header = clean[seg:p]
            stack.append((_classify(header, stack), _hname(header), p, header))
            seg = p + 1
        elif c == "}":
            if stack:
                match[stack[-1][2]] = p
                stack.pop()
            seg = p + 1
    if len(_cache) >= 8:  # a few sources at a time (impact.py walks one file's two sides and its scope order)
        _cache.pop(next(iter(_cache)))
    _cache[key] = (clean, clines, starts, frames_at, match, pp)
    return _cache[key]


_cond_cache: dict = {}


def cond_stack(src: str) -> dict:
    """mt-p1 pass 3 (rereview act-on 1, P13b / P14b): line -> the enclosing preprocessor conditions, outermost first,
    as ('#if X', '#else') pairs (the branch the line is in). An include guard (#ifndef X / #define X as the first
    directive pair at the top level) is left out, so a declaration moved between two guarded headers keeps its
    signature. Directive lines get the stack in force before them."""
    key = hash(src)
    if key in _cond_cache:
        return _cond_cache[key]
    raw = src.split("\n")
    clean0 = cxxscan._blank(src)[0].split("\n")
    out, stack = {}, []
    i = 0
    while i < len(raw):
        out[i + 1] = tuple(e for e in stack if e is not None)
        t = clean0[i].strip() if i < len(clean0) else ""
        start = i
        while raw[i].rstrip().endswith("\\") and i + 1 < len(raw):  # continuation lines of a directive
            i += 1
            out[i + 1] = out[start + 1]
        mm = re.match(r"#\s*(\w+)\s*(.*)", t)
        if mm:
            kw, rest = mm.group(1), " ".join(mm.group(2).split())
            if kw in ("if", "ifdef", "ifndef"):
                guard = None
                if kw == "ifndef" and not stack:
                    nxt = next((clean0[k].strip() for k in range(i + 1, len(clean0)) if clean0[k].strip()), "")
                    gm = re.match(r"#\s*define\s+(\w+)\s*$", nxt)
                    guard = gm is not None and gm.group(1) == rest
                stack.append(None if guard else (f"#{kw} {rest}", "#if"))
            elif kw in ("elif", "else", "elifdef", "elifndef") and stack:
                if stack[-1] is not None:
                    stack[-1] = (stack[-1][0], f"#{kw} {rest}".strip())
            elif kw == "endif" and stack:
                stack.pop()
        i += 1
    if len(_cond_cache) >= 8:
        _cond_cache.pop(next(iter(_cond_cache)))
    _cond_cache[key] = out
    return out


def _scope_key(frames) -> tuple:
    """mt-p1 pass 3 (act-on 1, P15-P17): the full scope of a statement: every enclosing namespace / class / enum header,
    normalised (template specialisation arguments, base classes and outer classes included)"""
    return tuple(f"{f[0]}:{' '.join(_strip_lead(f[3]).split())}" for f in frames if f[0] in ("ns", "class", "enum"))


def _classify(header: str, stack) -> str:
    if any(f[0] in ("func", "init") for f in stack):
        return "inner"
    h = _strip_lead(header)
    if re.match(r"^(?:inline\s+)?namespace\b", h) or re.match(r"^extern\b", h):
        return "ns"
    if re.match(r"^(?:typedef\s+)?enum\b", h):
        return "enum"
    if cxxscan._func_name(header):
        return "func"
    if re.search(r"\b(class|struct|union)\s+(?:alignas\([^)]*\)\s*)?[A-Za-z_]\w*[^(){};=]*$", cxxscan._untemplate(h)):
        return "class"
    return "init"


def _hname(header: str):
    h = cxxscan._untemplate(_strip_lead(header))
    m = re.search(r"\b(?:class|struct|union|enum(?:\s+(?:class|struct))?)\s+(?:alignas\([^)]*\)\s*)?(?:[A-Z_][A-Z0-9_]+\s+)?([A-Za-z_]\w*)", h)
    return m.group(1) if m else None


def _strip_lead(s: str) -> str:
    s = s.strip()
    while True:
        t = _access.sub("", s)
        t = re.sub(r"^\[\[[^\]]*\]\]\s*", "", t)
        if t == s:
            return s
        s = t.strip()


def _top(s: str, chars: str):
    """positions of chars in s outside (), [], {} and template <> (a '<' after an identifier opens a template)"""
    out, d, a = [], 0, 0
    for k, c in enumerate(s):
        if c in "([{":
            d += 1
        elif c in ")]}":
            d -= 1
        elif c == "<" and k and (s[k - 1].isalnum() or s[k - 1] in "_ ") and d == 0 and not s[k + 1:k + 2] == "<":
            a += 1
        elif c == ">" and a and d == 0:
            a -= 1
        elif c in chars and d == 0 and a == 0:
            out.append(k)
    return out


def _parse_stmt(stmt: str, cls, term: str):
    """one declaration statement -> (kind, decls)"""
    s = _strip_lead(stmt)
    s = cxxscan._untemplate(s).strip() if s.startswith("template") else s
    s = _strip_lead(s)
    if not s:
        return "punct", []
    if re.match(r"^(?:inline\s+)?namespace\b", s) or re.fullmatch(r"extern\s*", s):
        return "punct", []
    if s.startswith("static_assert"):
        return "assert", []
    if re.match(r"^using\s+namespace\b", s):
        return "scope", []
    m = re.match(r"^using\s+([A-Za-z_]\w*)\s*(?:\[\[[^\]]*\]\]\s*)?=(.*)$", s, re.S)
    if m:
        return "decl", [_d(m.group(1), cls, "alias", m.group(2))]
    m = re.match(r"^using\s+(?:typename\s+)?[\w:]*::(~?[A-Za-z_]\w*)\s*$", s)
    if m:
        return "decl", [_d(m.group(1), cls, "alias", "")]
    s = re.sub(r"^friend\s+", "", s)
    s = re.sub(r"^extern\s+", "", s)
    if s.startswith("typedef"):
        body = s[len("typedef"):]
        fp = re.search(r"\(\s*[*&]\s*([A-Za-z_]\w*)\s*\)", body)
        nm = fp.group(1) if fp else (re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", body)) or [None])[-1]
        return ("decl", [_d(nm, cls, "alias", body)]) if nm else ("unparsed", [])
    m = re.match(r"^(?:typedef\s+)?(?:enum(?:\s+(?:class|struct))?|struct|class|union)\s+(?:alignas\([^)]*\)\s*)?"
                 r"(?:[A-Z_][A-Z0-9_]+\s+)?([A-Za-z_]\w*)\s*(?::(?!:)(.*))?$", s, re.S)
    if m and (term in "{;" or term == ""):
        return "decl", [_d(m.group(1), None if term == "{" else cls, "type", m.group(2) or "")]
    # mt-p1 review A3: operators first (operator== has '=' in its name), then mask template arguments and decltype(...)
    # (same length, so positions map back to the original text): std::function<void(int)> on_done is the member
    # on_done, decltype(&hipFree) free_fn is free_fn; the masked text is searched, the types come from the original
    mo = re.search(r"\boperator\s*(\(\)|\[\]|[^\w\s(]+|\s+[\w:<>*& ]+?)\s*\(", s)
    if mo:
        return "decl", [_d("operator" + re.sub(r"\s+", "", mo.group(1)), cls, "function", s[:mo.start()] + s[mo.end():])]
    orig = s
    s = _mask(s)
    eq = _top(s, "=")
    head = s[:eq[0]] if eq else s
    par = _top(head, "(")
    # a '(' at top level: function (or a function pointer / a constructor-style initializer)
    first_paren = None
    dd = 0
    for k, c in enumerate(head):
        if c == "(" and dd == 0:
            first_paren = k
            break
        if c in "[{":
            dd += 1
        elif c in "]}":
            dd -= 1
    if first_paren is not None:
        fp = re.match(r"\(\s*[*&]\s*(?:[A-Za-z_]\w*::\s*\*)?\s*([A-Za-z_]\w*)\s*\)", head[first_paren:])
        if fp:
            return "decl", [_d(fp.group(1), cls, "variable", orig[:first_paren])]
        pre = head[:first_paren].rstrip()
        mo = re.search(r"(operator\s*(?:\(\)|[^\s(]+|\s+[\w:<>*& ]+?))\s*$", pre)
        if mo:
            return "decl", [_d("operator" + re.sub(r"\s+", "", mo.group(1)[len("operator"):]), cls, "function", pre[:mo.start()])]
        mn = re.search(r"(~?[A-Za-z_]\w*)\s*(?:<[^<>]*>)?\s*$", pre)
        if mn and mn.group(1) not in ("decltype", "alignas", "__declspec", "sizeof", "__attribute__"):
            nm = mn.group(1)
            if "::" in pre[:mn.start()].rstrip()[-2:] and cls is None:
                q = re.search(r"([A-Za-z_]\w*)\s*::\s*$", pre[:mn.start()])
                cl = q.group(1) if q else None
            else:
                cl = cls
            return "decl", [_d(nm, cl, "function", orig[:mn.start()] + orig[first_paren:len(head)])]
        if not mn:
            return "unparsed", []
    # variables: cut the initializer, the bit-field width and array bounds; split the declarators
    parts = []
    last = 0
    for k in _top(s, ","):
        parts.append((last, s[last:k]))
        last = k + 1
    parts.append((last, s[last:]))
    decls = []
    typetxt = ""
    for n, (p0, p) in enumerate(parts):
        e = _top(p, "=")
        p = p[:e[0]] if e else p
        bf = [k for k in _top(p, ":") if not (p[k - 1:k] == ":" or p[k + 1:k + 2] == ":")]
        p = p[:bf[0]] if bf else p
        p = re.sub(r"\[[^\]]*\]", lambda x: " " * len(x.group(0)), p)
        p = re.sub(r"\{[^{}]*\}", lambda x: " " * len(x.group(0)), p)
        ids = [x for x in re.findall(r"[A-Za-z_]\w*", p)]
        if not ids:
            continue
        nm = ids[-1]
        if n == 0:
            typetxt = orig[p0:p0 + p.rfind(nm)]
        decls.append(_d(nm, cls, "member" if cls else "variable", typetxt))
    if not decls:
        return "unparsed", []
    return "decl", decls


def _mask(s: str) -> str:
    """s with the contents of template argument lists (after an identifier) and of decltype(...) blanked (same length)"""
    out = list(s)
    for m in re.finditer(r"\bdecltype\s*\(", s):
        d, k = 1, m.end()
        while k < len(s) and d:
            d += {"(": 1, ")": -1}.get(s[k], 0)
            k += 1
        for i in range(m.start(), k):
            out[i] = " "
    t = "".join(out)
    out = list(t)
    k = 0
    while k < len(t):
        if t[k] == "<" and k and (t[k - 1].isalnum() or t[k - 1] == "_") and t[k + 1:k + 2] != "<" and t[k - 1:k] != "<":
            d, j = 1, k + 1
            while j < len(t) and d:
                d += {"<": 1, ">": -1}.get(t[j], 0)
                if t[j] in ";{}":
                    break
                j += 1
            if d == 0:
                for i in range(k + 1, j - 1):
                    out[i] = " "
                k = j
                continue
        k += 1
    return "".join(out)


def _d(name, cls, what, typetxt, sig=None):
    txt = re.sub(r"[A-Za-z_]\w*\s*::", " ", typetxt or "")  # namespace / class qualifiers are not types
    types = sorted({x for x in re.findall(r"[A-Za-z_]\w*", txt)} - NOISE - {name})
    if what == "variable" and cls:
        what = "member"
    return {"name": name, "cls": cls, "what": what, "types": types, "sig": sig}


def declared(src: str, line: int) -> dict:
    clean, clines, starts, frames_at, match, pp = _prep(src)
    if line in pp:
        st, txt = pp[line]
        t = txt.strip()
        m = re.match(r"#\s*(\w+)\s*(.*)", t)
        kw, rest = (m.group(1), m.group(2)) if m else ("", "")
        if kw == "include":
            tg = re.match(r'[<"]([^">]+)[">]', rest)
            return {"kind": "include", "decls": [], "target": tg.group(1) if tg else rest}
        if kw in ("define", "undef"):
            nm = re.match(r"([A-Za-z_]\w*)", rest)
            rl = src.split("\n")
            k, full = st - 1, []
            while k < len(rl):
                full.append(rl[k].rstrip().rstrip("\\"))
                if not rl[k].rstrip().endswith("\\"):
                    break
                k += 1
            ds = [_d(nm.group(1), None, "macro", "", ("#", cond_stack(src).get(st, ()), " ".join(" ".join(full).split())))] if nm else []
            return {"kind": "macro", "decls": ds, "target": None}
        if kw == "pragma":
            return {"kind": "pragma", "decls": [], "target": None}
        return {"kind": "cond", "decls": [], "target": None}
    if line < 1 or line > len(clines):
        return {"kind": "punct", "decls": [], "target": None}
    frames = frames_at.get(line, ())
    ls, le = starts[line - 1], starts[line - 1] + len(clines[line - 1])
    if not clines[line - 1].strip():
        return {"kind": "punct", "decls": [], "target": None}
    # inside an initializer: the declaration is the statement that opened the outermost initializer block
    for k, f in enumerate(frames):
        if f[0] == "init":
            outer_cls = next((g[1] for g in reversed(frames[:k]) if g[0] in ("class", "enum")), None)
            kd, ds = _parse_stmt(f[3].rstrip().rstrip("="), outer_cls, "{")
            # mt-p1 pass 3 (rereview act-on 3, device_profile.h D2): a designated initializer `.field = v` on the line
            # also declares a change of that field: its readers (.field / ->field) are users, not only the table's
            tcls = next((t for x in ds for t in x["types"]), None) if ds else None
            for g in re.finditer(r"(?:^|[{,(])\s*((?:\.\s*[A-Za-z_]\w*\s*(?:\[[^\]]*\]\s*)?)+)=(?!=)", clines[line - 1]):
                for fld in re.findall(r"\.\s*([A-Za-z_]\w*)", g.group(1)):
                    ds.append(_d(fld, tcls or "?", "field", ""))
            # mt-p1 pass 4 (REVIEW3 C-1, V6 / V7b): the designators in force on a continuation line (value on the
            # line after `.x =`, or an element of `.x = {1, 1,\n 2, 2}`): in each enclosing initializer brace, the last
            # top-level element before the line (or before the inner brace) when it starts with `.x =`
            cur = ls
            for g2 in reversed(frames[k:]):  # frames[k] is the outermost init; the braces inside it are "inner"
                if g2[0] not in ("init", "inner"):
                    break
                seg = clean[g2[2] + 1:cur]
                dd, last = 0, 0
                for i, c in enumerate(seg):
                    if c in "([{":
                        dd += 1
                    elif c in ")]}":
                        dd -= 1
                    elif c == "," and dd == 0:
                        last = i + 1
                mm2 = re.match(r"\s*((?:\.\s*[A-Za-z_]\w*\s*(?:\[[^\]]*\]\s*)?)+)=(?!=)", seg[last:])
                if mm2:
                    for fld in re.findall(r"\.\s*([A-Za-z_]\w*)", mm2.group(1)):
                        if not any(x["name"] == fld and x["what"] == "field" for x in ds):
                            ds.append(_d(fld, tcls or "?", "field", ""))
                cur = g2[2]
            return {"kind": kd if ds else "unparsed", "decls": ds, "target": None}
    inner = frames[-1] if frames else None
    if inner and inner[0] == "enum":
        txt = clean[max(ls, inner[2] + 1):le]
        cut = txt.find("}")
        if cut >= 0:
            txt = txt[:cut]
        ds = []
        for part in txt.split(","):
            mm = re.match(r"\s*([A-Za-z_]\w*)", part)
            if mm:
                ds.append(_d(mm.group(1), inner[1], "enumerator", "", (_scope_key(frames), cond_stack(src).get(line, ()),
                                                                       " ".join(clines[line - 1].split()))))
        if ds:
            return {"kind": "enumerator", "decls": ds, "target": None}
        if not re.sub(r"[\s{}();,:]", "", txt):
            return {"kind": "punct", "decls": [], "target": None}
        return {"kind": "unparsed", "decls": [], "target": None}
    # statements at class / namespace scope that overlap the line
    lo = inner[2] + 1 if inner else 0
    p = ls
    while p > lo and clean[p - 1] not in ";{}":
        p -= 1
    decls, kinds = [], []
    first = True
    while p < len(clean) and (first or (p < le and clean[p:le].strip())):
        first = False
        q = p
        d = 0
        while q < len(clean):
            c = clean[q]
            if c in "([":
                d += 1
            elif c in ")]":
                d -= 1
            elif c in ";{}" and d <= 0:
                break
            q += 1
        stmt = clean[p:q]
        term = clean[q] if q < len(clean) else ""
        cls = next((g[1] for g in reversed(frames) if g[0] in ("class", "enum")), None) if inner and inner[0] == "class" else None
        if inner and inner[0] == "class":
            cls = inner[1]
        on_line = _strip_lead(clean[max(p, ls):min(q, le)])  # the part of the statement on this line (access spec dropped)
        if (stmt.strip() and on_line) or (term == "{" and ls <= q < le):
            kd, ds = _parse_stmt(stmt, cls, term)
            # the whole statement in its full scope (mt-p1 pass 3, rereview act-on 1: namespaces, enclosing class
            # headers with their specialisation arguments, the #if branch): an identical signature on the other side of
            # the diff = moved / unchanged. "at" = the statement's offset (one statement over several changed lines)
            sline = bisect.bisect_right(starts, p + len(clean[p:q]) - len(clean[p:q].lstrip()))
            for x in ds:
                x["sig"] = (_scope_key(frames), cond_stack(src).get(sline, ()), " ".join(stmt.split()) + term)
                x["at"] = p
            if term == "{" and kd == "decl" and ds and ds[0]["what"] == "type":
                ds[0]["cls"] = None
            kinds.append(kd)
            decls += ds
        if term == "{":
            kd2 = _classify(stmt, list(frames))
            if kd2 == "class":
                frames = frames + ((kd2, _hname(stmt), q, stmt),)
                inner = frames[-1]
                p = q + 1
                continue
            if kd2 == "enum" and q < le:
                end = match.get(q, len(clean))
                for part in clean[q + 1:min(end, le)].split(","):
                    mm = re.match(r"\s*([A-Za-z_]\w*)", part)
                    if mm:
                        decls.append(_d(mm.group(1), _hname(stmt), "enumerator", ""))
                p = end + 1
                continue
            p = match.get(q, len(clean)) + 1
            continue
        if term == "}":
            frames = frames[:-1] if frames else frames
            inner = frames[-1] if frames else None
        p = q + 1
    if decls:
        return {"kind": "decl", "decls": decls, "target": None}
    for k in ("unparsed", "scope", "assert"):
        if k in kinds:
            return {"kind": k, "decls": [], "target": None}
    return {"kind": "punct", "decls": [], "target": None}


def scope_order(src: str, cls: str) -> list:
    """mt-p1 review A1: the declared names of the class / struct / enum `cls` in declaration order (members, member
    functions, enumerators; nested types and inline function bodies left out)"""
    clean, clines, starts, frames_at, match, pp = _prep(src)
    out, seen = [], set()
    for ln in range(1, len(clines) + 1):
        fr = frames_at.get(ln, ())
        if not fr or fr[-1][0] not in ("class", "enum") or fr[-1][1] != cls or ln in pp or not clines[ln - 1].strip():
            continue
        for x in declared(src, ln)["decls"]:
            if x["cls"] == cls and x["name"] not in seen:
                seen.add(x["name"])
                out.append(x["name"])
    return out


def ns_sequence(src: str) -> list:
    """mt-p1 pass 3 (rereview act-on 2, P1 / P2): the namespace-scope entities of a file in source order: every
    declaration statement outside classes and functions (its full signature) and every function definition
    (('fn', qualified name)). Order at namespace scope changes meaning (two-phase lookup at a template's definition,
    same-TU dynamic initialisation order), so a same-file "move" is accepted only when this order is unchanged."""
    clean, clines, starts, frames_at, match, pp = _prep(src)
    sc = cxxscan.scan(src)
    fstart = {f.start: f for f in sc.funcs}
    out, seen_at = [], set()
    ln = 1
    while ln <= len(clines):
        f = fstart.get(ln)
        if f is not None:
            out.append(("fn", f.name))
            ln = max(ln, f.end) + 1
            continue
        fr = frames_at.get(ln, ())
        if clines[ln - 1].strip() and ln not in pp and not any(g[0] in ("class", "enum", "init", "func", "inner") for g in fr):
            for x in declared(src, ln)["decls"]:
                k = (x.get("at"), x["name"])
                if x.get("sig") and k not in seen_at and x["cls"] is None:
                    seen_at.add(k)
                    out.append(x["sig"])
        ln += 1
    return out


def brace_partners(src: str, line: int) -> list:
    """1-based lines of the partner of every '{' / '}' on line (comments, strings, preprocessor lines ignored);
    None for a brace with no partner. impact.py: a changed brace line whose partner line is unchanged means a
    block moved its edge (scope of the unchanged lines between them changed): fail closed."""
    clean, clines, starts, frames_at, match, pp = _prep(src)
    if line < 1 or line > len(clines):
        return []
    rev = {v: k for k, v in match.items()}
    out = []
    import bisect
    for p in range(starts[line - 1], starts[line - 1] + len(clines[line - 1])):
        c = clean[p]
        if c in "{}":
            q = match.get(p) if c == "{" else rev.get(p)
            out.append(None if q is None else bisect.bisect_right(starts, q))
    return out
