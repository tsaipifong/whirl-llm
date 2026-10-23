# SPDX-License-Identifier: Apache-2.0
"""Light C++ / HIP scanner for the impact map (no compiler, stdlib only).

Gives, per source file: the #include list (with the arch guard each include sits under), the function
definitions (qualified name, first / last line) and, per function, the identifiers it calls and the
members it touches. It is a text scanner, not a parser: it over-approximates (a call `run(` links to
every function named `run`), which is the safe direction for an impact map. Generic names are cut by
the caller (impact_build.py, STOP_NAMES / max fan-in).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "alignof", "decltype", "static_assert",
    "new", "delete", "throw", "case", "do", "else", "typeid", "noexcept", "requires", "co_await", "co_return",
    "defined", "__launch_bounds__", "alignas", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
}
# std / library names that never resolve to a WHIRL function
STD_CALLS = {
    "size", "data", "begin", "end", "push_back", "emplace_back", "resize", "reserve", "clear", "empty", "find",
    "insert", "erase", "count", "at", "front", "back", "substr", "c_str", "min", "max", "move", "forward", "swap",
    "get", "set", "reset", "release", "load", "store", "lock", "unlock", "join", "printf", "fprintf", "snprintf",
    "format", "to_string", "memcpy", "memset", "strlen", "abs", "fabs", "exp", "log", "sqrt", "floor", "ceil",
    "make_unique", "make_shared", "emplace", "append", "assign", "value_or", "has_value", "value", "first", "second",
    "fill", "copy", "sort", "accumulate", "string", "vector", "array", "optional", "pair", "tuple", "span",
    "operator", "assert", "fflush", "fopen", "fclose", "fwrite", "fread", "wait", "notify_one", "notify_all",
    "push", "pop", "top", "rfind", "compare", "starts_with", "ends_with", "what", "str", "clamp", "round",
}


@dataclass
class Func:
    name: str          # qualified, e.g. whirl::server::Engine::decodeCycle -> "Engine::decodeCycle"
    short: str         # last component
    start: int         # 1-based line of the header
    end: int           # 1-based line of the closing brace
    calls: set = field(default_factory=set)
    members: set = field(default_factory=set)
    idents: set = field(default_factory=set)


@dataclass
class FileScan:
    includes: list    # (header text, guard) guard: None | "gfx1151" | "not_gfx1151" | other macro text
    funcs: list       # Func
    strings: list     # (line, literal) string literals (kernel names, env names)
    classes: list = field(default_factory=list)  # (name, start line, end line) of class / struct / union bodies (arch024-S12)


_ws = re.compile(r"\s+")


def _blank(text: str) -> tuple[str, list]:
    """Comments, strings and char literals -> spaces (newlines kept). Returns (clean, string literals)."""
    out = []
    lits = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            seg = text[i:j]
            line += seg.count("\n")
            out.append(re.sub(r"[^\n]", " ", seg))
            i = j
        elif c == 'R' and text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'R"([^(\s]*)\(', text[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            close = ")" + m.group(1) + '"'
            j = text.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            seg = text[i:j]
            lits.append((line, seg[m.end():-len(close)] if j <= n else ""))
            line += seg.count("\n")
            out.append(re.sub(r"[^\n]", " ", seg))
            i = j
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            q = c
            j = i + 1
            while j < n and text[j] != q and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            if q == '"':
                lits.append((line, text[i + 1:j - 1]))
            out.append(" " * (j - i))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out), lits


_inc = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')
_if = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _includes(raw: str) -> list:
    res = []
    stack = []  # guard text per #if level
    for l in raw.splitlines():
        m = _if.match(l)
        if m:
            kw, rest = m.group(1), m.group(2).strip()
            if kw in ("if", "ifdef", "ifndef"):
                g = rest if kw == "if" else (("defined(%s)" if kw == "ifdef" else "!defined(%s)") % rest)
                stack.append(g)
            elif kw in ("elif", "else"):
                if stack:
                    stack[-1] = "else(" + stack[-1] + ")" if kw == "else" else rest
            elif kw == "endif" and stack:
                stack.pop()
            continue
        m = _inc.match(l)
        if m:
            guard = None
            for gtxt in stack:
                t = gtxt.replace(" ", "")
                if t in ("!defined(__gfx1151__)", "else(defined(__gfx1151__))"):
                    guard = "not_gfx1151"
                elif t in ("defined(__gfx1151__)", "else(!defined(__gfx1151__))"):
                    guard = "gfx1151"
                elif t in ("defined(__gfx1201__)",):
                    guard = "gfx1201"
            res.append((m.group(1), guard))
    return res


_hdr_tail = re.compile(r"\)\s*(const|noexcept(\([^)]*\))?|override|final|mutable|&|&&|\s|->\s*[\w:<>,\s\*&]+)*$")


def _func_name(header: str):
    """Name of the function a '{' opens, or None. header: text since the previous ; { or }."""
    h = header.strip()
    if not h or "(" not in h:
        return None
    if re.match(r"^(namespace|struct|class|union|enum|extern)\b", h):
        return None
    top = []  # the header outside parentheses (default arguments may hold '=')
    d = 0
    for ch in h:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        elif d == 0:
            top.append(ch)
    top = "".join(top)
    if re.search(r"(^|[^=!<>])=($|[^=])", top) and not re.search(r"operator\s*\S*=", top):
        return None  # initializer: x = {...}, or a lambda assigned at namespace scope
    # constructor initializer list:  Foo::Foo(args) : a_(x), b_{y}   -> cut at the ':' after the first ')'
    depth = 0
    first_close = -1
    for k, ch in enumerate(h):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                first_close = k
                break
    if first_close < 0:
        return None
    tail = h[first_close:]
    m_init = re.match(r"\)\s*(noexcept)?\s*:(?!:)", tail)
    if not m_init and not _hdr_tail.match(tail):
        return None
    pre = h[:h.index("(")].rstrip()
    m = re.search(r"((?:[A-Za-z_]\w*(?:<[^<>]*>)?\s*::\s*)*~?(?:operator\s*\S+|[A-Za-z_]\w*))\s*$", pre)
    if not m:
        return None
    name = _ws.sub("", m.group(1))
    name = re.sub(r"<[^<>]*>", "", name)
    short = name.split("::")[-1]
    if short in KEYWORDS:
        return None
    return name


_ident_call = re.compile(r"([A-Za-z_]\w*)\s*(?:<[^<>;(){}]*>)?\s*\(")
_member = re.compile(r"(?:\.|->)\s*([A-Za-z_]\w*)")
_ident = re.compile(r"[A-Za-z_]\w*")
_local_callable = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*\[|std::function<[^;{}]*>\s*([A-Za-z_]\w*)")


def scan(raw: str) -> FileScan:
    clean, lits = _blank(raw)
    # preprocessor lines -> blank (keep newlines)
    clean = re.sub(r"(?m)^[ \t]*#.*(?:\\\n.*)*$", lambda m: re.sub(r"[^\n]", " ", m.group(0)), clean)
    funcs = []
    classes = []
    stack = []  # (kind, name, start_pos, start_line)
    last = 0
    line = 1
    pos_line = []  # not stored; compute line incrementally
    i = 0
    n = len(clean)
    cur_line = 1
    seg_start = 0
    while i < n:
        c = clean[i]
        if c == "\n":
            cur_line += 1
        elif c in ";":
            if not stack or stack[-1][0] != "func":
                seg_start = i + 1
        elif c == "{":
            in_func = any(s[0] == "func" for s in stack)
            if in_func:
                stack.append(("inner", None, i, cur_line))
            else:
                header = clean[seg_start:i]
                hs = header.strip()
                fname = _func_name(header)
                if fname:
                    hl = cur_line - header[len(header.rstrip()):].count("\n") - hs.count("\n")
                    stack.append(("func", fname, i, max(1, hl)))
                elif re.match(r"^(?:inline\s+)?namespace\b", hs) or re.match(r'^extern\s+"?C?', hs) or hs.startswith("extern"):
                    stack.append(("ns", None, i, cur_line))
                elif re.search(r"\b(class|struct|union)\s+(?:alignas\([^)]*\)\s*)?([A-Za-z_]\w*)[^(){};]*$", _untemplate(hs)):
                    # arch024-S12 (review N5): `template <class T> struct Foo {` names Foo, not T
                    m = re.search(r"\b(class|struct|union)\s+(?:alignas\([^)]*\)\s*)?([A-Za-z_]\w*)[^(){};]*$", _untemplate(hs))
                    stack.append(("class", m.group(2), i, cur_line))
                else:
                    stack.append(("other", None, i, cur_line))
            seg_start = i + 1
        elif c == "}":
            if stack:
                kind, name, spos, sline = stack.pop()
                if kind == "class":
                    classes.append((name, sline, cur_line))
                if kind == "func":
                    cls = [s[1] for s in stack if s[0] == "class"]
                    q = "::".join(cls + [name]) if cls and "::" not in name else name
                    body = clean[spos:i]
                    f = Func(name=q, short=name.split("::")[-1], start=sline, end=cur_line)
                    # local callables (lambdas, std::function variables) are not calls to other functions
                    local = {x for t in _local_callable.findall(body) for x in t if x}
                    for m in _ident_call.finditer(body):
                        nm = m.group(1)
                        if nm not in KEYWORDS and nm not in local:
                            f.calls.add(nm)
                    f.members = set(_member.findall(body))
                    f.idents = set(_ident.findall(body))
                    funcs.append(f)
            if not stack or stack[-1][0] != "func":
                seg_start = i + 1
        i += 1
    return FileScan(includes=_includes(raw), funcs=funcs, strings=lits, classes=classes)


def _untemplate(h: str) -> str:
    """header text without a leading template parameter list (nested <> allowed)"""
    m = re.match(r"\s*template\s*<", h)
    if not m:
        return h
    depth, i = 1, m.end()
    while i < len(h) and depth:
        depth += {"<": 1, ">": -1}.get(h[i], 0)
        i += 1
    return h[i:]


def func_at(fs: FileScan, line: int):
    """Innermost function whose [start, end] contains line (functions do not nest in this scan)."""
    best = None
    for f in fs.funcs:
        if f.start <= line <= f.end and (best is None or f.end - f.start < best.end - best.start):
            best = f
    return best


def classes_at(fs: FileScan, line: int):
    """Names of the class / struct / union bodies that contain line, innermost last."""
    return [n for n, a, b in sorted(fs.classes, key=lambda c: c[1]) if a <= line <= b]


def blank_code(raw: str) -> str:
    """The text with comments, string literals and preprocessor lines blanked (line structure kept)."""
    clean, _ = _blank(raw)
    return re.sub(r"(?m)^[ \t]*#.*(?:\\n.*)*$", lambda m: re.sub(r"[^\n]", " ", m.group(0)), clean)
