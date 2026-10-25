#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fn_equiv.py - V-move check of arch025.PLAN.md: a function moved to another file (or another namespace) is the
same code. For every function of the OLD side, the NEW side must have a function of the same name whose text is
identical after whitespace is ignored (``diff -w``); include lines, namespace braces and comments never count.

Grew out of the s15-review fn.py (extract one function by name, diff -w); arch025-S0a adds whole-file mode,
qualified names (Model::f), overloads (compared in order), braces inside strings / comments, renames and several
files per side.

  python -I tools/impact/fn_equiv.py --old A.cpp [--old B.cpp] --new C.cpp [--new D.cpp] [--fn name ...]
         [--map old=new ...] [--ignore name ...] [--quiet]
  python -I tools/impact/fn_equiv.py --selftest

A side may also be ``REV:path`` (read with ``git show REV:path`` from --repo, default the current directory).
--fn limits the check to these names (default: every function defined on the OLD side). --map compares OLD's
``old`` with NEW's ``new`` (a renamed function, e.g. loadOrTune=loadOrTuneImpl). --ignore skips a name.
Exit 0: all equal; 1: a function differs or is missing on the NEW side; 2: usage / unreadable input.
Last line: "fn_equiv: ok (N functions equal)" or "fn_equiv: FAIL (...)".
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile

_NAME = r"(?:~?[A-Za-z_]\w*(?:<[^<>;{}()]*>)?::)*(?:operator\s*[^\s(]+|~?[A-Za-z_]\w*)"
# a definition starts at a line that is not indented more than one namespace level would be (we allow any
# indentation of 0-4 spaces), begins with a type / qualifier / name and holds `name(`
_HEAD = re.compile(r"^ {0,4}(?![#}/])(?!(?:if|for|while|switch|return|else|do|case|namespace|using|typedef)\b)"
                   r"[A-Za-z_~].*?\b(" + _NAME + r")\s*\(")
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "decltype", "static_assert", "alignof",
             "defined", "noexcept", "requires"}


def strip_code(text: str) -> str:
    """Comments removed, string / char literal contents blanked (same length per line, so line numbers stay)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == "R" and text.startswith('R"', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'R"([^()\\ ]{0,16})\(', text[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            end = ")" + m.group(1) + '"'
            j = text.find(end, i + m.end())
            j = n if j < 0 else j + len(end)
            out.append('R"' + "".join(ch if ch == "\n" else " " for ch in text[i + 2:j - 1]) + '"')
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + " " * max(0, j - i - 2) + (c if j - i >= 2 else ""))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def functions(text: str) -> list[tuple[str, str]]:
    """[(name, body text)] of every function definition with a body, in file order (name without qualifiers)."""
    code = strip_code(text)
    lines = text.split("\n")
    clines = code.split("\n")
    res: list[tuple[str, str]] = []
    i = 0
    while i < len(clines):
        m = _HEAD.match(clines[i])
        if not m:
            i += 1
            continue
        name = re.sub(r"\s+", "", m.group(1))
        base = name.split("::")[-1]
        if base in _KEYWORDS:
            i += 1
            continue
        # from the head: the first ';' or '{' at parenthesis depth 0 decides declaration or definition
        depth, j, kind, pos = 0, i, None, None
        start_col = m.start(1)
        while j < len(clines) and kind is None:
            seg = clines[j][start_col:] if j == i else clines[j]
            for k, ch in enumerate(seg):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif depth == 0 and ch == ";":
                    kind = "decl"
                    break
                elif depth == 0 and ch == "{":
                    kind = "def"
                    pos = (j, k + (start_col if j == i else 0))
                    break
                elif depth == 0 and ch == "=" and seg[k:k + 2] != "==" and kind is None and j == i:
                    # `T x = f(...)` style initialisers are not definitions
                    pass
            if kind is None:
                j += 1
                if j - i > 12:
                    break
        if kind != "def":
            i += 1
            continue
        # body: brace matching on the stripped text
        bj, bk = pos
        d = 0
        end = None
        for r in range(bj, len(clines)):
            seg = clines[r][bk:] if r == bj else clines[r]
            for ch in seg:
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                    if d == 0:
                        end = r
                        break
            if end is not None:
                break
        if end is None:
            i += 1
            continue
        res.append((base, "\n".join(lines[i:end + 1])))
        i = end + 1
    return res


def normalize(body: str) -> list[str]:
    """diff -w over the whole body: comments removed, every whitespace run removed (line breaks too, so a reflowed
    signature or brace still compares equal), string / char literal text kept as written, and the qualifiers of the
    function's own name dropped (Model::f in one file, f inside the class's namespace in another). Returns the
    body as tokens-without-spaces split at ';', '{' and '}' (only for a readable diff)."""
    code = strip_code(body)
    merged = []
    inlit = False
    q = ""
    for a, b in zip(body, code):
        if not inlit and b in "\"'" and a == b:
            inlit, q = True, b
            merged.append(b)
        elif inlit:
            merged.append(a)
            if a == q and b == q:
                inlit = False
        else:
            merged.append(b)
    text = "".join(merged)
    # the head's qualifiers: in the text before the first '(' only
    k = text.find("(")
    if k > 0:
        text = re.sub(r"(?<![\w:])(?:\w+(?:<[^<>]*>)?\s*::\s*)+(?=~?\w+\s*$)", "", text[:k]) + text[k:]
    flat = re.sub(r"\s+", "", text)
    return [t for t in re.split(r"(?<=[;{}])", flat) if t]


def read_side(spec: str, repo: str) -> str:
    if os.path.exists(spec):
        with open(spec, encoding="utf-8", errors="replace") as f:
            return f.read()
    if ":" in spec and not re.match(r"^[A-Za-z]:[\\/]", spec):
        rev, path = spec.split(":", 1)
        r = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return r.stdout
        raise OSError(f"git show {spec}: {r.stderr.strip()}")
    raise OSError(f"cannot read {spec}")


def compare(old_texts: list[str], new_texts: list[str], only: list[str], renames: dict[str, str], ignore: set[str],
            quiet: bool, out=print) -> int:
    old: dict[str, list[str]] = {}
    for t in old_texts:
        for n, b in functions(t):
            old.setdefault(n, []).append(b)
    new: dict[str, list[str]] = {}
    for t in new_texts:
        for n, b in functions(t):
            new.setdefault(n, []).append(b)
    names = only if only else list(old.keys())
    same = differ = missing = 0
    for n in names:
        if n in ignore:
            continue
        if n not in old:
            out(f"  MISSING-OLD {n}: no definition on the old side")
            missing += 1
            continue
        nn = renames.get(n, n)
        if nn not in new:
            out(f"  MISSING {n}: no definition named {nn} on the new side")
            missing += 1
            continue
        a, b = old[n], new[nn]
        if len(a) != len(b):
            out(f"  DIFF {n}: {len(a)} definition(s) on the old side, {len(b)} on the new side")
            differ += 1
            continue
        for k, (x, y) in enumerate(zip(a, b)):
            nx, ny = normalize(x), normalize(y)
            if nn != n and ny:
                ny[0] = ny[0].replace(nn + "(", n + "(", 1)
            tag = n if len(a) == 1 else f"{n}#{k}"
            if nx == ny:
                same += 1
                if not quiet:
                    out(f"  same {tag}" + (f" (as {nn})" if nn != n else ""))
            else:
                differ += 1
                out(f"  DIFF {tag}:")
                for l in list(difflib.unified_diff(nx, ny, "old", "new", n=1, lineterm=""))[:40]:
                    out("    " + l)
    if differ or missing:
        out(f"fn_equiv: FAIL ({differ} differ, {missing} missing, {same} equal)")
        return 1
    out(f"fn_equiv: ok ({same} functions equal)")
    return 0


def selftest() -> int:
    cases = []
    OLD = r'''#include "a.h"
namespace whirl::server {
namespace {
int helper(int x) {
    return x + 1;  // one
}
}  // namespace

void Engine::step(int n,
                  int m) {
    const char* s = "{ not a brace";
    if (n > m) {
        helper(n);
    }
}
int over(int a) { return a; }
int over(double a) { return static_cast<int>(a) * 2; }
void decl_only(int);
}  // namespace whirl::server
'''
    MOVED = r'''#include "b.h"
#include "c.h"
namespace whirl::server::engine_impl {

// moved here
int helper(int x)
{
  return x+1;
}

void Engine::step(int n, int m) {
    const char* s = "{ not a brace";   /* } */
    if (n > m) { helper(n); }
}
int over(int a) { return a; }
int over(double a) { return static_cast<int>(a) * 2; }
}
'''
    CHANGED = MOVED.replace("return x+1;", "return x+2;")
    STRING = MOVED.replace('"{ not a brace"', '"{ not a brace!"')
    RENAMED = MOVED.replace("void Engine::step(", "void Engine::stepImpl(")
    ORDER = MOVED.replace("int over(int a) { return a; }\nint over(double a) { return static_cast<int>(a) * 2; }",
                          "int over(double a) { return static_cast<int>(a) * 2; }\nint over(int a) { return a; }")
    cases = [
        ("moved + reformatted + other namespace: equal", [OLD], [MOVED], [], {}, set(), 0),
        ("one body changed: FAIL", [OLD], [CHANGED], [], {}, set(), 1),
        ("string literal changed: FAIL", [OLD], [STRING], [], {}, set(), 1),
        ("missing on the new side: FAIL", [OLD], [MOVED.replace("int helper(int x)\n{\n  return x+1;\n}\n", "")], [], {}, set(), 1),
        ("renamed without --map: FAIL", [OLD], [RENAMED], [], {}, set(), 1),
        ("renamed with --map: equal", [OLD], [RENAMED], [], {"step": "stepImpl"}, set(), 0),
        ("overloads swapped: FAIL", [OLD], [ORDER], [], {}, set(), 1),
        ("--fn limits the check", [OLD], [CHANGED], ["step", "over"], {}, set(), 0),
        ("--ignore skips a name", [OLD], [CHANGED], [], {}, {"helper"}, 0),
        ("split over two new files: equal", [OLD], [MOVED.split("void Engine::step")[0] + "}\n",
                                                   "namespace x {\nvoid Engine::step" + MOVED.split("void Engine::step")[1]], [], {}, set(), 0),
        ("--fn name not on the old side: FAIL", [OLD], [MOVED], ["nosuch"], {}, set(), 1),
        ("qualified head vs in-class-namespace head: equal", [OLD], [MOVED.replace("void Engine::step(", "void step(")], [], {}, set(), 0),
        ("qualifier inside the body still counts: FAIL", [OLD], [MOVED.replace("helper(n);", "detail::helper(n);")], [], {}, set(), 1),
    ]
    fails = 0
    # extraction facts
    fs = functions(OLD)
    names = [n for n, _ in fs]
    ok = names == ["helper", "step", "over", "over"]
    print(f"  {'ok  ' if ok else 'FAIL'} extraction: {names} (decl_only is a declaration, not listed)")
    fails += 0 if ok else 1
    for (title, o, n, only, ren, ign, want) in cases:
        lines: list[str] = []
        rc = compare(o, n, only, ren, ign, True, out=lines.append)
        good = rc == want
        fails += 0 if good else 1
        print(f"  {'ok  ' if good else 'FAIL'} {title}: rc {rc} (want {want}) | {lines[-1] if lines else ''}")
    # file and CLI path
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "old.cpp"), os.path.join(d, "new.cpp")
        with open(a, "w", encoding="utf-8") as f:
            f.write(OLD)
        with open(b, "w", encoding="utf-8") as f:
            f.write(MOVED)
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--old", a, "--new", b, "--quiet"],
                           capture_output=True, text=True)
        good = r.returncode == 0 and r.stdout.strip().endswith("fn_equiv: ok (4 functions equal)")
        fails += 0 if good else 1
        print(f"  {'ok  ' if good else 'FAIL'} CLI on files: rc {r.returncode} | {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()}")
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--old", os.path.join(d, "nope.cpp"), "--new", b],
                           capture_output=True, text=True)
        good = r.returncode == 2
        fails += 0 if good else 1
        print(f"  {'ok  ' if good else 'FAIL'} unreadable input is a usage error: rc {r.returncode}")
    total = len(cases) + 3
    if fails:
        print(f"fn_equiv selftest: FAIL ({fails} of {total})")
        return 1
    print(f"fn_equiv selftest: ok ({total} cases)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", action="append", default=[])
    ap.add_argument("--new", action="append", default=[])
    ap.add_argument("--fn", action="append", default=[])
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.old or not a.new:
        print("fn_equiv: FAIL (usage: --old and --new are required)")
        return 2
    renames = {}
    for m in a.map:
        if "=" not in m:
            print(f"fn_equiv: FAIL (usage: --map {m}: want old=new)")
            return 2
        k, v = m.split("=", 1)
        renames[k] = v
    try:
        olds = [read_side(s, a.repo) for s in a.old]
        news = [read_side(s, a.repo) for s in a.new]
    except OSError as e:
        print(f"fn_equiv: FAIL (usage: {e})")
        return 2
    return compare(olds, news, a.fn, renames, set(a.ignore), a.quiet)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
