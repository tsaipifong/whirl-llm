# SPDX-License-Identifier: Apache-2.0
# arch025-engine S0 (from scratch/a25_engine/v3/census3.py): census of class Engine AND of every component the cut
# names. Run through tools/gates/engine_split_check.py (census + graph + self-test).
#   python -I engine_split_census.py <src_root> <out_dir> --cut tools/gates/engine_cut.json
# S0 changes against census3 (third review A1-A3, progress/arch025-engine.DESIGN.REVIEW3.md):
#  A1 writes: a member hit is a WRITE when (a) an assignment / ++ / -- ends its access chain (`queue_.front()->id = 0`),
#     (b) any method called on the chain is not on the measured const list (CONST_OK: names that src/ declares only
#     as const member functions, plus std read-only names) - so `.append`, `.update`, `cc.observe` through a binding,
#     `ps_.spes().clear()` are writes, (c) it is bound to a non-const local reference / pointer / range-for reference
#     (`auto& q = stop_;`), (d) it is an argument of a function call (`std::swap(a_, b_)`, `dropAll(queue_)`,
#     `std::fill(x_.begin(), ...)`) unless the callee is on READ_CALLEES, (e) its address is taken (`&x_`) outside a
#     const declaration. Conservative: unknown = write (clang-tidy misc-const-correctness does the same).
#  A2 nested types: a nested struct / class with a body inside a tracked class, and a namespace-level struct / class
#     with a body in an engine file, become entries `type:X` (their inline bodies are scanned); in-class member
#     initializers (`X y_{a_, b_}`, `int n_ = k_;`) are scanned as part of the class's constructor entry.
#  A3 receivers: a local alias of *this / of a tracked class (`Engine& me = *this`, `auto* p = this`,
#     `SlotPages& s = sc_`) is a known receiver; a call `r.f(` / `r->f(` with any other receiver whose name f is a
#     tracked entry and NOT a method of any untracked class in src/ is recorded as a call (conservative); the rest
#     are listed in unresolved.txt (reported, not dropped silently). A data member of a tracked class whose type
#     names Engine or a tracked class, and whose name is not a cut holder of that class, is a FINDING.
# <src_root> = a tree with src/ (git archive c1ca7ce / abe3512 exports).
# v3 changes against census2.py (v2):
#  R1 (blind after the first move): the class bodies parsed are Engine AND every class in cut["classes"] (wherever it is
#     defined under src/); out-of-line definitions are `C::f` for every tracked C; in the "engine files" (files with a
#     tracked definition, files a module lists in cut["files"], and files that include a tracked header but are not
#     in cut["users"]) EVERY namespace-level function definition is an entry: an untracked `X::f` or an unknown free
#     function becomes an entry the cut does not know -> graph3 fails (unassigned). Free functions the cut lists
#     (cut["free_funcs"]) are found in any file under src/. Entry keys are the cut's names: `C::f` -> cut["aliases"]
#     or f (graph3 checks that f belongs to C's module).
#  R2 (this->, Engine::, macros): a data member is used wherever its identifier appears (any receiver: bare, this->,
#     e., holder_.x, Engine::x); a function is called when its name is followed by `(` and the receiver is none,
#     this, e, Engine, a tracked class, or a holder object (cut["holders"]); `C::f` / `&C::f` without a call also
#     counts. Every `#define` in an engine file, every `#define` anywhere in src/ whose replacement names a tracked
#     data member or class, and every `friend` in a tracked class body is a FINDING (exit 2).
#  R3 support: writes are recorded per entry for every data member (graph3 checks them against the cut's writer lists).
# Outputs: members.tsv, funcs.tsv (+ column 'def' = the qualified definition), files.txt, findings.txt, census3.json.
# Exit: 0 clean, 2 findings (the census itself is complete; graph3 still runs), 1 crash.
import os, re, sys, json

ROOT, OUT = sys.argv[1], sys.argv[2]
CUTP = sys.argv[sys.argv.index("--cut") + 1]
cut = json.load(open(CUTP, encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(ROOT, "src")
CLASSES = {k: v for k, v in cut.get("classes", {}).items() if not k.startswith("_")}  # class -> module
CLASSES.setdefault("Engine", "server.engine")
HOLDERS = {k: v for k, v in cut.get("holders", {}).items() if not k.startswith("_")}   # object name -> class
ALIASES = {k: v for k, v in cut.get("aliases", {}).items() if not k.startswith("_")}   # "C::f" -> entry
FREE = set(f for m, v in cut["modules"].items() for f in v.get("free_funcs", []))
USERS = set(cut.get("users", []))          # files allowed to include engine headers without being engine files
LISTED = set(f for m, v in cut["modules"].items() for f in v.get("files", []))
KW = {"if", "for", "while", "switch", "return", "sizeof", "catch", "static_assert", "decltype", "alignof", "noexcept",
      "defined", "new", "delete", "throw", "operator", "assert"}
findings = []


def strip(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith("//", i):
            j = s.find("\n", i); j = n if j < 0 else j
            out.append(" " * (j - i)); i = j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2); j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", s[i:j])); i = j
        elif s.startswith('R"', i) and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
            m = re.match(r'R"([^(]*)\(', s[i:])
            end = s.find(")" + m.group(1) + '"', i)
            j = end + len(m.group(1)) + 2
            out.append(re.sub(r"[^\n]", " ", s[i:j])); i = j
        elif c == "'" and i > 0 and s[i - 1].isdigit() and i + 1 < n and s[i + 1].isdigit():
            out.append(" "); i += 1
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            out.append(c + " " * (j - i - 1) + c); i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out)


def match_close(s, i, op, cl):
    d = 0
    while i < len(s):
        if s[i] == op: d += 1
        elif s[i] == cl:
            d -= 1
            if d == 0: return i
        i += 1
    raise ValueError("unbalanced")


def detpl(x):
    prev = None
    while prev != x:
        prev = x; x = re.sub(r"<[^<>;{}]*>", "", x)
    return x


def lineno(s, i):
    return s.count("\n", 0, i) + 1


def code_depth(s):
    stack, cdepth, cd, last = [], [0] * (len(s) + 1), 0, 0
    for k, ch in enumerate(s):
        cdepth[k] = cd
        if ch == "{":
            isns = re.search(r"(namespace\s*[\w:]*|extern\s*\"\"\s*)\s*$", s[last:k]) is not None
            stack.append(isns)
            if not isns: cd += 1
            last = k + 1
        elif ch == "}":
            if stack and not stack.pop(): cd -= 1
            last = k + 1
        elif ch == ";":
            last = k + 1
    cdepth[len(s)] = cd
    return cdepth


# ---------------- read all of src/ ----------------
texts = {}
for dp, dn, fn in os.walk(SRC):
    for f in sorted(fn):
        if f.endswith((".cpp", ".h", ".hip", ".inc", ".hpp")):
            p = os.path.join(dp, f)
            raw = open(p, encoding="utf-8", errors="replace").read()
            texts[os.path.relpath(p, SRC).replace("\\", "/")] = (raw, strip(raw))
# A1: declarations (const list, parameter kinds) are also read from include/ (wddm_watch.h, mem_governor.h ...);
# a tree without include/ gets fewer const names = more writes (the safe direction).
decl_texts = dict(texts)
for dp, dn, fn in os.walk(os.path.join(ROOT, "include")):
    for f in sorted(fn):
        if f.endswith((".h", ".hpp", ".inc")):
            p = os.path.join(dp, f)
            decl_texts["include:" + os.path.relpath(p, ROOT).replace("\\", "/")] = (None, strip(open(p, encoding="utf-8", errors="replace").read()))

data, funcs, types, statics = [], [], [], []   # data: (name, line, access, decl, class, file)
decls = []   # S0b: (class, member, file, line, start, end) of every data member declaration (revtext compares them)
cdecls = []  # rt3: (class, entry, file, line, start, end) of every OTHER declaration statement of a tracked class (revtext)
bodies = {}  # key -> (file, line, text, def)
branges = {}  # S0d: key -> [(file, start, end)] character offsets of the body text (the AST census maps clang positions)
class_files = {}
inits = {}   # class -> [(file, line, initializer text)]  (A2: in-class member initializers)


def add_body(key, f, ln, txt, dfn, rng=None):
    while key in bodies: key += "#"
    bodies[key] = (f, ln, txt, dfn)
    branges[key] = rng or []
    return key


def code_start(hs, i):
    """rt2 (S6 review act-on 1): the first code character at or after i, past whitespace and access specifiers (hs has its
    comments blanked). An in-class entry starts there: from stmt0 (the end of the statement before) the entry began on
    the line before it, and revtext (which takes whole lines) compared that line too."""
    m = re.compile(r"(?:\s|(?:public|private|protected)\s*:(?!:))*").match(hs, i)
    return m.end()


def parse_class(cname, rel, hs, a):
    cb0 = a.end()
    cb1 = match_close(hs, a.end() - 1, "{", "}")
    i, stmt0, access = cb0, cb0, ("private" if hs[a.start():a.end()].lstrip().startswith("class") else "public")
    while i < cb1:
        c = hs[i]
        if c == "{":
            head = hs[stmt0:i]
            if head.count("(") > head.count(")"):
                i = match_close(hs, i, "{", "}") + 1; continue
            hd = detpl(re.sub(r"WH_\w+\([^)]*\)", "", head)).strip()
            for acc in ("public:", "private:", "protected:"):
                while hd.startswith(acc): access = acc[:-1]; hd = hd[len(acc):].strip()
            j = match_close(hs, i, "{", "}")
            if re.match(r"(struct|class|enum|union)\b", hd):
                tn = re.findall(r"\w+", hd)[1]
                types.append((tn, lineno(hs, stmt0), cname))
                if not hd.startswith("enum"):   # A2: the nested type's inline bodies are an entry type:X
                    s0 = code_start(hs, stmt0)
                    add_body("type:" + tn, rel, lineno(hs, s0), hs[s0:j + 1], f"struct {cname}::{tn} (in-class)", [(rel, s0, j + 1)])
                k = hs.index(";", j)
                if hd.startswith("enum"):   # rt3: an enum's enumerators were in no entry; revtext compares the definition
                    me = re.match(r"enum\s+(?:(?:class|struct)\s+)?(\w+)", hd)
                    s0 = code_start(hs, stmt0)
                    cdecls.append((cname, "decl:enum:" + (me.group(1) if me else "(anonymous)"), rel, lineno(hs, s0), s0, k + 1))
                i = k + 1; stmt0 = i; continue
            hd = re.sub(r"\bdecltype\s*\(\s*auto\s*\)", "decltype_auto", hd)   # S0c (N11): a return type, not a call
            if "(" in hd.split("=")[0] and re.search(r"\w\s*$", hd.split("(")[0]) is not None:
                pre = hd.split("(")[0]
                nm = re.search(r"(operator\s*\S+|~?\w+)\s*$", pre).group(1)
                funcs.append((nm, lineno(hs, stmt0), access, re.sub(r"\s+", " ", hd)[:90], cname))
                s0 = code_start(hs, stmt0)
                add_body(resolve(cname, nm), rel, lineno(hs, s0), hs[s0:j + 1], f"{cname}::{nm} (inline)", [(rel, s0, j + 1)])
                i = j + 1; stmt0 = i; continue
            i = j + 1; continue
        if c == ";":
            s = re.sub(r"\s+", " ", hs[stmt0:i]).strip()
            a0 = stmt0 + (len(hs[stmt0:i]) - len(hs[stmt0:i].lstrip()))
            ln = lineno(hs, a0)
            span = (a0, i + 1)   # S0b: the declaration statement's offsets (strip() keeps offsets: raw[a0:a1] is the source)
            stmt0 = i + 1; i += 1
            for acc in ("public:", "private:", "protected:"):
                while s.startswith(acc):
                    access = acc[:-1]; s = s[len(acc):].strip()
            if s.startswith("friend"):
                findings.append(f"friend in tracked class {cname} ({rel}:{ln}): {s[:80]}")
                continue
            # rt3 (rt2 review act-on 1): every declaration statement that is not a data member is a class_decls row
            # (revtext compares the whole statement: default arguments, override, constexpr values, aliases)
            if s.startswith("using "):
                mu = re.match(r"using\s+(\w+)\s*=", s) or re.search(r"(\w+)\s*$", s)
                cdecls.append((cname, "decl:using:" + (mu.group(1) if mu else s), rel, ln, span[0], span[1]))
            elif s.startswith("static_assert"):
                cdecls.append((cname, "decl:static_assert", rel, ln, span[0], span[1]))
            if not s or s.startswith(("using ", "static_assert")):
                continue
            m = re.match(r"(struct|class) (\w+)$", s)
            if m:
                cdecls.append((cname, "decl:fwd:" + m.group(2), rel, ln, span[0], span[1]))
                types.append((m.group(2), ln, cname)); continue
            s2 = re.sub(r"\bdecltype\s*\(\s*auto\s*\)", "decltype_auto", detpl(re.sub(r"WH_\w+\([^)]*\)", "", s)))
            if s2.startswith("static constexpr"):
                statics.append((re.search(r"(\w+)\s*=", s2).group(1), ln))
                cdecls.append((cname, "member:" + statics[-1][0], rel, ln, span[0], span[1])); continue
            if "(" in re.sub(r"operator\s*=", "operator_assign", s2).split("=")[0] and not re.match(r"[\w:<>,\s*&]+\s+\w+\s*\{", s2):
                pre = s2.split("(")[0]
                nm = re.search(r"(operator\s*\S+|~?\w+)\s*$", pre).group(1)
                cdecls.append((cname, "decl:" + re.sub(r"\s+", "", nm), rel, ln, span[0], span[1]))
                funcs.append((nm, ln, access, s[:90], cname)); continue
            s3 = re.sub(r"\{[^{}]*\}", "", s2)
            parts, d, cur = [], 0, ""
            for ch in s3:
                if ch in "([": d += 1
                if ch in ")]": d -= 1
                if ch == "," and d == 0: parts.append(cur); cur = ""
                else: cur += ch
            parts.append(cur)
            for p in parts:
                w = re.findall(r"\w+", p.split("=")[0])
                if w:
                    data.append((w[-1], ln, access, s[:90], cname, rel))
                    decls.append((cname, w[-1], rel, ln, span[0], span[1]))
            # A2: in-class initializers (`= expr` or `{...}` after the declarator) belong to the constructor entry
            ini, d2, cur2, parts2 = [], 0, "", []
            for ch in s2:                     # declarators at depth 0 (commas inside (), [], {} , <> do not split)
                if ch in "([{<": d2 += 1
                if ch in ")]}>": d2 -= 1
                if ch == "," and d2 == 0: parts2.append(cur2); cur2 = ""
                else: cur2 += ch
            parts2.append(cur2)
            for p in parts2:
                m_eq = re.search(r"(?<![=!<>])=(?!=)(.*)$", p)
                if m_eq: ini.append(m_eq.group(1))
                mb = re.search(r"\w\s*(\{.*\})\s*$", p.split("=")[0])
                if mb: ini.append(mb.group(1))
            ini = [x for x in ini if re.search(r"[A-Za-z_]", x)]   # literal initializers use nothing
            if any(x.strip() for x in ini):
                inits.setdefault(cname, []).append((rel, ln, " ; ".join(ini), span))
            continue
        if c == ":" and hs[stmt0:i].strip() in ("public", "private", "protected"):
            access = hs[stmt0:i].strip(); stmt0 = i + 1
        i += 1


def resolve(cname, nm):
    q = f"{cname}::{nm}"
    if q in ALIASES: return ALIASES[q]
    if cname == "Engine" or cname in CLASSES: return nm
    return q


# class bodies (tracked classes), wherever they are
for rel, (raw, s) in texts.items():
    cd = None
    for cname in CLASSES:
        for a in re.finditer(r"(?:class|struct)\s+" + re.escape(cname) + r"\b(?:\s+final)?\s*(?::(?!:)[^;{]*)?\{", s):
            if cd is None: cd = code_depth(s)
            if cd[a.start()] != 0: continue
            class_files.setdefault(cname, []).append(rel)
            parse_class(cname, rel, s, a)

tracked_hdr_paths = {f for v in class_files.values() for f in v if f.endswith(".h")}
QUAL = "|".join(re.escape(c) for c in CLASSES)
files_with_defs = {}
engine_files = set()


def inc_paths(rel, raw):
    out = set()
    for x in re.findall(r'#include\s+"([^"]+)"', raw):
        out.add(os.path.normpath(os.path.join(os.path.dirname(rel), x)).replace(os.sep, "/"))
        out.add(os.path.normpath(x).replace(os.sep, "/"))
    return out


for rel, (raw, s) in texts.items():
    if rel in USERS: continue
    if re.search(r"(?<![\w:])(?:" + QUAL + r")::", s): engine_files.add(rel)
    if rel in LISTED: engine_files.add(rel)
    if inc_paths(rel, raw) & tracked_hdr_paths and not rel.endswith(".h") \
            and not any(rel in v for v in class_files.values()):
        engine_files.add(rel)
        if rel not in LISTED and not re.search(r"(?<![\w:])(?:" + QUAL + r")::", s):
            findings.append(f"file includes an engine header but is neither a module file nor a listed user: {rel}")

for rel, (raw, s) in sorted(texts.items()):
    is_eng = rel in engine_files
    if not is_eng and not re.search(r"(?<![\w:])(?:" + QUAL + r")::", s) and not any(n in s for n in FREE):
        continue
    cd = code_depth(s)
    if is_eng:
        for m in re.finditer(r"^[ \t]*#[ \t]*define\b[^\n]*", raw, re.M):
            findings.append(f"#define in an engine file {rel}:{lineno(raw, m.start())}: {m.group(0).strip()[:80]}")
    # qualified definitions X::f( / X::Y::f( and nested types struct X::Y
    pat = r"(?<![\w:])(\w+)::(?:(\w+)::)?(~?\w+)\s*\(|(?:struct|class)\s+(\w+)::(\w+)\b"
    for m in re.finditer(pat, s):
        if cd[m.start()] != 0: continue
        if m.group(5):
            owner = m.group(4)
            if owner not in CLASSES and not is_eng: continue
            k = m.end()
            while k < len(s) and s[k] not in "{;": k += 1
            if k >= len(s) or s[k] == ";": continue
            e = match_close(s, k, "{", "}")
            key = add_body("type:" + m.group(5), rel, lineno(s, m.start()), s[m.start():e + 1], f"struct {owner}::{m.group(5)}", [(rel, m.start(), e + 1)])
            files_with_defs.setdefault(rel, []).append(key)
            continue
        owner, sub, nm = m.group(1), m.group(2), m.group(3)
        if owner not in CLASSES and not is_eng: continue
        if owner in ("std", "whirl", "server", "chat", "qwen35", "tier", "memgov", "wddmwatch", "spec", "sampling",
                     "device", "vision", "protocol", "detail"):
            continue
        # must be a definition: preceded by a declaration context, followed by (...) quals [: init] {
        before = s[max(0, m.start() - 200):m.start()]
        if re.search(r"(=|return|\(|,|\.|->|!|&&|\|\||\?)\s*[&*]?\s*$", before): continue
        j = match_close(s, m.end() - 1, "(", ")") + 1
        while True:
            mm = re.match(r"\s*(const|noexcept|override|final|&&|&)(?!\w)", s[j:])
            mm2 = re.match(r"\s*WH_\w+\([^)]*\)", s[j:])
            if mm2: j += mm2.end(); continue
            if mm and mm.group(1): j += mm.end(); continue
            break
        mm = re.match(r"\s*->\s*[\w:<>,\s*&]+?(?=\s*\{)", s[j:])
        if mm: j += mm.end()
        mm = re.match(r"\s*:(?!:)", s[j:])
        if mm:
            j += mm.end()
            while True:
                mm = re.match(r"\s*[\w:]+(?:<[^>]*>)?\s*", s[j:])
                j += mm.end()
                if s[j] == "(": j = match_close(s, j, "(", ")") + 1
                elif s[j] == "{": j = match_close(s, j, "{", "}") + 1
                else: raise ValueError(f"init list parse {rel}:{lineno(s, j)}")
                mm = re.match(r"\s*(,|\{)", s[j:])
                if mm.group(1) == ",": j += mm.end(); continue
                j += mm.end() - 1; break
        mm = re.match(r"\s*([{;=])", s[j:])
        if not mm or mm.group(1) != "{":
            continue
        k = j + mm.end() - 1
        e = match_close(s, k, "{", "}")
        key = (sub + "::" + nm) if sub else resolve(owner, nm)   # Engine::Cycle::f -> nested type method (as v2)
        key = add_body(key, rel, lineno(s, m.start()), s[m.start():e + 1], f"{owner}::{(sub + '::') if sub else ''}{nm}", [(rel, m.start(), e + 1)])
        files_with_defs.setdefault(rel, []).append(key)
    # A2: namespace-level struct / class definitions with a body in an engine file (helper types with inline methods)
    if is_eng:
        for m in re.finditer(r"(?<![\w:])(?:struct|class)\s+(\w+)\b(?!\s*::)(?:\s+final)?\s*(?::(?!:)[^;{]*)?\{", s):
            if cd[m.start()] != 0 or m.group(1) in CLASSES: continue
            e = match_close(s, m.end() - 1, "{", "}")
            key = add_body("type:" + m.group(1), rel, lineno(s, m.start()), s[m.start():e + 1], f"struct {m.group(1)} (namespace)", [(rel, m.start(), e + 1)])
            files_with_defs.setdefault(rel, []).append(key)
            types.append((m.group(1), lineno(s, m.start()), "-"))
    # free functions: in engine files every namespace-level definition; elsewhere only the names the cut lists
    for m in re.finditer(r"(?<![\w:.>~])([A-Za-z_]\w*)\s*\(", s):
        nm = m.group(1)
        if cd[m.start()] != 0 or nm in KW: continue
        if not is_eng and nm not in FREE: continue
        before = s[max(0, m.start() - 200):m.start()]
        if not re.search(r"[\w>&*]\s+[&*]*\s*$|^\s*$", before): continue
        if re.search(r"(=|return|\(|,|\.|->|!|&&|\|\||\?|#\s*\w+)\s*[\w:<>]*\s*$", before): continue
        try:
            j = match_close(s, m.end() - 1, "(", ")") + 1
        except ValueError:
            continue
        mm = re.match(r"(?:\s*(?:const|noexcept|&&|&)(?!\w)|\s*WH_\w+\([^)]*\))*\s*(?:->\s*[\w:<>,\s*&]+?)?\s*\{", s[j:])
        if not mm: continue
        k = j + mm.end() - 1
        e = match_close(s, k, "{", "}")
        key = add_body(nm, rel, lineno(s, m.start()), s[m.start():e + 1], f"free {nm}", [(rel, m.start(), e + 1)])
        files_with_defs.setdefault(rel, []).append(key)

# A2: in-class member initializers are part of the class's constructor entry
for cname, lst in sorted(inits.items()):
    add_body(resolve(cname, cname), lst[0][0], lst[0][1], "{ " + " ;\n".join(t for _, _, t, _ in lst) + " }",
             f"{cname}::{cname} (in-class initializers)", [(r, a0, a1) for r, _, _, (a0, a1) in lst])

# A3: no component holds Engine (DESIGN 2.1 rule 3) - not even as a listed holder
for h, c in sorted(HOLDERS.items()):
    if c == "Engine":
        findings.append(f"cut holder of Engine: {h}  [no component may hold Engine& / Engine*]")
# A3: data members of tracked classes whose type names Engine or a tracked class must be cut holders of that class
for nm, ln, acc, decl, cname, rel in data:
    dtype = decl.split(nm)[0] if nm in decl else decl
    named = [c for c in CLASSES if re.search(r"(?<![\w:])" + re.escape(c) + r"\b", dtype)]
    if named and not (nm in HOLDERS and HOLDERS[nm] in named):
        findings.append(f"member of tracked class {cname} holds {'/'.join(named)} but is not a cut holder of it "
                        f"({rel}:{ln}): {nm}  [no component may hold Engine& / another component under an unlisted name]")

# public forwarders: once a component class defines f, Engine's own f is the forwarder entry "Engine.f"
comp_keys = {k.split("#")[0] for k, v in bodies.items() if v[3].split()[-1].split("::")[0] in CLASSES
             and not v[3].split()[-1].startswith("Engine::") and not v[3].startswith("free")}
for k in list(bodies):
    v = bodies[k]
    if v[3].split()[-1].startswith("Engine::") and k.split("#")[0] in comp_keys and k.split("#")[0] not in ("Engine", "~Engine"):
        nk = "Engine." + k.split("#")[0]
        while nk in bodies: nk += "#"
        bodies[nk] = bodies.pop(k)
        branges[nk] = branges.pop(k)

# macros anywhere that name a tracked data member or class
dnames = sorted({d[0] for d in data})
dset = set(dnames)
for rel, (raw, s) in texts.items():
    if rel in engine_files: continue
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)([^\n]*)", raw, re.M):
        toks = set(re.findall(r"\w+", m.group(2)))
        hit = sorted(toks & (dset | set(CLASSES)))
        if hit:
            findings.append(f"#define naming tracked names {rel}:{lineno(raw, m.start())}: {m.group(1)} -> {','.join(hit)}")

fnames = sorted({k.split("#")[0] for k in bodies if not k.startswith("type:")} | {f[0] for f in funcs if "::" not in f[0]}
                | set(ALIASES.values()))
fnames = [n for n in fnames if "::" not in n]
WOPS = r"(?:(?:\s*(?:\.\w+|->\w+|\[[^\]]*\]))*\s*(?:=(?!=)|\+=|-=|\*=|/=|\|=|&=|\^=|<<=|>>=|\+\+|--)|(?:\s*(?:\.\w+|\[[^\]]*\]))*\s*\.\s*(?:store|fetch_add|fetch_sub|push_back|emplace_back|emplace|clear|reset|resize|assign|erase|insert|pop_front|pop_back|push_front|swap|exchange)\s*\()"
OKRECV = {"this", "e", "Engine", "self"} | set(CLASSES) | set(HOLDERS)
# method name of a tracked class -> entry (for holder.f( calls): C::f resolves through aliases / f
meth = {}
for nm, ln, acc, s, cname in funcs:
    meth.setdefault(nm.split("::")[-1], set()).add(resolve(cname, nm.split("::")[-1]))
for q, e in ALIASES.items():
    meth.setdefault(q.split("::")[-1], set()).add(e)

# arch025-emut2 (S4 requirement, progress/arch025-engine.HANDOFF.md S4 "cut 改動"): a non-const call on a cut holder
# member (`dc_.dflReset(..)`, cut "holders" name -> class) is a CALL EDGE to the holder class's method (the call graph
# above records it: OKRECV / HOLDERS), not a data write of the holder; the writes inside the callee are judged in the
# callee's own body. Only when (1) every declaration of the holder has one shape: the class by value / reference (call
# with `.`) or a raw / unique_ptr pointer to it (call with `->`), (2) the callee is a method the class declares (not an
# operator, not a destructor), (3) every census body of that method is free of `this` other than `this->` (a callee
# that assigns / swaps / hands out *this writes the holder as a whole, which no member use inside it shows). Everything
# else stays a write: assigning / resetting the holder, binding it to a non-const reference, passing it to a function,
# writing h.x, calling through an alias.
_cls_meth = {}
for nm, ln, acc, s, cname in funcs:
    if "::" not in nm and not nm.startswith(("operator", "~")): _cls_meth.setdefault(cname, set()).add(nm)
_hold_op = {}
for nm in HOLDERS:
    c = re.escape(HOLDERS[nm])
    kinds = set()
    for dn, ln, acc, s, cname, rel in data:
        if dn != nm: continue
        pre = s.split("=")[0].split("{")[0]
        pre = pre[:pre.rfind(nm)] if nm in pre else pre
        if re.search(r"(?<![\w:])(?:\w+::)*" + c + r"\s*&?\s*$", pre): kinds.add(".")
        elif re.search(r"(?<![\w:])(?:\w+::)*" + c + r"\s*\*\s*(?:const\s*)?$|unique_ptr\s*<\s*(?:\w+::)*" + c + r"\s*>\s*$", pre): kinds.add("->")
        else: kinds.add("?")
    if len(kinds) == 1 and "?" not in kinds: _hold_op[nm] = kinds.pop()
_THIS_RX = re.compile(r"(?<![\w])this\b(?!\s*->)")
_self_free = {}
for _c, _ms in _cls_meth.items():
    for _f in _ms:
        _e = resolve(_c, _f)
        _bt = [t for k, (_fl, _l, t, _d) in bodies.items() if k.split("#")[0] == _e]
        _self_free[(_c, _f)] = bool(_bt) and not any(_THIS_RX.search(t[t.find("{"):] if "{" in t else t) for t in _bt)
_HCALL = re.compile(r"\s*(\.|->)\s*(\w+)\s*\(")


def holder_call(text, i, mem):
    """the use of member `mem` at text[i:] is a holder call edge: its reason, else None"""
    op = _hold_op.get(mem)
    if not op or text[i:i + len(mem)] != mem: return None
    m = _HCALL.match(text, i + len(mem))
    if not m or m.group(1) != op: return None
    c, f = HOLDERS[mem], m.group(2)
    if f not in _cls_meth.get(c, ()) or not _self_free.get((c, f)): return None
    return f"holder call {mem}{op}{f}() (call edge to {c}::{f}, judged in its own body)"


# A1: the measured const list. A method name counts as read-only when src/ declares it ONLY as a const member
# function (some `name(...) const` declaration and no non-const declaration of the same name), or it is a std
# read-only name. Everything else called on a member's access chain is a write.
STD_CONST = {"size", "empty", "load", "find", "count", "contains", "at", "front", "back", "data", "begin", "end",
             "cbegin", "cend", "rbegin", "rend", "get", "c_str", "str", "substr", "length", "capacity", "value",
             "has_value", "value_or", "compare", "starts_with", "ends_with", "rfind", "find_first_of", "test",
             "time_since_epoch", "count_ms", "lower_bound", "upper_bound", "equal_range", "max_size", "native_handle"}
_decl_const, _decl_mut = set(), set()
_params = {}   # function / method name -> [parameter list per declaration] (A1 (d): by-value / const& = read)


def _split_top(x):
    out, d, cur = [], 0, ""
    for ch in x:
        if ch in "([{<": d += 1
        if ch in ")]}>": d -= 1
        if ch == "," and d == 0: out.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out


def _param_reads(cal, idx):
    """True when every src/ declaration of `cal` takes argument idx by value or by const reference / pointer."""
    decls = _params.get(cal)
    if not decls: return False
    seen = False
    for ps in decls:
        if idx >= len(ps) and not any("..." in x for x in ps): continue   # cannot bind: not this overload
        if idx >= len(ps): return False
        seen = True
        t = ps[idx].split("=")[0]
        if re.search(r"[&*]", t) and not re.match(r"\s*const\b", t): return False
        if "&&" in t: return False
    return seen
_DECL = re.compile(r"(?<![\w.>:])(?:[\w:<>,]+[\s*&]+)(~?[A-Za-z_]\w*)\s*\(((?:[^()]|\([^()]*\))*)\)\s*(const)?(?![\w])")
_NOTTYPE = KW | {"return", "else", "case", "goto", "co_return", "co_await", "throw", "new", "delete", "typename"}
for _rel, (_raw, _s) in decl_texts.items():
    for _m in _DECL.finditer(_s):
        _t = _s[max(0, _m.start() - 1):_m.start(1)].strip().split()
        if not _t or _t[0].strip("*&") in _NOTTYPE: continue
        _after = _s[_m.end():_m.end() + 40]
        if not re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?(?:WH_\w+\([^)]*\)\s*)*[;{=]", _after):
            continue
        (_decl_const if _m.group(3) else _decl_mut).add(_m.group(1))
        _params.setdefault(_m.group(1), []).append(_split_top(_m.group(2)))
CONST_OK = STD_CONST | (_decl_const - _decl_mut)
# accessor pairs (declared both const and non-const, e.g. PrefixStore::spes()): the call itself is not a write; what
# follows on the chain decides (`ps_.spes().size()` reads, `ps_.spes().clear()` / `= x` / `f(ps_.spes())` write).
PAIRS = _decl_const & _decl_mut
# A1 (d): callees that only read their arguments (by value or const reference) - measured on c1ca7ce by hand, each
# checked at its declaration; anything else that receives a member as an argument is a write.
# S0b (S0 review act-on 1): span and move_if_noexcept left the list (a span is a writable view, a move writes the source);
# static_cast stays for value casts only - a cast to a non-const reference / pointer is unwrapped by _outer (write if written).
READ_CALLEES = {"min", "max", "clamp", "static_cast", "size_t", "uint32_t", "uint64_t", "int", "double", "float",
                "bool", "to_string", "duration_cast", "duration", "logf", "log", "snprintf",
                "printf", "fprintf", "strlen", "memcmp", "abs", "llabs", "ceil", "floor", "round", "sqrt", "pow", "exp",
                "isfinite", "isnan", "string", "string_view", "hash", "equal", "distance",
                "logI", "logW", "logE", "logD",   # log*: printf-style varargs (by value)
                "WH_REQUIRES", "WH_REQUIRES_SHARED", "WH_GUARDED_BY", "WH_PT_GUARDED_BY", "WH_EXCLUDES",
                "WH_RETURN_CAPABILITY", "WH_ACQUIRED_AFTER", "WH_ACQUIRED_BEFORE"}   # TSA annotations, not operations
_ASSIGN = re.compile(r"\s*(?:=(?!=)|\+=|-=|\*=|/=|%=|\|=|&=|\^=|<<=|>>=|\+\+|--)")


def _chain(txt, i):
    """Walk an access chain after a member hit: .x  ->x  [..]  (..). Returns ([(name, called)], end)."""
    out, n = [], len(txt)
    while i < n:
        m = re.match(r"\s*(?:\.|->)\s*(~?\w+)", txt[i:])
        if m:
            i += m.end()
            mm = re.match(r"\s*(?:<[^<>;(){}]*>)?\s*\(", txt[i:])
            if mm:
                try: i = match_close(txt, i + mm.end() - 1, "(", ")") + 1
                except ValueError: return out, n
                out.append((m.group(1), True))
            else:
                out.append((m.group(1), False))
            continue
        m = re.match(r"\s*\[", txt[i:])
        if m:
            try: i = match_close(txt, i + m.end() - 1, "[", "]") + 1
            except ValueError: return out, n
            continue
        break
    return out, i


def _callee(txt, k):
    """txt[k] is the '(' that opens an argument list: the callee name, or None (not a call: if/while/(..) cast)."""
    m = re.search(r"([A-Za-z_]\w*)\s*(?:<[^<>;(){}]*>)?\s*$", txt[:k])
    if not m or m.group(1) in KW | {"return", "else", "case"}: return None
    return m.group(1)


CASTS = {"static_cast", "const_cast", "reinterpret_cast"}
# S0c: data members declared as std::function<...>: calling one is a read (std::function::operator() is a const member)
FUNC_MEMBERS = {nm for nm, _ln, _acc, decl, _c, _r in data if re.match(r"\s*(?:mutable\s+)?std::function\s*<", decl)}
# S0b: std accessors that hand out a writable handle (iterator / pointer / reference) in a non-const context
HANDLE = {"begin", "end", "rbegin", "rend", "data", "front", "back", "at"}


def _outer(txt, h0, end):
    """S0b (S0 review W4 / W5 / W10): one step outwards over `( ... )` around the expression txt[h0:end], with an optional
    unary `*` inside: `(x_)`, `(*x_.begin())`, `static_cast<T&>(x_)` (T& / T* not const). Returns (h0, end) unchanged when
    the parentheses belong to a call, a keyword (if / while / return ...), a value cast or there are none."""
    q = h0
    while q > 0 and txt[q - 1].isspace(): q -= 1
    if q > 0 and txt[q - 1] == "*":
        r = q - 1
        while r > 0 and txt[r - 1].isspace(): r -= 1
        if r == 0 or txt[r - 1] in "(=,;{}!?:&|[": q = r      # unary * (dereference)
    while q > 0 and txt[q - 1].isspace(): q -= 1
    m = re.match(r"\s*\)", txt[end:])
    if q == 0 or txt[q - 1] != "(" or not m:
        return h0, end
    k = q - 1
    mc = re.search(r"(static_cast|const_cast|reinterpret_cast)\s*<([^<>;(){}]*(?:<[^<>;(){}]*>[^<>;(){}]*)*)>\s*$", txt[:k])
    if mc:
        if not re.search(r"[&*]", mc.group(2)) or re.search(r"\bconst\b", mc.group(2)): return h0, end
        k = mc.start()
        mm = re.search(r"std\s*::\s*$", txt[:k])
        if mm: k = mm.start()
    elif re.search(r"\)\s*$", txt[:k]):
        # `if (...) (x_) = ..`: the ) closes a statement header -> syntax, unwrap; else `f(..)(x_)` is a call
        j, d = len(txt[:k].rstrip()) - 1, 0
        while j >= 0:
            d += txt[j] == ")"; d -= txt[j] == "("
            if d == 0: break
            j -= 1
        if j < 0 or not re.search(r"\b(?:if|while|for|switch)\s*$", txt[:j]): return h0, end
    elif re.search(r"[\w>\]]\s*$", txt[:k]):
        return h0, end                                          # a call, a keyword, a subscript or a function cast
    return k, end + m.end()


def _unary_star(txt, h0):
    """S0c (N1, N5): a unary `*` (dereference) right before txt[h0:] without parentheses: its position, else None"""
    q = h0
    while q > 0 and txt[q - 1].isspace(): q -= 1
    if q == 0 or txt[q - 1] != "*": return None
    r = q - 1
    while r > 0 and txt[r - 1].isspace(): r -= 1
    if r == 0 or txt[r - 1] in "(=,;{}!?:&|[<>+-" or re.search(r"\breturn\s*$", txt[:r]): return r
    return None


def _ccast(txt, h0):
    """S0c (N3): a C-style cast to a non-const reference / pointer `(T&)x_` / `(T*)x_` right before txt[h0:]"""
    m = re.search(r"\(\s*[\w:<>,\s]*?[\w>]\s*([&*]+)\s*\)\s*$", txt[:h0])
    if not m or re.search(r"\bconst\b", m.group(0)): return None
    if re.search(r"[\w\]]\s*$", txt[:m.start()]) and not re.search(r"\b(?:return|case)\s*$", txt[:m.start()]): return None
    return m.start()


_CMP = r"(?:==|!=|<=|>=|<(?![<=])|>(?![>=]))"
_BINR = r"(?:==|!=|<=|>=|&&|\|\||<<|>>|<|>|\+|-(?!>)|\*|/|%|&|\||\^|\?|:(?!:))"
_STMT0 = r"(?:^|[;{}]|\)|\belse|\bdo)\s*$"


def write_kind(txt, h0, h1, handle0=False):
    """A1: why the member hit txt[h0:h1] is a write, or None for a read.
    S0c (S0b review act-on 1, ExprMutationAnalyzer direction): the rule is inverted - a hit is a READ only in one of the
    listed read forms (after unwrapping parentheses / derefs / casts and walking the access chain): a const method on
    the chain, an operand of a comparison / arithmetic / logic operator, `return`, the right side of a non-reference
    copy, a condition, a subscript, a read-only call argument, a member initializer; ANYTHING ELSE is a write. A writable
    handle (begin / data / front / at ... or a returned iterator / pointer / view) is a write unless it is only compared."""
    hit = txt[h0:h1]
    ch, end = _chain(txt, h1)
    deref = False
    hk = lambda c, dflt: (c[-1][1] and c[-1][0] in HANDLE) if c else dflt
    handle = hk(ch, handle0)
    # S0b / S0c: unwrap parentheses / dereference / casts around the hit, continuing the access chain after them
    for _ in range(12):
        n0, n1 = _outer(txt, h0, end)
        if (n0, n1) != (h0, end):
            inner = txt[n0:h0]
            if re.search(r"\*\s*$", re.sub(r"\(", "", inner)) and "*" in inner: deref = True; handle = False
            h0 = n0
            ch2, end = _chain(txt, n1)
            ch += ch2
            handle = hk(ch2, handle)
            continue
        u = _unary_star(txt, h0)
        if u is not None:
            h0 = u; deref = True; handle = False
            continue
        c = _ccast(txt, h0)
        if c is not None:
            h0 = c
            continue
        break
    pre = txt[:h0]
    for nm, called in ch:
        if called and nm not in CONST_OK and nm not in PAIRS: return f"method {nm}()"
    if _ASSIGN.match(txt[end:]): return "assignment"
    if re.search(r"(\+\+|--)\s*(?:\(\s*)*$", pre): return "prefix ++/--"
    seg = re.split(r"[;{}]", pre)[-1]
    # S0b (W1 / W2 / W3): `auto&&`, range-for `auto&& j :`, structured bindings `auto& [a, b] =`
    if re.search(r"(?:&&|&)\s*(?:\[[^\]]*\]|[A-Za-z_]\w*)\s*(?:=|:)\s*$|\*\s*[A-Za-z_]\w*\s*=\s*&\s*$", seg) and not re.search(r"\bconst\b", seg):
        return "bound to a non-const reference / pointer"
    # S0b (W9): a pointer to a data member (&Engine::x_ / &C::x_) can write through ->* / .*
    if re.search(r"&\s*(?:" + QUAL + r")\s*::\s*$", pre): return "data member pointer taken"
    if re.search(r"(?<![&\w)\]])&\s*$", pre) and not re.search(r"\bconst\b", seg): return "address taken"
    # S0b (W6 / W7): a writable handle (iterator / pointer) of a std accessor kept in a variable (`auto it = x_.begin()`,
    # `auto* p = x_.data()`, `it = x_.end()`); a const declaration or a c-accessor (cbegin ...) stays a read
    if ch and ch[-1][1] and ch[-1][0] in HANDLE and not re.search(r"\bconst\b", seg) \
            and re.search(r"(?:[\w\]>)]\s*(?<![=!<>])=(?!=)|[A-Za-z_]\w*\s*\{)\s*$", pre):
        return f"handle {ch[-1][0]}() kept"
    post = txt[end:]
    L = pre.rstrip()
    # S0c (N2): brace-initialized reference `auto& q{x_}`; (N4) a reference bound through ?: `auto& q = c ? x_ : y_`
    mb = re.search(r"(?:&&|&)\s*[A-Za-z_]\w*\s*\{\s*$", pre)   # (the `{` itself ends `seg`: look at the text before it)
    if mb and not re.search(r"\bconst\b", re.split(r"[;{}]", pre[:mb.start()])[-1]):
        return "bound to a non-const reference (brace)"
    if re.search(r"(?<!:)[?:]\s*$", L) and re.search(r"(?:&&|&|\*)\s*[A-Za-z_]\w*\s*(?:=|\{)", seg) \
            and not re.search(r"\bconst\b", seg):
        return "bound to a non-const reference / pointer through ?:"
    # S0c (N5-N8, N15): a writable handle is a write unless it is only compared (`it != x_.end()`)
    if handle:
        if re.match(r"\s*" + _CMP, post) or re.search(_CMP + r"\s*$", L): return None
        return f"writable handle {ch[-1][0] if ch else 'result'}() used ({' '.join(post[:12].split()) or 'end'})"
    # argument of a call: the hit (with its chain) is a whole argument
    if re.search(r"[(,]\s*(?:std::move\s*\(\s*)?$", pre) and re.match(r"\s*[,)]", txt[end:]):
        d, k = 0, len(pre) - 1
        while k >= 0:
            c = pre[k]
            if c == ")": d += 1
            elif c == "(":
                if d == 0: break
                d -= 1
            elif c in ";{}": k = -1; break
            k -= 1
        if k >= 0:
            cal = _callee(pre, k)
            if cal is not None and cal not in READ_CALLEES:
                inner = pre[k + 1:]
                idx, dd = 0, 0
                for ch in inner:
                    if ch in "([{": dd += 1
                    elif ch in ")]}": dd -= 1
                    elif ch == "," and dd == 0: idx += 1
                # S0c (Z1): a std:: callee is never matched against src/ declarations of the same bare name
                # (std::ref(x_) is not PagePool::ref(int32_t)); only READ_CALLEES make a std call a read
                is_std = re.search(r"\bstd\s*::\s*(?:\w+\s*::\s*)*" + re.escape(cal) + r"\s*(?:<[^<>;(){}]*>)?\s*$", pre[:k]) is not None
                if is_std or not _param_reads(cal, idx):
                    return f"argument {idx} of {cal}()"
    # S0c: the positive read forms; everything else is a write
    R = re.match(r"\s*(->\*|\.\*|==|!=|<=|>=|&&|\|\||<<|>>|\S)", post)
    R = R.group(1) if R else ""
    if R in ("->*", ".*"): return "member pointer access on it"
    stmt0 = re.search(_STMT0, L) is not None
    if R in ("<<", ">>") and stmt0: return f"stream operator {R} on it"
    if R == "(":
        if re.search(r"[:,]\s*$", L): return None   # member initializer `: x_(v)` (initialization, not a mutation)
        if not ch and hit in FUNC_MEMBERS: return None   # std::function::operator() is const (its declaration)
        return "called (operator())"
    if R == "{":
        if re.search(r"[:,]\s*$", L): return None   # member initializer `: x_{v}`
        return "brace after it"
    r_ok = R in (")", ",", ";", "]", "}", "") or re.match(_BINR, R) is not None
    l_ok = (stmt0 or re.search(r"(?:[(,\[{!~]|\breturn|\bcase|\bco_return|" + _BINR + r"|(?<![=!<>])=)\s*$", L) is not None)
    if r_ok and l_ok:
        return None
    return f"not a known read form ({' '.join(L[-12:].split()) or 'start'} .. {R or 'end'})"


# untracked classes' method names (A3: an unknown receiver calling one of these is ambiguous -> reported only)
_untracked_methods = set()
for _rel, (_raw, _s) in decl_texts.items():
    for _m in re.finditer(r"(?:class|struct)\s+(\w+)\b[^;{]*\{", _s):
        if _m.group(1) in CLASSES: continue
        try: _e = match_close(_s, _m.end() - 1, "{", "}")
        except ValueError: continue
        for _d in _DECL.finditer(_s, _m.end(), _e):
            _untracked_methods.add(_d.group(1))
unresolved = []
wlog = []   # entry, member, why it is a write, context (writes.tsv)

# S0b (S0 review W11): tracked functions that return a non-const reference / pointer hand out write access to the members
# their return statements name: the function itself counts as writing them, and a caller that writes through the
# result (`holdRef() = true`, `auto& h = holdRef(); h = ..`, `holdRef().clear()`) writes them too.
REFACC = {}
REFHANDLE = set()   # S0c: accessors whose result is itself a handle (pointer / iterator / span / auto)
# S0c (S0b review act-on 1 (c); N10, N12, N13, N15): the return type text includes a trailing `-> T`; a return type with
# & / * / span / iterator / view / auto / decltype(auto) and no const (and a non-const method) hands out write access to
# EVERY member its body uses (a local reference in between does not hide it)
_RETW = r"[&*]|\bspan\b|(?<!const_)iterator\b|(?<!string_)view\b|\bauto\b|decltype"
for _nm, _ln, _acc, _decl, _cname in funcs:
    _dd = re.sub(r"\bdecltype\s*\(\s*auto\s*\)", "decltype_auto", detpl(_decl))
    _ret = _dd.split("(")[0]
    _ret = _ret[:_ret.rfind(_nm)] if _nm in _ret else _ret
    _ret = re.sub(r"\b(static|inline|virtual|constexpr|explicit|\[\[[^\]]*\]\])\b", "", _ret)
    _k = resolve(_cname, _nm)
    _bl = [(_bk, _t) for _bk, (_f, _l, _t, _d) in bodies.items() if _bk.split("#")[0] == _k]
    _tr = ""
    for _bk, _t in _bl[:1]:
        _mt = re.search(r"\)\s*((?:const|noexcept|override|final|&|\s)*)(->\s*[^{;]*)?\{", detpl(_t))
        if _mt: _tr = (_mt.group(2) or "")[2:]; _quals = _mt.group(1) or ""
    _mq = re.search(r"\)\s*((?:const|noexcept|override|final|\s)*)(?:->|;|=|\{|$)", _dd)
    _mconst = bool(_mq and re.search(r"\bconst\b", _mq.group(1)))
    _rt = (_ret + " " + _tr).replace("&&", " ")
    if re.search(_RETW, _rt) and not re.search(r"\bconst\b", _rt) and not _mconst:
        for _bk, _t in _bl:
            _body = _t[_t.find("{"):] if "{" in _t else _t
            REFACC.setdefault(_k, set()).update(n for n in dnames if re.search(r"(?<![\w])" + re.escape(n) + r"\b", _body))
        if re.search(r"\*|\bspan\b|iterator\b|\bauto\b|decltype", _rt) and not re.search(r"&\s*$", _ret.strip()):
            REFHANDLE.add(_k)


def uses(key, txt):
    ud, uw = [], []
    for n in dnames:
        hits = list(re.finditer(r"(?<![\w])" + re.escape(n) + r"\b", txt))
        if hits:
            ud.append(n)
            for h in hits:
                if holder_call(txt, h.start(), n): continue   # arch025-emut2: a call edge, not a write of the holder
                why = write_kind(txt, h.start(), h.end())
                if why:   # S0f (S0e review consider 2): every text write reason of the member is kept, not only the first
                    if n not in uw: uw.append(n)
                    wlog.append(f"{key}\t{n}\t{why}\t{' '.join(txt[max(0, h.start() - 40):h.end() + 40].split())}")
    me = key.split("#")[0]
    # S0b: a non-const reference / pointer accessor writes what it returns
    for n in sorted(REFACC.get(me, ())):
        if n not in uw:
            uw.append(n); wlog.append(f"{key}\t{n}\treturned by a non-const reference / pointer\t-")
    # A3: local aliases of *this / of a tracked object are known receivers
    # S0b (S0 review U2): `auto&&` too, and an alias of an alias inherits its class (`auto& me2 = me`)
    alias = {}
    for m in re.finditer(r"(?<![\w:])(" + QUAL + r"|auto)\s*(?:const\s*)?(?:&&|[&*])\s*(\w+)\s*=\s*(\*?\s*this|[\w.>*-]+)", txt):
        tgt = m.group(3).replace(" ", "")
        if m.group(1) == "auto":
            if tgt in ("*this", "this"): alias[m.group(2)] = "Engine"
            elif tgt.lstrip("*") in HOLDERS: alias[m.group(2)] = HOLDERS[tgt.lstrip("*")]
            elif tgt.lstrip("*") in alias: alias[m.group(2)] = alias[tgt.lstrip("*")]
        else:
            alias[m.group(2)] = m.group(1)
    called = set()
    for h in re.finditer(r"(?<![\w])(~?\w+)\s*\(", txt):
        n = h.group(1)
        pre = txt[:h.start()]
        if n in REFACC:   # S0b (W11): a write through the returned reference / pointer is a write of what it returns
            try:
                k1 = match_close(txt, h.end() - 1, "(", ")") + 1
                mrc = re.search(r"(?:(?:\(\s*\*\s*)?\w+\s*\)?\s*(?:\.|->)\s*)+$", pre)
                why = write_kind(txt, mrc.start() if mrc else h.start(), k1, n in REFHANDLE)
            except ValueError:
                why = "unparsed call"
            if why:
                for d0 in sorted(REFACC[n]):
                    if d0 not in uw:
                        uw.append(d0)
                        wlog.append(f"{key}\t{d0}\tthrough {n}(): {why}\t{' '.join(txt[max(0, h.start() - 40):h.end() + 40].split())}")
        mr = re.search(r"(\w+)\s*(?:\.|->|::)\s*$", pre)
        recv = mr.group(1) if mr else None
        mt = re.search(r"\(\s*\*\s*(\w+)\s*\)\s*(?:\.|->)\s*$", pre)   # S0b (U1): (*this).f( / (*me).f(
        if mt: recv = mt.group(1)
        if recv in alias:
            rc = alias[recv]
            if rc == "Engine":
                if n in fnames: called.add(n)
            else:
                for t in meth.get(n, set()) | ({n} if n in fnames else set()): called.add(t)
            continue
        if recv is None:
            if re.search(r"(?:\.|->|::)\s*$", pre):
                if n in fnames: unresolved.append(f"{me}: <expr>.{n}(")
                continue      # ).f( / ]->f( : unknown receiver
            if n in fnames: called.add(n)
        elif recv in OKRECV:
            if recv in HOLDERS or (recv in CLASSES and recv != "Engine"):
                for t in meth.get(n, set()) | ({n} if n in fnames else set()):
                    called.add(t)
            elif n in fnames:
                called.add(n)
        elif n in fnames and not re.search(r"::\s*$", pre):
            # A3: unknown receiver calling a tracked entry name: a call unless an untracked class also has that method
            if n in _untracked_methods: unresolved.append(f"{me}: {recv}.{n}(  (also a method of an untracked class)")
            else: called.add(n)
    for h in re.finditer(r"(?<![\w:])(" + QUAL + r")::(~?\w+)\b(?!\s*\()", txt):
        t = resolve(h.group(1), h.group(2))
        if t in fnames: called.add(t)
    called.discard(me)
    ut = [t for t, _, _ in types if re.search(r"(?<![\w:])" + t + r"\b", txt) and key != "type:" + t]
    return ud, uw, sorted(called), sorted(set(ut))


rows = {}
for key, (f, ln, txt, dfn) in bodies.items():
    rows[key] = (f, ln) + uses(key, txt)

# ---------------- S0d: the member writes come from the clang AST (tools/gates/engine_split_ast.py) ----------------
# The text rules above stay as a cross-check (writes_text.tsv, ast_xcheck.tsv: every row where they disagree is listed
# and counted on stdout); the writes the graph judges (funcs.tsv data_written, members.tsv writers) are the AST's.
# A body the AST cannot see (no parsed TU reads its file, or its TU has clang errors) keeps text OR AST writes and is
# a finding. Clang missing = crash (exit 1), never a silent fallback.
HERE = os.path.dirname(os.path.abspath(__file__))
text_wlog = list(wlog)
wlog = []
_tus = sorted({"src/" + f for f in set(files_with_defs) | engine_files if f.endswith(".cpp")})
_cache = os.environ.get("WHIRL_ENGINE_AST_CACHE") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "build", "engine_split_ast_cache")
_callable = sorted(k for k in cut.get("callable_reads", {}) if not k.startswith("_"))
_astout = os.path.join(OUT, "ast.json")
import subprocess
_cmd = [sys.executable, "-I", os.path.join(HERE, "engine_split_ast.py"), "--root", ROOT, "--out", _astout, "--cache", _cache,
        "--classes", ",".join(sorted(CLASSES)), "--callable-ok", ",".join(_callable), "--jobs", os.environ.get("WHIRL_ENGINE_AST_JOBS", "6")]
for _t in _tus: _cmd += ["--tu", _t]
# arch025-emut2 (PLAN N4-emut2): clang's own ExprMutationAnalyzer (tools/gates/emut, emut.exe on the official LLVM release)
# runs next to the Python analyzer on the same TUs; writes = clang w UNION Python w, every disagreement is classified or a
# finding (tools/gates/engine_split_emut.py). emut.exe / LLVM missing or stale = crash (exit 1): never one analyzer alone.
sys.path.insert(0, HERE)
import engine_split_emut as ESE
_emut = os.environ.get("WHIRL_EMUT") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "build", "emut", "emut.exe")
_ecache = os.environ.get("WHIRL_ENGINE_EMUT_CACHE") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "build", "engine_split_emut_cache")
_emout = os.path.join(OUT, "emut.json")
_ecmd = [sys.executable, "-I", os.path.join(HERE, "emut", "emut_run.py"), "--root", ROOT, "--out", _emout, "--cache", _ecache,
         "--emut", _emut, "--classes", ",".join(sorted(CLASSES)), "--jobs", os.environ.get("WHIRL_ENGINE_AST_JOBS", "6")]
for _t in _tus: _ecmd += ["--tu", _t]
if os.path.exists(_emout): os.remove(_emout)
_ep = subprocess.Popen(_ecmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
_r = subprocess.run(_cmd, capture_output=True, text=True)
_eo = _ep.communicate()[0]
if _r.returncode != 0 or not os.path.exists(_astout):
    print(_r.stdout + _r.stderr)
    print("census3: AST census failed (engine_split_ast.py rc %d)" % _r.returncode)
    sys.exit(1)
if _ep.returncode != 0 or not os.path.exists(_emout):
    print(_eo)
    print("census3: emut census failed (tools/gates/emut/emut_run.py rc %d; build emut.exe with tools/gates/emut/build_emut.cmd "
          "<dir> and set WHIRL_EMUT, or into build/emut/; the official LLVM: WHIRL_LLVM) - the census never runs one analyzer alone"
          % _ep.returncode)
    sys.exit(1)
_ast = json.load(open(_astout, encoding="utf-8"))
_east = json.load(open(_emout, encoding="utf-8"))
for _t in _ast["tus"]:
    if _t["errors"] or _t["rc"]:
        findings.append(f"clang errors in {_t['tu']} ({len(_t['errors'])}; the AST census of its bodies falls back to text OR AST): "
                        + (_t["errors"][0][-160:] if _t["errors"] else f"rc {_t['rc']}"))
for _t in _east["tus"]:
    if _t["errors"] or _t["rc"]:
        findings.append(f"emut (clang ExprMutationAnalyzer) errors in {_t['tu']} ({len(_t['errors'])}; its bodies fall back to "
                        f"text OR Python OR emut): " + (_t["errors"][0][-160:] if _t["errors"] else f"rc {_t['rc']}"))
_PX, _CX = ESE.index(_ast), ESE.index(_east)
_pos, _calls, _fname, _leaks = _PX["pos"], _PX["calls"], _PX["fname"], _PX["leaks"]
_covered, _errfiles = _PX["covered"], _PX["errfiles"]
findings.extend(ESE.check_tus(_tus, _ast, _east))
if not _PX["errfiles"] and not _CX["errfiles"]:   # with clang errors the error findings above already fail the check
    findings.extend(ESE.check_positions(_PX, _CX))
_eok = {k: v for k, v in cut.get("emut_read_ok", {}).items() if not k.startswith("_")}
_eused = set()
_etext = lambda rel: texts.get(rel, ("", ""))[1]
_skip = lambda rel, idx, mem: holder_call(_etext(rel), idx, mem)
_erows = []
_inbody = set()
xrows = []
for key in list(rows):
    f, ln, ud, uw_text, uf, ut = rows[key]
    rng = branges.get(key, [])
    p_all, rd_of, inb = ESE.body_writes(_PX, rng, _skip)
    c_all, _crd, cinb = ESE.body_writes(_CX, rng, _skip)
    _inbody |= inb | cinb
    files_k = {r for r, _, _ in rng}
    blind = [r for r in files_k if r in _errfiles or r not in _covered]
    cblind = [r for r in files_k if r in _CX["errfiles"] or r not in _CX["covered"]]
    if (blind or cblind) and (ud or uw_text):
        if not any(r in _errfiles for r in blind) and blind:
            findings.append(f"no AST coverage (no parsed TU reads {', '.join(sorted(blind))}): {key}; text census writes used")
        if not any(r in _CX["errfiles"] for r in cblind) and cblind:
            findings.append(f"no emut coverage (no emut TU reads {', '.join(sorted(cblind))}): {key}; text OR Python OR emut writes used")
        why_of = {m: v[0] for m, v in c_all.items()}
        for m, v in p_all.items(): why_of.setdefault(m, v[0])
        for m in uw_text:
            why_of.setdefault(m, ("text census (no AST)", 0, f))
    else:
        why_of, er, ef, eu = ESE.judge(key, p_all, c_all, _eok, _etext)
        _erows += er; findings.extend(ef); _eused |= eu
    aw = sorted(why_of)
    for m in sorted(set(aw) | set(uw_text)):
        if (m in aw) != (m in uw_text):
            tw = " | ".join(sorted({l.split("\t")[2] for l in text_wlog if l.split("\t")[0] == key and l.split("\t")[1] == m})) or "-"
            # S0e: for a text-write / AST-read row, the AST read reasons (proof kind decided below)
            ar = "; ".join(sorted(set(rd_of.get(m, [])))) if m not in aw else why_of.get(m, ('-',))[0]
            xrows.append(f"{key}\t{m}\t{'w' if m in uw_text else 'r'}\t{'w' if m in aw else 'r'}\t{ar or 'NO AST HIT'}\t{tw}")
    for m in aw:
        w, idx, rel = why_of[m]
        rawt = texts.get(rel, ("", ""))[1]
        wlog.append(f"{key}\t{m}\t{w}\t{' '.join(rawt[max(0, idx - 40):idx + 60].split()) if rawt else '-'}")
    rows[key] = (f, ln, ud, aw, uf, ut)
for k2 in sorted(set(_eok) - _eused):
    findings.append(f"cut emut_read_ok entry no longer occurs (remove it): {k2}")
open(os.path.join(OUT, "emut_xcheck.tsv"), "w", encoding="utf-8").write(
    "entry\tmember\tpython\tclang\tfinal\tclass\tpython_why\tclang_why\n" + "".join("\t".join(r) + "\n" for r in _erows))
_ecnt = {}
for r in _erows: _ecnt[r[5].split(":")[0].split(" (")[0]] = _ecnt.get(r[5].split(":")[0].split(" (")[0], 0) + 1
print(f"emut cross-check: {len(_east['tus'])} TUs ({sum(1 for t in _east['tus'] if t['errors'] or t['rc'])} with clang errors), "
      f"{len(_CX['pos'])} member uses (Python {len(_PX['pos'])}); (entry, member) pairs where they disagree: {len(_erows)} "
      + (", ".join(f"{k} {v}" for k, v in sorted(_ecnt.items())) or "-") + " (emut_xcheck.tsv)")
# a write the AST sees outside every census entry (a body the text census did not find) is a finding
for _ix, _nm in ((_PX, "AST"), (_CX, "emut")):
    for (rel, idx, mem), e in sorted(_ix["pos"].items()):
        if (rel, idx, mem) not in _inbody and ESE.verdict(e) and not rel.startswith("include:"):
            findings.append(f"{_nm} write outside every census entry: {rel}:{texts.get(rel, ('', ''))[1].count(chr(10), 0, idx) + 1} {mem} ({ESE.verdict(e)})")

# S0e (S0d review act-on 2): the cross-check bites - a row the text rules call a write and the AST a read is a FINDING
# unless the cut pins it (cut "text_write_ok": "entry:member" -> {"text_why": the text rule's reason, "reason": why the
# text rule is wrong there}); a pinned row that no longer occurs is a finding too (the list stays exact)
# S0e (coordinator 08:5x): a row is NOT a finding when the AST PROVES the read from types / the call target: every AST
# verdict for that member in that body is a read whose reason comes from the AST itself (a const object / const cast /
# const& parameter, a value read, a compared / discarded handle, a std::format target by its basic_format_string
# parameter). Reads that rest on a measured NAME list (value algorithms, lock guards, value wrappers, copy types,
# callable_reads, non-const accessor value results) and bodies without any AST hit still need the pin.
_DIRECT = {"assignment", "prefix ++/--"}
_LISTED = ("handle read by std::", "locked by", "copied into", "std::function call (cut callable_reads)", "returns a value")
_two = {k: v for k, v in cut.get("text_write_ok", {}).items() if not k.startswith("_")}
_seen_two = set()
for x in xrows:
    q = x.split("\t")
    if q[2] == "w" and q[3] == "r":
        k2 = f"{q[0]}:{q[1]}"
        pw = _two.get(k2, {}).get("text_why")
        pw = set(pw) if isinstance(pw, list) else ({pw} if pw else set())
        if k2 in _two and pw == set(q[5].split(" | ")):   # S0f: the pin names the row's WHOLE set of text reasons
            _seen_two.add(k2); continue
        # S0f (S0e review act-on 2): a direct write in the text census (assignment / compound assignment / ++ / --, the
        # text rules that do not guess) is never excused by an AST read - the walker itself may be wrong there (E1)
        if any(r.split(" (")[0] in _DIRECT for r in q[5].split(" | ")):
            findings.append(f"text census DIRECT write the AST calls a read: {k2} (text: {q[5]}; AST: {q[4]}) - an AST rule gap, or pin it")
            continue
        if q[4] != "NO AST HIT" and not any(t in q[4] for t in _LISTED):
            continue   # the AST proves the read (ast_xcheck.tsv column ast_why lists the proof)
        mt = re.match(r"through (~?\w+)\(\)", q[5])
        if q[4] == "NO AST HIT" and mt:
            # the text rule's accessor write: the AST knows the callee; it proves the read when no definition of that
            # name hands the member out (no return of a non-const reference / handle to it, S0d leak analysis)
            mg = [g for g, nm in _fname.items() if nm == mt.group(1)]
            if mg and not any(q[1] in _leaks.get(g, ()) for g in mg):
                continue
        findings.append(f"text census write the AST calls a read, not pinned in the cut's text_write_ok: {k2} (text: {q[5]})")
for k2 in sorted(set(_two) - _seen_two):
    findings.append(f"cut text_write_ok entry no longer occurs (remove it): {k2}")
# S0e (S0d review consider 6 / 7): AST hits without a position, writes in files outside src/ include/, tracked class names
# that are class templates / nested classes (their fields would not be tracked)
for _t in _ast["tus"]:
    if _t.get("nopos"): findings.append(f"AST: {_t['nopos']} tracked member uses without a source position in {_t['tu']}")
    for u in _t.get("unmapped", []): findings.append(f"AST write outside src/ and include/ ({_t['tu']}): {u}")
for o_ in sorted({o for _t in _ast["tus"] for o in _t.get("odd", [])}): findings.append(f"AST: {o_}")
open(os.path.join(OUT, "writes_text.tsv"), "w", encoding="utf-8").write("entry\tmember\twhy\tcontext\n" + "\n".join(text_wlog) + "\n")
open(os.path.join(OUT, "ast_xcheck.tsv"), "w", encoding="utf-8").write(
    "entry\tmember\ttext\tast\tast_why\ttext_why\n" + "\n".join(xrows) + ("\n" if xrows else ""))
print(f"AST census: {len(_ast['tus'])} TUs ({sum(1 for t in _ast['tus'] if t['errors'] or t['rc'])} with clang errors), "
      f"{len(_pos)} member uses, {len(_leaks)} accessors handing out members; "
      f"AST vs text census: {len(xrows)} rows disagree (ast_xcheck.tsv: text w / AST r {sum(1 for x in xrows if x.split(chr(9))[2] == 'w')}, "
      f"text r / AST w {sum(1 for x in xrows if x.split(chr(9))[3] == 'w')})")
# S0b (S0 review act-on 2): an unresolved call is a finding unless the cut lists it exactly (cut "unresolved_ok", which is
# inside the cut pin); a listed entry that no longer occurs is a finding too (the list stays exact).
_uok = [x for x in cut.get("unresolved_ok", []) if not x.startswith("_")]
for u in sorted(set(unresolved)):
    if u not in _uok: findings.append(f"unresolved call not in the cut's unresolved_ok: {u}")
for u in _uok:
    if u not in set(unresolved): findings.append(f"cut unresolved_ok entry no longer occurs (remove it): {u}")

with open(os.path.join(OUT, "funcs.tsv"), "w", encoding="utf-8") as o:
    o.write("entry\tfile\tline\tlines\tdata_used\tdata_written\tfuncs_called\ttypes_used\tdef\n")
    for key in sorted(rows):
        f, ln, ud, uw, uf, ut = rows[key]
        nlines = bodies[key][2].count("\n") + 1
        o.write(f"{key}\t{f}\t{ln}\t{nlines}\t{','.join(ud)}\t{','.join(uw)}\t{','.join(uf)}\t{','.join(ut)}\t{bodies[key][3]}\n")
json.dump({k: v[2] for k, v in bodies.items()}, open(os.path.join(OUT, "bodies.json"), "w", encoding="utf-8"))
with open(os.path.join(OUT, "members.tsv"), "w", encoding="utf-8") as o:
    o.write("kind\tname\tline\taccess\tusers\twriters\tdecl\tclass\tfile\n")
    for nm, ln, acc, s, cname, rel in data:
        us = sorted({k.split("#")[0] for k, r in rows.items() if nm in r[2]})
        ws = sorted({k.split("#")[0] for k, r in rows.items() if nm in r[3]})
        o.write(f"data\t{nm}\t{ln}\t{acc}\t{','.join(us)}\t{','.join(ws)}\t{s}\t{cname}\t{rel}\n")
    for nm, ln in statics:
        o.write(f"static\t{nm}\t{ln}\t\t\t\t\t\t\n")
    for nm, ln, cname in types:
        o.write(f"type\t{nm}\t{ln}\t\t\t\t\t{cname}\t\n")
    for nm, ln, acc, s, cname in funcs:
        o.write(f"func\t{nm}\t{ln}\t{acc}\t\t\t{s}\t{cname}\t\n")
with open(os.path.join(OUT, "member_decls.tsv"), "w", encoding="utf-8") as o:   # S0b: for revtext (S0 review act-on 4)
    o.write("class\tmember\tfile\tline\tstart\tend\n")
    for cname, nm, rel, ln, a0, a1 in decls:
        o.write(f"{cname}\t{nm}\t{rel}\t{ln}\t{a0}\t{a1}\n")
# rt3: a declaration Engine keeps for a public forwarder while a component class declares the same function (S7:
# `void initPool();` in Engine and in SlotPages) is "decl:Engine.<f>", like the forwarder's body entry "Engine.<f>";
# the component's declaration keeps "decl:<f>" (it is the moved one)
_comp_decls = {e for c, e, *_ in cdecls if c != "Engine" and e.startswith("decl:") and ":" not in e[5:]}
cdecls = [(c, ("decl:Engine." + e[5:]) if c == "Engine" and e in _comp_decls and e[5:] not in ("Engine", "~Engine") else e, *r)
          for c, e, *r in cdecls]
with open(os.path.join(OUT, "class_decls.tsv"), "w", encoding="utf-8") as o:   # rt3: for revtext (rt2 review act-on 1)
    o.write("class\tentry\tfile\tline\tstart\tend\n")
    for cname, e, rel, ln, a0, a1 in cdecls:
        o.write(f"{cname}\t{e}\t{rel}\t{ln}\t{a0}\t{a1}\n")
with open(os.path.join(OUT, "files.txt"), "w", encoding="utf-8") as o:
    for f in sorted(files_with_defs):
        o.write(f"{f}\t{len(files_with_defs[f])}\n")
declared = {resolve(c, n) for n, _, _, _, c in funcs if "::" not in n}
have = {k.split("#")[0] for k in bodies}
missing = sorted(n for n in declared if n not in have and n not in ("Engine", "operator="))
for n in missing:
    findings.append(f"declared in a tracked class, no body found: {n}")
open(os.path.join(OUT, "findings.txt"), "w", encoding="utf-8").write("\n".join(findings) + ("\n" if findings else ""))
open(os.path.join(OUT, "writes.tsv"), "w", encoding="utf-8").write("entry\tmember\twhy\tcontext\n" + "\n".join(wlog) + "\n")
open(os.path.join(OUT, "unresolved.txt"), "w", encoding="utf-8").write("\n".join(sorted(set(unresolved))) + "\n")
print(f"unresolved calls (unresolved.txt; each must be in the cut's unresolved_ok): {len(set(unresolved))}")
print(f"census3: {len(set(d[0] for d in data))} data members ({len(data)} declarations), {len(statics)} static, "
      f"{len(types)} nested types, {len(funcs)} declared functions, {len(bodies)} bodies; classes found: "
      + ", ".join(f"{c}@{'/'.join(v)}" for c, v in sorted(class_files.items())))
print("files with tracked definitions:", ", ".join(f"{f}({len(v)})" for f, v in sorted(files_with_defs.items())))
print("findings:", len(findings), *(["\n  " + x for x in findings][:20]))
json.dump({"data": len(data), "bodies": len(bodies), "missing": missing, "findings": findings,
           "files": {f: len(v) for f, v in files_with_defs.items()}, "classes": class_files},
          open(os.path.join(OUT, "census3.json"), "w"), indent=1)
sys.exit(2 if findings else 0)
