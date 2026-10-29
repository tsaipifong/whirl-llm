# SPDX-License-Identifier: Apache-2.0
"""arch025-engine S0d: member-write census from the clang AST (replaces the text rules of engine_split_census.py).

Three review rounds (S0 / S0b / S0c, progress/arch025-engine.S0c.REVIEW.md act-on 1) each found 12-17 write shapes a text
rule cannot see, because only the declared types decide them (a typedef to a reference, reference_wrapper, span,
decltype((x)), aggregate reference members, unresolved callees). This tool asks clang:

  clang++ -fsyntax-only -Xclang -ast-dump=json -Xclang -ast-dump-filter=whirl::server <TU>   (product defines / -std)

and decides, for every MemberExpr (and DeclRefExpr) that names a data member of a tracked class, whether the evaluated
expression can modify it - the rules of clang's ExprMutationAnalyzer (clang/lib/Analysis/ExprMutationAnalyzer.cpp,
the base of clang-tidy misc-const-correctness), conservative: what cannot be proven const is a write.

  object  (a glvalue that IS the member or a sub-object of it): assignment / compound assignment / ++ / -- on it,
          a non-const method or operator call on it (const ones get a NoOp / DerivedToBase cast to const first), binding
          to a non-const reference (variable, parameter, constructor parameter, aggregate member, by-ref capture,
          structured binding, range-for loop variable), returning it as a non-const reference, a cast to a non-const
          reference or to an rvalue reference (move), taking its address (-> handle); propagation through parentheses,
          member access, subscripts, ?: branches, the right side of ',', derived-to-base casts.
  handle  (a pointer / iterator / view INTO the member's storage: &x_, x_.data(), x_.begin(), array decay, a returned
          handle): a read only when it is compared, discarded, or passed to a measured std read-only algorithm
          (STD_READ_ALGOS, callee not declared in whirl::server); a dereference / subscript / arrow writes through it
          when the result is written; kept in a variable, stored, returned, passed anywhere else = write.
  loaded pointer (a pointer VALUE read out of the member, e.g. queue_.front()->id): a write when the pointee is written
          through the expression itself (as the text census A1 (a)); copying the pointer is a read.

Accessors (S0b W11, S0c N10-N15, R16 / R17): a function whose return statement hands out a member as a non-const
reference / handle writes it, and every call site that does not provably only read the result writes it too
(callers are matched by mangled name across TUs).

Usage (engine_split_census.py runs it):
  python -I engine_split_ast.py --root R --out ast.json --classes Engine,JobQueue,... --tu src/server/engine.cpp ...
         [--cache DIR] [--jobs N] [--callable-ok vram_probe_,...]
Output ast.json: {"hits": [[rel, index, member, "w"|"r", why, weak]], "calls": [[rel, index, callee, verdict, why,
ret_of]], "leaks": {mangled: [[member, why]]}, "funcs": {mangled: name}, "tus": {tu: {"errors": [...], "deps": [...]}}}
(index = character offset in the file read as text, the same offsets engine_split_census.py uses).
Exit 0 ok, 3 clang missing / crash.
"""
import hashlib, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "s0d-1"


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a else default


def find_clang():
    c = os.environ.get("WHIRL_CLANG")
    if c and os.path.exists(c): return c
    cands = []
    hp = os.environ.get("HIP_PATH")
    if hp: cands.append(os.path.join(hp, "bin", "clang++.exe"))
    cands += [r"C:\Program Files\AMD\ROCm\7.2\bin\clang++.exe", r"C:\Program Files\AMD\ROCm\6.4\bin\clang++.exe"]
    for c in cands:
        if os.path.exists(c): return c
    return shutil.which("clang++")


# product build flags (build.ninja whirl_server_core: DEFINES / -std:c++20; INCLUDES src, include, generated)
DEFINES = ["-DNOMINMAX", "-DWHIRL_WITH_HIP=1", "-DWIN32_LEAN_AND_MEAN", "-D_CRT_SECURE_NO_WARNINGS"]
STDFLAGS = ["--target=x86_64-pc-windows-msvc", "-std=c++20", "-fsyntax-only", "-Wno-everything"]
VERSION_H = ('#pragma once\n#define WHIRL_VERSION_MAJOR 0\n#define WHIRL_VERSION_MINOR 0\n#define WHIRL_VERSION_PATCH 0\n'
             '#define WHIRL_VERSION_STRING "0.0.0"\n#define WHIRL_GPU_ARCHS_STRING "census"\n')

# ---- measured lists (each name checked against its std declaration; anything else is a write) ----
# std methods that hand out access to the object (a non-const overload exists next to a const one): the call itself
# is not a write; what is done with the result decides
ACCESSORS = {"begin", "end", "rbegin", "rend", "data", "front", "back", "at", "operator[]", "operator*", "operator->",
             "find", "lower_bound", "upper_bound", "equal_range", "get", "value", "top", "operator std::basic_string_view"}
# std algorithms / functions that only read through the iterators / pointers they get (callee not declared in whirl::server)
STD_READ_ALGOS = {"find", "find_if", "find_if_not", "any_of", "all_of", "none_of", "count", "count_if", "distance",
                  "equal", "mismatch", "search", "lower_bound", "upper_bound", "binary_search", "equal_range",
                  "accumulate", "min_element", "max_element", "minmax_element", "is_sorted", "adjacent_find",
                  "includes", "lexicographical_compare", "memcmp", "strlen", "strncmp", "strcmp", "next", "prev",
                  "advance_copy", "string_view", "basic_string_view"}
# std functions taking forwarding references that only read them (std::format: its arguments are formatted, not changed)
STD_FWD_READ = {"format", "print", "println", "vformat", "make_format_args", "formatted_size"}
# whirl functions with forwarding-reference parameters that only format them (src/server/log.h:27-38: std::format(f,
# std::forward<Args>(args)...)); checked at their definition
WHIRL_FWD_READ = {"logI", "logW", "logE", "logD"}
# lock guards: constructing one from a mutex member locks it (synchronization, not a change of the guarded data; the
# text census never counted it either). A direct q_mutex_.lock() stays a write (non-const method).
LOCK_TYPES = ("whirl::MutexLock", "whirl::UniqueLock", "whirl::SharedLock", "std::lock_guard<", "std::unique_lock<",
              "std::scoped_lock<", "std::shared_lock<", "MutexLock", "UniqueLock", "lock_guard<", "unique_lock<", "scoped_lock<")
# std value wrappers whose converting constructor takes U&& and copies (no reference template argument)
STD_VALUE_WRAPPERS = ("std::optional<", "std::pair<", "std::variant<", "std::tuple<", "optional<", "pair<", "variant<", "tuple<")
# std class templates that, constructed from iterators / pointers, copy what they point to (read)
STD_COPY_TYPES = ("std::basic_string<", "std::basic_string_view<", "std::vector<", "std::deque<", "basic_string<",
                  "basic_string_view<")
CMP_OPS = {"==", "!=", "<", ">", "<=", ">=", "<=>"}
STMT_KINDS = {"CompoundStmt", "IfStmt", "WhileStmt", "ForStmt", "DoStmt", "SwitchStmt", "CaseStmt", "DefaultStmt",
              "LabelStmt", "AttributedStmt"}
FUNC_KINDS = {"FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl", "CXXConversionDecl"}
TRANSP = {"ParenExpr", "MaterializeTemporaryExpr", "CXXBindTemporaryExpr", "ExprWithCleanups", "ConstantExpr",
          "SubstNonTypeTemplateParmExpr", "CXXDefaultArgExpr", "CXXDefaultInitExpr", "FullExpr"}
XCASTS = {"CStyleCastExpr", "CXXStaticCastExpr", "CXXFunctionalCastExpr", "CXXConstCastExpr", "CXXReinterpretCastExpr",
          "BuiltinBitCastExpr", "CXXAddrspaceCastExpr", "CXXDynamicCastExpr"}
DEPENDENT = {"CXXDependentScopeMemberExpr", "UnresolvedMemberExpr", "UnresolvedLookupExpr", "CXXUnresolvedConstructExpr",
             "ParenListExpr", "CXXFoldExpr", "DependentScopeDeclRefExpr", "PackExpansionExpr", "RecoveryExpr"}


def T(n):
    t = n.get("type") or {}
    return t.get("desugaredQualType") or t.get("qualType") or ""


def TQ(n):
    t = n.get("type") or {}
    return (t.get("qualType") or "") + " | " + (t.get("desugaredQualType") or "")


def is_const(t):
    return t.startswith("const ") or t.startswith("volatile const ") or t.startswith("const volatile ")


def ptr_const_pointee(n):
    """a handle whose pointee is const: `const T *`, a const_iterator, a span / view of const"""
    q = TQ(n)
    for t in q.split(" | "):
        t = t.strip()
        if not t: continue
        if t.endswith("*") and is_const(t): return True
        if t.endswith("* const") and is_const(t): return True
        if re.search(r"const_iterator|const_reverse_iterator", t): return True
        if re.search(r"\bspan<const\b|string_view|basic_string_view", t): return True
    return False


def is_pointer(t):
    return t.rstrip().endswith("*") or t.rstrip().endswith("* const")


class TU:
    def __init__(self, objs, root):
        self.root = root
        self.parent = {}
        self.pos = {}       # id(node) -> (file, offset)
        self.decls = {}     # clang id -> node (decls in the dump)
        self.pattern = set()  # id(node) of nodes inside template patterns
        self.refs = {}      # clang decl id -> [DeclRefExpr nodes] (a reference / pointer variable's uses)
        self.files = set()
        self.objs = objs
        for o in objs:
            self._index(o)

    def _index(self, root):
        # iterative walk in document order: the dumper elides "file" when it equals the last printed one
        st = {"file": None}
        stack = [(root, None, False)]
        while stack:
            n, par, pat = stack.pop()
            if not isinstance(n, dict): continue
            self.parent[id(n)] = par
            if pat: self.pattern.add(id(n))
            if "id" in n and n.get("kind", "").endswith("Decl"): self.decls.setdefault(n["id"], n)
            if n.get("kind") == "DeclRefExpr" and isinstance(n.get("referencedDecl"), dict):
                self.refs.setdefault(n["referencedDecl"].get("id"), []).append(n)
            first = None
            for key in ("loc", "range"):
                v = n.get(key)
                if v is None: continue
                want = None
                if key == "range" and isinstance(v.get("begin"), dict):
                    b = v["begin"]
                    want = b.get("expansionLoc") if "expansionLoc" in b else b
                for loc in self._bare(v):
                    if "file" in loc: st["file"] = loc["file"]; self.files.add(loc["file"])
                    if loc is want and "offset" in loc: first = (st["file"], loc["offset"])
            if first: self.pos[id(n)] = first
            kids = n.get("inner") or []
            k = n.get("kind")
            # a template's first FunctionDecl / CXXRecordDecl is the pattern (dependent); the specializations follow
            pat_child = None
            if k in ("FunctionTemplateDecl", "ClassTemplateDecl"):
                for c in kids:
                    if isinstance(c, dict) and c.get("kind") in FUNC_KINDS | {"CXXRecordDecl"}:
                        pat_child = c; break
            for c in reversed(kids):
                stack.append((c, n, pat or c is pat_child))

    @staticmethod
    def _bare(v):
        """bare locations in printing order: loc | {spellingLoc, expansionLoc} | range {begin, end}"""
        out = []
        if "begin" in v or "end" in v:
            for k in ("begin", "end"):
                if k in v: out += TU._bare(v[k])
            return out
        if "spellingLoc" in v or "expansionLoc" in v:
            for k in ("spellingLoc", "expansionLoc"):
                if k in v: out.append(v[k])
            # the expansion location is the one inside the body text; put it last so it wins for the node's position
            return out
        return [v]

    def par(self, n):
        return self.parent.get(id(n))

    def kids(self, n):
        return [c for c in (n.get("inner") or []) if isinstance(c, dict)]

    def index_in(self, P, n):
        for i, c in enumerate(P.get("inner") or []):
            if c is n: return i
        return -1

    def callee_decl(self, call):
        """the callee of a CallExpr / CXXOperatorCallExpr: (name, decl node or None, decl id)"""
        k = self.kids(call)
        if not k: return None, None, None
        c = k[0]
        for _ in range(6):
            if c.get("kind") in ("ImplicitCastExpr", "ParenExpr") and self.kids(c): c = self.kids(c)[0]; continue
            break
        if c.get("kind") == "DeclRefExpr":
            rd = c.get("referencedDecl") or {}
            return rd.get("name"), self.decls.get(rd.get("id")), rd.get("id")
        if c.get("kind") == "MemberExpr":
            return c.get("name"), self.decls.get(c.get("referencedMemberDecl")), c.get("referencedMemberDecl")
        if c.get("kind") in DEPENDENT: return "<dependent>", None, None
        return None, None, None

    def enclosing_func(self, n):
        p = self.par(n)
        while p is not None:
            k = p.get("kind")
            if k == "LambdaExpr": return p
            if k in FUNC_KINDS:
                return p
            p = self.par(p)
        return None

    def is_lambda_fn(self, f):
        if f is None or f.get("kind") == "LambdaExpr": return True
        if f.get("name") != "operator()": return False
        p = self.par(f)
        while p is not None and p.get("kind") == "FunctionTemplateDecl": p = self.par(p)
        return p is not None and p.get("kind") == "CXXRecordDecl" and not p.get("name")


class Walk:
    """ExprMutationAnalyzer-style upward walk from one expression; returns (verdict, why, returned_from) with verdict
    'w' | 'r'; returned_from = the named function node whose return statement the expression flows into."""

    def __init__(self, tu, callable_ok=()):
        self.tu = tu
        self.callable_ok = set(callable_ok)

    def run(self, E, mode, member=None, depth=0):
        v, why, rf = self._run(E, mode, member)
        # findReferenceMutation / findPointeeMutation: a non-const reference / handle variable made from the member is
        # followed through its uses; one that flows into a return hands the member out (S0b review N12)
        var = getattr(self, "_var", None)
        self._var = None
        if v == "w" and var is not None and rf is None and depth < 3:
            m2 = "obj" if "reference" in why else "ptr"
            for ref in self.tu.refs.get(var.get("id"), []):
                v2, why2, rf2 = self.run(ref, m2, member, depth + 1)
                if rf2 is not None:
                    return v, f"{why}; {why2}", rf2
        return v, why, rf

    def _run(self, E, mode, member=None):
        tu = self.tu
        self._var = None
        cur = E
        for _ in range(400):
            P = tu.par(cur)
            if P is None: return "w", "no parent", None
            k = P.get("kind")
            i = tu.index_in(P, cur)
            if k in DEPENDENT or "<dependent type>" in T(P):
                return "w", f"dependent {k} (weak)", None
            if k in TRANSP:
                cur = P; continue
            r = getattr(self, "m_" + mode)(cur, P, k, i, member)
            if r is None:
                return "w", f"unknown parent {k} ({mode})", None
            if r[0] == "go":
                cur, mode = r[1], r[2]; continue
            return r
        return "w", "walk too deep", None

    # ---------------- object mode ----------------
    def m_obj(self, cur, P, k, i, member):
        tu = self.tu
        if cur.get("valueCategory") in ("lvalue", "xvalue") and is_const(T(cur)):
            return self._const_obj(cur, cur, member)
        if k.endswith("Attr"): return ("r", f"attribute {k}", None)
        if k == "ImplicitCastExpr":
            ck = P.get("castKind")
            if ck in ("LValueToRValue", "ToVoid", "LValueToRValueBitCast"):
                if ck == "LValueToRValue" and is_pointer(T(P)) and not ptr_const_pointee(P):
                    return ("go", P, "lptr")
                return ("r", "read (value)", None)
            if P.get("valueCategory") in ("lvalue", "xvalue"):
                if is_const(T(P)):
                    return self._const_obj(cur, P, member)
                if P.get("valueCategory") == "xvalue": return ("w", "cast to an rvalue reference (move)", None)
                return ("go", P, "obj")
            if ck == "ArrayToPointerDecay": return ("go", P, "ptr")
            if ck in ("NoOp", "UserDefinedConversion", "ConstructorConversion"): return ("go", P, "obj")
            return ("w", f"implicit cast {ck}", None)
        if k in XCASTS:
            ck = P.get("castKind")
            if ck == "ToVoid": return ("r", "discarded", None)
            vc = P.get("valueCategory")
            if vc == "xvalue": return ("w", "cast to an rvalue reference (move)", None)
            if vc == "lvalue":
                if is_const(T(P)): return ("r", "cast to a const reference", None)
                return ("go", P, "obj")
            if ck in ("LValueToRValue",): return ("r", "value cast", None)
            if ck in ("NoOp", "ConstructorConversion", "UserDefinedConversion"): return ("go", P, "obj")
            return ("w", f"explicit cast {ck}", None)
        if k == "MemberExpr":
            if T(P) == "<bound member function type>" or (P.get("type") or {}).get("qualType") == "<bound member function type>":
                return self._method(cur, P, member, base_mode="obj")
            return ("go", P, "obj")   # a sub-object (field) of the member
        if k == "ArraySubscriptExpr":
            if i == 0: return ("go", P, "obj")
            return ("r", "subscript index", None)
        if k == "UnaryOperator":
            op = P.get("opcode")
            if op in ("++", "--"): return ("w", f"{op}", None)
            if op == "&": return ("go", P, "ptr")
            if op in ("*", "__extension__"): return ("go", P, "obj")
            if op in ("real", "imag"): return ("go", P, "obj")
            return ("r", f"operand of {op}", None)
        if k == "BinaryOperator":
            op = P.get("opcode")
            if op == "=": return ("w", "assignment", None) if i == 0 else ("r", "assigned from", None)
            if op == ",": return ("go", P, "obj") if i == 1 else ("r", "left of ','", None)
            if op in (".*", "->*"): return ("go", P, "obj") if i == 0 else ("r", "member pointer operand", None)
            return ("r", f"operand of {op}", None)
        if k == "CompoundAssignOperator":
            return ("w", f"compound assignment {P.get('opcode')}", None) if i == 0 else ("r", "operand", None)
        if k in ("ConditionalOperator", "BinaryConditionalOperator"):
            if i == 0: return ("r", "condition", None)
            return ("go", P, "obj")
        if k == "CXXOperatorCallExpr":
            name, d, _ = tu.callee_decl(P)
            if i == 0: return ("r", "callee", None)
            if i == 1:
                if name in ("operator[]", "operator*", "operator->"):
                    return self._result(P, f"non-const {name}")
                if name == "operator()":
                    return ("w", "non-const operator() called", None)
                return ("w", f"{name or 'operator'} on it", None)
            return ("w", f"argument of {name or 'operator'} (non-const reference)", None)
        if k == "CallExpr":
            name, d, _ = tu.callee_decl(P)
            if i == 0: return ("w", "called through", None)
            if name == "<dependent>": return ("w", "argument of a dependent call (weak)", None)
            if d is None and name in STD_FWD_READ: return ("r", f"formatted by std::{name}", None)
            if name in WHIRL_FWD_READ: return ("r", f"formatted by {name}", None)
            return ("w", f"argument of {name or '<unknown callee>'}() bound to a non-const reference", None)
        if k == "CXXMemberCallExpr":
            if i == 0: return ("r", "callee", None)
            mx = tu.kids(P)[0] if tu.kids(P) else {}
            return ("w", f"argument of {mx.get('name', '?')}() bound to a non-const reference", None)
        if k in ("CXXConstructExpr", "CXXTemporaryObjectExpr"):
            t = T(P)
            if any(t.startswith(x) for x in LOCK_TYPES): return ("r", f"locked by {t[:40]}", None)
            if any(t.startswith(x) for x in STD_VALUE_WRAPPERS) and "&" not in t and "reference_wrapper" not in t:
                return ("r", f"copied into {t[:40]}", None)
            return ("w", f"bound to a non-const reference parameter of a constructor of {t[:60]}", self._leak_ret(P))
        if k == "InitListExpr":
            return ("w", "bound to a reference member of an aggregate / init list", self._leak_ret(P))
        if k == "CXXStdInitializerListExpr": return ("go", P, "obj")
        if k in ("VarDecl", "DecompositionDecl", "FieldDecl", "BindingDecl"):
            if k == "VarDecl" and P.get("isImplicit") and (P.get("name") or "").startswith("__range"):
                return self._range_for(P)
            if k == "VarDecl": self._var = P
            return ("w", f"bound to a non-const reference {'variable ' + (P.get('name') or '') if k == 'VarDecl' else k}", None)
        if k == "LambdaExpr": return ("w", "captured by reference", None)
        if k == "ReturnStmt":
            f = tu.enclosing_func(P)
            return ("w", "returned as a non-const reference", None if tu.is_lambda_fn(f) else f)
        if k in STMT_KINDS: return ("r", "discarded / condition", None)
        if k in ("UnaryExprOrTypeTraitExpr", "CXXTypeidExpr", "CXXNoexceptExpr", "DecltypeType", "GenericSelectionExpr"):
            return ("r", "unevaluated", None)
        if k == "CXXPseudoDestructorExpr": return ("r", "pseudo destructor", None)
        if k == "CXXThrowExpr": return ("w", "thrown", None)
        if k == "CXXDeleteExpr": return ("w", "deleted", None)
        if k == "CXXForRangeStmt": return ("w", "range of a range-for", None)
        if k == "CXXRewrittenBinaryOperator": return ("go", P, "obj")
        if k == "OpaqueValueExpr": return ("go", P, "obj")
        if k == "CXXCtorInitializer": return ("w", "bound to a non-const reference member", None)
        return None

    def _const_obj(self, cur, P, member):
        """the object became const (a const method / operator / const& parameter follows): a read, except a
        std::function call (S0c review consider 1: a member listed in the cut's callable_reads only) and a smart
        pointer's -> / * / get() (they hand out a non-const pointee)"""
        tu = self.tu
        G = P
        while tu.par(G) is not None and tu.par(G).get("kind") in TRANSP | {"ImplicitCastExpr"}: G = tu.par(G)
        GP = tu.par(G)
        if GP is not None and GP.get("kind") == "CXXOperatorCallExpr" and tu.index_in(GP, G) == 1:
            name, _, _ = tu.callee_decl(GP)
            if name == "operator()" and re.search(r"std::(?:function|_Func_class|move_only_function)", TQ(P) + TQ(cur)):
                if member in self.callable_ok: return ("r", "std::function call (cut callable_reads)", None)
                return ("w", "std::function member called (not in the cut's callable_reads)", None)
            if name in ("operator->", "operator*") and re.search(r"std::(?:unique_ptr|shared_ptr)", TQ(P)):
                return self._result(GP, f"smart pointer {name}", pointee=True)
        if GP is not None and GP.get("kind") == "MemberExpr" and GP.get("name") == "get" and re.search(r"std::(?:unique_ptr|shared_ptr)", TQ(P)):
            call = tu.par(GP)
            if call is not None: return ("go", call, "lptr")
        return ("r", "const use", None)

    def _result(self, call, why, pointee=False):
        vc = call.get("valueCategory")
        if vc == "xvalue": return ("w", f"{why} (rvalue)", None)
        if vc == "lvalue": return ("go", call, "obj")
        t = T(call)
        if is_pointer(t) or re.search(r"iterator|span<|reference_wrapper", t):
            return ("go", call, "lptr" if pointee else "ptr")
        return ("r", f"{why} returns a value", None)

    def _method(self, cur, ME, member, base_mode):
        tu = self.tu
        call = tu.par(ME)
        name = ME.get("name") or "?"
        d = tu.decls.get(ME.get("referencedMemberDecl"))
        pairs = False
        if d is not None:
            # a whirl method with a const overload of the same name in its class (PrefixStore::spes()): an accessor
            cls = tu.par(d)
            if cls is not None:
                pairs = any(c.get("kind") == "CXXMethodDecl" and c.get("name") == name and
                            re.search(r"\)\s*const\b", (c.get("type") or {}).get("qualType", "")) for c in tu.kids(cls))
        if call is None or call.get("kind") != "CXXMemberCallExpr":
            return ("w", f"method {name} referenced", None)
        if name in ACCESSORS and d is None or pairs or name.startswith("operator std::basic_string_view"):
            return self._result(call, f"non-const {name}()")
        return ("w", f"non-const method {name}()", None)

    def _range_for(self, rv):
        tu = self.tu
        ds = tu.par(rv)
        fr = tu.par(ds) if ds is not None else None
        if fr is None or fr.get("kind") != "CXXForRangeStmt": return ("w", "range variable", None)
        loopv = None
        for c in tu.kids(fr):
            if c.get("kind") == "DeclStmt":
                for v in tu.kids(c):
                    if v.get("kind") in ("VarDecl", "DecompositionDecl") and not v.get("isImplicit"): loopv = v
        if loopv is None: return ("w", "range-for without a loop variable", None)
        t = T(loopv)
        if "&" in t and not is_const(t): return ("w", f"range-for loop variable {loopv.get('name') or '[..]'} is a non-const reference ({t[:40]})", None)
        if is_pointer(t) and not is_const(t) and "*" in t:
            return ("r", "range-for copies pointers", None)
        return ("r", "range-for by value / const reference", None)

    def _leak_ret(self, node):
        """a construct / init list made from the member: does it flow into a return statement of a named function"""
        tu = self.tu
        cur = node
        for _ in range(40):
            P = tu.par(cur)
            if P is None: return None
            k = P.get("kind")
            if k in TRANSP or k in ("CXXConstructExpr", "CXXFunctionalCastExpr", "ImplicitCastExpr", "InitListExpr"):
                cur = P; continue
            if k == "ReturnStmt":
                f = tu.enclosing_func(P)
                return None if tu.is_lambda_fn(f) else f
            return None
        return None

    # ---------------- handle mode: a pointer / iterator / view into the member's storage ----------------
    def m_ptr(self, cur, P, k, i, member):
        tu = self.tu
        if k.endswith("Attr"): return ("r", f"attribute {k}", None)
        if k == "ImplicitCastExpr":
            ck = P.get("castKind")
            if ck in ("PointerToBoolean", "PointerToIntegral", "IntegralCast", "ToVoid"): return ("r", "handle tested / converted", None)
            if is_pointer(T(P)) and ptr_const_pointee(P): return ("r", "handle to const", None)
            if re.search(r"const_iterator", TQ(P)): return ("r", "const iterator", None)
            return ("go", P, "ptr")
        if k in XCASTS:
            if P.get("castKind") == "ToVoid": return ("r", "discarded", None)
            if ptr_const_pointee(P): return ("r", "handle cast to const", None)
            return ("go", P, "ptr")
        if k == "UnaryOperator":
            op = P.get("opcode")
            if op == "*": return ("go", P, "obj")
            if op == "!": return ("r", "handle tested", None)
            if op in ("++", "--"): return ("go", P, "ptr")
            return ("w", f"handle operand of {op}", None)
        if k == "ArraySubscriptExpr": return ("go", P, "obj")
        if k == "MemberExpr":
            if (P.get("type") or {}).get("qualType") == "<bound member function type>":
                if P.get("isArrow"): return self._method(cur, P, member, "ptr")
                call = tu.par(P)
                nm = P.get("name") or "?"
                if nm in ("base", "operator->", "operator*", "get"):
                    return self._result(call, nm) if call is not None else ("w", nm, None)
                return ("w", f"method {nm}() on a handle", None)
            if P.get("isArrow"): return ("go", P, "obj")
            return ("w", "member of a handle", None)
        if k == "BinaryOperator":
            op = P.get("opcode")
            if op in CMP_OPS: return ("r", "handle compared", None)
            if op in ("+", "-"):
                if is_pointer(T(P)): return ("go", P, "ptr")
                return ("r", "handle difference", None)
            if op == ",": return ("go", P, "ptr") if i == 1 else ("r", "discarded", None)
            if op in ("&&", "||"): return ("r", "handle tested", None)
            if op == "=": return ("w", "handle stored", None) if i == 1 else ("r", "handle variable assigned", None)
            return ("w", f"handle operand of {op}", None)
        if k == "CompoundAssignOperator": return ("w", "handle stored / moved", None)
        if k in ("ConditionalOperator", "BinaryConditionalOperator"):
            return ("r", "condition", None) if i == 0 else ("go", P, "ptr")
        if k == "CXXOperatorCallExpr":
            name, d, _ = tu.callee_decl(P)
            op = (name or "").replace("operator", "")
            if op in CMP_OPS: return ("r", "handle compared", None)
            if i == 1 and name in ("operator*", "operator[]"): return self._result(P, name)
            if i == 1 and name == "operator->": return ("go", P, "ptr")
            if i == 1 and name in ("operator+", "operator-", "operator++", "operator--"):
                return ("go", P, "ptr") if not re.search(r"\b(?:long|int|ptrdiff|difference_type)\b", T(P)) else ("r", "handle difference", None)
            if i == 2 and name in ("operator+", "operator-"): return ("go", P, "ptr")
            if name == "operator=" and i == 2: return ("w", "handle stored", None)
            return ("w", f"handle passed to {name}", None)
        if k == "CallExpr":
            name, d, _ = tu.callee_decl(P)
            if i == 0: return ("w", "handle called", None)
            if d is None and name in STD_READ_ALGOS: return ("r", f"handle read by std::{name}", None)
            return ("w", f"handle passed to {name or '<unknown callee>'}()", None)
        if k == "CXXMemberCallExpr":
            mx = tu.kids(P)[0] if tu.kids(P) else {}
            return ("w", f"handle passed to {mx.get('name', '?')}()", None)
        if k in ("CXXConstructExpr", "CXXTemporaryObjectExpr"):
            t = T(P)
            if any(t.startswith(x) or (" " + x) in t for x in STD_COPY_TYPES): return ("r", f"copied into {t[:40]}", None)
            if len(tu.kids(P)) == 1 and (T(P).split("<")[0].split("::")[-1] in T(cur) or "iterator" in t):
                return ("go", P, "ptr")   # a copy of the handle
            return ("w", f"handle passed to a constructor of {t[:60]}", self._leak_ret(P))
        if k in ("VarDecl", "DecompositionDecl", "FieldDecl", "BindingDecl"):
            if k == "VarDecl" and P.get("isImplicit") and (P.get("name") or "").startswith("__"):
                return ("w", "handle in an implicit variable", None)
            if k == "VarDecl": self._var = P
            return ("w", f"handle kept in {'variable ' + (P.get('name') or '') if k == 'VarDecl' else k}", None)
        if k == "ReturnStmt":
            f = tu.enclosing_func(P)
            return ("w", "handle returned", None if tu.is_lambda_fn(f) else f)
        if k in ("InitListExpr", "LambdaExpr", "CXXStdInitializerListExpr"): return ("w", f"handle kept ({k})", self._leak_ret(P))
        if k in STMT_KINDS: return ("r", "handle discarded / tested", None)
        if k in ("UnaryExprOrTypeTraitExpr", "CXXTypeidExpr", "CXXNoexceptExpr"): return ("r", "unevaluated", None)
        if k == "CXXRewrittenBinaryOperator": return ("r", "handle compared", None)
        if k == "CXXDeleteExpr": return ("w", "deleted", None)
        if k == "CXXForRangeStmt": return ("w", "handle range", None)
        return None

    # ---------------- loaded pointer: a pointer value read out of the member ----------------
    def m_lptr(self, cur, P, k, i, member):
        tu = self.tu
        if k == "ImplicitCastExpr" or k in XCASTS:
            if is_pointer(T(P)) and ptr_const_pointee(P): return ("r", "pointer to const", None)
            if is_pointer(T(P)) or P.get("castKind") == "NoOp": return ("go", P, "lptr")
            return ("r", "pointer value used", None)
        if k == "UnaryOperator" and P.get("opcode") == "*": return ("go", P, "obj")
        if k == "ArraySubscriptExpr" and i == 0: return ("go", P, "obj")
        if k == "MemberExpr" and P.get("isArrow"):
            if (P.get("type") or {}).get("qualType") == "<bound member function type>":
                return self._method(cur, P, member, "lptr")
            return ("go", P, "obj")
        if k in ("ConditionalOperator", "BinaryConditionalOperator") and i > 0: return ("go", P, "lptr")
        if k == "BinaryOperator" and P.get("opcode") == "," and i == 1: return ("go", P, "lptr")
        if k == "BinaryOperator" and P.get("opcode") in ("+", "-") and is_pointer(T(P)): return ("go", P, "lptr")
        return ("r", "pointer value copied / compared", None)


def tracked_fields(tu, classes):
    """clang id -> (class, field name) for the data members of the tracked classes (definitions in whirl::server)"""
    out = {}
    for d in list(tu.decls.values()):
        if d.get("kind") == "CXXRecordDecl" and d.get("name") in classes and d.get("completeDefinition"):
            p = tu.par(d)
            if p is not None and p.get("kind") not in ("NamespaceDecl", "TranslationUnitDecl", "LinkageSpecDecl"): continue
            for c in tu.kids(d):
                if c.get("kind") == "FieldDecl": out[c["id"]] = (d["name"], c.get("name"))
    return out


def analyze(objs, root, classes, callable_ok):
    tu = TU(objs, root)
    fields = tracked_fields(tu, classes)
    W = Walk(tu, callable_ok)
    hits, calls = [], []
    seen = set()
    implicit_fn = set()
    for d in tu.decls.values():
        if d.get("kind") in FUNC_KINDS and d.get("isImplicit"): implicit_fn.add(id(d))
    stack = [o for o in objs]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict): continue
        if id(n) in implicit_fn: continue
        k = n.get("kind")
        if k == "MemberExpr" and n.get("referencedMemberDecl") in fields or \
                k == "DeclRefExpr" and (n.get("referencedDecl") or {}).get("id") in fields:
            fid = n.get("referencedMemberDecl") if k == "MemberExpr" else n["referencedDecl"]["id"]
            cls, mem = fields[fid]
            pos = tu.pos.get(id(n))
            if pos and (pos, id(n)) not in seen:
                seen.add((pos, id(n)))
                if k == "DeclRefExpr":
                    P = tu.par(n)
                    if P is not None and P.get("kind") == "UnaryOperator" and P.get("opcode") == "&":
                        v, why, rf = "w", "data member pointer taken", None
                    else:
                        v, why, rf = W.run(n, "obj", mem)
                else:
                    v, why, rf = W.run(n, "obj", mem)
                weak = id(n) in tu.pattern or "(weak)" in why
                hits.append([pos, mem, cls, v, why, weak, mangled(rf)])
        if k in ("CXXMemberCallExpr", "CallExpr"):
            name, d, did = tu.callee_decl(n)
            if d is not None and d.get("mangledName"):
                pos = tu.pos.get(id(n))
                vc = n.get("valueCategory")
                if vc == "lvalue": v, why, rf = W.run(n, "obj")
                elif vc == "xvalue": v, why, rf = "w", "rvalue result", None
                else: v, why, rf = W.run(n, "ptr")
                if pos: calls.append([pos, d["mangledName"], v, why, mangled(rf), id(n) in tu.pattern])
        stack.extend(reversed(n.get("inner") or []))
    funcs = {}
    for d in tu.decls.values():
        if d.get("kind") in FUNC_KINDS and d.get("mangledName"):
            funcs[d["mangledName"]] = d.get("name")
    return hits, calls, funcs, tu.files


def mangled(f):
    return f.get("mangledName") if isinstance(f, dict) else None


def read_objs(txt):
    dec = json.JSONDecoder()
    i, out = 0, []
    ws = re.compile(r"\S")
    while True:
        m = ws.search(txt, i)
        if not m: break
        o, i = dec.raw_decode(txt, m.start())
        out.append(o)
    return out


class PosMap:
    """byte offset -> character index of the file as Python text mode reads it (universal newlines)"""
    cache = {}

    @classmethod
    def get(cls, path):
        if path in cls.cache: return cls.cache[path]
        raw = open(path, "rb").read()
        mp = [0] * (len(raw) + 1)
        ci, bi, n = 0, 0, len(raw)
        while bi < n:
            b = raw[bi]
            if b == 0x0D and bi + 1 < n and raw[bi + 1] == 0x0A:
                mp[bi] = ci; mp[bi + 1] = ci; bi += 2; ci += 1; continue
            ln = 1 if b < 0x80 else 2 if b >> 5 == 6 else 3 if b >> 4 == 14 else 4 if b >> 3 == 30 else 1
            for q in range(ln):
                if bi + q < n: mp[bi + q] = ci
            bi += ln; ci += 1
        mp[n] = ci
        cls.cache[path] = mp
        return mp


def rel_of(root, f):
    p = os.path.normcase(os.path.normpath(f))
    for sub in ("src", "include"):
        b = os.path.normcase(os.path.normpath(os.path.join(root, sub))) + os.sep
        if p.startswith(b):
            r = os.path.relpath(os.path.normpath(f), os.path.join(root, sub)).replace("\\", "/")
            return r if sub == "src" else "include:" + r
    return None


def tu_flags(root, gen):
    return STDFLAGS + DEFINES + ["-I" + os.path.join(root, "src"), "-I" + os.path.join(root, "include"), "-I" + gen]


def deps_hash(root, deps, extra):
    h = hashlib.sha256(extra.encode())
    for d in deps:
        p = os.path.join(root, d)
        h.update(d.encode() + b"\0")
        try: h.update(hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).digest())
        except OSError: h.update(b"<missing>")
    return h.hexdigest()


def run_tu(root, tu_rel, classes, callable_ok, cache, clang, gen):
    """one TU: clang dump + analysis, cached by the content of every user file it reads (clang -MMD)"""
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]
    extra = "|".join([VERSION, me, ",".join(sorted(classes)), ",".join(sorted(callable_ok)), tu_rel])
    cdir = os.path.join(cache, re.sub(r"[^\w.]+", "_", tu_rel)) if cache else None
    if cdir and os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".deps"): continue
            deps = open(os.path.join(cdir, f), encoding="utf-8").read().split("\n")
            deps = [d for d in deps if d]
            h = deps_hash(root, deps, extra)
            rp = os.path.join(cdir, h + ".json")
            if os.path.exists(rp):
                try: return json.load(open(rp, encoding="utf-8"))
                except ValueError: pass
    src = os.path.join(root, tu_rel)
    dfile = os.path.join(cache or os.path.dirname(src), f"_dep_{os.getpid()}_{abs(hash(tu_rel))}.d")
    cmd = [clang] + tu_flags(root, gen) + ["-MMD", "-MF", dfile, "-Xclang", "-ast-dump=json",
                                          "-Xclang", "-ast-dump-filter=whirl::server", src]
    r = subprocess.run(cmd, capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    errors = [l for l in err.splitlines() if re.search(r"\berror\b:", l)][:20]
    deps = []
    try:
        dt = open(dfile, encoding="utf-8", errors="replace").read().replace("\\\n", " ")
        os.remove(dfile)
        body = dt.split(":", 2)[-1] if re.match(r"^[A-Za-z]:", dt) else dt.split(":", 1)[-1]
        # Windows paths: split on spaces not escaped; clang escapes spaces as '\ '
        toks = re.findall(r"(?:\\ |[^\s])+", dt[dt.find(": ") + 2:] if ": " in dt else body)
        for t in toks:
            t = t.replace("\\ ", " ")
            rr = os.path.relpath(os.path.normpath(t), root) if os.path.isabs(t) else t
            rr = rr.replace("\\", "/")
            if not rr.startswith(".."): deps.append(rr)
    except OSError:
        deps = [tu_rel]
    deps = sorted(set(deps) | {tu_rel})
    res = {"tu": tu_rel, "errors": errors, "rc": r.returncode, "deps": deps, "hits": [], "calls": [], "funcs": {}}
    if out.strip():
        objs = read_objs(out)
        hits, calls, funcs, files = analyze(objs, root, classes, callable_ok)
        def cpos(p):
            f, off = p
            rel = rel_of(root, f) if f else None
            if rel is None: return None
            full = os.path.join(root, "src", rel) if not rel.startswith("include:") else os.path.join(root, "include", rel[8:])
            try: return [rel, PosMap.get(full)[min(off, len(PosMap.get(full)) - 1)]]
            except OSError: return None
        for p, mem, cls, v, why, weak, rf in hits:
            q = cpos(p)
            if q: res["hits"].append(q + [mem, cls, v, why, weak, rf])
        for p, callee, v, why, rf, weak in calls:
            q = cpos(p)
            if q: res["calls"].append(q + [callee, v, why, rf, weak])
        res["funcs"] = funcs
    if cdir and not errors and r.returncode == 0:
        os.makedirs(cdir, exist_ok=True)
        h = deps_hash(root, deps, extra)
        tmp = os.path.join(cdir, f"{h}.json.{os.getpid()}.tmp")
        json.dump(res, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, os.path.join(cdir, h + ".json"))
        dp = os.path.join(cdir, hashlib.sha256("\n".join(deps).encode()).hexdigest()[:16] + ".deps")
        if not os.path.exists(dp):
            tmp = dp + f".{os.getpid()}.tmp"
            open(tmp, "w", encoding="utf-8").write("\n".join(deps) + "\n")
            os.replace(tmp, dp)
    return res


def main():
    root = arg("--root")
    out = arg("--out")
    classes = set(x for x in (arg("--classes", "Engine") or "").split(",") if x)
    callable_ok = set(x for x in (arg("--callable-ok", "") or "").split(",") if x)
    tus = [sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--tu"]
    cache = arg("--cache")
    jobs = int(arg("--jobs", "8"))
    clang = find_clang()
    if not clang:
        print("engine split ast: NO CLANG (set WHIRL_CLANG or HIP_PATH; the HIP SDK's clang++.exe)"); return 3
    gen = os.path.join(cache or os.path.join(root, "build"), "_generated")
    os.makedirs(os.path.join(gen, "whirl"), exist_ok=True)
    vh = os.path.join(gen, "whirl", "version.h")
    if not os.path.exists(vh): open(vh, "w").write(VERSION_H)
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(lambda t: run_tu(root, t, classes, callable_ok, cache, clang, gen), tus))
    json.dump({"clang": clang, "tus": res}, open(out, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
