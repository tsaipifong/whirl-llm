# SPDX-License-Identifier: Apache-2.0
"""arch025-emut2 (PLAN N4-emut2): the two member-write analyzers side by side, for engine_split_census.py.

The census runs BOTH the Python analyzer (engine_split_ast.py, clang -ast-dump=json) and clang's own
ExprMutationAnalyzer (tools/gates/emut, emut.exe on the official LLVM release) on the same TUs and compares them
(progress/arch025-emut.HANDOFF.md, section 整合提案; decided by the main session):
  writes = clang w  UNION  Python w, per (census entry, member):
  (a) clang write / Python read  -> a write, unless the cut pins the pair in "emut_read_ok"
      ("entry:member" -> {"clang_why": [every clang reason of the pair, positions dropped], "class", "reason"});
      a pin whose reason set differs does not apply; a pin that no longer occurs is a finding.
  (b) Python write / clang read  -> a write; every Python reason must belong to a known wider-rule class
      (WIDER below: the cross-check's (b) classes), otherwise a finding.
  (c) everything else fails closed: a TU list that differs from the census's, clang (emut) errors in a TU, a member use
      only one analyzer reports (position mismatch), emut.exe / LLVM missing or stale (census exit 1).
Holder calls (S4 requirement): a non-const call h.f( / h->f( on a cut holder member h (cut "holders" name -> class),
f a method of that class, is a CALL EDGE (the census's call graph has it), not a data write of h; the writes inside f
are judged in f's own body. The census decides which uses are holder calls (it has the class bodies) and passes them
in as `skip`; this module only applies it to both analyzers alike.

  python -I engine_split_emut.py --selftest      (unit cases on synthetic analyzer results)
"""
import json
import re
import sys


def index(ast, src_prefix="src/"):
    """an analyzer's ast.json -> positions, calls, function names, accessor leaks, coverage"""
    pos, calls, fname = {}, {}, {}
    covered, errfiles = set(), set()
    for t in ast["tus"]:
        if t["errors"] or t["rc"]:
            errfiles.add(t["tu"][len(src_prefix):] if t["tu"].startswith(src_prefix) else t["tu"])
        else:
            covered.update(d[len(src_prefix):] for d in t["deps"] if d.startswith(src_prefix))
        for rel, idx, mem, cls, v, why, weak, rf in t["hits"]:
            e = pos.setdefault((rel, idx, mem), {"strong": [], "weak": [], "rf": set()})
            e["weak" if weak else "strong"].append((v, why))
            if rf: e["rf"].add(rf)
        for rel, idx, callee, v, why, rf, weak in t["calls"]:
            e = calls.setdefault((rel, idx, callee), {"strong": [], "weak": [], "rf": set()})
            e["weak" if weak else "strong"].append((v, why))
            if rf: e["rf"].add(rf)
        fname.update(t["funcs"])
    ix = {"pos": pos, "calls": calls, "fname": fname, "covered": covered, "errfiles": errfiles}
    ix["leaks"] = _leaks(ix)
    ix["byfile"], ix["cbyfile"] = {}, {}
    for (rel, idx, mem), e in pos.items(): ix["byfile"].setdefault(rel, []).append((idx, mem, e))
    for (rel, idx, callee), e in calls.items(): ix["cbyfile"].setdefault(rel, []).append((idx, callee, e))
    return ix


def verdict(e):
    lst = e["strong"] or e["weak"]
    w = [why for v, why in lst if v == "w"]
    return w[0] if w else None


def _leaks(ix):
    """accessors: a function whose return hands out a member writes it; a call that does not only read the result writes
    what the callee hands out (fixpoint: an accessor returning another accessor's result)"""
    lk = {}
    for (rel, idx, mem), e in ix["pos"].items():
        if verdict(e):
            for rf in e["rf"]: lk.setdefault(rf, set()).add(mem)
    for _ in range(20):
        grew = False
        for (rel, idx, callee), e in ix["calls"].items():
            if callee in lk and verdict(e):
                for rf in e["rf"]:
                    s0 = lk.setdefault(rf, set()); n0 = len(s0); s0 |= lk[callee]; grew |= len(s0) != n0
        if not grew: break
    return lk


def body_writes(ix, ranges, skip=None):
    """writes of one census body (ranges = [(rel, start, end)]): {member: [(why, idx, rel)]} (every reason, in position
    order), {member: [read reasons]}, the positions inside the body. skip(rel, idx, member) -> reason or None: a use that
    is a holder call edge (counted as a read)."""
    allw, rd, inb = {}, {}, set()
    for rel, a, b in ranges:
        for idx, mem, e in sorted(ix["byfile"].get(rel, []), key=lambda x: (x[0], x[1])):
            if not (a <= idx < b): continue
            inb.add((rel, idx, mem))
            w = verdict(e)
            if w and skip is not None:
                s = skip(rel, idx, mem)
                if s:
                    rd.setdefault(mem, []).append(s); continue
            if w: allw.setdefault(mem, []).append((w, idx, rel))
            else: rd.setdefault(mem, []).extend(why for _v, why in (e["strong"] or e["weak"]))
        for idx, callee, e in sorted(ix["cbyfile"].get(rel, []), key=lambda x: (x[0], x[1])):
            if a <= idx < b and callee in ix["leaks"] and verdict(e):
                for mem in sorted(ix["leaks"][callee]):
                    allw.setdefault(mem, []).append((f"through {ix['fname'].get(callee, callee)}(): {verdict(e)}", idx, rel))
    return allw, rd, inb


def norm_why(w):
    """a clang reason without its position (`@line:col`) and the weak marker: what an emut_read_ok pin names"""
    return re.sub(r"\s*\(weak\)\s*$", "", re.sub(r"\s@\d+:\d+", "", w)).strip()


# (b) the Python analyzer's rules that are deliberately wider than ExprMutationAnalyzer (cross-check classes of
# progress/arch025-emut.HANDOFF.md): a Python-only write must come from one of them
WIDER = [
    ("loaded pointer / handle", r"^(?:pointer|handle)\b|pointee reachable|\bpointee\b|^unique_ptr|^span\b|T \*const|pointer FIELD|"
                                r"^pointer-to|two-level copy|^range-for|^range of a range-for|^range variable"),
    ("std::function call (cut callable_reads)", r"^std::function member called"),
    ("rvalue / move", r"\(rvalue\)|^rvalue result|^cast to an rvalue reference"),
    ("accessor caller (returned handle followed)", r"^through ~?\w+\(\)"),
]


def chain_pointee(text, idx, mem):
    """the use reaches a pointee: `*m...`, `m...->x`, `*m[i]`, `m.get()` copied (the access chain after the member)"""
    if not text or text[idx:idx + len(mem)] != mem: return False
    j = idx - 1
    while j >= 0 and text[j] in " \t(": j -= 1
    if j >= 0 and text[j] == "*" and (j == 0 or text[j - 1] not in "*/"): return True
    k, n, d = idx + len(mem), len(text), 0
    while k < n:
        c = text[k]
        if c in "([": d += 1
        elif c in ")]":
            if d == 0: break
            d -= 1
        elif d == 0 and (c in ";,{}=" and not text.startswith("==", k)): break
        if text.startswith("->", k): return True
        k += 1
    return False


def wider_class(why, text=None, idx=None, mem=None):
    for part in [p.strip() for p in why.split(";")]:
        for name, rx in WIDER:
            if re.search(rx, part): return name
    if text is not None and chain_pointee(text, idx, mem): return "pointee through a sub-object / handle (`->`, `*`)"
    return None


def judge(key, py_w, cl_w, pins, text_of):
    """one census body: py_w / cl_w = {member: [(why, idx, rel)]}. Returns (writes {member: (why, idx, rel)},
    rows [(key, member, py, clang, final, class, py_why, clang_why)] (the why columns: JSON lists), findings, pins used)"""
    out, rows, fnd, used = {}, [], [], set()
    for m in sorted(set(py_w) | set(cl_w)):
        k2 = f"{key}:{m}"
        cw = sorted({norm_why(w) for w, _, _ in cl_w.get(m, [])})
        pw = sorted({w for w, _, _ in py_w.get(m, [])})
        if m in py_w and m in cl_w:
            out[m] = cl_w[m][0]
            continue
        if m in cl_w:   # (a)
            pin = pins.get(k2)
            pv = pin.get("clang_why") if isinstance(pin, dict) else None
            pv = set(pv) if isinstance(pv, list) else ({pv} if pv else set())
            if pin is not None and pv == set(cw):
                used.add(k2)
                rows.append((key, m, "r", "w", "r", "a pinned: " + str(pin.get("class", "-")), "-", json.dumps(cw, ensure_ascii=False)))
                continue
            out[m] = cl_w[m][0]
            rows.append((key, m, "r", "w", "w", "a unpinned (clang write counted)", "-", json.dumps(cw, ensure_ascii=False)))
            continue
        # (b)
        cls = [wider_class(w, text_of(rel), idx, m) for w, idx, rel in py_w[m]]
        out[m] = py_w[m][0]
        if all(cls):
            rows.append((key, m, "w", "r", "w", "b " + " / ".join(sorted(set(cls))), json.dumps(pw, ensure_ascii=False), "-"))
        else:
            bad = [w for (w, _, _), c in zip(py_w[m], cls) if not c]
            rows.append((key, m, "w", "r", "w", "b UNCLASSIFIED", json.dumps(pw, ensure_ascii=False), "-"))
            fnd.append(f"Python write the clang analyzer calls a read, not a known wider rule: {k2} (Python: {' | '.join(sorted(set(bad)))})")
    return out, rows, fnd, used


def check_tus(want, py_ast, cl_ast):
    f = []
    for name, a in (("Python", py_ast), ("emut", cl_ast)):
        have = sorted(t["tu"] for t in a["tus"])
        if have != sorted(want):
            f.append(f"emut cross-check: {name} analyzer TU list differs from the census's ({len(have)} vs {len(want)}: "
                     f"missing {sorted(set(want) - set(have))[:4]}, extra {sorted(set(have) - set(want))[:4]})")
    return f


def check_positions(py_ix, cl_ix, limit=8):
    """every tracked member use must be seen by both analyzers (position = file, character index, member)"""
    a, b = set(py_ix["pos"]), set(cl_ix["pos"])
    f = []
    for name, d in (("only the Python analyzer", sorted(a - b)), ("only clang (emut)", sorted(b - a))):
        if d:
            f.append(f"emut cross-check: {len(d)} member uses seen by {name}: "
                     + ", ".join(f"{r}@{i} {m}" for r, i, m in d[:limit]) + (" ..." if len(d) > limit else ""))
    return f


# ---------------- unit self-test ----------------
def _ast(tus, hits=(), calls=(), errors=None, deps=None):
    return {"tus": [{"tu": t, "errors": (errors or {}).get(t, []), "rc": 0, "deps": deps or [t], "hits": list(hits) if i == 0 else [],
                     "calls": list(calls) if i == 0 else [], "funcs": {}} for i, t in enumerate(tus)]}


def selftest():
    T = "src/server/engine.cpp"
    txt = {"server/engine.cpp": "void f() { *rh_.p = 1; q_.front()->id = 0; x_ = 1; m_.f(a); }"}
    tof = lambda r: txt.get(r, "")
    i_rh, i_q, i_x = txt["server/engine.cpp"].index("rh_"), txt["server/engine.cpp"].index("q_"), txt["server/engine.cpp"].index("x_")
    res = []

    def case(cid, what, ok):
        res.append((cid, ok)); print(f"  {'ok  ' if ok else 'FAIL'} {cid} {what}")

    # (a)
    cw = {"lk_": [("write: CXXConstructExpr `lk(q_mutex_)` @12:3", 5, "server/engine.cpp")]}
    o, r, f, u = judge("runLoop", {}, cw, {}, tof)
    case("U1", "(a) clang write / Python read, no pin: counted as a write, no finding", "lk_" in o and not f and r[0][5].startswith("a unpinned"))
    pins = {"runLoop:lk_": {"clang_why": ["write: CXXConstructExpr `lk(q_mutex_)`"], "class": "d lock", "reason": "x"}}
    o, r, f, u = judge("runLoop", {}, cw, pins, tof)
    case("U2", "(a) pinned with the exact clang reason set: a read, pin used", "lk_" not in o and u == {"runLoop:lk_"})
    pins2 = {"runLoop:lk_": {"clang_why": ["write: something else"], "class": "d", "reason": "x"}}
    o, r, f, u = judge("runLoop", {}, cw, pins2, tof)
    case("U3", "(a) a pin whose clang reason differs does not apply", "lk_" in o and not u)
    cw2 = {"lk_": cw["lk_"] + [("write: BinaryOperator `lk_ = 1` @13:3", 9, "server/engine.cpp")]}
    o, r, f, u = judge("runLoop", {}, cw2, pins, tof)
    case("U4", "(a) a new clang reason next to a pinned one: the pin no longer covers the pair", "lk_" in o and not u)
    # (b)
    pw = {"rh_": [("assignment", i_rh, "server/engine.cpp")]}
    o, r, f, u = judge("kvTokBytes", pw, {}, {}, tof)
    case("U5", "(b) Python write through *rh_.p (pointee of a sub-object): a write, class known", "rh_" in o and not f)
    pw = {"q_": [("assignment", i_q, "server/engine.cpp")]}
    o, r, f, u = judge("kvTokBytes", pw, {}, {}, tof)
    case("U6", "(b) Python write through q_.front()->id: a write, class known", "q_" in o and not f)
    pw = {"m_": [("pointer passed to a CXXMemberCallExpr (non-const pointee)", 0, "server/engine.cpp")]}
    o, r, f, u = judge("govDrain", pw, {}, {}, tof)
    case("U7", "(b) loaded pointer (m_.stream() passed on): known", "m_" in o and not f)
    pw = {"on_gov_": [("std::function member called (not in the cut's callable_reads)", 0, "server/engine.cpp")]}
    case("U8", "(b) std::function call: known", not judge("runLoop", pw, {}, {}, tof)[2])
    pw = {"x_": [("assignment", i_x, "server/engine.cpp")]}
    o, r, f, u = judge("runLoop", pw, {}, {}, tof)
    case("U9", "(b) Python 'assignment' to x_ itself that clang calls a read: still a write AND a finding", "x_" in o and len(f) == 1)
    pw = {"y_": [("non-const method clear()", 0, "server/engine.cpp")]}
    case("U10", "(b) Python non-const method call that clang calls a read: a finding", len(judge("runLoop", pw, {}, {}, tof)[2]) == 1)
    # both
    o, r, f, u = judge("runLoop", {"z_": [("assignment", 1, "a")]}, {"z_": [("write: BinaryOperator `z_ = 1` @1:1", 1, "a")]}, {}, tof)
    case("U11", "both write: a write, no row, no finding", "z_" in o and not r and not f)
    # (c)
    want = [T, "src/server/engine_gov.cpp"]
    case("U12", "(c) TU list differs (emut ran one TU less): a finding", len(check_tus(want, _ast(want), _ast([T]))) == 1)
    case("U13", "(c) same TU lists: no finding", not check_tus(want, _ast(want), _ast(list(reversed(want)))))
    h1 = ["server/engine.cpp", 3, "x_", "Engine", "r", "read", False, None]
    h2 = ["server/engine.cpp", 7, "y_", "Engine", "r", "read", False, None]
    a1, a2 = index(_ast([T], [h1, h2])), index(_ast([T], [h1]))
    case("U14", "(c) a use only the Python analyzer reports: a finding", len(check_positions(a1, a2)) == 1)
    case("U15", "(c) a use only clang reports: a finding", len(check_positions(a2, a1)) == 1)
    case("U16", "(c) same uses: no finding", not check_positions(a1, index(_ast([T], [h2, h1]))))
    # holder skip
    hw = ["server/engine.cpp", 3, "dc_", "Engine", "w", "non-const method dflReset()", False, None]
    ix = index(_ast([T], [hw]))
    w0 = body_writes(ix, [("server/engine.cpp", 0, 99)])[0]
    w1, rd1, _ = body_writes(ix, [("server/engine.cpp", 0, 99)], skip=lambda rel, i, m: "holder call dflReset()" if m == "dc_" else None)
    case("U17", "holder call skip: dc_.dflReset( is a read (call edge) only through skip", "dc_" in w0 and "dc_" not in w1 and rd1["dc_"] == ["holder call dflReset()"])
    case("U18", "norm_why drops the position and the weak marker", norm_why("write: CallExpr `f(x_)` @12:3 (weak)") == "write: CallExpr `f(x_)`")
    case("U19", "chain_pointee: x_ = 1 is not a pointee, *rh_.p and q_.front()->id are",
         not chain_pointee(txt["server/engine.cpp"], i_x, "x_") and chain_pointee(txt["server/engine.cpp"], i_rh, "rh_")
         and chain_pointee(txt["server/engine.cpp"], i_q, "q_"))
    good = sum(ok for _, ok in res)
    print(f"engine split emut selftest: {good}/{len(res)} as expected")
    return 0 if good == len(res) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(selftest())
    print(__doc__); sys.exit(2)
