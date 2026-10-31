# SPDX-License-Identifier: Apache-2.0
"""arch025-emut: self-test of emut.exe / emut_run.py (clang's own ExprMutationAnalyzer on the census TUs).

The expected verdicts below were written BEFORE the tool existed, from the analyzer's rules at llvmorg-21.1.8
(clang/lib/Analysis/ExprMutationAnalyzer.cpp, line numbers in progress/arch025-emut.PRIOR_ART.md):

  w  some use of the member is an object write  (ExprMutationAnalyzer::findMutation != null)
  p  no object write, but a pointee write       (findPointeeMutation != null)
  r  every use is a read (both null)
  -  the member must NOT be reported (untracked class, unevaluated only ...)

`py` is what the Python analyzer (tools/gates/engine_split_ast.py, S0g working copy) expects for the same shape; the
cases come from its own selftest (S0d / S0e / S0f review probes) plus the S0f review live shape (std::exchange through
queue_.front()) and analyzer-specific shapes.  When the analyzer is more conservative or narrower than the census rule,
the row says so - those are the analyzer's verdicts, not bugs of this tool.

`u` marks predictions the source does not settle without running it (template bodies of the MSVC STL, matcher arg
handling); after the first run each one is pinned to what clang said, with the reason in `note` (never silently).

Checks besides the verdicts: rows have engine_split_ast.py's 8-field schema, the char index points at the member's
text (or `this`), an accessor returning a member by non-const reference names its function in returned_from, a
member of an untracked class is never reported, call rows exist for that accessor.

Usage: python -I emut_selftest.py [--emut PATH_TO_emut.exe] [--work DIR]
Exit 0 all as expected, 1 some differ, 3 tool missing.
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

HDR = r"""#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <atomic>
#include <bitset>
#include <deque>
#include <format>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
namespace whirl::util {
inline void count(int* p) { *p = 1; }
inline void format(int& x) { x = 1; }
template <class... A> void logI(A&... a) { ((a = 0), ...); }
}
inline int* next(int* p) { *p = 1; return p; }
inline bool find(int* p, int* e) { *p = 1; return p == e; }
namespace other { inline int* prev(int* p) { *p = 2; return p; } }
struct GS { int x = 0; };
struct H { int* p; GS* g; };
inline void wr(int* p) { *p = 1; }
inline void cr(const int& v) { (void)v; }
inline std::size_t* next(std::size_t* p) { *p = 1; return p; }
namespace other { inline std::string* find(std::string* p) { p->clear(); return p; } }
namespace whirl::server { struct Other { int o1_ = 0; void f() { o1_ = 1; } }; }
"""

# (member declaration, method body ("CONST:" = const method; None = special method below), expect, py, u, note)
CASES = [
    ("const char* c1_ = nullptr;", "c1_ = nullptr;", "w", "w", 0, "pointer-to-const member assigned"),
    ("const int* c2_ = nullptr;", "++c2_;", "w", "w", 0, "pointer-to-const member incremented"),
    ("int c3_ = 0;", None, "w", "w", 0, "const_cast<int&>(c3_) = 1 in a const method (findCastMutation)"),
    ("std::future<int> c4_;", "(void)c4_.get();", "w", "w", 0, "future::get non-const"),
    ("int c5_ = 0;", "auto f = [](auto& e) { e.c5_ = 1; }; f(*this);", "w", "w", 1, "generic lambda e.c5_ = 1 (instantiated operator() must be visited)"),
    ("int c7_ = 0;", None, "w", "w", 0, "member template never instantiated: dependent c7_ = v (weak)"),
    ("int c8_ = 0;", "tf(*this);", "w", "w", 0, "function template tf(E& e) { e.c8_ = 1; } instantiation"),
    ("int c9_ = 0;", "SETONE(c9_);", "w", "w", 0, "macro SETONE(x) x = 1"),
    ("std::map<int, int> c18_;", "(void)c18_[1];", "w", "w", 0, "map::operator[] non-const"),
    ("std::map<int, int> c19_;", "int v = c19_[1]; (void)v;", "w", "w", 0, "map::operator[] read still non-const"),
    ("std::vector<bool> c20_;", "c20_[0] = true;", "w", "w", 0, "vector<bool> proxy"),
    ("std::bitset<8> c21_;", "c21_[3] = true;", "w", "w", 0, "bitset proxy"),
    ("std::map<int, int> c22_;", "auto r = c22_.equal_range(1); r.first->second = 5;", "w", "w", 0, "equal_range non-const overload"),
    ("std::vector<char> c23_;", "std::optional<std::span<char>> o{c23_}; (*o)[0] = 'x';", "w", "w", 1, "optional<span> from the member (forwarding ctor followed into the STL)"),
    ("std::vector<int> c24_;", "std::sort(c24_.begin(), c24_.end());", "w", "w", 0, "begin() non-const"),
    ("std::unordered_map<int, int> c25_;", "c25_[2];", "w", "w", 0, "unordered_map::operator[]"),
    ("int c26_ = 0;", "[this] { c26_ = 1; }();", "w", "w", 0, "lambda capturing this"),
    ("std::vector<int> c27_;", "for (auto& x : c27_) x = 0;", "w", "w", 0, "range-for by reference"),
    ("int* c28_ = nullptr;", "*c28_ = 1;", "p", "w", 0, "pointee written through a pointer member"),
    ("std::unique_ptr<int> c29_;", "*c29_ = 1;", "r", "w", 0, "PINNED after run 1 (predicted w): ANALYZER LIMIT with the MSVC STL - isMoveOnly() needs a move constructor, MSVC unique_ptr declares it as a constrained template (memory:3384-3385, 3412-3413), so operator* is not followed"),
    ("std::atomic<int> c30_{0};", "c30_.store(1);", "w", "w", 0, "atomic store"),
    ("std::vector<int> c31_;", "std::fill(c31_.begin(), c31_.end(), 0);", "w", "w", 0, "std::fill (begin non-const)"),
    ("std::vector<int> c32_;", "auto it = std::find(c32_.begin(), c32_.end(), 1); *it = 2;", "w", "w", 0, "begin non-const"),
    ("std::vector<int> c33_;", "auto it = std::max_element(c33_.begin(), c33_.end()); *it = 2;", "w", "w", 0, "begin non-const"),
    ("std::vector<int> c34_;", "std::span<int> s(c34_.data(), c34_.size()); s[0] = 1;", "w", "w", 0, "data() non-const"),
    ("std::string c35_;", "std::string_view v = c35_; (void)v;", "r", "r", 0, "string_view conversion operator is const"),
    ("std::vector<int> c36_;", "(void)c36_.size();", "r", "r", 0, "size() const"),
    ("std::vector<int> c37_;", "(void)std::count(c37_.begin(), c37_.end(), 1);", "w", "r", 0, "ANALYZER MORE CONSERVATIVE: non-const begin() on a non-const object (census: measured std read algorithm)"),
    ("std::map<int, int> c38_;", "auto it = c38_.find(1); if (it != c38_.end()) it->second = 3;", "w", "w", 0, "find non-const"),
    ("std::vector<int> c39_;", "auto p = std::next(c39_.begin()); *p = 1;", "w", "w", 0, "begin non-const"),
    ("std::vector<int> c40_;", "auto* p = &*std::next(c40_.begin()); *p = 1;", "w", "w", 0, "begin non-const"),
    ("std::vector<int> c41_;", "*std::find(c41_.begin(), c41_.end(), 1) = 2;", "w", "w", 0, "begin non-const"),
    ("std::vector<int> c42_;", "if (std::find(c42_.begin(), c42_.end(), 1) != c42_.end()) {}", "w", "r", 0, "ANALYZER MORE CONSERVATIVE: begin() non-const though only compared"),
    ("std::map<int, int> c43_;", "(void)c43_.at(1);", "w", "r", 0, "ANALYZER MORE CONSERVATIVE: non-const at() overload selected"),
    ("std::vector<int> c44_;", "int v = c44_[0]; (void)v;", "w", "r", 0, "ANALYZER MORE CONSERVATIVE: non-const operator[] selected"),
    ("const char* c45_ = nullptr;", "const char* p = c45_; (void)p;", "r", "r", 0, "copy to pointer-to-const"),
    ("int c46_ = 0;", "whirl::util::count(&c46_);", "w", "w", 0, "address taken"),
    ("int c47_ = 0;", "whirl::util::logI(c47_);", "w", "w", 0, "template A& (not forwarding) = write"),
    ("int c48_ = 0;", "whirl::util::format(c48_);", "w", "w", 0, "int& parameter"),
    ("std::string c49_;", "auto s = std::format(\"{}\", c49_); (void)s;", "w", "r", 1, "forwarding into std::format -> make_format_args(Args&...) (STL body decides)"),
    ("int* p1_ = nullptr;", "CONST:*p1_ = 1;", "p", "w", 0, "pointee written in a const method"),
    ("int* const p2_ = nullptr;", "*p2_ = 1;", "p", "w", 0, "T *const member pointee"),
    ("GS* p3_ = nullptr;", "CONST:p3_->x = 1;", "p", "w", 0, "findPointeeMemberMutation"),
    ("int* p3b_ = nullptr;", "*p3b_ = 1;", "p", "w", 0, "pointee write, non-const method"),
    ("int p4_ = 0;", "(void)::next(&p4_);", "w", "w", 0, "address taken"),
    ("int p5_ = 0;", "int e = 0; if (::find(&p5_, &e)) {}", "w", "w", 0, "address taken"),
    ("int p6_ = 0;", "(void)other::prev(&p6_);", "w", "w", 0, "address taken"),
    ("std::span<int> p7_;", "CONST:p7_[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: span::operator[] is const, view constness does not cover the elements"),
    ("int* p17_ = nullptr;", "CONST:int* q = p17_; *q = 1;", "p", "w", 0, "init of a non-const pointer variable (findPointeeToNonConst)"),
    ("int* p19_ = nullptr;", "CONST:int v = *p19_; (void)v;", "r", "r", 0, "pointee read"),
    ("const int* p20_ = nullptr;", "CONST:const int* q = p20_; (void)q;", "r", "r", 0, "copy to const pointee"),
    ("int* q1_ = nullptr;", "CONST:int* q = nullptr; q = q1_; *q = 1;", "p", "w", 0, "assign to a non-const pointer"),
    ("int* q2_ = nullptr;", "CONST:int* q{q2_}; *q = 1;", "p", "w", 0, "list-init of a non-const pointer"),
    ("int* q3_ = nullptr;", "CONST:std::memset(q3_, 0, 4);", "p", "w", 1, "void* parameter: matcher arg vs BitCast (canResolveToExprPointee stops at BitCast)"),
    ("int* q3n_ = nullptr;", "std::memset(q3n_, 0, 4);", "p", "w", 1, "same, non-const method"),
    ("int* q4_ = nullptr;", "CONST:std::fill(q4_, q4_ + 1, 0);", "p", "w", 0, "non-const pointer parameter of an instantiation"),
    ("int* q5_ = nullptr;", "CONST:wr(q5_);", "p", "w", 0, "int* parameter"),
    ("int* q5n_ = nullptr;", "wr(q5n_);", "p", "w", 0, "int* parameter"),
    ("std::span<int> q7_;", "CONST:q7_.subspan(1)[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: subspan const"),
    ("std::span<int> q8_;", "CONST:auto s = q8_; s[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: span copy (const& copy ctor)"),
    ("std::span<int> q8n_;", "auto s = q8n_; s[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: span copy"),
    ("std::vector<int>::iterator q9_;", "CONST:*(q9_ + 1) = 1;", "r", "w", 0, "ANALYZER LIMIT: iterator operator+ / operator* const"),
    ("std::vector<int>::iterator q10_;", "CONST:*std::next(q10_) = 1;", "r", "w", 0, "ANALYZER LIMIT: iterator passed by value"),
    ("std::size_t q11_ = 0;", "(void)::next(&q11_);", "w", "w", 0, "address taken"),
    ("std::string q12_;", "(void)other::find(&q12_);", "w", "w", 0, "address taken"),
    ("std::unique_ptr<GS> q14_;", "CONST:q14_->x = 1;", "r", "w", 0, "PINNED after run 1 (predicted w): ANALYZER LIMIT with the MSVC STL - same isMoveOnly() miss for operator->"),
    ("std::unique_ptr<GS> q14b_;", "CONST:GS* g = q14b_.get(); g->x = 1;", "r", "w", 0, "ANALYZER LIMIT: get() const, the copy is not followed"),
    ("std::shared_ptr<GS> q15_;", "CONST:auto c = q15_; c->x = 1;", "r", "w", 0, "ANALYZER LIMIT: shared_ptr copy"),
    ("int** q16_ = nullptr;", "CONST:**q16_ = 1;", "r", "w", 0, "ANALYZER LIMIT: pointee of pointee (one level only)"),
    ("H q17_{};", "CONST:*q17_.p = 1;", "r", "w", 0, "ANALYZER LIMIT: pointer FIELD's pointee is not the member's pointee"),
    ("H q17b_{};", "CONST:q17b_.g->x = 1;", "r", "w", 0, "ANALYZER LIMIT: pointer field ->x"),
    ("H q17n_{};", "*q17n_.p = 1;", "r", "w", 0, "ANALYZER LIMIT: pointer field pointee (non-const method)"),
    ("int* q18_[2] = {};", "CONST:*q18_[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: array of pointers element pointee"),
    ("std::vector<int*> q19_;", "CONST:*q19_[0] = 1;", "r", "w", 0, "ANALYZER LIMIT: const operator[] then pointee"),
    ("std::vector<int*> q19b_;", "CONST:for (int* p : q19b_) *p = 1;", "r", "w", 0, "ANALYZER LIMIT: by-value loop variable"),
    ("std::vector<int*> q19n_;", "*q19n_[0] = 1;", "w", "w", 0, "non-const operator[]"),
    ("std::map<int, GS*> q20_;", "CONST:q20_.at(0)->x = 1;", "r", "w", 0, "ANALYZER LIMIT: const at() result pointee"),
    ("int* q21_ = nullptr;", "CONST:int* const& r = q21_; *r = 1;", "r", "w", 0, "ANALYZER LIMIT: T* const& binding (fixed upstream by PR #190421, not in 21.1.8)"),
    ("int* q22_ = nullptr;", "CONST:auto* q = q22_; q[0] = 1;", "p", "w", 0, "auto* init"),
    ("int* q23_ = nullptr;", "CONST:int* a = q23_; int* b = a; *b = 1;", "p", "w", 0, "init of a non-const pointer"),
    ("int* q24_ = nullptr;", "CONST:int* q = q24_ + 1; *q = 1;", "p", "w", 0, "additive resolves"),
    ("int* r1_ = nullptr;", "CONST:int v = *r1_; (void)v;", "r", "r", 0, "pointee read"),
    ("H r2_{};", "CONST:int v = *r2_.p; (void)v;", "r", "r", 0, "pointer field read"),
    ("std::vector<int*> r3_;", "CONST:int v = *r3_[0]; (void)v;", "r", "r", 0, "const operator[] read"),
    # analyzer-specific shapes (arch025-emut)
    ("std::deque<GS*> e5_;", "CONST:(void)std::exchange(e5_.front()->x, 0);", "r", "w", 0, "ANALYZER LIMIT: S0f review live shape (queue_.front()->id) - const front() result pointee"),
    ("int e6_ = 0;", "int v = std::move(e6_); (void)v;", "r", "w", 0, "PINNED after run 1 (predicted w): findCastMutation only counts std::move when its RESULT is mutated; an int initialised from it is a read (ExprMutationAnalyzer.cpp:562-568)"),
    ("std::string e6b_;", "std::string s = std::move(e6b_); (void)s;", "w", "w", 0, "added after run 1, predicted before run 2: the move constructor takes string&& (findFunctionArgMutation)"),
    ("int e7_ = 0;", "[&] { e7_ = 1; }();", "w", "w", 0, "[&] lambda"),
    ("int e8_ = 0;", "auto& r = e8_; r = 1;", "w", "w", 0, "non-const reference variable followed"),
    ("int e9_ = 0;", "const auto& r = e9_; (void)r;", "r", "r", 0, "const reference"),
    ("int e10_ = 0;", "cr(e10_);", "r", "r", 0, "const& parameter"),
    ("int e11_ = 0;", "decltype(e11_) z = 0; (void)z; (void)sizeof(e11_ = 1);", "r", "r", 0, "unevaluated only (decltype / sizeof)"),
    ("int e12_ = 0;", None, "w", "w", 0, "accessor int& a12() { return e12_; } -> ReturnStmt, returned_from = a12"),
    ("int e13_ = 0;", None, "w", "r", 0, "PINNED after run 1 (predicted r): ANALYZER FALSE POSITIVE - AsNonConstRefReturn assumes a const-ref return shows a NoOp cast (ExprMutationAnalyzer.cpp:483-489); a member of a const method is already const, so `return e13_` in a const accessor counts as a write"),
    ("int e14_ = 0;", "this->e14_ += 2;", "w", "w", 0, "explicit this-> (position at `this`)"),
    ("int e15_ = 0;", "CONST:int v = e15_; (void)v;", "r", "r", 0, "plain read"),
    ("int* e16_ = nullptr;", "CONST:const int* q = e16_; (void)q;", "r", "r", 0, "copy into const pointee"),
    ("int e17_ = 0;", "int x = e17_; x = 2; (void)x;", "r", "r", 0, "copy is not the member"),
    ("H e18_{};", "auto& [hp, hg] = e18_; hp = nullptr; (void)hg;", "w", "w", 0, "structured binding by reference, binding assigned (S0 review W3 shape; added after the cross-check probe scratch/a25_emut/probe_w3.py showed w)"),
    ("H e19_{};", "CONST:const auto& [hp, hg] = e19_; (void)hp; (void)hg;", "r", "r", 0, "const structured binding, read"),
]


def build_tu():
    src = [HDR, "namespace whirl::server {", "#define SETONE(x) x = 1", "template <class E> void tf(E& e) { e.c8_ = 1; }",
           "struct Engine {"]
    src += ["  " + d for d, *_ in CASES]
    src.append("  void m3() const { const_cast<int&>(c3_) = 1; }")
    src.append("  template <class T> void m7(T v) { c7_ = v; }")
    src.append("  int& a12() { return e12_; }")
    src.append("  const int& a13() const { return e13_; }")
    src.append("  void use12() { a12() = 5; }")
    for i, (_d, b, *_r) in enumerate(CASES):
        if b is None: continue
        src.append(f"  void m{i}x() const {{ {b[6:]} }}" if b.startswith("CONST:") else f"  void m{i}x() {{ {b} }}")
    src += ["};", "}"]
    return "\n".join(src) + "\n"


def member_of(decl):
    return decl.split("=")[0].split("{")[0].split("[")[0].replace(";", "").split()[-1].lstrip("*")


def main():
    a = sys.argv[1:]
    emut = a[a.index("--emut") + 1] if "--emut" in a else os.environ.get("WHIRL_EMUT")
    work = a[a.index("--work") + 1] if "--work" in a else tempfile.mkdtemp(prefix="emut_selftest_")
    if not emut or not os.path.exists(emut):
        print(f"emut selftest: NO TOOL ({emut}; build tools/gates/emut first or set WHIRL_EMUT)"); return 3
    root = os.path.join(work, "root")
    os.makedirs(os.path.join(root, "src", "server"), exist_ok=True)
    text = build_tu()
    p = os.path.join(root, "src", "server", "p.cpp")
    open(p, "w", encoding="utf-8", newline="\n").write(text)
    out = os.path.join(work, "emut.json")
    cmd = [sys.executable, "-I", os.path.join(HERE, "emut_run.py"), "--emut", emut, "--root", root, "--out", out,
           "--cache", os.path.join(work, "cache"), "--classes", "Engine", "--tu", "src/server/p.cpp"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print("emut selftest: emut_run.py failed rc", r.returncode, (r.stdout + r.stderr)[-2000:]); return 1
    res = json.load(open(out, encoding="utf-8"))
    tu = res["tus"][0]
    if tu["errors"] or tu["rc"]:
        print("emut selftest: synthetic TU does not compile:", tu["errors"][:3]); return 1
    bad = []
    by = {}
    for row in tu["hits"]:
        if len(row) != 8: bad.append(f"row with {len(row)} fields (schema is 8): {row}"); continue
        rel, idx, mem, cls, v, why, weak, rf = row
        by.setdefault(mem, []).append(row)
        if rel != "server/p.cpp": bad.append(f"rel {rel}")
        frag = text[idx:idx + 8]
        if not (frag.startswith(mem) or frag.startswith("this") or frag.startswith("SETONE") or mem == "c8_" or mem == "c5_"):
            bad.append(f"{mem}: index {idx} points at {frag!r}")
        if v not in ("w", "r"): bad.append(f"{mem}: verdict {v}")
    good = upred = 0
    lines = []
    for i, (d, _b, exp, py, u, note) in enumerate(CASES):
        m = member_of(d)
        hs = by.get(m, [])
        strong = [h for h in hs if not h[6]] or hs
        if not hs: got = "-"
        elif any(h[4] == "w" and h[5].startswith("write") for h in strong): got = "w"
        elif any(h[4] == "w" and h[5].startswith("pointee-write") for h in strong): got = "p"
        elif any(h[4] == "w" for h in strong): got = "?"
        else: got = "r"
        ok = got == exp
        good += ok
        if not ok and u: upred += 1
        why = next((h[5] for h in strong if (h[4] == "w") == (got in "wp")), "-")
        tag = "ok  " if ok else ("PRED" if u else "FAIL")
        lines.append(f"  {tag} {m:6} expect {exp} got {got} (py {py}){' u' if u else ''}: {note}  [{why[:90]}]")
    for l in lines: print(l)
    # structural checks
    if "o1_" in by: bad.append("o1_ of the untracked class Other is reported")
    rf12 = [h[7] for h in by.get("e12_", []) if h[4] == "w"]
    if not rf12 or not all(rf12) or "a12" not in (rf12[0] or ""): bad.append(f"e12_ returned_from {rf12} (expected the mangled a12)")
    if not any("a12" in (c[2] or "") for c in tu.get("calls", [])): bad.append("no call row for a12()")
    else:
        c = [c for c in tu["calls"] if "a12" in (c[2] or "")]
        if not any(x[3] == "w" for x in c): bad.append(f"call a12() = 5 not a write: {c}")
    for b in bad: print("  BAD", b)
    print(f"emut selftest: {good}/{len(CASES)} as expected ({upred} of the misses are unsettled predictions 'u'), "
          f"{len(bad)} structural problems")
    return 0 if good == len(CASES) and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
