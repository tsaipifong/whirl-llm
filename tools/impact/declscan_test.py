# SPDX-License-Identifier: Apache-2.0
"""Block / pass tests of declscan.py, the declared-name parser of a changed declaration line (measure-tools P1, B1.1-1.2).
CPU, < 1 s.

    python -I tools/impact/declscan_test.py        exit 0 = "declscan test: N/N ok"

Each case names a source snippet, the 1-based line that changed, and the declared names the parser must return
(and, for the block side, identifiers it must NOT return: type names, casts, keywords).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import declscan  # noqa: E402

ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


def names(src, line):
    return {(d["cls"], d["name"]) for d in declscan.declared(src, line)["decls"]}


def kind(src, line):
    return declscan.declared(src, line)["kind"]


MODEL = """namespace whirl {
struct Model {
    std::vector<device::DevBuf> allocations;
    hip::EventH events[4];
    void alloc(size_t n,
               hip::Stream s);
    ~Model();
    size_t n_ = 0, m_ = 1;
    std::vector<int> v{1, 2};
    void (*cb)(int);
    static constexpr int kX = static_cast<int>(3);
    Model(const Model&) = delete;
    explicit operator bool() const;
    template <class T> T get(int i) const;
  public:
    int   x : 3;
};
}
"""
r = names(MODEL, 3)
expect("member: vector<DevBuf> allocations -> Model::allocations", r == {("Model", "allocations")}, r)
d = declscan.declared(MODEL, 3)["decls"][0]
expect("member: type names recorded as type_changed, not as names (DevBuf, device)", "DevBuf" in d["types"] and "device" not in d["types"]
       and "vector" not in d["types"], d["types"])
expect("member array: events[4] -> Model::events", names(MODEL, 4) == {("Model", "events")}, names(MODEL, 4))
expect("member function, changed continuation line -> Model::alloc", names(MODEL, 6) == {("Model", "alloc")}, names(MODEL, 6))
expect("member function first line -> Model::alloc (no Stream / size_t)", names(MODEL, 5) == {("Model", "alloc")}, names(MODEL, 5))
expect("destructor -> Model::~Model", names(MODEL, 7) == {("Model", "~Model")}, names(MODEL, 7))
expect("two declarators -> n_, m_", names(MODEL, 8) == {("Model", "n_"), ("Model", "m_")}, names(MODEL, 8))
expect("brace init -> v", names(MODEL, 9) == {("Model", "v")}, names(MODEL, 9))
expect("function pointer -> cb", names(MODEL, 10) == {("Model", "cb")}, names(MODEL, 10))
expect("static_cast is not a name (block) -> kX", names(MODEL, 11) == {("Model", "kX")}, names(MODEL, 11))
expect("deleted copy ctor -> Model::Model", names(MODEL, 12) == {("Model", "Model")}, names(MODEL, 12))
expect("conversion operator -> operator", any(n.startswith("operator") for _, n in names(MODEL, 13)), names(MODEL, 13))
expect("template member -> get", names(MODEL, 14) == {("Model", "get")}, names(MODEL, 14))
expect("access specifier line is punctuation", kind(MODEL, 15) == "punct", kind(MODEL, 15))
expect("bit field -> x", names(MODEL, 16) == {("Model", "x")}, names(MODEL, 16))
expect("closing brace is punctuation", kind(MODEL, 17) == "punct", kind(MODEL, 17))
expect("struct opener -> type Model", names(MODEL, 2) == {(None, "Model")}, names(MODEL, 2))

FREE = """#include "whirl/model.h"
#define WHIRL_X 3
#define WHIRL_F(a) \\
    ((a) + 1)
#if defined(__gfx1151__)
using Clock = std::chrono::steady_clock;
#endif
namespace whirl::server {
int foo(const Bar& b);
WHIRL_API void bar();
enum class Mode : uint8_t {
    kA, kB = 2,
    kC
};
static const Row kRows[] = {
    {"a", 1},
    {"b", 2},
};
typedef int (*Fn)(int);
using whirl::other::baz;
struct Fwd;
int a = 1, b = 2;
}
"""
dd = declscan.declared(FREE, 1)
expect("#include -> include kind with its target", dd["kind"] == "include" and dd["target"] == "whirl/model.h", dd)
expect("#define -> macro WHIRL_X", kind(FREE, 2) == "macro" and names(FREE, 2) == {(None, "WHIRL_X")}, declscan.declared(FREE, 2))
expect("macro continuation line -> macro WHIRL_F", kind(FREE, 4) == "macro" and names(FREE, 4) == {(None, "WHIRL_F")}, declscan.declared(FREE, 4))
expect("#if -> cond (file scope)", kind(FREE, 5) == "cond", kind(FREE, 5))
expect("using alias -> Clock (not steady_clock / chrono)", names(FREE, 6) == {(None, "Clock")}, names(FREE, 6))
expect("free function -> foo (not Bar)", names(FREE, 9) == {(None, "foo")}, names(FREE, 9))
expect("attribute macro before a function -> bar", names(FREE, 10) == {(None, "bar")}, names(FREE, 10))
expect("enum opener -> type Mode (not uint8_t)", names(FREE, 11) == {(None, "Mode")}, names(FREE, 11))
r = names(FREE, 12)
expect("enumerators -> kA, kB in Mode", r == {("Mode", "kA"), ("Mode", "kB")}, r)
expect("enumerator line kind", kind(FREE, 12) == "enumerator", kind(FREE, 12))
expect("initializer row -> the table kRows (not Row)", names(FREE, 16) == {(None, "kRows")}, names(FREE, 16))
expect("typedef function pointer -> Fn", names(FREE, 19) == {(None, "Fn")}, names(FREE, 19))
expect("using-declaration -> baz", names(FREE, 20) == {(None, "baz")}, names(FREE, 20))
expect("forward declaration -> type Fwd", names(FREE, 21) == {(None, "Fwd")}, names(FREE, 21))
expect("two variables -> a, b", names(FREE, 22) == {(None, "a"), (None, "b")}, names(FREE, 22))
expect("namespace opener is punctuation", kind(FREE, 8) == "punct", kind(FREE, 8))

ONE = "struct P { int a; float b; };\n"
expect("one-line struct -> P, a, b", names(ONE, 1) == {(None, "P"), ("P", "a"), ("P", "b")}, names(ONE, 1))
CMT = "struct Q {\n    int a;  // was: DevBuf a\n};\n"
expect("comment text is not parsed (block DevBuf)", names(CMT, 2) == {("Q", "a")}, names(CMT, 2))
NEST = "struct Outer {\n  struct Inner {\n    int deep;\n  };\n  Inner in;\n};\n"
expect("nested struct member -> Inner::deep", names(NEST, 3) == {("Inner", "deep")}, names(NEST, 3))
expect("member of a nested type -> Outer::in", names(NEST, 5) == {("Outer", "in")}, names(NEST, 5))
# mt-p1 review A3: callback members (template arguments with a function type, decltype, operator==)
CB = """struct S {
    std::function<void(int)> on_done;
    std::function<bool(const Job&)> filter = nullptr;
    std::map<int, std::function<int(int)>> handlers;
    std::unique_ptr<FILE, int (*)(FILE*)> fh{nullptr, &fclose};
    decltype(&hipFree) free_fn = &hipFree;
    bool operator==(const Key&) const = default;
    std::function<void(std::uint64_t, int)> on_page;
    decltype(sizeof(int)) width = 4;
    void (*cb)(int) = nullptr;
    auto step() -> int;
};
"""
for ln, want in ((2, "on_done"), (3, "filter"), (4, "handlers"), (5, "fh"), (6, "free_fn"), (7, "operator=="), (8, "on_page"),
                 (9, "width"), (10, "cb"), (11, "step")):
    got = names(CB, ln)
    expect(f"A3 callback / decltype member line {ln} -> S::{want}", got == {("S", want)}, got)
d6 = declscan.declared(CB, 6)["decls"]
expect("A3 decltype(&hipFree): hipFree is a type-side identifier, not the name", d6 and "hipFree" in d6[0]["types"], d6)
d2 = declscan.declared(CB, 2)["decls"]
expect("A3 std::function<void(int)> on_done is a member, not a function", d2 and d2[0]["what"] == "member", d2)
# mt-p1 pass 3 (rereview act-on 1): the signature carries the full scope and the #if branch; include guards are left out
G = "#ifndef X_H\n#define X_H\nnamespace a {\n#if defined(G1)\nconstexpr int k = 1;\n#else\nconstexpr int k = 1;\n#endif\n}\n#endif\n"
s5, s7 = declscan.declared(G, 5)["decls"][0]["sig"], declscan.declared(G, 7)["decls"][0]["sig"]
expect("pass3 #if / #else branches give different signatures", s5 != s7, (s5, s7))
expect("pass3 include guard is not in the #if stack", s5[1] == (("#if defined(G1)", "#if"),), s5)
expect("pass3 namespace in the scope", s5[0] == ("ns:namespace a",), s5)
SP = "template <> struct Cfg<1151> {\n    static constexpr int kTile = 32;\n};\ntemplate <> struct Cfg<1201> {\n    static constexpr int kTile = 32;\n};\n"
expect("pass3 specialisation arguments in the scope", declscan.declared(SP, 2)["decls"][0]["sig"] != declscan.declared(SP, 5)["decls"][0]["sig"])
DI = "struct P {\n    int a;\n    int b;\n};\nconstexpr P k_p = {\n    .a = 1, .b = 2,\n};\n"
got = {(x["cls"], x["name"], x["what"]) for x in declscan.declared(DI, 6)["decls"]}
expect("pass3 designated initializer row -> fields P::a, P::b + the table", {("P", "a", "field"), ("P", "b", "field")} <= got
       and any(n == "k_p" for _, n, _ in got), got)
# mt-p1 pass 4 (REVIEW3 C-1): the designator in force on a continuation line
DC = ("struct P {\n    int a;\n    int r[4];\n    int b;\n};\nconstexpr P k_p = {\n    .a =\n        128,\n    .r = {1, 1,\n          2, 2},\n"
      "    .b = 1,\n};\n")
for ln, want, not_ in ((8, "a", "r"), (10, "r", "a"), (11, "b", "r")):
    got = {x["name"] for x in declscan.declared(DC, ln)["decls"] if x["what"] == "field"}
    expect(f"pass4 continuation line {ln} -> field {want} only", want in got and not_ not in got, got)
print(f"declscan test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
