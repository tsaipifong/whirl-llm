"""Block / pass test for impact_build's root check (arch024 A9). CPU only, a few seconds per case.

    python tools/impact/impact_build_test.py [--root REPO]

Cases (each runs impact_build.main() in-process on the real tree):
  pass        : the declared roots as committed -> exit 0, map written, 0 errors
  block_path  : one PATHS root renamed to a function that does not exist -> exit 1, no map written
  block_gate  : one CLI_GATES root renamed likewise -> exit 1, no map written
  block_check : --check with a missing root -> exit 1 (a missing root is not "map is current")
  block_why   : --why with a missing root -> exit 1
  ns_*        : (toolfix) same short name in two namespaces on a synthetic tree: call edges stay name-based (every call
                of `f`, qualified or not, keeps both functions: over-select is the safe side), while a PATHS root
                `ns::f` resolves to one function and the short root `f` still resolves to both
Prints "impact_build_test: N/N as expected" and exits 0 only when every case behaves as listed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impact_build  # noqa: E402

MISSING = "Engine::noSuchFunctionArch024A9"


def run(root, argv, mutate=None):
    """impact_build.main() with argv; mutate(PATHS, CLI_GATES) edits copies of the declarations. -> (rc, stderr)."""
    paths0, gates0 = impact_build.PATHS, impact_build.CLI_GATES
    impact_build.PATHS, impact_build.CLI_GATES = copy.deepcopy(paths0), copy.deepcopy(gates0)
    if mutate:
        mutate(impact_build.PATHS, impact_build.CLI_GATES)
    old_argv = sys.argv
    sys.argv = ["impact_build.py", "--root", root] + argv
    err, out = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = impact_build.main()
    finally:
        sys.argv = old_argv
        impact_build.PATHS, impact_build.CLI_GATES = paths0, gates0
    return rc, err.getvalue() + out.getvalue()


def ns_cases():
    """-> [(name, ok)]: same-named functions in different namespaces are distinct nodes (toolfix)."""
    nl = chr(10)
    src = {"src/kernels_host/kernels_abi.cpp": "", "kernels/whirl_kernels.hip": "",
           "src/a.cpp": nl.join(["namespace w { namespace one {", "int draw(int x) { return x; }", "} }", ""]),
           "src/b.cpp": nl.join(["namespace w { namespace two {", "int draw(int x) { return x + 1; }", "} }", ""]),
           "tests/t.cpp": nl.join(["namespace sp = w::one;", "void testAlias() { sp::draw(1); }",
                                   "void testFull() { w::two::draw(1); }", "void testBare() { draw(1); }",
                                   "void testBoth() { one::draw(1); draw(2); }", "void testStd() { std::draw(1); }", ""])}
    A, B = "src/a.cpp::draw", "src/b.cpp::draw"
    with tempfile.TemporaryDirectory() as d:
        for p, text in src.items():
            os.makedirs(os.path.dirname(os.path.join(d, p)), exist_ok=True)
            open(os.path.join(d, p), "w", encoding="utf-8", newline=chr(10)).write(text)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            try:
                impact_build.build(d)
            except Exception as e:  # the synthetic tree has no CMake / paths: only the call graph is judged
                err.write(repr(e))
        e = impact_build.DEBUG["edges"]
        g = lambda f: sorted(e.get("tests/t.cpp::" + f, ()))
        rs = impact_build.DEBUG["resolve"]
    return [("ns_alias: sp::draw keeps both (no call-edge narrowing)", g("testAlias") == [A, B]),
            ("ns_full: w::two::draw keeps both", g("testFull") == [A, B]),
            ("ns_bare: draw keeps both", g("testBare") == [A, B]),
            ("ns_mixed: one::draw + bare draw keeps both", g("testBoth") == [A, B]),
            ("ns_foreign: std::draw keeps both", g("testStd") == [A, B]),
            ("ns_root: one::draw is one node, draw is two", rs("one::draw") == [A] and sorted(rs("draw")) == [A, B])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.normpath(os.path.join(HERE, "..", "..")))
    a = ap.parse_args()

    def bad_path(p, g):
        p["server.loop"]["roots"] = p["server.loop"]["roots"] + [MISSING]

    def bad_gate(p, g):
        k = sorted(g)[0]
        g[k] = list(g[k]) + ["noSuchCliFunctionArch024A9"]

    results = []
    with tempfile.TemporaryDirectory() as td:
        def case(name, argv, mutate, want_rc, want_file, want_text):
            out = os.path.join(td, name + ".json")
            rc, text = run(a.root, ["--out", out] + argv, mutate)
            ok = rc == want_rc and os.path.exists(out) == want_file and (want_text is None or want_text in text)
            results.append(ok)
            print(f"{'ok  ' if ok else 'FAIL'} {name}: rc={rc} (want {want_rc}), map written={os.path.exists(out)} "
                  f"(want {want_file})" + ("" if ok else "\n" + text))

        case("pass", [], None, 0, True, "0 error(s)")
        case("block_path", [], bad_path, 1, False, f"root '{MISSING}' not found")
        case("block_gate", [], bad_gate, 1, False, "root 'noSuchCliFunctionArch024A9' not found")
        case("block_check", ["--check"], bad_path, 1, False, "map not written")
        case("block_why", ["--why", "Engine::runLoop", "Engine::decodeCycle"], bad_path, 1, False, "map not written")
    n = sum(results)
    print(f"impact_build_test: {n}/{len(results)} as expected")
    ns = ns_cases()
    for name, ok in ns:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    k = sum(ok for _, ok in ns)
    print(f"impact_build_test ns: {k}/{len(ns)} ok")
    return 0 if n == len(results) and k == len(ns) else 1


if __name__ == "__main__":
    sys.exit(main())
