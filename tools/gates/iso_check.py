#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ARCH-1 8.5 isolation check of src/sampling (arch025-X3), MSVC, CPU only, no GPU, no model.

    python -I tools/gates/iso_check.py [--mutants] [--work DIR] [--root REPO] [--vcvars BAT] [--keep]

Builds what cmake/sampling_iso.cmake builds, by hand: the files of tests/sampling/isolation.txt are copied into
<work>/<case>/{mod,test} (leading src/ include/ tests/ dropped); the module sources are compiled with mod/ as the
ONLY include path, the test with mod/ + test/, and whirl-sampling-tests is linked from those objects alone.
  control              unchanged copy: compiles, links, the test program prints "N passed, 0 failed"     (must PASS)
--mutants adds four boundary mutants; each must FAIL the isolated build at the named stage AND compile with the
product flags (/I src /I include, as whirl_sampling in cmake/server.cmake), which is what makes the check worth having:
  include_engine       sampler.cpp gains #include "server/engine.h"                 -> isolated compile C1083
  include_relative     sampler.cpp gains #include "../server/engine.h"              -> isolated compile C1083
  call_whirl           filterCands calls whirl::nowSeconds() (src/common, lib whirl) -> isolated link LNK2019
  call_server          filterCands calls whirl::server::Log::line (whirl_server_core) -> isolated link LNK2019
The product compile of a mutant is done on a temporary file next to the real one (src/sampling/_iso_mut_<case>.cpp, so
the relative include resolves as in the tree) and removed right after. A link-stage mutant resolves in the product
build because whirl_sampling links whirl PUBLIC and whirl_server_core links it (cmake/server.cmake); the product link is
not redone here (it needs the full HIP build).
Last line "iso check: ok (sampling, N/N cases as expected)" and exit 0; else "iso check: FAIL ..." and exit 1;
exit 2 = setup (no vcvars, bad manifest).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# the compile options of the top-level CMakeLists.txt (MSVC branch) + Release
CL_FLAGS = ["/nologo", "/std:c++20", "/W4", "/permissive-", "/utf-8", "/EHsc", "/Zc:__cplusplus", "/O2", "/DNDEBUG",
            "/D_CRT_SECURE_NO_WARNINGS", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
FILTER_HEAD = "bool filterCands(Sampler& s, std::size_t count, bool complete) {"


def read_manifest(root):
    path = os.path.join(root, "tests", "sampling", "isolation.txt")
    items = []
    exe = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"([a-z]+)\s+([^\s#]+)", line)
        if not m:
            continue
        kind, p = m.group(1), m.group(2)
        if kind == "exe":
            exe = p
            continue
        if kind not in ("src", "hdr", "testsrc", "testhdr"):
            raise SystemExit(f"iso check: FAIL (setup: {path}: unknown kind {kind})")
        if not os.path.isfile(os.path.join(root, p)):
            raise SystemExit(f"iso check: FAIL (setup: {path}: no such file {p})")
        items.append((kind, p))
    if exe != "whirl-sampling-tests" or not any(k == "src" for k, _ in items) or not any(k == "testsrc" for k, _ in items):
        raise SystemExit(f"iso check: FAIL (setup: {path}: needs exe whirl-sampling-tests, src and testsrc lines)")
    return items


def vc_env(bat):
    if not os.path.isfile(bat):
        print(f"iso check: FAIL (setup: no {bat})")
        sys.exit(2)
    out = subprocess.run(f'cmd /c ""{bat}" >nul 2>nul && set"', capture_output=True, text=True, shell=False).stdout
    env = {}
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep:
            env[k] = v
    env["VSLANG"] = "1033"  # English cl / link messages: the C1083 / LNK2019 text is matched (arch025-X3fix)
    if "INCLUDE" not in env or not shutil.which("cl", path=env.get("PATH", "")):
        print("iso check: FAIL (setup: vcvars64 gave no cl / INCLUDE)")
        sys.exit(2)
    return env


def run(cmd, env, cwd):
    # CreateProcess looks a bare name up in THIS process's PATH, not in env= (CPython subprocess docs, Windows), so
    # cl / lib / link from the vcvars env are resolved to full paths first (arch025-X3fix)
    if not os.path.isabs(cmd[0]):
        full = shutil.which(cmd[0], path=env.get("PATH") or env.get("Path", ""))
        if not full:
            raise SystemExit(f"iso check: FAIL (setup: {cmd[0]} not on the vcvars PATH)")
        cmd = [full, *cmd[1:]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def stage(root, items, d):
    shutil.rmtree(d, ignore_errors=True)
    out = {"src": [], "testsrc": []}
    for kind, p in items:
        rel = re.sub(r"^(src|include|tests)/", "", p)
        dst = os.path.join(d, "test" if kind.startswith("test") else "mod", *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(root, p), dst)
        if kind in out:
            out[kind].append(dst)
    return out


def mutate(text, case):
    if case in ("include_engine", "include_relative"):
        inc = '#include "server/engine.h"' if case == "include_engine" else '#include "../server/engine.h"'
        anchor = '#include "sampling/sampler.h"\n'
        assert anchor in text
        return text.replace(anchor, anchor + inc + "  // iso mutant\n", 1)
    assert text.count(FILTER_HEAD) == 1
    if case == "call_whirl":
        decl = "namespace whirl { double nowSeconds(); }  // iso mutant: src/common/util.cpp (lib whirl)\n"
        call = "\n    if (count == 0xFFFFFFFFull) (void)whirl::nowSeconds();  // iso mutant"
    else:
        decl = ("#include <string_view>\nnamespace whirl::server { class Log { public: static void line(char level, std::string_view msg); }; }"
                "  // iso mutant: src/server/log.h (whirl_server_core)\n")
        call = '\n    if (count == 0xFFFFFFFFull) whirl::server::Log::line(\'I\', "iso");  // iso mutant'
    text = text.replace("namespace whirl::sampling {", decl + "namespace whirl::sampling {", 1)
    return text.replace(FILTER_HEAD, FILTER_HEAD + call, 1)


def build_iso(d, files, env):
    """-> (stage, rc, output) of the first failing stage, or ('run', rc, output) of the test program."""
    mod, test = os.path.join(d, "mod"), os.path.join(d, "test")
    objs = []
    for i, s in enumerate(files["src"]):
        o = os.path.join(d, f"m{i}.obj")
        rc, out = run(["cl", *CL_FLAGS, "/c", f"/I{mod}", s, f"/Fo{o}"], env, d)
        if rc:
            return "compile", rc, out
        objs.append(o)
    lib = os.path.join(d, "whirl_sampling_iso.lib")
    rc, out = run(["lib", "/nologo", f"/OUT:{lib}", *objs], env, d)
    if rc:
        return "lib", rc, out
    tobjs = []
    for i, s in enumerate(files["testsrc"]):
        o = os.path.join(d, f"t{i}.obj")
        rc, out = run(["cl", *CL_FLAGS, "/c", f"/I{mod}", f"/I{test}", s, f"/Fo{o}"], env, d)
        if rc:
            return "test-compile", rc, out
        tobjs.append(o)
    exe = os.path.join(d, "whirl-sampling-tests.exe")
    rc, out = run(["link", "/nologo", f"/OUT:{exe}", *tobjs, lib], env, d)
    if rc:
        return "link", rc, out
    rc, out = run([exe], env, d)
    return "run", rc, out


def product_compile(root, rel_src, text, case, env, d):
    tmp = os.path.join(root, os.path.dirname(rel_src), f"_iso_mut_{case}.cpp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        rc, out = run(["cl", *CL_FLAGS, "/c", f"/I{os.path.join(root, 'src')}", f"/I{os.path.join(root, 'include')}", tmp,
                       f"/Fo{os.path.join(d, 'prod.obj')}"], env, d)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return rc, out


def tail(out, n=6):
    return "\n".join("    " + l for l in out.strip().splitlines()[-n:])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "whirl_iso_check"))
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--vcvars", default=VCVARS)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    # python -I ignores PYTHONIOENCODING; a cp950 console cannot print U+FFFD from a decoded cl line (arch025-X3fix)
    sys.stdout.reconfigure(errors="replace")
    items = read_manifest(a.root)
    env = vc_env(a.vcvars)
    os.makedirs(a.work, exist_ok=True)
    cases = [("control", None, "run")]
    if a.mutants:
        cases += [("include_engine", "compile", "C1083"), ("include_relative", "compile", "C1083"),
                  ("call_whirl", "link", "LNK2019"), ("call_server", "link", "LNK2019")]
    target_rel = "src/sampling/sampler.cpp"
    good = 0
    for name, want_stage, want_code in cases:
        d = os.path.join(a.work, name)
        files = stage(a.root, items, d)
        if name == "control":
            st, rc, out = build_iso(d, files, env)
            ok = st == "run" and rc == 0 and re.search(r"\b[1-9]\d* passed, 0 failed", out) is not None
            summ = re.findall(r"^sampling tests: .*$", out, re.M)
            print(f"{'ok  ' if ok else 'FAIL'} control: isolated build + run -> {st} rc={rc} {summ[-1] if summ else ''}")
            if not ok:
                print(tail(out, 12))
        else:
            staged = os.path.join(d, "mod", "sampling", "sampler.cpp")
            src_text = open(os.path.join(a.root, target_rel), encoding="utf-8").read()
            mut = mutate(src_text, name)
            with open(staged, "w", encoding="utf-8", newline="\n") as f:
                f.write(mut)
            st, rc, out = build_iso(d, files, env)
            caught = st == want_stage and rc != 0 and want_code in out
            prc, pout = product_compile(a.root, target_rel, mut, name, env, d)
            ok = caught and prc == 0
            msg = [l.strip() for l in out.splitlines() if want_code in l][:1]
            print(f"{'ok  ' if ok else 'FAIL'} {name}: isolated -> {st} rc={rc} ({want_code} {'seen' if want_code in out else 'MISSING'}); "
                  f"product flags compile rc={prc} (must be 0)")
            if msg:
                print("    " + msg[0])
            if not ok:
                print(tail(out))
                if prc:
                    print(tail(pout))
        good += ok
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)
    n = len(cases)
    if good == n:
        print(f"iso check: ok (sampling, {good}/{n} cases as expected)")
        return 0
    print(f"iso check: FAIL (sampling, {good}/{n} cases as expected)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
