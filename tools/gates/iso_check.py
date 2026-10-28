#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ARCH-1 8.5 isolation check of every isolated module (arch025-X3 for src/sampling, arch025-isoall for the rest), MSVC,
CPU only, no GPU, no model.

    python -I tools/gates/iso_check.py [--module NAME[,NAME...]] [--mutants] [--list] [-j N] [--work DIR] [--root REPO]
                                       [--vcvars BAT] [--keep]

Manifests: tests/sampling/isolation.txt and tests/iso/<module>/isolation.txt (one per module; cmake/iso.cmake reads the
same files). Kinds: module NAME, exe TARGET, lib TARGET (optional object-library name), mutate PATH (the source the
mutants edit), mutsym hip|log|now|engine|model (link mutants), syslib X.lib, then src / hdr (module + declared dependencies)
and testsrc / testhdr (test program only). Every file is copied into <work>/<module>/<case>/{mod,test} (leading src/
include/ tests/ dropped); module sources are compiled with mod/ as the ONLY include path, the test with mod/ + test/, and
the test program is linked from ALL those objects directly (no static-library member selection) with /OPT:NOREF, so an
undeclared symbol in any module object is LNK2019 even when no test calls it.
  control              unchanged copy: compiles, links, the program prints "N passed, 0 failed"           (must PASS)
--mutants adds, per module, boundary mutants on the `mutate` source; each must FAIL the isolated build at the named stage
AND compile with the product flags (/I src /I include), which is what makes the check worth having:
  include_engine       gains #include "server/engine.h"                                -> isolated compile C1083
  include_relative     gains #include "<relative path>/server/engine.h"                -> isolated compile C1083
  call_<sym>           a function appended at the end calls a symbol outside the tree  -> isolated link LNK2019
       hip    whirl::hip::deviceCount       (src/hip, whirl_hip: the device runtime)
       log    whirl::server::Log::line      (src/server/log.cpp, whirl_server_core)
       now    whirl::nowSeconds             (src/common/util.cpp, lib whirl)
       engine whirl::server::serveMain      (src/server/server_main.cpp, whirl_server_core: server startup)
       model  whirl::qwen35::loadOrTune     (src/model/tune.cpp, whirl_model: the model core, class Model)
The product compile of a mutant is done on a temporary file next to the real one (_iso_mut_<module>_<case>.cpp, so the relative
include resolves as in the tree) and removed right after. The product link is not redone here (it needs the HIP build).
Per module the last line is "iso check: ok (<module>, N/N cases as expected)" or "iso check: FAIL (<module>, ...)"; with
more than one module a final "iso check: ok (all: M modules, N/N cases as expected)". Exit 0 ok, 1 FAIL, 2 setup.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
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
KINDS = ("src", "hdr", "testsrc", "testhdr")
SYMS = {
    "hip": ("namespace whirl::hip { int deviceCount(); }  // iso mutant: src/hip (whirl_hip)\n",
            "(void)whirl::hip::deviceCount();"),
    "log": ("#include <string_view>\nnamespace whirl::server { class Log { public: static void line(char level, std::string_view msg); }; }"
            "  // iso mutant: src/server/log.h (whirl_server_core)\n", "whirl::server::Log::line('I', \"iso\");"),
    "now": ("namespace whirl { double nowSeconds(); }  // iso mutant: src/common/util.cpp (lib whirl)\n",
            "(void)whirl::nowSeconds();"),
    "engine": ("namespace whirl::server { int serveMain(int argc, char** argv, const char* program); }"
               "  // iso mutant: src/server/server_main.cpp (whirl_server_core)\n",
               "(void)whirl::server::serveMain(0, nullptr, \"iso\");"),
    # arch025-lane2 leg 3: the model core (class Model's load / tune), for the modules below it (forward.*)
    "model": ("#include <string>\nnamespace whirl::qwen35 { class Model; void loadOrTune(Model& m, const std::string& model_path, "
              "std::string& log); }  // iso mutant: src/model/tune.cpp (whirl_model, model.core)\n",
              "std::string s; whirl::qwen35::loadOrTune(*reinterpret_cast<whirl::qwen35::Model*>(&s), s, s);"),
}
ENGINE_H = "server/engine.h"


class Manifest:
    def __init__(self, path, root):
        self.path = path
        self.rel = os.path.relpath(path, root).replace("\\", "/")
        self.module = None
        self.exe = None
        self.mutate = None
        self.mutsyms = []
        self.syslibs = []
        self.items = []
        for line in open(path, encoding="utf-8"):
            m = re.match(r"([a-z]+)\s+([^\s#]+)", line)
            if not m:
                continue
            kind, p = m.group(1), m.group(2)
            if kind == "module":
                self.module = p
            elif kind == "exe":
                self.exe = p
            elif kind == "lib":
                pass  # cmake only
            elif kind == "mutate":
                self.mutate = p
            elif kind == "mutsym":
                if p not in SYMS:
                    raise SetupError(f"{self.rel}: unknown mutsym {p} (known: {', '.join(SYMS)})")
                self.mutsyms.append(p)
            elif kind == "syslib":
                self.syslibs.append(p)
            elif kind in KINDS:
                if not os.path.isfile(os.path.join(root, p)):
                    raise SetupError(f"{self.rel}: no such file {p}")
                self.items.append((kind, p))
            else:
                raise SetupError(f"{self.rel}: unknown kind {kind}")
        if self.module is None and self.rel == "tests/sampling/isolation.txt":
            self.module = "sampling"
        if not self.module or not self.exe or not any(k == "src" for k, _ in self.items) or \
                not any(k == "testsrc" for k, _ in self.items):
            raise SetupError(f"{self.rel}: needs module, exe, src and testsrc lines")
        if self.mutate and (("src", self.mutate) not in self.items):
            raise SetupError(f"{self.rel}: mutate {self.mutate} is not one of its src lines")


class SetupError(Exception):
    pass


def manifests(root):
    paths = [os.path.join(root, "tests", "sampling", "isolation.txt")] + \
        sorted(glob.glob(os.path.join(root, "tests", "iso", "*", "isolation.txt")))
    out = [Manifest(p, root) for p in paths if os.path.isfile(p)]
    names = [m.module for m in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SetupError(f"module listed twice: {', '.join(dup)}")
    return out


def vc_env(bat):
    if not os.path.isfile(bat):
        raise SetupError(f"no {bat}")
    out = subprocess.run(f'cmd /c ""{bat}" >nul 2>nul && set"', capture_output=True, text=True, shell=False).stdout
    env = {}
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep:
            env[k] = v
    env["VSLANG"] = "1033"  # English cl / link messages: the C1083 / LNK2019 text is matched (arch025-X3fix)
    if "INCLUDE" not in env or not shutil.which("cl", path=env.get("PATH", "")):
        raise SetupError("vcvars64 gave no cl / INCLUDE")
    return env


def run(cmd, env, cwd):
    # CreateProcess looks a bare name up in THIS process's PATH, not in env= (CPython subprocess docs, Windows), so
    # cl / link from the vcvars env are resolved to full paths first (arch025-X3fix)
    if not os.path.isabs(cmd[0]):
        full = shutil.which(cmd[0], path=env.get("PATH") or env.get("Path", ""))
        if not full:
            raise SetupError(f"{cmd[0]} not on the vcvars PATH")
        cmd = [full, *cmd[1:]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def staged(p):
    return re.sub(r"^(src|include|tests)/", "", p)


def stage(root, man, d):
    shutil.rmtree(d, ignore_errors=True)
    out = {"src": [], "testsrc": []}
    for kind, p in man.items:
        if staged(p) == p:
            # arch025-lane2 leg 3: outside src/ include/ tests/ (kernels/gfx1151/attn_ws4_index.h, included from
            # src/ops/attention.cpp as "../../kernels/..."): next to mod/, as next to src/ in the tree (cmake/iso.cmake)
            dst = os.path.join(d, *p.split("/"))
        else:
            dst = os.path.join(d, "test" if kind.startswith("test") else "mod", *staged(p).split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(root, p), dst)
        if kind in out:
            out[kind].append((p, dst))
    return out


def relative_engine(src_rel):
    # from the source's directory to src/server/engine.h, as a relative include ("../server/engine.h" from src/sampling)
    up = len(staged(src_rel).split("/")) - 1
    return "/".join([".."] * up + [ENGINE_H]) if up else ENGINE_H


def mutate(text, case, src_rel):
    if case in ("include_engine", "include_relative"):
        inc = ENGINE_H if case == "include_engine" else relative_engine(src_rel)
        m = re.search(r'^#include "[^"]+"[^\n]*\n', text, re.M)
        assert m, f"{src_rel}: no quoted include to anchor the include mutant"
        return text[:m.end()] + f'#include "{inc}"  // iso mutant\n' + text[m.end():]
    decl, call = SYMS[case[len("call_"):]]
    return text + "\n// iso mutant: a function nobody calls; /OPT:NOREF keeps it, so its reference must resolve\n" + decl + \
        f"void whirl_iso_mutant_probe() {{ {call} }}\n"


def compile_obj(src, incs, obj, env, d):
    return run(["cl", *CL_FLAGS, "/c", *[f"/I{i}" for i in incs], src, f"/Fo{obj}"], env, d)


def build_iso(man, d, files, env, objs_from=None, only_src=None):
    """-> (stage, rc, output) of the first failing stage, or ('run', rc, output) of the test program.
    objs_from: reuse the control's objects for every source except only_src (mutants)."""
    mod, test = os.path.join(d, "mod"), os.path.join(d, "test")
    objs = []
    for i, (rel, s) in enumerate(files["src"]):
        o = os.path.join(d, f"m{i}.obj")
        if objs_from and rel != only_src:
            objs.append(os.path.join(objs_from, f"m{i}.obj"))
            continue
        rc, out = compile_obj(s, [mod], o, env, d)
        if rc:
            return "compile", rc, out
        objs.append(o)
    for i, (rel, s) in enumerate(files["testsrc"]):
        o = os.path.join(d, f"t{i}.obj")
        if objs_from:
            objs.append(os.path.join(objs_from, f"t{i}.obj"))
            continue
        rc, out = compile_obj(s, [mod, test], o, env, d)
        if rc:
            return "test-compile", rc, out
        objs.append(o)
    exe = os.path.join(d, man.exe + ".exe")
    rc, out = run(["link", "/nologo", "/OPT:NOREF", f"/OUT:{exe}", *objs, *man.syslibs], env, d)
    if rc:
        return "link", rc, out
    rc, out = run([exe], env, d)
    return "run", rc, out


def product_compile(root, rel_src, text, tag, env, d):
    # tag = <module>_<case>: modules checked in parallel can mutate sources of one directory
    tmp = os.path.join(root, os.path.dirname(rel_src), f"_iso_mut_{tag}.cpp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        rc, out = compile_obj(tmp, [os.path.join(root, "src"), os.path.join(root, "include")], os.path.join(d, "prod.obj"), env, d)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return rc, out


def tail(out, n=6):
    return "\n".join("    " + l for l in out.strip().splitlines()[-n:])


def cases_of(man, mutants):
    cs = [("control", None, "run")]
    if mutants:
        if not man.mutate:
            raise SetupError(f"{man.rel}: --mutants needs a mutate line")
        cs += [("include_engine", "compile", "C1083"), ("include_relative", "compile", "C1083")]
        cs += [(f"call_{s}", "link", "LNK2019") for s in (man.mutsyms or ["now", "log"])]
    return cs


def check_module(root, man, work, env, mutants, keep):
    lines = []
    cases = cases_of(man, mutants)
    good = 0
    base = os.path.join(work, man.module)
    ctl = os.path.join(base, "control")
    ctl_ok = False
    for name, want_stage, want_code in cases:
        d = os.path.join(base, name)
        files = stage(root, man, d)
        if name == "control":
            st, rc, out = build_iso(man, d, files, env)
            ok = st == "run" and rc == 0 and re.search(r"\b[1-9]\d* passed, 0 failed", out) is not None
            ctl_ok = ok
            summ = re.findall(r"^.* tests: \d+ passed, \d+ failed$", out, re.M)
            lines.append(f"{'ok  ' if ok else 'FAIL'} {man.module} control: isolated build + run -> {st} rc={rc} {summ[-1] if summ else ''}")
            if not ok:
                lines.append(tail(out, 14))
        else:
            src_text = open(os.path.join(root, man.mutate), encoding="utf-8").read()
            mut = mutate(src_text, name, man.mutate)
            target = dict(files["src"])[man.mutate]
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                f.write(mut)
            # the control's objects are reused for the unchanged sources (only when the control built them)
            st, rc, out = build_iso(man, d, files, env, objs_from=ctl if ctl_ok else None, only_src=man.mutate)
            caught = st == want_stage and rc != 0 and want_code in out
            prc, pout = product_compile(root, man.mutate, mut, f"{man.module}_{name}", env, d)
            ok = caught and prc == 0
            msg = [l.strip() for l in out.splitlines() if want_code in l][:1]
            lines.append(f"{'ok  ' if ok else 'FAIL'} {man.module} {name}: isolated -> {st} rc={rc} ({want_code} "
                         f"{'seen' if want_code in out else 'MISSING'}); product flags compile rc={prc} (must be 0)")
            if msg:
                lines.append("    " + msg[0])
            if not ok:
                lines.append(tail(out))
                if prc:
                    lines.append(tail(pout))
            if not keep:
                shutil.rmtree(d, ignore_errors=True)
        good += ok
    if not keep:
        shutil.rmtree(base, ignore_errors=True)
    n = len(cases)
    lines.append(f"iso check: {'ok' if good == n else 'FAIL'} ({man.module}, {good}/{n} cases as expected)")
    return good, n, lines


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="", help="comma-separated module names (default: every manifest)")
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--list", action="store_true", help="print module, exe and manifest of every isolated module")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "whirl_iso_check"))
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--vcvars", default=VCVARS)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    # python -I ignores PYTHONIOENCODING; a cp950 console cannot print U+FFFD from a decoded cl line (arch025-X3fix)
    sys.stdout.reconfigure(errors="replace")
    try:
        mans = manifests(a.root)
        if a.list:
            for m in mans:
                print(f"{m.module}\t{m.exe}\t{m.rel}")
            return 0
        if a.module:
            want = [x for x in a.module.split(",") if x]
            known = {m.module: m for m in mans}
            bad = [x for x in want if x not in known]
            if bad:
                raise SetupError(f"unknown module(s) {', '.join(bad)} (--list)")
            mans = [known[x] for x in want]
        for m in mans:
            cases_of(m, a.mutants)  # setup errors before any build
        env = vc_env(a.vcvars)
    except SetupError as ex:
        print(f"iso check: FAIL (setup: {ex})")
        return 2
    os.makedirs(a.work, exist_ok=True)
    tot_good = tot_n = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = [ex.submit(check_module, a.root, m, a.work, env, a.mutants, a.keep) for m in mans]
        for m, f in zip(mans, futs):
            try:
                good, n, lines = f.result()
            except SetupError as e:
                print(f"iso check: FAIL (setup: {e})")
                return 2
            print("\n".join(lines), flush=True)
            tot_good += good
            tot_n += n
            if good != n:
                failed.append(m.module)
    if len(mans) > 1:
        if failed:
            print(f"iso check: FAIL (all: {len(mans)} modules, {tot_good}/{tot_n} cases as expected; failed: {', '.join(failed)})")
        else:
            print(f"iso check: ok (all: {len(mans)} modules, {tot_good}/{tot_n} cases as expected)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
