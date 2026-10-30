#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ARCH-1 8.5 isolation check of every isolated module (arch025-X3 for src/sampling, arch025-isoall for the rest), MSVC,
CPU only, no GPU, no model.

    python -I tools/gates/iso_check.py [--module NAME[,NAME...]] [--mutants] [--list] [-j N] [--work DIR] [--root REPO]
                                       [--vcvars BAT] [--keep] [--build DIR]

Manifests: tests/sampling/isolation.txt and tests/iso/<module>/isolation.txt (one per module; cmake/iso.cmake reads the
same files). Kinds: module NAME, exe TARGET, lib TARGET (optional object-library name), mutate PATH (the source the
mutants edit), mutinc top|end (where the include mutants put the line: after the first quoted include, the default, or
at the end of the file), mutsym hip|log|now|engine|model|tier|archtable (link mutants), syslib X.lib, define NAME[=VALUE] (arch025-iso4: /D for
the module's sources and test program), sdkinc hip (arch025-iso4: the HIP SDK include/ after the staged tree), gen PATH, then src / hdr (module + declared dependencies)
and testsrc / testhdr (test program only). Every file is copied into <work>/<module>/<case>/{mod,test} (leading src/
include/ tests/ dropped); module sources are compiled with mod/ as the ONLY include path, the test with mod/ + test/, and
the test program is linked from ALL those objects directly (no static-library member selection) with /OPT:NOREF, so an
undeclared symbol in any module object is LNK2019 even when no test calls it.
  control              unchanged copy: compiles, links, the program prints "N passed, 0 failed"           (must PASS)
--mutants adds, per module, boundary mutants on the `mutate` source; each must FAIL the isolated build at the named stage
AND compile with the product flags (/I src /I include), which is what makes the check worth having:
  include_engine       gains #include "server/engine.h"                                -> isolated compile C1083
                       (mutinc end, arch025-lane2 leg 4: at the end of the file, for a source whose anonymous-namespace
                       helpers share a name with engine.h's (tier_bridge.cpp: kv_page, msSince), where the line at the
                       top would not compile with the product flags either; the product source stays as it is)
  include_relative     gains #include "<relative path>/server/engine.h"                -> isolated compile C1083
  call_<sym>           a function appended at the end calls a symbol outside the tree  -> isolated link LNK2019
       hip    whirl::hip::deviceCount       (src/hip, whirl_hip: the device runtime)
       log    whirl::server::Log::line      (src/server/log.cpp, whirl_server_core)
       now    whirl::nowSeconds             (src/common/util.cpp, lib whirl)
       engine whirl::server::serveMain      (src/server/server_main.cpp, whirl_server_core: server startup)
       model  whirl::qwen35::loadOrTune     (src/model/tune.cpp, whirl_model: the model core, class Model)
       tier   whirl::tier::blockHash        (src/tier/kv_tier.cpp, whirl_tier_core: the KV host tier)
       archtable whirl::model::builtinArchs (src/model/qwen35/module.cpp, model.qwen35: the product architecture table)
  inline_<sym>         (header-only modules, arch025-iso4-fix) an uncalled inline function at the end of the header calls
                       the symbol                                                      -> isolated decl-link LNK2019
Header-only modules also get the decl-link (arch025-iso4-fix): each hdr line is compiled alone by the ROCm clang-cl
/Od -Xclang -femit-all-decls (cl never emits an uncalled inline function) and linked with the module and test objects
(/OPT:NOREF); `declbase PATH` names a dependency header whose own inline bodies reach outside the tree (its unresolved
references are the baseline the other headers may not add to). `sdkinc hip` stages the SDK's include/hip ALONE into
<work>/_sdk; a `define` that a product file of a src / hdr line reads is refused unless DEFINE_ALLOW has it.
gen PATH (arch025-lane2 leg 5): a header the build generates (<build>/generated/PATH, e.g. whirl/version.h from
src/release/version.h.in), staged as mod/PATH; --build names the build tree (default <root>/build/Release).
Header-only module (arch025-lane2 leg 6, model.types): `mutate` may name one of its hdr lines, and then src lines are
optional; the test program is the only translation unit, so it is rebuilt for every mutant (the header goes into it),
and the product compile of a mutant compiles the mutated header as a source of its own (so it must be self-contained).
The relative include mutant of a file outside src/ (include/whirl/x.h) names the tree's src/server/engine.h
("../../src/server/engine.h"). --mutants first runs the script's own selftest (pure Python, no compiler; --selftest
alone runs only it): last line "iso check selftest: N/N as expected".
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
import threading

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
    # arch025-lane2 leg 4: the KV host tier (tier.kv, whirl_tier_core), for the model core (the model never reaches the
    # server's cache stack)
    "tier": ("#include <array>\n#include <cstdint>\nnamespace whirl::tier { std::array<std::uint64_t, 2> blockHash(const void* p, "
             "std::uint64_t n); }  // iso mutant: src/tier/kv_tier.cpp (whirl_tier_core, tier.kv)\n",
             "(void)whirl::tier::blockHash(nullptr, 0);"),
    # arch025-lane2 leg 5: the product architecture table (model.qwen35, src/model/qwen35/module.cpp), for the model core
    # (Config::fromGguf checks the qwen35 family's own names, src/model/qwen35/arch_names.h)
    "archtable": ("#include <span>\nnamespace whirl::model { struct ArchModule; const std::span<const ArchModule>& builtinArchs(); }"
                  "  // iso mutant: src/model/qwen35/module.cpp (model.qwen35)\n",
                  "(void)whirl::model::builtinArchs().size();"),
}
ENGINE_H = "server/engine.h"
# sdkinc NAME -> (environment variables that name the SDK root, default root, include sub folder, the ONE folder of it that
# is staged). arch025-iso4-fix (REVIEW 2): the whole ROCm include/ also holds hipblas, hipblaslt, rocblas, rocprim, hipcub,
# thrust, rocwmma, ...; only include/hip is copied into <work>/_sdk/hip/include (once per run) and that copy is the /I, so
# an undeclared ROCm library header is C1083 like an undeclared WHIRL header. The product compile of a mutant keeps the
# full include/ (the product flags).
HIP_ENVS = ("WHIRL_HIP_ROOT", "HIP_PATH")
HIP_DEFAULT = r"C:\Program Files\AMD\ROCm\7.2"
SDK_DIRS = {"hip": (HIP_ENVS, HIP_DEFAULT, "include", "hip")}
SDK_STAGE = None  # <work>/_sdk (main)
# define names a product file may read: the product build defines them the same way (CMakeLists.txt:159,174: __HIP_PLATFORM_AMD__
# for the HIP sources; cmake/vision.cmake:51 WHIRL_VISION_DEFS: WHIRL_VISION_GFX1201 / WHIRL_VISION_GFX1151 for src/vision/kernel_image.cpp)
DEFINE_ALLOW = frozenset({"__HIP_PLATFORM_AMD__", "WHIRL_VISION_GFX1201", "WHIRL_VISION_GFX1151"})
_sdk_lock = threading.Lock()


def sdk_root(envs, default):
    return next((os.environ[e] for e in envs if os.environ.get(e)), default)


def sdk_dirs(man, product=False):
    out = []
    for n in man.sdkinc:
        envs, default, sub, only = SDK_DIRS[n]
        d = os.path.join(sdk_root(envs, default), sub)
        if not os.path.isdir(os.path.join(d, only)):
            raise SetupError(f"{man.rel}: sdkinc {n}: no {os.path.join(d, only)} (set {envs[0]})")
        if product:
            out.append(d)
            continue
        if SDK_STAGE is None:
            raise SetupError("sdkinc needs the staged SDK folder (main sets SDK_STAGE)")
        dst = os.path.join(SDK_STAGE, n, sub)
        with _sdk_lock:
            done = os.path.join(SDK_STAGE, n, "STAGED")
            want = os.path.join(d, only)
            if not os.path.isfile(done) or open(done, encoding="utf-8").read() != want:
                shutil.rmtree(os.path.join(SDK_STAGE, n), ignore_errors=True)
                shutil.copytree(os.path.join(d, only), os.path.join(dst, only))
                with open(done, "w", encoding="utf-8") as f:
                    f.write(want)
        out.append(dst)
    return out


def clang_cl():
    # arch025-iso4-fix (REVIEW 1): the ROCm clang-cl (the compiler of the kernels; MSVC-ABI COFF objects that link.exe links)
    p = os.path.join(sdk_root(HIP_ENVS, HIP_DEFAULT), "bin", "clang-cl.exe")
    if not os.path.isfile(p):
        raise SetupError(f"header-only module: no {p} (the inline-reference check needs the ROCm clang-cl; set {HIP_ENVS[0]})")
    return p


# the header TUs of a header-only module: CL_FLAGS without /O2 (/O2 drops the unreferenced bodies again) and /W4, plus the
# cc1 option that emits every definition (clang/test/CodeGen/emit-all-decls.c)
DECL_FLAGS = [f for f in CL_FLAGS if f not in ("/O2", "/W4")] + ["/Od", "/w", "-Xclang", "-femit-all-decls"]


def def_flags(man):
    return [f"/D{d}" for d in man.defines]


class Manifest:
    def __init__(self, path, root):
        self.path = path
        self.rel = os.path.relpath(path, root).replace("\\", "/")
        self.module = None
        self.exe = None
        self.mutate = None
        self.mutinc = "top"
        self.mutsyms = []
        self.syslibs = []
        self.items = []
        self.gens = []
        self.defines = []
        self.sdkinc = []
        self.declbase = []
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
            elif kind == "mutinc":
                if p not in ("top", "end"):
                    raise SetupError(f"{self.rel}: mutinc {p} (known: top, end)")
                self.mutinc = p
            elif kind == "mutsym":
                if p not in SYMS:
                    raise SetupError(f"{self.rel}: unknown mutsym {p} (known: {', '.join(SYMS)})")
                self.mutsyms.append(p)
            elif kind == "syslib":
                self.syslibs.append(p)
            elif kind == "gen":
                self.gens.append(p)
            elif kind == "define":
                # arch025-iso4: a compile definition of the module's sources and test program (/DNAME[=VALUE]); the one
                # module that needs it is release (its only unit case is compiled under WHIRL_TEST_DLL_DEPS)
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.]+)?", p):
                    raise SetupError(f"{self.rel}: define {p} (NAME or NAME=VALUE)")
                self.defines.append(p)
            elif kind == "sdkinc":
                # arch025-iso4: a system SDK include directory (the platform, like the MSVC / Windows headers): hip = a copy
                # of the HIP SDK's include/hip alone (hip/hip_runtime_api.h, which src/hip/hip_api.h includes for its types;
                # arch025-iso4-fix: no other ROCm library header). It is put AFTER the staged tree, so it can only supply what
                # the tree does not have and never shadows a module header.
                if p not in SDK_DIRS:
                    raise SetupError(f"{self.rel}: unknown sdkinc {p} (known: {', '.join(SDK_DIRS)})")
                self.sdkinc.append(p)
            elif kind == "declbase":
                # arch025-iso4-fix: a declared dependency header of a header-only module whose OWN inline bodies reach code
                # outside the tree (model.arch_open: whirl/model.h's Model / Context members); its unresolved references are
                # the decl-link baseline (see build_iso); must also be a hdr line
                self.declbase.append(p)
            elif kind in KINDS:
                if not os.path.isfile(os.path.join(root, p)):
                    raise SetupError(f"{self.rel}: no such file {p}")
                self.items.append((kind, p))
            else:
                raise SetupError(f"{self.rel}: unknown kind {kind}")
        if self.module is None and self.rel == "tests/sampling/isolation.txt":
            self.module = "sampling"
        # arch025-lane2 leg 6: a header-only module mutates one of its hdr lines and needs no src line
        self.header_only = bool(self.mutate) and ("hdr", self.mutate) in self.items
        if not self.module or not self.exe or not (self.header_only or any(k == "src" for k, _ in self.items)) or \
                not any(k == "testsrc" for k, _ in self.items):
            raise SetupError(f"{self.rel}: needs module, exe, src (or a mutate hdr line) and testsrc lines")
        for p in self.declbase:
            if ("hdr", p) not in self.items or p == self.mutate or not self.header_only:
                raise SetupError(f"{self.rel}: declbase {p} must be a hdr line of a header-only module, not its mutate header")
        if self.mutate and (("src", self.mutate) not in self.items) and not self.header_only:
            raise SetupError(f"{self.rel}: mutate {self.mutate} is not one of its src or hdr lines")
        # arch025-iso4-fix (REVIEW 7): a define reaches the module's product sources too; one that a product file (not under
        # tests/) of a src / hdr line reads could hide a crossing from the iso build (#ifndef WHIRL_ISO_X), so it is refused
        # unless the product build defines it the same way (DEFINE_ALLOW)
        for dname in (x.split("=", 1)[0] for x in self.defines):
            if dname in DEFINE_ALLOW:
                continue
            rx = re.compile(r"\b" + re.escape(dname) + r"\b")
            for kind, p in self.items:
                if kind in ("src", "hdr") and not p.startswith("tests/"):
                    if rx.search(open(os.path.join(root, p), encoding="utf-8", errors="replace").read()):
                        raise SetupError(f"{self.rel}: define {dname} is read by the product file {p} (only test files may read an "
                                         f"iso define; allowed product macros: {', '.join(sorted(DEFINE_ALLOW))})")


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


RUN_TIMEOUT_S = 600  # arch025-iso4: a test program that hangs (a socket test) fails the case instead of the gate step


def out_text(x):
    # arch025-iso4-fix (REVIEW 14): TimeoutExpired.stdout / .stderr are bytes even with text=True
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x or ""


def run(cmd, env, cwd, timeout=None):
    # CreateProcess looks a bare name up in THIS process's PATH, not in env= (CPython subprocess docs, Windows), so
    # cl / link from the vcvars env are resolved to full paths first (arch025-X3fix)
    if not os.path.isabs(cmd[0]):
        full = shutil.which(cmd[0], path=env.get("PATH") or env.get("Path", ""))
        if not full:
            raise SetupError(f"{cmd[0]} not on the vcvars PATH")
        cmd = [full, *cmd[1:]]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as ex:
        return 124, f"timed out after {timeout} s: {cmd[0]}\n" + out_text(ex.stdout) + out_text(ex.stderr)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def staged(p):
    return re.sub(r"^(src|include|tests)/", "", p)


GEN_ROOT = None  # <build>/generated (main: --build)


def stage(root, man, d):
    shutil.rmtree(d, ignore_errors=True)
    out = {"src": [], "testsrc": [], "hdr": []}
    for p in man.gens:
        g = os.path.join(GEN_ROOT or os.path.join(root, "build", "Release", "generated"), *p.split("/"))
        if not os.path.isfile(g):
            raise SetupError(f"{man.rel}: gen {p}: no {g} (configure the build first, or --build)")
        dst = os.path.join(d, "mod", *p.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(g, dst)
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
    if not src_rel.startswith("src/"):
        # arch025-lane2 leg 6: a header under include/ (include/whirl/model_types.h): up to the root, then src/
        return "/".join([".."] * (len(src_rel.split("/")) - 1) + ["src", ENGINE_H])
    up = len(staged(src_rel).split("/")) - 1
    return "/".join([".."] * up + [ENGINE_H]) if up else ENGINE_H


def mutate(text, case, src_rel, inc_at="top"):
    if case in ("include_engine", "include_relative"):
        inc = ENGINE_H if case == "include_engine" else relative_engine(src_rel)
        if inc_at == "end":
            return text + ("" if text.endswith("\n") else "\n") + f'#include "{inc}"  // iso mutant (mutinc end)\n'
        m = re.search(r'^#include "[^"]+"[^\n]*\n', text, re.M)
        assert m, f"{src_rel}: no quoted include to anchor the include mutant"
        return text[:m.end()] + f'#include "{inc}"  // iso mutant\n' + text[m.end():]
    if case.startswith("inline_"):
        # arch025-iso4-fix (REVIEW 1): the crossing inside an inline function nobody calls, the form a header-only module's
        # real code has (inline / constexpr / in-class); only the emitted-definitions objects of the header TUs give it a body
        decl, call = SYMS[case[len("inline_"):]]
        return text + ("" if text.endswith("\n") else "\n") + \
            "// iso mutant: an inline function nobody calls; the -femit-all-decls header objects give it a body\n" + decl + \
            f"inline void whirl_iso_mutant_inline_probe() {{ {call} }}\n"
    decl, call = SYMS[case[len("call_"):]]
    return text + "\n// iso mutant: a function nobody calls; /OPT:NOREF keeps it, so its reference must resolve\n" + decl + \
        f"void whirl_iso_mutant_probe() {{ {call} }}\n"


def compile_obj(src, incs, obj, env, d, defs=()):
    return run(["cl", *CL_FLAGS, *defs, "/c", *[f"/I{i}" for i in incs], src, f"/Fo{obj}"], env, d)


def build_iso(man, d, files, env, objs_from=None, only_src=None, decl_from=None):
    """-> (stage, rc, output) of the first failing stage, or ('run', rc, output) of the test program.
    objs_from: reuse the control's objects for every source except only_src (mutants); decl_from: the same for the header
    TUs of a header-only module (every one but the mutated header's)."""
    mod, test = os.path.join(d, "mod"), os.path.join(d, "test")
    objs = []
    for i, (rel, s) in enumerate(files["src"]):
        o = os.path.join(d, f"m{i}.obj")
        if objs_from and rel != only_src:
            objs.append(os.path.join(objs_from, f"m{i}.obj"))
            continue
        rc, out = compile_obj(s, [mod, *sdk_dirs(man)], o, env, d, def_flags(man))
        if rc:
            return "compile", rc, out
        objs.append(o)
    for i, (rel, s) in enumerate(files["testsrc"]):
        o = os.path.join(d, f"t{i}.obj")
        if objs_from:
            objs.append(os.path.join(objs_from, f"t{i}.obj"))
            continue
        rc, out = compile_obj(s, [mod, test, *sdk_dirs(man)], o, env, d, def_flags(man))
        if rc:
            return "test-compile", rc, out
        objs.append(o)
    decl_objs, base_objs = [], []
    if man.header_only:
        # arch025-iso4-fix (REVIEW 1): cl never emits an inline function nobody calls (not with /Zc:inline- either,
        # scratch/a25_iso4fix/exp1), so the test program alone proves only the paths the test walks. Every hdr line gets a
        # TU of its own (#include of the staged copy), compiled by clang-cl /Od -Xclang -femit-all-decls with mod/ as the only
        # include path: every non-template inline / constexpr / in-class function then has a body in an object, and the
        # decl-link (/OPT:NOREF) makes its references resolve inside the listed sources (LNK2019 otherwise). Mutants reuse the
        # control's objects for every header but the mutated one.
        for i, (rel, s) in enumerate(files["hdr"]):
            o = os.path.join(d, f"h{i}.obj")
            if decl_from and rel != man.mutate:
                o = os.path.join(decl_from, f"h{i}.obj")
            else:
                tu = os.path.join(d, f"h{i}.cpp")
                with open(tu, "w", encoding="utf-8", newline="\n") as f:
                    f.write(f'#include "{s.replace(chr(92), "/")}"  // iso: {rel}\n')
                rc, out = run([clang_cl(), *DECL_FLAGS, *def_flags(man), "/c", f"/I{mod}", *[f"/I{x}" for x in sdk_dirs(man)], tu,
                               f"/Fo{o}"], env, d)
                if rc:
                    return "decl-compile", rc, out
            decl_objs.append(o)
            if rel in man.declbase:
                base_objs.append(o)
    exe = os.path.join(d, man.exe + ".exe")
    rc, out = run(["link", "/nologo", "/OPT:NOREF", f"/OUT:{exe}", *objs, *man.syslibs], env, d)
    if rc:
        return "link", rc, out
    if decl_objs:
        dexe = os.path.join(d, man.exe + "-decls.exe")
        rc, out = run(["link", "/nologo", "/OPT:NOREF", f"/OUT:{dexe}", *objs, *decl_objs, *man.syslibs], env, d)
        if rc and man.declbase:
            # a `declbase` header (a declared dependency whose own inline bodies reach code outside the tree, reviewed and
            # listed in docs/dev/isolation_blockers.md): its unresolved references are the baseline; the module's other
            # headers may not add any reference to it
            got = unresolved(out)
            brc, bout = run(["link", "/nologo", "/OPT:NOREF", f"/OUT:{os.path.join(d, man.exe + '-base.exe')}", *objs, *base_objs,
                             *man.syslibs], env, d)
            base = unresolved(bout) if brc else set()
            extra = sorted(got - base)
            if got and not extra:
                rc = 0
            else:
                out += "\niso: references beyond the declbase baseline: " + ", ".join(extra[:8]) + "\n" + \
                    "\n".join(l for l in out.splitlines() if any(x in l for x in extra[:8]))
        if rc:
            return "decl-link", rc, out
    rc, out = run([exe], env, d, RUN_TIMEOUT_S)
    return "run", rc, out


def unresolved(link_out):
    """The unresolved symbols of a failed link (LNK2019 / LNK2001 lines): the decorated name (the first (?...) of the line),
    else the line after the error code (a C name). -> set"""
    out = set()
    for line in link_out.splitlines():
        m = re.search(r"\bLNK20(?:19|01)\b:?(.*)$", line)
        if not m:
            continue
        dec = re.search(r"\((\?[^()\s]+)\)", m.group(1))
        out.add(dec.group(1) if dec else m.group(1).split('"')[0].strip())
    return out


def engine_includes(root):
    """The repo files src/server/engine.h includes, transitively (quoted includes; resolved next to the including file,
    then under src/, then include/). -> set of repo-relative paths."""
    seen, todo = set(), ["src/" + ENGINE_H]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            text = open(os.path.join(root, *rel.split("/")), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.M):
            for base in (os.path.dirname(rel), "src", "include"):
                cand = os.path.normpath(os.path.join(base, inc)).replace("\\", "/")
                if os.path.isfile(os.path.join(root, *cand.split("/"))):
                    todo.append(cand)
                    break
    return seen


def probe_plan(rel_hdr, case, engine_back):
    """How the product compile of a src/ or out-of-tree header mutant is done: "copy" = the mutated text as a copy next to
    the real header (what is compiled is the mutant); "engine_first" = #include of the added engine include, then the real
    header (same text order) - only for an include mutant of a header that server/engine.h includes back, where a second
    copy would define its classes twice (arch025-iso4)."""
    # arch025-iso4-fix (REVIEW 6): every other header gets its mutated copy, so the product compile checks the mutant's text
    if case in ("include_engine", "include_relative") and rel_hdr in engine_back:
        return "engine_first"
    return "copy"


def product_incs(root, man):
    """The include path of the product compile of a mutant: src/, include/, the SDK of the sdkinc lines, and <build>/generated
    when the manifest stages a generated header (gen lines; arch025-n8b2: release.cpp includes whirl/version.h, which the
    product build finds there)."""
    out = [os.path.join(root, "src"), os.path.join(root, "include"), *sdk_dirs(man, product=True)]
    if man.gens:
        out.append(GEN_ROOT or os.path.join(root, "build", "Release", "generated"))
    return out


def product_compile_header(root, rel_hdr, text, tag, env, d, man, case=""):
    # arch025-lane2 leg 6 (header-only module): the mutated header shadows the real one. It is written to
    # <root>/_iso_mut_<tag>/<path without include/ or src/> (same depth as the real file, so a relative include resolves
    # as in the tree) and a one-line source includes it with that directory first on the include path: the product
    # headers that include it (server/engine.h -> whirl/model_types.h) then reach the same copy (#pragma once).
    if staged(rel_hdr) == rel_hdr or rel_hdr.startswith("src/"):
        # arch025-iso4: a header outside src/ include/ tests/ (kernels/gemm_small_addr.h) or under src/ (src/tier/ram_size.h,
        # src/server/backend.h, whose same-directory includes such as "vision_iface.h" a shadow folder would not have): the mutated
        # copy sits next to the real one under a temporary name, as product_compile does for a .cpp, so its relative includes
        # resolve as in the tree. An include mutant of a header that server/engine.h includes back (backend.h, cache_model.h)
        # can not be a second copy of the header (the classes would be defined twice): its product compile is the added include
        # first, then the real header, which is the same text order.
        base = os.path.join(root, os.path.dirname(rel_hdr))
        hdr = os.path.join(base, f"_iso_mut_{tag}.h")
        src = os.path.join(base, f"_iso_mut_{tag}_probe.cpp")
        probe = f'#include "_iso_mut_{tag}.h"\n'
        write_copy = True
        if probe_plan(rel_hdr, case, engine_includes(root)) == "engine_first":
            inc = ENGINE_H if case == "include_engine" else relative_engine(rel_hdr)
            probe = f'#include "{inc}"\n#include "{os.path.basename(rel_hdr)}"\n'
            write_copy = False
        try:
            if write_copy:
                with open(hdr, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(probe)
            return compile_obj(src, product_incs(root, man),
                               os.path.join(d, "prod.obj"), env, d, def_flags(man))
        finally:
            for x in (hdr, src):
                try:
                    os.remove(x)
                except OSError:
                    pass
    top = os.path.join(root, f"_iso_mut_{tag}")
    hdr = os.path.join(top, *staged(rel_hdr).split("/"))
    src = os.path.join(top, "_iso_probe.cpp")
    try:
        os.makedirs(os.path.dirname(hdr), exist_ok=True)
        with open(hdr, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f'#include "{staged(rel_hdr)}"\n')
        rc, out = compile_obj(src, [top, *product_incs(root, man)],
                              os.path.join(d, "prod.obj"), env, d, def_flags(man))
    finally:
        shutil.rmtree(top, ignore_errors=True)
    return rc, out


def product_compile(root, rel_src, text, tag, env, d, man, case=""):
    # tag = <module>_<case>: modules checked in parallel can mutate sources of one directory
    if not rel_src.endswith(".cpp"):
        return product_compile_header(root, rel_src, text, tag, env, d, man, case)
    tmp = os.path.join(root, os.path.dirname(rel_src), f"_iso_mut_{tag}.cpp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        rc, out = compile_obj(tmp, product_incs(root, man),
                              os.path.join(d, "prod.obj"), env, d, def_flags(man))
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
        # a header-only module's header is compiled inside the test program (arch025-lane2 leg 6)
        inc_stage = "test-compile" if man.header_only else "compile"
        cs += [("include_engine", inc_stage, "C1083"), ("include_relative", inc_stage, "C1083")]
        cs += [(f"call_{s}", "link", "LNK2019") for s in (man.mutsyms or ["now", "log"])]
        if man.header_only:
            # arch025-iso4-fix (REVIEW 1): the same symbols reached from an uncalled inline function of the header
            cs += [(f"inline_{s}", "decl-link", "LNK2019") for s in (man.mutsyms or ["now", "log"])]
    return cs


def mutate_target(man, d, files):
    # the staged copy the mutant edits: a src line's copy, or (header-only module) the hdr line's copy under mod/
    if not man.header_only:
        return dict(files["src"])[man.mutate]
    if staged(man.mutate) == man.mutate:
        # arch025-iso4: a header outside src/ include/ tests/ (kernels/gemm_small_addr.h) is staged next to mod/
        return os.path.join(d, *man.mutate.split("/"))
    return os.path.join(d, "mod", *staged(man.mutate).split("/"))


def selftest():
    """Pure-Python checks of what the header-only mode added (arch025-lane2 leg 6). -> (good, n, lines)"""
    lines = []
    res = []

    def case(name, ok):
        res.append(bool(ok))
        lines.append(f"{'ok  ' if ok else 'FAIL'} selftest {name}")

    with tempfile.TemporaryDirectory() as root:
        for p in ("include/whirl/t.h", "kernels/k.h", "src/a/t.cpp", "tests/iso/t/t_tests.cpp"):
            os.makedirs(os.path.join(root, os.path.dirname(p)), exist_ok=True)
            open(os.path.join(root, p), "w").close()
        mp = os.path.join(root, "tests", "iso", "t", "isolation.txt")

        def man(text):
            with open(mp, "w", encoding="utf-8") as f:
                f.write(text)
            try:
                return Manifest(mp, root)
            except SetupError:
                return None

        head = "module t\nexe t-tests\ntestsrc tests/iso/t/t_tests.cpp\nhdr include/whirl/t.h\n"
        m = man(head + "mutate include/whirl/t.h\n")
        case("header-only manifest (mutate hdr, no src) accepted", m is not None and m.header_only)
        case("no src and no mutate hdr refused", man(head) is None)
        case("mutate naming no src / hdr line refused", man(head + "mutate src/a/t.cpp\n") is None)
        m2 = man(head + "src src/a/t.cpp\nmutate src/a/t.cpp\n")
        case("module with a mutate src line is not header-only", m2 is not None and not m2.header_only)
        case("header mutant edits mod/<header>",
             m is not None and mutate_target(m, "D", {"src": []}).replace("\\", "/") == "D/mod/whirl/t.h")
        case("header-only include mutants expected at test-compile",
             m is not None and ("include_engine", "test-compile", "C1083") in cases_of(m, True))
        case("src module include mutants expected at compile",
             m2 is not None and ("include_engine", "compile", "C1083") in cases_of(m2, True))
        md = man(head + "src src/a/t.cpp\nmutate src/a/t.cpp\ndefine WHIRL_X\ndefine WHIRL_Y=1\nsdkinc hip\n")
        case("define lines become /D flags", md is not None and def_flags(md) == ["/DWHIRL_X", "/DWHIRL_Y=1"])
        case("sdkinc hip accepted", md is not None and md.sdkinc == ["hip"])
        case("define with a bad name refused", man(head + "src src/a/t.cpp\nmutate src/a/t.cpp\ndefine 1bad\n") is None)
        case("unknown sdkinc refused", man(head + "src src/a/t.cpp\nmutate src/a/t.cpp\nsdkinc zz\n") is None)
        case("a module with neither has no flags and no sdk dirs", m2 is not None and def_flags(m2) == [] and sdk_dirs(m2) == [])
        # arch025-n8b2: the product compile of a mutant sees the build's generated headers when the manifest stages one (gen
        # whirl/version.h: release.cpp includes it, as the product build does from <build>/generated)
        mg = man(head + "src src/a/t.cpp\nmutate src/a/t.cpp\ngen whirl/version.h\n")
        gen_root = os.path.join(root, "build", "Release", "generated")
        case("gen line: <build>/generated on the product include path",
             mg is not None and product_incs(root, mg) == [os.path.join(root, "src"), os.path.join(root, "include"), gen_root])
        case("no gen line: product include path is src/ include/ only",
             m2 is not None and product_incs(root, m2) == [os.path.join(root, "src"), os.path.join(root, "include")])
        # arch025-iso4-fix (REVIEW 7): a define that a product file of the module reads would let the iso build differ from
        # the product build unseen (a crossing wrapped in #ifndef WHIRL_ISO_X)
        with open(os.path.join(root, "src", "a", "d.cpp"), "w", encoding="utf-8") as f:
            f.write("#ifndef WHIRL_ISO_X\nint crossing();\n#endif\n#if __HIP_PLATFORM_AMD__\n#endif\n")
        os.makedirs(os.path.join(root, "tests", "iso", "t", "f"), exist_ok=True)
        with open(os.path.join(root, "tests", "iso", "t", "f", "fake.cpp"), "w", encoding="utf-8") as f:
            f.write("#ifdef WHIRL_ISO_X\n#endif\n")
        dsrc = head + "src src/a/d.cpp\nmutate src/a/d.cpp\n"
        case("define read by a product src file refused", man(dsrc + "define WHIRL_ISO_X\n") is None)
        case("define on the product allow list accepted", man(dsrc + "define __HIP_PLATFORM_AMD__\n") is not None)
        case("define read only by a tests/ src file accepted",
             man(head + "src src/a/t.cpp\nsrc tests/iso/t/f/fake.cpp\nmutate src/a/t.cpp\ndefine WHIRL_ISO_X\n") is not None)
        case("define read by a product hdr file refused",
             man(head + "src src/a/t.cpp\nhdr src/a/d.cpp\nmutate src/a/t.cpp\ndefine WHIRL_ISO_X=1\n") is None)
        case("declbase naming a hdr line of a header-only module accepted",
             man(head + "hdr kernels/k.h\nmutate include/whirl/t.h\ndeclbase kernels/k.h\n") is not None)
        case("declbase naming no hdr line refused", man(head + "mutate include/whirl/t.h\ndeclbase kernels/k.h\n") is None)
        case("declbase naming the mutate header refused", man(head + "mutate include/whirl/t.h\ndeclbase include/whirl/t.h\n") is None)
        mk = man("module t\nexe t-tests\ntestsrc tests/iso/t/t_tests.cpp\nhdr kernels/k.h\nmutate kernels/k.h\n")
        case("header outside src/ include/ tests/ is staged next to mod/",
             mk is not None and mk.header_only and mutate_target(mk, "D", {"src": []}).replace(chr(92), "/") == "D/kernels/k.h")
    # arch025-iso4-fix (REVIEW 6): only a header engine.h includes back takes the engine-first product compile
    back = {"src/server/engine.h", "src/server/backend.h", "src/cache/cache_model.h"}
    case("include mutant of a header engine.h includes back: engine first",
         probe_plan("src/server/backend.h", "include_engine", back) == "engine_first")
    case("include mutant of any other src/ header: the mutated copy is compiled",
         probe_plan("src/tier/ram_size.h", "include_engine", back) == "copy"
         and probe_plan("src/config/device_choice.h", "include_relative", back) == "copy")
    case("include mutant of an out-of-tree header: the mutated copy is compiled",
         probe_plan("kernels/gemm_small_addr.h", "include_relative", back) == "copy")
    case("call / inline mutants: always the mutated copy",
         probe_plan("src/server/backend.h", "call_hip", back) == "copy" and probe_plan("src/server/backend.h", "inline_hip", back) == "copy")
    # REVIEW 1: the inline mutant kind
    case("header-only modules get inline_<sym> mutants at link",
         m is not None and ("inline_now", "decl-link", "LNK2019") in cases_of(m, True))
    case("src modules get no inline mutants", m2 is not None and not any(c[0].startswith("inline_") for c in cases_of(m2, True)))
    case("inline mutant appends an uncalled inline function",
         "inline void whirl_iso_mutant_inline_probe() { (void)whirl::nowSeconds(); }" in mutate("x", "inline_now", "include/whirl/t.h"))
    case("unresolved: decorated names of LNK2019 / LNK2001 lines",
         unresolved('h9.obj : error LNK2019: x "int f(void)" (?f@@YAHXZ) y "void g(void)" (?g@@YAXXZ)\n'
                    'h2.obj : error LNK2001: x "int f(void)" (?f@@YAHXZ)\nother line\n') == {"?f@@YAHXZ"})
    # REVIEW 14: a timed-out run's bytes are decoded, not printed as b'...'
    case("timeout output decoded", out_text(b"abc\xff") == "abc�" and out_text(None) == "" and out_text("s") == "s")
    case("relative include from include/whirl",relative_engine("include/whirl/model_types.h") == "../../src/server/engine.h")
    case("relative include from src/sampling unchanged", relative_engine("src/sampling/x.cpp") == "../server/engine.h")
    case("relative include from src/server unchanged", relative_engine("src/server/x.cpp") == "../server/engine.h")
    lines.append(f"iso check selftest: {sum(res)}/{len(res)} as expected")
    return sum(res), len(res), lines


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
            mut = mutate(src_text, name, man.mutate, man.mutinc)
            target = mutate_target(man, d, files)
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                f.write(mut)
            # the control's objects are reused for the unchanged sources (only when the control built them; never for
            # a header-only module, whose header is compiled into the test program)
            st, rc, out = build_iso(man, d, files, env, objs_from=ctl if ctl_ok and not man.header_only else None,
                                    only_src=man.mutate, decl_from=ctl if ctl_ok and man.header_only else None)
            caught = st == want_stage and rc != 0 and want_code in out
            prc, pout = product_compile(root, man.mutate, mut, f"{man.module}_{name}", env, d, man, name)
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
    ap.add_argument("--build", default="", help="build tree whose generated/ headers the gen lines stage (default <root>/build/Release)")
    ap.add_argument("--selftest", action="store_true", help="only the script's own pure-Python selftest")
    a = ap.parse_args(argv)
    if a.selftest or a.mutants:
        sgood, sn, slines = selftest()
        print("\n".join(slines), flush=True)
        if sgood != sn:
            return 2
        if a.selftest:
            return 0
    global GEN_ROOT
    GEN_ROOT = os.path.join(a.build or os.path.join(a.root, "build", "Release"), "generated")
    global SDK_STAGE
    SDK_STAGE = os.path.join(os.path.abspath(a.work), "_sdk")
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
