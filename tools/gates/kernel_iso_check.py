#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile boundary of every kernel family (arch025 N14, CPU only, no GPU).

The kernels are one translation unit per architecture (kernels/whirl_kernels.hip, one code object each; kept so for
bit-identical code objects), so a fragment can silently use anything an earlier fragment defined. This check gives each
family the boundary iso_check.py gives the host modules: per architecture, each family is compiled alone
(-fsyntax-only) from a translation unit that holds ONLY its own fragments and those of its declared dependencies
(transitively), in the order of the real translation unit. A use of an undeclared family's code fails that compile.

Manifest: kernels/families.json  {"families": {"kernels.<name>": {"paths": [globs], "deps": [families], "desc": ...}}}
A fragment is a file the root (kernels/whirl_kernels.hip) or a set root (kernels/gfx1151/whirl_kernels_gfx1151.hip)
includes directly; headers they include (kernels/shared/*.h, kernel_args.h, ...) travel with their includer. Every
fragment of every architecture must belong to exactly one family; a dependency must be a family whose fragments come
in the family graph without cycles (fragments of different families may interleave in the translation unit).

    python -I tools/gates/kernel_iso_check.py [--family NAME[,NAME]] [--arch gfx1201,gfx1151] [--selftest] [--list]
--selftest: on a temporary copy of kernels/ + include/whirl/kernels_abi_layout.h, boundary mutants (a fragment gains a
use of an undeclared earlier family's helper) must FAIL their family's isolated compile while the real translation unit
still compiles (-fsyntax-only), and the unchanged copy must pass.
Last line "kernel iso check: ok (N families x M archs)" or "kernel iso check: FAIL (...)". Exit 0 ok, 1 FAIL, 2 setup.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHS = ("gfx1201", "gfx1151")
ROOT = "kernels/whirl_kernels.hip"
SET_ROOTS = ("kernels/gfx1151/whirl_kernels_gfx1151.hip",)
CLANG_DEFAULT = r"C:\Program Files\AMD\ROCm\7.2\bin\clang.exe"
# boundary mutants for --selftest: (file, appended text, family that must fail). Each uses a helper that an EARLIER
# fragment of the real translation unit defines (so the product compile still passes) but that the mutated
# fragment's family does not declare.
MUTANTS = [
    ("kernels/sample_cand.hip",
     "\n__device__ int kiso_mutant_attn(const KvArgs &a) { return kv_tb(a, 0); }  // kernel_iso_check mutant\n",
     "kernels.sample"),
    ("kernels/gfx1151/attn.hip",
     "\n__device__ int kiso_mutant_gemv(const GvArgs &g) { return gv_sel(g).b; }  // kernel_iso_check mutant\n",
     "kernels.attn"),
    # a standalone family (its own code object, not in the root's translation unit): compiled alone per file
    ("kernels/vision/vision.hip",
     "\n__device__ int kiso_mutant_gemv(const GvArgs &g) { return gv_sel(g).b; }  // kernel_iso_check mutant\n",
     "kernels.vision"),
]


def clang():
    c = os.environ.get("WHIRL_HIP_CLANG", CLANG_DEFAULT)
    if not os.path.isfile(c):
        c = shutil.which("clang") or ""
    if not c:
        print("kernel iso check: setup error: HIP clang not found")
        sys.exit(2)
    return c


def rel(p, repo):
    return os.path.relpath(os.path.abspath(p), repo).replace("\\", "/")


def fragments(repo, arch):
    """the files the root / set roots include directly, in translation-unit order (clang -H)"""
    p = subprocess.run([clang(), "-x", "hip", f"--offload-arch={arch}", "--cuda-device-only", "-std=c++17",
                        "-I", os.path.join(repo, "kernels"), "-E", "-H", "-o", os.devnull, os.path.join(repo, ROOT)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{arch}: the real translation unit does not preprocess:\n{p.stderr[-2000:]}")
    roots = {ROOT, *SET_ROOTS}
    stack = [ROOT]
    out = []
    for ln in p.stderr.splitlines():
        m = re.match(r"^(\.+) (.*)$", ln)
        if not m:
            continue
        depth = len(m.group(1))
        path = os.path.normpath(m.group(2).strip())
        try:
            r = rel(path, repo)
        except ValueError:
            r = path
        del stack[depth:]
        includer = stack[-1] if stack else ROOT
        stack.append(r)
        if r.startswith("..") or ":" in r or not r.startswith("kernels/"):
            continue  # system / HIP headers
        if includer in roots and r not in roots:
            out.append(r)
    return out


def load_manifest(repo):
    with open(os.path.join(repo, "kernels", "families.json"), encoding="utf-8") as fh:
        return json.load(fh)["families"]


def family_of(frag, fams):
    hits = [n for n, f in fams.items() if any(fnmatch.fnmatch(frag, g) for g in f.get("paths", []))]
    return hits


def closure(name, fams):
    seen = set()
    todo = [name]
    while todo:
        n = todo.pop()
        for d in fams.get(n, {}).get("deps", []):
            if d not in seen:
                seen.add(d)
                todo.append(d)
    return seen


def compile_tu(repo, arch, files, extra=""):
    tu = "#include <hip/hip_runtime.h>\n" + "".join(
        f'#include "{os.path.join(repo, f).replace(chr(92), "/")}"\n' for f in files) + extra
    p = subprocess.run([clang(), "-x", "hip", f"--offload-arch={arch}", "--cuda-device-only", "-std=c++17", "-fsyntax-only",
                        "-ferror-limit=5", "-I", os.path.join(repo, "kernels"), "-"],
                       input=tu, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stderr


def check(repo, only=None, archs=ARCHS, quiet=False, jobs=6):
    fams = load_manifest(repo)
    problems = []
    plan = []
    for arch in archs:
        frs = fragments(repo, arch)
        fam = {}
        for f in frs:
            h = family_of(f, fams)
            if len(h) != 1:
                problems.append(f"{arch}: fragment {f} belongs to {len(h)} families ({', '.join(h) or 'none'})")
                continue
            fam[f] = h[0]
        for n in fams:
            for d in fams[n].get("deps", []):
                if d not in fams:
                    problems.append(f"{n}: unknown dependency {d}")
            if n in closure(n, fams):
                problems.append(f"{n}: dependency cycle")
        for n in fams:
            if only and n not in only:
                continue
            own = [f for f in frs if fam.get(f) == n]
            if not own:
                continue
            allowed = closure(n, fams) | {n}
            plan.append((n, arch, [f for f in frs if fam.get(f) in allowed]))
    for n, f in fams.items():
        if not f.get("standalone") or (only and n not in only):
            continue
        # own code object (cmake/vision.cmake): not in the root's translation unit, so it compiles alone, no deps
        if f.get("deps"):
            problems.append(f"{n}: a standalone family has no dependencies")
        files = [rel(g, repo) for pat in f.get("paths", []) for g in glob.glob(os.path.join(repo, pat))]
        if not files:
            problems.append(f"{n}: standalone family has no file")
        for arch in archs:
            if files:
                plan.append((n, arch, files))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(compile_tu, repo, arch, files): (n, arch) for n, arch, files in plan}
        for fu in concurrent.futures.as_completed(futs):
            n, arch = futs[fu]
            rc, err = fu.result()
            results.append((n, arch, rc, err))
    for n, arch, rc, err in sorted(results):
        if rc != 0:
            errs = [ln for ln in err.splitlines() if "error:" in ln][:5]
            problems.append(f"{arch} {n}: does not compile with only its declared dependencies:\n    " +
                            "\n    ".join(e.replace(repo.replace("\\", "/"), "").replace(repo, "") for e in errs))
        elif not quiet:
            print(f"ok   {arch} {n}")
    if not quiet:
        for p in problems:
            print("FAIL", p)
    return problems, len(plan)


def selftest(repo):
    bad = 0
    with tempfile.TemporaryDirectory(prefix="kiso_") as td:
        shutil.copytree(os.path.join(repo, "kernels"), os.path.join(td, "kernels"),
                        ignore=None)
        os.makedirs(os.path.join(td, "include", "whirl"))
        shutil.copy(os.path.join(repo, "include", "whirl", "kernels_abi_layout.h"), os.path.join(td, "include", "whirl"))
        probs, _ = check(td, quiet=True)
        if probs:
            print("SELFTEST FAIL: the unchanged copy does not pass:", *probs, sep="\n  ")
            return 1
        for f, text, famname in MUTANTS:
            p = os.path.join(td, f)
            orig = open(p, encoding="utf-8", newline="").read()
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(orig + text)
            arch = "gfx1151" if "/gfx1151/" in f else "gfx1201"
            # the real translation unit must still compile (the mutant is a boundary break, not a build break)
            rc_real, err_real = compile_tu(td, arch, [ROOT])
            probs, _ = check(td, only={famname}, archs=(arch,), quiet=True)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(orig)
            ok = rc_real == 0 and any(famname in x for x in probs)
            why = "real TU fails too" if rc_real != 0 else ("isolated compile passed" if not ok else "caught")
            print(f"{'ok  ' if ok else 'MISS'} {f} -> {famname} ({why})")
            if rc_real != 0:
                print("    ", [ln for ln in err_real.splitlines() if "error:" in ln][:2])
            bad += 0 if ok else 1
    print("kernel iso check selftest:", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--family", default="")
    ap.add_argument("--arch", default=",".join(ARCHS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.repo)
    archs = tuple(x for x in a.arch.split(",") if x)
    if a.list:
        fams = load_manifest(a.repo)
        for arch in archs:
            for f in fragments(a.repo, arch):
                print(arch, f, ",".join(family_of(f, fams)) or "-")
        return 0
    only = set(x for x in a.family.split(",") if x) or None
    probs, n = check(a.repo, only, archs, jobs=a.j)
    print(f"kernel iso check: {'ok' if not probs else 'FAIL'} ({n} family x arch compiles, {len(probs)} problem(s))")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
