#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel ABI layout check (arch025 N13, CPU only).

The by-value kernel argument structs (KvArgs, GvArgs, Tok16, RowTab, RowIdx, AwGroups, DSampArgs, GdnSeg / GdnSegs)
exist twice by necessity: the host definition in include/whirl/kernels_abi.h (DevPtr fields, MSVC) and the device
definition in kernels/kernel_args.h (typed pointers, HIP clang --cuda-device-only; shared by the gfx1201 and gfx1151
kernel sets). Their layout is defined once, in include/whirl/kernels_abi_layout.h, and both sides static_assert
sizeof / offsetof against it, so a change on one side fails that side's compile.

This check
  1. compiles kernels/kernel_args.h for every device arch (-fsyntax-only) and include/whirl/kernels_abi.h as host
     C++20 (-fsyntax-only, the same clang), both must pass;
  2. requires that the device structs are defined only in kernels/kernel_args.h (no second copy under kernels/);
  3. --selftest: on temporary copies, a one-field change on the device side, on the host side and in the layout table
     each must make the check fail (the static_asserts really bind both sides), and the unchanged copy must pass.

    python tools/gates/kernel_abi_check.py [--repo R] [--selftest]
Exit 0 = pass, 1 = fail, 2 = tool error (clang missing).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ARCHS = ("gfx1201", "gfx1151")
STRUCTS = ("KvArgs", "GvArgs", "Tok16", "RowTab", "RowIdx", "AwGroups", "DSampArgs", "GdnSeg", "GdnSegs")
DEVICE_HDR = "kernels/kernel_args.h"
HOST_HDR = "include/whirl/kernels_abi.h"
LAYOUT_HDR = "include/whirl/kernels_abi_layout.h"
CLANG_DEFAULT = r"C:\Program Files\AMD\ROCm\7.2\bin\clang.exe"


def clang_path() -> str:
    c = os.environ.get("WHIRL_HIP_CLANG", CLANG_DEFAULT)
    if not os.path.isfile(c):
        found = shutil.which("clang")
        if not found:
            print(f"kernel_abi_check: clang not found ({c})")
            sys.exit(2)
        c = found
    return c


def run(cmd, stdin=None):
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def check(repo: str, quiet: bool = False) -> list[str]:
    """returns the list of failures (empty = pass)"""
    clang = clang_path()
    fails = []
    dev = os.path.join(repo, DEVICE_HDR)
    for need, what in ((DEVICE_HDR, "the one device definition of the kernel argument structs"),
                       (LAYOUT_HDR, "the one layout table")):
        if not os.path.isfile(os.path.join(repo, need)):
            fails.append(f"missing {need} ({what})")
    if fails:
        if not quiet:
            for f in fails:
                print("FAIL", f)
        return fails
    tu = '#include <hip/hip_runtime.h>\n#include "kernel_args.h"\n'
    for a in ARCHS:
        rc, out = run([clang, "-x", "hip", f"--offload-arch={a}", "--cuda-device-only", "-std=c++17", "-fsyntax-only",
                       "-I", os.path.join(repo, "kernels"), "-"], stdin=tu)
        if rc != 0:
            fails.append(f"device {a}: {DEVICE_HDR} does not match {LAYOUT_HDR}:\n{out[-3000:]}")
    rc, out = run([clang, "-x", "c++", "-std=c++20", "-fsyntax-only", "-Wno-pragma-once-outside-header",
                   "-I", os.path.join(repo, "include"), os.path.join(repo, HOST_HDR)])
    if rc != 0:
        fails.append(f"host: {HOST_HDR} does not match {LAYOUT_HDR}:\n{out[-3000:]}")
    # one device definition
    pat = re.compile(r"^\s*struct\s+(" + "|".join(STRUCTS) + r")\s*\{", re.M)
    kdir = os.path.join(repo, "kernels")
    for root, _dirs, files in os.walk(kdir):
        for f in files:
            if not f.endswith((".hip", ".h")):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, repo).replace("\\", "/")
            if rel == DEVICE_HDR:
                continue
            with open(p, encoding="utf-8", errors="replace") as fh:
                for m in pat.finditer(fh.read()):
                    fails.append(f"{rel}: second device definition of struct {m.group(1)} (define it only in {DEVICE_HDR})")
    if not quiet:
        for f in fails:
            print("FAIL", f)
    return fails


def selftest(repo: str) -> int:
    """mutate one field per side on a temporary copy; each mutant must fail, the unchanged copy must pass"""
    mutants = [
        (DEVICE_HDR, "    int tab0;", "    long long tab0;"),
        (DEVICE_HDR, "    int t[16];", "    int t[17];"),
        (DEVICE_HDR, "    float *pend;", "    float *pend;\n    int extra_;"),
        (HOST_HDR, "    std::int32_t tab0 = 0;", "    std::int64_t tab0 = 0;"),
        (HOST_HDR, "    std::int32_t pos[kMaxVerifyRows] = {};", "    std::int32_t pos[kMaxVerifyRows + 1] = {};"),
        (LAYOUT_HDR, "kKvArgsWin = 52", "kKvArgsWin = 48"),
    ]
    bad = 0
    with tempfile.TemporaryDirectory(prefix="kabi_") as td:
        for d in ("kernels", "include"):
            shutil.copytree(os.path.join(repo, d), os.path.join(td, d),
                            ignore=shutil.ignore_patterns("vision", "*.hip") if d == "kernels" else None)
        if check(td, quiet=True):
            print("SELFTEST FAIL: the unchanged copy does not pass")
            return 1
        for rel, old, new in mutants:
            p = os.path.join(td, rel)
            with open(p, encoding="utf-8") as fh:
                orig = fh.read()
            if old not in orig:
                print(f"SELFTEST FAIL: mutant anchor not found in {rel}: {old!r}")
                bad += 1
                continue
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(orig.replace(old, new, 1))
            fails = check(td, quiet=True)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(orig)
            ok = bool(fails)
            print(f"{'ok  ' if ok else 'MISS'} {rel}: {old.strip()!r} -> {new.strip()!r}")
            bad += 0 if ok else 1
        # a second device copy must be caught
        extra = os.path.join(td, "kernels", "dup_probe.hip")
        with open(extra, "w", encoding="utf-8") as fh:
            fh.write("struct KvArgs {\n    void *k;\n};\n")
        ok = bool(check(td, quiet=True))
        os.remove(extra)
        print(f"{'ok  ' if ok else 'MISS'} second device definition of KvArgs")
        bad += 0 if ok else 1
    print("kernel_abi_check selftest:", "PASS" if bad == 0 else f"FAIL ({bad} missed)")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.repo)
    fails = check(a.repo)
    print("kernel_abi_check:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
