# SPDX-License-Identifier: Apache-2.0
"""Compile WHIRL device code objects outside cmake, for kernel_isa_cmp.py (arch025-sc11c, ADR-0005; replaces
scratch/a25_sc11k/cobuild.sh).

The clang flags are read from the build files themselves, so they cannot drift from the real build:
  whirl   CMakeLists.txt     the add_custom_command that writes whirl_kernels.<arch>.hsaco
  vision  cmake/vision.cmake the add_custom_command that writes whirl_vision.<arch>.hsaco
Only the depfile options (-Xclang -dependency-file / -MT, no effect on the output) are dropped; ${arch},
${CMAKE_SOURCE_DIR}, ${co}/${vco} are substituted (the source dir = --tree, so a scratch copy of the kernels can be built
with the flags of the worktree's build files).

    python -I tools/gates/kernel_co_build.py --tree <dir with kernels/> --out <dir> --side <name>
           [--arch gfx1201,gfx1151] [--what whirl,vision] [--jobs N] [--repo <dir with CMakeLists.txt>]
           [--print-commands]
Writes <out>/<side>.<what>.<arch>.hsaco and .log; last action <out>/<side>.DONE ("kernel_co_build <side>: <what>.<arch>=rc0
(hh:mm:ss-hh:mm:ss) ..."). Exit 0 only if every compile returned 0. Whole run: about 6-7 min for whirl gfx1201 (the
largest), the rest in parallel; start it detached and wait for the DONE file.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLANG = os.environ.get("WHIRL_HIP_CLANG", r"C:\Program Files\AMD\ROCm\7.2\bin\clang.exe")
TARGETS = {"whirl": ("CMakeLists.txt", "whirl_kernels."), "vision": (os.path.join("cmake", "vision.cmake"), "whirl_vision.")}


def cmake_command(repo, what):
    """The tokens of the COMMAND "${WHIRL_HIP_CLANG}" ... of the custom command that writes the <what> code object."""
    rel, stem = TARGETS[what]
    text = open(os.path.join(repo, rel), encoding="utf-8").read()
    for m in re.finditer(r"add_custom_command\((.*?)\n\s*VERBATIM\)", text, re.S):
        body = m.group(1)
        if stem + "${arch}.hsaco" not in body and not re.search(r"OUTPUT\s+\"\$\{v?co\}\"", body):
            continue
        cm = re.search(r'COMMAND\s+"\$\{WHIRL_HIP_CLANG\}"(.*?)\n\s*(?:#|DEPFILE|DEPENDS|COMMENT)', body, re.S)
        if not cm:
            continue
        toks = re.findall(r'"[^"]*"|\S+', cm.group(1))
        toks = [t[1:-1] if t.startswith('"') else t for t in toks]
        if "-o" in toks:
            return toks
    raise SystemExit("kernel_co_build: no clang command for %s in %s" % (what, rel))


def flags_for(repo, what, arch, tree, out_path):
    toks = cmake_command(repo, what)
    res, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == "-Xclang" and i + 1 < len(toks) and toks[i + 1] in ("-dependency-file", "-MT"):
            i += 4  # -Xclang -dependency-file -Xclang <file>  /  -Xclang -MT -Xclang <target>
            continue
        if t == "-o":
            res += ["-o", out_path]
            i += 2
            continue
        t = t.replace("${arch}", arch).replace("${CMAKE_SOURCE_DIR}", tree.replace("\\", "/"))
        res.append(t)
        i += 1
    if any("${" in t for t in res):
        raise SystemExit("kernel_co_build: unsubstituted variable in %s command: %s" % (what, res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--arch", default="gfx1201,gfx1151")
    ap.add_argument("--what", default="whirl,vision")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--print-commands", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for what in a.what.split(","):
        for arch in a.arch.split(","):
            co = os.path.join(a.out, "%s.%s.%s.hsaco" % (a.side, what, arch))
            jobs.append((what, arch, co, [CLANG] + flags_for(a.repo, what, arch, os.path.abspath(a.tree), co)))
    if a.print_commands:
        for what, arch, _, cmd in jobs:
            print("%s.%s: %s" % (what, arch, " ".join(cmd)))
        return 0
    done = os.path.join(a.out, a.side + ".DONE")
    if os.path.exists(done):
        os.remove(done)

    def run(job):
        what, arch, co, cmd = job
        t0 = time.strftime("%H:%M:%S")
        with open(co[:-6] + ".log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        return "%s.%s=rc%d(%s-%s)" % (what, arch, rc, t0, time.strftime("%H:%M:%S")), rc

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        res = list(ex.map(run, jobs))
    line = "kernel_co_build %s: %s" % (a.side, " ".join(r[0] for r in res))
    with open(done + ".tmp", "w") as f:
        f.write(line + "\n")
    os.replace(done + ".tmp", done)
    print(line)
    return 1 if any(r[1] for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
