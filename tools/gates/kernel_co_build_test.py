# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/gates/kernel_co_build.py (arch025-sc11c / SC-17), CPU only.

Part 1 (no compiler): the clang commands come from the real build files - CMakeLists.txt (whirl_kernels) and
  cmake/vision.cmake (whirl_vision) - with the depfile options dropped and the source dir / arch / output substituted;
  a build file with an extra flag changes the command (no hard-coded copy that could drift).
Part 2 (ROCm clang, seconds): a tiny tree (kernels/whirl_kernels.hip + kernels/vision/vision.hip) built per architecture
  -> exit 0, every .hsaco written, DONE line with rc0 for each; a tree whose vision source does not compile -> exit 1 and
  the DONE line names vision.<arch>=rc1 (a failed compile is never reported as built).
Last line "kernel_co_build test: PASS" or "kernel_co_build test: FAIL (n)". Exit 0 / 1.
    python -I tools/gates/kernel_co_build_test.py [--arch gfx1201] [--no-compile]
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOL = os.path.join(HERE, "kernel_co_build.py")
sys.path.insert(0, HERE)
import kernel_co_build as kcb  # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def part1():
    tree = os.path.join(REPO, "some_tree")
    w = kcb.flags_for(REPO, "whirl", "gfx1201", tree, "OUT.hsaco")
    v = kcb.flags_for(REPO, "vision", "gfx1151", tree, "OUT.hsaco")
    t = tree.replace("\\", "/")
    check(w[:7] == ["-x", "hip", "--offload-arch=gfx1201", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++17"],
          "whirl: the flags of CMakeLists.txt in order (%s)" % " ".join(w[:7]))
    check(["-I", t + "/kernels"] == w[7:9], "whirl: -I <tree>/kernels")
    check(w[-3:] == ["-o", "OUT.hsaco", t + "/kernels/whirl_kernels.hip"], "whirl: -o OUT <tree>/kernels/whirl_kernels.hip")
    check(v[:7] == ["-x", "hip", "--offload-arch=gfx1151", "--cuda-device-only", "--no-gpu-bundle-output", "-O3", "-std=c++17"],
          "vision: the flags of cmake/vision.cmake in order")
    check(v[-3:] == ["-o", "OUT.hsaco", t + "/kernels/vision/vision.hip"], "vision: -o OUT <tree>/kernels/vision/vision.hip")
    check("-I" not in v, "vision: no include dir (as cmake/vision.cmake)")
    for name, f in (("whirl", w), ("vision", v)):
        check(not any("dependency-file" in x or x == "-MT" or "${" in x for x in f), "%s: depfile options dropped, nothing unsubstituted" % name)
    # drift: an extra flag in the build file shows up in the command
    d = tempfile.mkdtemp(prefix="kcb_t1_")
    try:
        os.makedirs(os.path.join(d, "cmake"))
        cm = open(os.path.join(REPO, "CMakeLists.txt"), encoding="utf-8").read()
        cm2, n = re.subn(r"(--no-gpu-bundle-output -O3 -std=c\+\+17)(\s+-I \"\$\{CMAKE_SOURCE_DIR\}/kernels\")", r"\1 -ffp-contract=off\2", cm, count=1)
        check(n == 1, "drift fixture: CMakeLists.txt clang line found")
        open(os.path.join(d, "CMakeLists.txt"), "w", encoding="utf-8").write(cm2)
        shutil.copy(os.path.join(REPO, "cmake", "vision.cmake"), os.path.join(d, "cmake", "vision.cmake"))
        w2 = kcb.flags_for(d, "whirl", "gfx1201", tree, "OUT.hsaco")
        check("-ffp-contract=off" in w2, "drift: a flag added to CMakeLists.txt is in the command")
        os.remove(os.path.join(d, "cmake", "vision.cmake"))
        open(os.path.join(d, "cmake", "vision.cmake"), "w", encoding="utf-8").write("# no clang command here\n")
        try:
            kcb.flags_for(d, "vision", "gfx1201", tree, "OUT.hsaco")
            check(False, "missing clang command -> error")
        except SystemExit as e:
            check("no clang command" in str(e), "missing clang command -> error (%s)" % e)
    finally:
        shutil.rmtree(d, ignore_errors=True)


WK = '#include <hip/hip_runtime.h>\nextern "C" __global__ void kw(float *o) { o[threadIdx.x] *= 2.f; }\n'
VK = '#include <hip/hip_runtime.h>\nextern "C" __global__ void kv(float *o) { o[threadIdx.x] += 1.f; }\n'


def make_tree(d, vision_src):
    os.makedirs(os.path.join(d, "kernels", "vision"))
    open(os.path.join(d, "kernels", "whirl_kernels.hip"), "w").write(WK)
    open(os.path.join(d, "kernels", "vision", "vision.hip"), "w").write(vision_src)


def run(tree, out, side, arch):
    return subprocess.run([sys.executable, "-I", TOOL, "--tree", tree, "--out", out, "--side", side, "--arch", arch],
                          capture_output=True, text=True)


def part2(arch):
    d = tempfile.mkdtemp(prefix="kcb_t2_")
    try:
        good, bad, out = os.path.join(d, "good"), os.path.join(d, "bad"), os.path.join(d, "out")
        make_tree(good, VK)
        make_tree(bad, VK.replace("+= 1.f;", "+= undeclared_name;"))
        archs = arch.split(",")
        r = run(good, out, "g", arch)
        done = os.path.join(out, "g.DONE")
        line = open(done).read().strip() if os.path.exists(done) else ""
        check(r.returncode == 0, "good tree: exit 0 (%d)" % r.returncode)
        for a in archs:
            for wh in ("whirl", "vision"):
                p = os.path.join(out, "g.%s.%s.hsaco" % (wh, a))
                check(os.path.exists(p) and os.path.getsize(p) > 0, "good tree: %s written" % os.path.basename(p))
                check(re.search(r"\b%s\.%s=rc0\(" % (wh, a), line) is not None, "good tree: DONE has %s.%s=rc0" % (wh, a))
        check(line.startswith("kernel_co_build g: "), "DONE line format (%s)" % line[:60])
        r = run(bad, out, "b", arch)
        done = os.path.join(out, "b.DONE")
        line = open(done).read().strip() if os.path.exists(done) else ""
        check(r.returncode == 1, "broken vision source: exit 1 (%d)" % r.returncode)
        for a in archs:
            check(re.search(r"\bvision\.%s=rc[1-9]" % a, line) is not None, "broken vision source: DONE has vision.%s=rc!=0" % a)
            check(re.search(r"\bwhirl\.%s=rc0\(" % a, line) is not None, "broken vision source: whirl.%s still rc0" % a)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    arch = "gfx1201,gfx1151"
    if "--arch" in sys.argv:
        arch = sys.argv[sys.argv.index("--arch") + 1]
    part1()
    if "--no-compile" not in sys.argv:
        part2(arch)
    print("kernel_co_build test: " + ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
