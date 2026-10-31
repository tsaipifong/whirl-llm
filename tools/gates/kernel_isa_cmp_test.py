# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/gates/kernel_isa_cmp.py (arch025-sc11k, ADR-0005), CPU only.

Part 1 (no compiler): the pure pieces - msgpack decoding, PC-relative literal symbolization, descriptor normalization.
Part 2 (ROCm clang, a few seconds): tiny synthetic code objects compiled per architecture -
  same source at another path (different __hip_cuid_ symbol)   -> identical, exit 0
  kernels / device function defined in another order            -> identical, reordered, exit 0
  one kernel's arithmetic changed                               -> exit 1, names that kernel only, shows the instruction diff
  a __constant__ table value changed (ISA unchanged)            -> exit 1, names the data object
  a kernel removed                                              -> exit 1, "only in base"
  a kernel's LDS size changed                                   -> exit 1, descriptor / metadata of that kernel
Last line "kernel_isa_cmp test: PASS" or "kernel_isa_cmp test: FAIL (n)". Exit 0 / 1.
    python -I tools/gates/kernel_isa_cmp_test.py [--arch gfx1201,gfx1151] [--no-compile]
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "kernel_isa_cmp.py")
sys.path.insert(0, HERE)
CLANG = os.environ.get("WHIRL_HIP_CLANG", r"C:\Program Files\AMD\ROCm\7.2\bin\clang.exe")
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


HEAD = "#include <hip/hip_runtime.h>\n__constant__ float tab[4] = {1.f, 2.f, 3.f, %s};\n"
HELPER = "__device__ __noinline__ float helper(float x) { return x * tab[(int)x & 3]; }\n"
K1 = "extern \"C\" __global__ void k1(float *o, int n) { int i = threadIdx.x; if (i < n) o[i] = helper(o[i]) + tab[i & 3]; }\n"
K2 = ("extern \"C\" __global__ void k2(float *o) { __shared__ float s[%d]; s[threadIdx.x] = o[threadIdx.x]; __syncthreads();"
      " o[threadIdx.x] = s[63 - threadIdx.x] * %s; }\n")
K3 = "extern \"C\" __global__ void k3(int *o) { o[threadIdx.x] += 7; }\n"


def src(order=("h", "k1", "k2", "k3"), last="4.f", lds=64, mul="2.f", drop=()):
    parts = {"h": HELPER, "k1": K1, "k2": K2 % (lds, mul), "k3": K3}
    return HEAD % last + "".join(parts[p] for p in order if p not in drop)


def compile_co(d, name, text, arch):
    sub = os.path.join(d, name)  # its own folder: the __hip_cuid_ symbol hashes the input path
    os.makedirs(sub, exist_ok=True)
    hip = os.path.join(sub, "k.hip")
    with open(hip, "w") as f:
        f.write(text)
    co = os.path.join(sub, "k.%s.hsaco" % arch)
    r = subprocess.run([CLANG, "-x", "hip", "--offload-arch=" + arch, "--cuda-device-only", "--no-gpu-bundle-output", "-O3",
                        "-std=c++17", "-o", co, hip], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("compile failed %s %s:\n%s" % (name, arch, r.stderr[-2000:]))
    return co


def run(a, b, *extra):
    p = subprocess.run([sys.executable, "-I", TOOL, a, b] + list(extra), capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def part1():
    import kernel_isa_cmp as K
    # msgpack: fixmap / fixstr / positive + negative fixint / uint16 / array / str8 / true
    blob = bytes([0x83, 0xa1, 0x61, 0x05, 0xa1, 0x62, 0x92, 0xff, 0xcd, 0x01, 0x00, 0xa1, 0x63, 0xd9, 0x02, 0x68, 0x69])
    check(K.msgpack_decode(blob) == {"a": 5, "b": [-1, 256], "c": "hi"}, "msgpack decode")
    syms = [K.Sym("tab", 0x800, 16, "OBJECT"), K.Sym("helper", 0x1900, 0x60, "FUNC")]
    # gfx12 sequence: getpc at 0x1918 (next 0x191c) + lo 0xffffeee4 + hi -1 -> 0x800 = tab
    ins = [(0x1918, "s_getpc_b64 s[0:1]"), (0x191c, "s_wait_alu 0xfffe"), (0x1920, "s_sext_i32_i16 s1, s1"),
           (0x1924, "s_add_co_u32 s0, s0, 0xffffeee4"), (0x192c, "s_add_co_ci_u32 s1, s1, -1"),
           (0x1938, "v_and_b32_e32 v1, 3, v1")]
    out = K.symbolize(ins, syms)
    check(out[3] == "s_add_co_u32 s0, s0, <tab>" and out[4] == "s_add_co_ci_u32 s1, s1, <tab:hi>",
          "gfx12 getpc literal -> <tab>: %r" % out[3:5])
    # same code placed 0x100 later with a literal that still lands on tab: identical text
    ins2 = [(a + 0x100, t) for a, t in ins]
    ins2[3] = (0x1a24, "s_add_co_u32 s0, s0, 0xffffede4")
    check(K.symbolize(ins2, syms) == out, "moved code, same target -> same text")
    # gfx11 spelling (s_add_u32 / s_addc_u32), target inside an object -> <sym+0xoff>
    ins3 = [(0x2000, "s_getpc_b64 s[4:5]"), (0x2004, "s_add_u32 s4, s4, 0xffffe804"), (0x200c, "s_addc_u32 s5, s5, -1")]
    out3 = K.symbolize(ins3, syms)
    check(out3[1] == "s_add_u32 s4, s4, <tab+0x8>", "gfx11 getpc literal -> <tab+0x8>: %r" % out3[1])
    check(K.norm_name("__hip_cuid_24e9dba9f98b1a29") == "__hip_cuid_", "cuid symbol name normalized")


def part2(archs):
    d = tempfile.mkdtemp(prefix="kisacmp_")
    try:
        for arch in archs:
            base = compile_co(d, "base_" + arch, src(), arch)
            same = compile_co(d, "same_" + arch, src(), arch)
            reord = compile_co(d, "reord_" + arch, src(order=("k3", "h", "k2", "k1")), arch)
            chg = compile_co(d, "chg_" + arch, src(mul="3.f"), arch)
            tabc = compile_co(d, "tab_" + arch, src(last="5.f"), arch)
            drop = compile_co(d, "drop_" + arch, src(drop=("k3",)), arch)
            lds = compile_co(d, "lds_" + arch, src(lds=128), arch)
            rc, out = run(base, base)
            check(rc == 0 and "byte-identical" in out, "%s: file vs itself -> byte-identical, exit 0 (rc %d)" % (arch, rc))
            rc, out = run(base, same)
            check(rc == 0 and "IDENTICAL" in out and "byte-identical" not in out,
                  "%s: other path (cuid) -> IDENTICAL, exit 0 (rc %d)" % (arch, rc))
            rc, out = run(base, reord)
            check(rc == 0 and "IDENTICAL" in out and "reordered" in out, "%s: reordered -> IDENTICAL (reordered), exit 0 (rc %d)" % (arch, rc))
            rc, out = run(base, chg)
            check(rc == 1 and "DIFFERENT code k2" in out and "\n    - " in out and "\n    + " in out,
                  "%s: one kernel changed -> exit 1, k2 named with its instruction diff (rc %d)" % (arch, rc))
            check("DIFFERENT code k1" not in out and "DIFFERENT code k3" not in out, "%s: unchanged kernels not reported" % arch)
            rc, out = run(base, tabc)
            check(rc == 1 and "DIFFERENT object tab" in out and "DIFFERENT code" not in out,
                  "%s: constant table changed -> exit 1, object tab named, no code diff (rc %d)" % (arch, rc))
            rc, out = run(base, drop)
            check(rc == 1 and "only in base" in out and "k3" in out, "%s: kernel removed -> exit 1, only in base k3 (rc %d)" % (arch, rc))
            rc, out = run(base, lds)
            check(rc == 1 and ("DIFFERENT descriptor k2" in out or "DIFFERENT metadata k2" in out),
                  "%s: LDS size changed -> exit 1, k2 descriptor / metadata (rc %d)" % (arch, rc))
            # directories: matched by file name
            da, db = os.path.join(d, "dirA_" + arch), os.path.join(d, "dirB_" + arch)
            os.makedirs(da)
            os.makedirs(db)
            shutil.copy(base, os.path.join(da, "x.hsaco"))
            shutil.copy(reord, os.path.join(db, "x.hsaco"))
            rc, out = run(da, db)
            check(rc == 0 and "1 code object(s)" in out, "%s: directory pair -> 1 code object, exit 0 (rc %d)" % (arch, rc))
            shutil.copy(chg, os.path.join(db, "x.hsaco"))
            rc, out = run(da, db)
            check(rc == 1, "%s: directory pair with a change -> exit 1 (rc %d)" % (arch, rc))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    archs = ["gfx1201", "gfx1151"]
    if "--arch" in sys.argv:
        archs = sys.argv[sys.argv.index("--arch") + 1].split(",")
    if not os.path.exists(TOOL):
        check(False, "tool exists: " + TOOL)
    else:
        part1()
        if "--no-compile" not in sys.argv:
            part2(archs)
    print("kernel_isa_cmp test: " + ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
