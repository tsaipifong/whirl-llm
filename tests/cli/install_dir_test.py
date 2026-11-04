# PATH-256 P3 (v0.2.5.1 PLAN V1-22; progress/pathlen256.DISCOVERY.md v3.1 2c #2, 2d-3; Sir Pifeng 2026-11-03 17:44
# 「安裝資料夾 ≤150」): whirl.exe, whirl-server.exe and whirl-vis.exe (when built) refuse to start from an install folder
# longer than 150 UTF-8 bytes - exit 2 and a message in English and Traditional Chinese that names the unit - as the
# first thing their main function does, before the --whirl-gpu-probe branch and the GPU-runtime preflight, so before
# anything can load the AMD runtime. 150 bytes start as before.
#
# Made on the CPU like preflight_order_test.py: each exe is hard-linked (copied when the volume does not allow it) into a
# test folder next to an "amdhip64_7.dll" that is NOT a DLL. A file of that name in the exe's folder is the only
# candidate the runtime loader tries (src/hip/runtime_loader.cpp probeTable step 1), so the real AMD runtime is never
# loaded and no GPU is touched: a command that reaches the preflight (or the probe) exits 3 (exit_gpu) - the positive
# control that this test sees a runtime load being attempted - while a refused start exits 2 before it.
# The limit is in UTF-8 bytes, so the test folders reach 150 / 151 / 153 bytes with Chinese characters (3 bytes each)
# while every exe path stays at most 150 characters: nothing is ever run from a path longer than 150 characters
# (checked below; the test refuses to run otherwise). A pure-ASCII folder of 151 characters is covered by the unit test
# of path_budget::installDirRefusal (tests/iso/platform.path_budget), not run here.
#   python -I tests/cli/install_dir_test.py --bin <build\Release>
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

EXIT_USAGE = 2
EXIT_GPU = 3
EN = "the install folder path is %d bytes in UTF-8 (English letters and digits count 1, Chinese characters count 3); the limit is 150."
ZH = "安裝資料夾路徑長 %d 位元組"  # 安裝資料夾路徑長 N 位元組
ZH_LIMIT = "上限 150"  # 上限 150
GPU_MSG = "AMD GPU runtime"
MAX_RUN_CHARS = 150  # never run anything from a path longer than this many characters


def make_folder(base, tag, nbytes, cjk):
    """<base>\\<tag>\\<cjk Chinese characters><ASCII padding>: a folder whose full path is exactly nbytes UTF-8 bytes."""
    head = os.path.join(base, tag, "中" * cjk)
    pad = nbytes - len(head.encode("utf-8")) - 1
    if pad < 1:
        raise RuntimeError("base folder too long for %d bytes" % nbytes)
    d = os.path.join(head, "d" * pad)
    assert len(d.encode("utf-8")) == nbytes, (d, nbytes)
    os.makedirs(d)
    with open(os.path.join(d, "amdhip64_7.dll"), "wb") as f:
        f.write(b"not a DLL (PATH-256 install_dir_test)\n")
    return d


def place(src, d):
    dst = os.path.join(d, os.path.basename(src))
    if len(dst) > MAX_RUN_CHARS:
        raise RuntimeError("exe path %d characters > %d: not run" % (len(dst), MAX_RUN_CHARS))
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def run(exe, args, env, cwd):
    assert len(exe) <= MAX_RUN_CHARS and len(cwd) <= MAX_RUN_CHARS
    p = subprocess.run([exe] + args, capture_output=True, env=env, timeout=120, cwd=cwd)
    return p.returncode, p.stdout.decode("utf-8", errors="replace"), p.stderr.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    o = ap.parse_args()
    exes = {n: os.path.join(o.bin, n + ".exe") for n in ("whirl", "whirl-server", "whirl-vis")}
    for n in ("whirl", "whirl-server"):
        if not os.path.isfile(exes[n]):
            print("install dir test: setup error, %s missing" % exes[n])
            return 1
    have_vis = os.path.isfile(exes["whirl-vis"])
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("WHIRL_")}
    tmp = tempfile.mkdtemp(prefix="whirl_idir_")
    ok = bad = 0

    def case(name, good, detail):
        nonlocal ok, bad
        if good:
            ok += 1
            print("ok   %s" % name)
        else:
            bad += 1
            print("FAIL %s: %s" % (name, detail[:400]))

    try:
        base = os.path.realpath(tmp)
        if len(base) > 80:
            print("install dir test: setup error, temp folder %s too long (%d characters)" % (base, len(base)))
            return 1
        f = {}
        f[150] = make_folder(base, "a", 150, 30)
        f[151] = make_folder(base, "b", 151, 30)
        f[153] = make_folder(base, "c", 153, 30)
        for k in (150, 151, 153):
            print("folder %d bytes = %d characters" % (k, len(f[k])))
        at = {k: {n: place(p, d) for n, p in exes.items() if os.path.isfile(p)} for k, d in f.items()}

        def refused(rc, err, nbytes, prog):
            return (rc == EXIT_USAGE and (prog + ": " + EN % nbytes) in err and (prog + "：" + ZH % nbytes) in err
                    and ZH_LIMIT in err)

        # 150 bytes: everything starts as before
        rc, out, err = run(at[150]["whirl"], ["--version"], env, f[150])
        case("150 whirl --version", rc == 0 and "whirl" in out and "install folder" not in err, "exit %d, %s" % (rc, err))
        rc, out, err = run(at[150]["whirl-server"], ["--version"], env, f[150])
        case("150 whirl-server --version", rc == 0 and "install folder" not in err, "exit %d, %s" % (rc, err))
        rc, out, err = run(at[150]["whirl"], ["devices"], env, f[150])
        case("150 whirl devices reaches the preflight (control)", rc == EXIT_GPU and GPU_MSG in err, "exit %d, %s" % (rc, err))
        rc, out, err = run(at[150]["whirl-server"], ["no_such_model.gguf"], env, f[150])
        case("150 whirl-server start reaches the preflight (control)", rc == EXIT_GPU and GPU_MSG in err, "exit %d, %s" % (rc, err))
        rc, out, err = run(at[150]["whirl"], ["--whirl-gpu-probe"], env, f[150])
        case("150 whirl --whirl-gpu-probe tries the runtime (control)", rc == EXIT_GPU, "exit %d, %s" % (rc, err))
        # 151 bytes: refused first, whatever the arguments (also --version / --help / the probe), before the preflight
        for prog, args in (("whirl", ["--version"]), ("whirl", ["--help"]), ("whirl", ["devices"]), ("whirl", ["--whirl-gpu-probe"]),
                           ("whirl", ["serve", "no_such_model.gguf"]), ("whirl-server", ["--version"]),
                           ("whirl-server", ["no_such_model.gguf"]), ("whirl-server", ["--whirl-gpu-probe"])):
            rc, out, err = run(at[151][prog], args, env, f[151])
            case("151 %s %s refused" % (prog, " ".join(args)), refused(rc, err, 151, prog), "exit %d, %s" % (rc, err))
        if have_vis:
            for args in (["--whirl-gpu-probe"], ["vis-encode", "x.gguf", "y.png"]):
                rc, out, err = run(at[151]["whirl-vis"], args, env, f[151])
                case("151 whirl-vis %s refused" % " ".join(args), refused(rc, err, 151, "whirl-vis"), "exit %d, %s" % (rc, err))
            rc, out, err = run(at[150]["whirl-vis"], ["--whirl-gpu-probe"], env, f[150])
            case("150 whirl-vis --whirl-gpu-probe tries the runtime (control)", rc == EXIT_GPU, "exit %d, %s" % (rc, err))
        else:
            print("note: whirl-vis.exe not built, its cases not run")
        # 153 bytes in about 95 characters: the unit is bytes, not characters
        rc, out, err = run(at[153]["whirl-server"], ["--version"], env, f[153])
        case("153 bytes whirl-server --version refused", refused(rc, err, 153, "whirl-server"), "exit %d, %s" % (rc, err))
        rc, out, err = run(at[153]["whirl"], ["devices"], env, f[153])
        case("153 bytes whirl devices refused", refused(rc, err, 153, "whirl"), "exit %d, %s" % (rc, err))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("install dir test: %d passed, %d failed" % (ok, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
