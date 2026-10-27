# arch025-surv (mutation survivor c2-raw-early, progress\arch025.MUTATION.md class C): whirl.exe runs the GPU-runtime
# preflight (app::gpuPreflight, src/cli/main.cpp) BEFORE every command that needs the GPU, including the two commands
# with their own option parsing (serve, vis-encode: CmdInfo::raw). A machine without a loadable AMD runtime is made on
# the CPU: whirl.exe is copied into a temporary folder next to an "amdhip64_7.dll" that is not a DLL (the loader tries
# the executable's folder first: src/hip/runtime_loader.cpp), so the preflight fails with exit 3 (exit_gpu) and its
# "AMD GPU runtime" message, whatever the command. A command dispatched before the preflight answers its own way
# (missing model -> exit 4, or its option errors) instead. CPU only; nothing is loaded.
#   python -I tests/cli/preflight_order_test.py --bin <build\Release>
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

EXIT_GPU = 3
MSG = "AMD GPU runtime"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    o = ap.parse_args()
    src = os.path.join(o.bin, "whirl.exe")
    if not os.path.isfile(src):
        print("preflight order test: setup error, %s missing" % src)
        return 1
    tmp = tempfile.mkdtemp(prefix="whirl_preflight_order_")
    ok = bad = 0
    try:
        exe = os.path.join(tmp, "whirl.exe")
        shutil.copy2(src, exe)
        with open(os.path.join(tmp, "amdhip64_7.dll"), "wb") as f:
            f.write(b"not a DLL (arch025-surv preflight_order_test)\n")
        miss = os.path.join(tmp, "no_such_model.gguf")
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("WHIRL_")}
        cases = [
            ("serve", ["serve", miss]),                  # raw command
            ("vis-encode", ["vis-encode", miss, miss]),  # raw command
            ("chat", ["chat", miss, "x"]),               # parsed command (control: always after the preflight)
            ("selftest", ["selftest", miss]),
        ]
        for name, args in cases:
            p = subprocess.run([exe] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                               timeout=120, cwd=tmp)
            good = p.returncode == EXIT_GPU and MSG in p.stderr
            if good:
                ok += 1
                print("ok   %s" % name)
            else:
                bad += 1
                print("FAIL %s: exit %d (want %d), stderr: %s" % (name, p.returncode, EXIT_GPU, p.stderr.strip()[:300]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("preflight order test: %d passed, %d failed" % (ok, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
