# bug-cli-dumponly: whirl.exe commands that do not load + tune (selftest, devices, dflash-check, vis-encode) must refuse
# --dump-only (stderr message naming the flag, exit 2 = bad command line) instead of eating it and running anyway;
# the commands that honour it (chat, bench, seqtest, golden) must behave exactly as without the flag up to the model
# load (same exit code and stderr for a missing model file). CPU only: every model path is missing, so no case loads a
# model; a refused case exits before the GPU runtime is loaded. 'devices' on an exe WITHOUT the fix lists the GPUs
# (a GPU probe), so --no-devices skips that case when running against an old exe.
#   python -I tests/cli/dump_only_test.py --bin <build\Release> [--no-devices]
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
import sys
import tempfile

EXIT_USAGE = 2
REFUSE_TEXT = "does not support --dump-only"


def run(exe, args):
    env = dict(os.environ)
    for k in ("WHIRL_DUMP_EXEC", "WHIRL_DUMP_KV", "WHIRL_ALLOC_LOG"):
        env.pop(k, None)
    p = subprocess.run([exe] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=120)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--no-devices", action="store_true")
    o = ap.parse_args()
    exe = os.path.join(o.bin, "whirl.exe")
    miss = os.path.join(tempfile.gettempdir(), "whirl_dump_only_test_no_such_model.gguf")
    if os.path.exists(miss):
        print("dump-only cli test: setup error, %s exists" % miss)
        return 1
    refuse = [
        ("selftest", ["selftest", miss, "--dump-only"]),
        ("selftest_first", ["selftest", "--dump-only", miss]),
        ("dflash-check", ["dflash-check", miss, miss, miss, "--dump-only"]),
        ("vis-encode", ["vis-encode", miss, miss, "--dump-only"]),
    ]
    if not o.no_devices:
        refuse.append(("devices", ["devices", "--dump-only"]))
    honour = [
        ("chat", ["chat", miss, "x", "--max-tokens", "1"]),
        ("bench", ["bench", miss]),
        ("seqtest", ["seqtest", miss]),
        ("golden", ["golden", miss]),
    ]
    ok = bad = 0

    def check(name, cond, detail):
        nonlocal ok, bad
        if cond:
            ok += 1
            print("ok   %s" % name)
        else:
            bad += 1
            print("FAIL %s: %s" % (name, detail))

    for name, args in refuse:
        rc, out, err = run(exe, args)
        check("refuse_" + name, rc == EXIT_USAGE and REFUSE_TEXT in err and args[0] in err and out == "",
              "exit %d, stdout %d chars, stderr: %s" % (rc, len(out), err.strip()[:300]))
    for name, args in honour:
        rc0, out0, err0 = run(exe, args)
        rc1, out1, err1 = run(exe, args + ["--dump-only"])
        check("honour_" + name, (rc1, out1, err1) == (rc0, out0, err0) and REFUSE_TEXT not in err1,
              "without flag exit %d / with flag exit %d, stderr: %s" % (rc0, rc1, err1.strip()[:300]))
    # no flag: the refused commands run as before (missing model -> exit 4, no refusal text)
    rc, out, err = run(exe, ["selftest", miss])
    check("noflag_selftest", rc == 4 and REFUSE_TEXT not in err, "exit %d, stderr: %s" % (rc, err.strip()[:300]))
    # --help still wins (help for the command, exit 0)
    rc, out, err = run(exe, ["selftest", "--dump-only", "--help"])
    check("help_wins", rc == 0 and "selftest" in out, "exit %d" % rc)
    print("dump-only cli test: %d passed, %d failed" % (ok, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
