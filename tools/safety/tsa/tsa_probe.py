# SPDX-License-Identifier: Apache-2.0
"""cpu_tsa probe for the Engine split S10 (arch025-engine S0, review2 R11): compiles tools/safety/tsa/tsa_probe_jq.cpp
with clang-cl (-fsyntax-only -Wthread-safety -Werror=thread-safety, the include/whirl/thread_annotations.h wrapper) in
each variant and checks the expected outcome. The S10 design uses the variant this probe shows to work.

  python -I tools/safety/tsa/tsa_probe.py        exit 0 "tsa probe: ok (...)", 1 an outcome differs, 2 no clang-cl
clang-cl: WHIRL_CLANG_CL, else %HIP_PATH%\\bin\\clang-cl.exe, else clang-cl on PATH (same rule as run_tsa_build.bat).
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "tsa_probe_jq.cpp")
# (name, defines, expect: "ok" | "tsa" (thread-safety error) | "access" (private member error))
CASES = [
    ("private mu, WH_REQUIRES(jq_.mu)", ["V_PRIVATE_MEMBER"], "access"),
    ("public mu, WH_REQUIRES(jq_.mu), lock held", ["V_PUBLIC_MEMBER"], "ok"),
    ("public mu, WH_REQUIRES(jq_.mu), lock NOT held", ["V_PUBLIC_MEMBER", "FAULT"], "tsa"),
    ("private mu + accessor WH_RETURN_CAPABILITY, WH_REQUIRES(jq_.mutex()), lock held", ["V_ACCESSOR"], "ok"),
    ("private mu + accessor, lock NOT held", ["V_ACCESSOR", "FAULT"], "tsa"),
]


def clang():
    c = os.environ.get("WHIRL_CLANG_CL")
    if c and os.path.exists(c): return c
    hp = os.environ.get("HIP_PATH")
    if hp and os.path.exists(os.path.join(hp, "bin", "clang-cl.exe")): return os.path.join(hp, "bin", "clang-cl.exe")
    return shutil.which("clang-cl")


def main():
    cc = clang()
    if not cc:
        print("tsa probe: SKIP (no clang-cl)"); return 2
    bad = 0
    for name, defs, want in CASES:
        cmd = [cc, "/std:c++20", "/EHsc", "/nologo", "/I" + os.path.join(REPO, "include"), "/clang:-fsyntax-only",
               "/clang:-Wthread-safety", "/clang:-Werror=thread-safety"] + ["/D" + d for d in defs] + [SRC]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode == 0: got = "ok"
        elif "private member" in out or "is a private member" in out or "inaccessible" in out: got = "access"
        elif "thread-safety" in out or "requires holding" in out: got = "tsa"
        else: got = "other"
        good = got == want
        bad += not good
        first = next((l.strip() for l in out.splitlines() if "error" in l), "")
        print(f"  {'ok  ' if good else 'FAIL'} {name}: expected {want}, got {got}" + (f"  [{first[:160]}]" if first else ""))
    print(f"tsa probe: {'ok' if not bad else 'FAIL'} ({len(CASES) - bad}/{len(CASES)} as expected; {os.path.basename(cc)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
