# SPDX-License-Identifier: Apache-2.0
"""Fault injection (mutation check) for the modularization verification plan (audit/ARCH-1_modules.md 8.4).

For each mutation in tools/impact/mutations.json: break one module on purpose, ask the impact map which tests
cover the change, run the module's test command, and check that (a) the tests fail and (b) the impact map
listed the test that caught it. The file is always restored (also on Ctrl+C / errors).

    python tools/impact/mutate.py --cmd cache="build_and_run_cache_tests.bat" --cmd tier="..." [--only ID,...]
                                  [--tsv progress/arch1_mutations.tsv]

--cmd KEY=COMMAND: the shell command that builds and runs the tests a mutation names in "run" (exit 0 = pass).
A baseline run of every command used, with no mutation, must pass first; otherwise nothing is judged.
A "run" value may name several keys ("server,cli"): every key needs a command, each runs, and the mutant is caught when
any of them fails (a compile error counts). Built-in commands (used when no --cmd gives the key; a --cmd overrides):
  cli_preflight: build whirl-cli in build/Release (cmake --build; run from an MSVC x64 shell, as the --cmd scripts do),
                 then tests/cli/preflight_order_test.py on that whirl.exe (arch025-surv2: mutant c2-raw-early)
Mutation kinds: "replace" (find -> replace, the find text must occur exactly once) and "revert" (reverse-apply
one commit's diff of that file, e.g. re-introduce a fixed bug).
A mutation that the tests do not catch is a SURVIVOR: a test gap, reported, never hidden.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


BUILTIN_CMDS = {
    "cli_preflight": (f'cmake --build "{os.path.join(ROOT, "build", "Release")}" --target whirl-cli && '
                      f'"{sys.executable}" -I "{os.path.join(ROOT, "tests", "cli", "preflight_order_test.py")}" '
                      f'--bin "{os.path.join(ROOT, "build", "Release")}"'),
}


def run_keys(m):
    return [k.strip() for k in m["run"].split(",") if k.strip()]


def sh(cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or ""), time.time() - t0


def apply(m):
    path = os.path.join(ROOT, m["file"])
    orig = open(path, "rb").read()
    if m["kind"] == "replace":
        t = orig.decode("utf-8")
        if t.count(m["find"]) != 1:
            raise RuntimeError(f"{m['id']}: find text occurs {t.count(m['find'])} times in {m['file']} (want 1)")
        open(path, "wb").write(t.replace(m["find"], m["replace"]).encode("utf-8"))
    elif m["kind"] == "revert":
        diff = subprocess.run(["git", "-C", ROOT, "show", "--format=", m["commit"], "--", m["file"]], capture_output=True).stdout
        r = subprocess.run(["git", "-C", ROOT, "apply", "-R", "--whitespace=nowarn", "-"], input=diff, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"{m['id']}: cannot reverse-apply {m['commit']} to {m['file']}: {r.stderr.decode(errors='replace')[:200]}")
    else:
        raise RuntimeError(f"{m['id']}: unknown kind {m['kind']}")
    return path, orig


def impact_lists(expect):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "impact", "impact.py"), "--worktree", "--json"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        rep = list(json.loads(r.stdout).values())[0]
    except (ValueError, IndexError):
        return False, "impact.py failed"
    exe, _, name = expect.partition(":")
    # "release:<step>": a GPU mutation caught by a release_gate step (e.g. release:srv_tier_verify)
    listed = name in rep["release_gate_steps"] if exe == "release" else (name in rep["cpu_tests"].get(exe, []) or name in rep["kernel_tests"])
    return listed, f"{len(rep['paths'])} path(s): {', '.join(rep['paths'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", action="append", default=[], help="KEY=COMMAND")
    ap.add_argument("--only")
    ap.add_argument("--spec", default=os.path.join(ROOT, "tools", "impact", "mutations.json"))
    ap.add_argument("--tsv")
    a = ap.parse_args()
    cmds = dict(BUILTIN_CMDS)
    cmds.update(dict(c.split("=", 1) for c in a.cmd))
    muts = json.load(open(a.spec, encoding="utf-8"))["mutations"]
    if a.only:
        keep = set(a.only.split(","))
        muts = [m for m in muts if m["id"] in keep]
    muts = [m for m in muts if all(k in cmds for k in run_keys(m))]
    if not muts:
        print("mutate: no mutation has a --cmd for its 'run' key")
        return 2
    if subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "--", *{m["file"] for m in muts}]).returncode != 0:
        print("mutate: the files to mutate have uncommitted changes; commit or stash them first")
        return 2
    for k in sorted({k for m in muts for k in run_keys(m)}):
        rc, out, sec = sh(cmds[k])
        print(f"baseline {k}: exit {rc} ({sec:.0f} s)")
        if rc != 0:
            print("\n".join(out.splitlines()[-8:]))
            print("mutate: baseline fails: nothing can be judged")
            return 2
    rows = []
    for m in muts:
        path, orig = apply(m)
        try:
            listed, ipaths = impact_lists(m["expect_test"])
            rc, out, sec = 0, "", 0.0
            for k in run_keys(m):
                krc, kout, ksec = sh(cmds[k])
                out += f"[{k}] exit {krc}\n" + kout
                sec += ksec
                rc = rc or krc
        finally:
            open(path, "wb").write(orig)
        caught = rc != 0
        fails = [l.strip() for l in out.splitlines() if "FAIL" in l or "failed" in l][:3]
        verdict = "CAUGHT" if caught else "SURVIVED"
        print(f"{verdict:8} {m['id']} ({m['module']}): exit {rc} in {sec:.0f} s; impact map lists {m['expect_test']}: "
              f"{'yes' if listed else 'NO'} [{ipaths}]")
        for l in fails:
            print(f"         {l[:160]}")
        rows.append([datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"), m["id"], m["module"], m["what"], verdict, m["expect_test"],
                     "listed" if listed else "NOT listed", f"{sec:.0f}", (fails[0] if fails else "")[:120]])
    if a.tsv:
        new = not os.path.exists(a.tsv)
        with open(a.tsv, "a", encoding="utf-8", newline="\n") as f:
            if new:
                f.write("ts\tid\tmodule\tmutation\tverdict\texpected_test\timpact_map\tseconds\tfirst_failure\n")
            for r in rows:
                f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
    s = sum(1 for r in rows if r[4] == "SURVIVED")
    nl = sum(1 for r in rows if r[6] != "listed")
    print(f"mutate: {len(rows) - s}/{len(rows)} caught, {s} survived; impact map missed the expected test in {nl}")
    return 1 if s or nl else 0


if __name__ == "__main__":
    sys.exit(main())
