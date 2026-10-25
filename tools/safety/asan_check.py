#!/usr/bin/env python
"""arch025-Q1: verdict of one tools/safety/run_asan.bat run (release_gate step cpu_asan) and the ASan opt-out whitelist.

  asan_check.py check <build-asan dir> <source root> [--allow tools/safety/asan_allow.txt]
  asan_check.py sites <source root> [--allow ...]          the whitelist check alone (no build needed)

PASS (exit 0, last line "asan check: ok (...)") only when ALL of these hold:
  - RUN_DONE says rc=0 and steps.log has an "ok" line for the build and for every step in STEPS (a step that did not run,
    was skipped or ended with an unexpected exit code fails);
  - every suite log has its "N passed, 0 failed" line with N >= 1 (pins: "pins: ok") and no "ERROR: AddressSanitizer";
  - every fault mode shows the checker that must catch it: the leak modes a "LEAKCHECK: ... leaked" line with >= 1 leaked,
    the memory-error modes their ASan report kind (heap-use-after-free, heap-buffer-overflow, double-free), the clean mode
    "0 leaked" and no ASan report. A fault that exits non-zero for another reason does not count as caught;
  - the source tree has no ASan opt-out (__declspec(no_sanitize_address), __attribute__((no_sanitize_address)),
    no_sanitize("address"), a __SANITIZE_ADDRESS__ guard) that is not listed in the whitelist, and every whitelist entry
    still points at such a site. MSVC ASan has no runtime suppression file (progress/arch025-Q135.PRIOR_ART.md), so this
    list IS the false-positive whitelist: one line per site, "file:line<TAB>reason" (reason required).
Exit 1 = FAIL (every problem printed), 2 = usage.
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (step name in steps.log, log file, rule)
SUITES = [
    ("whirl-device-tests", "whirl-device-tests.log", r"device tests: [1-9]\d* passed, 0 failed"),
    ("whirl-cache-tests", "whirl-cache-tests.log", r"cache tests: [1-9]\d* passed, 0 failed"),
    ("whirl-tier-tests", "whirl-tier-tests.log", r"tier tests: [1-9]\d* passed, 0 failed"),
    ("whirl-tests", "whirl-tests.log", r"unit tests: [1-9]\d* passed, 0 failed"),
    ("whirl-model-tests", "whirl-model-tests.log", r"\b[1-9]\d* passed, 0 failed"),
    ("whirl-ops-tests", "whirl-ops-tests.log", r"\b[1-9]\d* passed, 0 failed"),
    ("whirl-vision-tests", "whirl-vision-tests.log", r"\b[1-9]\d* passed, 0 failed"),
    ("whirl-server-tests", "whirl-server-tests.log", r"server tests: [1-9]\d* passed, 0 failed"),
    ("pins", "pins.log", r"(?m)^pins: ok"),
]
LEAK = re.compile(r"LEAKCHECK: (\d+) NullBackend instances checked, (\d+) leaked")
FAULTS = {  # mode -> required evidence
    "clean": ("clean", None),
    "missing_free": ("leak", None),
    "missing_host": ("leak", None),
    "missing_stream": ("leak", None),
    "uaf": ("asan", "heap-use-after-free"),
    "uaf_vector": ("asan", "heap-use-after-free"),
    "overflow": ("asan", "heap-buffer-overflow"),
    "double_free": ("asan", "double-free"),
}
STEPS = ["build"] + [s[0] for s in SUITES] + [f"fault_{m}" for m in FAULTS]
ASAN_ERR = re.compile(r"ERROR: AddressSanitizer: (?:attempting )?([\w-]+)")  # "attempting double-free on ..." (MSVC runtime text)
OPT_OUT = re.compile(r"no_sanitize_address|no_sanitize\s*\(\s*\"address\"|__SANITIZE_ADDRESS__")
SRC_DIRS = ("src", "include", "tests", "kernels", "tools")
SRC_EXT = (".cpp", ".h", ".hpp", ".inc", ".c", ".cc", ".hip")
SELF = {"tools/safety/asan_check.py", "tools/safety/test_asan_check.py"}


def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def check_run(bdir):
    probs = []
    L = os.path.join(bdir, "logs")
    rd = read(os.path.join(bdir, "RUN_DONE"))
    m = re.search(r"rc=(-?\d+)", rd or "")
    if not m:
        probs.append("RUN_DONE missing or without rc=N (run_asan.bat did not finish)")
    elif m.group(1) != "0":
        probs.append(f"RUN_DONE rc={m.group(1)} (unexpected steps)")
    rs = os.path.join(bdir, "RUN_START")
    t_start = os.path.getmtime(rs) if os.path.exists(rs) else None
    if t_start is None:
        probs.append("RUN_START missing (not a run of the current run_asan.bat): logs cannot be told from an earlier run's")

    def fresh(name, path):
        # arch025-Q135: a log left by an earlier run must not count (run 2 read a stale whirl-vision-tests.log)
        if t_start is not None and os.path.exists(path) and os.path.getmtime(path) < t_start:
            probs.append(f"{name}: log {os.path.basename(path)} is older than RUN_START (left by an earlier run)")
    st = read(os.path.join(L, "steps.log")) or ""
    seen = {}
    for line in st.splitlines():
        x = re.match(r"\[[^\]]*\]\s+(\S+) exit (\S+) expect (\S+) (ok|UNEXPECTED)", line)
        if x:
            seen[x.group(1)] = (x.group(2), x.group(3), x.group(4))
    for s in STEPS:
        if s not in seen:
            probs.append(f"step {s}: not in steps.log (did not run)")
        elif seen[s][2] != "ok":
            probs.append(f"step {s}: exit {seen[s][0]}, expected {seen[s][1]}")
    n_suites = 0
    for name, log, rule in SUITES:
        t = read(os.path.join(L, log))
        if t is None:
            probs.append(f"{name}: no log {log}")
            continue
        fresh(name, os.path.join(L, log))
        e = ASAN_ERR.findall(t)
        if e:
            probs.append(f"{name}: AddressSanitizer report(s): {', '.join(sorted(set(e)))}")
        if not re.search(rule, t):
            probs.append(f"{name}: no summary line matching {rule!r}")
        else:
            n_suites += 1
        lk = LEAK.findall(t)
        if lk and any(int(b) > 0 for _, b in lk):
            probs.append(f"{name}: leak check reports leaked NullBackend instances")
    n_faults = 0
    for mode, (kind, word) in FAULTS.items():
        t = read(os.path.join(L, f"fault_{mode}.log"))
        if t is None:
            probs.append(f"fault {mode}: no log")
            continue
        fresh(f"fault {mode}", os.path.join(L, f"fault_{mode}.log"))
        e = ASAN_ERR.findall(t)
        lk = LEAK.findall(t)
        ok = False
        if kind == "clean":
            ok = not e and bool(lk) and all(int(b) == 0 for _, b in lk)
        elif kind == "leak":
            ok = bool(lk) and any(int(b) > 0 for _, b in lk)
        elif kind == "asan":
            ok = word in e
        if ok:
            n_faults += 1
        else:
            got = (", ".join(sorted(set(e))) or "no ASan report") + ("; " + "; ".join(f"{a} checked / {b} leaked" for a, b in lk) if lk else "")
            probs.append(f"fault {mode}: expected {'0 leaked, no ASan report' if kind == 'clean' else (word or 'a leaked instance')}, got {got}")
    return probs, n_suites, n_faults


def load_allow(path):
    rows, probs = {}, []
    t = read(path)
    if t is None:
        return rows, [f"whitelist {path} missing"]
    for i, line in enumerate(t.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        m = re.match(r"^([^:\t]+):(\d+)$", parts[0].strip())
        if not m or len(parts) < 2 or len(parts[1].strip()) < 10:
            probs.append(f"whitelist line {i}: need 'file:line<TAB>reason' (reason >= 10 chars): {line!r}")
            continue
        rows[(m.group(1).replace("\\", "/"), int(m.group(2)))] = parts[1].strip()
    return rows, probs


def opt_out_sites(root):
    out = []
    for d in SRC_DIRS:
        for dp, dirs, files in os.walk(os.path.join(root, d)):
            dirs[:] = [x for x in dirs if not x.startswith(("build", ".")) and x != "__pycache__"]
            for f in files:
                if not f.endswith(SRC_EXT):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                if rel in SELF:
                    continue
                for i, line in enumerate((read(p) or "").splitlines(), 1):
                    if OPT_OUT.search(line):
                        out.append((rel, i))
    return out


def check_sites(root, allow):
    rows, probs = load_allow(allow)
    sites = opt_out_sites(root)
    for s in sites:
        if s not in rows:
            probs.append(f"ASan opt-out not in the whitelist: {s[0]}:{s[1]} (add 'file:line<TAB>reason' to {os.path.basename(allow)})")
    for s in rows:
        if s not in sites:
            probs.append(f"whitelist entry {s[0]}:{s[1]} points at no ASan opt-out (stale: fix the line or remove it)")
    return probs, len(sites)


def main(argv):
    if len(argv) < 3 or argv[1] not in ("check", "sites"):
        print(__doc__)
        return 2
    allow = None
    if "--allow" in argv:
        i = argv.index("--allow")
        allow = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if argv[1] == "sites":
        root = argv[2]
        allow = allow or os.path.join(root, "tools", "safety", "asan_allow.txt")
        probs, n = check_sites(root, allow)
        for p in probs:
            print("FAIL", p)
        print(f"asan sites: {'ok' if not probs else 'FAIL'} ({n} opt-out site(s), whitelist {allow})")
        return 1 if probs else 0
    if len(argv) < 4:
        print(__doc__)
        return 2
    bdir, root = argv[2], argv[3]
    allow = allow or os.path.join(root, "tools", "safety", "asan_allow.txt")
    probs, ns, nf = check_run(bdir)
    sp, n = check_sites(root, allow)
    probs += sp
    for p in probs:
        print("FAIL", p)
    if probs:
        print(f"asan check: FAIL ({len(probs)} problem(s); suites ok {ns}/{len(SUITES)}, faults caught {nf}/{len(FAULTS)})")
        return 1
    print(f"asan check: ok ({ns}/{len(SUITES)} suites clean, {nf}/{len(FAULTS)} fault modes caught, {n} whitelisted opt-out site(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
