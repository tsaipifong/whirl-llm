#!/usr/bin/env python
"""safe025: MSVC /analyze baseline gate - only NEW warnings fail, and only a complete full-rebuild run can pass.

  analyze_check.py prepare <build_dir> <target>...   before the build: list every cl compile of the targets (ninja -t commands),
                                                      delete those .obj / .sarif files so every TU is compiled again, write the
                                                      start marker and the expected list (analyze_expected.txt)
  analyze_check.py check   <build_dir> <baseline.txt> exit 1 on an incomplete run or on a key that occurs more often than in the baseline
  analyze_check.py update  <build_dir> <baseline.txt> write the baseline (sorted "count<TAB>key" lines); refuses an incomplete run
  analyze_check.py list    <build_dir>                print the keys
  analyze_check.py reasons <baseline.txt> <whitelist.tsv> [<build_dir>]
                                                      arch025-Q3: every baseline key has one whitelist row "key<TAB>file:line<TAB>reason"
                                                      (reason >= 10 chars, file:line in the key's file), no row for a key outside the
                                                      baseline; with <build_dir> (a complete run) no whitelisted key may occur more
                                                      often than its baseline count; the row's line is only a hint: a moved warning
                                                      prints "INFO line drift" with its new line and does not fail
  analyze_check.py faultproof <out_dir>               arch025-Q3: verdict of tools/safety/analyze_fault.bat (real cl, 3 cases): the
                                                      baseline-only TU PASSES, 3 planted keys FAIL as NEW, a 2nd warning of a
                                                      baselined key FAILS as NEW x1

Input is the SARIF 2.1.0 file cl writes next to each object with /analyze:log:format:sarif (<obj>.nativecodeanalysis.sarif), not the
console text: the console text follows the installed MSVC language pack (e.g. zh-TW in code page 950; VSLANG=1033 does nothing
without the 1033 resources) and carries absolute paths, line numbers and stack byte counts.
A key is  <file relative to the source tree>|<ruleId>|<function>  (SARIF ruleId and logicalLocations[0].fullyQualifiedName), so it
does not depend on the language, the tree path, the line, or the numbers in the message. A header warning reported by several TUs is
counted once per (key, line, column); the same key appearing more often than in the baseline is new.

A run is complete only when (safe025 review A3):
  - <build_dir>/DONE says rc=0 (the build finished without error; run_analyze.bat writes it),
  - <build_dir>/analyze_start exists (prepare ran, so the old objects were deleted), and
  - every object in analyze_expected.txt exists, is newer than the start marker, and has its SARIF file, also newer.
An incremental build that skipped a TU, a failed or partial build, or a plain ninja run without prepare all FAIL.
"""
import collections
import json
import os
import re
import subprocess
import sys
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

START = "analyze_start"
EXPECTED = "analyze_expected.txt"
CODES = re.compile(r"^C(?:6\d{3}|26\d{3}|28\d{3})$")  # the code-analysis ranges (the /W4 warnings are not part of this gate)


def norm(p):
    return os.path.normcase(os.path.normpath(p)).replace("\\", "/")


def cache_var(build_dir, name):
    try:
        with open(os.path.join(build_dir, "CMakeCache.txt"), encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(name + ":"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def sarif_of(obj):
    base = obj[:-4] if obj.lower().endswith(".obj") else obj
    return base + ".nativecodeanalysis.sarif"


def analyze_on(cmd):
    """True when a cl command line compiles with /analyze: the last plain /analyze or /analyze- wins (arch025-Q3: the bin2c kernel
    byte arrays get /analyze- from cmake/safety.cmake and write no SARIF, so they are not expected)"""
    flags = re.findall(r"(?:^|\s)[/-]analyze(-?)(?=\s|$)", cmd)
    return bool(flags) and flags[-1] == ""


def expected_objects(build_dir, targets):
    ninja = cache_var(build_dir, "CMAKE_MAKE_PROGRAM") or "ninja"
    # arch025-Q135 run 2: a changed cmake file re-runs CMake only inside the build, AFTER prepare had read the old build.ninja
    # (4 kernel byte-array TUs expected with /analyze although the new build.ninja says /analyze-). Bring build.ninja up to date first.
    regen = subprocess.run([ninja, "-C", build_dir, "build.ninja"], capture_output=True, text=True, errors="replace")
    if regen.returncode != 0:
        raise SystemExit(f"prepare: ninja build.ninja (CMake re-run) failed ({regen.returncode}): {(regen.stdout + regen.stderr).strip()[-400:]}")
    out = subprocess.run([ninja, "-C", build_dir, "-t", "commands", *targets], capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"prepare: ninja -t commands failed ({out.returncode}): {out.stderr.strip()[:400]}")
    objs = []
    for line in out.stdout.splitlines():
        if "cl.exe" not in line.lower() or not analyze_on(line):
            continue
        m = re.search(r'/Fo"?([^"\s]+\.obj)"?', line)
        if m:
            objs.append(m.group(1))
    return sorted(set(objs))


def prepare(build_dir, targets):
    objs = expected_objects(build_dir, targets)
    if not objs:
        raise SystemExit("prepare: no /analyze compile in the targets (is WHIRL_ANALYZE=ON in this build directory?)")
    for o in objs:
        for p in (o, sarif_of(o)):
            fp = os.path.join(build_dir, p)
            if os.path.exists(fp):
                os.remove(fp)
    for name in ("DONE",):
        fp = os.path.join(build_dir, name)
        if os.path.exists(fp):
            os.remove(fp)
    with open(os.path.join(build_dir, EXPECTED), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(objs) + "\n")
    with open(os.path.join(build_dir, START), "w", encoding="utf-8") as f:
        f.write("targets " + " ".join(targets) + "\n")
    print(f"prepare: {len(objs)} TUs will be compiled with /analyze; objects and SARIF files removed")
    return 0


def completeness(build_dir):
    """Return (problems, sarif paths)."""
    probs = []
    done = os.path.join(build_dir, "DONE")
    rc = None
    if os.path.exists(done):
        with open(done, encoding="utf-8", errors="replace") as f:
            m = re.search(r"rc=(-?\d+)", f.read())
        rc = int(m.group(1)) if m else None
    if rc is None:
        probs.append("DONE missing or without rc=N (build not finished, or not run by run_analyze.bat)")
    elif rc != 0:
        probs.append(f"build failed: DONE rc={rc}")
    start = os.path.join(build_dir, START)
    exp = os.path.join(build_dir, EXPECTED)
    if not os.path.exists(start) or not os.path.exists(exp):
        probs.append("no analyze_start / analyze_expected.txt: not a full rebuild prepared by 'analyze_check.py prepare'")
        return probs, []
    t0 = os.path.getmtime(start)
    with open(exp, encoding="utf-8") as f:
        objs = [l.strip() for l in f if l.strip()]
    if not objs:
        probs.append("analyze_expected.txt is empty")
    missing, stale, nosarif = [], [], []
    sarifs = []
    for o in objs:
        fo = os.path.join(build_dir, o)
        fs = sarif_of(fo)
        if not os.path.exists(fo):
            missing.append(o)
            continue
        if os.path.getmtime(fo) < t0:
            stale.append(o)
        if not os.path.exists(fs) or os.path.getmtime(fs) < t0:
            nosarif.append(o)
            continue
        sarifs.append(fs)
    for what, lst in (("objects missing (TU not compiled)", missing), ("objects older than the start marker (not recompiled)", stale),
                      ("SARIF missing or stale", nosarif)):
        if lst:
            probs.append(f"{len(lst)} of {len(objs)} {what}: " + ", ".join(lst[:5]) + (" ..." if len(lst) > 5 else ""))
    return probs, sarifs


def keys(build_dir, sarifs, lines=None):
    src = norm(cache_var(build_dir, "CMAKE_HOME_DIRECTORY") or "")
    bld = norm(os.path.abspath(build_dir))
    out = collections.Counter()
    seen = set()
    for fs in sarifs:
        with open(fs, encoding="utf-8-sig", errors="replace") as f:
            d = json.load(f)
        for run in d.get("runs", []):
            arts = [urllib.parse.unquote(a.get("location", {}).get("uri", "")) for a in run.get("artifacts", [])]
            for r in run.get("results", []):
                rule = r.get("ruleId", "")
                if not CODES.match(rule) or r.get("suppressions"):
                    continue
                loc = (r.get("locations") or [{}])[0]
                phys = loc.get("physicalLocation", {})
                al = phys.get("artifactLocation", {})
                uri = al.get("uri") or (arts[al["index"]] if "index" in al and al["index"] < len(arts) else "")
                path = norm(re.sub(r"^file:/*", "", urllib.parse.unquote(uri)))
                if bld and path.startswith(bld + "/"):
                    rel = "<build>/" + path[len(bld) + 1:]
                elif src and path.startswith(src + "/"):
                    rel = path[len(src) + 1:]
                else:
                    continue  # SDK / STL / other trees are not ours
                logical = loc.get("logicalLocations") or [{}]
                fn = logical[0].get("fullyQualifiedName") or logical[0].get("decoratedName") or "-"
                key = f"{rel}|{rule}|{fn}"
                reg = phys.get("region", {})
                dedup = (key, reg.get("startLine"), reg.get("startColumn"))
                if dedup in seen:
                    continue
                seen.add(dedup)
                out[key] += 1
                if lines is not None:
                    lines.setdefault(key, set()).add(reg.get("startLine"))
    return out


def load_baseline(path):
    c = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            n, k = line.split("\t", 1)
            c[k] = int(n)
    return c


def reasons(baseline, whitelist, build_dir=None):
    """arch025-Q3: the baseline's false-positive / accepted-warning whitelist (file:line + reason per key)."""
    probs = []
    base = load_baseline(baseline)
    rows = {}
    try:
        with open(whitelist, encoding="utf-8") as f:
            text = f.read().splitlines()
    except OSError:
        text = None
        probs.append(f"whitelist {whitelist} missing")
    for i, line in enumerate(text or [], 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            probs.append(f"whitelist line {i}: need key<TAB>file:line<TAB>reason, got {len(parts)} column(s)")
            continue
        k, fl, why = (x.strip() for x in parts)
        m = re.match(r"^(.+):(\d+)$", fl)
        if k in rows:
            probs.append(f"whitelist line {i}: key listed twice: {k}")
        elif not m or m.group(1) != k.split("|", 1)[0]:
            probs.append(f"whitelist line {i}: file:line {fl!r} is not in the key's file {k.split('|', 1)[0]}")
        elif len(why) < 10 or why.upper().startswith("TODO"):
            probs.append(f"whitelist line {i}: no reason for {k}")
        rows[k] = (int(m.group(2)) if m else None, why)
    for k in sorted(base):
        if k not in rows:
            probs.append(f"baseline key without a whitelist reason: {k}")
    for k in sorted(rows):
        if k not in base:
            probs.append(f"whitelist row for a key not in the baseline (stale): {k}")
    # arch025-Q3 (coordinator 2026-10-26): the verdict keys on file|rule|function and the count, never on the line. Unrelated code
    # above a warning moves its line (server_tests.cpp 4531/6845/7009 -> 4600/6914/7078 on arch025-next with no new warning), and a
    # gate that fails on that breaks on every merge. The whitelist line is a hint for the reader: drift is printed as INFO.
    checked, drift = 0, []
    if build_dir:
        cp, sarifs = completeness(build_dir)
        if cp:
            probs += [f"build dir incomplete: {p}" for p in cp]
        else:
            lines = {}
            cur = keys(build_dir, sarifs, lines)
            for k in sorted(cur):
                if k not in rows:
                    probs.append(f"{k}: cl reports it {cur[k]} time(s) and the whitelist has no reason for it (new key, e.g. the "
                                 "warning is now in another function)")
            for k, (ln, _) in sorted(rows.items()):
                if k not in base:
                    continue  # already reported as a stale row
                n = cur.get(k, 0)
                if n > base[k]:
                    probs.append(f"{k}: cl reports it {n} time(s), the baseline accepts {base[k]} (a new occurrence has no reason)")
                now = sorted(x for x in lines.get(k, set()) if x is not None)
                if ln is not None and now and ln not in now:
                    drift.append(f"{k}: whitelist line {ln}, cl reports line(s) {', '.join(map(str, now))} now (hint only; "
                                 "refresh the row's line when convenient)")
                checked += 1
    for p in probs:
        print("FAIL", p)
    for d in drift:
        print("INFO line drift:", d)
    print(f"analyze reasons: {'ok' if not probs else 'FAIL'} ({len(rows)} row(s), {len(base)} baseline key(s)"
          + (f", {checked} key(s) checked against the SARIF of {build_dir}, {len(drift)} line drift(s) (info)" if build_dir
             else ", not checked against a build (no build dir)") + ")")
    return 1 if probs else 0


FAULT_FILE = "tests/safety/analyze_fault.cpp"
FAULT_ACCEPTED = FAULT_FILE + "|C6001|acceptedOld"   # the one key of tests/safety/analyze_fault_baseline.txt
FAULT_NEW_MIN = 3                                       # usesUninitialized, derefsNull, overrun


def faultproof(out_dir):
    """arch025-Q3: verdict of tools/safety/analyze_fault.bat (real cl /analyze, three cases). Both directions must hold:
    clean   (only the baselined warning)            -> gate rc=0, "analyze gate: PASS", the accepted key present, nothing NEW
    new     (+3 planted defects, 3 new keys)        -> gate rc=1, >= 3 "NEW x1: tests/safety/analyze_fault.cpp|..." lines, the
                                                       accepted key not among them
    samekey (2nd C6001 in the baselined function)   -> gate rc=1, exactly "NEW x1: <accepted key>" (review C3: same key, count 2 > 1)
    A case whose compile failed (DONE rc != 0) or whose gate did not run is a FAIL of the proof, never a pass."""
    probs = []
    res = {}
    for case in ("clean", "new", "samekey"):
        d = os.path.join(out_dir, case)
        log = os.path.join(d, "gate.log")
        rcs = {}
        for name in ("DONE", "GATE"):
            try:
                with open(os.path.join(d, name), encoding="utf-8", errors="replace") as f:
                    m = re.search(r"rc=(-?\d+)", f.read())
                rcs[name] = int(m.group(1)) if m else None
            except OSError:
                rcs[name] = None
        try:
            with open(log, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        if rcs["DONE"] != 0:
            probs.append(f"{case}: compile DONE rc={rcs['DONE']} (see {case}\\analyze_fault.log)")
        if not text:
            probs.append(f"{case}: gate.log missing or empty")
        new = re.findall(r"(?m)^NEW x(\d+): (\S+)$", text)
        cur = {}
        try:
            cp, sarifs = completeness(d)
            if not cp:
                cur = keys(d, sarifs)
        except (OSError, ValueError) as e:
            probs.append(f"{case}: SARIF unreadable: {e}")
        res[case] = (rcs["GATE"], new, cur, text)
    rc, new, cur, text = res["clean"]
    if rc != 0 or "analyze gate: PASS" not in text or new:
        probs.append(f"clean: the baseline-only TU must PASS (GATE rc={rc}, NEW {len(new)})")
    if cur.get(FAULT_ACCEPTED, 0) != 1:
        probs.append(f"clean: the accepted warning {FAULT_ACCEPTED} must be reported once by cl, got {cur.get(FAULT_ACCEPTED, 0)} "
                     "(otherwise the PASS proves nothing)")
    rc, new, cur, text = res["new"]
    planted = [k for n, k in new if k.startswith(FAULT_FILE + "|") and k != FAULT_ACCEPTED]
    if rc != 1 or "analyze gate: FAIL" not in text or len(planted) < FAULT_NEW_MIN:
        probs.append(f"new: the planted warnings must FAIL as NEW (GATE rc={rc}, planted NEW {len(planted)} < {FAULT_NEW_MIN}?)")
    if any(k == FAULT_ACCEPTED for _, k in new):
        probs.append("new: the accepted key was reported NEW although its count did not grow")
    rc, new, cur, text = res["samekey"]
    if rc != 1 or "analyze gate: FAIL" not in text or new != [("1", FAULT_ACCEPTED)]:
        probs.append(f"samekey: a 2nd warning of a baselined key must FAIL as 'NEW x1: {FAULT_ACCEPTED}' only "
                     f"(GATE rc={rc}, NEW {new}, cl count {cur.get(FAULT_ACCEPTED, 0)})")
    for p in probs:
        print("FAIL", p)
    n_new = len([k for n, k in res["new"][1] if k.startswith(FAULT_FILE + "|")])
    print(f"analyze fault proof: {'ok' if not probs else 'FAIL'} (clean GATE rc={res['clean'][0]}; new GATE rc={res['new'][0]}, "
          f"{n_new} NEW; samekey GATE rc={res['samekey'][0]}, NEW {['x' + n + ' ' + k for n, k in res['samekey'][1]]})")
    return 1 if probs else 0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, bdir = argv[1], argv[2]
    if cmd == "faultproof":
        return faultproof(bdir)
    if cmd == "reasons":
        if len(argv) < 4:
            print(__doc__)
            return 2
        return reasons(argv[2], argv[3], argv[4] if len(argv) > 4 else None)
    if cmd == "prepare":
        return prepare(bdir, argv[3:])
    probs, sarifs = completeness(bdir)
    if cmd == "list":
        for p in probs:
            print("INCOMPLETE:", p)
        for k, n in sorted(keys(bdir, sarifs).items()):
            print(f"{n}\t{k}")
        return 0
    if cmd in ("check", "update") and len(argv) < 4:
        print(__doc__)
        return 2
    if probs:
        for p in probs:
            print("INCOMPLETE:", p)
        print("analyze gate: FAIL (incomplete run)" if cmd == "check" else "baseline NOT written (incomplete run)")
        return 1
    cur = keys(bdir, sarifs)
    if cmd == "update":
        with open(argv[3], "w", encoding="utf-8", newline="\n") as f:
            f.write("# safe025 /analyze baseline: count<TAB>file|ruleId|function (from SARIF; no paths, lines or message text). "
                    "Regenerate: tools/safety/run_analyze.bat, then analyze_check.py update <build_dir> <this file>\n")
            for k, n in sorted(cur.items()):
                f.write(f"{n}\t{k}\n")
        print(f"baseline written: {sum(cur.values())} warnings, {len(cur)} keys, {len(sarifs)} TUs -> {argv[3]}")
        return 0
    if cmd == "check":
        base = load_baseline(argv[3])
        new = [(k, n - base.get(k, 0)) for k, n in sorted(cur.items()) if n > base.get(k, 0)]
        gone = [k for k in base if cur.get(k, 0) < base[k]]
        print(f"analyze: {len(sarifs)} TUs, {sum(cur.values())} warnings in {len(cur)} keys; baseline {sum(base.values())} in {len(base)} keys")
        for k in gone:
            print(f"fixed since baseline (refresh the baseline with 'update'): {k}")
        for k, d in new:
            print(f"NEW x{d}: {k}")
        print("analyze gate:", "FAIL (new warnings)" if new else "PASS")
        return 1 if new else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
