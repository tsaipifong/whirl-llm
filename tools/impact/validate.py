# SPDX-License-Identifier: Apache-2.0
"""Validates the impact map on real bug-fix commits: does it flag the test / gate that caught each bug?

    python tools/impact/validate.py            named cases (today's bugs) + history replay
    python tools/impact/validate.py --cases    named cases only
    python tools/impact/validate.py --history N [--rev REV] [--skip K]   replay N commits of REV (default HEAD), skipping the
                                                newest K (hold-out: the commits the tools were not tuned on)
    python tools/impact/validate.py --baseline REV2 [--history N --rev REV --skip K]   also replay REV2's tools/impact on
                                                the same commits and list every commit the current tools select less for

Named cases: a fix commit (product code only: its test changes are left out of the diff, so the map has to
find the test from the code), the test that reproduces the bug (the repro commit's touched test functions)
and the gate that caught it in practice. The map used is built at the fix commit (git archive of that revision
into a temp folder, then impact_build.py), so tests that exist only on that branch are known to it.
History replay: every commit that changes product code AND a registered test function in the same commit; the
touched test functions are taken as "the test for this change". Hit = the map lists that test for the product
part of the diff. Executable hit = it lists at least that test executable.
The history hit rate counts CPU tests only (a test touched in the same commit). It says nothing about the GPU steps:
those are reported separately (commits with / without a GPU release step) and checked against a baseline with
--baseline (narrowed commits). Cases marked "lock" pin a fix the tools were tuned on (regression lock, not evidence).
Exit (arch024-S12, review C7): 0 every named case hit, every history hit rate >= 95 %, no narrowed commit; 1 otherwise;
2 setup error.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cxxscan  # noqa: E402
import impact  # noqa: E402
import impact_build  # noqa: E402

ROOT = impact.ROOT

# fix commits, repro commits, what caught it (evidence), what the map must list
CASES = [
    {"name": "COLD-RESUME (precise not bitwise after a cache restore, since 0.2.0)",
     "fix": ["99f69a1", "754b58b"], "repro": [], "map_at": "754b58b",
     "caught_by": "release gate srv_cache_balance 'tier p1: T turn 2 after restart (SSD restore)' (progress/rel021.log 2026-10-19 20:18)",
     "expect": ["release:srv_cache_balance", "release:srv_cache_precise", "srvgate:tier", "golden:server"]},
    {"name": "Fingerprint included a diagnostic variable (WHIRL_PREFIX_CACHE_VERIFY)",
     "fix": ["7ec63b4"], "repro": ["a73a0ce"], "map_at": "7ec63b4",
     "caught_by": "CACHEBAL review; repro whirl-cache-tests (a73a0ce)",
     "expect": ["release:cpu_cache", "srvgate:tier"]},
    {"name": "KV tier dedup race (finished, not yet reaped base write)",
     "fix": ["bf31527"], "repro": ["3fd8af8"], "map_at": "bf31527",
     "caught_by": "release gate srv_cache_* flake (CB-FLAKE); repro whirl-tier-tests (3fd8af8)",
     "expect": ["release:cpu_tier", "release:srv_cache_balance", "srvgate:restore_conc"]},
    {"name": "HTTP: unanswered connect after server-side TIME_WAIT (HTTP-DROP)",
     "fix": ["fc80ed4"], "repro": ["d25d41e"], "map_at": "fc80ed4",
     "caught_by": "server gate http_stress loop (progress/httpdrop.log 2026-10-20 10:42); repro whirl-server-tests http_port_reuse (d25d41e)",
     "expect": ["release:cpu_server", "srvgate:basic"]},
    {"name": "rocm_kpack.dll missing from the package (issue #2, ROCm 10.1 driver)",
     "fix": ["2d2621e", "0f3ee97"], "repro": [], "map_at": "0f3ee97",
     "caught_by": "a user (GitHub issue #2); no gate existed",
     "expect": ["extra:package smoke", "test:whirl-tests:testDllDeps"]},
    {"name": "lock: struct member added before any reader (PinnedGpu::adapter_shared, B0-1 C3 test first)", "lock": True,
     "fix": ["a4626b9"], "repro": [], "map_at": "a4626b9",
     "caught_by": "S12 history replay miss; review A1 (the struct fallback must add to the whole-file fallback, not replace it)",
     "expect": ["test:whirl-tests:testB0SizingFollowUps", "release:srv_cache_balance", "release:srv_cache_precise", "release:golden_swift"]},
    {"name": "lock: Watch::check -> testWddmWatch edge ('check' is not a stop name; fan-out after the class filter)", "lock": True,
     "fix": ["392874b"], "repro": [], "map_at": "392874b",
     "caught_by": "S12 history replay miss; review C2 (fan-out margin)",
     "expect": ["test:whirl-tests:testWddmWatch"]},
    {"name": "Server gate asserted the wrong thing for precise (history reuse instead of no gen-end resume)",
     "fix": ["56dea43"], "repro": [], "map_at": "56dea43", "keep_tests": True,
     "caught_by": "COLD-RESUME review: the gate passed on wrong behaviour",
     "expect": ["extra:cpu_gate_selftest", "extra:the changed suite on one GPU"]},
]


def git(*a):
    return impact.git(*a)


def map_at(rev, cache={}):
    """Impact map of revision rev (git archive -> temp dir -> impact_build)."""
    if rev in cache:
        return cache[rev]
    data = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", rev, "src", "include", "kernels", "tests", "tools", "cmake",
                           "CMakeLists.txt"], capture_output=True).stdout
    d = tempfile.mkdtemp(prefix="impact_at_")
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(d, filter="data")
    m = impact_build.build(d)
    m["source_commit"] = rev
    shutil.rmtree(d, ignore_errors=True)  # the map is all that is used; one tree per replayed commit filled %TEMP%
    cache[rev] = (m, None)
    return cache[rev]


def touched_tests(rev, with_file=False):
    """registered-test-style functions (test*) touched by commit rev in tests/ files ({name} or, with_file, {(file, name)})"""
    out = set()
    ch = impact.parse_diff(git("diff", "-U0", "--no-color", f"{rev}^", rev))
    for path, c in ch.items():
        if not (path.startswith("tests/") and path.endswith(".cpp")) or not c["new"]:
            continue
        src = impact.show(rev, path)
        if src is None:
            continue
        sc = cxxscan.scan(src)
        for ln in c["new"]:
            f = cxxscan.func_at(sc, ln)
            if f and re.match(r"(test|bench)[A-Z]", f.short):
                out.add((path, f.short) if with_file else f.short)
    return out


def product_only(changes, keep_tests=False):
    if keep_tests:
        return changes
    return {p: c for p, c in changes.items() if not p.startswith("tests/")}


def flagged(report, item):
    kind, _, val = item.partition(":")
    if kind == "release":
        return val in report["release_gate_steps"]
    if kind == "srvgate":
        return val in report["server_gate_suites"]
    if kind == "golden":
        return any(f"suites" in g and val in g.split("suites ")[1].split(",") for g in report["golden"])
    if kind == "extra":
        return any(k.startswith(val) for k in report["extra"])
    if kind == "test":
        exe, name = val.split(":", 1)
        return name in report["cpu_tests"].get(exe, []) or name in report["kernel_tests"]
    return False


def run_cases():
    res = []
    for c in CASES:
        m, _ = map_at(c["map_at"])
        changes = {}
        for f in c["fix"]:
            ch = impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{f}^", f))
            changes.update(product_only(ch, c.get("keep_tests", False)))
        # analyze each fix commit against its own revision, merge the reports
        reports = []
        for f in c["fix"]:
            ch = product_only(impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{f}^", f)), c.get("keep_tests", False))
            reports.append(impact.analyze(ch, m, f, f"{f}^"))
        rep = merge(reports)
        exp = list(c["expect"])
        for r in c["repro"]:
            for t in sorted(touched_tests(r)):
                exe = [e for e, fl in m["executables"].items() if any(fl_.startswith("tests/") for fl_ in fl)
                       and f"{e}:{t}" in m["tests"]]
                for e in exe:
                    exp.append(f"test:{e}:{t}")
        rows = [(e, flagged(rep, e)) for e in exp]
        res.append((c, rep, rows))
        print(f"== {c['name']}")
        print(f"   fix {', '.join(c['fix'])}; map built at {c['map_at']}; caught by: {c['caught_by']}")
        print(f"   map: paths {', '.join(rep['paths'])}; release steps {', '.join(rep['release_gate_steps']) or '-'}")
        if rep["not_in_release_gate"]:
            print(f"   map: suites with no release step: {', '.join(rep['not_in_release_gate'])}")
        for e, ok in rows:
            print(f"   {'HIT ' if ok else 'MISS'} {e}")
        print("   " + selectivity(rep, m))
    return res


def selectivity(rep, m):
    """how much of the test / gate set the map asks for (a map that flags everything would hit everything)"""
    ncpu = sum(len(v) for v in rep["cpu_tests"].values())
    tcpu = sum(1 for t, d in m["tests"].items() if not d["gpu"])
    nrel = len([s for s in rep["release_gate_steps"]])
    trel = len(m["release_srv_steps"]) + 12 + 3 * 3 + 2 + 5 + 2  # srv steps + cpu/kt + st (2 models x 3 modes) + golden + seq + dfl approx
    nsu = len(rep["server_gate_suites"])
    return (f"selectivity: CPU tests {ncpu}/{tcpu}, release steps {nrel}/~{trel}, server-gate suites {nsu}/{len(m['declared']['gate_suites'])}, "
            f"paths {len(rep['paths'])}/{len(m['declared']['paths'])}")


def merge(reports):
    out = {"paths": {}, "release_gate_steps": [], "server_gate_suites": [], "golden": [], "extra": {}, "cpu_tests": {},
           "kernel_tests": [], "not_in_release_gate": []}
    for r in reports:
        out["paths"].update(r["paths"])
        for k in ("release_gate_steps", "server_gate_suites", "golden", "kernel_tests", "not_in_release_gate"):
            out[k] = sorted(set(out[k]) | set(r[k]))
        for k, v in r["extra"].items():
            out["extra"][k] = sorted(set(out["extra"].get(k, [])) | set(v))
        for k, v in r["cpu_tests"].items():
            out["cpu_tests"][k] = sorted(set(out["cpu_tests"].get(k, [])) | set(v))
    return out


def gpu_steps(rep):
    return sorted(x for x in impact.expand_steps(rep["release_gate_steps"]) if not x.startswith("cpu_"))


def pair(m, tt):
    """touched (file, test) -> registered test ids of the executables that link the file; and the unpaired ones"""
    regs, unp = {}, []
    for path, t in tt:
        got = False
        for exe, srcs in m["executables"].items():
            tid = f"{exe}:{t}"
            if path in srcs and tid in m["tests"]:
                regs[tid] = m["tests"][tid]["exe"]
                got = True
        if not got:
            unp.append((path, t))
    return regs, unp


def run_history(n, rev, skip=0, at_commit=False):
    m_head = impact_build.build(ROOT) if rev == "HEAD" else map_at(rev)[0]
    shas = git("log", "--format=%h", "--no-merges", f"--skip={skip}", f"-{n}", rev).split()
    tot = fhit = ehit = 0
    misses, unpaired, at_rev = [], [], []
    nprod = ngpu = 0
    for s in shas:
        ch = impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{s}^", s))
        prod = {p: c for p, c in ch.items() if p.startswith(("src/", "include/", "kernels/"))}
        if not prod:
            continue
        nprod += 1
        tt = touched_tests(s, with_file=True)
        m = map_at(s)[0] if at_commit else m_head
        regs, unp = pair(m, tt)
        if unp and not at_commit:
            # review C3: a test renamed or removed after this commit is not in the map of REV: pair and analyze this
            # commit with the map built at the commit itself, and report what still does not pair
            m_s = map_at(s)[0]
            regs2, unp2 = pair(m_s, tt)
            if len(regs2) >= len(regs):
                m, regs, unp = m_s, regs2, unp2
                at_rev.append(s)
            unpaired += [(s, p, t) for p, t in unp]
        rep = impact.analyze(prod, m, s, f"{s}^")
        if gpu_steps(rep):
            ngpu += 1
        if not regs:
            continue
        listed = {f"{e}:{x}" for e, v in rep["cpu_tests"].items() for x in v} | {f"whirl-kernel-test:{x}" for x in rep["kernel_tests"]}
        for tid, exe in regs.items():
            tot += 1
            if tid in listed:
                fhit += 1
            elif exe in rep["cpu_tests"] or (exe == "whirl-kernel-test" and rep["kernel_tests"]):
                ehit += 1
                misses.append((s, tid, "exe only"))
            else:
                misses.append((s, tid, "MISS"))
    subj = lambda s: git("log", "-1", "--format=%s", s).strip()[:70]
    print(f"== history replay: {n} commits of {rev} skipping the newest {skip}{' (map built at every commit)' if at_commit else ''}: {nprod} change product code; {tot} (commit, test) pairs "
          f"where a commit changed product code and a test")
    rate = 100 * fhit / tot if tot else 0.0
    if tot:
        print(f"   CPU test listed by the map: {fhit}/{tot} ({rate:.0f}%); executable listed only: {ehit}; missed: {tot - fhit - ehit}"
              f"  [CPU tests only: not evidence for the GPU step selection]")
    print(f"   GPU release steps: {ngpu}/{nprod} product commits get >= 1 GPU step; {nprod - ngpu} get none (check them with --baseline)")
    if at_rev:
        print(f"   paired with the map at the commit itself (test renamed / removed later): {', '.join(at_rev)}")
    for s, p, t in unpaired:
        print(f"   UNPAIRED {s} {p}:{t}  (no executable registers it at REV or at the commit)  ({subj(s)})")
    for s, t, w in misses:
        print(f"   {w:8} {s} {t}  ({subj(s)})")
    near = m_head.get("fanout_near_cut") or {}
    if near:
        print(f"   WARN     called names within 2 of MAX_FANOUT {impact_build.MAX_FANOUT} (one more definition cuts their edges): "
              + ", ".join(f"{k}={v}" for k, v in near.items()))
    return tot, fhit, ehit, rate


def run_baseline(base_rev, n, rev, skip):
    """replay.py with base_rev's tools/impact and with these tools on the same commits; list narrowed commits"""
    import json as _json
    tmp = tempfile.mkdtemp(prefix="validate_base_")
    data = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", base_rev, "tools/impact"], capture_output=True)
    if data.returncode != 0:
        print(f"validate: git archive {base_rev} tools/impact failed")
        return None
    with tarfile.open(fileobj=io.BytesIO(data.stdout)) as t:
        t.extractall(tmp, filter="data")
    here = os.path.dirname(os.path.abspath(__file__))
    outs, procs = {}, []
    for tag, tools in (("base", os.path.join(tmp, "tools", "impact")), ("new", here)):
        outs[tag] = os.path.join(tmp, f"{tag}.json")
        procs.append(subprocess.Popen([sys.executable, "-I", os.path.join(here, "replay.py"), "--tools", tools, "--root", ROOT,
                                       "--rev", rev, "--n", str(n), "--skip", str(skip), "--out", outs[tag]]))
    if any(p.wait() != 0 for p in procs):
        print("validate: replay failed")
        return None
    b = _json.load(open(outs["base"], encoding="utf-8"))
    w = _json.load(open(outs["new"], encoding="utf-8"))
    shutil.rmtree(tmp, ignore_errors=True)
    nar, grew = [], 0
    nsteps = {"base": 0, "new": 0}
    for s in b:
        nsteps["base"] += len(b[s]["steps"])
        nsteps["new"] += len(w[s]["steps"])
        lost = {k: sorted(set(b[s][k]) - set(w[s][k])) for k in ("steps", "paths", "cpu", "kt")}
        if any(lost[k] for k in ("steps", "paths")):
            nar.append((s, lost))
        if any(set(w[s][k]) - set(b[s][k]) for k in ("steps", "paths", "cpu", "kt")):
            grew += 1
    print(f"== baseline: tools/impact of {base_rev} vs these tools on {len(b)} product commits of {rev} (skip {skip}): "
          f"release steps {nsteps['base']} -> {nsteps['new']}; {grew} commit(s) select more; {len(nar)} narrowed")
    for s, lost in nar:
        print(f"   NARROWED {s}: steps {lost['steps']} paths {lost['paths']}")
    for s in b:
        lc = sorted(set(b[s]["cpu"]) - set(w[s]["cpu"]))
        if lc and not any(s == x for x, _ in nar):
            print(f"   note     {s}: {len(lc)} CPU test(s) fewer, steps / paths unchanged: {lc[:4]}")
    return len(nar)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="store_true")
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--baseline")
    ap.add_argument("--map-at-commit", action="store_true", help="diagnostic: build the map at every replayed commit")
    a = ap.parse_args()
    bad = []
    if not a.history or a.cases:
        for c, rep, rows in run_cases():
            bad += [f"named case {c['name']}: MISS {e}" for e, ok in rows if not ok]
    if a.history or not a.cases:
        tot, fhit, ehit, rate = run_history(a.history or 150, a.rev, a.skip, a.map_at_commit)
        if not tot or rate < 95:
            bad.append(f"history: CPU hit rate {rate:.0f}% < 95% ({fhit}/{tot})")
    if a.baseline:
        nn = run_baseline(a.baseline, a.history or 150, a.rev, a.skip)
        if nn is None:
            return 2
        if nn:
            bad.append(f"baseline {a.baseline}: {nn} narrowed commit(s)")
    for b in bad:
        print(f"validate: FAIL {b}")
    print("validate: " + ("ok" if not bad else f"FAIL ({len(bad)})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
