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
measure-tools P1 additions: RANGE_CASES (a whole release diff: no path reached only through a type name, declarations
of a header select at most N functions, every needed step and GPU check outside release_gate is selected; run with
--cases), --arch024 (ground truth: the GPU checks each arch024 step's runner ran, fit / hold-out), and --baseline now
accepts a narrowed commit only when the declared-name rule explains it (the new tools saw declared names or moved
declarations in it) and it is not a ground-truth commit losing a GPU step.
Exit (arch024-S12, review C7; P1): 0 every named / range case hit, every history hit rate >= 95 %, no unexplained or
ground-truth narrowed commit, every arch024 needed check covered; 1 otherwise; 2 setup error.
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
from collections import defaultdict

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


# measure-tools P1: named RANGE cases (a whole release diff). "type_only": identifiers that are only type names on the
# changed declaration lines; no path may be reached "uses <type>" through them (B1.1). "decl_max": the most functions
# a header's declarations may select (before P1: model.h 147, engine_api.h 101, vision.h 43 users,
# scratch/s15/impact_605281b_9844fde.txt). "expect": what the release really needed (S15 runner + s15-blast GAP list),
# including the GPU checks outside release_gate.
RANGE_CASES = [
    {"name": "0.2.4 S15 range 605281b..9844fde (S15 over-selection from type names; s15-blast GAP list)",
     "range": "605281b..9844fde", "map_at": "9844fde",
     "type_only": ["DevBuf", "DevPtr", "EventH", "PinnedBuf", "StreamH", "Event", "Stream", "hip", "device"],
     "decl_max": {"include/whirl/model.h": 20, "include/whirl/vision.h": 10, "src/server/engine_api.h": 60},
     "expect": ["release:golden_swift", "release:golden_moe", "release:seq_swift_<mode>", "release:seq_moe_<mode>",
                "release:srv_cache_balance", "release:srv_cache_precise", "release:srv_mt_precise", "release:srv_tier_verify",
                "release:srv_gpu_select", "release:srv_env_e1", "release:srv_env_e3", "release:srv_env_e4", "release:srv_env_e7",
                "check:E2", "check:kvram719", "check:visab", "check:dense", "check:E6"]},
]

# measure-tools P1: ground truth of the arch024 steps (0.2.4): the product commits of each step (subjects tagged
# "(arch024-<step>" / B0-, B0b-; tools-only steps S0b, S1-deletions-only kept out) and the GPU checks its runner ran /
# its PLAN row required (progress/arch024-<step>.HANDOFF.md, read 2026-10-25; S2 P-size and S3 E7 were explicitly NOT
# run and are left out). golden = golden_swift or golden_moe; P-size = sizing class (startup_lines.py over the gate's
# server logs); kvram719 / visab / E2 / E6 = the named check outside release_gate. fit = the steps the P1 rules were
# designed on (S11 and the S15 range via the s15-blast GAP list); every other step is hold-out.
ARCH024_STEPS = [
    {"step": "B0", "fit": True, "commits": ["06a1540", "8e79cbc", "fd07e8a", "d754891", "b0acad1", "cf3f4f3"],
     "needed": ["check:kvram719", "golden", "check:P-size"]},
    {"step": "S0a", "fit": False, "commits": ["cc03dd3", "703c666"], "needed": ["golden", "check:P-size"]},
    {"step": "S2", "fit": False, "commits": ["9fd84f3", "25accd6"], "needed": ["release:srv_mt_precise", "release:srv_mt_precise_moe"]},
    {"step": "S3", "fit": False, "commits": ["ef40a1f"], "needed": ["golden", "release:srv_cache_balance", "release:srv_mt_precise",
                                                                   "release:srv_mt_precise_moe", "release:srv_gpu_select",
                                                                   "check:kvram719", "check:P-size"]},
    {"step": "S4", "fit": False, "commits": ["920ae63"], "needed": ["golden", "release:srv_cache_balance", "release:srv_mt_precise",
                                                                   "release:srv_mt_precise_moe", "check:P-size"]},
    {"step": "S5a", "fit": False, "commits": ["dadae41"], "needed": ["golden", "release:srv_cache_precise", "release:srv_cache_balance",
                                                                    "release:srv_precise_cli", "release:srv_mt_precise",
                                                                    "release:srv_mt_precise_moe", "check:P-size"]},
    {"step": "S6a", "fit": False, "commits": ["b5ed025"], "needed": ["golden", "release:srv_tier_verify", "release:srv_gpu_select",
                                                                    "release:srv_cache_balance", "check:kvram719", "release:srv_mt_precise",
                                                                    "release:srv_mt_precise_moe", "check:P-size"]},
    {"step": "S6b", "fit": True, "commits": ["d96dc9d"], "needed": ["golden", "release:srv_tier_verify", "release:srv_cache_balance",
                                                                    "release:srv_mt_precise", "release:srv_mt_precise_moe",
                                                                    "release:srv_gpu_select", "check:kvram719", "check:P-size"]},
    {"step": "S7", "fit": True, "commits": ["43205f5"], "needed": ["golden", "release:srv_tier_verify", "release:srv_mt_precise",
                                                                   "release:srv_mt_precise_moe", "release:srv_cache_precise",
                                                                   "release:srv_cache_balance", "check:P-size"]},
    {"step": "S8", "fit": False, "commits": ["51de103"], "needed": ["golden", "release:srv_fs_swift", "release:srv_fs_moe", "check:P-size"]},
    {"step": "S9", "fit": False, "commits": ["a214b6f"], "needed": ["golden", "release:srv_gpu_select", "release:srv_balance",
                                                                   "check:kvram719", "release:srv_mt_precise", "check:P-size"]},
    {"step": "S10", "fit": False, "commits": ["1b17eb9"], "needed": ["golden", "release:srv_tier_verify", "release:srv_mt_precise",
                                                                    "release:srv_balance"]},
    {"step": "S11", "fit": True, "commits": ["3bfaac7"], "needed": ["golden", "check:visab", "check:kvram719", "release:srv_mt_precise",
                                                                   "release:srv_mt_precise_moe", "release:srv_gpu_select", "check:P-size"]},
    {"step": "S11c", "fit": False, "commits": ["9cd2100", "6251d2f"], "needed": ["golden"]},
]


# review A4: commits the P1 rules were tuned on after a validation miss (never hold-out evidence): f6e4236 (vision ->
# golden), the B0 / S6b / S7 / S11 arch024 commits (golden steps, P-719 pinned class, removed-declaration rule, design)
TUNED = {"f6e4236", "06a1540", "8e79cbc", "fd07e8a", "d754891", "b0acad1", "cf3f4f3", "d96dc9d", "43205f5", "3bfaac7"}


def covered(rep, item):
    """a needed GPU check is covered by the report (release steps with <mode> expanded, GPU checks, behaviour classes)"""
    kind, _, val = item.partition(":")
    steps = set(rep["release_gate_steps"]) | impact.expand_steps(rep["release_gate_steps"])
    if item == "golden":
        return bool({"golden_swift", "golden_moe"} & steps)
    if kind == "release":
        return val in steps
    if kind == "check":
        if val == "P-size":
            return "sizing" in (rep.get("behaviour") or {})
        return any(c.startswith(val) for c in rep.get("outside_gate_checks") or [])
    return flagged(rep, item)


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


def run_range_cases():
    """measure-tools P1 named range cases: the type-name over-selection is gone, every needed step / check is selected"""
    bad = []
    for c in RANGE_CASES:
        m, _ = map_at(c["map_at"])
        x, y = c["range"].split("..")
        base = git("merge-base", x, y).strip()
        ch = product_only(impact.parse_diff(git("diff", "-U0", "--no-color", "-M", base, y)), keep_tests=True)
        rep = impact.analyze(ch, m, y, base)
        print(f"== {c['name']}")
        print(f"   range {c['range']}; map built at {c['map_at']}")
        tol = set(c["type_only"])
        via_type = sorted({w for p, ws in rep["paths"].items() for w in ws
                           if (mm := re.search(r"\(uses ([^)]*) of ", w)) and
                           {s.split("::")[-1].strip() for s in mm.group(1).split(",")} <= tol})
        print(f"   {'ok  ' if not via_type else 'FAIL'} no path reached only through a type name ({len(via_type)} reasons)"
              + (f": {via_type[:3]}" if via_type else ""))
        if via_type:
            bad.append(f"range case {c['range']}: {len(via_type)} path reason(s) through type names")
        for f, mx in c["decl_max"].items():
            n = [int(mm.group(1)) for p, w in rep["touched"] if p == f and (mm := re.search(r"declarations \(.*\): (\d+) function", w))]
            got = sum(n)
            okk = got <= mx
            print(f"   {'ok  ' if okk else 'FAIL'} {f}: declarations select {got} function(s) (limit {mx})")
            if not okk:
                bad.append(f"range case {c['range']}: {f} selects {got} > {mx}")
        for e in c["expect"]:
            hit = covered(rep, e)
            print(f"   {'HIT ' if hit else 'MISS'} {e}")
            if not hit:
                bad.append(f"range case {c['range']}: MISS {e}")
        print(f"   map defects: {len(rep['map_defects'])}" + "".join(f"\n     - {d}" for d in rep["map_defects"]))
        print("   " + selectivity(rep, m))
    return bad


def run_arch024():
    """measure-tools P1: the arch024 steps' needed GPU checks (ground truth) vs the map built at each commit"""
    bad = []
    tot = {True: [0, 0], False: [0, 0]}
    for st in ARCH024_STEPS:
        reps = []
        for s in st["commits"]:
            m, _ = map_at(s)
            ch = product_only(impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{s}^", s)))
            reps.append(impact.analyze(ch, m, s, f"{s}^"))
        rep = merge(reps)
        miss = [e for e in st["needed"] if not covered(rep, e)]
        tot[st["fit"]][0] += len(st["needed"]) - len(miss)
        tot[st["fit"]][1] += len(st["needed"])
        print(f"   {'fit     ' if st['fit'] else 'hold-out'} {st['step']:5} {len(st['needed']) - len(miss)}/{len(st['needed'])} needed GPU checks"
              + (f"  MISS {', '.join(miss)}" if miss else "") + f"  (map defects {len(rep['map_defects'])})")
        bad += [f"arch024 {st['step']}: MISS {e}" for e in miss]
    for f in (True, False):
        h, n = tot[f]
        print(f"== arch024 {'fit' if f else 'hold-out'}: {h}/{n} needed GPU checks selected ({100 * h / n if n else 0:.1f}%)")
    return bad


def merge(reports):
    out = {"paths": {}, "release_gate_steps": [], "server_gate_suites": [], "golden": [], "extra": {}, "cpu_tests": {},
           "kernel_tests": [], "not_in_release_gate": [], "outside_gate_checks": [], "map_defects": [], "behaviour": {}}
    for r in reports:
        out["paths"].update(r["paths"])
        for k in ("release_gate_steps", "server_gate_suites", "golden", "kernel_tests", "not_in_release_gate",
                  "outside_gate_checks", "map_defects"):
            out[k] = sorted(set(out[k]) | set(r.get(k, [])))
        for k, v in (r.get("behaviour") or {}).items():
            out["behaviour"][k] = sorted(set(out["behaviour"].get(k, [])) | set(v))
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


def score_pairs(regs, unp, rep):
    """(test id, verdict) per touched test: HIT, "exe only", MISS, or UNPAIRED (review K3: a touched test that no
    executable of the map registers counts as a miss, so an executable missing from the map cannot shrink the
    denominator and hide its misses)"""
    listed = {f"{e}:{x}" for e, v in rep["cpu_tests"].items() for x in v} | {f"whirl-kernel-test:{x}" for x in rep["kernel_tests"]}
    out = []
    for tid, exe in regs.items():
        if tid in listed:
            out.append((tid, "HIT"))
        elif exe in rep["cpu_tests"] or (exe == "whirl-kernel-test" and rep["kernel_tests"]):
            out.append((tid, "exe only"))
        else:
            out.append((tid, "MISS"))
    out += [(f"{p}:{t}", "UNPAIRED") for p, t in unp]
    return out


def run_history(n, rev, skip=0, at_commit=False, exclude=()):
    m_head = impact_build.build(ROOT) if rev == "HEAD" else map_at(rev)[0]
    shas = [x for x in git("log", "--format=%h", "--no-merges", f"--skip={skip}", f"-{n}", rev).split() if x[:7] not in exclude]
    tot = fhit = ehit = 0
    misses, unpaired, at_rev = [], [], []
    nprod = ngpu = 0
    roots = set(git("rev-list", "--max-parents=0", rev).split())
    for s in shas:
        if any(r.startswith(s) for r in roots):
            print(f"   note     {s} is a root commit (no parent to diff against): skipped")
            continue
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
        unpaired += [(s, p, t) for p, t in unp]  # review N2: also in --map-at-commit mode
        rep = impact.analyze(prod, m, s, f"{s}^")
        if gpu_steps(rep):
            ngpu += 1
        for tid, v in score_pairs(regs, unp, rep):
            tot += 1
            if v == "HIT":
                fhit += 1
            else:
                ehit += v == "exe only"
                misses.append((s, tid, v))
    subj = lambda s: git("log", "-1", "--format=%s", s).strip()[:70]
    print(f"== history replay: {n} commits of {rev} skipping the newest {skip}{' (map built at every commit)' if at_commit else ''}: {nprod} change product code; {tot} (commit, test) pairs "
          f"where a commit changed product code and a test")
    rate = 100 * fhit / tot if tot else 0.0
    if tot:
        print(f"   test listed by the map: {fhit}/{tot} ({rate:.1f}%); executable listed only: {ehit}; missed: {tot - fhit - ehit} "
              f"(UNPAIRED included: {len(unpaired)})  [test functions only (CPU tests and kernel-test functions): not evidence "
              f"for the GPU release steps]")
    print(f"   GPU release steps: {ngpu}/{nprod} product commits get >= 1 GPU step; {nprod - ngpu} get none (check them with --baseline)")
    if at_rev:
        print(f"   paired with the map at the commit itself (test renamed / removed later): {', '.join(at_rev)}")
    for s, t, w in misses:
        print(f"   {w:8} {s} {t}  ({subj(s)})")
    near = m_head.get("fanout_near_cut") or {}
    if near:
        print(f"   WARN     called names within 2 of MAX_FANOUT {impact_build.MAX_FANOUT} (one more definition cuts their edges): "
              + ", ".join(f"{k}={v}" for k, v in near.items()))
    return tot, fhit, ehit, rate


def ground_truth():
    """ground-truth commit (7 chars) -> the release steps it needed (ARCH024_STEPS, named CASES; golden = both models)"""
    gt = defaultdict(set)
    for st in ARCH024_STEPS:
        for c in st["commits"]:
            for e in st["needed"]:
                if e == "golden":
                    gt[c[:7]] |= {"golden_swift", "golden_moe"}
                elif e.startswith("release:"):
                    gt[c[:7]].add(e.split(":", 1)[1])
    for cs in CASES:
        for c in cs["fix"]:
            gt[c[:7]] |= {e.split(":", 1)[1] for e in cs["expect"] if e.startswith("release:")}
    return gt


RULES = {
    "R-decl": "the step was reached only through identifiers the declared-name rule drops: types / template arguments / "
              "decltype (a type changes on its own declaration line), parameter names and types (no meaning outside the "
              "declaration), initializer references (read, not changed), qualifiers",
    "R-member": "no identifier was dropped: the 0.2.4 rule matched functions that only name a same-named local / other "
                "member; the P1 rule requires .n / ->n, a member function of the class, or the class named with n",
    "R-moved": "the statement, in its full scope (namespaces, enclosing class headers with specialisation arguments, #if "
               "branch), vanishes from the changed lines of one side and appears on the other with the same number of "
               "occurrences: in another file (moved between headers), or in the same file with the declaration order of "
               "its class / enum, or at namespace scope of the file, unchanged (diff alignment); mt-p1 pass 3",
    "R-block": "a namespace / brace block added or removed whole (both braces changed; a lone brace still fails closed): "
               "the 0.2.4 rule read the identifier-less brace lines as a file-scope change",
}


def attribute(nar, rev, skip_cpu=True):
    """review A4: for every narrowed commit that loses a GPU step, re-run these tools with the legacy switches (map built
    at REV, as replay.py does): steps that come back with legacy "ids" / "moved", and the dropped identifiers by kind"""
    m = map_at(rev)[0]
    out = {}
    for s, lost in nar:
        gl = [x for x in impact.expand_steps(lost["steps"]) if not x.startswith("cpu_")]
        if not gl and skip_cpu:
            out[s] = {"ids": [], "moved": [], "dropped": {}}
            continue
        ch = impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{s}^", s))
        prod = {p_: c for p_, c in ch.items() if p_.startswith(("src/", "include/", "kernels/"))}
        r = impact.analyze(prod, m, s, f"{s}^")
        base = impact.expand_steps(r["release_gate_steps"])
        li = impact.expand_steps(impact.analyze(prod, m, s, f"{s}^", legacy=("ids",))["release_gate_steps"])
        lm = impact.expand_steps(impact.analyze(prod, m, s, f"{s}^", legacy=("moved",))["release_gate_steps"])
        lb = impact.expand_steps(impact.analyze(prod, m, s, f"{s}^", legacy=("ids", "moved"))["release_gate_steps"])
        lk = impact.expand_steps(impact.analyze(prod, m, s, f"{s}^", legacy=("braces",))["release_gate_steps"])
        out[s] = {"ids": sorted(li - base), "moved": sorted(lm - base), "both": sorted(lb - base), "braces": sorted(lk - base),
                  "dropped": r.get("dropped_ids", {})}
    return out


def classify_narrowed(nar, w, gt=None, attrib=None):
    """measure-tools P1 + review A4. A narrowed commit (the base tools select a step / path these tools do not) passes
    only when it loses no ground-truth needed step AND every lost GPU step is shown unneeded by a named rule (RULES):
    R-moved = the step comes back with the moved rule off; R-decl / R-member = it comes back with the 0.2.4 identifier
    rule and every dropped identifier is of a named kind (type / param / init / qualifier; none dropped = R-member).
    A dropped identifier of kind "other" (declarator position, not declared) or a step no variant brings back fails.
    attrib: commit -> {"ids": steps back with legacy ids, "moved": steps back with legacy moved, "dropped": {id: kind}}.
    Returns (failing commits, report lines, accepted commits, accepted commits that lose a GPU step)."""
    gt = ground_truth() if gt is None else gt
    fails, lines, expl, gpu_expl = [], [], [], 0
    for s, lost in nar:
        ls = impact.expand_steps(lost["steps"])
        gpu_lost = sorted(x for x in ls if not x.startswith("cpu_"))
        need_lost = sorted(set(ls) & gt.get(s[:7], set()))
        if need_lost:
            fails.append(s)
            lines.append(f"   NARROWED-GT {s}: ground-truth commit loses needed steps {need_lost} (all lost {gpu_lost})")
            continue
        if not gpu_lost:
            expl.append(s)
            lines.append(f"   narrowed {s}: no GPU step lost (CPU tests / paths {lost['paths'] or '-'})")
            continue
        a = (attrib or {}).get(s)
        if a is None:
            fails.append(s)
            lines.append(f"   NARROWED-UNATTRIBUTED {s}: GPU steps {gpu_lost} (no attribution run)")
            continue
        dropped = a.get("dropped") or {}
        kinds = {}
        for x, k in dropped.items():
            kinds.setdefault(k, []).append(x)
        per, bad = {}, []
        for st in gpu_lost:
            if st in a.get("moved", []):
                per[st] = "R-moved"
            elif st in a.get("ids", []):
                if kinds.get("other"):
                    bad.append(f"{st} (dropped identifier(s) of kind other: {', '.join(sorted(kinds['other'])[:4])})")
                else:
                    per[st] = "R-decl" if dropped else "R-member"
            elif st in a.get("braces", []):
                per[st] = "R-block"
            elif st in a.get("both", []) and not kinds.get("other"):
                per[st] = "R-decl+R-moved"
            else:
                bad.append(f"{st} (no legacy variant brings it back)")
        ev = "; ".join(f"{k}: {', '.join(sorted(v)[:5])}{' ...' if len(v) > 5 else ''}" for k, v in sorted(kinds.items())) or "none"
        if bad:
            fails.append(s)
            lines.append(f"   NARROWED-UNEXPLAINED {s}: {'; '.join(bad)} | dropped {ev}")
        else:
            expl.append(s)
            gpu_expl += 1
            rules = sorted(set(per.values()))
            lines.append(f"   narrowed {s}: GPU steps {gpu_lost} -> {', '.join(rules)} | dropped {ev}")
    return fails, lines, expl, gpu_expl


def run_baseline(base_rev, n, rev, skip, exclude=()):
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
                                       "--rev", rev, "--n", str(n), "--skip", str(skip), "--out", outs[tag]]
                                      + (["--exclude", ",".join(sorted(exclude))] if exclude else [])))
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
    fails, lines, expl, gpu_expl = classify_narrowed(nar, w, attrib=attribute(nar, rev))
    for ln in lines:
        print(ln)
    print(f"   narrowed: {len(expl)} explained by the declared-name rule ({gpu_expl} lose a GPU step), {len(fails)} failing "
          f"(ground truth / unexplained)")
    for s in b:
        lc = sorted(set(b[s]["cpu"]) - set(w[s]["cpu"]))
        if lc and not any(s == x for x, _ in nar):
            print(f"   note     {s}: {len(lc)} CPU test(s) fewer, steps / paths unchanged: {lc[:4]}")
    return len(fails)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="store_true")
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--baseline")
    ap.add_argument("--map-at-commit", action="store_true", help="diagnostic: build the map at every replayed commit")
    ap.add_argument("--arch024", action="store_true", help="measure-tools P1: needed GPU checks of the arch024 steps (fit / hold-out)")
    ap.add_argument("--exclude-tuned", action="store_true", help="review A4: leave the TUNED commits out of the replay (hold-out)")
    a = ap.parse_args()
    bad = []
    if (not a.history and not a.arch024) or a.cases:
        for c, rep, rows in run_cases():
            bad += [f"named case {c['name']}: MISS {e}" for e, ok in rows if not ok]
        bad += run_range_cases()
    if a.arch024:
        bad += run_arch024()
    if a.history or not (a.cases or a.arch024):
        ex = TUNED if a.exclude_tuned else ()
        if ex:
            print(f"   excluded (tuned on): {', '.join(sorted(ex))}")
        tot, fhit, ehit, rate = run_history(a.history or 150, a.rev, a.skip, a.map_at_commit, exclude=ex)
        if not tot or rate < 95:
            bad.append(f"history: test hit rate {rate:.1f}% < 95% ({fhit}/{tot}, UNPAIRED counted as missed)")
    if a.baseline:
        nn = run_baseline(a.baseline, a.history or 150, a.rev, a.skip, exclude=TUNED if a.exclude_tuned else ())
        if nn is None:
            return 2
        if nn:
            bad.append(f"baseline {a.baseline}: {nn} narrowed commit(s) not explained by the declared-name rule or on ground truth")
    for b in bad:
        print(f"validate: FAIL {b}")
    print("validate: " + ("ok" if not bad else f"FAIL ({len(bad)})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
