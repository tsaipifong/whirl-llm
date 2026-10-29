r"""# SPDX-License-Identifier: Apache-2.0
mt_equiv_compare.py (arch025-engine S0, review3 A4; DESIGN 5 P4 / 7 E7): the EQUIVALENT-MODULO-TIMING compare of a
GPU batch against the previous batch, from scratch/a25_f2_gpu/f2_compare.py. S0 change: --hh <B> <B2> (repeatable)
gives the HEAD-vs-HEAD pair of THIS batch (same exe run twice, case folders); a timing class passes only when that
pair shows the same class for the unit. The old sources (S0b-1 B / B2 folders and the s0b2 fd_run1 / fd_run2 pair)
are used only with --hh-fallback-s0b1 and are labelled [fallback S0b-1] in the report. Without --hh and without the
fallback every timing class is INCOMPLETE (no HEAD-vs-HEAD evidence).
  python -I tools/gates/mt_equiv_compare.py --card r9700 --base <A root, A data in <base>\B\<case>> --data <B data>
         --bin <B build\Release> --hh <B2 case dir> <B case dir> [--hh ...] [--tag leg2] [--out <dir>]
  python -I tools/gates/mt_equiv_compare.py --selftest

Original header:
arch025 F2 V-work comparer (CPU, offline from saved data; bug-f2vwork 2026-10-26).

A = S0b-1 side B (the arch025 base build the F-steps start from, wt_a25_int), B = the F2 run of f2gpu.ps1.
Reuses scratch\a25_s0b\s0b1_compare.py (units / classify / case_verdict) and tools\impact\work_equal.py, so the rules
are the ones S0b-1 already used and the main session accepted:
  - code objects: PRODUCT exes only (whirl.exe, whirl-server.exe). whirl-server-gate.exe is the host-only test client
    (no AMDGPU code object; work_equal.compare_golden binding says it is not product work).
  - launch: CLI units must be SAME; server units are classified (s0b1_compare.classify) and a timing class
    (mtp-timing / env-tiers-off) passes only when S0b-1 saw the same class HEAD vs HEAD (B vs B2) for that unit,
    giving EQUIVALENT-MODULO-TIMING.
  - golden: the golden summaries (A golden.txt, B out.txt: "golden: exe .. sha256", reference set, OK lines) and the
    golden output folders: every case OK on both sides against the same reference set, output files' sha256 equal.
Units that exist only in B (the F2 release gate runs more steps than S0b-1 kept) are listed, not blocking.
cmp-fix-1027 (arch025-X4mt2): the S0b-1 HEAD-vs-HEAD classes are re-derived live from the S0b-1 B / B2 data with the
current s0b1_compare rules (find_log '<unit>_server_logs' fix, seq-timing), the S0b-1 json is only the fallback; every
srv_mt_* server run compares its finish lines (s0b1_compare.mt_finish).

  python -I f2_compare.py --card r9700 [--data <scratch\a25_f2_gpu\data_r9700>] [--base <s0b1\r9700>]
         [--bin <F2 build\Release>] [--tools <wt\tools\impact>] [--ignore-file ignore_server.txt] [--out <data\cmp2>] [--tag f2|f3|f4|f5|f6]
Exit 0 every case EQUIVALENT*, 1 some NOT_EQUIVALENT, 2 some INCOMPLETE, 3 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

W = r"C:\ClaudeWorkspace\whirl-work"
S0B = os.path.dirname(os.path.abspath(__file__))   # mt_equiv_rules.py lives here (S0)
REPO = os.path.dirname(os.path.dirname(S0B))
PRODUCT = ("whirl.exe", "whirl-server.exe")
GOLD_EXE = re.compile(r"^golden: (?:(server|gate) )?exe (.+?) sha256 ([0-9a-f]{64})")
# case: (A sub-folder of <base>\B, B sub-folder of <data>, A gate file, B gate file, golden?)
CASES = (
    ("st", "st", "rg_full", "release_gate.txt", "out.txt", False),
    ("seq", "seq", "rg_full", "release_gate.txt", "out.txt", False),
    ("mt", "mt", "rg_full", "release_gate.txt", "out.txt", False),
    ("golden", "golden", "golden_ll", "golden.txt", "out.txt", True),
)


def golden_exes(p):
    out = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = GOLD_EXE.match(line.rstrip("\n"))
                if m:
                    out[os.path.basename(m.group(2)).lower()] = m.group(3)
    return out


SCHED_RX = re.compile(r"(req \d+ \| (?:POST \S+|slot \d+ \| (?:assigned|starting after|prefill: \d+ tok)))")
# extra HEAD-vs-HEAD pairs (same exe twice) whose server units add timing classes to the S0b-1 B vs B2 set:
# arch024 S0b-2 fd_run1 / fd_run2 (whirl-server sha256 97694962..., both cards): req 4 / req 5 of the golden cold
# run were prefilled jointly in run1 and one after the other in run2 (bug-f2vwork).
HH_EXTRA = {c: [(os.path.join(W, "scratch", "s0b2", f"case_{c}_fd_run1"), os.path.join(W, "scratch", "s0b2", f"case_{c}_fd_run2"))]
            for c in ("r9700", "8060s")}


def hh_s0b1(sc, base, ign, ign_rx):
    """cmp-fix-1027: {unit: set(classes)} of the S0b-1 HEAD-vs-HEAD pairs (B vs B2, sc.HH_PAIRS, server units; golden:
    srv_* units) re-derived from the saved S0b-1 data with the current sc.classify -> (hh, source) or (None, why)"""
    exes = os.path.join(base, "exes.tsv")
    if not os.path.isfile(exes):
        return None, f"no {exes}"
    sha, _bins = sc.read_exes(exes)
    hh, used = {}, []
    for bc, b2c in sc.HH_PAIRS:
        x, y = os.path.join(base, "B", bc), os.path.join(base, "B2", b2c)
        if not (os.path.isdir(x) and os.path.isdir(y)):
            continue
        for u in sc.compare_units(x, y, sha.get("B", {}), sha.get("B2", {}), ign, ign_rx):
            if not u.get("server") or (bc == "golden" and not u["unit"].startswith("srv_")):
                continue
            hh.setdefault(u["unit"], set()).update(c["class"] for c in u.get("classes", []))
        used.append(f"{bc}/{b2c}")
    if not used:
        return None, f"no B / B2 case folders under {base}"
    return hh, "live from " + base + " (" + ", ".join(used) + ")"


def sched_lines(p):
    if not p or not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return [m.group(1) for m in (SCHED_RX.search(x) for x in f) if m]


def slot_timing(sc, info, xcase, ycase, unit):
    """bug-f2vwork: a count-other run of a --parallel >= 2 server whose per-request MTP lines are equal but whose request
    arrival / slot assignment order differs (e.g. A: 'req 5 | slot 0 | starting after N ms in queue', B: 'assigned'):
    concurrent client requests met a busy or a free slot, so the decode batches overlapped differently and every
    kernel count moved. Timing, like mtp-timing; it passes only when also seen HEAD vs HEAD (case_verdict), else
    INCOMPLETE. The output check (golden server cases) is what proves the responses equal."""
    if info.get("class") != "count-other":
        return info
    m = re.search(r"--parallel (\d+)", info["key"])
    if not m or int(m.group(1)) < 2:
        return info
    la, lb = sc.find_log(xcase, unit, info.get("log")), sc.find_log(ycase, unit, info.get("log"))
    sa, sb = sched_lines(la), sched_lines(lb)
    if sa is not None and sb is not None and sa != sb:
        info["class"] = "slot-timing"
        info["sched_differing"] = [[x, y] for x, y in zip(sa, sb) if x != y][:6]
        info["logs"] = [la, lb]
    return info


STARTED_RX = re.compile(r"server (\S+) started \(pid (\d+)\)")
# bug-launchlog-pid: a launch table file name, old (launch_<pid>.txt) or new (launch_<pid>_<start>[_<n>].txt, written
# exclusively by builds with src/hip/log_name.h, bug:log-pid-names); group 1 = pid
LOG_NAME_RX = re.compile(r"^launch_(\d+)(?:_(\d{8}T\d{9}))?(?:_(\d+))?\.txt$")
# bug-launchlog-pid: a launch log is written in one go at exit (creation -> last write <= ~20 ms); a later process
# that got the same PID truncated and rewrote launch_<pid>.txt (fopen "wb"), which keeps the NTFS creation time of the
# first writer and moves only the last-write time (bug:n3-golden8060-pidreuse, scratch\n3_golden8060\pid_reuse_check.py)
OVERWRITE_GAP_S = 2.0


def is_launch_log(fn):
    return LOG_NAME_RX.match(fn) is not None


def has_pid_log(unit_dir, pid):
    """a launch table of this pid in unit_dir, under the old or the new name"""
    if not os.path.isdir(unit_dir):
        return False
    return any((m := LOG_NAME_RX.match(fn)) and m.group(1) == str(pid) for fn in os.listdir(unit_dir))


def overwritten_logs(d, gap=OVERWRITE_GAP_S):
    """launch tables directly in folder d whose last write is > gap s after their creation: a second process with the
    same PID rewrote the file (the first run's table is lost). [{'path', 'created', 'written', 'surviving_cmd'}]"""
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not is_launch_log(fn):
            continue
        p = os.path.join(d, fn)
        st = os.stat(p)
        c = getattr(st, "st_birthtime", st.st_ctime)  # Windows: st_ctime is the creation time (pre-3.12 Pythons)
        if st.st_mtime - c > gap:
            cmd = ""
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("# cmd: "):
                        cmd = line[7:].rstrip("\n"); break
            out.append({"path": p, "created": c, "written": st.st_mtime, "surviving_cmd": cmd})
    return out


def _ts(x):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)) + f".{int((x % 1) * 1000):03d}"


def _split_key(unit_dir, key):
    """work_equal.load_side keys runs of a step sub-folder as '<rel>/<key>': -> (folder of the run, key in that folder)"""
    for root, _ds, fs in os.walk(unit_dir):
        rel = os.path.relpath(root, unit_dir).replace("\\", "/")
        if rel != "." and key.startswith(rel + "/") and any(is_launch_log(f) for f in fs):
            return root, key[len(rel) + 1:]
    return unit_dir, key


def resolve_cli_only(lt, e, have_dir, miss_dir, miss_root, miss_bin, claimed):
    """bug-launchlog-pid: a CLI run with a launch table on one side only (cli-only_a / cli-only_b).
    - The missing side's folder holds an overwritten table (overwritten_logs, not yet used for another lost run of
      this unit) whose surviving '# cmd:' line is another command (or the lost key is a '#n' repeat): the run's table
      was lost to PID reuse. Then a substitute record is looked for in the missing side's data folder (miss_root, the
      run's own folder excluded): same exe (miss_bin in its '# cmd:' line), same normalised command, not itself
      overwritten. Its table SAME as the present side's -> 'same-vs-substitute' (equal; the file is named);
      none, or none SAME -> 'pid-reuse-lost' (INCOMPLETE, never NOT_EQUIVALENT: nothing says the work differs).
    - No overwritten table on the missing side: unchanged 'cli-only_a' / 'cli-only_b' (NOT_EQUIVALENT).
    claimed: set of overwritten paths already used for a lost run (one overwrite accounts for one lost run)."""
    base = {"key": e["key"], "class": "cli-" + e["status"].lower(), "status": e["status"]}
    sub_have, k_have = _split_key(have_dir, e["key"])
    sub_miss = os.path.join(miss_dir, os.path.relpath(sub_have, have_dir)) if sub_have != have_dir else miss_dir
    plain = re.sub(r" #\d+$", "", k_have)
    cand = [o for o in overwritten_logs(sub_miss) if o["path"] not in claimed
            and (lt.norm_cmd(o["surviving_cmd"]) != plain or plain != k_have)]
    if not cand:
        return base
    ow = cand[0]
    claimed.add(ow["path"])
    lost = {"overwritten": ow["path"], "first_writer_exit": _ts(ow["created"]), "last_writer_exit": _ts(ow["written"]),
            "surviving_cmd": lt.norm_cmd(ow["surviving_cmd"])[:160]}
    have = lt.table_of(sub_have).get(k_have)
    subs, tried = None, []
    if have is not None and miss_bin and os.path.isdir(miss_root):
        skip = os.path.normcase(os.path.abspath(sub_miss))
        for root, _ds, fs in os.walk(miss_root):
            if os.path.normcase(os.path.abspath(root)) == skip:
                continue
            for fn in sorted(fs):
                if not is_launch_log(fn):
                    continue
                p = os.path.join(root, fn)
                st = os.stat(p)
                if st.st_mtime - getattr(st, "st_birthtime", st.st_ctime) > OVERWRITE_GAP_S:
                    continue  # itself overwritten: not one run's record
                cmd, n, seq, rows, aseq, args = lt.read_log(p)
                if n == 0 or lt.norm_cmd(cmd) != plain or os.path.normcase(miss_bin) not in os.path.normcase(cmd):
                    continue
                r = lt.compare({plain: have}, {plain: {"n": n, "seq": seq, "rows": rows, "aseq": aseq, "args": args}})[0]
                tried.append({"path": p, "status": r["status"]})
                if r["status"] == "SAME":
                    subs = p
                    break
            if subs:
                break
    if subs:
        return dict(base, **{"class": "same-vs-substitute", "pid_reuse": lost, "substitute": subs})
    why = ("present side has no table for the key" if have is None else "no exe path for the missing side" if not miss_bin
           else f"no same-exe same-command record SAME in {miss_root}" + (f" (tried {[(os.path.basename(t['path']), t['status']) for t in tried]})" if tried else ""))
    return dict(base, **{"class": "pid-reuse-lost", "pid_reuse": lost, "why": why, "substitutes_tried": tried})


def lost_runs(case_dir, unit, unit_dir):
    """arch025-F2-mt: server runs the gate started ('server TAG started (pid N)' in its <unit>.txt) whose launch_N.txt
    is missing from the unit's launch folder. The launch log is written at process exit (atexit); a teardown crash
    (bug:server-exit-av-amdhip64-8f725, S0b-1 8060S B mtp_cache pid 21372) loses it, so the run shows up as
    one-side-only although it ran. {'server_TAG.log': pid}"""
    b = unit.split("/")[-1]
    out = {}
    for root, _ds, fs in os.walk(case_dir):
        if b + ".txt" not in fs:
            continue
        with open(os.path.join(root, b + ".txt"), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STARTED_RX.search(line)
                if m and not has_pid_log(unit_dir, m.group(2)):  # bug-launchlog-pid: old or new file name
                    out[f"server_{m.group(1)}.log"] = int(m.group(2))
    return out


def resolve_only_b(sc, lt, e, x, y, unit, ux_dir, uy_dir, b2case, a_bin, ign_rx):
    """arch025-F2-mt: an ONLY_B server run (B ran it, A has no launch table for it).
    - A really ran it but lost the launch log (lost_runs) and S0b-1 B2 (the same exe as A, rerun) has the run:
      compare B against B2 for that run and classify it like any other run (timing classes still need HH evidence).
    - A lost it and B2 cannot stand in: 'a-lost-evidence' (INCOMPLETE, not NOT_EQUIVALENT: nothing says the work differs).
    - A never started it: 'one-side-only' (unchanged; NOT_EQUIVALENT)."""
    m = re.search(r"--log-file (\S+)", e["key"])
    logname = m.group(1) if m else None
    lost = lost_runs(x, unit, ux_dir)
    if not logname or logname not in lost:
        return {"key": e["key"], "class": "one-side-only", "status": e["status"]}
    base = {"key": e["key"], "a_lost": {"log": logname, "pid": lost[logname]}}
    d2 = sc.units(b2case).get(unit) if b2case and os.path.isdir(b2case) else None
    if d2:
        bound = all(os.path.normcase(a_bin) in os.path.normcase(lt.read_log(os.path.join(d2, fn))[0])
                    for fn in os.listdir(d2) if is_launch_log(fn)
                    and lt.norm_cmd(lt.read_log(os.path.join(d2, fn))[0]).startswith("whirl-server"))
        t2, ty = lt.table_of(d2, False, True), lt.table_of(uy_dir, False, True)
        if bound and e["key"] in t2 and e["key"] in ty:
            e2 = lt.compare({e["key"]: t2[e["key"]]}, {e["key"]: ty[e["key"]]})[0]
            if e2["status"] == "SAME":
                return dict(base, **{"class": "same-vs-b2", "substituted_a": b2case})
            ci = slot_timing(sc, sc.classify(e2, ign_rx, b2case, y, unit), b2case, y, unit)
            ci.update(base, substituted_a=b2case)
            return ci
        base["why"] = "B2 run not bound to the A exe" if not bound else "run missing in B2 or B"
    else:
        base["why"] = "no B2 unit"
    return dict(base, **{"class": "a-lost-evidence", "status": e["status"]})


def selftest():
    """CPU only, fake folders: lost_runs / resolve_only_b (arch025-F2-mt)."""
    import tempfile
    sys.path.insert(0, S0B)
    import mt_equiv_rules as sc  # noqa: E402
    sc.load_tools(os.path.join(REPO, "tools", "impact"))
    lt = sc.lt
    bad = 0
    abin = r"C:\fake\A\Release"
    srv = lambda log: rf"{abin}\whirl-server.exe C:\m\Ornith.gguf --port 1 --log-file C:\w\{log}"
    rows = {"k1 1x1x1 1x1x1 0": 5}
    with tempfile.TemporaryDirectory() as t:
        def case(name, runs, gate_pids):
            c = os.path.join(t, name); u = os.path.join(c, "ll", "srv_x"); os.makedirs(u)
            os.makedirs(os.path.join(c, "rg", "s"))
            for pid, log, n, mtp in runs:
                lt._log(u, pid, srv(log), n, "aa", rows, "bb", {})
                lp = os.path.join(c, "rg", "s", "srv_x_server_logs"); os.makedirs(lp, exist_ok=True)
                with open(os.path.join(lp, log), "w", encoding="utf-8") as f:
                    f.write(f"x I req 1 | slot 0 | MTP: drafted {mtp}, accepted 1\n")
            with open(os.path.join(c, "rg", "s", "srv_x.txt"), "w", encoding="utf-8") as f:
                for pid, tag in gate_pids:
                    f.write(f"[00:00:00] server {tag} started (pid {pid}): x\n")
            return c, u
        x, ux = case("A", [(2, "server_mtp_verify.log", 10, 3)], [(1, "mtp_cache"), (2, "mtp_verify")])
        b2, _ = case("B2", [(5, "server_mtp_cache.log", 10, 3), (6, "server_mtp_verify.log", 10, 3)], [(5, "mtp_cache"), (6, "mtp_verify")])
        y, uy = case("F", [(7, "server_mtp_cache.log", 11, 4), (8, "server_mtp_verify.log", 10, 3)], [(7, "mtp_cache"), (8, "mtp_verify")])
        n, _ = case("N", [], [])
        key = lt.norm_cmd(srv("server_mtp_cache.log"))
        checks = [
            ("lost_runs finds the started pid without a launch log", lost_runs(x, "srv_x", ux) == {"server_mtp_cache.log": 1}),
            ("lost + B2 has it -> classified vs B2 (mtp-timing)",
             resolve_only_b(sc, lt, {"key": key, "status": "ONLY_B"}, x, y, "srv_x", ux, uy, b2, abin, [])["class"] == "mtp-timing"),
            ("lost + no B2 -> a-lost-evidence",
             resolve_only_b(sc, lt, {"key": key, "status": "ONLY_B"}, x, y, "srv_x", ux, uy, n, abin, [])["class"] == "a-lost-evidence"),
            ("lost + B2 bound to another exe -> a-lost-evidence",
             resolve_only_b(sc, lt, {"key": key, "status": "ONLY_B"}, x, y, "srv_x", ux, uy, b2, r"C:\other", [])["class"] == "a-lost-evidence"),
            ("never started on A -> one-side-only",
             resolve_only_b(sc, lt, {"key": lt.norm_cmd(srv("server_other.log")), "status": "ONLY_B"}, x, y, "srv_x", ux, uy, b2, abin, [])["class"] == "one-side-only"),
        ]
        ok_ent = [{"unit": "srv_x", "server": True, "result": "different", "classes": [{"key": key, "class": "a-lost-evidence"}]}]
        v, _n = finish_verdict(sc, ok_ent, {}, {"code_objects": "identical"})
        checks.append(("a-lost-evidence alone -> INCOMPLETE", v == "INCOMPLETE"))
        ok_ent[0]["classes"].append({"key": "k2", "class": "one-side-only"})
        v, _n = finish_verdict(sc, ok_ent, {}, {"code_objects": "identical"})
        checks.append(("a-lost-evidence + one-side-only -> NOT_EQUIVALENT", v == "NOT_EQUIVALENT"))
        checks += selftest_pid_reuse(sc, lt, t)
        checks += selftest_cmpfix(sc, lt, t)
        for name, ok in checks:
            print(f"selftest {'PASS' if ok else 'FAIL'}: {name}")
            bad += not ok
    print(f"selftest: {len(checks) - bad}/{len(checks)} PASS")
    return 0 if bad == 0 else 1


def selftest_cmpfix(sc, lt, t):
    """cmp-fix-1027: hh_s0b1 re-derives the S0b-1 B vs B2 classes with the current classify (a seq-only HH pair with equal
    outputs gives seq-timing), returns None without folders; the s0b1_compare rule selftests (find_log, seq-timing,
    finish) run here too, so f2 cannot run against an s0b1_compare without them."""
    out = [(f"s0b1_compare rule: {n}", ok) for n, ok in sc.selftest_cmpfix()]
    base = os.path.join(t, "cf_base")
    srv = r"C:\fake\B\Release\whirl-server.exe C:\m\Ornith.gguf --port 1 --parallel 4 --log-file C:\w\server_mtp_cache.log"
    rows = {"gdn_conv_state 1x1x1 64x1x1 0": 8, "k_a 1x1x1 64x1x1 0": 4}
    for side, seq, a in (("B", "d001", "00000000000000a1"), ("B2", "d002", "00000000000000a2")):
        u = os.path.join(base, side, "mt", "ll", "srv_mt_precise_moe"); os.makedirs(u)
        lt._log(u, 3, srv, 12, seq, rows, "e" + seq, {"gdn_conv_state 1x1x1 64x1x1 0 " + a: 8, "k_a 1x1x1 64x1x1 0 00000000000000c1": 4})
        d = os.path.join(base, side, "mt", "rg", "s", "srv_mt_precise_moe_server_logs"); os.makedirs(d)
        with open(os.path.join(d, "server_mtp_cache.log"), "w", encoding="utf-8") as f:
            f.write("x I req 1 | slot 0 | MTP: drafted 9, accepted 5\n"
                    "x I req 1 | slot 0 | finish_reason stop, completion 9 tok, reasoning 0 B, content 30 B, tool_calls 0, cache now 50 tok\n")
    with open(os.path.join(base, "exes.tsv"), "w") as f:
        f.write("B\twhirl-server.exe\t" + "0" * 64 + "\t" + r"C:\fake\B\Release" + "\tx\n")
    hh, src = hh_s0b1(sc, base, [], [])
    out.append((f"hh_s0b1: S0b-1 B vs B2 seq-only with equal outputs -> seq-timing HH evidence (got {hh}, {src[:40]})",
                hh is not None and hh.get("srv_mt_precise_moe") == {"seq-timing"}))
    hn, why = hh_s0b1(sc, os.path.join(t, "cf_none"), [], [])
    out.append((f"hh_s0b1: no S0b-1 folders -> None (json fallback) ({why[:40]})", hn is None))
    return out


def selftest_pid_reuse(sc, lt, t):
    """bug-launchlog-pid: CLI cli-only_a / cli-only_b against PID-reuse evidence (fake folders under t)."""
    a_bin, b_bin = r"C:\fake\A\Release", r"C:\fake\B\Release"
    cli = lambda b, s: rf"{b}\whirl.exe golden C:\m\Swift.gguf --suite {s} --precise --device gfx1151"
    rows, rows2 = {"k1 1x1x1 1x1x1 0": 5}, {"k1 1x1x1 1x1x1 0": 6}

    def mk(path, files, overwrite=()):
        os.makedirs(path, exist_ok=True)
        for pid, cmd, r in files:
            lt._log(path, pid, cmd, sum(r.values()), f"{sum(r.values()):04x}", r)
            if pid in overwrite:  # a later process with the same PID rewrote it: last write 100 s after creation
                p = os.path.join(path, f"launch_{pid}.txt")
                m = os.stat(p).st_mtime + 100
                os.utime(p, (m, m))
        return path

    def run(name, b_over, subs, direction="ONLY_A", bind=("sha", "sha")):
        r = os.path.join(t, "pr_" + name)
        have_bin, miss_bin = (a_bin, b_bin) if direction == "ONLY_A" else (b_bin, a_bin)
        have = mk(os.path.join(r, "have", "golden_ll", "ll"), [(1, cli(have_bin, "pair"), rows), (2, cli(have_bin, "ngram"), rows)])
        # the missing side ran both, but the pair run's table was rewritten by the ngram run (same PID 40184)
        miss = mk(os.path.join(r, "miss", "golden_ll", "ll"), [(40184, cli(miss_bin, "ngram"), rows)],
                  overwrite=(40184,) if b_over else ())
        for i, (sb, sr, sub_over) in enumerate(subs):
            mk(os.path.join(r, "miss", "rg_full", "ll", f"golden_swift{i}"), [(34396 + i, cli(sb or miss_bin, "pair"), sr)],
               overwrite=(34396 + i,) if sub_over else ())
        e = {"key": lt.norm_cmd(cli(have_bin, "pair")), "status": direction}
        c = resolve_cli_only(lt, e, have, miss, os.path.join(r, "miss"), miss_bin, set())
        ent = [{"unit": ".", "server": False, "result": "missing", "binding": list(bind), "classes": [c]}]
        v, _n = finish_verdict(sc, ent, {}, {"code_objects": "identical"})
        return c["class"], v

    out = []
    for name, args, want in (
        ("overwrite + substitute (same exe, same command, SAME table) -> same-vs-substitute, EQUIVALENT",
         ("s1", True, [(None, rows, False)]), ("same-vs-substitute", "EQUIVALENT")),
        ("overwrite, no substitute -> pid-reuse-lost, INCOMPLETE", ("s2", True, []), ("pid-reuse-lost", "INCOMPLETE")),
        ("genuine only_a (no overwrite) -> cli-only_a, NOT_EQUIVALENT", ("s3", False, [(None, rows, False)]), ("cli-only_a", "NOT_EQUIVALENT")),
        ("overwrite + substitute with another table -> pid-reuse-lost, INCOMPLETE (not equal, not NOT_EQUIVALENT)",
         ("s4", True, [(None, rows2, False)]), ("pid-reuse-lost", "INCOMPLETE")),
        ("overwrite + substitute of another exe -> pid-reuse-lost, INCOMPLETE", ("s5", True, [(r"C:\other\Release", rows, False)]),
         ("pid-reuse-lost", "INCOMPLETE")),
        ("overwrite + an overwritten substitute only -> pid-reuse-lost, INCOMPLETE", ("s6", True, [(None, rows, True)]),
         ("pid-reuse-lost", "INCOMPLETE")),
        ("cli-only_b, A side overwritten + A-side substitute -> same-vs-substitute, EQUIVALENT",
         ("s7", True, [(None, rows, False)], "ONLY_B"), ("same-vs-substitute", "EQUIVALENT")),
        ("genuine only_b (no overwrite) -> cli-only_b, NOT_EQUIVALENT", ("s8", False, [], "ONLY_B"), ("cli-only_b", "NOT_EQUIVALENT")),
        ("substitute equal but evidence unbound -> INCOMPLETE", ("s9", True, [(None, rows, False)], "ONLY_A", ("sha", "unverified")),
         ("same-vs-substitute", "INCOMPLETE")),
    ):
        got = run(*args)
        out.append((f"{name} (got {got})", got == want))
    # one overwrite accounts for one lost run only
    r = os.path.join(t, "pr_claim")
    have = mk(os.path.join(r, "have", "ll"), [(1, cli(a_bin, "pair"), rows), (2, cli(a_bin, "multi"), rows), (3, cli(a_bin, "ngram"), rows)])
    miss = mk(os.path.join(r, "miss", "ll"), [(7, cli(b_bin, "ngram"), rows)], overwrite=(7,))
    claimed = set()
    c1 = resolve_cli_only(lt, {"key": lt.norm_cmd(cli(a_bin, "pair")), "status": "ONLY_A"}, have, miss, os.path.join(r, "miss"), b_bin, claimed)
    c2 = resolve_cli_only(lt, {"key": lt.norm_cmd(cli(a_bin, "multi")), "status": "ONLY_A"}, have, miss, os.path.join(r, "miss"), b_bin, claimed)
    out.append((f"two lost runs, one overwritten table: first pid-reuse-lost, second stays cli-only_a (got {c1['class']}, {c2['class']})",
                (c1["class"], c2["class"]) == ("pid-reuse-lost", "cli-only_a")))
    # file names: lost_runs takes the new launch_<pid>_<start>[_<n>].txt names (bug:log-pid-names writers)
    d = os.path.join(t, "pr_names")
    os.makedirs(d)
    open(os.path.join(d, "launch_77_20261026T170842184.txt"), "w").close()
    open(os.path.join(d, "launch_78_20261026T170842184_2.raw.txt"), "w").close()
    out.append(("has_pid_log: new name found, .raw.txt and other pids not",
                has_pid_log(d, 77) and not has_pid_log(d, 78) and not has_pid_log(d, 7)))
    return out


def finish_verdict(sc, ur, hh, extra):
    """case_verdict, with arch025-F2-mt classes: same-vs-b2 passes, a-lost-evidence makes at most INCOMPLETE;
    bug-launchlog-pid (CLI units): same-vs-substitute passes, pid-reuse-lost makes at most INCOMPLETE. A CLI unit
    whose every non-SAME run is one of those counts as identical only if both sides' evidence is bound to its exe
    (work_equal binding sha / db), else it is 'missing' (INCOMPLETE)."""
    lost, view = [], []
    resolved, lossy = ("same-vs-b2", "same-vs-substitute"), ("a-lost-evidence", "pid-reuse-lost")
    for u in ur:
        cl = [c for c in u.get("classes", []) if c["class"] not in resolved + lossy]
        lost += [f"{u['unit']}: A launch log lost (pid {c.get('a_lost', {}).get('pid')}), {c.get('why', '')} {c['key'][:60]}"
                 for c in u.get("classes", []) if c["class"] == "a-lost-evidence"]
        lost += [f"{u['unit']}: {'B' if c['status'] == 'ONLY_A' else 'A'} launch table lost to PID reuse "
                 f"({os.path.basename(c['pid_reuse']['overwritten'])}: first writer exit {c['pid_reuse']['first_writer_exit']}, "
                 f"rewritten {c['pid_reuse']['last_writer_exit']} by {c['pid_reuse']['surviving_cmd'][:60]}); {c.get('why', '')}; {c['key'][:60]}"
                 for c in u.get("classes", []) if c["class"] == "pid-reuse-lost"]
        uu = dict(u, classes=cl)
        if u.get("server") and u["result"] != "identical" and not cl:
            uu["result"] = "identical"  # every differing run was resolved against B2 as SAME, or is a-lost-evidence
        elif not u.get("server") and u["result"] != "identical" and u.get("classes") and not cl:
            bound = all(s in ("sha", "db") for s in (u.get("binding") or ["?"]))
            uu["result"] = "identical" if bound else "missing"
            if not bound:
                uu["why"] = f"runs resolved by substitute / PID-reuse evidence, but launch evidence not bound: {u.get('binding')}"
        view.append(uu)
    v, notes = sc.case_verdict(view, hh, extra)
    if lost and v in ("EQUIVALENT", "EQUIVALENT-MODULO-TIMING"):
        v = "INCOMPLETE"
    return v, notes + lost


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, choices=("r9700", "8060s"))
    ap.add_argument("--data")
    ap.add_argument("--base")
    ap.add_argument("--bin", default=os.path.join(W, "wt_a25_f2", "build", "Release"))
    ap.add_argument("--tools", default=os.path.join(REPO, "tools", "impact"))
    ap.add_argument("--ignore-file", default=os.path.join(S0B, "mt_equiv_ignore_server.txt"))
    ap.add_argument("--hh", nargs=2, action="append", default=[], metavar=("B2_CASE", "B_CASE"),
                    help="this batch's HEAD-vs-HEAD pair (same exe twice): two case folders; repeatable (S0, A4)")
    ap.add_argument("--hh-fallback-s0b1", action="store_true",
                    help="also use the S0b-1 B / B2 folders and the s0b2 pair (old evidence, labelled)")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="f2", help="runner tag for report names / SUMMARY (f2..f6; fix-fNgpu-compare)")
    a = ap.parse_args()
    data = a.data or os.path.join(W, "scratch", "a25_f2_gpu", f"data_{a.card}")
    base = a.base or os.path.join(W, "scratch", "arch025", "s0b", "s0b1", a.card)
    outd = a.out or os.path.join(data, "cmp2")
    os.makedirs(outd, exist_ok=True)
    sys.path.insert(0, S0B)
    import mt_equiv_rules as sc  # noqa: E402
    sc.load_tools(a.tools)
    we, lt = sc.we, sc.lt
    sc.SAFE.add("slot-timing")  # timing class: EQUIVALENT-MODULO-TIMING only when also seen HEAD vs HEAD

    sha_tsv, bins = sc.read_exes(os.path.join(base, "exes.tsv"))
    a_sha, a_bin = sha_tsv.get("B", {}), bins.get("B", "")  # S0b-1 side B = our A
    b_gold = golden_exes(os.path.join(data, "golden_ll", "out.txt"))
    ign = lt.read_ignore_file(a.ignore_file) if a.ignore_file else []
    ign_rx = [re.compile(p) for p in ign]
    lines = [f"# arch025 {a.tag.upper()} V-work compare {a.card} {time.strftime('%Y-%m-%dT%H:%M:%S')}; A {a_bin} (S0b-1 B data {base}\\B); "
             f"B {a.bin} (data {data}); ignore {ign}"]
    rep = {"tag": a.tag, "card": a.card, "a_bin": a_bin, "b_bin": a.bin, "base": base, "data": data, "ignore": ign, "exes": {}, "cases": {}}

    # code objects, product exes; stale check: A against exes.tsv, B against the sha256 golden.ps1 logged during the run
    co_ok, co_bad = True, False
    sa, sb = {}, {}
    for n in PRODUCT:
        pa, pb = os.path.join(a_bin, n), os.path.join(a.bin, n)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            rep["exes"][n] = {"result": "missing"}; co_ok = False
            lines.append(f"code objects {n}: MISSING ({pa} / {pb})"); continue
        e = we.compare_exe(pa, pb)
        sa[n], sb[n] = e["a"]["sha256"], e["b"]["sha256"]
        stale = [s for s, want, got in (("A", a_sha.get(n), sa[n]), ("B", b_gold.get(n), sb[n])) if want and want != got]
        r = e["code_objects"]["result"]
        rep["exes"][n] = {"code_objects": r, "host": e["host"]["result"], "n": [e["code_objects"]["n_a"], e["code_objects"]["n_b"]],
                          "sha": [sa[n], sb[n]], "rebuilt_since_run": stale}
        co_ok = co_ok and r in ("identical", "identical-except-cuid") and not stale
        co_bad = co_bad or r == "different"
        lines.append(f"code objects {n}: {r} ({e['code_objects']['n_a']}/{e['code_objects']['n_b']}); host {e['host']['result']}; "
                     f"A {sa[n][:16]} B {sb[n][:16]}{'; EXE REBUILT SINCE THE RUN: ' + ','.join(stale) if stale else ''}")
    gx = golden_exes(os.path.join(data, "golden_ll", "out.txt"))
    lines.append("not compared: whirl-server-gate.exe (host-only test client, no AMDGPU code object; B gate sha "
                 f"{gx.get('whirl-server-gate.exe', '?')[:16]})")

    # HEAD vs HEAD timing classes seen by S0b-1 (B vs B2, same exe twice)
    # cmp-fix-1027: the S0b-1 B vs B2 classes are re-derived from the saved S0b-1 data with the current classify
    # (find_log fix, seq-timing); the S0b-1 json (written by the old rules) is the fallback when the folders are gone
    hh, hh_src = (hh_s0b1(sc, base, ign, ign_rx) if a.hh_fallback_s0b1 else ({}, "not used (no --hh-fallback-s0b1)"))
    if hh is None and a.hh_fallback_s0b1:
        hh = {}
        try:
            h = json.load(open(os.path.join(base, "compare", f"cmp_{a.card}.json"), encoding="utf-8"))
            for _case, ur in h.get("hh", {}).items():
                for u in ur:
                    hh.setdefault(u["unit"], set()).update(c["class"] for c in u.get("classes", []))
            hh_src = f"S0b-1 json (live re-derivation impossible: {hh_src})"
        except Exception as ex:  # noqa: BLE001
            lines.append(f"HH classes: S0b-1 compare json not readable ({ex}); timing classes will be INCOMPLETE")
    lines.append(f"HH classes [fallback S0b-1] B vs B2 source: {hh_src}")
    hh_pairs = [(x, y, "own") for x, y in a.hh] + ([(x, y, "fallback S0b-1") for x, y in HH_EXTRA.get(a.card, [])]
                                                  if a.hh_fallback_s0b1 else [])
    if not hh_pairs:
        lines.append("HH pairs: NONE (no --hh): every timing class is INCOMPLETE")
    for x2, y2, hh_kind in hh_pairs:
        if not (os.path.isdir(x2) and os.path.isdir(y2)):
            lines.append(f"HH [{hh_kind}] {x2} vs {y2}: missing"); continue
        ux2, uy2 = sc.units(x2), sc.units(y2)
        for u in sorted(set(ux2) & set(uy2)):
            if not (sc.unit_has_server(ux2[u]) or sc.unit_has_server(uy2[u])):
                continue
            got = set()
            for e in lt.compare(lt.table_of(ux2[u], False, True), lt.table_of(uy2[u], False, True)):
                if e["status"] == "DIFF":
                    ci = sc.classify(e, ign_rx, x2, y2, u)
                    got.add(slot_timing(sc, dict(ci), x2, y2, u)["class"])
                    # the HH pair's schedule itself differed (classify may have named it mtp-timing first because the
                    # per-request MTP lines moved with the request order): that is HEAD-vs-HEAD evidence of slot-timing
                    st = slot_timing(sc, dict(ci, **{"class": "count-other"}), x2, y2, u)
                    if st["class"] == "slot-timing":
                        got.add("slot-timing")
            hh.setdefault(u, set()).update(got)
            lines.append(f"HH [{hh_kind}] {x2} vs {y2} {u}: {sorted(got) or 'identical'}")
    lines.append("HH classes (own --hh pairs" + (" + fallback S0b-1" if a.hh_fallback_s0b1 else "") + "): " + "; ".join(f"{u} {sorted(c)}" for u, c in sorted(hh.items())))

    worst = []
    for name, asub, bsub, agate, bgate, is_gold in CASES:
        x, y = os.path.join(base, "B", asub), os.path.join(data, bsub)
        if not (os.path.isdir(x) and os.path.isdir(y)):
            rep["cases"][name] = {"verdict": "INCOMPLETE", "notes": [f"folder missing A {os.path.isdir(x)} B {os.path.isdir(y)}"]}
            worst.append("INCOMPLETE"); lines.append(f"AB {name}: INCOMPLETE (folder missing: A {x} {os.path.isdir(x)}, B {y} {os.path.isdir(y)})")
            continue
        ux, uy = sc.units(x), sc.units(y)
        keep = sorted(ux)
        ur = []
        for u in keep:
            if u not in uy:
                ur.append({"unit": u, "result": "missing", "why": "only in A"}); continue
            srv = sc.unit_has_server(ux[u]) or sc.unit_has_server(uy[u])
            r = we.compare_launch(ux[u], uy[u], server=srv, ignore=ign if srv else (), a_exes=sa, b_exes=sb)
            ent = {"unit": u, "server": srv, "result": r["result"], "n_same": r["n_same"], "n_runs": r["n_runs"],
                   "binding": [r["binding"]["a"]["state"], r["binding"]["b"]["state"]], "reasons": r["reasons"], "classes": []}
            if srv:
                ent["finish"] = sc.mt_finish(ux[u], uy[u], x, y, u)  # cmp-fix-1027 rule 3 (srv_mt_* only)
                for e in lt.compare(lt.table_of(ux[u], False, True), lt.table_of(uy[u], False, True)):
                    if e["status"] == "DIFF":
                        ent["classes"].append(slot_timing(sc, sc.classify(e, ign_rx, x, y, u), x, y, u))
                    elif e["status"] == "ONLY_B":
                        ent["classes"].append(resolve_only_b(sc, lt, e, x, y, u, ux[u], uy[u], os.path.join(base, "B2", asub),
                                                             a_bin, ign_rx))
                    elif e["status"] != "SAME":
                        ent["classes"].append({"key": e["key"], "class": "one-side-only", "status": e["status"]})
            else:
                claimed = set()  # bug-launchlog-pid: one overwritten table accounts for one lost run
                for e in r["runs"]:
                    if e["status"] in ("ONLY_A", "ONLY_B"):
                        # the missing side: B for ONLY_A (data folder `data`, exe a.bin), A for ONLY_B (S0b-1 B data, a_bin)
                        if e["status"] == "ONLY_A":
                            ent["classes"].append(resolve_cli_only(lt, e, ux[u], uy[u], data, a.bin, claimed))
                        else:
                            ent["classes"].append(resolve_cli_only(lt, e, uy[u], ux[u], os.path.join(base, "B"), a_bin, claimed))
                    elif e["status"] != "SAME":
                        ent["classes"].append({"key": e["key"], "class": "cli-" + e["status"].lower(),
                                               "rows_diff": e.get("rows_diff", [])[:5], "args_diff": e.get("args_diff", [])[:5]})
            ur.append(ent)
        b_only = sorted(set(uy) - set(ux))
        extra = {"code_objects": "identical" if co_ok else "different" if co_bad else "missing",
                 "gate_A": "identical" if sc.gate_pass(x, agate) else "missing",
                 "gate_B": "identical" if sc.gate_pass(y, bgate) else "missing"}
        g = None
        if is_gold:
            g = we.compare_golden([os.path.join(x, agate)], [os.path.join(y, bgate)], os.path.join(x, "out"), os.path.join(y, "out"), sa, sb)
            extra["golden"] = g["result"]
        v, notes = finish_verdict(sc, ur, hh, extra)
        notes += [f"{u['unit']}: A launch log lost (pid {c['a_lost']['pid']}, {c['a_lost']['log']}; A exe teardown, launch log is "
                  f"written at exit): run compared B vs S0b-1 B2 (same A exe) -> {c['class']}"
                  for u in ur for c in u.get("classes", []) if c.get("substituted_a")]
        notes += [f"{u['unit']}: {'B' if c['status'] == 'ONLY_A' else 'A'} launch table lost to PID reuse "
                  f"({c['pid_reuse']['overwritten']}: first writer exit {c['pid_reuse']['first_writer_exit']}, rewritten "
                  f"{c['pid_reuse']['last_writer_exit']} by {c['pid_reuse']['surviving_cmd'][:60]}): run compared against "
                  f"the same exe's record {c['substitute']} -> same-vs-substitute; {c['key'][:80]}"
                  for u in ur for c in u.get("classes", []) if c["class"] == "same-vs-substitute"]
        worst.append(v)
        rep["cases"][name] = {"verdict": v, "notes": notes, "units": ur, "b_only_units": b_only, "extra": extra,
                              "golden": None if g is None else {k: g[k] for k in ("result", "reasons", "outputs", "binding", "cases")}}
        lines.append(f"AB {name}: {v}; units {len(ur)} ({sum(u['result'] == 'identical' for u in ur)} identical); B-only units {len(b_only)}; "
                     + "; ".join(f"{k} {vv}" for k, vv in extra.items()))
        for u in ur:
            cl = ", ".join(c["class"] for c in u.get("classes", [])) or "-"
            lines.append(f"    {u['unit']}: {u['result']} ({u.get('n_same', 0)}/{u.get('n_runs', 0)} same{' after ignore' if u.get('server') else ''}); "
                         f"binding {u.get('binding')}; raw diff classes: {cl}")
        for n in notes[:12]:
            lines.append(f"    note: {n}")
        if g is not None:
            lines.append(f"    golden: {g['result']}; cases {len(g['cases'])} ({sum(c['same'] for c in g['cases'])} OK on both); outputs compared "
                         f"{g['outputs']['compared']}, differ {g['outputs']['differing']}, only A {g['outputs']['only_a']}, only B {g['outputs']['only_b']}; "
                         f"binding {g['binding']}")
            for r in g["reasons"][:4]:
                lines.append(f"    golden reason: {r}")
    ex = 1 if "NOT_EQUIVALENT" in worst else 2 if "INCOMPLETE" in worst else 0
    summ = f"SUMMARY {a.tag} {a.card} A (S0b-1 B) vs B ({a.tag.upper()}): " + ", ".join(f"{c} {rep['cases'][c]['verdict']}" for c, *_ in CASES if c in rep["cases"])
    lines.append(summ)
    rep["exit"] = ex
    with open(os.path.join(outd, f"{a.tag}cmp_{a.card}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, default=list)
    with open(os.path.join(outd, f"{a.tag}cmp_{a.card}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines:
        print(l)
    return ex


if __name__ == "__main__":
    sys.exit(main())
