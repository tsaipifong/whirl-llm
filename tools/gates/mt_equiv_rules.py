r"""# SPDX-License-Identifier: Apache-2.0
mt_equiv_rules.py (arch025-engine S0, review3 A4): the classification rules of the EQUIVALENT-MODULO-TIMING compare,
copied unchanged from scratch/a25_s0b/s0b1_compare.py (cmp-fix-1027 version) so the gate does not depend on scratch;
tools/gates/mt_equiv_compare.py imports it. Original header:

arch025 S0b-1 comparer (CPU): A (arch025-base exe) vs B (S0a exe) per case, and B vs B2 (same S0a exe twice: the
server launch-table words that are time-dependent). Uses tools/impact/work_equal.py (ADR-0003: code objects + launch
sequence + golden) as a module, and classifies every server launch-table difference:
  args-known      only argument digests differ, only in kernels of the ignore file (get_rows_q8_0: launch-log tool defect)
  args-other      only argument digests differ, in other kernels                                  -> UNEXPLAINED
  seq-only        same rows and counts, different sequence hash                                    -> UNEXPLAINED
  seq-timing      seq-only AND the unit's own logs have equal finish lines + equal MTP / n-gram lines AND args
                  differ only in SEQ_KERNELS (per-cycle rows / state offsets; cmp-fix-1027, arch025-X4mt2)
                  -> timing class (passes only when also seen HEAD vs HEAD)
  (cmp-fix-1027) server logs: find_log takes '<unit>_server_logs' first and never another unit's folder;
  every srv_mt_* server run present on both sides compares its finish lines (differ -> NOT_EQUIVALENT,
  log missing -> INCOMPLETE)
  mtp-timing      kernel counts differ AND the per-request MTP / n-gram log lines of that server differ
                  (timing-driven draft count: engine.cpp pickDrafts on timing_, Ngram choose; arch024-S0b2 (3)B)
  env-tiers-off   kernel counts differ, MTP lines equal, a 'host tiers are off' line in either server log (arch024 (3)C)
  count-other     kernel counts differ with equal MTP lines and tiers on                           -> UNEXPLAINED
A case is
  EQUIVALENT              code objects identical, every launch unit SAME (server: after the ignore file), golden identical
                          (golden case), par4 texts equal (par4 cases)
  EQUIVALENT-MODULO-TIMING the same, except server runs whose A-vs-B difference is mtp-timing / env-tiers-off, a class
                          that also occurs B vs B2 (HEAD vs HEAD) for the same unit: not caused by the S0a hooks
  NOT_EQUIVALENT / INCOMPLETE otherwise (CLI run differs; UNEXPLAINED class; missing evidence)
Writes <root>\compare\cmp_<card>.txt / .json, whitelist_candidates_<card>.txt; with --decisions one TSV row per case
(+ the work hashes); with --import lt_import.py for every case's ll folder (results.db work_traces).

  python -I s0b1_compare.py --card r9700 --root <data>\\r9700 --exes <data>\\r9700\\exes.tsv --ignore-file ignore_server.txt
         [--decisions progress\\arch025.decisions.tsv] [--import] [--we <tools\\impact dir>] [--selftest]
Exit 0 every A/B case EQUIVALENT*, 1 some case NOT_EQUIVALENT, 2 some INCOMPLETE (nothing NOT_EQUIVALENT), 3 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

W = r"C:\ClaudeWorkspace\whirl-work"
DEF_WE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools", "impact")
LT_IMPORT = os.path.join(W, "progress", "tools", "lt_import.py")
PY = r"C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
MTP_RX = re.compile(r"(req \d+ \| slot \d+ \| (?:MTP: drafted|n-gram drafts in).*)$")
PRODUCT = ("whirl.exe", "whirl-server.exe")
AB_CASES = ("st", "golden", "mt", "par4_swift", "par4_moe")
HH_PAIRS = (("golden", "goldsrv"), ("mt", "mt"), ("par4_swift", "par4_swift"), ("par4_moe", "par4_moe"))

we = None
lt = None


def load_tools(path):
    global we, lt
    sys.path.insert(0, path)
    import launch_table as _lt  # noqa: E402
    import work_equal as _we  # noqa: E402
    we, lt = _we, _lt


def read_exes(path):
    sha, bins = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                sha.setdefault(p[0], {})[p[1]] = p[2]
                bins.setdefault(p[0], p[3])
    sha["B2"] = sha.get("B", {})
    bins["B2"] = bins.get("B", "")
    return sha, bins


SYM_SECTIONS = {".dynstr", ".dynsym", ".gnu.hash", ".hash", ".strtab", ".symtab", ".dynamic", ".rela.dyn"}
CUID_RX = re.compile(rb"__hip_cuid_[0-9a-f]+")


def cuid_only(pa, pb, e):
    """True when every differing code object differs only in symbol-table sections and the only printable strings that
    differ are __hip_cuid_<hash> (the per-TU id clang derives from the source path: two worktrees always differ there).
    Same rule as s15-blast §2 / PLAN I1 (.text / .rodata / .note equal). work_equal.py itself reports such exes as
    'different' (tool follow-up for the main session)."""
    objs = e["code_objects"].get("objects", [])
    if e["code_objects"]["result"] != "different" or not objs:
        return False
    bad = [o for o in objs if not o["same"]]
    if any(o.get("only") or not set(o.get("differing_sections") or ["?"]) <= SYM_SECTIONS for o in bad):
        return False
    da, db = open(pa, "rb").read(), open(pb, "rb").read()
    oa, ob = we.amdgpu_objects(da), we.amdgpu_objects(db)
    for o in bad:
        i = o["index"]
        xa = da[oa[i]["offset"]:oa[i]["offset"] + oa[i]["size"]]
        xb = db[ob[i]["offset"]:ob[i]["offset"] + ob[i]["size"]]
        sa, sb = strtab_names(xa), strtab_names(xb)
        if sa is None or sb is None or any(not CUID_RX.search(s) for s in sa ^ sb):
            return False
    return True


def strtab_names(blob):
    """the NUL-separated names of the .dynstr and .strtab sections of one ELF64 code object (None if unparsable)"""
    import struct
    try:
        shoff = struct.unpack_from("<Q", blob, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
        sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", blob, shoff + i * shentsize)
        stro = sh(shstrndx)[4]
        out = set()
        for i in range(shnum):
            s = sh(i)
            name = blob[stro + s[0]:blob.index(b"\0", stro + s[0])]
            if name in (b".dynstr", b".strtab"):
                out.update(x for x in blob[s[4]:s[4] + s[5]].split(b"\0") if x)
        return out
    except Exception:  # noqa: BLE001
        return None


def units(case_dir):
    """leaf launch folders under <case>\\ll: {rel: abs}"""
    ll = os.path.join(case_dir, "ll")
    out = {}
    if not os.path.isdir(ll):
        return out
    for root, _ds, fs in os.walk(ll):
        if any(f.startswith("launch_") and f.endswith(".txt") and not f.endswith(".raw.txt") for f in fs):
            out[os.path.relpath(root, ll).replace("\\", "/")] = root
    return out


def unit_has_server(d):
    for fn in os.listdir(d):
        if fn.startswith("launch_") and fn.endswith(".txt") and not fn.endswith(".raw.txt"):
            if lt.norm_cmd(lt.read_log(os.path.join(d, fn))[0]).startswith("whirl-server"):
                return True
    return False


UNIT_DIR_RX = re.compile(r"^(?:(.+)_server_logs|srvwork_(.+)|work_(.+)|(srv_.+))$")


def _dir_unit(name):
    """the unit a log folder belongs to by its name (release_gate '<unit>_server_logs', golden 'srvwork_<x>' = unit
    'srv_<x>', 'work_<unit>', a launch folder 'srv_<x>'), or None for a generic folder (work, out, ...)"""
    m = UNIT_DIR_RX.match(name)
    if not m:
        return None
    if m.group(1):
        return m.group(1)
    if m.group(2):
        return "srv_" + m.group(2)
    return m.group(3) or m.group(4)


def find_log(case_dir, unit, logname):
    """the server log named in a run key (--log-file NAME) of this unit.
    cmp-fix-1027 (arch025-X4mt2): 1. the exactly named '<unit>_server_logs' folder (release_gate layout
    rg\\<tag>\\<unit>_server_logs); 2. the unit's work folders (srvwork_<x> for srv_<x>, work_<unit>, <unit>);
    3. a generic folder (one whose name names no unit, e.g. par4 'work'). A log in a folder that belongs to ANOTHER
    unit is never returned (the old fallback took the first os.walk hit: srv_mt_precise was classified with the
    srv_mt_precise_moe logs) -> None = 'server log not found'."""
    if not logname:
        return None
    b = unit.split("/")[-1]
    exact, pref, generic = [], [], []
    for root, _ds, fs in os.walk(case_dir):
        if logname not in fs:
            continue
        p, d = os.path.join(root, logname), os.path.basename(root)
        owner = _dir_unit(d)
        if d == b + "_server_logs":
            exact.append(p)
        elif owner == b:
            pref.append(p)
        elif owner is None:
            generic.append(p)
    return (exact or pref or generic or [None])[0]


def mtp_lines(p):
    if not p or not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return [m.group(1) for m in (MTP_RX.search(x) for x in f) if m]


def tiers_off(p):
    if not p or not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8", errors="replace") as f:
        return any("host tiers are off" in x for x in f)


# cmp-fix-1027 (arch025-X4mt2): per-request finish line of a server log (output evidence: finish reason, completion
# tok, reasoning B, content B, tool calls, cache tok), timestamp stripped
FIN_RX = re.compile(r"(req \d+ \| slot \d+ \| finish_reason .*)$")
# kernels whose argument digests carry per-cycle row counts / state offsets (DeltaNet chunk / conv / state, KV row
# writes, the q8_0 row gather): with a wall-clock adaptive draft count (engine.cpp pickDrafts on timing_, dtiming_,
# msSince) the same total work lands in other cycles, so only these arguments and the sequence hash move
SEQ_KERNELS = frozenset({"gdn_chunk_scan", "gdn_conv_l2", "gdn_conv_l2n", "gdn_conv_state", "gdn_seq_128",
                         "gdn_step_norm", "set_rows", "get_rows_q8_0"})


def log_name(key):
    m = re.search(r"--log-file (\S+)", key)
    return m.group(1) if m else ("server_par4.log" if "--parallel 4" in key else None)


def _rx_lines(p, rx):
    if not p or not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return [m.group(1) for m in (rx.search(x) for x in f) if m]


def finish_lines(p):
    return _rx_lines(p, FIN_RX)


def classify(e, ign_rx, xcase, ycase, unit):
    """class of one DIFF run entry of launch_table.compare (raw, no ignore)"""
    kern = lambda rows: sorted({k.split()[0] for k, _a, _b in rows})
    rows, args = e.get("rows_diff", []), e.get("args_diff", [])
    logname = log_name(e["key"])
    info = {"key": e["key"], "launches": [e["n_a"], e["n_b"]], "count_kernels": kern(rows), "arg_kernels": kern(args), "log": logname}
    if not rows and e["n_a"] == e["n_b"]:
        if e["seq_a"] == e["seq_b"] and args:
            bad = [k for k in info["arg_kernels"] if not any(r.search(k + " ") for r in ign_rx)]
            info["class"] = "args-known" if not bad else "args-other"
            if bad:
                info["unexplained_kernels"] = bad
        else:
            info["class"] = "seq-only"
            seq_timing(info, xcase, ycase, unit)
        return info
    la, lb = find_log(xcase, unit, logname), find_log(ycase, unit, logname)
    ma, mb = mtp_lines(la), mtp_lines(lb)
    info["mtp_lines"] = [None if ma is None else len(ma), None if mb is None else len(mb)]
    if ma is not None and mb is not None and ma != mb:
        info["class"] = "mtp-timing"
        info["mtp_differing"] = sum(1 for x, y in zip(ma, mb) if x != y) + abs(len(ma) - len(mb))
    elif tiers_off(la) or tiers_off(lb):
        info["class"] = "env-tiers-off"
    else:
        info["class"] = "count-other"
        info["why"] = "MTP lines equal" if ma is not None and mb is not None else "server log not found"
    return info


def seq_timing(info, xcase, ycase, unit):
    """cmp-fix-1027 rule 2: a seq-only run (every (kernel, geometry) count equal, only the sequence hash and maybe
    argument digests differ) becomes 'seq-timing' when ALL hold: the unit's own server logs (find_log) have identical
    per-request finish lines and identical MTP / n-gram summary lines, and every argument-differing kernel is in
    SEQ_KERNELS. A timing class: case_verdict passes it only when it is also seen HEAD vs HEAD for the unit.
    Otherwise it stays 'seq-only' (UNEXPLAINED) with the failed condition in 'why'."""
    la, lb = find_log(xcase, unit, info.get("log")), find_log(ycase, unit, info.get("log"))
    fa, fb = finish_lines(la), finish_lines(lb)
    ma, mb = mtp_lines(la), mtp_lines(lb)
    ak = set(info.get("arg_kernels", []))
    why = []
    if fa is None or fb is None:
        why.append("server log not found")
    else:
        if fa != fb or not fa:
            why.append("finish lines differ" if fa != fb else "no finish lines")
        if ma != mb:
            why.append("MTP / n-gram lines differ")
    if not ak <= SEQ_KERNELS:
        why.append("args differ outside SEQ_KERNELS: " + ",".join(sorted(ak - SEQ_KERNELS)))
    info["finish_n"] = [None if fa is None else len(fa), None if fb is None else len(fb)]
    if why:
        info["why"] = "; ".join(why)
    else:
        info["class"] = "seq-timing"
        info["why"] = ("equal counts, equal finish + MTP lines (" + str(len(fa)) + " / " + str(len(ma or [])) + "), args differ only in "
                       + (",".join(sorted(ak)) or "(none)"))
    return info


def mt_finish(ux_dir, uy_dir, xcase, ycase, unit, lt_mod=None):
    """cmp-fix-1027 rule 3: the output check of an mt server unit (srv_mt_*): for every server run present on both
    sides, the per-request finish lines of the unit's own logs. -> [{'key', 'log', 'equal': True/False/None, 'n'}]
    (None = a log missing on a side). Other units -> []."""
    if not unit.split("/")[0].startswith("srv_mt_"):
        return []
    L = lt_mod or lt
    ta, tb = L.table_of(ux_dir, False, True), L.table_of(uy_dir, False, True)
    out = []
    for k in sorted(set(ta) & set(tb)):
        if not k.startswith("whirl-server"):
            continue
        ln = log_name(k)
        fa, fb = finish_lines(find_log(xcase, unit, ln)), finish_lines(find_log(ycase, unit, ln))
        out.append({"key": k, "log": ln, "equal": None if fa is None or fb is None or not (fa or fb) else fa == fb,
                    "n": [None if fa is None else len(fa), None if fb is None else len(fb)]})
    return out


def compare_units(xcase, ycase, xs, ys, ign, ign_rx):
    ux, uy = units(xcase), units(ycase)
    out = []
    for u in sorted(set(ux) | set(uy)):
        if u not in ux or u not in uy:
            out.append({"unit": u, "result": "missing", "why": f"only in {'A' if u in ux else 'B'}"})
            continue
        srv = unit_has_server(ux[u]) or unit_has_server(uy[u])
        r = we.compare_launch(ux[u], uy[u], server=srv, ignore=ign if srv else (), a_exes=xs, b_exes=ys)
        ent = {"unit": u, "server": srv, "result": r["result"], "n_same": r["n_same"], "n_runs": r["n_runs"],
               "binding": [r["binding"]["a"]["state"], r["binding"]["b"]["state"]], "reasons": r["reasons"], "classes": []}
        if srv:
            ent["finish"] = mt_finish(ux[u], uy[u], xcase, ycase, u)  # cmp-fix-1027 rule 3 (srv_mt_* only)
            raw = lt.compare(lt.table_of(ux[u], False, True), lt.table_of(uy[u], False, True))
            for e in raw:
                if e["status"] == "DIFF":
                    ent["classes"].append(classify(e, ign_rx, xcase, ycase, u))
                elif e["status"] != "SAME":
                    ent["classes"].append({"key": e["key"], "class": "one-side-only", "status": e["status"]})
        else:
            for e in r["runs"]:
                if e["status"] != "SAME":
                    ent["classes"].append({"key": e["key"], "class": "cli-" + e["status"].lower(),
                                           "rows_diff": e.get("rows_diff", [])[:5], "args_diff": e.get("args_diff", [])[:5]})
        out.append(ent)
    return out


def par4_cmp(xcase, ycase):
    try:
        a = json.load(open(os.path.join(xcase, "par4_results.json"), encoding="utf-8"))
        b = json.load(open(os.path.join(ycase, "par4_results.json"), encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        return {"result": "missing", "why": str(ex)[:200]}
    keys = sorted(set(a) | set(b))
    diff = [k for k in keys if a.get(k) != b.get(k)]
    return {"result": "identical" if keys and not diff else "missing" if not keys else "different", "keys": len(keys), "differing": diff}


def gate_pass(case_dir, name):
    p = os.path.join(case_dir, name)
    if not os.path.isfile(p):
        return None
    t = open(p, encoding="utf-8", errors="replace").read()
    return bool(re.search(r"\(ALL PASS\)|golden check gfx\d+: ok|release gate \S+: PASS", t))


def work_hash(case_dir):
    """sha256 over the sorted (unit, key, launches, seq, aseq, rows digest) of every run (server sequences included)"""
    h = hashlib.sha256()
    for u, d in sorted(units(case_dir).items()):
        for k, r in sorted(lt.table_of(d, False, True).items()):
            rd = hashlib.sha256("\n".join(f"{x} {r['rows'][x]}" for x in sorted(r["rows"])).encode()).hexdigest()[:16]
            h.update(f"{u}|{k}|{r['n']}|{r['seq']}|{r.get('aseq', '')}|{rd}\n".encode())
    return h.hexdigest()[:16]


SAFE = {"args-known", "mtp-timing", "env-tiers-off", "seq-timing"}


def case_verdict(units_res, hh_classes, extra):
    """hh_classes: {unit: set(classes seen B vs B2)}.
    cmp-fix-1027 rule 3: a unit's 'finish' list (mt_finish, srv_mt_* units) is checked first, also for launch-identical
    units: finish lines differ -> NOT_EQUIVALENT; a log missing -> INCOMPLETE."""
    notes = []
    modulo = False
    worst = "EQUIVALENT"
    for u in units_res:
        for fz in u.get("finish") or []:
            if fz["equal"] is False:
                worst = "NOT_EQUIVALENT"; notes.append(f"{u['unit']}: finish lines differ ({fz['log']}, {fz['n'][0]} vs {fz['n'][1]} lines) {fz['key'][:60]}")
            elif fz["equal"] is None:
                worst = "INCOMPLETE" if worst == "EQUIVALENT" else worst
                notes.append(f"{u['unit']}: finish lines not checked ({fz['log']}: log missing or no finish lines, n {fz['n']}) {fz['key'][:60]}")
        if u["result"] == "identical":
            continue
        if u["result"] == "missing" and not u.get("classes"):
            worst = "INCOMPLETE" if worst == "EQUIVALENT" else worst
            notes.append(f"{u['unit']}: missing ({'; '.join(u.get('reasons') or [u.get('why', '')])[:160]})")
            continue
        if not u.get("server"):
            worst = "NOT_EQUIVALENT"; notes.append(f"{u['unit']}: CLI launch table differs"); continue
        for c in u["classes"]:
            cl = c["class"]
            if cl == "args-known":
                continue
            if cl in SAFE and cl in hh_classes.get(u["unit"], set()):
                modulo = True
                notes.append(f"{u['unit']}: {cl} (also HEAD vs HEAD) {c['key'][:60]}")
            elif cl == "seq-timing":  # cmp-fix-1027 rule 2: without HEAD-vs-HEAD evidence it stays UNEXPLAINED
                worst = "NOT_EQUIVALENT"; notes.append(f"{u['unit']}: UNEXPLAINED seq-timing NOT seen HEAD vs HEAD {c['key'][:60]}")
            elif cl in SAFE:
                modulo = True
                notes.append(f"{u['unit']}: {cl} NOT seen HEAD vs HEAD (B vs B2 missing or different) {c['key'][:60]}")
                worst = "INCOMPLETE" if worst == "EQUIVALENT" else worst
            else:
                worst = "NOT_EQUIVALENT"; notes.append(f"{u['unit']}: UNEXPLAINED {cl} {c['key'][:60]}")
    for k, v in extra.items():
        if v == "different":
            worst = "NOT_EQUIVALENT"; notes.append(f"{k}: different")
        elif v in ("missing", None):
            worst = "INCOMPLETE" if worst == "EQUIVALENT" else worst; notes.append(f"{k}: missing")
    if worst == "EQUIVALENT" and modulo:
        worst = "EQUIVALENT-MODULO-TIMING"
    return worst, notes


def run(a):
    sha, bins = read_exes(a.exes)
    ign = lt.read_ignore_file(a.ignore_file) if a.ignore_file else []
    ign_rx = [re.compile(p) for p in ign]
    outd = os.path.join(a.root, "compare")
    os.makedirs(outd, exist_ok=True)
    lines = [f"# arch025 S0b-1 compare {a.card} {time.strftime('%Y-%m-%dT%H:%M:%S')}; root {a.root}; ignore {ign}"]
    rep = {"card": a.card, "root": a.root, "ignore": ign, "exes": {}, "hh": {}, "ab": {}, "b_only": {}}
    # code objects: product exes, A vs B (once)
    co_ok = True
    for n in PRODUCT:
        pa, pb = os.path.join(bins.get("A", ""), n), os.path.join(bins.get("B", ""), n)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            rep["exes"][n] = {"result": "missing"}; co_ok = False
            lines.append(f"code objects {n}: MISSING ({pa} / {pb})")
            continue
        e = we.compare_exe(pa, pb)
        cuid = cuid_only(pa, pb, e)
        if cuid:
            e["code_objects"]["result"] = "identical-except-cuid"
        stale = [s for s, p in (("A", pa), ("B", pb)) if sha.get(s, {}).get(n) and sha[s][n] != (e["a"] if s == "A" else e["b"])["sha256"]]
        rep["exes"][n] = {"code_objects": e["code_objects"]["result"], "host": e["host"]["result"], "n": [e["code_objects"]["n_a"], e["code_objects"]["n_b"]],
                          "sha": [e["a"]["sha256"], e["b"]["sha256"]], "rebuilt_since_run": stale}
        co_ok = co_ok and e["code_objects"]["result"] in ("identical", "identical-except-cuid") and not stale
        lines.append(f"code objects {n}: {e['code_objects']['result']} ({e['code_objects']['n_a']}/{e['code_objects']['n_b']}); host {e['host']['result']}; "
                     f"A {e['a']['sha256'][:16]} B {e['b']['sha256'][:16]}{'; EXE REBUILT SINCE THE RUN: ' + ','.join(stale) if stale else ''}")
    # HEAD vs HEAD: B vs B2
    hh_classes = {}
    for bc, b2c in HH_PAIRS:
        x, y = os.path.join(a.root, "B", bc), os.path.join(a.root, "B2", b2c)
        if not (os.path.isdir(x) and os.path.isdir(y)):
            lines.append(f"HH {bc} vs B2 {b2c}: missing"); continue
        ur = [u for u in compare_units(x, y, sha["B"], sha["B2"], ign, ign_rx) if u.get("server") or bc == "golden"]
        if bc == "golden":  # B golden holds the CLI runs too; B2 goldsrv only the server ones
            ur = [u for u in ur if u["unit"].startswith("srv_")]
        rep["hh"][bc] = ur
        for u in ur:
            hh_classes.setdefault(u["unit"], set()).update(c["class"] for c in u.get("classes", []))
            cl = ", ".join(f"{c['class']}" for c in u.get("classes", [])) or "-"
            lines.append(f"HH {bc}/{u['unit']}: {u['result']} ({u.get('n_same', 0)}/{u.get('n_runs', 0)} same after ignore); raw diff classes: {cl}")
    # A vs B per case
    worst_all = []
    for c in AB_CASES:
        x, y = os.path.join(a.root, "A", c), os.path.join(a.root, "B", c)
        if not (os.path.isdir(x) and os.path.isdir(y)):
            lines.append(f"AB {c}: missing (A {os.path.isdir(x)}, B {os.path.isdir(y)})")
            rep["ab"][c] = {"verdict": "INCOMPLETE", "notes": ["case folder missing"]}
            worst_all.append("INCOMPLETE")
            continue
        ur = compare_units(x, y, sha["A"], sha["B"], ign, ign_rx)
        extra = {"code_objects": "identical" if co_ok else "different" if any(v.get("code_objects") == "different" for v in rep["exes"].values()) else "missing",
                 "gate_A": "identical" if gate_pass(x, {"st": "release_gate.txt", "mt": "release_gate.txt", "golden": "golden.txt"}.get(c, "par4_gate.txt")) else "missing",
                 "gate_B": "identical" if gate_pass(y, {"st": "release_gate.txt", "mt": "release_gate.txt", "golden": "golden.txt"}.get(c, "par4_gate.txt")) else "missing"}
        g = None
        if c == "golden":
            g = we.compare_golden([os.path.join(x, "golden.txt")], [os.path.join(y, "golden.txt")], os.path.join(x, "out"), os.path.join(y, "out"),
                                  sha["A"], sha["B"])
            extra["golden"] = g["result"]
        p4 = None
        if c.startswith("par4"):
            p4 = par4_cmp(x, y)
            extra["par4_texts"] = p4["result"]
        v, notes = case_verdict(ur, hh_classes, extra)
        wa, wb = work_hash(x), work_hash(y)
        rep["ab"][c] = {"verdict": v, "notes": notes, "units": ur, "extra": extra, "work_hash": [wa, wb],
                        "golden": None if g is None else {k: g[k] for k in ("result", "reasons", "outputs", "binding")}, "par4": p4}
        worst_all.append(v)
        lines.append(f"AB {c}: {v}; units {len(ur)} ({sum(u['result'] == 'identical' for u in ur)} identical); work hash A {wa} B {wb}; "
                     + "; ".join(f"{k} {vv}" for k, vv in extra.items()))
        for n in notes[:12]:
            lines.append(f"    {n}")
        if g is not None:
            for r in g["reasons"][:4]:
                lines.append(f"    golden: {r}")
    for c in ("seq",):
        y = os.path.join(a.root, "B", c)
        if os.path.isdir(y):
            u = units(y)
            rep["b_only"][c] = {"units": sorted(u), "work_hash": work_hash(y), "gate": gate_pass(y, "release_gate.txt")}
            lines.append(f"B only {c}: {len(u)} launch units, gate PASS {rep['b_only'][c]['gate']}, work hash {rep['b_only'][c]['work_hash']} (baseline, no A)")
    # whitelist candidates: classes seen HEAD vs HEAD, per kernel
    cand = {}
    for bc, ur in rep["hh"].items():
        for u in ur:
            for c in u.get("classes", []):
                for k in c.get("arg_kernels", []) if c["class"].startswith("args") else []:
                    cand.setdefault(("args", k), set()).add(f"{bc}/{u['unit']}")
                if c["class"] in ("mtp-timing", "env-tiers-off", "count-other", "seq-only", "seq-timing"):
                    cand.setdefault((c["class"], "(whole run)"), set()).add(f"{bc}/{u['unit']}")
    wl = os.path.join(outd, f"whitelist_candidates_{a.card}.txt")
    with open(wl, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# arch025 S0b-1 {a.card}: launch-table differences seen HEAD vs HEAD (B vs B2, same S0a exe). Candidates only:\n")
        f.write("# an args row may enter ignore_server.txt only with a named cause (arch024 rule); whole-run classes cannot be row-ignored.\n")
        for (kind, k), where in sorted(cand.items()):
            known = any(r.search(k + " ") for r in ign_rx)
            f.write(f"{kind}\t{k}\t{'in ignore file' if known else 'NEW'}\t{','.join(sorted(where))}\n")
    order = {"EQUIVALENT": 0, "EQUIVALENT-MODULO-TIMING": 0, "INCOMPLETE": 2, "NOT_EQUIVALENT": 1}
    ex = 1 if "NOT_EQUIVALENT" in worst_all else 2 if "INCOMPLETE" in worst_all else 0
    summ = (f"SUMMARY {a.card} A vs B: " + ", ".join(f"{c} {rep['ab'][c]['verdict']}" for c in AB_CASES if c in rep["ab"])
            + f"; HH units {sum(len(v) for v in rep['hh'].values())}; whitelist candidates {len(cand)} ({sum(1 for (k, n) in cand if k == 'args' and not any(r.search(n + ' ') for r in ign_rx))} new args kernels)")
    lines.append(summ)
    rep["exit"] = ex
    with open(os.path.join(outd, f"cmp_{a.card}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, default=list)
    with open(os.path.join(outd, f"cmp_{a.card}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines:
        print(l)
    if a.decisions:
        new = not os.path.isfile(a.decisions)
        with open(a.decisions, "a", encoding="utf-8", newline="\n") as f:
            if new:
                f.write("time\tdecision\twhy\tevidence\tresult\n")
            ts = time.strftime("%Y-%m-%d %H:%M")
            for c, r in rep["ab"].items():
                f.write(f"{ts}\tS0b-1 {a.card} {c}: A (arch025-base) vs B (S0a) {r['verdict']}\tADR-0003 work equivalence (code objects + launch + golden)"
                        f"\t{outd}\\cmp_{a.card}.json\twork_hash A {r.get('work_hash', ['-', '-'])[0]} B {r.get('work_hash', ['-', '-'])[1]}; {'; '.join(r['notes'][:3])}\n")
            for c, r in rep["b_only"].items():
                f.write(f"{ts}\tS0b-1 {a.card} {c}: B-only baseline (S0a exe)\tno A run by plan\t{a.root}\\B\\{c}\twork_hash {r['work_hash']} gate {r['gate']}\n")
    if a.do_import:
        rc_all = 0
        for side in ("A", "B", "B2"):
            sd = os.path.join(a.root, side)
            if not os.path.isdir(sd):
                continue
            for c in sorted(os.listdir(sd)):
                ll = os.path.join(sd, c, "ll")
                if not os.path.isdir(ll):
                    continue
                cmd = [PY, "-I", LT_IMPORT, ll, "--gate-dir", os.path.join(sd, c), "--series", "arch025-s0b1", "--item", "arch025:S0b-1",
                       "--note", f"arch025 S0b-1 {a.card} side {side} case {c}"]
                for n, h in sha.get(side, {}).items():
                    cmd += ["--exe-sha", f"{n}={h}"]
                if a.dry_import:
                    cmd.append("--dry-run")
                p = subprocess.run(cmd, capture_output=True, text=True)
                rc_all |= p.returncode
                print(f"lt_import {side}/{c}: exit {p.returncode} {(p.stdout.strip().splitlines() or [''])[-1][:160]}")
        print(f"SUMMARY {a.card} lt_import exit {rc_all}")
    print(summ)
    return ex


def selftest():
    """synthetic A / B / B2 trees: identical -> EQUIVALENT; CLI change -> NOT_EQUIVALENT; get_rows args only -> EQUIVALENT;
    server count change with MTP lines differing, also HEAD vs HEAD -> MODULO-TIMING; count change with MTP equal -> NOT_EQUIVALENT"""
    import tempfile
    real_cmp = we.compare_exe

    def fake_cmp(pa, pb):  # the fake exes carry no AMDGPU code object: report them identical, keep the sha binding
        e = real_cmp(pa, pb)
        e["code_objects"] = {"result": "identical", "n_a": 1, "n_b": 1, "objects": []}
        return e
    we.compare_exe = fake_cmp
    srv = lambda d: rf'"{d}\whirl-server.exe" m.gguf --port 1 --log-file server_golden_a.log --parallel 1'
    cli = lambda d: rf'"{d}\whirl.exe" golden m.gguf --precise'
    res = []
    with tempfile.TemporaryDirectory() as td:
        bins = {s: os.path.join(td, "bin" + s) for s in ("A", "B")}
        for b in bins.values():
            os.makedirs(b)
            for n in PRODUCT + ("whirl-server-gate.exe",):
                open(os.path.join(b, n), "wb").write(b"MZ same")
        shaf = we.file_sha(os.path.join(bins["A"], "whirl.exe"))
        shas = we.file_sha(os.path.join(bins["A"], "whirl-server.exe"))
        with open(os.path.join(td, "exes.tsv"), "w") as f:
            for s, b in bins.items():
                for n in PRODUCT + ("whirl-server-gate.exe",):
                    f.write(f"{s}\t{n}\t{we.file_sha(os.path.join(b, n))}\t{b}\tx\n")

        def mk(root, side, case, cli_n=5, srv_rows=None, srv_args=None, mtp="drafted 10", tiers=False):
            b = bins["B" if side == "B2" else side]
            cd = os.path.join(root, side, case)
            ll = os.path.join(cd, "ll"); os.makedirs(ll, exist_ok=True)
            sl = os.path.join(ll, "srv_swift_precise"); os.makedirs(sl, exist_ok=True)
            if case != "goldsrv":
                lt._log(ll, 1, cli(b), cli_n, f"c{cli_n:03d}", {"k_a 1x1x1 64x1x1 0": cli_n}, "aa", {"k_a 1x1x1 64x1x1 0 0000000000000001": cli_n})
            rows = srv_rows or {"k_a 1x1x1 64x1x1 0": 10, "get_rows_q8_0 1x1x1 64x1x1 0": 2}
            args = srv_args or {"k_a 1x1x1 64x1x1 0 0000000000000001": 10, "get_rows_q8_0 1x1x1 64x1x1 0 00000000000000aa": 2}
            n = sum(rows.values())
            lt._log(sl, 2, srv(b), n, f"e{n:03d}", rows, "bb" if "00000000000000aa" in str(args) else "bc", args)
            wd = os.path.join(cd, "out", "srvwork_swift_precise"); os.makedirs(wd, exist_ok=True)
            with open(os.path.join(wd, "server_golden_a.log"), "w") as f:
                f.write(f"x I req 1 | slot 0 | MTP: {mtp}, accepted 5\n" + ("W adjusted: host tiers are off (RAM and SSD)\n" if tiers else ""))
            with open(os.path.join(cd, "golden.txt"), "w") as f:
                f.write(f"golden: exe {b}\\whirl.exe sha256 {shaf}\ngolden: server exe {b}\\whirl-server.exe sha256 {shas}\n"
                        "golden: reference set gfx1201 sha256 x; all files of gfx1201: " + "0" * 64 + "\nOK    swift/precise_single  x\ngolden check gfx1201: ok\n")
            os.makedirs(os.path.join(cd, "out"), exist_ok=True)
            open(os.path.join(cd, "out", "swift_precise_single.txt"), "w").write("G same\n")

        def tree(name, a_kw=None, b_kw=None, b2_kw=None):
            root = os.path.join(td, name)
            mk(root, "A", "golden", **(a_kw or {}))
            mk(root, "B", "golden", **(b_kw or {}))
            mk(root, "B2", "goldsrv", **(b2_kw or {}))
            return root
        cnt = {"srv_rows": {"k_a 1x1x1 64x1x1 0": 12, "get_rows_q8_0 1x1x1 64x1x1 0": 2}}
        cases = [
            ("identical -> EQUIVALENT", tree("same"), "EQUIVALENT"),
            ("CLI count change -> NOT_EQUIVALENT", tree("cli", b_kw={"cli_n": 6}), "NOT_EQUIVALENT"),
            ("get_rows_q8_0 args only -> EQUIVALENT", tree("gr", b_kw={"srv_args": {"k_a 1x1x1 64x1x1 0 0000000000000001": 10, "get_rows_q8_0 1x1x1 64x1x1 0 00000000000000ab": 2}}), "EQUIVALENT"),
            ("other kernel args -> NOT_EQUIVALENT", tree("oa", b_kw={"srv_args": {"k_a 1x1x1 64x1x1 0 0000000000000002": 10, "get_rows_q8_0 1x1x1 64x1x1 0 00000000000000aa": 2}}), "NOT_EQUIVALENT"),
            ("count + MTP differ, also HH -> MODULO-TIMING", tree("mt", b_kw=dict(cnt, mtp="drafted 12"), b2_kw={"mtp": "drafted 9", "srv_rows": {"k_a 1x1x1 64x1x1 0": 9, "get_rows_q8_0 1x1x1 64x1x1 0": 2}}), "EQUIVALENT-MODULO-TIMING"),
            ("count + MTP differ, not HH -> INCOMPLETE", tree("mt2", b_kw=dict(cnt, mtp="drafted 12"), b2_kw=dict(cnt, mtp="drafted 12")), "INCOMPLETE"),
            ("count differs, MTP equal -> NOT_EQUIVALENT", tree("cnt", b_kw=cnt), "NOT_EQUIVALENT"),
            ("count differs, tiers off, also HH -> MODULO-TIMING", tree("toff", b_kw=dict(cnt, tiers=True), b2_kw={"tiers": True, "srv_rows": {"k_a 1x1x1 64x1x1 0": 9, "get_rows_q8_0 1x1x1 64x1x1 0": 2}}), "EQUIVALENT-MODULO-TIMING"),
        ]
        bad = 0
        for name, root, want in cases:
            ns = argparse.Namespace(card="t", root=root, exes=os.path.join(td, "exes.tsv"), ignore_file=None, decisions=None, do_import=False, dry_import=False)
            ign_file = os.path.join(td, "ign.txt"); open(ign_file, "w").write("^get_rows_q8_0\\s\n"); ns.ignore_file = ign_file
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                run(ns)
            got = json.load(open(os.path.join(root, "compare", "cmp_t.json")))["ab"]["golden"]["verdict"]
            ok = got == want
            bad += 0 if ok else 1
            res.append(f"{'ok  ' if ok else 'FAIL'} {name}: {got}")
    for name, ok in selftest_cmpfix():
        bad += 0 if ok else 1
        res.append(f"{'ok  ' if ok else 'FAIL'} {name}")
    for r in res:
        print(r)
    print(f"s0b1_compare selftest: {len(res) - bad}/{len(res)} as expected")
    return 1 if bad else 0


def selftest_cmpfix():
    """cmp-fix-1027 rules on fake release_gate trees (rg\\<tag>\\<unit>_server_logs): find_log, seq-timing (with the
    arch025-X4mt2 prototype's negative cases), mt finish lines, case_verdict. -> [(name, ok)]"""
    import tempfile
    out = []
    MTP = "x I req {r} | slot 0 | MTP: drafted {d}, accepted 5, acceptance rate 50.0%, 2.00 tokens per cycle (5 verify cycles)\n"
    FIN = "x I req {r} | slot 0 | finish_reason stop, completion {c} tok, reasoning 0 B, content {b} B, tool_calls 0, cache now 90 tok\n"
    srv = lambda log: rf"C:\fake\Release\whirl-server.exe C:\m\Swift.gguf --port 1 --parallel 4 --log-file C:\w\{log}"
    rows = {"gdn_chunk_scan 1x1x1 64x1x1 0": 10, "set_rows 1x1x1 64x1x1 0": 4, "k_a 1x1x1 64x1x1 0": 6}
    args0 = {"gdn_chunk_scan 1x1x1 64x1x1 0 00000000000000a1": 10, "set_rows 1x1x1 64x1x1 0 00000000000000b1": 4,
             "k_a 1x1x1 64x1x1 0 00000000000000c1": 6}
    args_seq = {"gdn_chunk_scan 1x1x1 64x1x1 0 00000000000000a2": 10, "set_rows 1x1x1 64x1x1 0 00000000000000b2": 4,
                "k_a 1x1x1 64x1x1 0 00000000000000c1": 6}
    args_bad = {"gdn_chunk_scan 1x1x1 64x1x1 0 00000000000000a2": 10, "set_rows 1x1x1 64x1x1 0 00000000000000b1": 4,
                "k_a 1x1x1 64x1x1 0 00000000000000c2": 6}
    with tempfile.TemporaryDirectory() as t:
        def case(name, unit="srv_mt_precise", seq="c001", args=None, mtp=(10, 12), fin=((60, 135), (200, 587)),
                 other_first=True, own_log=True, logname="server_mtp_cache.log"):
            c = os.path.join(t, name)
            u = os.path.join(c, "ll", unit); os.makedirs(u)
            lt._log(u, 7, srv(logname), sum(rows.values()), seq, rows, "aa" if args is None else "a" + seq, args or args0)
            rg = os.path.join(c, "rg", "x4A1")
            # another unit's folder with the same log name, created first (the old os.walk fallback took it)
            for un, write in (("srv_mt_precise_moe", other_first), (unit, own_log)):
                if not write:
                    continue
                d = os.path.join(rg, un + "_server_logs"); os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, logname), "w", encoding="utf-8") as f:
                    if un == unit:
                        for i, (cc, bb) in enumerate(fin, 1):
                            f.write(MTP.format(r=i, d=mtp[i - 1]) + FIN.format(r=i, c=cc, b=bb))
                    else:
                        f.write(MTP.format(r=1, d=99) + FIN.format(r=1, c=1, b=1))
            return c, u

        # find_log
        c, _u = case("fl_exact")
        p = find_log(c, "srv_mt_precise", "server_mtp_cache.log")
        out.append(("find_log: exact <unit>_server_logs over another unit's folder",
                    p is not None and os.path.basename(os.path.dirname(p)) == "srv_mt_precise_server_logs"))
        c, _u = case("fl_foreign", own_log=False)
        out.append(("find_log: only another unit's folder has the log -> None (missing, never the other unit's)",
                    find_log(c, "srv_mt_precise", "server_mtp_cache.log") is None))
        g = os.path.join(t, "fl_generic", "work"); os.makedirs(g); open(os.path.join(g, "server_par4.log"), "w").close()
        out.append(("find_log: generic folder (par4 'work') still found",
                    find_log(os.path.join(t, "fl_generic"), "srv_par4_swift", "server_par4.log") is not None))
        s = os.path.join(t, "fl_srvwork", "out")
        for d in ("srvwork_moe_precise", "srvwork_swift_precise"):
            os.makedirs(os.path.join(s, d)); open(os.path.join(s, d, "server_golden_a.log"), "w").close()
        p = find_log(os.path.join(t, "fl_srvwork"), "srv_swift_precise", "server_golden_a.log")
        out.append(("find_log: golden srvwork_<x> of srv_<x>, not srvwork_moe_*",
                    p is not None and os.path.basename(os.path.dirname(p)) == "srvwork_swift_precise"))
        out.append(("find_log: golden unit with only another unit's srvwork -> None",
                    find_log(os.path.join(t, "fl_srvwork"), "srv_swift_fast", "server_golden_a.log") is None))

        # seq-timing via classify on real launch tables
        def cls(xa, yb, unit="srv_mt_precise"):
            ta, tb = lt.table_of(xa[1], False, True), lt.table_of(yb[1], False, True)
            e = [x for x in lt.compare(ta, tb) if x["status"] == "DIFF"]
            if len(e) != 1:
                return {"class": f"(no single DIFF: {len(e)})"}
            return classify(e[0], [re.compile(r"^get_rows_q8_0\s")], xa[0], yb[0], unit)
        A = case("A")
        r = cls(A, case("B_ok", seq="c002", args=args_seq))
        out.append((f"seq-only + equal finish + equal MTP + args only gdn_*/set_rows -> seq-timing (got {r['class']})", r["class"] == "seq-timing"))
        r = cls(A, case("B_fin", seq="c002", args=args_seq, fin=((60, 135), (200, 588))))
        out.append((f"negative: finish differs -> seq-only (got {r['class']}: {r.get('why')})", r["class"] == "seq-only"))
        r = cls(A, case("B_mtp", seq="c002", args=args_seq, mtp=(10, 13)))
        out.append((f"negative: MTP line differs -> seq-only (got {r['class']}: {r.get('why')})", r["class"] == "seq-only"))
        r = cls(A, case("B_arg", seq="c002", args=args_bad))
        out.append((f"negative: arg kernel k_a outside SEQ_KERNELS -> seq-only (got {r['class']}: {r.get('why')})", r["class"] == "seq-only"))
        r = cls(A, case("B_nolog", seq="c002", args=args_seq, own_log=False))
        out.append((f"negative: own log missing (only the other unit's) -> seq-only (got {r['class']}: {r.get('why')})", r["class"] == "seq-only"))
        r = cls(A, case("B_pure", seq="c002"))
        out.append((f"seq hash only, no arg difference, outputs equal -> seq-timing (got {r['class']})", r["class"] == "seq-timing"))

        # case_verdict: HH rule for seq-timing, finish lines
        st = {"unit": "srv_mt_precise_moe", "server": True, "result": "different", "classes": [{"key": "k", "class": "seq-timing"}]}
        out.append(("seq-timing also seen HEAD vs HEAD -> EQUIVALENT-MODULO-TIMING",
                    case_verdict([st], {"srv_mt_precise_moe": {"seq-timing"}}, {})[0] == "EQUIVALENT-MODULO-TIMING"))
        out.append(("negative: seq-timing NOT seen HEAD vs HEAD -> NOT_EQUIVALENT (UNEXPLAINED)",
                    case_verdict([st], {"srv_mt_precise_moe": {"mtp-timing"}}, {})[0] == "NOT_EQUIVALENT"))
        so = dict(st, classes=[{"key": "k", "class": "seq-only"}])
        out.append(("seq-only stays UNEXPLAINED even if seq-only seen HH -> NOT_EQUIVALENT",
                    case_verdict([so], {"srv_mt_precise_moe": {"seq-only"}}, {})[0] == "NOT_EQUIVALENT"))
        Bf = case("B_samelaunch_fin", fin=((60, 135), (200, 588)))
        f = mt_finish(A[1], Bf[1], A[0], Bf[0], "srv_mt_precise")
        ident = {"unit": "srv_mt_precise", "server": True, "result": "identical", "classes": [], "finish": f}
        out.append((f"launch identical but finish lines differ -> NOT_EQUIVALENT (finish {[x['equal'] for x in f]})",
                    case_verdict([ident], {}, {})[0] == "NOT_EQUIVALENT"))
        Bs = case("B_same")
        f = mt_finish(A[1], Bs[1], A[0], Bs[0], "srv_mt_precise")
        out.append((f"launch identical, finish equal -> EQUIVALENT (finish {[x['equal'] for x in f]})",
                    case_verdict([dict(ident, finish=f)], {}, {})[0] == "EQUIVALENT" and f and all(x["equal"] for x in f)))
        Bn = case("B_nofin", own_log=False)
        f = mt_finish(A[1], Bn[1], A[0], Bn[0], "srv_mt_precise")
        out.append((f"finish log missing on a side -> INCOMPLETE (finish {[x['equal'] for x in f]})",
                    case_verdict([dict(ident, finish=f)], {}, {})[0] == "INCOMPLETE"))
        out.append(("mt_finish: non-mt unit -> no check", mt_finish(A[1], Bs[1], A[0], Bs[0], "srv_swift_precise") == []))
        # end to end through compare_units (we.compare_launch + mt_finish + classify)
        ur = compare_units(A[0], case("B_e2e", seq="c002", args=args_seq)[0], {}, {}, [r"^get_rows_q8_0\s"], [re.compile(r"^get_rows_q8_0\s")])
        cl = [c["class"] for u in ur for c in u.get("classes", [])]
        v = case_verdict(ur, {"srv_mt_precise": {"seq-timing"}}, {})[0]
        out.append((f"compare_units end to end: seq-timing + finish equal + HH -> MODULO-TIMING (classes {cl}, {v})",
                    cl == ["seq-timing"] and v == "EQUIVALENT-MODULO-TIMING"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card")
    ap.add_argument("--root")
    ap.add_argument("--exes")
    ap.add_argument("--ignore-file")
    ap.add_argument("--decisions")
    ap.add_argument("--import", dest="do_import", action="store_true")
    ap.add_argument("--dry-import", action="store_true", help="lt_import --dry-run")
    ap.add_argument("--we", default=DEF_WE)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    load_tools(a.we)
    if a.selftest:
        return selftest()
    if not (a.card and a.root and a.exes):
        print("usage: --card --root --exes [--ignore-file] [--decisions] [--import]", file=sys.stderr)
        return 3
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
