# SPDX-License-Identifier: Apache-2.0
"""E8 speed judge of the Engine split (arch025-engine S0; DESIGN 7 E8, review3 C5 / A6 / C1).

GPU cells (server decode / prefill, anchor25.py output TSVs ab_<gpu>_<model>_<set>.tsv, one row per round and engine
label <gpu>-<tag>-<mode>): per cell (gpu, model, test, target, metric, mode) the paired AB/BA judge of abjudge.py with
band = max(2 % (abjudge SMALL), the A/A noise of that cell from the G-AA run (--aa-noise noise.tsv)).
  verdict PASS          abjudge IN / IN-small, or faster beyond the band
          EXTEND        INCONCLUSIVE (or EXTEND) with fewer than 10 rounds: add 2 rounds (AB/BA stays balanced)
          NOT ACCEPTED  INCONCLUSIVE after 10 rounds (C5): the step is not accepted; investigate with whirl-bench-check
                        (whirl-bench-check skill), never extend further and never widen the band
          SLOWER        OUT and slower: whirl-bench-check, the step is not accepted
Host cells (whirl-host-bench --suite sched jsonl, --host-a A1.jsonl,A2.. --host-b B1.jsonl,..; alternate the order):
  per (metric, cell) the paired judge too, printed as INFO (DESIGN 7 E8 (a): host time is not a gate); the work
  counters must be identical (a difference is a FAIL: the host path did different work).
--aa: the input is an A/A run (same exe as both tags): prints the noise per cell (to noise.tsv with --write-noise).
S0b (S0 review act-on 5, fail closed): the cells that must be judged come from --expect-cells (TSV gpu model test target
metric mode, a header line allowed) or else from the --aa-noise table; without either the judge refuses (exit 2). An
expected cell with no rows, or a cell with rows on one side only (every round of the new build ERROR), is MISSING = a
failure; rows whose note contains ERROR are counted and printed per file and tag (never dropped silently).

S0c (S0b review act-on 3; llama-bench exit(1) / hyperfine abort on any failed repetition): --expect-cells is required
(also with --aa; --aa-noise only gives the bands). A cell with any ERROR row or blank metric round (timeout, silent
failure) on either side is NOT ACCEPTED (the other rounds are never judged alone); so are unpaired rounds, an unbalanced
AB / BA order, more than MAX_ROUNDS pairs, and an expected cell without an A/A band when --aa-noise is given. An A/A run
that loses an expected cell fails and writes no noise table; a blank noise column is exit 4 with a message.

  python -I tools/bench/e8_judge.py --dir D --base base --new fin --expect-cells cells.tsv [--aa-noise noise.tsv]
                                    [--host-a ..] [--host-b ..]
  python -I tools/bench/e8_judge.py --dir D --base base --new aa --aa --expect-cells cells.tsv --write-noise noise.tsv
  python -I tools/bench/e8_judge.py --selftest
S0d (S0c review act-on 3, probes Q1-Q4): a row that names no cell (blank / malformed engine label) fails the whole run; a
(tag, round) seen twice (a retry appended), a row without rN in its note, a value <= 0 or not a number, a note saying
TIMEOUT / FAIL are bad rounds of their cell; a cell with fewer than MIN_ROUNDS (6, C5) pairs is NOT ACCEPTED. A crash of
the judge itself is exit 5 (never 1, which means "not accepted").
Exit 0 every expected GPU cell PASS (or a clean --aa), 1 a cell NOT ACCEPTED / SLOWER / MISSING / host work differs,
3 EXTEND needed, 2 usage (also: no expected-cell list, a bad expected-cell line), 4 a blank noise column in --aa-noise,
5 the judge crashed (traceback printed).
"""
import csv, glob, json, math, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abjudge  # noqa: E402

MAX_ROUNDS = 10
MIN_ROUNDS = 6
NOISE_MAX = 10.0   # S0e (S0d review consider 8): an A/A band above 10 % would pass any regression: the owner decides   # S0d (S0c review Q3): C5 starts at 6 rounds; fewer pairs is never accepted


def arg(n, d=None):
    a = sys.argv[1:]
    return a[a.index(n) + 1] if n in a else d


def load_gpu(d, base, new, errs=None, bad=None, orphans=None):
    """per cell {tag: {round: value}}. S0c (S0b review act-on 3; llama-bench exit(1) on any failed repetition): a row whose
    note says ERROR, or whose metric column is empty in a file that has the column (timeout / silent failure), is a BAD
    ROUND of its cell: bad[cell][tag][round] = why (evaluate() rejects a cell with any bad round, never judges the rest)."""
    cells = defaultdict(lambda: {base: {}, new: {}, "t": {base: {}, new: {}}})
    pend = []
    dup_seen = set()
    for path in sorted(glob.glob(os.path.join(d, "ab_*_*_*.tsv"))):
        gpu, model = os.path.basename(path)[:-4].split("_")[1:3]
        rd = csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")
        cols = [(met, col) for met, col in (("pp", "prefill_tps"), ("tg", "decode_tps")) if col in (rd.fieldnames or [])]
        for r in rd:
            note = r.get("note") or ""
            err = "ERROR" in note
            if err and errs is not None:   # S0b: counted per file / engine label (printed by evaluate)
                k = (os.path.basename(path), r.get("engine", "?"))
                errs[k] = errs.get(k, 0) + 1
            m = re.match(r"r(\d+)", note)
            rnd = int(m.group(1)) if m else None
            parts = (r.get("engine") or "").split("-")
            if len(parts) < 3:   # S0d (Q1): a row that names no cell cannot be attributed: the whole run fails
                if orphans is not None: orphans.append(f"{os.path.basename(path)}: engine label {r.get('engine')!r}, note {note!r}")
                continue
            tag, mode = parts[1], parts[-1]
            if tag not in (base, new): continue
            for met, col in cols:
                key = (gpu, model, r.get("test", ""), r.get("target", ""), met, mode)
                v = (r.get(col) or "").strip()
                if rnd is None:   # S0d (Q3): no rN in the note - rounds would collapse into one
                    pend.append((key, tag, "?", "no round number in the note")); continue
                if err or not v:
                    pend.append((key, tag, rnd, "ERROR" if err else "blank " + col))
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    fv = None
                if fv is None or not math.isfinite(fv) or not fv > 0 or re.search(r"TIMEOUT|FAIL", note, re.I):   # S0d (Q4)
                    pend.append((key, tag, rnd, f"value {v!r} / note {note.strip()!r}")); continue
                c = cells[key]
                if rnd in c[tag] or rnd in (bad or {}).get(key, {}).get(tag, {}) or (key, tag, rnd) in dup_seen:   # S0d (Q2)
                    pend.append((key, tag, rnd, "duplicate round (a retry appended)")); continue
                dup_seen.add((key, tag, rnd))
                c[tag][rnd] = fv
                if r.get("time"):
                    c["t"][tag][rnd] = min(c["t"][tag].get(rnd, r["time"]), r["time"])
    for key, tag, rnd, why in pend:
        # a blank metric belongs to a cell that has values somewhere (a test without prefill has no pp cell);
        # an ERROR row always marks its cells
        if why.startswith("blank") and key not in cells: continue
        cells[key][tag].pop(rnd, None)
        if bad is not None:
            bad.setdefault(key, {}).setdefault(tag, {})[rnd] = why
    return cells


def judge(b, f, first, band, final):
    o = abjudge.paired(b, f, first, small=band, final=final)
    v = o["verdict"]
    n = o["n_pairs"]
    if v in ("IN", "IN-small"): e8 = "PASS"
    elif v == "OUT": e8 = "PASS (faster)" if (o["delta_paired_pct"] or o["median_d_pct"] or 0) > 0 else "SLOWER"
    elif v in ("INCONCLUSIVE", "EXTEND"): e8 = "EXTEND" if n < MAX_ROUNDS else "NOT ACCEPTED (INCONCLUSIVE after 10 rounds: whirl-bench-check)"
    else: e8 = "MISSING"
    return o, e8


def load_noise(p):
    """S0c: (noise per cell, cells whose noise column is empty / not a number) - a blank never becomes a traceback"""
    out, blank = {}, []
    if p and os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if len(q) >= 6 and not l.startswith("gpu") and l.strip():
                try:
                    v = float(q[6])
                except (IndexError, ValueError):
                    v = None
                # S0e (S0d review act-on 5): inf / nan / negative is no band (blank, exit 4); above NOISE_MAX too
                if v is None or not math.isfinite(v) or v < 0 or v > NOISE_MAX:
                    blank.append(tuple(q[:6]) + ((f"value {q[6]!r}",) if len(q) > 6 else ()))
                else:
                    out[tuple(q[:6])] = v
    return out, blank


def host(alist, blist):
    """paired host-bench judge (INFO) + work identity (FAIL)"""
    def rd(fs):
        res = []
        for f in fs:
            d = {}
            for l in open(f, encoding="utf-8"):
                j = json.loads(l); d[(j["metric"], j["cell"])] = (j["median"], json.dumps(j.get("work"), sort_keys=True))
            res.append(d)
        return res
    A, B = rd(alist), rd(blist)
    lines, bad = [], 0
    for k in sorted(set(A[0]) & set(B[0])):
        rounds = range(1, min(len(A), len(B)) + 1)
        b = {i: A[i - 1][k][0] for i in rounds if k in A[i - 1]}
        f = {i: B[i - 1][k][0] for i in rounds if k in B[i - 1]}
        first = {i: ("base" if i % 2 else "fin") for i in rounds}
        o = abjudge.paired(b, f, first)
        works = {A[i - 1][k][1] for i in rounds if k in A[i - 1]} | {B[i - 1][k][1] for i in rounds if k in B[i - 1]}
        wsame = len(works) == 1
        bad += not wsame
        lines.append(f"HOST {k[0]} [{k[1]}] work {'=' if wsame else 'DIFFERS (FAIL)'} | {abjudge.line(o)} | {o['verdict']} (info)")
    return lines, bad


def load_cells(p):
    """S0b: the expected cells (gpu model test target metric mode), one per line, '#' comments and a header allowed"""
    out = set()
    for l in open(p, encoding="utf-8"):
        q = l.rstrip("\n").split("\t")
        if l.startswith(("#", "gpu\t")) or not l.strip(): continue
        if len(q) < 6: raise SystemExit(f"e8 judge: bad expected-cell line in {p}: {l.strip()[:80]}")
        out.add(tuple(q[:6]))
    return out


def main():
    if "--selftest" in sys.argv: return selftest()
    d, base, new = arg("--dir"), arg("--base", "base"), arg("--new")
    if not d or not new: print(__doc__); return 2
    lines, rc = evaluate(d, base, new, arg("--aa-noise"), arg("--expect-cells"), "--aa" in sys.argv, arg("--write-noise"),
                         arg("--host-a"), arg("--host-b"))
    for l in lines: print(l)
    return rc


def evaluate(d, base, new, noise_p=None, cells_p=None, aa=False, write_noise=None, host_a=None, host_b=None):
    # S0c (S0b review act-on 3): the expected-cell list is always --expect-cells (A/A too); --aa-noise only gives bands
    if not cells_p:
        return [f"e8 judge: NO EXPECTED-CELL LIST (give --expect-cells, also with --aa; --aa-noise only gives the band; "
                f"a missing cell must not pass)"], 2
    noise, nblank = load_noise(noise_p)
    if nblank:
        return [f"e8 judge: NOISE TABLE BLANK in {noise_p}: {', '.join(' '.join(k) for k in nblank)} (the A/A run lost these "
                f"cells: re-run G-AA; no band is invented)"], 4
    expect = load_cells(cells_p)
    errs, badr, orphans = {}, {}, []
    cells = load_gpu(d, base, new, errs, badr, orphans)
    for k in sorted(expect - set(cells)):
        cells[k]   # an expected cell without rows: MISSING (defaultdict creates the empty cell)
    out, bad, ext, nrows = [], 0, 0, []
    for (f, eng), n in sorted(errs.items()):
        out.append(f"ERROR rows: {f} {eng}: {n} (the cell is not accepted)")
    for o_ in orphans:   # S0d (S0c review Q1)
        out.append(f"ROW WITHOUT A CELL: {o_} (the run is not accepted)")
    for k in sorted(cells):
        c = cells[k]
        first = {r: abjudge.first_of(c["t"][base], c["t"][new], r) for r in set(c[base]) | set(c[new])}
        band = max(abjudge.SMALL, noise.get(k, 0.0))
        o, e8 = judge(c[base], c[new], first, band, final=len(set(c[base]) & set(c[new])) >= MAX_ROUNDS)
        # S0c: llama-bench / hyperfine rule - any failed round rejects the cell; rounds on one side only, an unbalanced
        # AB / BA order, more than MAX_ROUNDS pairs, or no A/A band for an expected cell when a noise table is given too
        nb = sum(len(v) for v in badr.get(k, {}).values())
        unp = sorted(set(c[base]) ^ set(c[new]))
        pairs = sorted(set(c[base]) & set(c[new]))
        nbf = sum(1 for r in pairs if first.get(r) == "base")
        why = None
        if nb:
            why = (f"NOT ACCEPTED ({nb} ERROR / blank rounds: "
                   + "; ".join(f"{t} r{','.join(map(str, sorted(v, key=str)))} {sorted(set(v.values()))[0]}" for t, v in sorted(badr[k].items()))
                   + ")")
        elif e8 == "MISSING":   # S0b: no pairs (no rows, or one side only) is a failure, never a pass
            why = (f"MISSING (base rounds {len(c[base])}, new rounds {len(c[new])}"
                   + ("; an expected cell" if k in expect else "; not in the expected-cell list") + ")")
        elif unp:
            why = f"NOT ACCEPTED (unpaired rounds {unp})"
        elif abs(nbf - (len(pairs) - nbf)) > 1:
            why = f"NOT ACCEPTED (order not balanced: base-first {nbf}, new-first {len(pairs) - nbf})"
        elif len(pairs) < MIN_ROUNDS:   # S0d (S0c review Q3)
            why = f"NOT ACCEPTED ({len(pairs)} pairs < {MIN_ROUNDS}: C5 starts at {MIN_ROUNDS} rounds)"
        elif len(pairs) > MAX_ROUNDS:
            why = f"NOT ACCEPTED ({len(pairs)} pairs > {MAX_ROUNDS}: rounds were extended past C5)"
        elif noise_p and k in expect and k not in noise and not aa:
            why = f"NOT ACCEPTED (NO A/A NOISE for this expected cell in {os.path.basename(noise_p)})"
        if aa:
            nz = o["noise_pct"]
            if why is None and (nz is None or not isinstance(nz, (int, float)) or not math.isfinite(nz)):
                why = "NOT ACCEPTED (no finite noise estimate)"
            elif why is None and o["verdict"] in ("EXTEND", "INCONCLUSIVE", "OUT"):   # S0e (act-on 5 X4)
                why = f"NOT ACCEPTED (A/A verdict {o['verdict']}: the two runs of one exe do not agree)"
            elif why is None and nz > NOISE_MAX:
                why = f"NOT ACCEPTED (A/A noise {nz:.2f}% > {NOISE_MAX}%: the owner decides)"
            if why is not None and k in expect:
                bad += 1
            if why is None:
                nrows.append("\t".join(k) + f"\t{o['noise_pct']}")
            out.append(f"AA {' '.join(k)} | {abjudge.line(o)} | A/A verdict {why or o['verdict']}"
                       + ("" if k in expect else " (not in the expected-cell list: info)"))
            continue
        if why is not None:
            e8 = why
        elif k not in expect:
            e8 += " (not in the expected-cell list: info)"
        bad += e8.startswith(("NOT ACCEPTED", "SLOWER", "MISSING"))
        ext += e8 == "EXTEND"
        out.append(f"E8 {' '.join(k)} band {band:.2f}% | {abjudge.line(o)} | {e8}")
    hl, hbad = ([], 0)
    if host_a and host_b:
        hl, hbad = host(host_a.split(","), host_b.split(","))
    lines = out + hl
    if orphans:   # S0d (S0c review Q1): a row no cell owns makes the whole run fail (A/A writes no noise table)
        return lines + [f"e8 judge: FAIL ({len(orphans)} rows name no cell: a crashed / malformed runner row is never a pass"
                        + ("; no noise table written" if aa else "") + ")"], 1
    if aa:
        if bad or hbad:   # S0c: an A/A that lost an expected cell writes no noise table (it would only hide the loss)
            return lines + [f"e8 judge: A/A FAIL ({bad} expected cells without a clean A/A, {hbad} host cells with different work; "
                            f"no noise table written)"], 1
        if write_noise:
            open(write_noise, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\tnoise_pct\n" + "\n".join(nrows) + "\n")
        return lines + [f"e8 judge: A/A {len(nrows)} cells"], 0
    ng = sum(1 for l in out if l.startswith("E8 "))
    if not ng:
        return lines + [f"e8 judge: NO GPU CELLS in {d} (nothing judged; host lines are info only)"], 2
    if bad or hbad:
        return lines + [f"e8 judge: FAIL ({bad} GPU cells not accepted or MISSING, {hbad} host cells with different work)"], 1
    if ext:
        return lines + [f"e8 judge: EXTEND ({ext} cells: run 2 more rounds)"], 3
    return lines + [f"e8 judge: PASS ({ng} GPU cells, all {len(expect)} expected cells present)"], 0


def selftest():
    fs = lambda n: {r: ("base" if r % 2 else "fin") for r in range(1, n + 1)}
    cases = []
    b = {r: 100.0 for r in range(1, 7)}
    cases.append(("6 rounds, +0.5 % -> PASS", judge(b, {r: 100.5 for r in b}, fs(6), 2.0, False)[1] == "PASS"))
    cases.append(("6 rounds, -6 % both orders -> SLOWER", judge(b, {r: 94.0 + (0.1 if r % 2 else -0.1) for r in b}, fs(6), 2.0, True)[1] == "SLOWER"))
    sp = {1: 100, 2: 92, 3: 99, 4: 93, 5: 100, 6: 92}
    cases.append(("6 rounds, order effect dominates -> EXTEND", judge(b, sp, fs(6), 2.0, False)[1] == "EXTEND"))
    b10 = {r: 100.0 for r in range(1, 11)}
    sp10 = {r: (100 if r % 2 else 92) for r in b10}
    cases.append(("10 rounds still INCONCLUSIVE -> NOT ACCEPTED (C5)", judge(b10, sp10, fs(10), 2.0, True)[1].startswith("NOT ACCEPTED")))
    cases.append(("band from A/A noise 5 %: -3 % -> PASS", judge(b, {r: 97.0 for r in b}, fs(6), 5.0, True)[1] == "PASS"))
    # S0b (S0 review act-on 5; reviewer case scratch/a25_engine/s0review/e8t): file-level cases through evaluate()
    import tempfile
    work = tempfile.mkdtemp(prefix="e8_selftest_")

    def tsv(name, rows):
        os.makedirs(os.path.join(work, name), exist_ok=True)
        with open(os.path.join(work, name, "ab_r9700_swift_short.tsv"), "w", encoding="utf-8") as o:
            o.write("engine\tnote\ttest\ttarget\tdecode_tps\ttime\n")
            for r in rows: o.write("\t".join(r) + "\n")
        return os.path.join(work, name)

    def rounds(mode, base_v, fin_v, fin_err=False, err_r=(), blank_r=(), tag_new="fin"):
        rows = []
        for r in range(1, 7):
            first, second = (("base", base_v), (tag_new, fin_v)) if r % 2 else ((tag_new, fin_v), ("base", base_v))
            for j, (tag, v) in enumerate((first, second)):
                err = tag == tag_new and (fin_err or r in err_r)
                blank = tag == tag_new and r in blank_r
                rows.append((f"r9700-{tag}-{mode}", f"r{r}" + (" ERROR" if err else "") + (" timeout" if blank else ""), "chat",
                             "512", "" if (err or blank) else f"{v}", f"2026-10-29T10:{r:02d}:{j}0"))
        return rows

    cells_p = os.path.join(work, "cells.tsv")
    open(cells_p, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\n"
                                               "r9700\tswift\tchat\t512\ttg\tprecise\nr9700\tswift\tchat\t512\ttg\tbalance\n")
    d_ok = tsv("ok", rounds("precise", 40.0, 40.1) + rounds("balance", 45.0, 45.1))
    d_err = tsv("err", rounds("precise", 40.0, 40.1) + rounds("balance", 45.0, 45.0, fin_err=True))
    d_one = tsv("one", rounds("precise", 40.0, 40.1))
    for nm, d, cp, want_rc, want_re in (
            ("control: both expected cells present and equal -> PASS rc 0", d_ok, cells_p, 0, r"^e8 judge: PASS \(2 GPU cells, all 2 expected"),
            ("balance: every new-version round ERROR -> FAIL rc 1 (review e8t)", d_err, cells_p, 1, r"balance band .*\| NOT ACCEPTED \(6 ERROR / blank rounds"),
            ("the ERROR rows are counted and printed", d_err, cells_p, 1, r"^ERROR rows: ab_r9700_swift_short\.tsv r9700-fin-balance: 6"),
            ("an expected cell with no rows at all -> MISSING, FAIL rc 1", d_one, cells_p, 1, r"balance band .*\| MISSING \(base rounds 0, new rounds 0; an expected cell"),
            ("no expected-cell list -> refuses, rc 2", d_ok, None, 2, r"^e8 judge: NO EXPECTED-CELL LIST")):
        lines, rc = evaluate(d, "base", "fin", None, cp)
        txt = "\n".join(lines)
        cases.append((nm, rc == want_rc and re.search(want_re, txt, re.M) is not None))
    # S0c (S0b review act-on 3; reviewer probes scratch/a25_engine/s0b_review/runner/r_e8.py P1-P5): llama-bench style,
    # any ERROR / blank round of a cell rejects the cell; A/A needs the expected-cell list; a blank noise is its own exit
    one_p = os.path.join(work, "cells1.tsv")
    open(one_p, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\nr9700\tswift\tchat\t512\ttg\tprecise\n")
    two_p = cells_p
    d_p1 = tsv("p1", rounds("precise", 40.0, 40.0, err_r=(1, 2, 3, 4)))
    d_p2 = tsv("p2", rounds("precise", 40.0, 40.0, err_r=(5, 6)))
    d_p3 = tsv("p3", rounds("precise", 40.0, 40.0, blank_r=(2, 4, 6)))
    d_p4 = tsv("p4", rounds("precise", 40.0, 40.0, tag_new="aa") + rounds("balance", 45.0, 45.0, fin_err=True, tag_new="aa"))
    d_p5 = tsv("p5", rounds("precise", 40.0, 40.0, tag_new="aa"))
    d_aa = tsv("aaok", rounds("precise", 40.0, 40.1, tag_new="aa") + rounds("balance", 45.0, 45.1, tag_new="aa"))
    nz_blank = os.path.join(work, "noise_blank.tsv")
    open(nz_blank, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\tnoise_pct\n"
                                                "r9700\tswift\tchat\t512\ttg\tprecise\t0.3\nr9700\tswift\tchat\t512\ttg\tbalance\t\n")
    nz_one = os.path.join(work, "noise_one.tsv")
    open(nz_one, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\tnoise_pct\nr9700\tswift\tchat\t512\ttg\tprecise\t0.3\n")
    for nm, args, want_rc, want_re in (
            ("P1 new build ERROR in 4 of 6 rounds -> NOT ACCEPTED rc 1", (d_p1, "base", "fin", None, one_p), 1,
             r"precise band .*\| NOT ACCEPTED \(4 ERROR / blank rounds"),
            ("P2 new build ERROR in rounds 5-6 -> NOT ACCEPTED rc 1", (d_p2, "base", "fin", None, one_p), 1,
             r"precise band .*\| NOT ACCEPTED \(2 ERROR / blank rounds"),
            ("P3 three silent blank rounds (timeout, no ERROR) -> NOT ACCEPTED rc 1", (d_p3, "base", "fin", None, one_p), 1,
             r"precise band .*\| NOT ACCEPTED \(3 ERROR / blank rounds"),
            ("P4a A/A with one side of an expected cell lost -> FAIL rc 1", (d_p4, "base", "aa", None, two_p, True), 1,
             r"^AA r9700 swift chat 512 tg balance .*NOT ACCEPTED \(6 ERROR / blank rounds"),
            ("P4b a blank noise column -> its own exit code 4, a message, no traceback", (d_ok, "base", "fin", nz_blank, two_p), 4,
             r"^e8 judge: NOISE TABLE BLANK .*r9700 swift chat 512 tg balance"),
            ("P5a A/A without an expected-cell list -> refuses rc 2", (d_p5, "base", "aa", None, None, True), 2,
             r"^e8 judge: NO EXPECTED-CELL LIST"),
            ("P5b A/A missing an expected cell -> FAIL rc 1", (d_p5, "base", "aa", None, two_p, True), 1,
             r"^AA r9700 swift chat 512 tg balance .*MISSING"),
            ("P5c --aa-noise alone is not an expected-cell list -> refuses rc 2", (d_ok, "base", "fin", nz_one, None), 2,
             r"^e8 judge: NO EXPECTED-CELL LIST"),
            ("P5d noise table without an expected cell -> FAIL rc 1", (d_ok, "base", "fin", nz_one, two_p), 1,
             r"balance band .*NO A/A NOISE"),
            ("control: A/A with both cells, expected list -> rc 0", (d_aa, "base", "aa", None, two_p, True), 0,
             r"^e8 judge: A/A 2 cells")):
        try:
            lines, rc = evaluate(*args)
        except BaseException as ex:   # a traceback is a failure of the case, never an exit code
            lines, rc = [f"EXCEPTION {type(ex).__name__}: {ex}"], -1
        txt = "\n".join(lines)
        cases.append((nm, rc == want_rc and re.search(want_re, txt, re.M) is not None))
    # S0d (S0c review act-on 3; reviewer probes scratch/a25_engine/s0c_review/runner/r_e8.py Q1-Q4)
    clean = rounds("precise", 40.0, 40.0)
    q1 = tsv("q1", clean + [("", "r7 ERROR", "chat", "512", "", "2026-10-29T10:07:00")])
    q2 = tsv("q2", clean + [("r9700-fin-precise", "r3", "chat", "512", "48.0", "2026-10-29T11:03:00")])
    q3 = tsv("q3", [(e, "", t, g, v, ti) for (e, _n, t, g, v, ti) in clean])
    q3b = tsv("q3b", [x for x in clean if not re.match(r"r[5-6]\b", x[1])])
    q4r = list(clean); q4r[3] = (q4r[3][0], "r2 TIMEOUT", q4r[3][2], q4r[3][3], "0", q4r[3][5])
    q4 = tsv("q4", q4r)
    for nm, d, want_rc, want_re in (
            ("Q1 an ERROR row with a blank engine label + 6 clean pairs -> FAIL rc 1", q1, 1, r"^e8 judge: FAIL \(1 rows name no cell"),
            ("Q2 a duplicate r3 of the new build (retry appended) -> NOT ACCEPTED rc 1", q2, 1, r"precise band .*NOT ACCEPTED \(1 ERROR / blank rounds: fin r3 duplicate round"),
            ("Q3 notes without rN -> NOT ACCEPTED rc 1", q3, 1, r"precise band .*NOT ACCEPTED .*no round number"),
            ("Q3b 4 clean pairs (< 6, C5) -> NOT ACCEPTED rc 1", q3b, 1, r"precise band .*NOT ACCEPTED \(4 pairs < 6"),
            ("Q4 a TIMEOUT row with value 0 -> NOT ACCEPTED rc 1 (no ZeroDivisionError)", q4, 1, r"precise band .*NOT ACCEPTED \(1 ERROR / blank rounds")):
        try:
            lines, rc = evaluate(d, "base", "fin", None, one_p)
        except BaseException as ex:
            lines, rc = [f"EXCEPTION {type(ex).__name__}: {ex}"], -1
        txt = "\n".join(lines)
        cases.append((nm, rc == want_rc and re.search(want_re, txt, re.M) is not None))
    # S0e (S0d review act-on 5; reviewer probes scratch/a25_engine/s0d_review/runner/r_e8.py X1-X4): non-finite values
    def over(rows, fn):
        return [(e, n, t, g, fn(e, n, v), ti) for (e, n, t, g, v, ti) in rows]
    x1 = tsv("x1", over(clean, lambda e, n, v: "inf" if (e.endswith("-fin-precise") and n.startswith("r3")) else v))
    x1b = tsv("x1b", over(clean, lambda e, n, v: "inf" if "-fin-" in e else v))
    x1c = tsv("x1c", over(clean, lambda e, n, v: "nan" if "-fin-" in e else v))
    slow = tsv("x3", over(clean, lambda e, n, v: "32.0" if "-fin-" in e else v))
    for nm, d, want_rc, want_re in (
            ("X1 one new-build round inf -> NOT ACCEPTED rc 1", x1, 1, r"precise band .*NOT ACCEPTED \(1 ERROR / blank rounds"),
            ("X1b every new-build round inf -> NOT ACCEPTED rc 1", x1b, 1, r"precise band .*NOT ACCEPTED \(6 ERROR / blank rounds"),
            ("X1c every new-build round nan -> NOT ACCEPTED rc 1", x1c, 1, r"precise band .*NOT ACCEPTED \(6 ERROR / blank rounds")):
        lines, rc = evaluate(d, "base", "fin", None, one_p)
        cases.append((nm, rc == want_rc and re.search(want_re, "\n".join(lines), re.M) is not None))
    for nz in ("inf", "nan", "-5", "1e9", "abc"):
        p = os.path.join(work, f"noise_{nz}.tsv")
        open(p, "w", encoding="utf-8").write(f"gpu\tmodel\ttest\ttarget\tmetric\tmode\tnoise_pct\nr9700\tswift\tchat\t512\ttg\tprecise\t{nz}\n")
        lines, rc = evaluate(slow, "base", "fin", p, one_p)
        cases.append((f"X3 noise_pct {nz}, new build 20 % slower -> NOISE TABLE BLANK exit 4 (never PASS)",
                      rc == 4 and re.search(r"^e8 judge: NOISE TABLE BLANK", "\n".join(lines), re.M) is not None))
    x4 = tsv("x4", [(e.replace("-fin-", "-aa-"), n, t, g, ("inf" if "-fin-" in e else v), ti) for (e, n, t, g, v, ti) in clean])
    nz4 = os.path.join(work, "x4_noise.tsv")
    if os.path.exists(nz4): os.remove(nz4)
    lines, rc = evaluate(x4, "base", "aa", None, one_p, True, nz4)
    cases.append(("X4 A/A with every aa round inf -> A/A FAIL rc 1, no noise table", rc == 1 and not os.path.exists(nz4)
                  and re.search(r"^e8 judge: A/A FAIL", "\n".join(lines), re.M) is not None))
    # a crash of the judge is exit 5, a bad expected-cell line is a usage error 2 (never 1 = "not accepted")
    import subprocess
    badc = os.path.join(work, "cells_bad.tsv"); open(badc, "w", encoding="utf-8").write("r9700\tswift\n")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--dir", d_ok, "--new", "fin", "--expect-cells", badc],
                       capture_output=True, text=True)
    cases.append(("a bad expected-cell line -> usage exit 2", r.returncode == 2 and "USAGE ERROR" in r.stdout))
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import e8_judge as J; "
                        "J.evaluate = lambda *a, **k: 1 / 0; sys.argv = ['e8_judge.py', '--dir', 'x', '--new', 'fin']; sys.exit(J.run_main())",
                        os.path.dirname(os.path.abspath(__file__))], capture_output=True, text=True)
    cases.append(("a crash inside the judge -> exit 5 with CRASH, not 1", r.returncode == 5 and "e8 judge: CRASH" in r.stdout))
    ok = all(g for _, g in cases)
    for n, g in cases: print(f"  {'ok  ' if g else 'FAIL'} {n}")
    print(f"e8 judge selftest: {sum(g for _, g in cases)}/{len(cases)} as expected")
    return 0 if ok else 1


def run_main():
    """S0d (S0c review act-on 3 Q4): exit 1 means 'not accepted' only; a usage error is 2, a crash of the judge is 5"""
    try:
        return main()
    except SystemExit as ex:
        if isinstance(ex.code, int) or ex.code is None: return ex.code or 0
        print(ex.code); print("e8 judge: USAGE ERROR"); return 2
    except BaseException:
        import traceback
        traceback.print_exc()
        print("e8 judge: CRASH (exit 5: the judge failed, not the cells; nothing was judged)")
        return 5


if __name__ == "__main__":
    sys.exit(run_main())
