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

  python -I tools/bench/e8_judge.py --dir D --base base --new fin (--aa-noise noise.tsv | --expect-cells cells.tsv)
                                    [--host-a ..] [--host-b ..]
  python -I tools/bench/e8_judge.py --dir D --base base --new aa --aa --write-noise noise.tsv
  python -I tools/bench/e8_judge.py --selftest
Exit 0 every expected GPU cell PASS (or --aa), 1 a cell NOT ACCEPTED / SLOWER / MISSING / host work differs, 3 EXTEND
needed, 2 usage (also: no expected-cell list).
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abjudge  # noqa: E402

MAX_ROUNDS = 10


def arg(n, d=None):
    a = sys.argv[1:]
    return a[a.index(n) + 1] if n in a else d


def load_gpu(d, base, new, errs=None):
    cells = defaultdict(lambda: {base: {}, new: {}, "t": {base: {}, new: {}}})
    for path in sorted(glob.glob(os.path.join(d, "ab_*_*_*.tsv"))):
        gpu, model = os.path.basename(path)[:-4].split("_")[1:3]
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
            if "ERROR" in r.get("note", ""):
                if errs is not None:   # S0b: counted per file / engine label (printed by evaluate)
                    k = (os.path.basename(path), r.get("engine", "?"))
                    errs[k] = errs.get(k, 0) + 1
                continue
            m = re.match(r"r(\d+)", r.get("note", ""))
            rnd = int(m.group(1)) if m else 0
            parts = r["engine"].split("-")
            tag, mode = parts[1], parts[-1]
            if tag not in (base, new): continue
            for met, col in (("pp", "prefill_tps"), ("tg", "decode_tps")):
                if r.get(col):
                    c = cells[(gpu, model, r["test"], r["target"], met, mode)]
                    c[tag][rnd] = float(r[col])
                    if r.get("time"):
                        c["t"][tag][rnd] = min(c["t"][tag].get(rnd, r["time"]), r["time"])
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
    out = {}
    if p and os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if len(q) == 7 and not l.startswith("gpu"):
                out[tuple(q[:6])] = float(q[6])
    return out


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
    noise = load_noise(noise_p)
    if not aa and not cells_p and not noise:
        return [f"e8 judge: NO EXPECTED-CELL LIST (give --expect-cells or --aa-noise; a missing cell must not pass)"], 2
    expect = load_cells(cells_p) if cells_p else set(noise)
    errs = {}
    cells = load_gpu(d, base, new, errs)
    if not aa:
        for k in sorted(expect - set(cells)):
            cells[k]   # an expected cell without rows: MISSING (defaultdict creates the empty cell)
    out, bad, ext, nrows = [], 0, 0, []
    for (f, eng), n in sorted(errs.items()):
        out.append(f"ERROR rows: {f} {eng}: {n} (not judged)")
    for k in sorted(cells):
        c = cells[k]
        first = {r: abjudge.first_of(c["t"][base], c["t"][new], r) for r in set(c[base]) | set(c[new])}
        band = max(abjudge.SMALL, noise.get(k, 0.0))
        o, e8 = judge(c[base], c[new], first, band, final=len(set(c[base]) & set(c[new])) >= MAX_ROUNDS)
        if aa:
            nrows.append("\t".join(k) + f"\t{o['noise_pct'] if o['noise_pct'] is not None else ''}")
            out.append(f"AA {' '.join(k)} | {abjudge.line(o)} | A/A verdict {o['verdict']}")
            continue
        if e8 == "MISSING":   # S0b: no pairs (no rows, or one side only) is a failure, never a pass
            e8 = (f"MISSING (base rounds {len(c[base])}, new rounds {len(c[new])}"
                  + ("; an expected cell" if k in expect else "; not in the expected-cell list") + ")")
        elif k not in expect:
            e8 += " (not in the expected-cell list: info)"
        bad += e8.startswith(("NOT ACCEPTED", "SLOWER", "MISSING"))
        ext += e8 == "EXTEND"
        out.append(f"E8 {' '.join(k)} band {band:.2f}% | {abjudge.line(o)} | {e8}")
    hl, hbad = ([], 0)
    if host_a and host_b:
        hl, hbad = host(host_a.split(","), host_b.split(","))
    lines = out + hl
    if aa:
        if write_noise:
            open(write_noise, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\tnoise_pct\n" + "\n".join(nrows) + "\n")
        return lines + [f"e8 judge: A/A {len(nrows)} cells{', host work differs in ' + str(hbad) if hbad else ''}"], (1 if hbad else 0)
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

    def rounds(mode, base_v, fin_v, fin_err=False):
        rows = []
        for r in range(1, 7):
            first, second = (("base", base_v), ("fin", fin_v)) if r % 2 else (("fin", fin_v), ("base", base_v))
            for j, (tag, v) in enumerate((first, second)):
                err = fin_err and tag == "fin"
                rows.append((f"r9700-{tag}-{mode}", f"r{r}" + (" ERROR" if err else ""), "chat", "512", "" if err else f"{v}",
                             f"2026-10-29T10:{r:02d}:{j}0"))
        return rows

    cells_p = os.path.join(work, "cells.tsv")
    open(cells_p, "w", encoding="utf-8").write("gpu\tmodel\ttest\ttarget\tmetric\tmode\n"
                                               "r9700\tswift\tchat\t512\ttg\tprecise\nr9700\tswift\tchat\t512\ttg\tbalance\n")
    d_ok = tsv("ok", rounds("precise", 40.0, 40.1) + rounds("balance", 45.0, 45.1))
    d_err = tsv("err", rounds("precise", 40.0, 40.1) + rounds("balance", 45.0, 45.0, fin_err=True))
    d_one = tsv("one", rounds("precise", 40.0, 40.1))
    for nm, d, cp, want_rc, want_re in (
            ("control: both expected cells present and equal -> PASS rc 0", d_ok, cells_p, 0, r"^e8 judge: PASS \(2 GPU cells, all 2 expected"),
            ("balance: every new-version round ERROR -> MISSING, FAIL rc 1 (review e8t)", d_err, cells_p, 1, r"\| MISSING \(base rounds 6, new rounds 0"),
            ("the ERROR rows are counted and printed", d_err, cells_p, 1, r"^ERROR rows: ab_r9700_swift_short\.tsv r9700-fin-balance: 6"),
            ("an expected cell with no rows at all -> MISSING, FAIL rc 1", d_one, cells_p, 1, r"balance band .*\| MISSING \(base rounds 0, new rounds 0; an expected cell"),
            ("no expected-cell list -> refuses, rc 2", d_ok, None, 2, r"^e8 judge: NO EXPECTED-CELL LIST")):
        lines, rc = evaluate(d, "base", "fin", None, cp)
        txt = "\n".join(lines)
        cases.append((nm, rc == want_rc and re.search(want_re, txt, re.M) is not None))
    ok = all(g for _, g in cases)
    for n, g in cases: print(f"  {'ok  ' if g else 'FAIL'} {n}")
    print(f"e8 judge selftest: {sum(g for _, g in cases)}/{len(cases)} as expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
