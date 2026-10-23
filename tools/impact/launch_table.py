# SPDX-License-Identifier: Apache-2.0
"""Kernel launch tables (ARCH-1 K-0): what WHIRL launches, kernel name + grid + block + shared memory.

A process run with WHIRL_LAUNCH_LOG=<folder> writes <folder>/launch_<pid>.txt at exit (src/hip/launch_log.cpp):
its command line, the launch count, a hash of the whole launch sequence, and one line per distinct
(kernel, grid, block, shared bytes) with its count. This tool turns a folder of those into a table keyed by the
normalized command line, and compares two tables: the kernel-abstraction steps K-1..K-5 must reproduce the K-0
table exactly (same kernels, same grids, same sequence).

    python tools/impact/launch_table.py table <folder> [-o table.txt]       # one table per run folder
    python tools/impact/launch_table.py diff <table_or_folder_a> <table_or_folder_b>

REVIEW-ARCH4 P3-2: logs from builds with the argument digest also carry, per (kernel, geometry, argument digest),
a count ("@" lines) and an argument-sequence hash; when both sides have them they are compared too (scalars and
allocation-relative pointer offsets: head0, r0 / r1, copy offsets, n_split, group tables). A side without them (an
older build) is compared on kernels and geometry only; the summary line says in how many runs arguments were compared.

Only deterministic processes are compared by default: whirl.exe CLI runs (golden single / pair / ngram, seqtest,
selftest). Server processes (whirl-server.exe) batch requests by arrival time, so their sequences legitimately
vary; `--all` includes them (count tables only, no sequence hash).

arch024-S0a (REVIEW-ARCH6-code C-4 / REVIEW-ARCH6 C-M4): `--server` includes whirl-server.exe runs with their
sequence hashes kept, so the server's multi-slot / batch paths are pinned too. Whether a server table is deterministic
for a given workload is what S0b proves (same build twice, HEAD vs HEAD); kernels whose counts depend on time (tier
IO completion, prefetch, idle flush, governor ticks, loop wake-ups) are named with `--ignore REGEX` (repeatable,
matched against the row key "kernel grid block shared"; with `--ignore-file`, one regex per line, # comments). A
run with an ignored row has its sequence hashes not compared (they cover every launch) and says so.

    python tools/impact/launch_table.py diff A B --server [--ignore REGEX ...] [--ignore-file F]
    python tools/impact/launch_table.py selftest
"""
from __future__ import annotations

import argparse
import os
import re
import sys


def norm_cmd(cmd: str) -> str:
    """Command line without folders, ports and temp names: the exe base name and the arguments."""
    toks = re.findall(r'"[^"]*"|\S+', cmd)
    out = []
    skip = False
    for t in toks:
        t = t.strip('"')
        if skip:
            skip = False
            continue
        if t in ("--port", "--work", "--out", "--kv-ssd-dir", "--log"):
            skip = True
            continue
        if "\\" in t or "/" in t:
            t = re.split(r"[\\/]", t)[-1]
        out.append(t)
    if out:
        out[0] = out[0].lower()
        if not out[0].endswith(".exe"):
            out[0] += ".exe"
    return " ".join(out)


def read_log(path):
    cmd, n, seq, rows, aseq, args = "", 0, "", {}, "", None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# cmd: "):
                cmd = line[7:]
            elif line.startswith("# launches "):
                m = re.match(r"# launches (\d+) seqhash ([0-9a-f]+)", line)
                if m:
                    n, seq = int(m.group(1)), m.group(2)
            elif line.startswith("# argseqhash "):
                aseq, args = line.split()[2], {}
            elif line.startswith("@ "):
                k, _, c = line[2:].rpartition(" ")
                args[k] = args.get(k, 0) + int(c)
            elif line and not line.startswith("#"):
                k, _, c = line.rpartition(" ")
                rows[k] = rows.get(k, 0) + int(c)
    return cmd, n, seq, rows, aseq, args


def table_of(folder, include_all=False, server_seq=False):
    """include_all: server runs as counts only (no sequence hash); server_seq: server runs with their hashes."""
    include_all = include_all or server_seq
    runs = {}
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("launch_") and fn.endswith(".txt")) or fn.endswith(".raw.txt"):
            continue  # launch_<pid>.raw.txt: the WHIRL_LAUNCH_LOG_RAW per-launch dump (arch6e/rawcmp.py reads it)
        cmd, n, seq, rows, aseq, args = read_log(os.path.join(folder, fn))
        key = norm_cmd(cmd)
        server = key.startswith("whirl-server")
        if server and not include_all:
            continue
        if n == 0:
            continue  # a process that launched nothing (help, listing)
        if key in runs:
            # the same command twice in one folder (a rerun): keep both under a suffix, compared in order
            i = 2
            while f"{key} #{i}" in runs:
                i += 1
            key = f"{key} #{i}"
        drop = server and not server_seq
        runs[key] = {"n": n, "seq": "" if drop else seq, "rows": rows, "aseq": "" if drop else aseq, "args": args}
    return runs


def apply_ignore(runs, pats):
    """Rows (and "@" rows) whose key matches one of pats are left out; such a run's launch count and sequence
    hashes are cleared (they cover every launch) and the run is marked."""
    if not pats:
        return runs
    rx = [re.compile(p) for p in pats]
    out = {}
    for key, r in runs.items():
        hit = lambda k: any(x.search(k) for x in rx)
        rows = {k: v for k, v in r["rows"].items() if not hit(k)}
        args = None if r.get("args") is None else {k: v for k, v in r["args"].items() if not hit(k.rsplit(" ", 1)[0])}
        cut = len(rows) != len(r["rows"]) or (args is not None and len(args) != len(r["args"]))
        out[key] = dict(r, rows=rows, args=args)
        if cut:
            out[key].update(n=sum(rows.values()), seq="", aseq="" if args is not None else r.get("aseq", ""), ignored=True)
    return out


def read_ignore_file(path):
    pats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(line)
    return pats


def write_table(runs, out):
    for key in sorted(runs):
        r = runs[key]
        out.write(f"== {key}\n   launches {r['n']} seqhash {r['seq'] or '-'} kinds {len(r['rows'])}\n")
        if r.get("args") is not None:
            out.write(f"   argseqhash {r['aseq'] or '-'} argkinds {len(r['args'])}\n")
        for k in sorted(r["rows"]):
            out.write(f"   {k} {r['rows'][k]}\n")
        for k in sorted(r.get("args") or {}):
            out.write(f"   @ {k} {r['args'][k]}\n")


def read_table(path):
    runs, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("== "):
                cur = {"n": 0, "seq": "", "rows": {}, "aseq": "", "args": None}
                runs[line[3:]] = cur
            elif line.startswith("   launches "):
                m = re.match(r"   launches (\d+) seqhash (\S+)", line)
                cur["n"], cur["seq"] = int(m.group(1)), "" if m.group(2) == "-" else m.group(2)
            elif line.startswith("   argseqhash "):
                v = line.split()[1]
                cur["aseq"], cur["args"] = "" if v == "-" else v, {}
            elif line.startswith("   @ "):
                k, _, c = line[5:].rpartition(" ")
                cur["args"][k] = int(c)
            elif line.startswith("   "):
                k, _, c = line.strip().rpartition(" ")
                cur["rows"][k] = int(c)
    return runs


def load(p, include_all, server_seq=False, ignore=()):
    runs = table_of(p, include_all, server_seq) if os.path.isdir(p) else read_table(p)
    return apply_ignore(runs, ignore)


def diff(a, b):
    bad = 0
    n_args = 0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(f"ONLY {'B' if key not in a else 'A'}  {key}")
            bad += 1
            continue
        ra, rb = a[key], b[key]
        same_rows = ra["rows"] == rb["rows"]
        same_seq = ra["seq"] == rb["seq"]
        has_args = ra.get("args") is not None and rb.get("args") is not None
        same_args = not has_args or (ra["args"] == rb["args"] and ra["aseq"] == rb["aseq"])
        n_args += 1 if has_args else 0
        if same_rows and same_seq and same_args and ra["n"] == rb["n"]:
            extra = f", arguments {len(ra['args'])} kinds" if has_args else ""
            if ra.get("ignored") or rb.get("ignored"):
                extra += ", ignored rows left out, sequence not compared"
            print(f"SAME  {key}  ({ra['n']} launches, {len(ra['rows'])} kinds{extra})")
            continue
        bad += 1
        extra = f", argseqhash {ra['aseq'] or '-'} vs {rb['aseq'] or '-'}" if has_args else ""
        print(f"DIFF  {key}  launches {ra['n']} vs {rb['n']}, seqhash {ra['seq'] or '-'} vs {rb['seq'] or '-'}{extra}")
        for k in sorted(set(ra["rows"]) | set(rb["rows"])):
            ca, cb = ra["rows"].get(k, 0), rb["rows"].get(k, 0)
            if ca != cb:
                print(f"      {k}: {ca} vs {cb}")
        if has_args:
            for k in sorted(set(ra["args"]) | set(rb["args"])):
                ca, cb = ra["args"].get(k, 0), rb["args"].get(k, 0)
                if ca != cb:
                    print(f"      @ {k}: {ca} vs {cb}")
    print(f"launch tables: {len(set(a) | set(b)) - bad} same, {bad} different; arguments compared in {n_args}")
    return 1 if bad else 0


def _log(folder, pid, cmd, launches, seq, rows, aseq=None, args=None):
    with open(os.path.join(folder, f"launch_{pid}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# whirl launch log v1\n# cmd: {cmd}\n# launches {launches} seqhash {seq} kinds {len(rows)}\n")
        if aseq is not None:
            f.write(f"# argseqhash {aseq} argkinds {len(args)}\n")
        for k, c in rows.items():
            f.write(f"{k} {c}\n")
        for k, c in (args or {}).items():
            f.write(f"@ {k} {c}\n")


def selftest():
    """Block / pass cases of the server mode (arch024-S0a C-4) on synthetic logs."""
    import contextlib
    import io
    import tempfile
    srv = r'"C:\x\whirl-server.exe" m.gguf --port 8091 --parallel 4'
    cli = r'"C:\x\whirl.exe" m.gguf -p hi'
    rows = {"k_a 1x1x1 64x1x1 0": 10, "k_tick 1x1x1 32x1x1 0": 3}
    args = {"k_a 1x1x1 64x1x1 0 00000000000000aa": 10, "k_tick 1x1x1 32x1x1 0 00000000000000bb": 3}
    cases = []
    with tempfile.TemporaryDirectory() as td:
        def mk(name, srv_seq="1111", srv_rows=rows, srv_args=args, srv_n=13):
            d = os.path.join(td, name)
            os.makedirs(d)
            _log(d, 1, cli, 5, "c0c0", {"k_a 1x1x1 64x1x1 0": 5}, "c1c1", {"k_a 1x1x1 64x1x1 0 00000000000000aa": 5})
            _log(d, 2, srv, srv_n, srv_seq, srv_rows, "2222" if srv_seq == "1111" else "3333", srv_args)
            return d
        a = mk("a")
        same = mk("same")
        seq = mk("seq", srv_seq="9999")
        tick = mk("tick", srv_rows={"k_a 1x1x1 64x1x1 0": 10, "k_tick 1x1x1 32x1x1 0": 5},
                  srv_args={"k_a 1x1x1 64x1x1 0 00000000000000aa": 10, "k_tick 1x1x1 32x1x1 0 00000000000000bb": 5}, srv_seq="7777",
                  srv_n=15)
        arg = mk("arg", srv_args={"k_a 1x1x1 64x1x1 0 00000000000000ab": 10, "k_tick 1x1x1 32x1x1 0 00000000000000bb": 3})
        q = lambda *x: x
        cases = [
            ("default excludes the server", q(a, seq), {}, 0),
            ("--all: server counts only, a sequence change passes", q(a, seq), {"include_all": True}, 0),
            ("--server: same logs pass", q(a, same), {"server_seq": True}, 0),
            ("--server: a server sequence change blocks", q(a, seq), {"server_seq": True}, 1),
            ("--server: a time-dependent kernel's count blocks", q(a, tick), {"server_seq": True}, 1),
            ("--server --ignore k_tick: the same passes", q(a, tick), {"server_seq": True, "ignore": [r"^k_tick "]}, 0),
            ("--server --ignore k_tick: an argument change of another kernel still blocks", q(a, arg),
             {"server_seq": True, "ignore": [r"^k_tick "]}, 1),
        ]
        bad = 0
        for name, (x, y), kw, want in cases:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = diff(load(x, kw.get("include_all", False), kw.get("server_seq", False), kw.get("ignore", ())),
                           load(y, kw.get("include_all", False), kw.get("server_seq", False), kw.get("ignore", ())))
            ok = got == want
            bad += 0 if ok else 1
            print(f"{'ok  ' if ok else 'FAIL'} {name}: rc {got} (want {want})")
    print(f"launch_table selftest: {len(cases) - bad}/{len(cases)} as expected")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("folder")
    t.add_argument("-o")
    t.add_argument("--all", action="store_true")
    t.add_argument("--server", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--all", action="store_true")
    d.add_argument("--server", action="store_true")
    d.add_argument("--ignore", action="append", default=[])
    d.add_argument("--ignore-file")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "table":
        runs = table_of(a.folder, a.all, a.server)
        if a.o:
            with open(a.o, "w", encoding="utf-8", newline="\n") as f:
                write_table(runs, f)
        else:
            write_table(runs, sys.stdout)
        print(f"launch table: {len(runs)} run(s) from {a.folder}", file=sys.stderr)
        return 0 if runs else 1
    ign = list(a.ignore) + (read_ignore_file(a.ignore_file) if a.ignore_file else [])
    return diff(load(a.a, a.all, a.server, ign), load(a.b, a.all, a.server, ign))


if __name__ == "__main__":
    sys.exit(main())
