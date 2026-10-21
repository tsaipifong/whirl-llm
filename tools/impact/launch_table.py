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


def table_of(folder, include_all=False):
    runs = {}
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith("launch_") and fn.endswith(".txt")):
            continue
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
        runs[key] = {"n": n, "seq": "" if server else seq, "rows": rows, "aseq": "" if server else aseq, "args": args}
    return runs


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


def load(p, include_all):
    return table_of(p, include_all) if os.path.isdir(p) else read_table(p)


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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("folder")
    t.add_argument("-o")
    t.add_argument("--all", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "table":
        runs = table_of(a.folder, a.all)
        if a.o:
            with open(a.o, "w", encoding="utf-8", newline="\n") as f:
                write_table(runs, f)
        else:
            write_table(runs, sys.stdout)
        print(f"launch table: {len(runs)} run(s) from {a.folder}", file=sys.stderr)
        return 0 if runs else 1
    return diff(load(a.a, a.all), load(a.b, a.all))


if __name__ == "__main__":
    sys.exit(main())
