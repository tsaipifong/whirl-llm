# SPDX-License-Identifier: Apache-2.0
"""Block and pass tests of work_equal.py (measure-tools P5, DESIGN §B3) on synthetic evidence: PE files with embedded AMDGPU
ELF code objects, launch-log folders, golden summaries and a work_traces database. CPU, < 5 s.

    python -I tools/impact/work_equal_test.py        exit 0 = "work_equal test: N/N ok"
"""
import contextlib
import io
import json
import os
import sqlite3
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_table  # noqa: E402
import work_equal as we  # noqa: E402

ok = bad = 0


def expect(name, cond, info=""):
    global ok, bad
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {info}"))
    ok, bad = ok + bool(cond), bad + (not cond)


# ---------------------------------------------------------------- synthetic files
def mk_elf(text=b"\x01\x02\x03\x04" * 8, mach=0x4E, note=b"amdhsa note"):
    shstr = b"\0.shstrtab\0.text\0.note\0"
    body = bytearray(64)
    off_shstr = len(body); body += shstr
    off_text = len(body); body += text
    off_note = len(body); body += note
    while len(body) % 8:
        body += b"\0"
    shoff = len(body)
    secs = [(0, 0, 0, 0), (1, 3, off_shstr, len(shstr)), (11, 1, off_text, len(text)), (17, 7, off_note, len(note))]
    for nm, typ, off, sz in secs:
        body += struct.pack("<IIQQQQIIQQ", nm, typ, 0, 0, off, sz, 0, 0, 1, 0)
    body[0:4] = b"\x7fELF"
    body[4], body[5], body[6], body[7], body[8] = 2, 1, 1, 64, 4
    struct.pack_into("<HHI", body, 16, 3, 224, 1)
    struct.pack_into("<Q", body, 0x28, shoff)
    struct.pack_into("<I", body, 0x30, mach)
    struct.pack_into("<HHHHHH", body, 0x34, 64, 0, 0, 64, len(secs), 1)
    return bytes(body)


def mk_pe(path, text=b"\x90" * 64, strings=b"whirl version 0.2.4\0", elfs=(), rsrc=b"VS_VERSION 0.2.4", ts=0x11111111,
          guid=b"G" * 16, age=1):
    FA, SA = 0x200, 0x1000
    al = lambda n, a: (n + a - 1) // a * a
    # .rdata = strings + code objects + debug directory (1 CodeView entry) + RSDS record
    rd = bytearray(strings)
    for e in elfs:
        while len(rd) % 16:
            rd += b"\0"
        rd += e
    while len(rd) % 4:
        rd += b"\0"
    dbg_off = len(rd); rd += b"\0" * 28
    rsds_off = len(rd); rd += b"RSDS" + guid + struct.pack("<I", age) + b"whirl.pdb\0"
    secs = [(b".text", bytes(text)), (b".rdata", rd), (b".rsrc", bytes(rsrc))]
    hdr = 0x400
    raw, va, lay = hdr, SA, []
    for nm, data in secs:
        lay.append((nm, data, va, raw))
        raw += al(len(data), FA); va += al(len(data), SA)
    rdata_va, rdata_raw = lay[1][2], lay[1][3]
    struct.pack_into("<IIHHIIII", rd, dbg_off, 0, ts, 0, 0, 2, len(rd) - rsds_off, rdata_va + rsds_off, rdata_raw + rsds_off)
    lay[1] = (b".rdata", bytes(rd), rdata_va, rdata_raw)
    out = bytearray(raw)
    out[0:2] = b"MZ"
    pe = 0x80
    struct.pack_into("<I", out, 0x3C, pe)
    out[pe:pe + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", out, pe + 4, 0x8664, len(secs), ts, 0, 0, 240, 0x22)
    opt = pe + 24
    struct.pack_into("<H", out, opt, 0x20B)
    struct.pack_into("<I", out, opt + 64, ts ^ 0x5A5A)  # CheckSum: differs per build like the timestamp
    struct.pack_into("<I", out, opt + 108, 16)
    struct.pack_into("<II", out, opt + 112 + 6 * 8, rdata_va + dbg_off, 28)
    so = opt + 240
    for i, (nm, data, v, r) in enumerate(lay):
        struct.pack_into("<8sIIIIIIHHI", out, so + 40 * i, nm, len(data), v, al(len(data), FA), r, 0, 0, 0, 0, 0x40000040)
        out[r:r + len(data)] = data
    with open(path, "wb") as f:
        f.write(out)
    return path


def ll(folder, pid, cmd, n, seq, rows, aseq=None, args=None, mtime=None, name=None):
    os.makedirs(folder, exist_ok=True)
    launch_table._log(folder, pid, cmd, n, seq, rows, aseq, args, name=name)
    p = os.path.join(folder, name or f"launch_{pid}.txt")
    if mtime is not None:
        os.utime(p, ns=(mtime, mtime))
    return p


def golden(path, exe, sha, cases, ref_all="ab" * 32, server=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"golden: exe {exe} sha256 {sha}\n")
        if server:
            f.write(f"golden: server exe {server[0]} sha256 {server[1]}\n")
        f.write(f"golden: reference set gfx1201 sha256 {'cd' * 32} (13 files; all files of gfx1201: {ref_all})\n")
        for name, st in cases.items():
            f.write(f"{st:5s} {name:24s} 43 lines, 20 s\n")
        f.write("PASS\n" if all(s == "OK" for s in cases.values()) else "FAIL\n")
    return path


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


ROWS = {"k_a 1x1x1 64x1x1 0": 5}
ARGS = {"k_a 1x1x1 64x1x1 0 00000000000000aa": 5}

with tempfile.TemporaryDirectory() as td:
    J = lambda *p: os.path.join(td, *p)
    os.makedirs(J("a")); os.makedirs(J("b")); os.makedirs(J("c"))
    E1, E2 = mk_elf(), mk_elf(mach=0x4A, text=b"\x05" * 40)
    A = mk_pe(J("a", "whirl.exe"), elfs=(E1, E2), strings=b"whirl version 0.2.3\0", rsrc=b"VS_VERSION 0.2.3", ts=1, guid=b"A" * 16)
    B = mk_pe(J("b", "whirl.exe"), elfs=(E1, E2), ts=2, guid=b"B" * 16)
    sha_a, sha_b = we.file_sha(A), we.file_sha(B)

    # ---------------- code objects + host
    sa = we.scan_exe(A)
    expect("scan: finds both AMDGPU objects with their targets", [o["target"] for o in sa["objects"]] == ["gfx1201", "gfx1151"], sa["objects"])
    expect("scan: per-section sha of each object (.text, .note, .shstrtab)",
           set(sa["objects"][0]["sections"]) == {"", ".shstrtab", ".text", ".note"}, sa["objects"][0]["sections"].keys())
    expect("scan: PE sections", [s["name"] for s in sa["pe"]["sections"]] == ["(headers)", ".text", ".rdata", ".rsrc"], sa["pe"])
    expect("scan: build-stamp fields located (COFF time, checksum, debug dir time, CodeView GUID/age)",
           {w for _, _, w in sa["pe"]["stamps"]} >= {"coff_timestamp", "checksum", "debug_timestamp", "codeview_guid_age"}, sa["pe"]["stamps"])
    r = we.compare_exe(A, B)
    expect("exe pair: both sha256 named", r["a"]["sha256"] == sha_a and r["b"]["sha256"] == sha_b, r["a"])
    expect("version-only rebuild: code objects identical", r["code_objects"]["result"] == "identical", r["code_objects"])
    expect("version-only rebuild: every object compared", len(r["code_objects"]["objects"]) == 2 and all(o["same"] for o in r["code_objects"]["objects"]))
    expect("version-only rebuild: host .text identical, differs elsewhere (version strings)",
           r["host"]["result"] == "differs" and r["host"]["text_same"] is True, r["host"])
    expect("version-only rebuild: differing strings name the version", any("0.2.3" in s for s in r["host"]["strings_only_a"])
           and any("0.2.4" in s for s in r["host"]["strings_only_b"]), (r["host"]["strings_only_a"], r["host"]["strings_only_b"]))
    expect("version-only rebuild: stamp bytes are not counted as unexplained",
           all(s["stamp_bytes"] >= 0 for s in r["host"]["sections"]) and r["host"]["stamp_bytes"] > 0, r["host"])
    C = mk_pe(J("c", "whirl.exe"), elfs=(E1, E2), ts=3, guid=b"C" * 16)
    rc_ = we.compare_exe(B, C)
    expect("stamp-only rebuild: host result stamp_only, 0 unexplained bytes",
           rc_["host"]["result"] == "stamp_only" and rc_["host"]["unexplained_bytes"] == 0, rc_["host"])
    same = we.compare_exe(B, B)
    expect("same file: host identical", same["host"]["result"] == "identical" and same["code_objects"]["result"] == "identical")
    D = mk_pe(J("c", "whirl_k.exe"), elfs=(mk_elf(text=b"\x01\x02\x03\x05" + b"\x01\x02\x03\x04" * 7), E2), ts=2, guid=b"B" * 16)
    rd = we.compare_exe(B, D)
    expect("kernel byte change: code objects different", rd["code_objects"]["result"] == "different", rd["code_objects"])
    expect("kernel byte change: names object 0 section .text",
           rd["code_objects"]["objects"][0]["differing_sections"] == [".text"] and rd["code_objects"]["objects"][1]["same"], rd["code_objects"]["objects"])
    expect("kernel byte change: ELF bytes are not counted as host bytes", rd["host"]["result"] in ("identical", "stamp_only"), rd["host"])
    F = mk_pe(J("c", "whirl_n.exe"), elfs=(E1,), ts=2, guid=b"B" * 16)
    rf = we.compare_exe(B, F)
    expect("object count change: different, count named", rf["code_objects"]["result"] == "different"
           and (rf["code_objects"]["n_a"], rf["code_objects"]["n_b"]) == (2, 1), rf["code_objects"])
    G = mk_pe(J("c", "whirl_h.exe"), text=b"\x90" * 63 + b"\xc3", elfs=(E1, E2), ts=2, guid=b"B" * 16)
    rg = we.compare_exe(B, G)
    expect("host .text change: code identical, host differs with .text named",
           rg["code_objects"]["result"] == "identical" and rg["host"]["text_same"] is False
           and any(s["name"] == ".text" and s["differing_bytes"] == 1 for s in rg["host"]["sections"]), rg["host"])
    Z = J("c", "zero.exe")
    with open(Z, "wb") as f:
        f.write(b"MZ not a pe")
    rz = we.compare_exe(Z, Z)
    expect("no code object in either exe: missing (fail closed), not identical", rz["code_objects"]["result"] == "missing", rz["code_objects"])

    # ---------------- launch tables
    cmd_a = f'"{A}" golden m.gguf --suite single --precise --device gfx1201'
    cmd_b = f'"{B}" golden m.gguf --suite single --precise --device gfx1201'
    srv_a = f'"{J("a", "whirl-server.exe")}" m.gguf --port 1 --parallel 4'
    t_after = os.stat(B).st_mtime_ns + 10**9
    for side, cmd in (("la", cmd_a), ("lb", cmd_b)):
        ll(J(side, "golden_swift"), 1, cmd, 5, "c0c0", ROWS, "c1c1", ARGS, mtime=t_after)
        ll(J(side, "golden_swift"), 2, cmd.replace("single", "pair"), 5, "d0d0", ROWS, "d1d1", ARGS, mtime=t_after + 1)
    ll(J("la", "golden_swift"), 3, srv_a, 9, "eeee", ROWS, "ffff", ARGS, mtime=t_after + 2)
    ll(J("lb", "golden_swift"), 3, srv_a.replace("\\a\\", "\\b\\"), 9, "9999", ROWS, "ffff", ARGS, mtime=t_after + 2)
    exes_a, exes_b = {"whirl.exe": sha_a}, {"whirl.exe": sha_b}
    L = we.compare_launch(J("la"), J("lb"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: same sequences (CLI) identical, server left out by default", L["result"] == "identical" and len(L["runs"]) == 2, L)
    expect("launch: step folders keyed step/command", all(x["key"].startswith("golden_swift/") for x in L["runs"]), [x["key"] for x in L["runs"]])
    expect("launch: bound to both exes by sha (exe unchanged since the log)", L["binding"]["a"]["state"] == "sha" and L["binding"]["b"]["state"] == "sha", L["binding"])
    expect("launch: argument digests compared", all(x["args_compared"] for x in L["runs"]), L["runs"])
    Ls = we.compare_launch(J("la"), J("lb"), server=True, a_exes=exes_a, b_exes=exes_b)
    expect("launch --server: server sequence change -> different", Ls["result"] == "different", Ls)
    ll(J("lc", "golden_swift"), 1, cmd_b, 5, "c0c0", ROWS, "c1c2", ARGS, mtime=t_after)
    ll(J("lc", "golden_swift"), 2, cmd_b.replace("single", "pair"), 5, "d0d0", ROWS, "d1d1", ARGS, mtime=t_after + 1)
    Lc = we.compare_launch(J("la"), J("lc"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: argument sequence change -> different, run named", Lc["result"] == "different"
           and [x["status"] for x in Lc["runs"]].count("DIFF") == 1, Lc["runs"])
    ll(J("ld", "golden_swift"), 1, cmd_b, 5, "c0c0", ROWS, "c1c1", ARGS, mtime=t_after)
    Ld = we.compare_launch(J("la"), J("ld"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: a run only in A -> missing (incomplete), not identical", Ld["result"] == "missing", Ld["runs"])
    Le = we.compare_launch(J("la"), J("nonexistent"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: no evidence for B -> missing", Le["result"] == "missing", Le)
    Lf = we.compare_launch(J("la"), J("lb"), a_exes=exes_a, b_exes={"whirl.exe": "0" * 64})
    expect("launch: logs made by another exe than B -> unbound -> missing", Lf["result"] == "missing" and Lf["binding"]["b"]["state"] == "mismatch", Lf["binding"])
    ll(J("lg", "golden_swift"), 1, cmd_b, 5, "c0c0", ROWS, "c1c1", ARGS, mtime=os.stat(B).st_mtime_ns - 10**9)
    ll(J("lg", "golden_swift"), 2, cmd_b.replace("single", "pair"), 5, "d0d0", ROWS, "d1d1", ARGS, mtime=os.stat(B).st_mtime_ns - 10**9)
    Lg = we.compare_launch(J("la"), J("lg"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: exe rebuilt after the log -> unverified -> missing", Lg["result"] == "missing" and Lg["binding"]["b"]["state"] == "unverified", Lg["binding"])
    Lg2 = we.compare_launch(J("la"), J("lg"), a_exes=exes_a, b_exes=exes_b, allow_unbound=True)
    expect("launch --allow-unbound-launch: identical, binding recorded as unverified", Lg2["result"] == "identical"
           and Lg2["binding"]["b"]["state"] == "unverified" and Lg2["binding"]["allow_unbound"] is True, Lg2)
    tab = J("la_table.txt")
    with open(tab, "w", encoding="utf-8", newline="\n") as f:
        launch_table.write_table(launch_table.table_of(J("la", "golden_swift")), f)
    Lt = we.compare_launch(tab, J("lb", "golden_swift"), a_exes=exes_a, b_exes=exes_b, allow_unbound=True)
    expect("launch: a saved table vs a folder compares (table side unbound)", Lt["result"] == "identical" and Lt["binding"]["a"]["state"] == "table", Lt)
    # bug:log-pid-names: the new names launch_<pid>_<start>[_<n>].txt, one PID reused by the second run -> both read
    ll(J("ln", "golden_swift"), 1, cmd_b, 5, "c0c0", ROWS, "c1c1", ARGS, mtime=t_after, name="launch_1_20261025T134651112.txt")
    ll(J("ln", "golden_swift"), 1, cmd_b.replace("single", "pair"), 5, "d0d0", ROWS, "d1d1", ARGS, mtime=t_after + 1,
       name="launch_1_20261025T134808592_2.txt")
    ll(J("ln", "golden_swift"), 1, "x", 0, "0", {}, mtime=t_after + 1, name="launch_1_20261025T134808592.raw.txt")
    Ln = we.compare_launch(J("la"), J("ln"), a_exes=exes_a, b_exes=exes_b)
    expect("launch: new file names (same PID twice) both read, identical, both bound", Ln["result"] == "identical"
           and len(Ln["runs"]) == 2 and Ln["binding"]["b"]["state"] == "sha", (Ln["runs"], Ln["binding"]))
    expect("launch: _log_files takes old + new names, not .raw.txt", sorted(os.path.basename(x) for x in we._log_files(J("ln")))
           == ["launch_1_20261025T134651112.txt", "launch_1_20261025T134808592_2.txt"], we._log_files(J("ln")))

    # ---------------- launch_table refactor: structured compare() and the printed diff agree
    ra, rb = launch_table.load(J("la", "golden_swift"), False), launch_table.load(J("lc", "golden_swift"), False)
    rc, out = quiet(launch_table.diff, ra, rb)
    st = launch_table.compare(ra, rb)
    expect("launch_table.compare: statuses match the printed diff", [e["status"] for e in st] == [ln.split()[0] for ln in out.splitlines() if ln[:4] in ("SAME", "DIFF", "ONLY")], (st, out))
    expect("launch_table.diff: rc unchanged (1 = different)", rc == 1)

    # ---------------- golden
    ga = golden(J("ga.txt"), A, sha_a, {"swift/precise_single": "OK", "swift/precise_pair": "OK"})
    gb = golden(J("gb.txt"), B, sha_b, {"swift/precise_single": "OK", "swift/precise_pair": "OK"})
    G1 = we.compare_golden([ga], [gb], a_exes=exes_a, b_exes=exes_b)
    expect("golden: both OK against the same references -> identical", G1["result"] == "identical" and len(G1["cases"]) == 2, G1)
    expect("golden: bound to both exes", G1["binding"]["a"] == "sha" and G1["binding"]["b"] == "sha", G1["binding"])
    gd = golden(J("gd.txt"), B, sha_b, {"swift/precise_single": "OK", "swift/precise_pair": "DIFF"})
    expect("golden: a DIFF case -> different", we.compare_golden([ga], [gd], a_exes=exes_a, b_exes=exes_b)["result"] == "different")
    gr = golden(J("gr.txt"), B, sha_b, {"swift/precise_single": "OK", "swift/precise_pair": "OK"}, ref_all="ef" * 32)
    expect("golden: other reference set -> missing (not comparable)", we.compare_golden([ga], [gr], a_exes=exes_a, b_exes=exes_b)["result"] == "missing")
    gw = golden(J("gw.txt"), A, sha_a, {"swift/precise_single": "OK", "swift/precise_pair": "OK"})
    G4 = we.compare_golden([ga], [gw], a_exes=exes_a, b_exes=exes_b)
    expect("golden: B's evidence names A's exe -> unbound -> missing", G4["result"] == "missing" and G4["binding"]["b"] == "mismatch", G4)
    g1 = golden(J("g1.txt"), B, sha_b, {"swift/precise_single": "OK"})
    expect("golden: a case only in A -> missing", we.compare_golden([ga], [g1], a_exes=exes_a, b_exes=exes_b)["result"] == "missing")
    g3 = golden(J("g3.txt"), B, sha_b, {"swift/precise_single": "OK", "swift/precise_pair": "OK", "swift/precise_ngram": "OK"})
    G7 = we.compare_golden([ga], [g3], a_exes=exes_a, b_exes=exes_b)
    expect("golden: an extra case only in B -> identical on the common cases, extra case listed", G7["result"] == "identical"
           and G7["only_b"] == ["gfx1201:swift/precise_ngram"], G7)
    expect("golden: no common case -> missing", we.compare_golden([g1], [golden(J("g8.txt"), B, sha_b, {"moe/fast_pair": "OK"})], a_exes=exes_a,
                                                                   b_exes=exes_b)["result"] == "missing")
    for side, txt in (("oa", "1 2 3\n"), ("ob", "1 2 3\n"), ("oc", "1 2 4\n")):
        os.makedirs(J(side))
        with open(J(side, "swift_precise_single.txt"), "w") as f:
            f.write(txt)
        with open(J(side, "swift_precise_server.diag.txt"), "w") as f:
            f.write(side)
    G5 = we.compare_golden([ga], [gb], a_out=J("oa"), b_out=J("ob"), a_exes=exes_a, b_exes=exes_b)
    expect("golden out dirs: same output files (diag files left out) -> identical", G5["result"] == "identical" and G5["outputs"]["compared"] == 1, G5["outputs"])
    G6 = we.compare_golden([ga], [gb], a_out=J("oa"), b_out=J("oc"), a_exes=exes_a, b_exes=exes_b)
    expect("golden out dirs: an output file differs -> different", G6["result"] == "different", G6["outputs"])

    # ---------------- db mode (work_traces rows written by lt_import.py)
    db = J("t.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE work_traces (id INTEGER PRIMARY KEY, run_id TEXT, measurement_id INTEGER, scope TEXT, component TEXT, cell TEXT, "
                "exe_sha256 TEXT, work_hash TEXT, seq_hash TEXT, arg_hash TEXT, launches INTEGER, launch_table TEXT, work_json TEXT, recorded_at TEXT)")
    rows = [("test", "golden_swift", "k1", sha_a, "w1", "s1", "a1", 5), ("test", "golden_swift", "k2", sha_a, "w2", "s2", "a2", 5),
            ("test", "golden_swift", "k1", sha_b, "w1", "s1", "a1", 5), ("test", "golden_swift", "k2", sha_b, "w2", "s2", "a2", 5),
            ("test", "golden_swift", "k1", "c" * 64, "w1", "s1", "a1", 5), ("test", "golden_swift", "k2", "c" * 64, "wX", "s2", "aX", 5)]
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, gpu TEXT)")
    con.executemany("INSERT INTO work_traces (scope, component, cell, exe_sha256, work_hash, seq_hash, arg_hash, launches, work_json) "
                    "VALUES (?,?,?,?,?,?,?,?,'{\"gpu\": \"gfx1201\"}')", rows)
    # the same exe on the other GPU (8060S): another work hash for k1; kept apart by "@gpu"
    con.execute("INSERT INTO work_traces (scope, component, cell, exe_sha256, work_hash, seq_hash, arg_hash, launches, work_json) "
                "VALUES ('test','golden_swift','k1',?,'w9','s9','a9',5,'{\"gpu\": \"gfx1151\"}')", (sha_b,))
    con.commit(); con.close()
    Ldb = we.compare_launch(f"db:{sha_a}", f"db:{sha_b}@gfx1201", db=db)
    expect("db: stored traces of A and B identical, bound by stored exe sha", Ldb["result"] == "identical" and Ldb["binding"]["a"]["state"] == "db" and len(Ldb["runs"]) == 2, Ldb)
    expect("db: unfiltered, one exe with two GPUs' traces disagreeing -> different (inconsistent evidence)",
           we.compare_launch(f"db:{sha_a}", f"db:{sha_b}", db=db)["result"] == "different")
    Lg1 = we.compare_launch(f"db:{sha_a}@gfx1201", f"db:{sha_b}@gfx1151", db=db)
    expect("db @gpu: B on gfx1151 has only k1 (other hash) -> not identical", Lg1["result"] in ("different", "missing"), Lg1["runs"])
    expect("db @gpu: aliases match (R9700 stored, @gfx1201 asked)", we.norm_gpu("R9700") == we.norm_gpu("gfx1201") == we.norm_gpu("pci:71:0")
           and we.norm_gpu("8060S") == we.norm_gpu("gfx1151") and we.norm_gpu("gfx1151") != we.norm_gpu("gfx1201"))
    expect("db: argument digests counted as compared when stored", Ldb["args_compared"] == 2, Ldb)
    Ldb2 = we.compare_launch(f"db:{sha_a}", f"db:{'c' * 64}", db=db)
    expect("db: one stored work hash differs -> different", Ldb2["result"] == "different", Ldb2["runs"])
    Ldb3 = we.compare_launch(f"db:{sha_a}", f"db:{'d' * 64}", db=db)
    expect("db: no stored traces for B -> missing", Ldb3["result"] == "missing", Ldb3)

    # ---------------- whole report + verdict + CLI
    rep = we.report([A], [B], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
    expect("report: EQUIVALENT when code objects, launch, golden all identical", rep["verdict"] == "EQUIVALENT" and rep["exit"] == 0, rep["checks"])
    expect("report: names both exe sha per exe and the host summary", rep["exes"][0]["a"]["sha256"] == sha_a and rep["exes"][0]["b"]["sha256"] == sha_b
           and "host" in rep["exes"][0] and rep["checks"]["host_code"] == "differs", rep["checks"])
    expect("report: evidence files carry their sha256", all(e.get("sha256") for e in rep["evidence"]), rep["evidence"])
    rep2 = we.report([A], [B], golden_a=[ga], golden_b=[gb])
    expect("report: no launch evidence -> INCOMPLETE (exit 2), reason named", rep2["verdict"] == "INCOMPLETE" and rep2["exit"] == 2
           and any("launch" in x for x in rep2["reasons"]), rep2["reasons"])
    os.makedirs(J("d"))
    D2 = mk_pe(J("d", "whirl.exe"), elfs=(mk_elf(text=b"" + b"" * 7), E2), ts=2, guid=b"B" * 16)
    rep3 = we.report([B], [D2], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
    expect("report: code object change -> NOT_EQUIVALENT (exit 1) even with launch + golden identical", rep3["verdict"] == "NOT_EQUIVALENT" and rep3["exit"] == 1)
    rep4 = we.report([A], [J("c", "whirl.exe")], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
    expect("report: exe pairing by name; launch / golden evidence for another B exe -> INCOMPLETE", rep4["verdict"] == "INCOMPLETE", rep4["checks"])
    rep5 = we.report([A], [J("nowhere", "whirl.exe")], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
    expect("report: an exe file that does not exist -> INCOMPLETE with the path named, no crash", rep5["verdict"] == "INCOMPLETE"
           and any("not found" in x for x in rep5["reasons"]), rep5["reasons"])
    outj = J("rep.json")
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "work_equal.py"), "compare", "--a-exe", A, "--b-exe", B, "--a-launch", J("la"),
                        "--b-launch", J("lb"), "--a-golden", ga, "--b-golden", gb, "-o", outj], capture_output=True, text=True)
    js = json.load(open(outj, encoding="utf-8")) if os.path.isfile(outj) else {}
    expect("CLI compare: exit 0 + JSON verdict EQUIVALENT", p.returncode == 0 and js.get("verdict") == "EQUIVALENT", p.stdout + p.stderr)
    expect("CLI compare: JSON format tag for P7", js.get("format") == "whirl-work-equal/1", js.get("format"))
    p2 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "work_equal.py"), "compare", "--a-exe", A, "--b-exe", B], capture_output=True, text=True)
    expect("CLI compare: code objects only -> exit 2 (INCOMPLETE)", p2.returncode == 2, p2.stdout + p2.stderr)
    p3 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "work_equal.py"), "compare", "--a-exe", A], capture_output=True, text=True)
    expect("CLI compare: usage error -> exit 3", p3.returncode == 3, p3.stdout + p3.stderr)

# ---------------- __hip_cuid (arch025-weq): code objects of two worktrees differ only in the per-TU cuid symbol
CUID1, CUID2 = b"__hip_cuid_9b8e5fda6ccb5c45", b"__hip_cuid_952180f848eb64a9"
CUID_SHORT = b"__hip_cuid_9ea48a15715ed90"  # 15 hex digits: utohexstr drops a leading zero (real object #2, arch025 vs a25_int)


def _gnuh(n):
    h = 5381
    for c in n:
        h = (h * 33 + c) & 0xFFFFFFFF
    return h


def _sysvh(n):
    h = 0
    for c in n:
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def mk_cuid_elf(cuid=CUID1, text=b"\x11\x22\x33\x44" * 8, rodata=b"const table 0123", names=(b"kernel_a", b"kernel_b.kd", b"var_x"),
                cuid_value=0x40, addend=8, mutate=None):
    """an AMDGPU-like ELF64 code object with .text .rodata .dynsym .gnu.hash .hash .dynstr .rela.dyn .dynamic .symtab
    .strtab .shstrtab as lld lays it out: .dynsym sorted by .gnu.hash bucket (so the cuid moves the other symbols),
    string tables in symbol order, fixed section capacities (a shorter cuid leaves the layout unchanged, as in the real
    object). mutate(name, bytearray) may change one section's bytes afterwards."""
    syms = [(names[0], 0x12, 1, 0x100, 0x40), (names[1], 0x11, 1, 0x200, 0x40), (cuid, 0x11, 2, cuid_value, 1), (names[2], 0x11, 2, 0x10, 4)]
    nb = 4  # with 4 buckets the three cuids land at three different .dynsym indices
    dyn = sorted(syms, key=lambda s: (_gnuh(s[0]) % nb, s[0]))  # bucket order; symoffset 1
    dynstr = bytearray(b"\0"); stoff = {}
    for s in syms:  # lld adds names before sorting .dynsym by bucket: .dynstr keeps symbol order
        stoff[s[0]] = len(dynstr); dynstr += s[0] + b"\0"
    dynsym = bytearray(24)
    for s in dyn:
        dynsym += struct.pack("<IBBHQQ", stoff[s[0]], s[1], 0, s[2], s[3], s[4])
    n = len(dyn) + 1
    bloom, buckets, chains = 0, [0] * nb, []
    for i, s in enumerate(dyn, 1):
        h = _gnuh(s[0])
        bloom |= (1 << (h % 64)) | (1 << ((h >> 6) % 64))
        b = h % nb
        if not buckets[b]:
            buckets[b] = i
        last = i == n - 1 or _gnuh(dyn[i][0]) % nb != b
        chains.append((h & ~1) | (1 if last else 0))
    gnu = struct.pack("<IIII", nb, 1, 1, 6) + struct.pack("<Q", bloom) + struct.pack(f"<{nb}I", *buckets) + struct.pack(f"<{len(chains)}I", *chains)
    sb, sc = [0] * nb, [0] * n
    for i, s in enumerate(dyn, 1):
        b = _sysvh(s[0]) % nb
        sc[i] = sb[b]; sb[b] = i
    sysv = struct.pack("<II", nb, n) + struct.pack(f"<{nb}I", *sb) + struct.pack(f"<{n}I", *sc)
    vi = 1 + [s[0] for s in dyn].index(names[2])
    rela = struct.pack("<QQq", 0x1000, (vi << 32) | 1, addend)
    strtab = bytearray(b"\0"); toff = {}
    for s in syms:
        toff[s[0]] = len(strtab); strtab += s[0] + b"\0"
    symtab = bytearray(24)
    for s in syms:
        symtab += struct.pack("<IBBHQQ", toff[s[0]], s[1], 0, s[2], s[3], s[4])
    dynamic = struct.pack("<qQ", 10, len(dynstr)) + struct.pack("<qQ", 6, 0x800) + struct.pack("<qQ", 0, 0)
    shnames = [b"", b".text", b".rodata", b".dynsym", b".gnu.hash", b".hash", b".dynstr", b".rela.dyn", b".dynamic", b".symtab", b".strtab", b".shstrtab"]
    shstr = bytearray(b"\0"); shoffs = []
    for s in shnames:
        shoffs.append(len(shstr) - 1 if not s else len(shstr))
        if s:
            shstr += s + b"\0"
    shoffs[0] = 0
    content = {b".text": bytearray(text), b".rodata": bytearray(rodata), b".dynsym": dynsym, b".gnu.hash": bytearray(gnu), b".hash": bytearray(sysv),
               b".dynstr": dynstr, b".rela.dyn": bytearray(rela), b".dynamic": bytearray(dynamic), b".symtab": symtab, b".strtab": strtab, b".shstrtab": shstr}
    if mutate:
        for k, v in content.items():
            mutate(k.decode(), v)
    typ = {b".text": 1, b".rodata": 1, b".dynsym": 11, b".gnu.hash": 0x6FFFFFF6, b".hash": 5, b".dynstr": 3, b".rela.dyn": 4, b".dynamic": 6,
           b".symtab": 2, b".strtab": 3, b".shstrtab": 3}
    link = {b".dynsym": 6, b".gnu.hash": 3, b".hash": 3, b".rela.dyn": 3, b".dynamic": 6, b".symtab": 10}
    ent = {b".dynsym": 24, b".symtab": 24, b".rela.dyn": 24, b".dynamic": 16}
    body = bytearray(64 + 56)  # ELF header + one program header
    hdrs = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for i, s in enumerate(shnames[1:], 1):
        while len(body) % 64:
            body += b"\0"
        off = len(body); c = content[s]
        body += c + (b"\0" * (256 - len(c)) if s in (b".dynstr", b".strtab") else b"")  # fixed capacity: a shorter cuid keeps the layout
        hdrs.append((shoffs[i], typ[s], 2, 0, off, len(c), link.get(s, 0), 1 if s == b".symtab" else 0, 1, ent.get(s, 0)))
    while len(body) % 64:
        body += b"\0"
    shoff = len(body)
    for h in hdrs:
        body += struct.pack("<IIQQQQIIQQ", *h)
    body[0:4] = b"\x7fELF"
    body[4], body[5], body[6], body[7], body[8] = 2, 1, 1, 64, 4
    struct.pack_into("<HHI", body, 16, 3, 224, 1)
    struct.pack_into("<Q", body, 0x20, 64)
    struct.pack_into("<Q", body, 0x28, shoff)
    struct.pack_into("<I", body, 0x30, 0x4E)
    struct.pack_into("<HHHHHH", body, 0x34, 64, 56, 1, 64, len(hdrs), len(hdrs) - 1)
    struct.pack_into("<IIQQQQQQ", body, 64, 1, 5, 0, 0, 0, 0x800, 0x800, 0x1000)
    return bytes(body)


def _flip(sec, i=0):
    def m(name, b):
        if name == sec:
            b[i] ^= 1
    return m


def _rename(old, new):
    def m(name, b):
        if name in (".dynstr", ".strtab"):
            k = bytes(b).find(old)
            b[k:k + len(new)] = new
    return m


def _co(a, b, **kw):
    try:
        return we.compare_exe(a, b, **kw)["code_objects"]
    except TypeError as e:  # the pre-fix tool has no allow_cuid_length
        return {"result": f"TypeError {e}", "objects": []}


with tempfile.TemporaryDirectory() as td:
    J = lambda *p: os.path.join(td, *p)
    for d in ("a", "b", "m"):
        os.makedirs(J(d))
    E2 = mk_elf(mach=0x4A, text=b"\x05" * 40)
    ea, eb = mk_cuid_elf(CUID1), mk_cuid_elf(CUID2)
    expect("cuid fixture: same object size for both cuids (fixed layout)", len(ea) == len(eb) == len(mk_cuid_elf(CUID_SHORT)))
    A = mk_pe(J("a", "whirl.exe"), elfs=(ea, E2), ts=1, guid=b"A" * 16)
    B = mk_pe(J("b", "whirl.exe"), elfs=(eb, E2), ts=2, guid=b"B" * 16)
    sha_a, sha_b = we.file_sha(A), we.file_sha(B)
    r = _co(A, B)
    expect("cuid only: fixture really moves .dynsym / hash tables (not only the string bytes)",
           set(r["objects"][0].get("differing_sections") or []) >= {".dynstr", ".strtab", ".dynsym", ".gnu.hash", ".hash"}, r["objects"][:1])
    expect("cuid only: code objects identical-except-cuid", r["result"] == "identical-except-cuid", r)
    expect("cuid only: object 0 marked cuid_only, object 1 plain same", r["objects"][0].get("cuid_only") is True and r["objects"][1]["same"], r["objects"])
    # the launch / golden evidence of both sides identical and bound: the verdict depends on the code objects alone
    t_after = os.stat(B).st_mtime_ns + 10**9
    for side, exe in (("la", A), ("lb", B)):
        ll(J(side, "golden_swift"), 1, f'"{exe}" golden m.gguf --suite single --precise --device gfx1201', 5, "c0c0", ROWS, "c1c1", ARGS, mtime=t_after)
    ga = golden(J("ga.txt"), A, sha_a, {"swift/precise_single": "OK"})
    gb = golden(J("gb.txt"), B, sha_b, {"swift/precise_single": "OK"})
    rep = we.report([A], [B], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
    expect("cuid only + launch + golden identical: verdict EQUIVALENT (fails on the pre-fix tool: NOT_EQUIVALENT)",
           rep["verdict"] == "EQUIVALENT" and rep["exit"] == 0 and rep["checks"]["code_objects"] == "identical-except-cuid", (rep["verdict"], rep["checks"], rep["reasons"]))
    outj = J("cuid.json")
    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "work_equal.py"), "compare", "--a-exe", A, "--b-exe", B, "--a-launch", J("la"),
                        "--b-launch", J("lb"), "--a-golden", ga, "--b-golden", gb, "-o", outj], capture_output=True, text=True)
    js = json.load(open(outj, encoding="utf-8")) if os.path.isfile(outj) else {}
    expect("cuid only CLI: exit 0, JSON verdict EQUIVALENT, format unchanged (release_check P7 reads verdict / exit / format)",
           p.returncode == 0 and js.get("verdict") == "EQUIVALENT" and js.get("exit") == 0 and js.get("format") == "whirl-work-equal/1"
           and js.get("checks", {}).get("code_objects") == "identical-except-cuid", p.stdout + p.stderr)

    # every other difference still fails (each mutant: B side changed once more)
    def mutant(tag, **kw):  # named whirl.exe: report() pairs exes by base name
        os.makedirs(J("m", tag), exist_ok=True)
        return mk_pe(J("m", tag, "whirl.exe"), elfs=(mk_cuid_elf(**kw), E2), ts=2, guid=b"B" * 16)
    cases = [
        ("1 byte of .text", dict(cuid=CUID2, mutate=_flip(".text", 5))),
        ("1 byte of .rodata", dict(cuid=CUID2, mutate=_flip(".rodata", 3))),
        ("a renamed non-cuid symbol (same length, kernel_b.kd -> kernel_c.kd)", dict(cuid=CUID2, names=(b"kernel_a", b"kernel_c.kd", b"var_x"))),
        ("1 char of a non-cuid .dynstr name (S0b mutant)", dict(cuid=CUID2, mutate=lambda n, b: n == ".dynstr" and _rename(b"kernel_a", b"kernel_q")(n, b))),
        ("a cuid of a different length (15 hex digits) with --strict-cuid-length", dict(cuid=CUID_SHORT)),
        ("cuid-like name that is not lower-case hex", dict(cuid=b"__hip_cuid_952180F848EB64A9")),
        ("cuid symbol value changed", dict(cuid=CUID2, cuid_value=0x48)),
        ("relocation addend changed", dict(cuid=CUID2, addend=16)),
        ("a .gnu.hash bloom word cleared", dict(cuid=CUID2, mutate=lambda n, b: n == ".gnu.hash" and b.__setitem__(slice(16, 24), bytes(8)))),
        ("a .hash bucket changed", dict(cuid=CUID2, mutate=_flip(".hash", 8))),
        ("DT_STRSZ changed", dict(cuid=CUID2, mutate=_flip(".dynamic", 1))),
        ("a .symtab size field changed", dict(cuid=CUID2, mutate=_flip(".symtab", 24 + 16))),
    ]
    for tag, kw in cases:
        M = mutant(tag.split()[1] + str(len(tag)), **kw)
        # the length case is "different" only in strict mode (default accepts length changes since 2026-10-25)
        r = _co(A, M, allow_cuid_length=False) if "strict-cuid-length" in tag else _co(A, M)
        expect(f"cuid + {tag}: code objects different", r["result"] == "different", (r["result"], [o.get("cuid") for o in r["objects"]]))
        if tag.startswith("1 byte of .text"):
            rep = we.report([A], [M], launch=(J("la"), J("lb")), golden_a=[ga], golden_b=[gb])
            expect("cuid + 1 byte of .text: verdict NOT_EQUIVALENT", rep["verdict"] == "NOT_EQUIVALENT", rep["checks"])
        if tag.startswith("a cuid of a different"):
            expect("cuid length: reason names the length", "length" in str(r["objects"][0].get("cuid")), r["objects"][0])
            r2 = _co(A, M, allow_cuid_length=True)
            expect("cuid length with allow_cuid_length: identical-except-cuid (sh_size / DT_STRSZ / st_name shifts explained)",
                   r2["result"] == "identical-except-cuid", (r2["result"], [o.get("cuid") for o in r2["objects"]]))
            r3 = _co(A, mutant("lenrodata", cuid=CUID_SHORT, mutate=_flip(".rodata", 0)), allow_cuid_length=True)
            expect("cuid length allowed + 1 byte of .rodata: still different", r3["result"] == "different", r3["result"])
    r = _co(A, mutant("same", cuid=CUID1))
    expect("same cuid, same bytes: plain identical", r["result"] == "identical", r["result"])

# ---------------- stored evidence on the owner's machine (skipped elsewhere): the two manual proofs this tool replaces
RW = r"C:\ClaudeWorkspace\whirl-work"
ra_, rb_ = os.path.join(RW, r"wt_arch025\build\Release\whirl.exe"), os.path.join(RW, r"wt_a25_int\build\Release\whirl.exe")
if os.path.isfile(ra_) and os.path.isfile(rb_):  # read only; arch025:S0b-runners pair (two worktrees, same sources for the kernels)
    r = _co(ra_, rb_)
    cu = [o.get("cuid_only") for o in r["objects"]]
    if r.get("n_a") == 4 and cu == [True, True, False, True]:
        expect("real arch025 vs a25_int whirl.exe: objects 0/1/3 cuid-only, object 2 differs only by a 26- vs 27-byte cuid (default: different)",
               r["result"] == "different" and "length" in str(r["objects"][2].get("cuid")), r["objects"][2])
        r2 = _co(ra_, rb_, allow_cuid_length=True)
        expect("real arch025 vs a25_int whirl.exe with allow_cuid_length: identical-except-cuid", r2["result"] == "identical-except-cuid", r2["result"])
    else:
        print(f"skip real arch025 pair (objects not as recorded on 2026-10-25, rebuilt or pre-fix tool: {r.get('result')} {cu})")
else:
    print("skip real arch025 pair (not on this machine)")
s15, shipped = os.path.join(RW, r"scratch\s16\exe_9844fde\whirl.exe"), os.path.join(RW, r"release024\pkg\whirl-0.2.4-windows-x64\whirl.exe")
if os.path.isfile(s15) and os.path.isfile(shipped):
    r = we.compare_exe(s15, shipped)
    expect("stored 0.2.4: S15 exe vs shipped exe, 4 code objects identical (scratch\\s16\\cosect.txt)",
           r["code_objects"]["result"] == "identical" and r["code_objects"]["n_a"] == 4, r["code_objects"])
    expect("stored 0.2.4: host .text identical, only the version string + build stamps differ",
           r["host"]["text_same"] and r["host"]["strings_only_a"] == ["0.2.3"] and r["host"]["strings_only_b"] == ["0.2.4"], r["host"])
else:
    print("skip stored 0.2.4 evidence (not on this machine)")
la, lb = os.path.join(RW, r"arch6\lt\K0dfix2_r97"), os.path.join(RW, r"arch6\lt\G_r97")
if os.path.isdir(la) and os.path.isdir(lb):
    r = we.compare_launch(la, lb, allow_unbound=True)
    expect("stored 0.2.3: K0dfix2_r97 vs G_r97 launch tables 24/24 identical with arguments (arch6g\\lt_compare.txt)",
           r["result"] == "identical" and r["n_runs"] == 24 and r["args_compared"] == 24, (r["result"], r["n_runs"], r["reasons"]))
    r = we.compare_launch(os.path.join(RW, r"arch6\lt\F_r97"), os.path.join(RW, r"arch6\lt\F2_r97"), allow_unbound=True)
    expect("stored 0.2.3: F_r97 vs F2_r97 (pre-fix digest) -> different, 17 runs (ARCH6_R9700_RESULTS section 1a)",
           r["result"] == "different" and sum(x["status"] == "DIFF" for x in r["runs"]) == 17, r["result"])
else:
    print("skip stored 0.2.3 evidence (not on this machine)")

print(f"work_equal test: {ok}/{ok + bad} ok")
sys.exit(1 if bad else 0)
