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


def ll(folder, pid, cmd, n, seq, rows, aseq=None, args=None, mtime=None):
    os.makedirs(folder, exist_ok=True)
    launch_table._log(folder, pid, cmd, n, seq, rows, aseq, args)
    p = os.path.join(folder, f"launch_{pid}.txt")
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

# ---------------- stored evidence on the owner's machine (skipped elsewhere): the two manual proofs this tool replaces
RW = r"C:\ClaudeWorkspace\whirl-work"
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
