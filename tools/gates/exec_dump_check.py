#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""exec_dump_check.py (arch025-S0a): every scalar data member of qwen35::Model (include/whirl/model.h; since arch025-M1
also of its base qwen35::Weights, src/model/qwen35/weights.h; since arch025-M2 of its bases qwen35::KvPool and
qwen35::SeqTable, src/model/qwen35/kv.h; since arch025-M3 of its private bases qwen35::Scratch and qwen35::TuneTable,
src/model/qwen35/scratch.h / tune_table.h, read as m.scratch().<name> / m.tuneTable().<name>) is either
printed by WHIRL_DUMP_EXEC (src/model/exec_dump.cpp reads it as m.<name>) or named here as per-call state. A member
added to Model without either fails, so the S0b-2 dump cannot silently miss a new numerics flag.

  python -I tools/gates/exec_dump_check.py [--root R]
  python -I tools/gates/exec_dump_check.py --selftest
Last line: "exec dump check: ok (...)" (exit 0) or "exec dump check: FAIL (...)" (exit 1); exit 2 = usage.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# per-call state a forward sets and resets (the same after every load); keep in sync with exec_dump.cpp's comment
PER_CALL = {"row_n", "fwd_pos0", "xq_n", "x16_count", "x16_fp8", "cur_seq", "kv_off", "ctl_off", "cap_rows", "cap_n",
            "n_inj", "attn_win", "rdelta"}
SCALAR = (r"(?:bool|float|double|int|unsigned|std::u?int(?:8|16|32|64)_t|u32|std::size_t|KvFormat|std::optional<KvFormat>|hip::Arch|"
          r"GemvR|GemvW|whirl::relax::Params|whirl::numerics::(?:Request|Plan|KvChoice)|std::array<std::uint64_t,\s*n_types>)")
DECL = re.compile(r"^    (?:static\s+constexpr\s+)?" + SCALAR + r"\s+(.+?);")
# arch025-M2: KvPool's read-only format, `const KvFormat& kv_fmt = fmt_;` (dumped as kv_q8 / kv_rot / kv_kf16 / kv_q4)
REF = re.compile(r"^    const KvFormat&\s+([a-z_][a-z0-9_]*)\s*=")
# arch025-M2: private storage behind a dumped public name (not dumped itself, by design)
BY_DESIGN = {"fmt_": "KvPool's storage of kv_fmt (dumped through m.kv_fmt)",
             "fixed_": "KvPool::fixFormat's once-guard (true after every Model::load)"}


CLASS_MODEL = re.compile(r"^class Model\b[^{;]*\{", re.M)
STRUCT_WEIGHTS = re.compile(r"^struct Weights \{", re.M)
CLASS_KVPOOL = re.compile(r"^class KvPool \{", re.M)
STRUCT_SEQTABLE = re.compile(r"^struct SeqTable \{", re.M)
STRUCT_SCRATCH = re.compile(r"^struct Scratch \{", re.M)  # arch025-M3
STRUCT_TUNE = re.compile(r"^struct TuneTable \{", re.M)  # arch025-M3


def body_of(text: str, head: re.Pattern) -> str:
    m = head.search(text)
    if not m:
        raise ValueError(f"no {head.pattern!r}")
    s = text[m.start():]
    return s[:s.index("\n};")]


def members(model_h: str, weights_h: str = "") -> list[str]:
    body = body_of(model_h, CLASS_MODEL)
    if weights_h:  # arch025-M1: class Model : public Weights (src/model/qwen35/weights.h)
        body += "\n" + body_of(weights_h, STRUCT_WEIGHTS)
        # arch025-M2: ... public KvPool, public SeqTable (src/model/qwen35/kv.h, appended by weights_text)
        # arch025-M3: ... private Scratch, private TuneTable (scratch.h / tune_table.h)
        for head in (CLASS_KVPOOL, STRUCT_SEQTABLE, STRUCT_SCRATCH, STRUCT_TUNE):
            if head.search(weights_h):
                body += "\n" + body_of(weights_h, head)
    out = []
    for line in body.split("\n"):
        line = line.split("//")[0]
        r = REF.match(line)
        if r:
            out.append(r.group(1))
            continue
        m = DECL.match(line)
        if not m or "(" in m.group(1) or "static constexpr" in line:
            continue
        for part in m.group(1).split(","):
            name = part.split("=")[0].split("{")[0].strip()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                out.append(name)
    return out


def check(model_h: str, dump_cpp: str, out=print, weights_h: str = "") -> int:
    ms = members(model_h, weights_h)
    if len(ms) < 50:
        out(f"exec dump check: FAIL (usage: only {len(ms)} members parsed from model.h)")
        return 2
    used = set(re.findall(r"\bm\.(?:scratch\(\)\.|tuneTable\(\)\.)?([a-z_][a-z0-9_]*)", dump_cpp)) | set(re.findall(r"WD_(?:[BU]|SU|TB)\(([a-z_][a-z0-9_]*)\)", dump_cpp))
    missing = [n for n in ms if n not in used and n not in PER_CALL and n not in BY_DESIGN]
    for n in missing:
        out(f"  NOT DUMPED Model::{n}: print it in src/model/exec_dump.cpp or list it in PER_CALL here (per-call state)")
    stale = sorted(n for n in PER_CALL if n not in ms)
    for n in stale:
        out(f"  STALE PER_CALL {n}: not a Model member any more")
    stale_bd = sorted(n for n in BY_DESIGN if n not in ms) if weights_h and CLASS_KVPOOL.search(weights_h) else []
    for n in stale_bd:
        out(f"  STALE BY_DESIGN {n}: not a KvPool member any more")
    stale += stale_bd
    if missing or stale:
        out(f"exec dump check: FAIL ({len(missing)} not dumped, {len(stale)} stale, {len(ms)} scalar members)")
        return 1
    n_pc = len([n for n in ms if n in PER_CALL])
    n_bd = len([n for n in ms if n in BY_DESIGN])
    out(f"exec dump check: ok ({len(ms)} scalar members: {len(ms) - n_pc - n_bd} dumped, {n_pc} per-call"
        + (f", {n_bd} by design)" if n_bd else ")"))
    return 0


def selftest(root: str) -> int:
    h = open(os.path.join(root, "include", "whirl", "model.h"), encoding="utf-8").read()
    d = open(os.path.join(root, "src", "model", "exec_dump.cpp"), encoding="utf-8").read()
    w = weights_text(root)
    fails = 0
    # the injection point: the first scalar member line of class Model (arch025-N3-r1: the fixed anchor
    # "bool use_graph = true;" left Model in arch025-N2b, so both injection cases checked the unchanged tree)
    body_at = CLASS_MODEL.search(h).start()
    anchor = next((ln + "\n" for ln in h[body_at:].split("\n") if DECL.match(ln.split("//")[0])), None)
    if anchor is None:
        print("exec dump check selftest: FAIL (no scalar member line in class Model to inject after)")
        return 1
    at = h.index(anchor, body_at) + len(anchor)
    drop = "    WD_U(fd_splits);\n"
    cases = [
        ("the tree passes", h, d, 0),
        ("a new bool member fails", h[:at] + "    bool s0a_new_flag = false;\n" + h[at:], d, 1),
        ("a new multi-name member fails", h[:at] + "    std::uint32_t s0a_a = 0, s0a_b = 0;\n" + h[at:], d, 1),
        ("a flag dropped from the dump fails", h, d.replace(drop, "", 1), 1),
    ]
    if w:  # arch025-M1: the members of the base Weights count too
        wat = STRUCT_WEIGHTS.search(w).end() + 1
        cases.append(("a new Weights member fails", h, d, 1, w[:wat] + "    std::uint64_t m1_new_count = 0;\n" + w[wat:]))
        cases.append(("type_bytes dropped from the dump fails", h, d.replace("m.type_bytes", "m.tb_gone"), 1, w))
    if w and CLASS_KVPOOL.search(w):  # arch025-M2: the members of the bases KvPool / SeqTable count too
        kat = w.index("    std::uint32_t pool_pages = 0;\n")
        cases.append(("a new KvPool member fails", h, d, 1, w[:kat] + "    std::uint32_t m2_new_pages = 0;\n" + w[kat:]))
        cases.append(("pool_pages dropped from the dump fails", h, d.replace("WD_U(pool_pages);", ""), 1, w))
        sat = STRUCT_SEQTABLE.search(w).end() + 1
        cases.append(("a new SeqTable member fails", h, d, 1, w[:sat] + "    bool m2_seq_flag = false;\n" + w[sat:]))
        cases.append(("kv_fmt dropped from the dump fails", h, d.replace("m.kv_fmt", "m.kvf_gone"), 1, w))
    if w and STRUCT_SCRATCH.search(w) and STRUCT_TUNE.search(w):  # arch025-M3: the private bases count too
        cat = STRUCT_SCRATCH.search(w).end() + 1
        cases.append(("a new Scratch scalar fails", h, d, 1, w[:cat] + "    std::uint32_t m3_new_rows = 0;\n" + w[cat:]))
        cases.append(("dq_rows dropped from the dump fails", h, d.replace("WD_SU(dq_rows);", ""), 1, w))
        tat = STRUCT_TUNE.search(w).end() + 1
        cases.append(("a new TuneTable member fails", h, d, 1, w[:tat] + "    bool m3_tune_flag = false;\n" + w[tat:]))
        cases.append(("tune_cold dropped from the dump fails", h, d.replace("WD_TB(tune_cold);", ""), 1, w))
        cases.append(("tune_mask dropped from the dump fails", h, d.replace("m.tuneTable().tune_mask", "m.tuneTable().tm_gone"), 1, w))
    if drop not in d:
        print(f"exec dump check selftest: FAIL (exec_dump.cpp has no {drop.strip()!r} line to drop)")
        return 1
    for name, hh, dd, want, *ww in cases:
        lines: list[str] = []
        rc = check(hh, dd, out=lines.append, weights_h=ww[0] if ww else w)
        ok = rc == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: rc {rc} (want {want}) | {lines[-1]}")
    if fails:
        print(f"exec dump check selftest: FAIL ({fails} of {len(cases)})")
        return 1
    print(f"exec dump check selftest: ok ({len(cases)} cases)")
    return 0


def weights_text(root: str) -> str:
    # the bases' headers: weights.h (arch025-M1), kv.h (arch025-M2), scratch.h / tune_table.h (arch025-M3)
    out = ""
    for f in ("weights.h", "kv.h", "scratch.h", "tune_table.h"):
        p = os.path.join(root, "src", "model", "qwen35", f)
        if os.path.exists(p):
            out += open(p, encoding="utf-8").read() + "\n"
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.root)
    try:
        h = open(os.path.join(a.root, "include", "whirl", "model.h"), encoding="utf-8").read()
        d = open(os.path.join(a.root, "src", "model", "exec_dump.cpp"), encoding="utf-8").read()
    except OSError as e:
        print(f"exec dump check: FAIL (usage: {e})")
        return 2
    return check(h, d, weights_h=weights_text(a.root))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
