#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""exec_dump_check.py (arch025-S0a): every scalar data member of qwen35::Model (include/whirl/model.h) is either
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
            "n_inj", "attn_win", "rdelta",
            # feat026-draftq: set by setupSeqs (after the load + tune dump; false after every load) = mtp.has_value(),
            # which the dump prints as "mtp"
            "draft_samp_ok"}
SCALAR = (r"(?:bool|float|double|int|unsigned|std::u?int(?:8|16|32|64)_t|u32|std::size_t|KvFormat|std::optional<KvFormat>|hip::Arch|"
          r"GemvR|GemvW|whirl::relax::Params|whirl::numerics::(?:Request|Plan|KvChoice)|std::array<std::uint64_t,\s*n_types>)")
DECL = re.compile(r"^    (?:static\s+constexpr\s+)?" + SCALAR + r"\s+(.+?);")


def members(model_h: str) -> list[str]:
    s = model_h[model_h.index("class Model {"):]
    body = s[:s.index("\n};")]
    out = []
    for line in body.split("\n"):
        line = line.split("//")[0]
        m = DECL.match(line)
        if not m or "(" in m.group(1) or "static constexpr" in line:
            continue
        for part in m.group(1).split(","):
            name = part.split("=")[0].split("{")[0].strip()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                out.append(name)
    return out


def check(model_h: str, dump_cpp: str, out=print) -> int:
    ms = members(model_h)
    if len(ms) < 50:
        out(f"exec dump check: FAIL (usage: only {len(ms)} members parsed from model.h)")
        return 2
    used = set(re.findall(r"\bm\.([a-z_][a-z0-9_]*)", dump_cpp)) | set(re.findall(r"WD_[BU]\(([a-z_][a-z0-9_]*)\)", dump_cpp))
    missing = [n for n in ms if n not in used and n not in PER_CALL]
    for n in missing:
        out(f"  NOT DUMPED Model::{n}: print it in src/model/exec_dump.cpp or list it in PER_CALL here (per-call state)")
    stale = sorted(n for n in PER_CALL if n not in ms)
    for n in stale:
        out(f"  STALE PER_CALL {n}: not a Model member any more")
    if missing or stale:
        out(f"exec dump check: FAIL ({len(missing)} not dumped, {len(stale)} stale, {len(ms)} scalar members)")
        return 1
    out(f"exec dump check: ok ({len(ms)} scalar members: {len(ms) - len([n for n in ms if n in PER_CALL])} dumped, "
        f"{len([n for n in ms if n in PER_CALL])} per-call)")
    return 0


def selftest(root: str) -> int:
    h = open(os.path.join(root, "include", "whirl", "model.h"), encoding="utf-8").read()
    d = open(os.path.join(root, "src", "model", "exec_dump.cpp"), encoding="utf-8").read()
    fails = 0
    # the injection point: the first scalar member line of class Model (arch025-N3-r1: the fixed anchor
    # "bool use_graph = true;" left Model in arch025-N2b, so both injection cases checked the unchanged tree)
    body_at = h.index("class Model {")
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
    if drop not in d:
        print(f"exec dump check selftest: FAIL (exec_dump.cpp has no {drop.strip()!r} line to drop)")
        return 1
    for name, hh, dd, want in cases:
        lines: list[str] = []
        rc = check(hh, dd, out=lines.append)
        ok = rc == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: rc {rc} (want {want}) | {lines[-1]}")
    if fails:
        print(f"exec dump check selftest: FAIL ({fails} of {len(cases)})")
        return 1
    print(f"exec dump check selftest: ok ({len(cases)} cases)")
    return 0


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
    return check(h, d)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
