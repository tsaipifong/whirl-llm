#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""arch025-gaptests G20: static happens-before check of the vision stream-encode schedule (src/vision/vision.cpp
Vision::encode, the !res branch). CPU only: it reads the source, it runs nothing on a GPU.

bug-visstream-race (12011b4) was a missing hip::streamWaitEvent(stream, ev_copy_[0]) before pos_ln, which reads layer 0's
ln1 from weight slot 0 while the copy stream may still be filling it. This check walks Vision::encode in host order,
as the stream (!res) path runs it, and models the two streams the way the HIP runtime orders them (PyTorch CUDA Stream
Sanitizer model: per-stream program order + event record / wait edges + buffer accesses; PRIOR_ART arch025-gaptests):

  * slot aliases are read from the code: `const DevPtr slot[2] = {W, W + lb}` and every `const DevPtr x = ...;` whose
    stream branch names a slot (slot[k], W + lb, or an alias already known), e.g. misc -> slot 1, lw0 -> slot 0,
    L -> slot s, Ln -> slot 1 - s, mbase -> slot 1, g / b -> slot 1 and slot 1 - s
  * hip::copyAnyAsync(<slot>, ..., cs) writes the slot on the copy stream; hip::eventRecord(ev_copy_[k], cs) covers
    the copies into slot k queued before it
  * hip::streamWaitEvent(stream, ev_copy_[k]) makes the compute stream see the copies the last record covered
  * a read is any hip::launch / launchGemm on `stream` whose arguments name a slot alias

Rules (each violation is printed with its line):
  R1 (RAW) a read of slot k on stream needs a wait on ev_copy_[k] after the record that covers the latest copy into
     slot k (prologue: exact; loop body: the wait must be in the body before the read, no copy into k in between)
  R2 (WAR) a copy into slot k on cs needs a preceding streamWaitEvent(cs, ev_free_[k] | ev_start_) whose event was
     recorded on stream after the last read of slot k (prologue: ev_start_ before any read)
     (not for the merger copy into W + lb: whether slot 1 is free then depends on the parity of n_layer)
  R3 the merger reads after the loop (slot 1 = W + lb) need a wait on ev_copy_[1] in the loop body that comes after
     the body's eventRecord(ev_copy_[1], cs) of the merger copy in host order of the last iteration (checked as: the
     body has the wait and a record of ev_copy_[1] after the merger copy)
Scope / limits: indices are symbols (0, 1, s, 1 - s); s and 1 - s are never assumed to equal 0 or 1, so a wait for
one does not count for another (the safe side). The loop body is checked as one iteration; branch conditions inside
it (l + 2 < n, last) are not evaluated. Timing is not modelled: that the device really reorders, and that the fix
removes the wrong embedding, is shown only by the GPU repro (vis-encode WHIRL_TEST_VIS_COPY_DELAY_MB, vis_ab).

    python -I tools/gates/vis_stream_order_check.py [--src src/vision/vision.cpp]
    python -I tools/gates/vis_stream_order_check.py --selftest
Last line "vis stream order check: ok (N reads, M copies)" / "...: FAIL (K violation(s))"; exit 0 / 1, 2 = parse error.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CALLS = ("hip::copyAnyAsync", "hip::eventRecord", "hip::streamWaitEvent", "hip::launch", "launchGemm")


class ParseError(Exception):
    pass


def encode_body(src: str) -> tuple[str, int]:
    m = re.search(r"\n[^\n]*\bVision::encode\s*\(", src)
    if not m:
        raise ParseError("Vision::encode not found")
    i = src.index("{", m.end())
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1], src.count("\n", 0, i) + 1
        j += 1
    raise ParseError("Vision::encode: unbalanced braces")


def strip_comments(s: str) -> str:
    # line comments only (the file uses no block comments inside encode); keep the newlines for line numbers
    return re.sub(r"//[^\n]*", "", s)


def close_paren(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ParseError("unbalanced parentheses")


def split_args(a: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for c in a:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


def norm_idx(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip())


def stream_branch(expr: str) -> str:
    # `res ? A : B` -> B (the stream path); anything else unchanged
    m = re.match(r"\s*res\s*\?(.*?):(.*)$", expr, re.S)
    return m.group(2) if m else expr


def events(body: str, line0: int):
    """(kind, line, payload) in source order: alias definitions, calls, loop begin / end"""
    out = []
    loop = re.search(r"for\s*\(\s*u32\s+l\s*=\s*0\s*;\s*l\s*<\s*hp\.n_layer\s*;\s*\+\+l\s*\)\s*\{", body)
    if not loop:
        raise ParseError("the layer loop `for (u32 l = 0; l < hp.n_layer; ++l) {` not found")
    lb_open = loop.end() - 1
    depth, k = 0, lb_open
    while True:
        if body[k] == "{":
            depth += 1
        elif body[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    loop_end = k
    out.append((loop.start(), "loop_begin", None))
    out.append((loop_end, "loop_end", None))
    for m in re.finditer(r"const\s+DevPtr\s+slot\s*\[\s*2\s*\]\s*=\s*\{([^}]*)\}\s*;", body):
        out.append((m.start(), "slots", split_args(m.group(1))))
    for m in re.finditer(r"const\s+DevPtr\s+(\w+)\s*=\s*([^;]*);", body):
        out.append((m.start(), "alias", (m.group(1), m.group(2))))
    pat = re.compile(r"(?<![\w:])(" + "|".join(re.escape(c) for c in CALLS) + r")\s*\(")
    for m in pat.finditer(body):
        p = body.index("(", m.end() - 1)
        q = close_paren(body, p)
        out.append((m.start(), "call", (m.group(1), split_args(body[p + 1:q]))))
    out.sort(key=lambda e: e[0])
    return [(kind, line0 + body.count("\n", 0, pos), pay) for pos, kind, pay in out]


class Checker:
    def __init__(self):
        self.alias: dict[str, set[str]] = {}
        self.slot_expr: dict[str, str] = {}  # "W" -> "0", "W + lb" -> "1"
        self.viol: list[str] = []
        self.n_reads = 0
        self.n_copies = 0

    def slots_of(self, expr: str) -> set[str]:
        e = norm_idx(expr)
        got: set[str] = set()
        for m in re.finditer(r"\bslot\s*\[([^\]]+)\]", e):
            got.add(norm_idx(m.group(1)))
        e2 = re.sub(r"\bslot\s*\[[^\]]+\]", " ", e)
        for k, v in sorted(self.slot_expr.items(), key=lambda kv: -len(kv[0])):
            if re.search(r"(?<![\w.])" + re.escape(k).replace(r"\ ", r"\s*") + r"(?![\w])", e2):
                got.add(v)
                e2 = re.sub(r"(?<![\w.])" + re.escape(k).replace(r"\ ", r"\s*") + r"(?![\w])", " ", e2)
        for name, sl in self.alias.items():
            if re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w])", e2):
                got |= sl
        return got

    def run(self, src: str) -> None:
        body, line0 = encode_body(src)
        body = strip_comments(body)
        evs = events(body, line0)
        if not any(k == "slots" for k, _, _ in evs):
            raise ParseError("`const DevPtr slot[2] = {...};` not found")
        # state
        last_copy: dict[str, int] = {}     # slot -> id of the latest copy into it (host order)
        covered: dict[str, int] = {}       # slot -> latest copy id covered by a record of ev_copy_[slot]
        seen: dict[str, int] = {}          # slot -> copy id the compute stream has waited for
        last_read: dict[str, int] = {}     # slot -> step of the latest read on stream
        free_rec: dict[str, int] = {}      # ev_free_[k] / ev_start_ -> step it was recorded on stream
        cs_free: dict[str, int] = {}       # slot -> step of the record the copy stream waited for (WAR)
        step = 0
        in_loop = False
        loop_has_merger_wait = False
        merger_copy_recorded = False
        merger_copy_seen = False
        cid = 0
        for kind, line, pay in evs:
            step += 1
            if kind == "slots":
                self.slot_expr = {norm_idx(pay[0]): "0", norm_idx(pay[1]): "1"}
            elif kind == "alias":
                name, expr = pay
                sl = self.slots_of(stream_branch(expr)) if self.slot_expr else set()
                if name == "slot" or not sl:
                    continue
                self.alias[name] = sl
            elif kind == "loop_begin":
                in_loop = True
                for k in ("s", "1 - s"):  # the copies of earlier iterations / the prologue: recorded, not yet waited
                    cid += 1
                    last_copy[k] = covered[k] = cid
                    seen.pop(k, None)
                    last_read.pop(k, None)
            elif kind == "loop_end":
                in_loop = False
                if not loop_has_merger_wait:
                    self.viol.append(f"R3 line {line}: the loop body never waits ev_copy_[1] on stream (merger weights)")
                if merger_copy_seen and not merger_copy_recorded:
                    self.viol.append(f"R3 line {line}: the merger copy into slot 1 is not followed by eventRecord(ev_copy_[1], cs)")
                if merger_copy_seen and loop_has_merger_wait:
                    # the last iteration's wait follows the merger copy of the iteration before it: covered
                    seen["1"] = last_copy.get("1", 0)
            elif kind == "call":
                fn, args = pay
                if fn == "hip::copyAnyAsync":
                    if len(args) < 4 or args[3] != "cs":
                        continue
                    dst = self.slots_of(args[0])
                    for k in dst:
                        self.n_copies += 1
                        cid += 1
                        last_copy[k] = cid
                        if in_loop and k == "1" and args[0].replace(" ", "") == "W+lb":
                            # the merger copy: its WAR safety depends on the parity of n_layer (slot 1 is free for
                            # good only after the last odd layer), which this model does not evaluate: not checked
                            merger_copy_seen = True
                            merger_copy_recorded = False
                            cs_free.pop(k, None)
                            continue
                        w = cs_free.get(k)
                        if w is None:
                            self.viol.append(f"R2 line {line}: copy into slot {k} on cs without a preceding streamWaitEvent(cs, ev_free_[{k}] / ev_start_)")
                        elif last_read.get(k, -1) > w:
                            self.viol.append(f"R2 line {line}: copy into slot {k} on cs is ordered before a later read of slot {k} on stream")
                        cs_free.pop(k, None)
                elif fn == "hip::eventRecord":
                    ev, st = norm_idx(args[0]), norm_idx(args[1]) if len(args) > 1 else ""
                    m = re.match(r"ev_copy_\s*\[([^\]]+)\]", ev)
                    if m and st == "cs":
                        k = norm_idx(m.group(1))
                        if k in last_copy:
                            covered[k] = last_copy[k]
                        if in_loop and k == "1" and merger_copy_seen:
                            merger_copy_recorded = True
                        continue
                    m = re.match(r"(ev_free_\s*\[([^\]]+)\]|ev_start_)", ev)
                    if m and st == "stream":
                        key = norm_idx(m.group(2)) if m.group(2) else "start"
                        free_rec[key] = step
                elif fn == "hip::streamWaitEvent":
                    st, ev = norm_idx(args[0]), norm_idx(args[1])
                    m = re.match(r"ev_copy_\s*\[([^\]]+)\]", ev)
                    if m and st == "stream":
                        k = norm_idx(m.group(1))
                        if k in covered:
                            seen[k] = covered[k]
                        if in_loop and k == "1":
                            loop_has_merger_wait = True
                        continue
                    m = re.match(r"(ev_free_\s*\[([^\]]+)\]|ev_start_)", ev)
                    if m and st == "cs":
                        key = norm_idx(m.group(2)) if m.group(2) else "start"
                        rec = free_rec.get(key)
                        if rec is None:
                            continue
                        if key == "start":
                            for k in ("0", "1"):
                                cs_free[k] = rec
                        else:
                            cs_free[key] = rec
                            if key == "1":
                                cs_free["1"] = rec
                elif fn in ("hip::launch", "launchGemm"):
                    if fn == "hip::launch":
                        if len(args) < 5 or norm_idx(args[4]) != "stream":
                            continue
                        rd = args[5:]
                    else:
                        if not args or norm_idx(args[0]) != "stream":
                            continue
                        rd = args[1:]
                    sl: set[str] = set()
                    for a in rd:
                        sl |= self.slots_of(a)
                    for k in sorted(sl):
                        self.n_reads += 1
                        if k not in last_copy:
                            self.viol.append(f"R1 line {line}: read of slot {k} before any copy into it")
                        elif seen.get(k) != last_copy[k]:
                            self.viol.append(f"R1 line {line}: {fn} reads slot {k} without a wait on ev_copy_[{k}] covering its latest copy")
                        last_read[k] = step
        if self.n_copies == 0 or self.n_reads == 0:
            raise ParseError(f"nothing found ({self.n_copies} copies, {self.n_reads} reads): the parser no longer matches vision.cpp")


def check(src: str) -> Checker:
    c = Checker()
    c.run(src)
    return c


def selftest(src: str) -> int:
    """the real file passes; each planted reorder (the 12011b4 revert first) is caught"""
    fix_line = "if (!res) hip::streamWaitEvent(stream, ev_copy_[0].get());"
    cases = [
        ("revert 12011b4 (no ev_copy_[0] wait before pos_ln)", fix_line, ""),
        ("no ev_copy_[1] wait before patchify / patch GEMM", "if (!res) hip::streamWaitEvent(stream, ev_copy_[1].get());\n    hip::launch(f_.patchify",
         "hip::launch(f_.patchify"),
        ("no ev_copy_[s] wait at the top of the layer loop", "if (!res) hip::streamWaitEvent(stream, ev_copy_[s].get());", ""),
        ("no ev_copy_[1 - s] wait before the next layer's ln1", "if (!last && !res) hip::streamWaitEvent(stream, ev_copy_[1 - s].get());", ""),
        ("no ev_free_[s] wait before the next copy into slot s", "hip::streamWaitEvent(cs, ev_free_[s].get());", ""),
        ("no record of ev_copy_[0] after the layer-0 copy", "hip::eventRecord(ev_copy_[0].get(), cs);", ""),
    ]
    good = 0
    base = check(src)
    ok0 = not base.viol
    print(f"{'ok  ' if ok0 else 'FAIL'} real file: {len(base.viol)} violation(s), {base.n_reads} reads, {base.n_copies} copies")
    for v in base.viol:
        print("    " + v)
    good += ok0
    for name, find, repl in cases:
        if src.count(find) != 1:
            print(f"FAIL {name}: planting text found {src.count(find)} times (expected 1)")
            continue
        try:
            c = check(src.replace(find, repl))
            hit = bool(c.viol)
            detail = c.viol[0] if c.viol else "no violation"
        except ParseError as ex:
            hit, detail = False, f"parse error {ex}"
        print(f"{'ok  ' if hit else 'FAIL'} {name}: {detail}")
        good += hit
    n = len(cases) + 1
    print(f"vis stream order selftest: {good}/{n} as expected")
    return 0 if good == n else 1


def main(argv: list[str]) -> int:
    path = os.path.join(ROOT, "src", "vision", "vision.cpp")
    if "--src" in argv:
        path = argv[argv.index("--src") + 1]
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as ex:
        print(f"vis stream order check: cannot read {path}: {ex}")
        return 2
    try:
        if "--selftest" in argv:
            return selftest(src)
        c = check(src)
    except ParseError as ex:
        print(f"vis stream order check: parse error: {ex}")
        return 2
    for v in c.viol:
        print(v)
    if c.viol:
        print(f"vis stream order check: FAIL ({len(c.viol)} violation(s))")
        return 1
    print(f"vis stream order check: ok ({c.n_reads} reads, {c.n_copies} copies)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
