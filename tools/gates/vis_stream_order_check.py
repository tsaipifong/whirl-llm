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

Rules (each violation is printed with its line and the call):
  R0 the parse found exactly EXPECTED_READS slot reads and EXPECTED_COPIES copies (arch025-gt2 C3): a launch the parser
     silently stops seeing (a new launch wrapper, the stream moved to another argument) fails instead of passing on
     the reads that are left. When vision.cpp really gains / loses a read or a copy, check the new code by hand and
     update the two numbers.
  R1 (RAW) a read of slot k on stream needs a wait on ev_copy_[k] after the record that covers the latest copy into
     slot k (prologue: exact; loop body: the wait must be in the body before the read, no copy into k in between)
  R2 (WAR) a copy into slot k on cs needs a preceding streamWaitEvent(cs, ev_free_[k] | ev_start_) whose event was
     recorded on stream after the last read of slot k (prologue: ev_start_ before any read)
     (not for the merger copy into W + lb: whether slot 1 is free then depends on the parity of n_layer)
  R3 the merger reads after the loop (slot 1 = W + lb) need a wait on ev_copy_[1] in the loop body that comes after
     the body's eventRecord(ev_copy_[1], cs) of the merger copy in host order of the last iteration (checked as: the
     body has the wait and a record of ev_copy_[1] after the merger copy)
  R4 the guard of every stream-schedule call (a copy on cs, a record / wait of ev_copy_ / ev_free_ / ev_start_) is
     read from the enclosing `if (...)` / `else if` / `else` and split at top-level &&; every conjunct must be in
     GUARDS_OK (`!res`, the loop conditions last / !last / l + 2 < hp.n_layer / l + 2 == hp.n_layer, hp.n_layer > 1)
     or the call is unconditional (arch025-gt2 C2). A call whose guard names `res` any other way (e.g. `if (res)`)
     does not run on the stream path: it is a violation and the model also leaves it out, so the read it should
     have ordered fails R1 too. An unknown condition is a violation (fail closed), the model keeps the call.
  R5 inside the layer loop a copy goes into slot[s] (symbolic) or is the merger copy into W + lb; a copy into a
     concrete slot (slot[0], slot[1], W) inside the loop is a violation: not modelled (see the loop boundary below).
Scope / limits: indices are symbols (0, 1, s, 1 - s); s and 1 - s are never assumed to equal 0 or 1, so a wait for
one does not count for another (the safe side). The loop body is checked as one iteration. Branch conditions are not
evaluated beyond R4: the loop conditions (l + 2 < n, last, !last) and hp.n_layer > 1 are treated as true, so a
swapped `last` / `!last` is not seen. Loop boundary (arch025-gt2 C5): last_read of s / 1 - s starts empty at the loop;
this is sound because a copy into slot[s] in the loop needs a cs wait on ev_free_[s] recorded in the loop body
(prologue events are keyed 0 / 1 / start and never match s), and such a record comes, on `stream`, after every read
of the prologue and of earlier iterations; only reads of the same iteration before the record remain, and R2 checks
those. A loop copy into a concrete slot would break that argument (it could use the prologue's ev_free_[1] while the
loop's reads of s / 1 - s are not aliased to 0 / 1), hence R5. The merger copy into W + lb is not WAR-checked (above).
Timing is not modelled: that the device really reorders, and that the fix removes the wrong embedding, is shown only
by the GPU repro (vis-encode WHIRL_TEST_VIS_COPY_DELAY_MB, vis_ab).

    python -I tools/gates/vis_stream_order_check.py [--src src/vision/vision.cpp]
    python -I tools/gates/vis_stream_order_check.py --selftest
The selftest pins, per planted change, the exact set of (rule, line) it must produce; the line is found from an
anchor text in the planted source, so it follows edits of vision.cpp (arch025-gt2 C4).
Last line "vis stream order check: ok (N reads, M copies)" / "...: FAIL (K violation(s))"; exit 0 / 1, 2 = parse error.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CALLS = ("hip::copyAnyAsync", "hip::eventRecord", "hip::streamWaitEvent", "hip::launch", "launchGemm")
# R0: what the parse of the current vision.cpp finds (arch025-gt2 C3); update only after checking a real change
EXPECTED_READS = 17
EXPECTED_COPIES = 5
# R4: the conditions a stream-schedule call may sit under (normalised: single spaces)
GUARDS_OK = frozenset({"!res", "!last", "last", "hp.n_layer > 1", "l + 2 < hp.n_layer", "l + 2 == hp.n_layer",
                       "!(l + 2 < hp.n_layer)"})


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


def split_and(cond: str) -> list[str]:
    """top-level && conjuncts, normalised; outer parentheses around the whole condition are dropped"""
    c = norm_idx(cond)
    while c.startswith("(") and c.endswith(")") and close_paren(c, 0) == len(c) - 1:
        c = norm_idx(c[1:-1])
    out, depth, cur, i = [], 0, "", 0
    while i < len(c):
        ch = c[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and c.startswith("&&", i):
            out.append(norm_idx(cur))
            cur, i = "", i + 2
            continue
        cur += ch
        i += 1
    if cur.strip():
        out.append(norm_idx(cur))
    return out


def block_end(body: str, j: int) -> int:
    """index of the last char of the statement / block that starts at j"""
    if body[j] == "{":
        depth = 0
        for k in range(j, len(body)):
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    return k
        raise ParseError("unbalanced braces in a guarded block")
    depth = 0
    for k in range(j, len(body)):
        if body[k] in "([{":
            depth += 1
        elif body[k] in ")]}":
            depth -= 1
        elif body[k] == ";" and depth == 0:
            return k
    raise ParseError("guarded statement without ';'")


def guard_regions(body: str) -> list[tuple[int, int, list[str]]]:
    """(start, end, conjuncts) of every if / else if / else body (arch025-gt2 C2); an else branch gets the negation
    of each earlier condition of its chain as `!(cond)`"""
    regs: list[tuple[int, int, list[str], list[str]]] = []  # start, end, conjuncts, chain conditions so far
    for m in re.finditer(r"\b(?:if\s*\(|else\b(?!\s*if\b))", body):
        is_if = m.group(0).startswith("if")
        chain: list[str] = []
        before = body[:m.start()]
        else_kw = re.search(r"\belse\s*$", before) if is_if else True
        if else_kw:
            kw_at = else_kw.start() if is_if else m.start()
            prev = [r for r in regs if re.fullmatch(r"\s*", body[r[1] + 1:kw_at])]
            if not prev:
                raise ParseError("else without a matching if")
            chain = prev[-1][3]
        if is_if:
            p = body.index("(", m.start())
            q = close_paren(body, p)
            cond = norm_idx(body[p + 1:q])
            j = q + 1
            conj = split_and(cond) + [f"!({c})" for c in chain]
            chain2 = chain + [cond]
        else:
            j = m.end()
            conj = [f"!({c})" for c in chain]
            chain2 = chain
        while body[j].isspace():
            j += 1
        regs.append((j, block_end(body, j), conj, chain2))
    return [(a, b, c) for a, b, c, _ in regs]


def guard_at(regs, pos: int) -> list[str]:
    out: list[str] = []
    for a, b, c in regs:
        if a <= pos <= b:
            out += c
    return out


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
    regs = guard_regions(body)
    pat = re.compile(r"(?<![\w:])(" + "|".join(re.escape(c) for c in CALLS) + r")\s*\(")
    for m in pat.finditer(body):
        p = body.index("(", m.end() - 1)
        q = close_paren(body, p)
        out.append((m.start(), "call", (m.group(1), split_args(body[p + 1:q]), guard_at(regs, m.start()))))
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

    def run(self, src: str, pin_counts: bool = True) -> None:
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
                fn, args, guard = pay
                if self.sched_call(fn, args):
                    bad = [g for g in guard if g not in GUARDS_OK]
                    if bad:
                        self.viol.append(f"R4 line {line}: {label(fn, args)} sits under `{' && '.join(bad)}`: a stream-"
                                         f"schedule call may only be guarded by !res and the known loop conditions")
                    if any(re.search(r"(?<![\w.])res(?![\w])", g) and g != "!res" for g in guard):
                        continue  # not on the stream (!res) path: the model leaves it out
                if fn == "hip::copyAnyAsync":
                    if len(args) < 4 or args[3] != "cs":
                        continue
                    dst = self.slots_of(args[0])
                    for k in dst:
                        self.n_copies += 1
                        cid += 1
                        last_copy[k] = cid
                        merger = in_loop and k == "1" and args[0].replace(" ", "") == "W+lb"
                        if in_loop and k in ("0", "1") and not merger:
                            self.viol.append(f"R5 line {line}: {label(fn, args)} copies into concrete slot {k} inside the "
                                             f"layer loop: not modelled (loop copies go into slot[s]; the merger into W + lb)")
                            cs_free.pop(k, None)
                            continue
                        if merger:
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
                            self.viol.append(f"R1 line {line}: {label(fn, args)} reads slot {k} before any copy into it")
                        elif seen.get(k) != last_copy[k]:
                            self.viol.append(f"R1 line {line}: {label(fn, args)} reads slot {k} without a wait on ev_copy_[{k}] covering its latest copy")
                        last_read[k] = step
        if self.n_copies == 0 or self.n_reads == 0:
            raise ParseError(f"nothing found ({self.n_copies} copies, {self.n_reads} reads): the parser no longer matches vision.cpp")
        if pin_counts and (self.n_reads, self.n_copies) != (EXPECTED_READS, EXPECTED_COPIES):
            self.viol.append(f"R0: parsed {self.n_reads} reads, {self.n_copies} copies; expected {EXPECTED_READS} reads, "
                             f"{EXPECTED_COPIES} copies (a launch / copy the parser no longer sees, e.g. a new wrapper or a "
                             f"moved stream argument; after checking a real change update EXPECTED_READS / EXPECTED_COPIES)")

    @staticmethod
    def sched_call(fn: str, args: list[str]) -> bool:
        """a call of the stream schedule (R4): a copy on cs, a record / wait of a copy / free / start event"""
        if fn == "hip::copyAnyAsync":
            return len(args) >= 4 and args[3] == "cs"
        if fn in ("hip::eventRecord", "hip::streamWaitEvent"):
            return any(re.search(r"\bev_(copy_|free_|start_)", a) for a in args)
        return False


def label(fn: str, args: list[str]) -> str:
    """the call as it reads in vision.cpp, short: the kernel / the weight / the event"""
    if fn == "hip::launch" and args:
        return f"hip::launch({args[0]})"
    if fn == "launchGemm" and len(args) > 1:
        return f"launchGemm({args[0]}, {args[1]})"
    if fn == "hip::copyAnyAsync" and args:
        return f"hip::copyAnyAsync({args[0]})"
    return f"{fn}({', '.join(args[:2])})"


def check(src: str, pin_counts: bool = True) -> Checker:
    c = Checker()
    c.run(src, pin_counts)
    return c


def selftest(src: str, check_fn=None) -> int:
    """the real file passes with the pinned counts; each planted change (the 12011b4 revert first) yields exactly
    the pinned set of (rule, line). The line comes from an anchor text in the planted source (arch025-gt2 C4)."""
    check_fn = check_fn or check
    fix_line = "if (!res) hip::streamWaitEvent(stream, ev_copy_[0].get());"
    top_s = "if (!res) hip::streamWaitEvent(stream, ev_copy_[s].get());"
    pos_ln = "hip::launch(f_.pos_ln"
    down_b = "hip::launch(f_.resid_ln, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, X, T, L + lo.down_b"
    # every read of slot s in the layer body (the 8 the loop-top wait orders)
    body_s = ["launchGemm(stream, L + lo.qkv_w", "hip::launch(f_.qkv_prep", "launchGemm(stream, L + lo.out_w",
              "hip::launch(f_.resid_ln, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, X, T, L + lo.out_b",
              "launchGemm(stream, L + lo.up_w", "hip::launch(f_.gelu, Dim3{N", "launchGemm(stream, L + lo.down_w", down_b]
    # (name, find, replace, [(rule, anchor | None)], pin the counts)
    cases = [
        ("revert 12011b4 (no ev_copy_[0] wait before pos_ln)", fix_line, "", [("R1", pos_ln)], True),
        ("no ev_copy_[1] wait before patchify / patch GEMM", "if (!res) hip::streamWaitEvent(stream, ev_copy_[1].get());\n    hip::launch(f_.patchify",
         "hip::launch(f_.patchify", [("R1", "launchGemm(stream, misc + so_.patch_w"), ("R1", pos_ln)], True),
        ("no ev_copy_[s] wait at the top of the layer loop", top_s, "", [("R1", a) for a in body_s], True),
        ("no ev_copy_[1 - s] wait before the next layer's ln1", "if (!last && !res) hip::streamWaitEvent(stream, ev_copy_[1 - s].get());", "",
         [("R1", down_b)], True),
        ("no ev_free_[s] wait before the next copy into slot s", "hip::streamWaitEvent(cs, ev_free_[s].get());", "",
         [("R2", "hip::copyAnyAsync(slot[s]")], True),
        ("no record of ev_copy_[0] after the layer-0 copy", "hip::eventRecord(ev_copy_[0].get(), cs);", "", [("R1", pos_ln)], True),
        # arch025-gt2 C2: the guard of the 12011b4 wait inverted; an unknown extra condition on the loop-top wait
        ("C2 `if (!res)` -> `if (res)` on the ev_copy_[0] wait", fix_line, "if (res) hip::streamWaitEvent(stream, ev_copy_[0].get());",
         [("R4", "if (res) hip::streamWaitEvent(stream, ev_copy_[0]"), ("R1", pos_ln)], True),
        ("C2 unknown condition `!res && l > 0` on the loop-top wait", top_s, "if (!res && l > 0) hip::streamWaitEvent(stream, ev_copy_[s].get());",
         [("R4", "if (!res && l > 0) hip::streamWaitEvent")], True),
        # arch025-gt2 C3: a read behind a launch wrapper the parser does not know is silently skipped
        ("C3 a read moved behind a new wrapper (launchGemmEx)", "launchGemm(stream, L + lo.up_w", "launchGemmEx(stream, L + lo.up_w", [("R0", None)], True),
        # arch025-gt2 C5: a loop copy into concrete slot 1 that waits the prologue's ev_free_[1] (the loop's reads of
        # s / 1 - s are not aliased to slot 1, so R2 alone would pass it: in an odd iteration slot 1 is slot s, which
        # the iteration before read as Ln); recorded on ev_copy_[1], so no RAW violation hides the WAR
        ("C5 loop copy into slot[1] after the prologue's ev_free_[1]", top_s,
         "if (!res) hip::streamWaitEvent(cs, ev_free_[1].get());\n        if (!res) hip::copyAnyAsync(slot[1], host0, lb, cs);\n        "
         "if (!res) hip::eventRecord(ev_copy_[1].get(), cs);\n        " + top_s,
         [("R5", "if (!res) hip::copyAnyAsync(slot[1], host0, lb, cs)")], False),
    ]

    def got_set(c) -> set:
        out = set()
        for v in c.viol:
            m = re.match(r"(R\d)(?: line (\d+))?", v)
            out.add((m.group(1), int(m.group(2)) if m.group(2) else None) if m else (v, None))
        return out

    def want_set(s2: str, want) -> set | str:
        out = set()
        for rule, anchor in want:
            if anchor is None:
                out.add((rule, None))
                continue
            if s2.count(anchor) != 1:
                return f"anchor {anchor!r} found {s2.count(anchor)} times (expected 1)"
            out.add((rule, s2.count("\n", 0, s2.index(anchor)) + 1))
        return out

    good = 0
    base = check_fn(src, True)
    ok0 = not base.viol
    print(f"{'ok  ' if ok0 else 'FAIL'} real file: {len(base.viol)} violation(s), {base.n_reads} reads, {base.n_copies} copies")
    for v in base.viol:
        print("    " + v)
    good += ok0
    first = None
    for name, find, repl, want, pin in cases:
        if src.count(find) != 1:
            print(f"FAIL {name}: planting text found {src.count(find)} times (expected 1)")
            continue
        s2 = src.replace(find, repl)
        exp = want_set(s2, want)
        if isinstance(exp, str):
            print(f"FAIL {name}: {exp}")
            continue
        try:
            c = check_fn(s2, pin)
            got = got_set(c)
            hit = got == exp
            detail = "; ".join(c.viol) if c.viol else "no violation"
        except ParseError as ex:
            got, hit, detail = set(), False, f"parse error {ex}"
        if first is None:
            first = got
        print(f"{'ok  ' if hit else 'FAIL'} {name}: {detail}")
        if not hit:
            print(f"    expected {sorted(exp, key=str)}, got {sorted(got, key=str)}")
        good += hit
    # C4 control: the matcher itself rejects a right violation found for the wrong rule
    s1 = src.replace(fix_line, "")
    wrong = want_set(s1, [("R2", pos_ln)])
    ctl = first is not None and isinstance(wrong, set) and first != wrong
    print(f"{'ok  ' if ctl else 'FAIL'} control: an R2 expectation for the 12011b4 revert is not met")
    good += ctl
    n = len(cases) + 2
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
