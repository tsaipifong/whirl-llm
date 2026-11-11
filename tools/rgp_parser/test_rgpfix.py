#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""test_rgpfix.py -- WHIRL v0252-rgpfix (2026-11-10, PLAN V1-68): the two RGP parser errors found on 2026-11-10.

Error 1 (wave -> dispatch pairing): a wave is given to the last same-PC dispatch whose initiator time, read from SE0's
stream only, is <= the wave start (rgp_parser/struct/sqtt_decode.py, "attribute waves to dispatches"). SE0 logs a
dispatch initiator when SE0 itself gets to it, so the workgroups on engines that drained earlier start BEFORE that time
and land on the previous instance of the kernel, one decode cycle back; with the exp_waves cap this cascades and the
previous instance's wave span ends inside the next cycle. The reported duration min(span, next-launch gap)
(rgp_parser/build.py) then books the host round trip after the kernel as kernel time (v0252-f1arg:
argmax_rank_rows_to "976 us" for a 77-116 us kernel, R9700 int0252 capture q7 9522).
Error 2 (idle): gap_before_us = start minus the end of the PREVIOUS dispatch in start order (build.py), so time under a
long overlapped kernel counts as GPU idle (v0252-hostwait: Swift plain "50-150 us in-layer gaps" whose true idle is ~0).

The tests run the REAL decoder and trace builder (build.build_trace -> struct.sqtt.decode_tokens ->
struct.sqtt_decode.decode) on synthetic gfx12 SQTT token streams written by the small encoder below (token layouts
taken from struct/sqtt_decode.py: _LOOKUP12 patterns / lengths and the parse_se field shifts). Only the .rgp container
is faked (no file is read). Every check reads the corrected column when the parser has it and otherwise the column the
consumers used until 2026-11-10 (dur_us / gap_before_us / wave_span_us / idle_gaps), so on the unfixed parser it fails
with a wrong VALUE.
  (a) a wave whose end would pair with the next cycle      -> test_a_cross_cycle_kernel_time
  (b) a long GEMV overlapping several short kernels        -> test_b_overlap_is_not_idle
  (c) a true host gap between cycles                       -> test_c_host_gap_between_cycles
  (d) existing columns unchanged (pinned from the unfixed parser on the same streams) -> test_d_old_columns_pinned
  (e) error 1 when only SE0 logs initiators (fallback)     -> test_e_fallback_without_engine_initiators
  (f) busy = union of the dispatch's own waves             -> test_f_busy_is_union_of_own_waves
  (g) v0252-rgpse (PLAN V1-96): every engine's ticks converted with its OWN realtime stamps, not SE0's (engines' shader
      ticks drift apart; review v0252-rgpreview F1, minimal repro scratch\\v0252_rgpreview\\indep_test.py probe)
                                                           -> test_g_each_engine_on_its_own_clock (+10 %, -7 %, control)
  (h) the reviewer's own F1 probe (decode() fed per-engine records, SE1 tick base +100 us) -> test_h_reviewer_probe_f1
Run (exit 0 = all pass):
  C:\\ClaudeWorkspace\\tools\\rgp_parser\\.venv\\Scripts\\python.exe -I tools\\rgp_parser\\test_rgpfix.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rgp_parser import build  # noqa: E402

MASK32 = 0xFFFFFFFF
US = 100  # shader clocks per us when a capture has no realtime pairs (decode(): rt_freq 1e8 -> 0.01 us per clock)
T0 = 1000.0  # us added to every scenario time inside the stream (time 0 is the decoder's pre-capture time)
ME, PIPE = 1, 1  # the decode queue of the R9700 captures (me1/pipe1)


class Enc:
    """LSB-first gfx12 SQTT token writer for the tokens the parser reads (HEADER, TIMESTAMP, REG, REG_INIT,
    WAVE_START, WAVE_END). Time only moves through TIMESTAMP tokens (rt=0: the decoder adds the 52-bit value)."""

    def __init__(self, tick_rate=1.0, stamps=False):
        self.v = 0
        self.n = 0
        self.t = 0
        self.tick_rate = tick_rate  # this engine's shader ticks per true us, relative to US (v0252-rgpse: != 1 = drift)
        self.stamps = stamps  # write a realtime TIMESTAMP (true time, 100 MHz) at every new time (v0252-rgpse)
        self.last_rt = -1
        self._put(0b0010001 | (4 << 7), 64)  # HEADER, version 4 = gfx12 (RDNA4)

    def _put(self, val, length):
        self.v |= (val & ((1 << length) - 1)) << self.n
        self.n += length

    def at(self, us):
        sc = int(round((T0 + us * self.tick_rate) * US))
        if sc < self.t:
            raise ValueError("events of one engine must be written in time order")
        if sc > self.t:
            self._put(0b0000001 | ((sc - self.t) << 12), 64)  # TIMESTAMP rt=0 pl=0
            self.t = sc
        rt = int(round((T0 + us) * US))
        if self.stamps and rt > self.last_rt:
            self._put(0b0000001 | (1 << 7) | (rt << 12), 64)  # TIMESTAMP rt=1 pl=0: realtime value (true time)
            self.last_rt = rt

    def reg(self, cs, addr, data):
        self._put(0b1001 | (PIPE << 7) | (ME << 9) | (cs << 15) | (addr << 16) | ((data & MASK32) << 32), 64)

    def pgm(self, pc):
        addr = pc >> 8
        self.reg(1, 0xC, addr & MASK32)  # COMPUTE_PGM_LO
        self.reg(1, 0xD, addr >> 32)  # COMPUTE_PGM_HI
        self.reg(1, 0x7, 32)  # COMPUTE_NUM_THREAD_X: 32 threads = one wave32 per workgroup
        self.reg(1, 0x8, 1)
        self.reg(1, 0x9, 1)

    def init(self):  # REG_INIT: COMPUTE_DISPATCH_INITIATOR write (rtype 2, data bit 0 = 1)
        self._put(0b1110001 | (PIPE << 14) | (ME << 16) | (2 << 18) | (1 << 20), 64)

    def marker(self, grid):  # RGP event marker with dims: 6 dwords through USERDATA_2 / USERDATA_3
        for k, dw in enumerate((1 << 31, 0, 0, grid, 1, 1)):
            self.reg(0, 0xC342 + (k & 1), dw)

    def wstart(self, slot):
        sa, wgp, simd, wid = slot
        self._put(0b01100 | (sa << 7) | (simd << 8) | (wgp << 10) | (wid << 15) | (PIPE << 20) | (ME << 22), 32)

    def wend(self, slot):
        sa, wgp, simd, wid = slot
        self._put(0b10101 | (sa << 8) | (simd << 9) | (wgp << 11) | (wid << 15), 20)

    def blob(self):
        self._put(0, 512)  # NOP padding: every real token is decoded in the padded region
        return self.v.to_bytes((self.n + 7) // 8, "little")


def stream(events, tick_rate=1.0, stamps=False):
    """events: [(time_us, kind, arg)] of ONE engine, already in that engine's order. kind: marker(grid) / pgm(pc) /
    init / ws(slot) / we(slot). tick_rate / stamps: see Enc (scenario g)."""
    e = Enc(tick_rate, stamps)
    for t, kind, arg in events:
        e.at(t)
        if kind == "marker":
            e.marker(arg)
        elif kind == "pgm":
            e.pgm(arg)
        elif kind == "init":
            e.init()
        elif kind == "ws":
            e.wstart(arg)
        elif kind == "we":
            e.wend(arg)
        else:
            raise ValueError(kind)
    return e.blob()


class _FakeRdf:
    """The .rgp container: no chunks (no code objects, SPM, queue info). The SQTT blobs come from the test."""

    @classmethod
    def from_path(cls, _path):
        return cls()

    def by_id(self, _name):
        return []

    def counts(self):
        return {}

    def data(self, _c):
        return b""


def trace_of(blobs):
    old_rdf, old_extract = build.RdfFile, build.sqtt.extract_blobs
    build.RdfFile = _FakeRdf
    build.sqtt.extract_blobs = lambda _rdf: blobs
    try:
        return build.build_trace("synthetic.rgp", spm_enabled=False)
    finally:
        build.RdfFile, build.sqtt.extract_blobs = old_rdf, old_extract


# ---- scenarios ----------------------------------------------------------------------------------------------------
PC_G, PC_R = 0x10000, 0x20000  # decode cycle: a "GEMV" (2 workgroups) then an "argmax" (2 workgroups)
PC_L, PC_S1, PC_S2, PC_N, PC_M = 0x40000, 0x50000, 0x60000, 0x70000, 0x80000


def scenario_a(engine_inits=True, se1_tick_rate=None):
    """Three decode cycles, 1000 us apart. G_k: initiator seen by both engines at B (GPU idle before), one workgroup
    per engine (se0 1..100, se1 1..60). R_k: se1 drains first, logs R's initiator at B+61 and starts its workgroup
    there (61..140); se0 logs the initiator only at B+101 (when ITS G wave ended) and runs 101..180. True idle:
    180 -> 1001 and 1180 -> 2001 (the host round trip). With engine_inits=False se1 carries no REG_INIT tokens."""
    se0, se1 = [], []
    for k in range(3):
        b = 1000.0 * k
        se0 += [(b, "marker", 2), (b, "pgm", PC_G), (b, "init", None), (b + 1, "ws", (0, 0, 0, 0)),
                (b + 100, "we", (0, 0, 0, 0))]
        se1 += [(b, "pgm", PC_G)] + ([(b, "init", None)] if engine_inits else []) + [
            (b + 1, "ws", (0, 1, 0, 0)), (b + 60, "we", (0, 1, 0, 0))]
        se1 += [(b + 61, "pgm", PC_R)] + ([(b + 61, "init", None)] if engine_inits else []) + [
            (b + 61, "ws", (0, 1, 0, 1)), (b + 140, "we", (0, 1, 0, 1))]
        se0 += [(b + 101, "marker", 2), (b + 101, "pgm", PC_R), (b + 101, "init", None), (b + 101, "ws", (0, 0, 0, 1)),
                (b + 180, "we", (0, 0, 0, 1))]
    if se1_tick_rate is not None:  # scenario g: both engines carry realtime stamps, se1's shader ticks drift
        return [stream(se0, 1.0, True), stream(se1, se1_tick_rate, True)]
    return [stream(se0), stream(se1)]


PC_U = 0x90000


def scenario_c():
    """One kernel U with two workgroups that do not overlap: se0 10..30, se1 50..70 (both engines log U's initiator at
    9). The GPU is idle 30 -> 50 INSIDE the dispatch: its busy time is the union 40 us, its first start -> last end 60."""
    se0 = [(9, "marker", 2), (9, "pgm", PC_U), (9, "init", None), (10, "ws", (0, 0, 0, 0)), (30, "we", (0, 0, 0, 0))]
    se1 = [(9, "pgm", PC_U), (9, "init", None), (50, "ws", (0, 1, 0, 0)), (70, "we", (0, 1, 0, 0))]
    return [stream(se0), stream(se1)]


def scenario_b():
    """A long kernel L (one workgroup on se1, 1..500) with three short independent kernels S1 / S2 / N on se0 under it
    (10..20, 30..40, 60..70) and M after it (510..520). se1 logs the initiators of S1 / S2 / N when it drains (500).
    True idle: only 500 -> 510 (10 us, L -> M)."""
    se0 = [(0, "marker", 1), (0, "pgm", PC_L), (0, "init", None),
           (9, "marker", 1), (9, "pgm", PC_S1), (9, "init", None), (10, "ws", (0, 0, 0, 0)), (20, "we", (0, 0, 0, 0)),
           (29, "marker", 1), (29, "pgm", PC_S2), (29, "init", None), (30, "ws", (0, 0, 0, 1)), (40, "we", (0, 0, 0, 1)),
           (59, "marker", 1), (59, "pgm", PC_N), (59, "init", None), (60, "ws", (0, 0, 0, 2)), (70, "we", (0, 0, 0, 2)),
           (505, "marker", 1), (505, "pgm", PC_M), (505, "init", None), (510, "ws", (0, 0, 0, 3)),
           (520, "we", (0, 0, 0, 3))]
    se1 = [(0, "pgm", PC_L), (0, "init", None), (1, "ws", (0, 1, 0, 0)), (500, "we", (0, 1, 0, 0)),
           (500, "pgm", PC_S1), (500, "init", None), (500, "pgm", PC_S2), (500, "init", None),
           (500, "pgm", PC_N), (500, "init", None), (505, "pgm", PC_M), (505, "init", None)]
    return [stream(se0), stream(se1)]


# ---- helpers ------------------------------------------------------------------------------------------------------
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


def by_pc(trace, pc):
    return sorted((d for d in trace.dispatches if d.pc == pc), key=lambda d: d.launch_ts_us)


def kernel_time(d):
    """Corrected per-dispatch kernel time when the parser has it (busy_v2_us = union of the dispatch's own waves),
    else the duration every consumer read until 2026-11-10 (dur_us)."""
    v = getattr(d, "busy_v2_us", None)
    return ("busy_v2_us", v) if v is not None else ("dur_us", d.dur_us)


def wave_extent(d):
    v = getattr(d, "dur_v2_us", None)
    return ("dur_v2_us", v) if v is not None else ("wave_span_us", d.wave_span_us)


def idle_before(d):
    v = getattr(d, "idle_before_v2_us", None)
    return ("idle_before_v2_us", v) if v is not None else ("gap_before_us", d.gap_before_us)


def idle_windows(trace, min_us):
    w = getattr(trace, "idle_windows", None)
    if w is not None:
        return "idle_windows", [x for x in w if x["dur_us"] > min_us]
    return "idle_gaps()", [x for x in trace.idle_gaps(10_000) if x["dur_us"] > min_us]


def near(a, b, tol=0.05):
    return a is not None and abs(a - b) <= tol


# ---- tests --------------------------------------------------------------------------------------------------------
def test_a_cross_cycle_kernel_time():
    print("(a) a wave whose end would pair with the next cycle")
    t = trace_of(scenario_a())
    r = by_pc(t, PC_R)
    check(len(r) == 3, f"three R dispatches decoded (got {len(r)})")
    for k, d in enumerate(r):
        name, v = kernel_time(d)
        check(near(v, 119.0), f"R{k} kernel time = union of its two workgroups 119 us (field {name} = {v})")
        name, v = wave_extent(d)
        check(v is not None and v <= 200.0, f"R{k} first wave start -> last wave end stays in its cycle (<= 200 us; "
                                            f"field {name} = {v})")


def test_b_overlap_is_not_idle():
    print("(b) a long GEMV overlapping several short kernels")
    t = trace_of(scenario_b())
    (lk,), (s2,), (nk,), (mk,) = by_pc(t, PC_L), by_pc(t, PC_S2), by_pc(t, PC_N), by_pc(t, PC_M)
    name, v = kernel_time(lk)
    check(near(v, 499.0), f"L kernel time 499 us (field {name} = {v})")
    for label, d, want in (("S2", s2, 0.0), ("N", nk, 0.0), ("M", mk, 10.0)):
        name, v = idle_before(d)
        check(near(v, want), f"GPU idle before {label} = {want} us while L runs / after it (field {name} = {v})")
    src, w = idle_windows(t, 0.0)
    tot = round(sum(x["dur_us"] for x in w), 3)
    check(near(tot, 10.0), f"total GPU idle inside the window = 10 us ({src}: {tot})")


def test_c_host_gap_between_cycles():
    print("(c) a true host gap between cycles")
    t = trace_of(scenario_a())
    g = by_pc(t, PC_G)
    for k in (1, 2):
        name, v = idle_before(g[k])
        check(near(v, 821.0), f"GPU idle before G{k} (host round trip) = 821 us (field {name} = {v})")
    src, w = idle_windows(t, 50.0)
    durs = sorted(round(x["dur_us"], 3) for x in w)
    check(len(durs) == 2 and all(near(x, 821.0) for x in durs), f"two idle windows > 50 us of 821 us ({src}: {durs})")
    if w and "prev_pc" in w[0]:
        pairs = sorted((x["prev_pc"], x["next_pc"]) for x in w)
        check(pairs == [(PC_R, PC_G), (PC_R, PC_G)], f"windows labelled argmax -> next cycle's GEMV ({pairs})")


# Columns of the unfixed parser on the two scenarios (recorded 2026-11-10 from wt_0252_rgpfix tools/rgp_parser at
# int0252 63de269, before any change; hand-checked against the scenario arithmetic in the HANDOFF). Per dispatch in
# ts_us order: (pc, ts_us, launch_ts_us, dur_us, gap_before_us, idle_after_us, wavefronts, exp_waves, wave_span_us,
# occupancy_wavefronts).
PIN_A = [
    (PC_G, 1.0, 0.0, 99.0, 0.0, 0.0, 2, 2, 99.0, 2),
    (PC_R, 61.0, 101.0, 119.0, 0.0, 821.0, 2, 2, 119.0, 2),
    (PC_G, 1001.0, 1000.0, 99.0, 821.0, 1.0, 2, 2, 99.0, 2),
    (PC_R, 1101.0, 1101.0, 899.0, 1.0, 1.0, 2, 2, 1039.0, 2),
    (PC_G, 2001.0, 2000.0, 99.0, 1.0, 1.0, 2, 2, 99.0, 2),
    (PC_R, 2101.0, 2101.0, 79.0, 1.0, 0.0, 2, 2, 79.0, 1),
]
PIN_B = [
    (PC_L, 1.0, 0.0, 9.0, 0.0, 0.0, 1, 1, 499.0, 1),
    (PC_S1, 10.0, 9.0, 10.0, 0.0, 10.0, 1, 1, 10.0, 1),
    (PC_S2, 30.0, 29.0, 10.0, 10.0, 20.0, 1, 1, 10.0, 1),
    (PC_N, 60.0, 59.0, 10.0, 20.0, 440.0, 1, 1, 10.0, 1),
    (PC_M, 510.0, 505.0, 10.0, 440.0, 0.0, 1, 1, 10.0, 1),
]


def rows_of(trace):
    out = []
    for d in sorted(trace.dispatches, key=lambda x: x.ts_us):
        out.append((d.pc, round(d.ts_us, 3), round(d.launch_ts_us, 3), round(d.dur_us, 3), round(d.gap_before_us, 3),
                    round(d.idle_after_us, 3), d.wavefronts, d.exp_waves, round(d.wave_span_us, 3),
                    d.occupancy_wavefronts))
    return out


def test_d_old_columns_pinned():
    print("(d) existing columns unchanged")
    for name, blobs, pin in (("A", scenario_a(), PIN_A), ("B", scenario_b(), PIN_B)):
        got = rows_of(trace_of(blobs))
        check(len(got) == len(pin), f"scenario {name}: {len(pin)} dispatches (got {len(got)})")
        for i, (g, p) in enumerate(zip(got, pin)):
            check(g == p, f"scenario {name} row {i}: old columns {p} (got {g})")


def test_e_fallback_without_engine_initiators():
    print("(e) error 1 when only SE0 logs dispatch initiators (fallback attribution)")
    t = trace_of(scenario_a(engine_inits=False))
    r = by_pc(t, PC_R)
    for k, d in enumerate(r):
        name, v = kernel_time(d)
        check(near(v, 119.0), f"R{k} kernel time 119 us without engine initiators (field {name} = {v})")
    g = by_pc(t, PC_G)
    name, v = idle_before(g[2])
    check(near(v, 821.0), f"GPU idle before G2 = 821 us (field {name} = {v})")


def test_f_busy_is_union_of_own_waves():
    print("(f) per-dispatch busy = union of the dispatch's own waves (idle inside a dispatch is not kernel time)")
    t = trace_of(scenario_c())
    (u,) = by_pc(t, PC_U)
    name, v = kernel_time(u)
    check(near(v, 40.0), f"U kernel time = union 40 us, not first start -> last end 60 (field {name} = {v})")
    src, w = idle_windows(t, 0.0)
    durs = [round(x["dur_us"], 3) for x in w]
    check(durs == [20.0], f"one idle window of 20 us inside U ({src}: {durs})")


def edge_spread(trace):
    a = trace.meta.extra.get("attribution_v2")
    return a.get("host_wait_edges") if isinstance(a, dict) else None


def test_g_each_engine_on_its_own_clock(rate=1.1, label="se1 shader ticks run 10 % fast"):
    """v0252-rgpse (PLAN V1-96, review v0252-rgpreview F1): each engine's SQTT shader ticks are its own accumulated
    deltas and drift against the other engines (R9700 captures: up to ~300 us by the end of a 60 ms capture); only the
    realtime TIMESTAMPs are a common clock. Scenario a with realtime stamps on both engines and se1's ticks running at
    `rate`: every v2 quantity must equal the true one (each engine converted with ITS OWN realtime pairs), and the
    first wave after each host wait must start at the same time on every engine."""
    print(f"(g) each engine converted with its own realtime stamps ({label})")
    t = trace_of(scenario_a(se1_tick_rate=rate))
    r, g = by_pc(t, PC_R), by_pc(t, PC_G)
    for k, d in enumerate(r):
        name, v = kernel_time(d)
        check(near(v, 119.0), f"R{k} kernel time = union 61..180 = 119 us (field {name} = {v})")
        name, v = wave_extent(d)
        check(near(v, 119.0), f"R{k} first start -> last end = 119 us (field {name} = {v})")
        v = getattr(d, "launch_v2_us", None)
        check(near(v, 1000.0 * k + 61.0), f"R{k} first initiator on any engine = {1000 * k + 61} us (launch_v2_us = {v})")
    for k in (1, 2):
        name, v = idle_before(g[k])
        check(near(v, 821.0), f"GPU idle before G{k} (host round trip) = 821 us (field {name} = {v})")
    src, w = idle_windows(t, 50.0)
    durs = sorted(round(x["dur_us"], 3) for x in w)
    check(len(durs) == 2 and all(near(x, 821.0) for x in durs), f"two idle windows > 50 us of 821 us ({src}: {durs})")
    e = edge_spread(t)
    mx = e.get("spread_us_max") if isinstance(e, dict) else None
    check(isinstance(e, dict) and e.get("windows") == 2 and mx is not None and mx <= 0.05,
          f"first wave after each host wait starts together on every engine (attribution_v2.host_wait_edges = {e})")


def test_h_reviewer_probe_f1():
    """The reviewer's minimal F1 repro (scratch\\v0252_rgpreview\\indep_test.py, probe section; written independently of
    this file's encoder): decode() fed per-engine records directly. SE0 initiators G@0 P@100 G@1000 P@1100, SE1 G@0 P@60
    G@1000 P@1050; SE0 waves G 0-90, P 100-200, G 1000-1090, P 1100-1200; SE1 waves G 0-40, P 60-150, G 1000-1040,
    P 1050-1130 (true us). SE1's tick base runs +100 us ahead of SE0's and its realtime pairs say so. Truth: one idle
    window 200 -> 1000 (800 us), P#1 span 60 -> 200 (140 us)."""
    print("(h) reviewer's F1 probe: SE1 tick base +100 us, own realtime pairs")
    from rgp_parser.struct import sqtt_decode as sd
    G, P, U = 0x2000, 0x1000, 100

    def build_se(inits, waves, s):
        ev = [(t * U + s, 0, pc) for t, pc in inits]
        for k, (a, b, pc) in enumerate(waves):
            ev += [(a * U + s, 1, (k, pc)), (b * U + s, 2, (k, pc))]
        ev.sort(key=lambda x: (x[0], x[1]))
        occ, ins, n = [], [], 0
        for tick, kind, pl in ev:
            if kind == 0:
                n += 1
                ins.append((tick, ME, PIPE, pl))
            else:
                k, pc = pl
                occ.append([tick, 1, k, 0, 0, ME, PIPE, pc, n - 1] if kind == 1 else [tick, 0, k, 0, 0, 0, 0, pc, -1])
        return occ, ins, [(int(x * U) + s, int(x * U)) for x in range(0, 1301, 50)]

    i0 = [(0, G), (100, P), (1000, G), (1100, P)]
    o0, n0, r0 = build_se(i0, [(0, 90, G), (100, 200, P), (1000, 1090, G), (1100, 1200, P)], 0)
    o1, n1, r1 = build_se([(0, G), (60, P), (1000, G), (1050, P)],
                          [(0, 40, G), (60, 150, P), (1000, 1040, G), (1050, 1130, P)], 100 * U)
    disps = [{"se": 0, "me": ME, "pipe": PIPE, "t": t * U, "pc": pc, "vgpr": 16, "sgpr": 128, "lds": 0, "wg": 32,
              "flags": 0} for t, pc in i0]
    recs = [(disps, [], r0, o0, n0), ([], [], r1, o1, n1)]
    orig = sd.parse_se
    sd.parse_se = lambda blob, se_index, collect_de: recs[se_index]
    try:
        r = sd.decode([b"", b""], wave_size=32)
    finally:
        sd.parse_se = orig
    big = [w for w in r["idle"] if w["b"] - w["a"] > 50]
    ln = round(big[0]["b"] - big[0]["a"], 4) if len(big) == 1 else None
    check(near(ln, 800.0), f"one GPU idle window of 800 us (got {[round(w['b'] - w['a'], 4) for w in big]})")
    d1 = r["dispatches"][1]
    sp = round(d1["end2"] - d1["ts2"], 4) if d1.get("end2") is not None else None
    check(near(sp, 140.0), f"P#1 first start -> last end 140 us (got {sp})")
    check(near(d1.get("busy2"), 140.0), f"P#1 busy (union) 140 us (got {d1.get('busy2')})")


def main():
    for f in (test_a_cross_cycle_kernel_time, test_b_overlap_is_not_idle, test_c_host_gap_between_cycles,
              test_d_old_columns_pinned, test_e_fallback_without_engine_initiators, test_f_busy_is_union_of_own_waves,
              test_g_each_engine_on_its_own_clock, lambda: test_g_each_engine_on_its_own_clock(0.93, "se1 7 % slow"),
              lambda: test_g_each_engine_on_its_own_clock(1.0, "control: no drift, stamps on"),
              test_h_reviewer_probe_f1):
        f()
    print(f"RESULT {'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed check(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
