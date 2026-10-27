# SPDX-License-Identifier: Apache-2.0
"""P-size (arch024 PLAN v3 N3): the server's startup sizing, checked as "same inputs -> same decisions".

Fields of the startup lines fall in three classes (FIELDS below, with the server_main.cpp line that prints them):
  (A) set by the model or the user's settings: compared across runs as they are;
  (B) measured inputs (free / total VRAM, WDDM budget and usage, pinned memory of this and every process, RAM): never
      compared by value, only checked for presence and format in the "startup inputs:" line;
  (C) decisions computed from the inputs: recomputed by whirl-startup-replay (tests/pins/startup_replay.cpp, pure
      functions only) from this run's "startup inputs:" line and compared with this run's log lines, word for word.
  glue: (C) values only server_main's glue decides (the order of the steps, the off conditions, the line formats, the
      issue-#5 pool re-check, the --parallel adjust loop); the replay cannot cover them, the cross-run compare (when the
      (B) inputs are equal) and the B0 unit / regex tests do.

    python tools/gates/startup_lines.py check SERVER.log --replay whirl-startup-replay.exe
    python tools/gates/startup_lines.py selftest --replay whirl-startup-replay.exe
Exit 0 = every (C) line matches the replay and the inputs line is well formed; 1 = a mismatch (each printed).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# (field, class, server_main.cpp line on branch arch024-s0a (approximate: the lines move with later steps), what)
FIELDS = [
    ("others.ok / others.mib / others.display", "B", 588, "other processes' VRAM and display (PDH), before loading"),
    ("wddm_start.ok / wddm_start.budget", "B", 589, "WDDM budget before WHIRL allocates (margin input)"),
    ("fitp.* (FitInput)", "B", 785, "pre-load estimate fed to startfit::fitParallel; avail is glue over memInfo / WDDM / reserve"),
    ("fitp.per_slot / fitp.given", "A", 785, "per-slot fixed VRAM (model) and whether --parallel was given (user)"),
    ("fitf.* (FitInput + plan)", "B", 796, "input of startfit::fitFixed"),
    ("slots.*", "B", 1104, "after-load VRAM fed to startfit::slotsThatFit (av is glue over memInfo / WDDM)"),
    ("ram.total_phys / ram.avail_phys", "B", 1300, "GlobalMemoryStatusEx"),
    ("ram.explicit_mb / ram.explicit_src", "A", 1290, "--kv-ram-mb / WHIRL_KV_RAM_MB"),
    ("ram.floor_mb / ram.entry_mb / ram.integrated / ram.n_ck / ram.no_pc", "A", 1270, "model, slot context and settings"),
    ("pin.* (PinnedState)", "B", 1320, "DXGI / PDH pinned memory of this process and every program (B0-1 adapter_shared)"),
    ("pool.avail / pool.per_token", "B", 1478, "VRAM for the KV pool (glue over memInfo / WDDM / margin / vision reserve)"),
    ("pool.req / pool.*_explicit / pool.slot_ctx / pool.page / pool.cap / pool.mode / pool.fmt / pool.alt", "A", 1478,
     "settings and the mode's KV format"),
    ("'room for the RAM tier N MiB'", "C", 1341, "startfit::pinnedRoomMb"),
    ("'kv tier: RAM tier size N MiB (reason)'", "C", 1372, "tier::ramTierSize + startfit::ramTierFit + ramTierText.size_reason"),
    ("'kv tier: RAM tier off (reason)'", "C", 1382, "ramTierText.off_reason"),
    ("'adjusted: --kv-ram-mb A -> B MiB (why)'", "C", 1365, "ramTierFit + ramTierText.adjusted_why"),
    ("'slots: N;'", "C", 1651, "startfit::slotsThatFit (or fitParallel's slots when not re-checked)"),
    ("'kv pool: N tokens'", "C", 1478, "numerics::serverPoolFit; after an issue-#5 re-check the replay must equal the first "
     "'shrinking the KV pool from A' and the rounds must chain (smaller, whole pages, <= 3) to N"),
    ("'kv pool check: ... shrinking the KV pool from A to B tokens'", "glue", 1661,
     "the issue-#5 re-check (WDDM read again after the alloc, not in the inputs line): chain checked, values not replayed"),
    ("'adjusted: --parallel p -> p-1' lines", "glue", 1100, "the adjust loop that prints each step"),
    ("'startup adjustments (...)'", "glue", 2062, "summary of the adjusted: lines"),
    ("'adjusted: -c / --ctx-per-slot'", "glue", 1488, "from serverPoolFit's result, formatted by the glue"),
]

INPUT_RE = re.compile(r"startup inputs:(.*)$")
TOKEN_RE = re.compile(r"^[a-z_0-9.]+=[^\s=]+$")


def dec(v: str) -> str:
    if v == "%":
        return ""
    return re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), v)


def parse_kv(text: str) -> dict:
    out = {}
    for tok in text.split():
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def inputs_line(log: str):
    last = None
    for line in log.splitlines():
        m = INPUT_RE.search(line)
        if m:
            last = m.group(1)
    return last


def format_problems(text: str) -> list:
    bad = []
    seen = set()
    for tok in text.split():
        if not TOKEN_RE.match(tok):
            bad.append(f"inputs: bad token {tok!r}")
            continue
        k = tok.split("=", 1)[0]
        if k in seen:
            bad.append(f"inputs: duplicate key {k}")
        seen.add(k)
    for need in ("ram.total_phys",) if any(k.startswith("ram.") for k in seen) else ():
        if need not in seen:
            bad.append(f"inputs: missing {need}")
    return bad


def replay_lines(exe: str, inputs: str) -> list:
    r = subprocess.run([exe, "--line", inputs], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0:
        raise RuntimeError(f"replay failed (exit {r.returncode}): {r.stdout} {r.stderr}")
    return lines


def groups(lines: list) -> dict:
    g = {}
    for l in lines:
        name, _, rest = l.partition(" ")
        g[name] = {k: dec(v) for k, v in parse_kv(rest).items()}
    return g


def compare(log: str, rep: list, inp: dict) -> list:
    """(C) lines of the log against the replay's decisions; returns the mismatches."""
    bad = []
    g = groups(rep)
    lines = log.splitlines()
    find = lambda rx: [m for m in (re.search(rx, l) for l in lines) if m]
    if "ram" in g:
        r = g["ram"]
        if inp.get("pin.measured") == "1":
            m = find(r"-> room for the RAM tier (\d+ MiB|unknown) \(")
            if not m:
                bad.append("ram: no 'room for the RAM tier' line")
            elif m[-1].group(1) != r["room"]:
                bad.append(f"ram: room {m[-1].group(1)!r} in the log, replay {r['room']!r}")
        host_tiers = inp.get("ram.n_ck", "0") != "0" and inp.get("ram.no_pc", "0") == "0"
        size = find(r"kv tier: RAM tier size (\d+) MiB \((.*)\)$")
        off = find(r"kv tier: RAM tier off \((.*)\)$")
        if host_tiers:
            if r["final_mb"] != "0":
                if not size:
                    bad.append(f"ram: no 'RAM tier size' line, replay {r['final_mb']} MiB")
                else:
                    if size[-1].group(1) != r["final_mb"]:
                        bad.append(f"ram: size {size[-1].group(1)} MiB in the log, replay {r['final_mb']}")
                    if size[-1].group(2) != r["size_reason"]:
                        bad.append(f"ram: size reason differs:\n    log    {size[-1].group(2)}\n    replay {r['size_reason']}")
            else:
                if not off:
                    bad.append("ram: no 'RAM tier off' line, replay 0 MiB")
                elif off[-1].group(1) != r["off_reason"]:
                    bad.append(f"ram: off reason differs:\n    log    {off[-1].group(1)}\n    replay {r['off_reason']}")
        if inp.get("ram.explicit_mb", "-") != "-" and r.get("fit") == "1":
            src = dec(inp.get("ram.explicit_src", "--kv-ram-mb"))
            want = f"adjusted: {src} {r['size_mb']} -> {r['fit.mb']} MiB ({r['adjusted_why']})"
            if not any(l.endswith(want) for l in lines):
                bad.append(f"ram: no line ending with {want!r}")
    slots_want = None
    if "slots" in g:
        slots_want = g["slots"]["result"]
    elif "fitp" in g and "pool" in g:
        slots_want = g["fitp"]["parallel"]
    if slots_want is not None:
        m = find(r"\bI slots: (\d+);")
        if not m:
            bad.append("slots: no 'slots: N;' line")
        elif m[-1].group(1) != slots_want:
            bad.append(f"slots: {m[-1].group(1)} in the log, replay {slots_want}")
    if "pool" in g and g["pool"]["refuse"] == "0":
        m = find(r"\bkv pool: (\d+) tokens")
        if not m:
            bad.append("pool: no 'kv pool: N tokens' line")
        else:
            bad += pool_chain(int(g["pool"]["pool"]), int(m[0].group(1)),
                              [(int(x.group(1)), int(x.group(2))) for x in find(RECHECK_RE)],
                              int(inp.get("pool.page", "256") or 256))
    return bad


# server_main's issue-#5 re-check after the pool is allocated: it reads the WDDM budget / usage again (measured values
# that are not in the "startup inputs:" line) and shrinks the pool by whole pages, up to 3 rounds, one line per round.
# The replay decides the pool before it (serverPoolFit); the re-check is glue, checked here for consistency only.
RECHECK_RE = r"kv pool check: .* -> shrinking the KV pool from (\d+) to (\d+) tokens"


def pool_chain(replay: int, logged: int, rounds: list, page: int) -> list:
    """The replay's pool, then each re-check round (from, to), must lead to the logged 'kv pool: N tokens'."""
    if not rounds:
        return [] if logged == replay else [f"pool: {logged} tokens in the log, replay {replay}"]
    bad = []
    have = replay
    for i, (a, b) in enumerate(rounds, 1):
        if a != have:
            what = "the replay's pool" if i == 1 else f"round {i - 1}'s result"
            bad.append(f"pool: re-check round {i} shrinks from {a} tokens, {what} is {have}")
        if not b < a:
            bad.append(f"pool: re-check round {i} goes from {a} to {b} tokens (not smaller)")
        if page > 0 and b % page:
            bad.append(f"pool: re-check round {i} result {b} tokens is not whole pages of {page}")
        have = b
    if len(rounds) > 3:
        bad.append(f"pool: {len(rounds)} re-check rounds (server_main does at most 3)")
    if logged != have:
        bad.append(f"pool: {logged} tokens in the log, re-check ended at {have} (replay {replay})")
    return bad


# arch025-surv (mutation survivors c1a-main-mem0, c1b-mem-t1): the (B) measured VRAM values are not compared by value,
# but three of the VRAM lines have a sign that holds on every run: a measuring point that is never read prints 0 there
# (gib() / the MiB difference clamp at 0). Checked only when the line is present. Measured on the 93 recorded server
# logs of scratch25_c1b_gpu (both GPUs, 2026-10-27): weights + buffers 16.08 .. 21.19 GiB, at start >= after weights,
# tier stream 12.8 .. 18445.9 MiB; never 0.
VALUE_RULES = [
    (re.compile(r"VRAM use: weights \+ buffers ([0-9.]+) GiB"), lambda v: float(v[0]) > 0,
     "VRAM use: weights + buffers is 0 (the VRAM before the load, mem0, was not read)"),
    (re.compile(r"\(after weights ([0-9.]+), at start ([0-9.]+)\)"), lambda v: float(v[1]) > 0 and float(v[1]) >= float(v[0]),
     "VRAM free at start is 0 or below the free VRAM after the weights (mem0 not read)"),
    (re.compile(r"VRAM taken by the tier stream / pinned arena mapping: ([0-9.]+) MiB"), lambda v: float(v[0]) > 0,
     "kv tier: VRAM taken by the tier stream is 0 (mem_t1 not read after the tier stream)"),
]


def value_problems(log: str) -> list:
    bad = []
    for l in log.splitlines():
        for rx, ok, why in VALUE_RULES:
            m = rx.search(l)
            if m and not ok(m.groups()):
                bad.append(f"value: {why}: {l.strip()[:200]}")
    return bad


def check(log: str, exe: str) -> list:
    text = inputs_line(log)
    if text is None:
        return ["no 'startup inputs:' line"]
    bad = format_problems(text)
    if bad:
        return bad
    return compare(log, replay_lines(exe, text), parse_kv(text)) + value_problems(log)


def selftest(exe: str) -> int:
    base_in = ("ram.explicit_mb=17346 ram.explicit_src=--kv-ram-mb ram.total_phys=33981464576 ram.avail_phys=24696061952 "
               "ram.floor_mb=8192 ram.entry_mb=9216 ram.integrated=1 ram.n_ck=8 ram.no_pc=0 pin.measured=1 "
               "pin.total_phys=33981464576 pin.pinned_max=1575485440 pin.later=0 pin.gpus=1 pin.g0.nonlocal_budget=0 "
               "pin.g0.own_nonlocal=1448083456 pin.g0.shared_system=16991223808 pin.g0.adapter_shared=1575485440 "
               "slots.parallel=4 slots.avail=21474836480 slots.per_slot=1073741824 slots.shared=1342177280 "
               "slots.per_token=69632 slots.target=16384")

    def log_for(inputs: str, mutate=None) -> str:
        g = groups(replay_lines(exe, inputs))
        r = g["ram"]
        inp = parse_kv(inputs)
        src = dec(inp["ram.explicit_src"])
        ls = [
            f"2026-10-23 20:00:42.095 I kv tier: pinned memory now: 1 HIP GPU(s) [...] -> room for the RAM tier {r['room']} (reserves 2048 / 2048 MiB)",
        ]
        if r.get("fit") == "1" and inp["ram.explicit_mb"] != "-":
            ls.append(f"2026-10-23 20:00:42.095 W adjusted: {src} {r['size_mb']} -> {r['fit.mb']} MiB ({r['adjusted_why']})")
        if r["final_mb"] != "0":
            ls.append(f"2026-10-23 20:00:42.095 I kv tier: RAM tier size {r['final_mb']} MiB ({r['size_reason']})")
        else:
            ls.append(f"2026-10-23 20:00:42.095 I kv tier: RAM tier off ({r['off_reason']})")
        ls.append(f"2026-10-23 20:00:46.598 I slots: {g['slots']['result']}; per slot: recurrent state 149.6 MiB")
        ls.append(f"2026-10-23 20:00:46.602 I startup inputs: {inputs}")
        text = "\n".join(ls) + "\n"
        return mutate(text) if mutate else text

    # the KV pool line and the issue-#5 re-check (server_main: "kv pool check: ... -> shrinking the KV pool from A to
    # B tokens", one line per round, up to 3 rounds, measured WDDM reads that are not in the inputs line)
    pool_in = base_in + (" pool.avail=10578939904 pool.per_token=36992 pool.req=- pool.pool_explicit=0 pool.slot_ctx=131072"
                         " pool.slot_explicit=0 pool.page=256 pool.cap=2097152 pool.mode=balance pool.fmt=q8h pool.alt=%")

    def pool_log(shrinks: list, final=None, first_from=None) -> str:
        """log_for(pool_in) plus the pool lines: shrinks = the 'to' value of each re-check round (in tokens below
        the replay's pool); final = the 'kv pool:' value (default: the last 'to'); first_from = the first round's
        'from' (default: the replay's pool)."""
        p = int(groups(replay_lines(exe, pool_in))["pool"]["pool"])
        ls = []
        have = p if first_from is None else first_from
        for to in shrinks:
            ls.append(f"2026-10-24 00:21:10.389 W kv pool check: WDDM budget 31.02 GiB now (31.02 GiB when the pool was "
                      f"sized), local 28.97 GiB + margin 2101 MiB is 0.00 GiB over -> shrinking the KV pool from {have} to "
                      f"{p - to} tokens (9.85 -> 9.84 GiB) so WHIRL stays in VRAM")
            have = p - to
        if shrinks:
            ls.append(f"2026-10-24 00:21:10.840 I kv pool check: WDDM budget 31.02 GiB now (31.02 GiB when the pool was "
                      f"sized), local 28.91 GiB + margin 2101 MiB fits (KV pool {have} tokens)")
        n = have if final is None else final
        ls.append(f"2026-10-24 00:21:10.840 I kv pool: {n} tokens ({n // 256} pages of 256), q8h KV, 36.1 KiB/token, "
                  f"shared by 4 slots; max 131072 tokens per request")
        return log_for(pool_in) + "\n".join(ls) + "\n"

    p0 = int(groups(replay_lines(exe, pool_in))["pool"]["pool"])
    b_changed = base_in.replace("pin.g0.adapter_shared=1575485440", "pin.g0.adapter_shared=9663676416").replace(
        "pin.pinned_max=1575485440", "pinned_max_placeholder")
    b_changed = b_changed.replace("pinned_max_placeholder", "pin.pinned_max=9663676416")
    good = log_for(base_in)
    vram_lines = ("2026-10-27 19:30:01.100 I kv tier: VRAM taken by the tier stream / pinned arena mapping: 12.8 MiB (before the pool was sized)\n"
                  "2026-10-27 19:30:01.101 I VRAM use: weights + buffers 16.08 GiB (prefill batch 512, embedding in VRAM), slots 0.58 GiB, "
                  "other 0.10 GiB, checkpoints 0.50 GiB, KV pool 9.85 GiB, left 3.83 GiB\n"
                  "2026-10-27 19:30:01.102 I context 131072 tokens per request, pool 313856 tokens; VRAM free 3.83 / 31.86 GiB "
                  "(after weights 15.63, at start 31.71)\n")
    cases = [
        ("pass: log made from the replay", good, 0),
        ("block: a (C) value changed (RAM tier size)", re.sub(r"RAM tier size (\d+) MiB", lambda m: f"RAM tier size {int(m.group(1)) + 512} MiB", good), 1),
        ("block: a (C) value changed (room)", re.sub(r"room for the RAM tier (\d+) MiB", lambda m: f"room for the RAM tier {int(m.group(1)) - 512} MiB", good), 1),
        ("block: a (C) value changed (slots)", re.sub(r"I slots: (\d+);", lambda m: f"I slots: {int(m.group(1)) + 1};", good), 1),
        ("pass: a (B) value changed and the (C) lines replaced by the replay of it", log_for(b_changed), 0),
        ("block: a (B) value changed but the (C) lines kept", good.replace(base_in, b_changed), 1),
        ("block: no startup inputs line", "\n".join(l for l in good.splitlines() if "startup inputs:" not in l), 1),
        ("block: malformed inputs line", good.replace("ram.n_ck=8", "ram.n_ck 8"), 1),
        ("pass: kv pool line = the replay (no re-check)", pool_log([]), 0),
        ("block: kv pool line one page off, no re-check line", pool_log([], final=p0 - 256), 1),
        ("pass: issue-#5 re-check shrank the replay's pool by one page (S3 run, 11 logs)", pool_log([256]), 0),
        ("pass: re-check in two rounds, chain ends at the kv pool line", pool_log([256, 768]), 0),
        ("block: re-check started from a pool the replay did not decide", pool_log([256], first_from=p0 + 256), 1),
        ("block: kv pool line differs from the re-check's last value", pool_log([256], final=p0 - 512), 1),
        ("block: re-check chain broken (round 2 starts elsewhere)",
         pool_log([256, 512]).replace(f"from {p0 - 256} to {p0 - 512}", f"from {p0 - 1024} to {p0 - 512}"), 1),
        ("block: re-check 'shrank' to a larger pool", pool_log([-256]), 1),
        ("block: re-check to a size that is not whole pages", pool_log([100]), 1),
        ("pass: VRAM lines of a real run (R9700, c1b GPU check)", good + vram_lines, 0),
        ("block: weights + buffers 0 and at start 0 (mem0 not read: c1a-main-mem0)",
         good + vram_lines.replace("weights + buffers 16.08", "weights + buffers 0.00").replace("at start 31.71", "at start 0.00"), 1),
        ("block: tier stream 0 MiB (mem_t1 not read: c1b-mem-t1)", good + vram_lines.replace("mapping: 12.8 MiB", "mapping: 0.0 MiB"), 1),
    ]
    bad = 0
    for name, log, want in cases:
        try:
            got = 1 if check(log, exe) else 0
        except RuntimeError as ex:
            got = 1
            name += f" (replay error {ex})"
        ok = got == want
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'mismatch' if got else 'match'} (want {'mismatch' if want else 'match'})")
    print(f"startup_lines selftest: {len(cases) - bad}/{len(cases)} as expected")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("log")
    c.add_argument("--replay", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--replay", required=True)
    sub.add_parser("fields")
    a = ap.parse_args()
    if a.cmd == "fields":
        for f, k, line, what in FIELDS:
            print(f"({k}) {f}  [server_main.cpp:{line}]  {what}")
        return 0
    if a.cmd == "selftest":
        return selftest(a.replay)
    with open(a.log, encoding="utf-8", errors="replace") as f:
        log = f.read()
    bad = check(log, a.replay)
    for b in bad:
        print(f"MISMATCH {b}")
    print(f"startup lines: {'ok' if not bad else f'{len(bad)} mismatch(es)'} ({os.path.basename(a.log)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
