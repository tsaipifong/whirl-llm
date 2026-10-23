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
    ("'kv pool: N tokens'", "C", 1478, "numerics::serverPoolFit (glue: the issue-#5 re-check may shrink it later)"),
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
        elif m[0].group(1) != g["pool"]["pool"] and not find(r"kv pool: .*(shrunk|lowered)"):
            bad.append(f"pool: {m[0].group(1)} tokens in the log, replay {g['pool']['pool']}")
    return bad


def check(log: str, exe: str) -> list:
    text = inputs_line(log)
    if text is None:
        return ["no 'startup inputs:' line"]
    bad = format_problems(text)
    if bad:
        return bad
    return compare(log, replay_lines(exe, text), parse_kv(text))


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

    b_changed = base_in.replace("pin.g0.adapter_shared=1575485440", "pin.g0.adapter_shared=9663676416").replace(
        "pin.pinned_max=1575485440", "pinned_max_placeholder")
    b_changed = b_changed.replace("pinned_max_placeholder", "pin.pinned_max=9663676416")
    good = log_for(base_in)
    cases = [
        ("pass: log made from the replay", good, 0),
        ("block: a (C) value changed (RAM tier size)", re.sub(r"RAM tier size (\d+) MiB", lambda m: f"RAM tier size {int(m.group(1)) + 512} MiB", good), 1),
        ("block: a (C) value changed (room)", re.sub(r"room for the RAM tier (\d+) MiB", lambda m: f"room for the RAM tier {int(m.group(1)) - 512} MiB", good), 1),
        ("block: a (C) value changed (slots)", re.sub(r"I slots: (\d+);", lambda m: f"I slots: {int(m.group(1)) + 1};", good), 1),
        ("pass: a (B) value changed and the (C) lines replaced by the replay of it", log_for(b_changed), 0),
        ("block: a (B) value changed but the (C) lines kept", good.replace(base_in, b_changed), 1),
        ("block: no startup inputs line", "\n".join(l for l in good.splitlines() if "startup inputs:" not in l), 1),
        ("block: malformed inputs line", good.replace("ram.n_ck=8", "ram.n_ck 8"), 1),
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
